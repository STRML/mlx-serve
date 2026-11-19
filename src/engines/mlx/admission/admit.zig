//! AdmissionBill, fitAfterUnloadingIdle; prefillAdmissionBill, verdict/log, the scheduler hooks, admitPrompt

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const transformer_mod = @import("../transformer.zig");
const generate_mod = @import("../generate.zig");
const model_mod = @import("../../../sdk/config/model.zig");
const log = @import("../../../sdk/util/log.zig");
const scheduler_mod = @import("../scheduler.zig");
const model_registry_mod = @import("../../../host/registry/registry.zig");
const http_server = @import("../../../host/http/server.zig");

const configuredKvQuantFor = @import("../../../host/http/server.zig").configuredKvQuantFor;
const currentGpuMemoryCeiling = @import("../../../host/registry/memory_ceiling.zig").currentGpuMemoryCeiling;
const explicitPrefillChunk = @import("prefill_chunk.zig").explicitPrefillChunk;
const qsaScoreFusedOffGuard = @import("bills_test.zig").qsaScoreFusedOffGuard;
const WarmPrefix = @import("bills.zig").WarmPrefix;
const PrefillLimits = @import("prefill_chunk.zig").PrefillLimits;
const decodeShareCapped = @import("../generate/prefill_chunk.zig").decodeShareCapped;
const prefillNeededAtChunk = @import("prefill_chunk.zig").prefillNeededAtChunk;
const perRequestPrefillChunkEnabled = @import("prefill_chunk.zig").perRequestPrefillChunkEnabled;
const chooseRequestPrefillChunk = @import("prefill_chunk.zig").chooseRequestPrefillChunk;
const requestPrefillChunkNow = @import("prefill_chunk.zig").requestPrefillChunkNow;
const adaptivePrefillWidthNow = @import("prefill_chunk.zig").adaptivePrefillWidthNow;
const adaptivePrefillWidenStillFits = @import("prefill_chunk.zig").adaptivePrefillWidenStillFits;
const adaptivePrefillChunkEnabled = @import("prefill_chunk.zig").adaptivePrefillChunkEnabled;
const clearResolvedPrefixCacheMem = @import("hot_cache_plan.zig").clearResolvedPrefixCacheMem;
const qwen4ExpOomConfig = @import("bills_test.zig").qwen4ExpOomConfig;

/// The admission bill and the headroom it is compared against, so the connection thread's
/// guard and the inference thread's evict-or-refuse decision cannot compute them differently.
pub const AdmissionBill = @import("../../../abi/engine.zig").Bill;

/// Unloads idle resident models, least recently used first, until `fits` holds; false when
/// nothing idle is left. An idle model is memory its next request reloads.
pub fn fitAfterUnloadingIdle(ctx: anytype, fits: fn (@TypeOf(ctx)) bool, unloadOne: fn (@TypeOf(ctx)) bool) bool {
    while (!fits(ctx)) if (!unloadOne(ctx)) return false;
    return true;
}

pub fn prefillAdmissionBill(config: *const model_mod.ModelConfig, prompt_len: usize, max_tokens: u32, kv_override: ?transformer_mod.KVQuantConfig, prompt_tokens: ?[]const u32, warm: WarmPrefix, limits: PrefillLimits) AdmissionBill {
    const heads = config.num_attention_heads;
    if (heads == 0) return .{ .needed = 0, .available = std.math.maxInt(u64) };
    const seq: u64 = @intCast(prompt_len);

    const kv_cfg: transformer_mod.KVQuantConfig = kv_override orelse configuredKvQuantFor(config);
    const kv_bits: u64 = if (kv_cfg.scheme == .off) 16 else kv_cfg.bits;
    // Available = the lesser of Metal's static working-set max and what is physically reachable
    // now (#64). Read before the width is chosen: on a per-request arch the width is a function of it.
    var active_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);
    const total_limit: u64 = currentGpuMemoryCeiling(config, active_mem);
    const available = if (total_limit > active_mem) total_limit - active_mem else 0;

    // A vision prefill chunks like text since issue #197 (the splice resumes its row index across
    // chunks). The width comes from `chooseRequestPrefillChunk`, the same rule the scheduler pins
    // with. An ungated arch (or an explicit `--prefill-chunk`) gets the load-time pin.
    const chosen: u32 = chooseRequestPrefillChunk(
        config,
        seq,
        max_tokens,
        kv_bits,
        available,
        config.pinned_prefill_chunk,
        explicitPrefillChunk(),
        warm,
        limits,
    );
    const effective = generate_mod.effectivePrefillChunk(config.prefillScoreHeadDim(), config.num_attention_heads, prompt_len, config.has_sliding_window, config.isMoe(), config.longCtxGated(), chosen);
    const chunk: u64 = config.cappedPrefillChunk(decodeShareCapped(effective, limits.width_cap));
    // RAM hot-cache restores rebind MLX array handles by refcount; they do not
    // allocate another copy of the cached buffers. `active_mem` above already
    // includes the resident entry, while `prefillMemoryNeeded` bills the full
    // destination KV capacity that may be allocated when the restored cache
    // grows. Adding the resident entry here again would invent a third copy
    // and reject long warm prompts (and even cache misses) spuriously.
    const needed: u64 = prefillNeededAtChunk(config, seq, max_tokens, kv_bits, chunk, warm);
    // Published number, never the pointer: `hot_prefix_cache` is inference-thread state, freed
    // on every model switch. Stale by at most one entry, which is all a hint has to be.
    const evictable: u64 = if (http_server.launch().global_scheduler) |sch|
        sch.resident_hot_cache_bytes.load(.monotonic)
    else
        0;
    // Prompt-aware when the caller has the ids: an entry this prompt could not restore from is
    // genuinely evictable. Still a published snapshot, never the cache pointer.
    const reclaimable: u64 = if (http_server.launch().global_scheduler) |sch|
        if (prompt_tokens) |toks|
            scheduler_mod.reclaimableHotCacheBytesFor(sch, toks)
        else
            sch.reclaimable_hot_cache_bytes.load(.monotonic)
    else
        0;
    return .{ .needed = needed, .available = available, .evictable = evictable, .reclaimable = reclaimable, .chunk = chunk };
}

/// Bytes of the resident hot cache this prompt pins: residency less what an eviction can prove
/// it will return. Non-zero means an entry this prompt would restore from is holding memory
/// that neither goes back nor has to be allocated again; the connection thread cannot price
/// how much, so it defers to the inference thread's `WarmPrefix` bill.
pub fn pinnedResidentBytes(bill: AdmissionBill) u64 {
    return bill.evictable -| bill.evictionCredit();
}

/// What the admission guard decided, from one bill so the log line and the arm agree: fits
/// outright, fits once the hot cache gives back what it provably can, or does not fit.
pub const AdmissionVerdict = enum {
    admit,
    evict,
    refuse,

    pub fn name(self: AdmissionVerdict) []const u8 {
        return switch (self) {
            .admit => "admit",
            .evict => "evict",
            .refuse => "refuse",
        };
    }
};

pub fn admissionVerdict(bill: AdmissionBill) AdmissionVerdict {
    if (bill.fits()) return .admit;
    if (bill.fitsAfterEviction()) return .evict;
    return .refuse;
}

/// The generation of `markReadyLocked` the last info-level admission line was emitted for.
var admission_logged_generation = std.atomic.Value(u64).init(std.math.maxInt(u64));

/// Has a model become resident since the last time this returned true? Consuming, so N racing connection threads produce one line.
fn firstRequestAfterLoad() bool {
    const gen = model_registry_mod.generations().load.load(.monotonic);
    const last = admission_logged_generation.load(.monotonic);
    if (gen == last) return false;
    return admission_logged_generation.cmpxchgStrong(last, gen, .monotonic, .monotonic) == null;
}

/// The level one admission line is emitted at: info on the first request after a load and
/// whenever the bill is within 10% of what is available; debug otherwise.
pub fn admissionLogLevel(needed: u64, available: u64, first_after_load: bool) log.Level {
    if (first_after_load) return .info;
    // `needed > 0.9 * available` without floating point or overflow.
    if (needed *| 10 > available *| 9) return .info;
    return .debug;
}

/// One line per request at the admission decision, on admitted requests too. Every field comes
/// off the same `bill` the arms act on. The level check runs first.
fn logAdmissionDecision(bill: AdmissionBill) void {
    // `info` is the loudest level this line is emitted at, and `firstRequestAfterLoad` consumes its token.
    if (!log.enabled(.info)) return;
    const level = admissionLogLevel(bill.needed, bill.available, firstRequestAfterLoad());
    if (!log.enabled(level)) return;
    const mb = 1024 * 1024;
    log.atLevel(level, "[admission] needed={d} MB available={d} MB reclaimable={d} MB width={d} verdict={s}\n", .{
        bill.needed / mb,
        bill.available / mb,
        bill.evictionCredit() / mb,
        bill.chunk,
        admissionVerdict(bill).name(),
    });
}

/// Inference-thread predicate behind the scheduler's admission hook: does this request fit
/// right now, with live memory re-read? Every input is forwarded, none defaulted (a hardcoded
/// `kv_override = null` priced a `kv_quant: 4` request at fp16 and evicted a cache for nothing).
pub fn prefillFitsNow(config: *const model_mod.ModelConfig, prompt_len: usize, max_tokens: u32, kv_cfg: transformer_mod.KVQuantConfig, warm: WarmPrefix, limits: PrefillLimits) bool {
    const bill = prefillAdmissionBill(config, prompt_len, max_tokens, kv_cfg, null, warm, limits);
    return bill.needed +| limits.extra_bytes <= bill.available;
}

/// {needed, available} of the cold bill, live memory re-read, for the scheduler's sibling ledger.
pub fn prefillBillNumbersNow(config: *const model_mod.ModelConfig, prompt_len: usize, max_tokens: u32, kv_cfg: transformer_mod.KVQuantConfig, enable_mtp: bool, limits: PrefillLimits) [2]u64 {
    const bill = prefillAdmissionBill(config, prompt_len, max_tokens, kv_cfg, null, .{ .mtp_on = enable_mtp }, limits);
    return .{ bill.needed, bill.available };
}

/// The inference thread's refusal, quoting the numbers it compared; returns {needed, available} for the client.
pub fn logPrefillRefusal(config: *const model_mod.ModelConfig, prompt_len: usize, max_tokens: u32, kv_cfg: transformer_mod.KVQuantConfig, warm: WarmPrefix, limits: PrefillLimits) [2]u64 {
    var bill = prefillAdmissionBill(config, prompt_len, max_tokens, kv_cfg, null, warm, limits);
    bill.needed +|= limits.extra_bytes;
    const mb = 1024 * 1024;
    // "Narrowest width tried" is only true where the ladder was walked.
    const width_note: []const u8 = if (perRequestPrefillChunkEnabled(config) and explicitPrefillChunk() == 0) "the narrowest width tried" else "the pinned width";
    // The comparison first (`fitsAfterEviction` spelled out), then the residency as context, with
    // the part eviction cannot take named.
    log.warn("  prompt {d} tokens needs ~{d}MB at prefill chunk {d} ({s}), ~{d}MB available + ~{d}MB the hot cache can give back (~{d}MB resident, ~{d}MB of it pinned by this request's own prefix) — refused before prefill\n", .{
        prompt_len,
        bill.needed / mb,
        bill.chunk,
        width_note,
        bill.available / mb,
        bill.evictionCredit() / mb,
        bill.evictable / mb,
        pinnedResidentBytes(bill) / mb,
    });
    return .{ bill.needed, bill.available };
}

/// The scheduler's admission and prefill-width hooks, live while the server runs (`on`); unit tests leave them null.
pub fn installSchedulerHooks(on: bool) void {
    scheduler_mod.prefill_admission_fits = if (on) &prefillFitsNow else null;
    scheduler_mod.prefill_admission_numbers = if (on) &prefillBillNumbersNow else null;
    scheduler_mod.prefill_request_chunk = if (on) &requestPrefillChunkNow else null;
    // The published budget is per model; retire it whenever the scheduler drops the cache.
    scheduler_mod.hot_cache_budget_invalidate = if (on) &clearResolvedPrefixCacheMem else null;
    scheduler_mod.prefill_admission_refused_log = if (on) &logPrefillRefusal else null;
    // Asked by the prefill loop: the width of the next chunk, re-priced at every boundary.
    scheduler_mod.prefill_chunk_adapt = if (on) &adaptivePrefillWidthNow else null;
    scheduler_mod.prefill_chunk_widen_ok = if (on) &adaptivePrefillWidenStillFits else null;
    // Whether the hook above can move anything for a given model (process-wide, so presence is not a gate).
    scheduler_mod.prefill_chunk_adaptive_enabled = if (on) &adaptivePrefillChunkEnabled else null;
}

/// The connection thread's guard: true admits (outright, once the hot cache gives back what it
/// can, or deferred to the warm bill); false refuses, `out` holding the bill it compared after
/// unloading every idle model but `keep`.
pub fn admitPrompt(config: *const model_mod.ModelConfig, keep: []const u8, prompt_ids: []const u32, max_tokens: u32, kv_override: ?transformer_mod.KVQuantConfig, enable_mtp: bool, media_bytes: u64, out: *AdmissionBill) bool {
    const prompt_len: usize = prompt_ids.len;
    if (config.num_attention_heads == 0) return true; // unknown architecture, skip check
    // The connection thread has no slot and no cache: it bills cold and defers a warm prompt.
    const limits: PrefillLimits = .{ .extra_bytes = if (config.residentDsv41()) media_bytes else 0 };
    var bill = prefillAdmissionBill(config, prompt_len, max_tokens, kv_override, prompt_ids, .{ .mtp_on = enable_mtp }, limits);
    // The request's media rows are one more tensor the prefill allocates.
    bill.needed +|= media_bytes;
    logAdmissionDecision(bill);
    const needed = bill.needed;
    const available = bill.available;
    // Evict to admit (#353): when the bill only fails because the cache is holding memory, admit;
    // the inference thread evicts LRU entries until it fits and refuses by name only if it still does not.
    if (needed > available and bill.fitsAfterEviction()) {
        log.info("  prompt {d} tokens needs ~{d}MB, ~{d}MB available + ~{d}MB reclaimable hot cache (of ~{d}MB resident) — admitting, the prefill will evict\n", .{
            prompt_len, needed / (1024 * 1024), available / (1024 * 1024), bill.evictionCredit() / (1024 * 1024), bill.evictable / (1024 * 1024),
        });
        return true;
    }
    // Warm deferral. The bill above is cold; a non-zero `pinnedResidentBytes` means an entry
    // this prompt would restore from is holding bytes that will neither be allocated nor
    // reclaimed. How many is knowable only on the inference thread, which bills with `WarmPrefix`.
    if (needed > available and pinnedResidentBytes(bill) > 0) {
        log.info("  prompt {d} tokens needs ~{d}MB cold, ~{d}MB available, and ~{d}MB of hot cache is a prefix this prompt would SHARE — deferring to the prefill, which bills the warm request\n", .{
            prompt_len, needed / (1024 * 1024), available / (1024 * 1024), pinnedResidentBytes(bill) / (1024 * 1024),
        });
        return true;
    }
    if (needed > available) if (http_server.launch().global_scheduler) |sch| {
        const Rebill = struct {
            sch: *scheduler_mod.Scheduler,
            keep: []const u8,
            bill: *AdmissionBill,
            config: *const model_mod.ModelConfig,
            prompt_ids: []const u32,
            max_tokens: u32,
            kv_override: ?transformer_mod.KVQuantConfig,
            enable_mtp: bool,
            media_bytes: u64,
            fn fits(r: *@This()) bool {
                return r.bill.fits();
            }
            fn unloadOne(r: *@This()) bool {
                const id = scheduler_mod.idleModelToUnload(r.sch, r.keep) orelse return false;
                log.info("  prompt {d} tokens does not fit — unloading idle model {s} first\n", .{ r.prompt_ids.len, id });
                r.sch.unloadModelIfIdle(id, 0) catch return false;
                const retry_limits: PrefillLimits = .{ .extra_bytes = if (r.config.residentDsv41()) r.media_bytes else 0 };
                r.bill.* = prefillAdmissionBill(r.config, r.prompt_ids.len, r.max_tokens, r.kv_override, r.prompt_ids, .{ .mtp_on = r.enable_mtp }, retry_limits);
                r.bill.needed +|= r.media_bytes;
                return true;
            }
        };
        var rebill = Rebill{ .sch = sch, .keep = keep, .bill = &bill, .config = config, .prompt_ids = prompt_ids, .max_tokens = max_tokens, .kv_override = kv_override, .enable_mtp = enable_mtp, .media_bytes = media_bytes };
        if (fitAfterUnloadingIdle(&rebill, Rebill.fits, Rebill.unloadOne)) return true;
    };
    if (bill.needed > bill.available) {
        const needed_mb = bill.needed / (1024 * 1024);
        const avail_mb = bill.available / (1024 * 1024);
        log.warn("  prompt {d} tokens needs ~{d}MB (KV+working+margin) at prefill chunk {d}, ~{d}MB available + ~{d}MB reclaimable of ~{d}MB resident hot cache — rejecting\n", .{ prompt_len, needed_mb, bill.chunk, avail_mb, bill.evictionCredit() / (1024 * 1024), bill.evictable / (1024 * 1024) });
        out.* = bill;
        return false;
    }
    return true;
}

// ── tests ──

test "a prompt that does not fit unloads idle models, oldest first, until it fits" {
    const t = std.testing;
    const Fake = struct {
        needed: u64,
        available: u64,
        idle: []const u64, // resident bytes, least recently used first
        unloaded: usize = 0,
        fn fits(f: *@This()) bool {
            return f.needed <= f.available;
        }
        fn unloadOne(f: *@This()) bool {
            if (f.unloaded == f.idle.len) return false;
            f.available += f.idle[f.unloaded];
            f.unloaded += 1;
            return true;
        }
    };
    // Already fits: nothing is unloaded.
    var a = Fake{ .needed = 5, .available = 9, .idle = &.{ 15, 3 } };
    try t.expect(fitAfterUnloadingIdle(&a, Fake.fits, Fake.unloadOne));
    try t.expectEqual(@as(usize, 0), a.unloaded);
    // The live shape: GLM's prompt needs 10.5 GB beside 9.4 GB free and an idle 14.7 GB image model.
    var b = Fake{ .needed = 10_558, .available = 9_466, .idle = &.{ 14_730, 8_000 } };
    try t.expect(fitAfterUnloadingIdle(&b, Fake.fits, Fake.unloadOne));
    try t.expectEqual(@as(usize, 1), b.unloaded);
    // Nothing idle left to give back: still refused.
    var c = Fake{ .needed = 50, .available = 9, .idle = &.{ 15, 3 } };
    try t.expect(!fitAfterUnloadingIdle(&c, Fake.fits, Fake.unloadOne));
    try t.expectEqual(@as(usize, 2), c.unloaded);
}

test "a refusal names the credit it COMPARED, never the residency it cannot take" {
    const t = std.testing;
    const mb: u64 = 1024 * 1024;

    // The live shape: the whole residency is the checked-out entry.
    const pinned_only = AdmissionBill{
        .needed = 13_664 * mb,
        .available = 13_374 * mb,
        .evictable = 7_802 * mb,
        .reclaimable = 0,
        .chunk = 512,
    };
    try t.expectEqual(@as(u64, 0), pinned_only.evictionCredit());
    try t.expectEqual(7_802 * mb, pinnedResidentBytes(pinned_only));
    try t.expect(!pinned_only.fits());
    try t.expect(!pinned_only.fitsAfterEviction());
    try t.expectEqual(
        pinned_only.needed <= pinned_only.available +| pinned_only.evictionCredit(),
        pinned_only.fitsAfterEviction(),
    );

    // The same residency, genuinely reclaimable: the credit is real and the request is admitted to evict.
    var reclaimable = pinned_only;
    reclaimable.reclaimable = 7_802 * mb;
    try t.expectEqual(7_802 * mb, reclaimable.evictionCredit());
    try t.expectEqual(@as(u64, 0), pinnedResidentBytes(reclaimable));
    try t.expect(reclaimable.fitsAfterEviction());
}

test "the guard credits only PROVABLY reclaimable cache bytes, never the entry a restore will pin" {
    const t = std.testing;
    // The guard admitted on `available + evictable >= needed`, but the evictable 1,564 MB was
    // the one entry this prompt then restored from; the provable credit is residency minus the largest entry.
    const MB: u64 = 1024 * 1024;
    var bill = AdmissionBill{ .needed = 37_394 * MB, .available = 37_130 * MB, .evictable = 1_564 * MB, .reclaimable = 0 };
    try t.expect(!bill.fits());
    try t.expect(!bill.fitsAfterEviction()); // it DID, before this fix
    bill.reclaimable = 1_564 * MB;
    try t.expect(bill.fitsAfterEviction());
}

test "the admission probe bills the request's OWN kv-quant and chunking, not the process defaults" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // The probe used to hardcode `kv_override = null`, pricing a `kv_quant: 4` request at fp16 on the inference thread.
    const t = std.testing;
    const cfg = qwen4ExpOomConfig();
    const seq: usize = 200_000;
    const q4 = transformer_mod.KVQuantConfig.affine(4);
    const q8 = transformer_mod.KVQuantConfig.affine(8);
    const bill4 = prefillAdmissionBill(&cfg, seq, 2048, q4, null, .{}, .{});
    const bill8 = prefillAdmissionBill(&cfg, seq, 2048, q8, null, .{}, .{});
    const bill16 = prefillAdmissionBill(&cfg, seq, 2048, transformer_mod.KVQuantConfig.dense, null, .{}, .{});
    try t.expect(bill4.needed < bill8.needed);
    try t.expect(bill8.needed < bill16.needed);
}

test "pinnedResidentBytes names the entry a restore would share, and only that" {
    const t = std.testing;
    // Residency the eviction pass cannot return: the connection thread's one signal that this prompt is warm.
    try t.expectEqual(@as(u64, 0), pinnedResidentBytes(.{ .evictable = 0, .reclaimable = 0 }));
    try t.expectEqual(@as(u64, 0), pinnedResidentBytes(.{ .evictable = 4096, .reclaimable = 4096 }));
    // The live 768k reading: 24,826 MB resident, 10,294 MB reclaimable.
    const mb: u64 = 1024 * 1024;
    try t.expectEqual(@as(u64, 14_532 * mb), pinnedResidentBytes(.{ .evictable = 24_826 * mb, .reclaimable = 10_294 * mb }));
    // Saturating: the two numbers are published from different sites.
    try t.expectEqual(@as(u64, 0), pinnedResidentBytes(.{ .evictable = 1, .reclaimable = 2 }));
}

test "prefillAdmissionBill: both admit arms are driven by the two credits" {
    const t = std.testing;
    const MB: u64 = 1024 * 1024;

    const evict = AdmissionBill{ .needed = 30 * MB, .available = 20 * MB, .evictable = 15 * MB, .reclaimable = 15 * MB };
    try t.expect(!evict.fits());
    try t.expect(evict.fitsAfterEviction());
    try t.expectEqual(AdmissionVerdict.evict, admissionVerdict(evict));

    const deferral = AdmissionBill{ .needed = 30 * MB, .available = 20 * MB, .evictable = 15 * MB, .reclaimable = 0 };
    try t.expect(!deferral.fitsAfterEviction());
    try t.expect(pinnedResidentBytes(deferral) > 0); // warm deferral

    // An empty cache carries neither credit, and both arms go dead.
    const empty = AdmissionBill{ .needed = 30 * MB, .available = 20 * MB };
    try t.expectEqual(empty.fits(), empty.fitsAfterEviction());
    try t.expectEqual(@as(u64, 0), pinnedResidentBytes(empty));
    try t.expectEqual(AdmissionVerdict.refuse, admissionVerdict(empty));
    // A bill that fits still admits on both arms.
    const roomy = AdmissionBill{ .needed = 10 * MB, .available = 20 * MB };
    try t.expectEqual(AdmissionVerdict.admit, admissionVerdict(roomy));
}

test "dsv41 admission: the cold g14 1M prompt fits at 4096" {
    const t = std.testing;
    var cfg = try model_mod.parseConfigFromJson(t.allocator, model_mod.dsv41_release_config);
    cfg.pinned_prefill_chunk = 8192;
    const prompt = try t.allocator.alloc(u32, 1 << 20);
    defer t.allocator.free(prompt);
    @memset(prompt, 3);
    const mb: u64 = 1 << 20;
    const available = 13_062 * mb;
    const narrow = prefillNeededAtChunk(&cfg, prompt.len, 256, 16, 4096, .{});
    const wide = prefillNeededAtChunk(&cfg, prompt.len, 256, 16, 8192, .{});
    try t.expect(narrow <= available and wide > available);
    const restore = Dsv41AdmissionTestLimits.set(available);
    defer restore.undo();
    var refused: AdmissionBill = undefined;
    try t.expect(admitPrompt(&cfg, "fixture", prompt, 256, null, false, 0, &refused));
    const bill = prefillAdmissionBill(&cfg, prompt.len, 256, null, null, .{}, .{});
    try t.expectEqual(@as(u64, 4096), bill.chunk);
    try t.expectEqual(narrow, bill.needed);
    try t.expectEqual(available, bill.available);
}

test "dsv41 admission: 32k picks 8192 from a 4096 pin and explicit flags pin or cap at 8192" {
    const t = std.testing;
    var cfg = try model_mod.parseConfigFromJson(t.allocator, model_mod.dsv41_release_config);
    defer cfg.deinit(t.allocator);
    cfg.pinned_prefill_chunk = 4096;
    const prompt32: [32_768]u32 = @splat(3);
    const mb: u64 = 1 << 20;
    const g14_avail = 13_062 * mb;
    const wide32 = prefillNeededAtChunk(&cfg, prompt32.len, 256, 16, 8192, .{});
    const restore = Dsv41AdmissionTestLimits.set(wide32);
    defer restore.undo();

    // Omitted flag: 32k widens from 4096 pin to 8192 and admits.
    var refused: AdmissionBill = undefined;
    try t.expect(admitPrompt(&cfg, "fixture", &prompt32, 256, null, false, 0, &refused));
    try t.expectEqual(@as(u64, 8192), prefillAdmissionBill(&cfg, prompt32.len, 256, null, null, .{}, .{}).chunk);

    // Explicit 4096: pins at 4096 even with room for 8192.
    generate_mod.prefillCli().* = .{ .override = 4096, .explicit = true };
    try t.expectEqual(@as(u64, 4096), prefillAdmissionBill(&cfg, prompt32.len, 256, null, null, .{}, .{}).chunk);

    // Explicit 16384: capped at 8192 in bill and chunk.
    generate_mod.prefillCli().* = .{ .override = 16_384, .explicit = true };
    const capped_bill = prefillAdmissionBill(&cfg, prompt32.len, 256, null, null, .{}, .{});
    try t.expectEqual(@as(u64, 8192), capped_bill.chunk);
    try t.expectEqual(wide32, capped_bill.needed);

    // Explicit 8192 at g14 1M budget preserves its normal fit check and refuses.
    const prompt1m = try t.allocator.alloc(u32, 1 << 20);
    defer t.allocator.free(prompt1m);
    @memset(prompt1m, 3);
    const restore_g14 = Dsv41AdmissionTestLimits.set(g14_avail);
    defer restore_g14.undo();
    generate_mod.prefillCli().* = .{ .override = 8192, .explicit = true };
    try t.expect(!admitPrompt(&cfg, "fixture", prompt1m, 256, null, false, 0, &refused));
    try t.expectEqual(@as(u64, 8192), refused.chunk);
    try t.expect(refused.needed > g14_avail);
}

test "dsv41 admission: image rows fit at 4096 when they push 8192 over budget" {
    const t = std.testing;
    var cfg = try model_mod.parseConfigFromJson(t.allocator, model_mod.dsv41_release_config);
    cfg.pinned_prefill_chunk = 8192;
    cfg.has_vision = true;
    const prompt: [32_768]u32 = @splat(3);
    const available = prefillNeededAtChunk(&cfg, prompt.len, 256, 16, 8192, .{});
    const media_bytes: u64 = 16 << 20;
    try t.expect(prefillNeededAtChunk(&cfg, prompt.len, 256, 16, 4096, .{}) + media_bytes <= available);
    const restore = Dsv41AdmissionTestLimits.set(available);
    defer restore.undo();
    var refused: AdmissionBill = undefined;
    try t.expect(admitPrompt(&cfg, "fixture", &prompt, 256, null, false, media_bytes, &refused));
}

test "dsv41 admission: pending prefill prices decoder company and outstanding claims" {
    const t = std.testing;
    const Slot = @import("../slot.zig").Slot;
    const pending = @import("pending.zig");
    var cfg = try model_mod.parseConfigFromJson(t.allocator, model_mod.dsv41_release_config);
    cfg.pinned_prefill_chunk = 8192;
    const prompt: [32_768]u32 = @splat(3);
    const narrow = prefillNeededAtChunk(&cfg, prompt.len, 256, 16, 4096, .{});
    const wide = prefillNeededAtChunk(&cfg, prompt.len, 256, 16, 8192, .{});
    const promise = wide - narrow;
    const reserve: u64 = 64 << 20;
    const margin = @import("../scheduler/params.zig").SIBLING_MARGIN_BYTES;
    var model: model_registry_mod.LoadedModel = undefined;
    model.config = &cfg;
    var xfm: transformer_mod.Transformer = undefined;
    model.transformer = &xfm;
    var slot: Slot = undefined;
    slot.model = &model;
    slot.full_prompt = @constCast(&prompt);
    slot.cache.config = .dense;
    slot.max_tokens = 256;
    slot.enable_mtp = false;
    slot.prompt_logprobs = false;
    slot.memory_promised = 0;
    slot.memory_hold_logged = false;
    var sibling: Slot = undefined;
    sibling.cancelled = .init(false);
    sibling.finished = false;
    sibling.error_code = null;
    var decoding = [_]*Slot{&sibling};
    var sch: scheduler_mod.Scheduler = undefined;
    sch.io = std.Io.Threaded.global_single_threaded.io();
    sch.queue_mu = .init;
    sch.decoding = .empty;
    sch.decoding.items = &decoding;
    sch.decoding.capacity = decoding.len;
    sch.boundary_outer = null;
    sch.gen_reserve_bytes = reserve;
    const old_hook = scheduler_mod.prefill_admission_numbers;
    defer scheduler_mod.prefill_admission_numbers = old_hook;
    scheduler_mod.prefill_admission_numbers = &prefillBillNumbersNow;

    var active: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    const restore = Dsv41AdmissionTestLimits.set(wide + reserve + margin + promise);
    defer restore.undo();
    sch.promise_ledger = .{ .promised = promise, .base_active = active };
    try t.expect(!pending.slotHoldsForMemory(&sch, &slot));
    try t.expectEqual(narrow, slot.memory_promised);

    sch.decoding = .empty;
    sch.boundary_outer = &sibling;
    sch.promise_ledger = .{ .promised = promise, .base_active = active };
    slot.memory_promised = 0;
    const tighter = Dsv41AdmissionTestLimits.set(narrow + reserve + margin + promise);
    defer tighter.undo();
    try t.expect(!pending.slotHoldsForMemory(&sch, &slot));
    try t.expectEqual(narrow, slot.memory_promised);
    sch.promise_ledger = .{ .promised = promise + 1, .base_active = active };
    slot.memory_promised = 0;
    try t.expect(pending.slotHoldsForMemory(&sch, &slot));
}

test "dsv41 admission: slotPrefillLimits preserves sibling promises after current slot restores" {
    const t = std.testing;
    const Slot = @import("../slot.zig").Slot;
    const pending = @import("pending.zig");
    var cfg = try model_mod.parseConfigFromJson(t.allocator, model_mod.dsv41_release_config);
    defer cfg.deinit(t.allocator);
    const gib: u64 = 1 << 30;
    const sib_promise: u64 = 10 * gib;
    const self_promise: u64 = 10 * gib;
    const margin = @import("../scheduler/params.zig").SIBLING_MARGIN_BYTES;
    var slot: Slot = undefined;
    slot.prompt_logprobs = false;
    slot.memory_promised = self_promise;
    var sibling: Slot = undefined;
    var sch: scheduler_mod.Scheduler = undefined;
    sch.io = std.Io.Threaded.global_single_threaded.io();
    sch.queue_mu = .init;
    sch.decoding = .empty;
    sch.boundary_outer = &sibling;
    sch.gen_reserve_bytes = 0;

    // 10 GiB promised to each of two slots at base_active = 100 GiB; current slot restored 10 GiB
    // before the re-ask (ref_active = 100 GiB, active_now = 110 GiB): sibling's 10 GiB stays intact.
    const ref_active: u64 = 100 * gib;
    sch.promise_ledger = .{ .promised = sib_promise + self_promise, .base_active = ref_active };
    const limits = pending.slotPrefillLimits(&sch, &slot, &cfg, ref_active);
    try t.expectEqual(sib_promise + margin, limits.extra_bytes);

    // With sibling's 10 GiB + margin accounted for, re-ask steps down to 4096 when 8192 would exceed remaining headroom.
    const prompt: [32_768]u32 = @splat(3);
    const wide = prefillNeededAtChunk(&cfg, prompt.len, 256, 16, 8192, .{});
    const avail = wide + sib_promise + margin - 1;
    try t.expectEqual(@as(u32, 4096), chooseRequestPrefillChunk(&cfg, prompt.len, 256, 16, avail, 8192, 0, .{}, limits));
}

const Dsv41AdmissionTestLimits = struct {
    const mem = @import("../../../host/registry/memory_ceiling.zig");
    memory: mem.Overrides,
    cli: @TypeOf(generate_mod.prefillCli().*),

    fn set(available: u64) Dsv41AdmissionTestLimits {
        const old = Dsv41AdmissionTestLimits{ .memory = mem.overrides().*, .cli = generate_mod.prefillCli().* };
        var active: usize = 0;
        _ = mlx.mlx_get_active_memory(&active);
        mem.overrides().* = .{
            .static_ceiling = available + active,
            .wired_limit_mb = 0,
            .machine_free = .{ .idle = available, .request = available },
        };
        generate_mod.prefillCli().* = .{};
        return old;
    }

    fn undo(old: Dsv41AdmissionTestLimits) void {
        mem.overrides().* = old.memory;
        generate_mod.prefillCli().* = old.cli;
    }
};
