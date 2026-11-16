//! Admission of pending requests: the live-memory hold and the single-flight pending tick.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const generate_mod = @import("../generate.zig");
const log = @import("../../../sdk/util/log.zig");
const testing = std.testing;

const holdsForMemory = @import("../batching/verdict.zig").holdsForMemory;
const admissionFits = @import("../batching/verdict.zig").admissionFits;
const liveDecodingCount = @import("../batching/verdict.zig").liveDecodingCount;
const Scheduler = @import("../scheduler.zig").Scheduler;
const Slot = @import("../slot.zig").Slot;
const scheduler_mod = @import("../scheduler.zig");
const HotPrefixCache = @import("../cache/prefix_cache.zig").HotPrefixCache;
const PromiseLedger = @import("../scheduler/params.zig").PromiseLedger;
const publishHotCacheResidency = @import("../cache/budget.zig").publishHotCacheResidency;
const model_mod = @import("../../../sdk/config/model.zig");
const transformer_mod = @import("../transformer.zig");

/// What a pending request is billed against. Every ask re-reads the bill, live memory and the
/// siblings' promises, because an eviction moves all three.
pub const HoldProbe = struct {
    bill_ctx: ?*anyopaque,
    /// `{needed, available}` right now.
    bill_fn: *const fn (?*anyopaque) [2]u64,
    ledger: *const PromiseLedger,
    gen_reserve: u64,
    live: usize,

    /// `credit` stands in for memory an eviction could return: the feasibility check only, never the verdict.
    fn fitsWithCredit(self: *const HoldProbe, credit: u64) bool {
        const bill = self.bill_fn(self.bill_ctx);
        var active_now: usize = 0;
        _ = mlx.mlx_get_active_memory(&active_now);
        return admissionFits(bill[0], bill[1] +| credit, self.ledger.outstanding(active_now), self.gen_reserve, self.live);
    }

    fn fitsNow(ctx: ?*anyopaque) bool {
        const self: *const HoldProbe = @ptrCast(@alignCast(ctx.?));
        return self.fitsWithCredit(0);
    }

    fn hasCompany(self: *const HoldProbe) bool {
        return self.live > 0 or self.gen_reserve > 0;
    }
};

/// Does the request wait? It admits only on memory that is free: with company beside it and no
/// room, the requesting model's own cache gives back LRU entries (never one a live slot holds,
/// never the entry this prompt restores from) until it fits or nothing is left to give.
pub fn holdsAfterEviction(cache: ?*HotPrefixCache, prompt: []const u32, probe: *HoldProbe) bool {
    return holdsForMemory(probe.fitsWithCredit(0), probe.hasCompany());
}

/// Cold bill against live memory; the gated arch runs its own warm pass inside `runPrefill`.
pub fn slotHoldsForMemory(sch: *Scheduler, slot: *Slot) bool {
    const cfg = slot.model.config orelse return false;
    if (cfg.longCtxGated()) return false;
    if (slot.model.transformer == null) return false;
    const numbers_fn = scheduler_mod.prefill_admission_numbers orelse return false;
    // A prefill hosting a nested admission holds memory but is in no list: it is company too.
    const live = liveDecodingCount(sch) + @intFromBool(sch.boundary_outer != null);
    if (live == 0) sch.promise_ledger = .{};
    var numbers = SlotBill{ .cfg = cfg, .slot = slot, .numbers_fn = numbers_fn };
    var probe = HoldProbe{ .bill_ctx = &numbers, .bill_fn = SlotBill.call, .ledger = &sch.promise_ledger, .gen_reserve = sch.gen_reserve_bytes, .live = live };
    const held = holdsAfterEviction(slot.prefix_cache, slot.full_prompt, &probe);
    publishHotCacheResidency(sch);
    const bill = SlotBill.call(&numbers);
    const active_now = activeNow();
    if (!held) {
        if (slot.memory_promised == 0) {
            slot.memory_promised = bill[0];
            sch.promise_ledger.admit(bill[0], active_now);
        }
        return false;
    }
    if (!slot.memory_hold_logged) {
        slot.memory_hold_logged = true;
        const mib = 20;
        log.info("[admission] held: {d} tokens do not fit beside {d} live request(s) and {d} MB of media job; waiting for one to finish (needs {d} MB, {d} MB available, {d} MB promised)\n", .{ slot.full_prompt.len, live, sch.gen_reserve_bytes >> mib, bill[0] >> mib, bill[1] >> mib, sch.promise_ledger.outstanding(active_now) >> mib });
    }
    return true;
}

fn activeNow() u64 {
    var active_now: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_now);
    return active_now;
}

const SlotBill = struct {
    cfg: *const model_mod.ModelConfig,
    slot: *const Slot,
    numbers_fn: *const fn (*const model_mod.ModelConfig, usize, u32, transformer_mod.KVQuantConfig, bool) [2]u64,

    fn call(ctx: ?*anyopaque) [2]u64 {
        const self: *const SlotBill = @ptrCast(@alignCast(ctx.?));
        return self.numbers_fn(self.cfg, self.slot.full_prompt.len, self.slot.max_tokens, self.slot.cache.config, self.slot.enable_mtp);
    }
};

/// One pending-drain candidate (or live decoding slot), reduced to what
/// admission needs: an opaque model identity + the exclusive-decode bit.
pub const AdmitCand = struct { model: usize, exclusive: bool };

/// FIFO admission for one drain tick. An EXCLUSIVE candidate admits only if
/// no EXCLUSIVE `active` slot (live decoding) holds its model and no earlier
/// admitted exclusive candidate claimed it this tick
/// (a same-tick sibling is not in `decoding` yet — the claim covers the
/// window). Non-exclusive candidates always admit — no head-of-line
/// blocking behind a held exclusive request. Writes admitted candidate
/// indices to `out` (queue order preserved), returns the count. Held
/// candidates stay where they are and retry next tick; no release
/// bookkeeping exists — the busy signal IS presence in `decoding`.
pub fn admitPendingTick(cands: []const AdmitCand, active: []const AdmitCand, out: []usize) usize {
    var n: usize = 0;
    outer: for (cands, 0..) |c, i| {
        if (n >= out.len) break;
        if (c.exclusive) {
            for (active) |a| if (a.exclusive and a.model == c.model) continue :outer;
            for (out[0..n]) |j| {
                if (cands[j].exclusive and cands[j].model == c.model) continue :outer;
            }
        }
        out[n] = i;
        n += 1;
    }
    return n;
}

// ── tests ──

test "admitPendingTick: per-slot exclusivity (qwen4 MTP slot) blocks only its own class" {
    const A: usize = 0xA0;
    var out: [16]usize = undefined;
    // An MTP candidate beside a live PLAIN slot on the same model admits.
    {
        const cands = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        const active = [_]AdmitCand{.{ .model = A, .exclusive = false }};
        try testing.expectEqual(@as(usize, 1), admitPendingTick(&cands, &active, &out));
    }
    // An MTP candidate beside a live MTP slot holds.
    {
        const cands = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        const active = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        try testing.expectEqual(@as(usize, 0), admitPendingTick(&cands, &active, &out));
    }
    // A plain candidate beside a live MTP slot admits.
    {
        const cands = [_]AdmitCand{.{ .model = A, .exclusive = false }};
        const active = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        try testing.expectEqual(@as(usize, 1), admitPendingTick(&cands, &active, &out));
    }
}

test "admitPendingTick: exclusive single-flight FIFO contract" {
    const A: usize = 0xA0;
    const B: usize = 0xB0;
    var out: [16]usize = undefined;

    // Held while a live slot on the same exclusive model is decoding — the
    // dsv4 class: a second admitted slot deinit+rebuilds the module-owned
    // dec_state at cache.step==0 and both requests then interleave on it.
    {
        const cands = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        const active = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        try testing.expectEqual(@as(usize, 0), admitPendingTick(&cands, &active, &out));
    }
    // Admits once no live slot holds the model.
    {
        const cands = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        try testing.expectEqual(@as(usize, 1), admitPendingTick(&cands, &.{}, &out));
        try testing.expectEqual(@as(usize, 0), out[0]);
    }
    // Two exclusive candidates on the SAME model in one tick: only the
    // first admits — the tick-claim covers slots not yet in `decoding`.
    {
        const cands = [_]AdmitCand{
            .{ .model = A, .exclusive = true },
            .{ .model = A, .exclusive = true },
        };
        try testing.expectEqual(@as(usize, 1), admitPendingTick(&cands, &.{}, &out));
        try testing.expectEqual(@as(usize, 0), out[0]);
    }
    // Distinct exclusive models admit independently.
    {
        const cands = [_]AdmitCand{
            .{ .model = A, .exclusive = true },
            .{ .model = B, .exclusive = true },
        };
        try testing.expectEqual(@as(usize, 2), admitPendingTick(&cands, &.{}, &out));
        try testing.expectEqual(@as(usize, 0), out[0]);
        try testing.expectEqual(@as(usize, 1), out[1]);
    }
}

test "admitPendingTick: non-exclusive concurrency and queue order preserved" {
    const A: usize = 0xA0;
    const L: usize = 0x10;
    var out: [16]usize = undefined;

    // Non-exclusive candidates (laguna/hy3: per-slot state) admit freely
    // even beside live slots — their serial-tick interleave is safe.
    {
        const cands = [_]AdmitCand{
            .{ .model = L, .exclusive = false },
            .{ .model = L, .exclusive = false },
        };
        const active = [_]AdmitCand{.{ .model = L, .exclusive = false }};
        try testing.expectEqual(@as(usize, 2), admitPendingTick(&cands, &active, &out));
    }
    // A held exclusive candidate must not head-of-line-block a later
    // candidate on another model (requests for OTHER models keep flowing).
    {
        const cands = [_]AdmitCand{
            .{ .model = A, .exclusive = true },
            .{ .model = L, .exclusive = false },
        };
        const active = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        try testing.expectEqual(@as(usize, 1), admitPendingTick(&cands, &active, &out));
        try testing.expectEqual(@as(usize, 1), out[0]);
    }
    // `out` caps the admitted count (mirrors to_prefill's 16), in order.
    {
        const cands = [_]AdmitCand{
            .{ .model = L, .exclusive = false },
            .{ .model = L, .exclusive = false },
            .{ .model = L, .exclusive = false },
        };
        var small: [2]usize = undefined;
        try testing.expectEqual(@as(usize, 2), admitPendingTick(&cands, &.{}, &small));
        try testing.expectEqual(@as(usize, 0), small[0]);
        try testing.expectEqual(@as(usize, 1), small[1]);
    }
}

const gib: u64 = 1 << 30;

/// A resident entry with a billed size and no arrays: eviction frees nothing real, the fake bill below credits what was billed.
fn testEntry(hc: *HotPrefixCache, tokens: []const u32, kv_bytes: u64, used: u64, checked_out_by: ?usize) !void {
    try hc.entries.append(testing.allocator, .{
        .tokens = try testing.allocator.dupe(u32, tokens),
        .has_tools = false,
        .snapshot = .{ .entries = try testing.allocator.alloc(transformer_mod.KVCacheEntry, 0), .step = 0, .allocator = testing.allocator, .config = transformer_mod.KVQuantConfig.dense },
        .last_used = used,
        .quant_config = transformer_mod.KVQuantConfig.dense,
        .kv_bytes = kv_bytes,
        .ssm_checkpoints = null,
        .ssm_bytes = 0,
        .checked_out_by = checked_out_by,
    });
    hc.current_kv_bytes += kv_bytes;
}

/// The requested model's bill: `base` free plus whatever its cache has given back since `resident0`.
const FakeBill = struct {
    cache: *const HotPrefixCache,
    resident0: u64,
    need: u64,
    base: u64,

    fn init(cache: *const HotPrefixCache, need: u64, base: u64) FakeBill {
        return .{ .cache = cache, .resident0 = cache.residentBytes(), .need = need, .base = base };
    }

    fn call(ctx: ?*anyopaque) [2]u64 {
        const self: *const FakeBill = @ptrCast(@alignCast(ctx.?));
        return .{ self.need, self.base +| (self.resident0 -| self.cache.residentBytes()) };
    }

    fn probe(self: *FakeBill, ledger: *const PromiseLedger, gen_reserve: u64, live: usize) HoldProbe {
        return .{ .bill_ctx = self, .bill_fn = call, .ledger = ledger, .gen_reserve = gen_reserve, .live = live };
    }
};

fn promptOf(first: u32) [600]u32 {
    var t: [600]u32 = undefined;
    for (&t, 0..) |*x, i| x.* = first + @as(u32, @intCast(i));
    return t;
}

test "admission: a full RAM tier no longer serializes the requests: the hold evicts the cache and decides on what it freed" {
    // iQ at W=4: the 8 GB tier sits inside active memory, 5 GB of 13 GB is free, the second 3.5 GB bill waits.
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    const other = promptOf(10_000);
    try testEntry(&hc, &other, 6 * gib, 1, null);
    var fake = FakeBill.init(&hc, 7 * gib / 2, 5 * gib);
    var ledger: PromiseLedger = .{};
    var probe = fake.probe(&ledger, 0, 1);

    const prompt = promptOf(1);
    try testing.expect(!holdsAfterEviction(&hc, &prompt, &probe));
    try testing.expectEqual(@as(usize, 0), hc.entryCount());
    try testing.expect(probe.fitsWithCredit(0));
}

test "admission: a request alone proceeds without evicting, and the entry its prompt restores from is spared" {
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    const mine = promptOf(1);
    try testEntry(&hc, &mine, 6 * gib, 1, null);
    var ledger: PromiseLedger = .{};

    var fake = FakeBill.init(&hc, 7 * gib / 2, 1 * gib);
    var alone = fake.probe(&ledger, 0, 0);
    try testing.expect(!holdsAfterEviction(&hc, &mine, &alone));
    try testing.expectEqual(@as(usize, 1), hc.entryCount());

    var beside = fake.probe(&ledger, 0, 1);
    try testing.expect(holdsAfterEviction(&hc, &mine, &beside));
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
}

test "admission: a running media job's reserve and the siblings' promises are covered by freed memory, never by credit" {
    var ledger: PromiseLedger = .{};
    ledger.admit(3 * gib, activeNow()); // a sibling promised 3 GiB it has not allocated
    const prompt = promptOf(1);
    const other = promptOf(10_000);

    // 3 GiB need + 3 GiB promised + 4 GiB media + 2 GiB margin = 12 GiB against 5 GiB free.
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        defer hc.deinit();
        try testEntry(&hc, &other, 8 * gib, 1, null);
        var fake = FakeBill.init(&hc, 3 * gib, 5 * gib);
        var probe = fake.probe(&ledger, 4 * gib, 1);
        try testing.expect(!holdsAfterEviction(&hc, &prompt, &probe));
        try testing.expectEqual(@as(usize, 0), hc.entryCount());
        try testing.expect(probe.fitsWithCredit(0)); // 13 GiB free after the eviction, not 5 + 8 on faith
    }
    // The cache holds too little to ever make room: it is kept, and the request waits.
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        defer hc.deinit();
        try testEntry(&hc, &other, 6 * gib, 1, null);
        var fake = FakeBill.init(&hc, 3 * gib, 5 * gib);
        var probe = fake.probe(&ledger, 4 * gib, 1);
        try testing.expect(holdsAfterEviction(&hc, &prompt, &probe));
        try testing.expectEqual(@as(usize, 1), hc.entryCount());
    }
}

test "admission: a prefix cache checkout held by a live slot is never evicted or credited" {
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    const held = promptOf(10_000);
    try testEntry(&hc, &held, 6 * gib, 1, 42);
    var fake = FakeBill.init(&hc, 7 * gib / 2, 5 * gib);
    var ledger: PromiseLedger = .{};
    var probe = fake.probe(&ledger, 0, 1);

    const prompt = promptOf(1);
    try testing.expect(holdsAfterEviction(&hc, &prompt, &probe));
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqual(@as(?usize, 42), hc.entries.items[0].checked_out_by);
}

test "admission: another model's cache is never touched for this model's request" {
    var mine = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer mine.deinit();
    var theirs = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer theirs.deinit();
    const b_prompt = promptOf(10_000);
    try testEntry(&theirs, &b_prompt, 6 * gib, 1, null);
    var fake = FakeBill.init(&mine, 5 * gib, 4 * gib);
    var ledger: PromiseLedger = .{};
    var probe = fake.probe(&ledger, 0, 1);

    const prompt = promptOf(1);
    try testing.expect(holdsAfterEviction(&mine, &prompt, &probe));
    try testing.expectEqual(@as(usize, 1), theirs.entryCount());
}
