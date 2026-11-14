//! In-process tests for plan 60: peer prefix-cache export, wire import, identity/stale/memory refusals, and HTTP gates.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const sdk = @import("../../sdk/root.zig");
const model = @import("../../sdk/config/model.zig");
const driver = @import("../../engines/mlx/plugin/driver.zig");
const fixture = @import("../deepseek_v4/fixture.zig");
const cache_transfer = @import("../../host/api/cache_transfer.zig");
const prefix_cache = @import("../../engines/mlx/cache/prefix_cache.zig");
const peer_wire = @import("../../engines/mlx/cache/peer_wire.zig");
const peer_transfer = @import("../../engines/mlx/cache/peer_transfer.zig");
const prefill_request = @import("../../engines/mlx/prefill.zig");
const decode_request = @import("../../engines/mlx/decode.zig");
const commit_request = @import("../../engines/mlx/commit.zig");
const finish_request = @import("../../engines/mlx/finish.zig");
const scheduler = @import("../../engines/mlx/scheduler.zig");
const RequestSlot = @import("../../engines/mlx/slot.zig").Slot;
const KVQuantConfig = @import("../../engines/mlx/cache/kv_quant.zig").KVQuantConfig;
const Tokenizer = @import("../../sdk/tokenizer/tokenizer.zig").Tokenizer;
const registry = @import("../../host/registry/registry.zig");
const cpu_state = @import("../../host/registry/cpu_state.zig");
const memory_ceiling = @import("../../host/registry/memory_ceiling.zig");
const http_server = @import("../../host/http/server.zig");
const Conn = @import("../../host/http/conn.zig").Conn;
const router = @import("../../host/http/router.zig");
const HostScheduler = @import("../../host/gpu_thread/scheduler.zig").Scheduler;
const mlx_engine = @import("../../host/engines/mlx.zig");

const testing = std.testing;

const PeerHarness = struct {
    tmp: testing.TmpDir,
    path_buf: [512]u8 = undefined,
    path_len: usize = 0,
    cfg: model.ModelConfig,
    xfm: @import("../../engines/mlx/transformer.zig").Transformer,
    sch: scheduler.Scheduler = undefined,
    entry: *registry.LoadedModel,
    tok: Tokenizer,

    fn open(files: []const sdk.fixture.File) !*PeerHarness {
        const a = testing.allocator;
        const io = std.Io.Threaded.global_single_threaded.io();
        const h = try a.create(PeerHarness);
        errdefer a.destroy(h);
        h.tmp = testing.tmpDir(.{ .iterate = true });
        errdefer h.tmp.cleanup();
        for (files) |f| try h.tmp.dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });
        h.path_len = try h.tmp.dir.realPath(io, &h.path_buf);
        const dir = h.path_buf[0..h.path_len];

        h.cfg = try model.parseConfig(io, a, dir);
        errdefer h.cfg.deinit(a);
        h.xfm = try driver.load(io, a, &h.cfg, dir, mlx.gpuStream());
        errdefer h.xfm.deinit();

        const reg = try registry.ModelRegistry.init(a, io, null, 4, 0, null);
        errdefer reg.deinit();
        h.entry = try reg.registerStub("fixture", dir, 1);
        h.tok = Tokenizer.initEmptyForTests(a, .byte_level_bpe);
        errdefer h.tok.deinit();
        h.entry.config = &h.cfg;
        h.entry.transformer = &h.xfm;
        h.entry.tokenizer = &h.tok;
        h.entry.state = .ready;
        h.entry.prefix_cache = try a.create(prefix_cache.HotPrefixCache);
        h.entry.prefix_cache.?.* = prefix_cache.HotPrefixCache.init(a, 8);

        h.sch = .{
            .allocator = a,
            .io = io,
            .registry = reg,
            .current_model = h.entry,
            .xfm = null,
            .weights = null,
            .vision_encoder = null,
            .drafter = null,
            .drafter_block_size = 4,
            .kv_quant_config = KVQuantConfig.dense,
            .gguf_ctx_size = 0,
            .prefix_cache_capacity = 8,
            .prefix_cache_ram_enabled = true,
            .prefix_cache_mem_bytes = 0,
            .prefix_cache_mem_resolver = null,
            .prefix_cache_disk_bytes = 0,
            .ssm_checkpoint_stride = 0,
            .ssm_checkpoint_max = 16,
            .mtp_enabled = false,
            .mtp_head_kv_quant = false,
            .mtp_depth = 0,
            .llama = .{},
            .ds4_mtp = false,
            .ds4_dspark = false,
            .ds4_ssd_streaming = false,
            .ane_prefill = false,
            .ane = false,
            .ane_slots = 1,
            .ane_chunk_resolver = null,
            .ane_headroom_resolver = null,
            .no_drafter = true,
            .drafter_dir = "",
            .primary_model_dir = dir,
            .draft_block_size = 4,
            .draft_block_size_explicit = false,
            .drafter_path = "",
            .hot_prefix_cache = h.entry.prefix_cache,
            .max_concurrent = 4,
            .force_batched = false,
            .queue_mu = .init,
            .queue_cond = .init,
            .pending = .empty,
            .decoding = .empty,
            .vision_queue = .empty,
            .embed_queue = .empty,
            .load_queue = .empty,
            .gen_queue = .empty,
            .unload_queue = .empty,
            .cleanup_queue = .empty,
            .metrics = null,
            .inflight_generated_tokens = .init(0),
            .inflight_prefill_tokens = .init(0),
            .inflight_prefill_expected = .init(0),
            .requests_prefilling = .init(0),
            .in_flight = 0,
            .queue_cap = 36,
            .submit_cond = .init,
            .session_cond = .init,
            .inference_thread = null,
            .shutdown = .init(false),
            .started = .init(true),
            .started_mu = .init,
            .started_cond = .init,
            .load_failed = .init(false),
            .load_error_name = null,
        };
        return h;
    }

    fn close(h: *PeerHarness) void {
        h.entry.transformer = null;
        h.entry.config = null;
        h.entry.tokenizer = null;
        h.sch.registry.deinit();
        h.sch.pending.deinit(testing.allocator);
        h.sch.decoding.deinit(testing.allocator);
        h.sch.offloaded.deinit(testing.allocator);
        h.sch.vision_queue.deinit(testing.allocator);
        h.sch.embed_queue.deinit(testing.allocator);
        h.sch.load_queue.deinit(testing.allocator);
        h.sch.gen_queue.deinit(testing.allocator);
        h.sch.unload_queue.deinit(testing.allocator);
        h.sch.cleanup_queue.deinit(testing.allocator);
        h.sch.cache_export_queue.deinit(testing.allocator);
        h.sch.cache_import_queue.deinit(testing.allocator);
        testing.allocator.free(h.sch.hot_cache_digests);
        h.tok.deinit();
        h.xfm.deinit();
        h.cfg.deinit(testing.allocator);
        h.tmp.cleanup();
        testing.allocator.destroy(h);
    }

    fn attachDisk(h: *PeerHarness) !void {
        const dir = h.path_buf[0..h.path_len];
        const hc = h.entry.prefix_cache.?;
        hc.disk = try @import("../../engines/mlx/load.zig").initPrefixDisk(
            testing.allocator,
            h.sch.io,
            dir,
            "peer-disk",
            0,
            512,
            prefix_cache.ModuleBinding.init(&h.xfm).?,
            dir,
        );
        hc.disk.?.armTestSpace(1 << 40, 2 << 40);
        hc.armDiskTier(&h.cfg);
        hc.module_quiet_ms = 0;
    }

    fn runGreedy(h: *PeerHarness, prompt: []const u32, max_tokens: u32, commit: bool) !struct { cached: u32, tokens: []u32 } {
        const s = try RequestSlot.init(testing.allocator, h.sch.io, h.entry.config.?, .{
            .model = h.entry,
            .prompt_ids = prompt,
            .sampling = .{ .temperature = 0 },
            .eos_token_ids = &.{},
            .max_tokens = max_tokens,
        }, h.sch.kv_quant_config);
        defer s.deinit();
        try prefill_request.runPrefill(&h.sch, s);
        while (s.completion_tokens < max_tokens and !s.finished) {
            try decode_request.runSingleDecodeTick(&h.sch, s);
        }
        const gen_ids = try testing.allocator.dupe(u32, s.legacy_gen.?.generated_ids.items);
        errdefer testing.allocator.free(gen_ids);
        const cached = s.cached_tokens;
        if (!s.finished) {
            if (commit) commit_request.commitSlotIfApplicable(&h.sch, s);
            finish_request.releasePluginSlot(s);
        }
        return .{ .cached = cached, .tokens = gen_ids };
    }
};

fn buildPrompt(comptime len: usize) [len]u32 {
    var out: [len]u32 = undefined;
    for (&out, 0..) |*id, i| id.* = @intCast(3 + (i * 7) % 80);
    return out;
}

test "dsv41 peer fetch: v41ServableFiles builds a servable pack with byte tokenizer and chat template" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const files = try fixture.v41ServableFiles(arena.allocator(), 7);
    for (files) |f| try tmp.dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });
    var buf: [512]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(io, &buf)];

    var state = try cpu_state.preloadCpuState(testing.allocator, io, dir, 0, null);
    defer cpu_state.freeCpuState(testing.allocator, &state);
    try testing.expectEqual(@as(u32, 320), state.config.vocab_size);
    try testing.expect(state.chat_config.chat_template.len > 0);
    const ids = try state.tok.encode(testing.allocator, "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>");
    defer testing.allocator.free(ids);
    try testing.expect(ids.len >= 8);
    try testing.expectEqual(@as(u32, 256), ids[0]);
}

test "dsv41 peer fetch: export from model A and importWireModule into model B preserves cached length and greedy output" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const model_a = try PeerHarness.open(files);
    defer model_a.close();
    const model_b = try PeerHarness.open(files);
    defer model_b.close();

    const full_prompt = buildPrompt(33);
    const turn1 = try model_a.runGreedy(full_prompt[0..25], 2, true);
    defer testing.allocator.free(turn1.tokens);
    try testing.expectEqual(@as(u32, 0), turn1.cached);
    try testing.expectEqual(@as(usize, 1), model_a.entry.prefix_cache.?.entries.items.len);

    var expired_export: @import("../../host/gpu_thread/requests.zig").CacheExportRequest = .{
        .model = model_a.entry,
        .prompt_ids = &full_prompt,
        .has_tools = false,
        .cache_key = 42,
        .max_bytes = router.max_cache_export_bytes,
        .deadline_ms = 0,
        .allocator = testing.allocator,
    };
    peer_transfer.runCacheExportRequest(&model_a.sch, &expired_export);
    try testing.expectEqual(@as(@TypeOf(expired_export.outcome), .deadline_passed), expired_export.outcome);

    const exported = try peer_transfer.exportWireModule(
        testing.allocator,
        model_a.sch.io,
        &model_a.sch,
        model_a.entry,
        &full_prompt,
        false,
        null,
        0,
        42,
        router.max_cache_export_bytes,
    );
    try testing.expect(exported == .ok);
    const wire = exported.ok.bytes;
    defer testing.allocator.free(wire);
    try testing.expectEqual(@as(usize, 24), exported.ok.n);
    var future_export: @import("../../host/gpu_thread/requests.zig").CacheExportRequest = .{
        .model = model_a.entry,
        .prompt_ids = &full_prompt,
        .has_tools = false,
        .cache_key = 42,
        .max_bytes = router.max_cache_export_bytes,
        .deadline_ms = std.math.maxInt(u64),
        .allocator = testing.allocator,
    };
    peer_transfer.runCacheExportRequest(&model_a.sch, &future_export);
    try testing.expect(future_export.outcome == .ok);
    testing.allocator.free(future_export.outcome.ok.bytes);
    const capped = try peer_transfer.exportWireModule(
        testing.allocator,
        model_a.sch.io,
        &model_a.sch,
        model_a.entry,
        &full_prompt,
        false,
        null,
        0,
        42,
        wire.len - 1,
    );
    try testing.expectEqual(@as(@TypeOf(capped), .{ .too_large = @intCast(wire.len) }), capped);

    var expired_import: @import("../../host/gpu_thread/requests.zig").CacheImportRequest = .{
        .model = model_b.entry,
        .body = wire,
        .deadline_ms = 0,
        .allocator = testing.allocator,
    };
    peer_transfer.runCacheImportRequest(&model_b.sch, &expired_import);
    try testing.expectEqual(@as(@TypeOf(expired_import.outcome), .{ .refused = .deadline_passed }), expired_import.outcome);
    try testing.expectEqual(@as(usize, 0), model_b.entry.prefix_cache.?.entries.items.len);

    var future_import: @import("../../host/gpu_thread/requests.zig").CacheImportRequest = .{
        .model = model_b.entry,
        .body = wire,
        .deadline_ms = std.math.maxInt(u64),
        .allocator = testing.allocator,
    };
    peer_transfer.runCacheImportRequest(&model_b.sch, &future_import);
    const imported = future_import.outcome;
    try testing.expect(imported == .ok);
    try testing.expectEqual(@as(usize, 24), imported.ok.n);
    try testing.expectEqual(@as(usize, 1), model_b.entry.prefix_cache.?.entries.items.len);

    const warm_a = try model_a.runGreedy(&full_prompt, 6, false);
    defer testing.allocator.free(warm_a.tokens);
    const warm_b = try model_b.runGreedy(&full_prompt, 6, false);
    defer testing.allocator.free(warm_b.tokens);

    try testing.expectEqual(@as(u32, 24), warm_a.cached);
    try testing.expectEqual(warm_a.cached, warm_b.cached);
    try testing.expectEqualSlices(u32, warm_a.tokens, warm_b.tokens);

    // Disk-only record on holder is not exported (RAM-only match)
    {
        const disk_only = try PeerHarness.open(files);
        defer disk_only.close();
        try disk_only.attachDisk();
        var disk_prompt: [513]u32 = undefined;
        for (&disk_prompt, 0..) |*id, i| id.* = @intCast(3 + i % 80);
        const seeded = try disk_only.runGreedy(&disk_prompt, 2, true);
        defer testing.allocator.free(seeded.tokens);
        const hc = disk_only.entry.prefix_cache.?;
        const disk_id = hc.entries.items[0].module.?.disk_id.?;
        for (0..32) |_| {
            const r = hc.flushOwedSlice(disk_only.xfm.s);
            hc.disk.?.drainWriter();
            hc.disk.?.pollModuleWrites();
            if (r == .done) break;
        }
        try testing.expect(hc.disk.?.entryWholeOnDisk(disk_id));
        hc.setBudget(1);
        hc.setBudget(0);
        try testing.expectEqual(@as(usize, 0), hc.entries.items.len);
        const disk_match = hc.findModuleMatch(prefix_cache.ModuleBinding.init(&disk_only.xfm).?, &disk_prompt, false, &.{}, null, KVQuantConfig.dense);
        try testing.expect(disk_match != null and disk_match.?.source == .disk);
        const disk_exp = try peer_transfer.exportWireModule(
            testing.allocator,
            disk_only.sch.io,
            &disk_only.sch,
            disk_only.entry,
            &disk_prompt,
            false,
            null,
            0,
            0,
            router.max_cache_export_bytes,
        );
        try testing.expectEqual(@as(@TypeOf(disk_exp), .not_found), disk_exp);
    }
}

test "dsv41 peer fetch: importWireModule refuses layout key, pack identity, settings, quant config, stale entry and low memory headroom" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const model_a = try PeerHarness.open(files);
    defer model_a.close();
    const model_b = try PeerHarness.open(files);
    defer model_b.close();

    const full_prompt = buildPrompt(33);
    const seed_short = try model_a.runGreedy(full_prompt[0..17], 2, true);
    defer testing.allocator.free(seed_short.tokens);
    const exp_short = try peer_transfer.exportWireModule(
        testing.allocator,
        model_a.sch.io,
        &model_a.sch,
        model_a.entry,
        full_prompt[0..25],
        false,
        null,
        0,
        0,
        router.max_cache_export_bytes,
    );
    const wire_short = exp_short.ok.bytes;
    defer testing.allocator.free(wire_short);
    try testing.expectEqual(@as(usize, 16), exp_short.ok.n);

    const seed_long = try model_a.runGreedy(full_prompt[0..25], 2, true);
    defer testing.allocator.free(seed_long.tokens);
    const exp_long = try peer_transfer.exportWireModule(
        testing.allocator,
        model_a.sch.io,
        &model_a.sch,
        model_a.entry,
        &full_prompt,
        false,
        null,
        0,
        0,
        router.max_cache_export_bytes,
    );
    const wire_long = exp_long.ok.bytes;
    defer testing.allocator.free(wire_long);
    try testing.expectEqual(@as(usize, 24), exp_long.ok.n);

    // 1. layout_key refusal
    {
        const bad_layout = try testing.allocator.dupe(u8, wire_long);
        defer testing.allocator.free(bad_layout);
        bad_layout[32] ^= 1;
        peer_wire.stampChecksum(bad_layout);
        const out = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, bad_layout);
        try testing.expectEqual(@as(@TypeOf(out), .{ .refused = .layout_key }), out);
    }

    // 2. pack_identity refusal (changed config.json and changed weight file size)
    {
        const orig_cfg = try model_b.tmp.dir.readFileAlloc(model_b.sch.io, "config.json", testing.allocator, .limited(1 << 20));
        defer testing.allocator.free(orig_cfg);
        const alt_cfg = try std.fmt.allocPrint(testing.allocator, "{s} ", .{orig_cfg});
        defer testing.allocator.free(alt_cfg);
        try model_b.tmp.dir.writeFile(model_b.sch.io, .{ .sub_path = "config.json", .data = alt_cfg });
        model_b.entry.pack_identity = null;
        const out_cfg = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire_long);
        try testing.expectEqual(@as(@TypeOf(out_cfg), .{ .refused = .pack_identity }), out_cfg);
        try model_b.tmp.dir.writeFile(model_b.sch.io, .{ .sub_path = "config.json", .data = orig_cfg });

        try model_b.tmp.dir.writeFile(model_b.sch.io, .{ .sub_path = "extra.safetensors", .data = "extra" });
        model_b.entry.pack_identity = null;
        const out_size = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire_long);
        try testing.expectEqual(@as(@TypeOf(out_size), .{ .refused = .pack_identity }), out_size);
        try model_b.tmp.dir.deleteFile(model_b.sch.io, "extra.safetensors");
        model_b.entry.pack_identity = null;
    }

    // 3. settings refusal (changed model flag)
    {
        model_b.cfg.lm_head_mxfp8_override = true;
        defer model_b.cfg.lm_head_mxfp8_override = null;
        const out = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire_long);
        try testing.expectEqual(@as(@TypeOf(out), .{ .refused = .settings }), out);
    }

    // 4. quant_config refusal
    {
        model_b.sch.kv_quant_config = KVQuantConfig.affine(8);
        defer model_b.sch.kv_quant_config = KVQuantConfig.dense;
        const out = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire_long);
        try testing.expectEqual(@as(@TypeOf(out), .{ .refused = .quant_config }), out);
    }

    // 5. no memory headroom refusal before import_arrays
    {
        const limits = memory_ceiling.overrides();
        const prev = limits.*;
        defer limits.* = prev;
        limits.* = .{ .static_ceiling = 1, .wired_limit_mb = 0 };
        const out = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire_long);
        try testing.expectEqual(@as(@TypeOf(out), .{ .refused = .no_memory }), out);
        try testing.expectEqual(@as(usize, 0), model_b.entry.prefix_cache.?.entries.items.len);
    }

    // 6. stale check: shorter resident rung accepts a longer tip; duplicate and shorter incoming tips refuse
    {
        const ok_short = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire_short);
        try testing.expect(ok_short == .ok);
        const dup_short = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire_short);
        try testing.expectEqual(@as(@TypeOf(dup_short), .{ .refused = .stale }), dup_short);

        const ok_long = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire_long);
        try testing.expect(ok_long == .ok);
        const stale_short = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire_short);
        try testing.expectEqual(@as(@TypeOf(stale_short), .{ .refused = .stale }), stale_short);
    }
}

test "dsv41 peer fetch: a pack with the same config, names and sizes but other weights refuses the import as pack_identity" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files_a = try fixture.v41Files(arena.allocator(), 7);
    const files_b = try fixture.v41Files(arena.allocator(), 8);
    try testing.expectEqual(files_a[1].bytes.len, files_b[1].bytes.len);
    const model_a = try PeerHarness.open(files_a);
    defer model_a.close();
    const model_b = try PeerHarness.open(files_b);
    defer model_b.close();

    const prompt = buildPrompt(25);
    const seeded = try model_a.runGreedy(prompt[0..17], 2, true);
    defer testing.allocator.free(seeded.tokens);
    const exp = try peer_transfer.exportWireModule(testing.allocator, model_a.sch.io, &model_a.sch, model_a.entry, &prompt, false, null, 0, 0, router.max_cache_export_bytes);
    const wire = exp.ok.bytes;
    defer testing.allocator.free(wire);
    const out = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire);
    try testing.expectEqual(@as(@TypeOf(out), .{ .refused = .pack_identity }), out);
}

test "dsv41 peer fetch: a file-backed model answers transfer_unsupported on export and import" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const h = try PeerHarness.open(try fixture.v41Files(arena.allocator(), 7));
    defer h.close();
    try h.tmp.dir.writeFile(h.sch.io, .{ .sub_path = "model.gguf", .data = "GGUF" });
    const gguf = try std.fmt.allocPrint(arena.allocator(), "{s}/model.gguf", .{h.path_buf[0..h.path_len]});
    const dir_path = h.entry.path;
    h.entry.path = gguf;
    defer h.entry.path = dir_path;

    const prompt = buildPrompt(25);
    const exp = try peer_transfer.exportWireModule(testing.allocator, h.sch.io, &h.sch, h.entry, &prompt, false, null, 0, 0, router.max_cache_export_bytes);
    try testing.expectEqual(@as(@TypeOf(exp), .transfer_unsupported), exp);
    const imp = try peer_transfer.importWireModule(testing.allocator, h.sch.io, &h.sch, h.entry, "abc");
    try testing.expectEqual(@as(@TypeOf(imp), .{ .refused = .transfer_unsupported }), imp);
}

test "dsv41 peer fetch: an import that cannot fit is refused before it evicts a resident" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const model_a = try PeerHarness.open(files);
    defer model_a.close();
    const model_b = try PeerHarness.open(files);
    defer model_b.close();

    const full_prompt = buildPrompt(33);
    const seed = try model_a.runGreedy(full_prompt[0..25], 2, true);
    defer testing.allocator.free(seed.tokens);
    const exp = try peer_transfer.exportWireModule(testing.allocator, model_a.sch.io, &model_a.sch, model_a.entry, &full_prompt, false, null, 0, 0, router.max_cache_export_bytes);
    const wire = exp.ok.bytes;
    defer testing.allocator.free(wire);

    var other: [25]u32 = undefined;
    for (&other, 0..) |*id, i| id.* = @intCast(5 + (i * 11) % 70);
    const r1 = try model_b.runGreedy(other[0..17], 2, true);
    defer testing.allocator.free(r1.tokens);
    other[0] = 90;
    const r2 = try model_b.runGreedy(other[0..17], 2, true);
    defer testing.allocator.free(r2.tokens);
    const hc = model_b.entry.prefix_cache.?;
    try testing.expectEqual(@as(usize, 2), hc.entries.items.len);

    {
        const limits = memory_ceiling.overrides();
        const prev = limits.*;
        defer limits.* = prev;
        limits.* = .{ .static_ceiling = 1, .wired_limit_mb = 0 };
        const out = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire);
        try testing.expectEqual(@as(@TypeOf(out), .{ .refused = .no_memory }), out);
        try testing.expectEqual(@as(usize, 2), hc.entries.items.len);
    }
    {
        const prev = hc.max_kv_bytes;
        defer hc.max_kv_bytes = prev;
        hc.max_kv_bytes = 1;
        const out = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire);
        try testing.expectEqual(@as(@TypeOf(out), .{ .refused = .commit_declined }), out);
        try testing.expectEqual(@as(usize, 2), hc.entries.items.len);
    }
    {
        hc.ram_enabled = false;
        defer hc.ram_enabled = true;
        const out = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire);
        try testing.expectEqual(@as(@TypeOf(out), .{ .refused = .commit_declined }), out);
        try testing.expectEqual(@as(usize, 2), hc.entries.items.len);
    }
}

test "dsv41 peer fetch: an import_arrays failure on a memory latch answers no_memory and inserts nothing" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const model_a = try PeerHarness.open(files);
    defer model_a.close();
    const model_b = try PeerHarness.open(files);
    defer model_b.close();

    const full_prompt = buildPrompt(25);
    const turn1 = try model_a.runGreedy(full_prompt[0..17], 2, true);
    defer testing.allocator.free(turn1.tokens);
    const exp = try peer_transfer.exportWireModule(testing.allocator, model_a.sch.io, &model_a.sch, model_a.entry, &full_prompt, false, null, 0, 0, router.max_cache_export_bytes);
    const wire = exp.ok.bytes;
    defer testing.allocator.free(wire);

    const Probe = struct {
        var latch_msg: [:0]const u8 = "";

        fn importArrays(_: *anyopaque, _: abi.Span(u32), _: abi.Str, _: abi.Span(abi.AuxTensor), out: *abi.Handle) callconv(.c) abi.Verdict {
            out.* = null;
            mlx.latchErrorForTest(latch_msg);
            return abi.Verdict.refuse(.unsupported, "forced import failure");
        }
    };

    const bound = model_b.xfm.plugin.?;
    const old_table = bound.table;
    defer bound.table = old_table;
    var hooks = old_table.cache.aux.?.*;
    hooks.import_arrays = &Probe.importArrays;
    var cache_hooks = old_table.cache.*;
    cache_hooks.aux = &hooks;
    var table = old_table.*;
    table.cache = &cache_hooks;
    bound.table = &table;

    const cases = [_]struct { msg: [:0]const u8, want: @import("../../host/gpu_thread/requests.zig").CacheImportRefusal }{
        .{ .msg = "[METAL] Command buffer execution failed: Insufficient Memory", .want = .no_memory },
        .{ .msg = "[broadcast_shapes] Shapes cannot be broadcast", .want = .import_arrays },
    };
    for (cases) |c| {
        Probe.latch_msg = c.msg;
        defer mlx.dropLatchedErrorUnless(false);
        const out = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire);
        try testing.expectEqual(@as(@TypeOf(out), .{ .refused = c.want }), out);
        try testing.expectEqual(@as(usize, 0), model_b.entry.prefix_cache.?.entries.items.len);
    }
}

test "dsv41 peer fetch: a commit declined after import_arrays frees the imported snapshot once" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old_fits = scheduler.prefill_admission_fits;
    defer scheduler.prefill_admission_fits = old_fits;
    scheduler.prefill_admission_fits = null;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const model_a = try PeerHarness.open(files);
    defer model_a.close();
    const model_b = try PeerHarness.open(files);
    defer model_b.close();

    const full_prompt = buildPrompt(25);
    const turn1 = try model_a.runGreedy(full_prompt[0..17], 2, true);
    defer testing.allocator.free(turn1.tokens);
    const exp = try peer_transfer.exportWireModule(
        testing.allocator,
        model_a.sch.io,
        &model_a.sch,
        model_a.entry,
        &full_prompt,
        false,
        null,
        0,
        0,
        router.max_cache_export_bytes,
    );
    const wire = exp.ok.bytes;
    defer testing.allocator.free(wire);

    const Probe = struct {
        var real_free: *const fn (*anyopaque, abi.Handle) callconv(.c) void = undefined;
        var real_describe: *const fn (*anyopaque, abi.Handle, *abi.AuxSnapshotInfo) callconv(.c) abi.Verdict = undefined;
        var real_bytes: *const fn (*anyopaque, abi.Handle) callconv(.c) u64 = undefined;
        var free_calls: usize = 0;
        var corrupt_describe: bool = false;
        var inflate_bytes: bool = false;

        fn free(mp: *anyopaque, h: abi.Handle) callconv(.c) void {
            free_calls += 1;
            real_free(mp, h);
        }

        fn describe(mp: *anyopaque, h: abi.Handle, out: *abi.AuxSnapshotInfo) callconv(.c) abi.Verdict {
            const v = real_describe(mp, h, out);
            if (corrupt_describe and v.isOk()) out.n += 1;
            return v;
        }

        fn bytes(mp: *anyopaque, h: abi.Handle) callconv(.c) u64 {
            const b = real_bytes(mp, h);
            return if (inflate_bytes) b << 20 else b;
        }
    };

    const bound = model_b.xfm.plugin.?;
    const old_table = bound.table;
    defer bound.table = old_table;
    var hooks = old_table.cache.aux.?.*;
    Probe.real_free = hooks.free;
    Probe.real_describe = hooks.describe;
    Probe.real_bytes = hooks.bytes;
    hooks.free = &Probe.free;
    hooks.describe = &Probe.describe;
    hooks.bytes = &Probe.bytes;
    var cache_hooks = old_table.cache.*;
    cache_hooks.aux = &hooks;
    var table = old_table.*;
    table.cache = &cache_hooks;
    bound.table = &table;

    // Declined inside commitModule (the snapshot outgrows the budget the wire fit) -> commitModule's defer frees once.
    Probe.free_calls = 0;
    Probe.corrupt_describe = false;
    Probe.inflate_bytes = true;
    model_b.entry.prefix_cache.?.max_kv_bytes = wire.len << 4;
    const declined = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire);
    try testing.expectEqual(@as(@TypeOf(declined), .{ .refused = .commit_declined }), declined);
    try testing.expectEqual(@as(usize, 1), Probe.free_calls);
    try testing.expectEqual(@as(usize, 0), model_b.entry.prefix_cache.?.entries.items.len);

    // Mismatch in step 3(e) before commitModule -> freed directly once.
    Probe.free_calls = 0;
    Probe.corrupt_describe = true;
    Probe.inflate_bytes = false;
    model_b.entry.prefix_cache.?.max_kv_bytes = 0;
    const bad_desc = try peer_transfer.importWireModule(testing.allocator, model_b.sch.io, &model_b.sch, model_b.entry, wire);
    try testing.expectEqual(@as(@TypeOf(bad_desc), .{ .refused = .describe_mismatch }), bad_desc);
    try testing.expectEqual(@as(usize, 1), Probe.free_calls);
    try testing.expectEqual(@as(usize, 0), model_b.entry.prefix_cache.?.entries.items.len);
}

const LiveNode = struct {
    tmp: testing.TmpDir,
    path_buf: [512]u8 = undefined,
    path_len: usize = 0,
    reg: *registry.ModelRegistry,
    entry: *registry.LoadedModel,
    sch: *HostScheduler,

    fn open(io: std.Io, files: []const sdk.fixture.File) !*LiveNode {
        const a = testing.allocator;
        const n = try a.create(LiveNode);
        errdefer a.destroy(n);
        n.tmp = testing.tmpDir(.{ .iterate = true });
        errdefer n.tmp.cleanup();
        for (files) |f| try n.tmp.dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });
        n.path_len = try n.tmp.dir.realPath(io, &n.path_buf);
        const dir = n.path_buf[0..n.path_len];

        n.reg = try registry.ModelRegistry.init(a, io, null, 4, 0, null);
        errdefer n.reg.deinit();
        n.entry = try n.reg.registerStub("dsv41-tiny", dir, 1);
        try n.reg.setDefault("dsv41-tiny");

        var cpu = try cpu_state.preloadCpuState(a, io, dir, 0, null);
        errdefer cpu_state.freeCpuState(a, &cpu);

        n.sch = try mlx_engine.start(a, io, .{
            .registry = n.reg,
            .entry = n.entry,
            .config = cpu.config,
            .tok = cpu.tok,
            .chat_config = cpu.chat_config,
            .model_dir = dir,
            .warmup_eager = false,
            .prefix_cache_capacity = 8,
            .prefix_cache_ram_enabled = true,
            .prefix_cache_mem_bytes = 64 * 1024 * 1024,
        }, 4);
        return n;
    }

    fn close(n: *LiveNode) void {
        n.sch.deinit();
        n.reg.deinit();
        n.tmp.cleanup();
        testing.allocator.destroy(n);
    }
};

const HttpReply = struct {
    status: u16,
    raw: []u8,
    body: []const u8,

    fn deinit(r: *HttpReply, a: std.mem.Allocator) void {
        a.free(r.raw);
    }
};

fn expectStatus(name: []const u8, want: u16, reply: HttpReply) !void {
    if (reply.status != want) {
        std.debug.print("check {s}: got {d} want {d} body={s}\n", .{
            name,
            reply.status,
            want,
            reply.body[0..@min(reply.body.len, 256)],
        });
    }
    try testing.expectEqual(want, reply.status);
}

fn sendHttpRaw(
    io: std.Io,
    node: *LiveNode,
    peer_addr: std.Io.net.IpAddress,
    head_bytes: []const u8,
    body: []const u8,
    shutdown_write_early: bool,
) !HttpReply {
    const a = testing.allocator;
    const prev_reg = http_server.global_registry;
    const prev_sch = http_server.launch().global_scheduler;
    http_server.global_registry = node.reg;
    http_server.launch().global_scheduler = node.sch;
    defer {
        http_server.global_registry = prev_reg;
        http_server.launch().global_scheduler = prev_sch;
    }

    var sv: [2]std.posix.fd_t = undefined;
    const sp_rc = std.c.socketpair(1, 1, 0, &sv);
    if (sp_rc != 0) std.debug.print("check socketpair: got {d} want 0\n", .{sp_rc});
    try testing.expectEqual(@as(c_int, 0), sp_rc);
    defer _ = std.c.close(sv[1]);

    const worker = try std.Thread.spawn(.{}, struct {
        fn run(fd: std.posix.fd_t, addr: std.Io.net.IpAddress, wio: std.Io) void {
            var conn: Conn = undefined;
            Conn.init(&conn, .{ .socket = .{ .handle = fd, .address = addr } }, wio);
            defer conn.close();
            router.handleConnection(testing.allocator, &conn) catch |err| {
                std.debug.print("sendHttp worker error: {s}\n", .{@errorName(err)});
            };
        }
    }.run, .{ sv[0], peer_addr, io });

    try sendAllFd(sv[1], head_bytes);
    if (body.len > 0) try sendAllFd(sv[1], body);
    if (shutdown_write_early) _ = std.c.shutdown(sv[1], 1);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var buf: [16 * 1024]u8 = undefined;
    while (true) {
        const got = std.c.recv(sv[1], &buf, buf.len, 0);
        if (got < 0 and std.posix.errno(got) == .INTR) continue;
        if (got <= 0) break;
        try out.appendSlice(a, buf[0..@intCast(got)]);
    }
    if (!shutdown_write_early) _ = std.c.shutdown(sv[1], 1);
    worker.join();

    const raw = try out.toOwnedSlice(a);
    errdefer a.free(raw);
    const sep = std.mem.indexOf(u8, raw, "\r\n\r\n") orelse {
        std.debug.print("check http_reply_framing: got {d}B raw={s}\n", .{ raw.len, raw[0..@min(raw.len, 256)] });
        return error.MalformedHttpReply;
    };
    const line_end = std.mem.indexOf(u8, raw[0..sep], "\r\n") orelse sep;
    var sp = std.mem.splitScalar(u8, raw[0..line_end], ' ');
    _ = sp.next();
    const code_str = sp.next() orelse return error.MalformedHttpReply;
    const status = try std.fmt.parseInt(u16, code_str, 10);
    return .{ .status = status, .raw = raw, .body = raw[sep + 4 ..] };
}

fn sendHttp(
    io: std.Io,
    node: *LiveNode,
    peer_addr: std.Io.net.IpAddress,
    head: []const u8,
    body: []const u8,
) !HttpReply {
    const req_bytes = try std.fmt.allocPrint(testing.allocator, "{s}Content-Length: {d}\r\n\r\n", .{ head, body.len });
    defer testing.allocator.free(req_bytes);
    return sendHttpRaw(io, node, peer_addr, req_bytes, body, false);
}

fn sendAllFd(fd: std.posix.fd_t, data: []const u8) !void {
    var off: usize = 0;
    while (off < data.len) {
        const sent = std.c.send(fd, data[off..].ptr, data.len - off, 0);
        if (sent < 0 and std.posix.errno(sent) == .INTR) continue;
        if (sent <= 0) return error.SendFailed;
        off += @intCast(sent);
    }
}

test "dsv41 peer fetch: HTTP export and import of a file-backed model answer 409 transfer_unsupported" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var threaded: std.Io.Threaded = .init(testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const prev_key = http_server.g_api_key;
    const prev_strict = http_server.g_api_key_strict;
    defer {
        http_server.g_api_key = prev_key;
        http_server.g_api_key_strict = prev_strict;
    }
    http_server.g_api_key = "peer-secret-60";
    http_server.g_api_key_strict = false;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const node = try LiveNode.open(io, try fixture.v41ServableFiles(arena.allocator(), 7));
    defer node.close();
    try node.tmp.dir.writeFile(io, .{ .sub_path = "model.gguf", .data = "GGUF" });
    const dir_path = node.entry.path;
    node.entry.path = try std.fmt.allocPrint(arena.allocator(), "{s}/model.gguf", .{node.path_buf[0..node.path_len]});
    defer node.entry.path = dir_path;

    const loopback: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 127, 0, 0, 1 }, .port = 50000 } };
    const body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"temperature\":0,\"max_tokens\":4}";
    var exp = try sendHttp(io, node, loopback, "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\nAuthorization: Bearer peer-secret-60\r\nX-Cache-Export: 1\r\n", body);
    defer exp.deinit(testing.allocator);
    try expectStatus("export_transfer_unsupported_409", 409, exp);
    try testing.expect(std.mem.indexOf(u8, exp.body, "\"reason\":\"transfer_unsupported\"") != null);

    var imp = try sendHttp(io, node, loopback, "POST /v1/cache/import HTTP/1.1\r\nContent-Type: application/octet-stream\r\nAuthorization: Bearer peer-secret-60\r\n", "abc");
    defer imp.deinit(testing.allocator);
    try expectStatus("import_transfer_unsupported_409", 409, imp);
    try testing.expect(std.mem.indexOf(u8, imp.body, "\"reason\":\"transfer_unsupported\"") != null);
}

test "dsv41 peer fetch: HTTP export and import enforce peer/auth gates, 422 media/speech, 413 cap, 404 miss and warm B turn" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var threaded: std.Io.Threaded = .init(testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const prev_key = http_server.g_api_key;
    const prev_strict = http_server.g_api_key_strict;
    defer {
        http_server.g_api_key = prev_key;
        http_server.g_api_key_strict = prev_strict;
    }
    http_server.g_api_key = "peer-secret-60";
    http_server.g_api_key_strict = false;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41ServableFiles(arena.allocator(), 7);
    const loopback: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 127, 0, 0, 1 }, .port = 50000 } };
    const tb5_peer: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 10, 254, 13, 3 }, .port = 50001 } };
    const lan_peer: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 192, 168, 86, 42 }, .port = 50002 } };

    const turn1_body = "{\"messages\":[{\"role\":\"user\",\"content\":\"Explain cache locality in systems programming.\"}],\"temperature\":0,\"max_tokens\":4,\"stream\":true}";
    const turn2_body = "{\"messages\":[{\"role\":\"user\",\"content\":\"Explain cache locality in systems programming.\"},{\"role\":\"assistant\",\"content\":\"hi\"},{\"role\":\"user\",\"content\":\"Now add one example.\"}],\"temperature\":0,\"max_tokens\":6,\"stream\":false}";
    const turn2_stream_body = "{\"messages\":[{\"role\":\"user\",\"content\":\"Explain cache locality in systems programming.\"},{\"role\":\"assistant\",\"content\":\"hi\"},{\"role\":\"user\",\"content\":\"Now add one example.\"}],\"temperature\":0,\"max_tokens\":6,\"stream\":true}";

    var exp_reply: HttpReply = undefined;
    var exp_ready = false;
    defer if (exp_ready) exp_reply.deinit(testing.allocator);

    var a_turn2: HttpReply = undefined;
    var a_turn2_ready = false;
    defer if (a_turn2_ready) a_turn2.deinit(testing.allocator);

    {
        const node_a = try LiveNode.open(io, files);
        defer node_a.close();

        for ([_][]const u8{ "abc", "0", "-5" }) |deadline| {
            const export_head = try std.fmt.allocPrint(
                testing.allocator,
                "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\nAuthorization: Bearer peer-secret-60\r\nX-Cache-Export: 1\r\nX-Cache-Deadline-Ms: {s}\r\n",
                .{deadline},
            );
            defer testing.allocator.free(export_head);
            var export_reply = try sendHttp(io, node_a, loopback, export_head, turn1_body);
            defer export_reply.deinit(testing.allocator);
            try expectStatus("export_invalid_deadline_400", 400, export_reply);
        }

        {
            var r1 = try sendHttp(io, node_a, lan_peer, "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\nAuthorization: Bearer peer-secret-60\r\nX-Cache-Export: 1\r\n", turn1_body);
            defer r1.deinit(testing.allocator);
            try expectStatus("export_lan_403", 403, r1);
        }
        {
            var r1 = try sendHttp(io, node_a, loopback, "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\nX-Cache-Export: 1\r\n", turn1_body);
            defer r1.deinit(testing.allocator);
            try expectStatus("export_no_auth_401", 401, r1);
        }
        {
            const media_body = "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"not-a-valid-image\"}},{\"type\":\"text\",\"text\":\"hi\"}]}]}";
            var r1 = try sendHttp(io, node_a, loopback, "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\nAuthorization: Bearer peer-secret-60\r\nX-Cache-Export: 1\r\n", media_body);
            defer r1.deinit(testing.allocator);
            try expectStatus("export_media_422", 422, r1);

            const speech_body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"modalities\":[\"text\",\"audio\"]}";
            var r2 = try sendHttp(io, node_a, loopback, "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\nAuthorization: Bearer peer-secret-60\r\nX-Cache-Export: 1\r\n", speech_body);
            defer r2.deinit(testing.allocator);
            try expectStatus("export_speech_422", 422, r2);
        }
        {
            var miss = try sendHttp(io, node_a, loopback, "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\nAuthorization: Bearer peer-secret-60\r\nX-Cache-Export: 1\r\n", turn2_stream_body);
            defer miss.deinit(testing.allocator);
            try expectStatus("export_cold_miss_404", 404, miss);
        }
        {
            var t1 = try sendHttp(io, node_a, loopback, "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\n", turn1_body);
            defer t1.deinit(testing.allocator);
            try expectStatus("seed_turn1_200", 200, t1);
            const saw_done = std.mem.indexOf(u8, t1.body, "data: [DONE]") != null;
            if (!saw_done) {
                std.debug.print("check seed_turn1_done: got body({d}B)={s} want data: [DONE]\n", .{
                    t1.body.len,
                    t1.body[0..@min(t1.body.len, 256)],
                });
            }
            try testing.expect(saw_done);
        }

        exp_reply = try sendHttp(io, node_a, loopback, "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\nAuthorization: Bearer peer-secret-60\r\nX-Cache-Export: 1\r\nX-Cache-Deadline-Ms: 60000\r\n", turn2_stream_body);
        exp_ready = true;
        try expectStatus("export_warm_200", 200, exp_reply);
        if (exp_reply.body.len <= peer_wire.FIXED_HEADER_BYTES) {
            std.debug.print("check export_wire_len: got {d} want > {d}\n", .{ exp_reply.body.len, peer_wire.FIXED_HEADER_BYTES });
        }
        try testing.expect(exp_reply.body.len > peer_wire.FIXED_HEADER_BYTES);

        {
            cache_transfer.max_export_bytes_override = exp_reply.body.len - 1;
            defer cache_transfer.max_export_bytes_override = null;
            var over_cap = try sendHttp(io, node_a, loopback, "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\nAuthorization: Bearer peer-secret-60\r\nX-Cache-Export: 1\r\n", turn2_stream_body);
            defer over_cap.deinit(testing.allocator);
            try expectStatus("export_over_cap_413", 413, over_cap);
        }

        a_turn2 = try sendHttp(io, node_a, loopback, "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\n", turn2_body);
        a_turn2_ready = true;
        try expectStatus("node_a_turn2_200", 200, a_turn2);
    }

    var decoded_exp = try peer_wire.decode(testing.allocator, exp_reply.body);
    defer decoded_exp.deinit();
    const exported_n = decoded_exp.ids.len;
    if (exported_n == 0) {
        std.debug.print("check decoded_exported_n: got {d} want > 0\n", .{exported_n});
    }
    try testing.expect(exported_n > 0);

    const a_parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, a_turn2.body, .{});
    defer a_parsed.deinit();

    var b_turn2: HttpReply = undefined;
    var b_turn2_ready = false;
    defer if (b_turn2_ready) b_turn2.deinit(testing.allocator);

    {
        const node_b = try LiveNode.open(io, files);
        defer node_b.close();

        for ([_][]const u8{ "abc", "0", "-5" }) |deadline| {
            const import_head = try std.fmt.allocPrint(
                testing.allocator,
                "POST /v1/cache/import HTTP/1.1\r\nContent-Type: application/octet-stream\r\nAuthorization: Bearer peer-secret-60\r\nX-Cache-Deadline-Ms: {s}\r\n",
                .{deadline},
            );
            defer testing.allocator.free(import_head);
            var import_reply = try sendHttp(io, node_b, tb5_peer, import_head, "abc");
            defer import_reply.deinit(testing.allocator);
            try expectStatus("import_invalid_deadline_400", 400, import_reply);
        }

        {
            var r2 = try sendHttp(io, node_b, lan_peer, "POST /v1/cache/import HTTP/1.1\r\nContent-Type: application/octet-stream\r\nAuthorization: Bearer peer-secret-60\r\n", "abc");
            defer r2.deinit(testing.allocator);
            try expectStatus("import_lan_403", 403, r2);
        }
        {
            var r2 = try sendHttp(io, node_b, loopback, "POST /v1/cache/import HTTP/1.1\r\nContent-Type: application/octet-stream\r\n", "abc");
            defer r2.deinit(testing.allocator);
            try expectStatus("import_no_auth_401", 401, r2);
        }
        {
            const oversize_head = try std.fmt.allocPrint(
                testing.allocator,
                "POST /v1/cache/import HTTP/1.1\r\nContent-Type: application/octet-stream\r\nAuthorization: Bearer peer-secret-60\r\nContent-Length: {d}\r\n\r\n",
                .{router.max_request_bytes},
            );
            defer testing.allocator.free(oversize_head);
            var r2 = try sendHttpRaw(io, node_b, tb5_peer, oversize_head, "", true);
            defer r2.deinit(testing.allocator);
            try expectStatus("import_over_cap_413", 413, r2);
        }

        var imp_reply = try sendHttp(io, node_b, tb5_peer, "POST /v1/cache/import HTTP/1.1\r\nContent-Type: application/octet-stream\r\nAuthorization: Bearer peer-secret-60\r\n", exp_reply.body);
        defer imp_reply.deinit(testing.allocator);
        try expectStatus("import_tb5_200", 200, imp_reply);
        const imp_json = try std.json.parseFromSlice(std.json.Value, testing.allocator, imp_reply.body, .{});
        defer imp_json.deinit();
        const got_imp_n = imp_json.value.object.get("n").?.integer;
        if (got_imp_n != @as(i64, @intCast(exported_n))) {
            std.debug.print("check import_n: got {d} want {d} body={s}\n", .{ got_imp_n, exported_n, imp_reply.body });
        }
        try testing.expectEqual(@as(i64, @intCast(exported_n)), got_imp_n);

        b_turn2 = try sendHttp(io, node_b, loopback, "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\n", turn2_body);
        b_turn2_ready = true;
        try expectStatus("node_b_turn2_200", 200, b_turn2);
    }

    const b_parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, b_turn2.body, .{});
    defer b_parsed.deinit();
    const a_cached = a_parsed.value.object.get("usage").?.object.get("prompt_tokens_details").?.object.get("cached_tokens").?.integer;
    const b_cached = b_parsed.value.object.get("usage").?.object.get("prompt_tokens_details").?.object.get("cached_tokens").?.integer;
    if (a_cached != @as(i64, @intCast(exported_n))) {
        std.debug.print("check node_a_cached: got {d} want {d} body={s}\n", .{ a_cached, exported_n, a_turn2.body });
    }
    try testing.expectEqual(@as(i64, @intCast(exported_n)), a_cached);
    if (b_cached != a_cached) {
        std.debug.print("check node_b_cached: got {d} want {d} body={s}\n", .{ b_cached, a_cached, b_turn2.body });
    }
    try testing.expectEqual(a_cached, b_cached);

    const a_msg = a_parsed.value.object.get("choices").?.array.items[0].object.get("message").?.object;
    const b_msg = b_parsed.value.object.get("choices").?.array.items[0].object.get("message").?.object;
    const a_content = if (a_msg.get("content")) |v| (if (v == .string) v.string else "") else "";
    const b_content = if (b_msg.get("content")) |v| (if (v == .string) v.string else "") else "";
    const a_reason = if (a_msg.get("reasoning_content")) |v| (if (v == .string) v.string else "") else "";
    const b_reason = if (b_msg.get("reasoning_content")) |v| (if (v == .string) v.string else "") else "";
    if (!std.mem.eql(u8, a_content, b_content)) {
        std.debug.print("check warm_b_content: got '{s}' want '{s}'\n", .{ b_content, a_content });
    }
    try testing.expectEqualStrings(a_content, b_content);
    if (!std.mem.eql(u8, a_reason, b_reason)) {
        std.debug.print("check warm_b_reasoning: got '{s}' want '{s}'\n", .{ b_reason, a_reason });
    }
    try testing.expectEqualStrings(a_reason, b_reason);
}
