//! Inference-thread RAM prefix-cache export (`exportWireModule`), peer import (`importWireModule`), and side-queue RPC.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const plugin_abi = @import("../../../abi/plugin.zig");
const plugin_driver = @import("../plugin/driver.zig");
const kv_quant = @import("../../../sdk/blocks/attention/kv_quant.zig");
const decode_attn_quant = @import("../../../sdk/blocks/attention/decode_attn_quant.zig");
const model_mod = @import("../../../sdk/config/model.zig");
const io_util = @import("../../../sdk/util/io_util.zig");
const log = @import("../../../sdk/util/log.zig");
const transformer_mod = @import("../transformer.zig");
const prefix_cache_mod = @import("prefix_cache.zig");
const peer_wire = @import("peer_wire.zig");
const prefill_mod = @import("../prefill.zig");
const pack_identity_mod = @import("pack_identity.zig");
const dsv41_numerics = @import("../../../plugins/deepseek_v41/numerics.zig");
const env_vars = @import("../../../env.zig");
const forceGpuFamilyFallback = @import("../../../mlx/device.zig").forceGpuFamilyFallback;
const publishHotCacheResidency = @import("budget.zig").publishHotCacheResidency;
const stagedHostBytes = @import("budget.zig").stagedHostBytes;
const prefillHeadroomNow = @import("../admission/prefill_chunk.zig").prefillHeadroomNow;
const module_record = @import("ssd/module_record.zig");
const nowMsMonotonic = io_util.nowMsMonotonic;
const requests = @import("../../../host/gpu_thread/requests.zig");
const LoadedModel = @import("../../../host/registry/registry.zig").LoadedModel;
const Scheduler = @import("../scheduler.zig").Scheduler;

const Transformer = transformer_mod.Transformer;
const ModelConfig = model_mod.ModelConfig;

pub fn modelLayoutKey(cfg: *const ModelConfig, xfm: *const Transformer) u64 {
    var h = std.hash.Wyhash.init(0x6c61_796f_7574_3630);
    if (xfm.plugin) |b| {
        h.update(b.table.name.slice());
        if (b.table.cache.layout) |layout_fn| if (cfg.plugin) |pb| {
            var layers: [128]plugin_abi.LayerCache = undefined;
            const n = layout_fn(&pb.parsed, null, &layers, layers.len);
            for (layers[0..n]) |lc| {
                std.hash.autoHash(&h, .{ @backingInt(lc.kind), lc.nope, lc.kv_source, lc.window, lc.kv_heads, lc.k_dim, lc.v_dim });
            }
        };
    } else {
        h.update(cfg.model_type);
    }
    if (cfg.cacheLayoutNamespace()) |ns| {
        h.update("\x00");
        h.update(ns);
    }
    std.hash.autoHash(&h, .{
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.moe_intermediate_size,
        cfg.num_experts,
        cfg.vocab_size,
        cfg.sliding_window,
    });
    if (cfg.dist_shard) |sh| {
        std.hash.autoHash(&h, .{ @as(u8, 1), sh.rank, sh.size });
    } else {
        std.hash.autoHash(&h, @as(u8, 0));
    }
    return h.final();
}

fn optBoolTag(v: ?bool) u8 {
    return if (v) |b| (if (b) 2 else 1) else 0;
}

pub fn modelSettingsKey(lm: *const LoadedModel, cfg: *const ModelConfig, xfm: *const Transformer) u64 {
    var h = std.hash.Wyhash.init(0x7365_7474_696e_6773);
    const head_mxfp8 = cfg.lm_head_mxfp8_override orelse transformer_mod.cli().lm_head_mxfp8_cli;
    const mtp_loaded = lm.mtp != null or xfm.dsparkStages() > 0;
    const dq_active = decode_attn_quant.decodeAttnQuantEnabled() and xfm.dense_attn_proj;
    std.hash.autoHash(&h, .{
        optBoolTag(head_mxfp8),
        mtp_loaded,
        optBoolTag(cfg.mtp_override),
        optBoolTag(cfg.int8_prefill_override),
        dq_active,
        decode_attn_quant.decodeAttnQuantExplicit(),
        Transformer.mtpHeadKvQuantEnabled(),
    });
    if (cfg.stream_experts_override) |se| {
        switch (se) {
            .on => std.hash.autoHash(&h, .{ @as(u8, 1), @as(u64, 0) }),
            .rows => |r| std.hash.autoHash(&h, .{ @as(u8, 2), @as(u64, r) }),
        }
    } else {
        std.hash.autoHash(&h, @as(u8, 0));
    }
    std.hash.autoHash(&h, dsv41_numerics.resolved());
    std.hash.autoHash(&h, .{ env_vars.int(.LAYER_CAP, usize, 0), env_vars.on(.DSV4_SKIP_MOE), forceGpuFamilyFallback() });
    if (model_mod.getConfigOverrides()) |raw| {
        h.update("\x00");
        h.update(raw);
    }
    return h.final();
}

fn rawArrayBytes(a: mlx.mlx_array) ?[*]const u8 {
    return switch (mlx.mlx_array_dtype(a)) {
        .bool_ => @ptrCast(mlx.mlx_array_data_bool(a) orelse return null),
        .uint8, .int8 => @ptrCast(mlx.mlx_array_data_uint8(a) orelse return null),
        .uint32, .int32 => @ptrCast(mlx.mlx_array_data_uint32(a) orelse return null),
        .float32 => @ptrCast(mlx.mlx_array_data_float32(a) orelse return null),
        .float16 => @ptrCast(mlx.mlx_array_data_float16(a) orelse return null),
        .bfloat16 => @ptrCast(mlx.mlx_array_data_bfloat16(a) orelse return null),
        else => null,
    };
}

fn materializeExportTensors(tensors: []plugin_abi.AuxTensor, s: mlx.mlx_stream) !void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (tensors) |*t| {
        const src: mlx.mlx_array = .{ .ctx = t.array.ctx };
        var cont = mlx.mlx_array_new();
        {
            errdefer _ = mlx.mlx_array_free(cont);
            try mlx.check(mlx.mlx_contiguous(&cont, src, false, s));
        }
        _ = mlx.mlx_array_free(src);
        t.array.ctx = cont.ctx;
        try mlx.check(mlx.mlx_vector_array_append_value(vec, cont));
    }
    try mlx.check(mlx.mlx_eval(vec));
    if (mlx.errorPending()) return error.MlxError;
}

pub fn exportWireModule(
    allocator: std.mem.Allocator,
    io: std.Io,
    sch: *const Scheduler,
    lm: *LoadedModel,
    prompt_ids: []const u32,
    has_tools: bool,
    kv_quant_override: ?kv_quant.KVQuantConfig,
    variant: u64,
    cache_key: u64,
    max_bytes: usize,
) !requests.CacheExportOutcome {
    const t0 = io_util.nowUsMonotonic(io);
    const had_error = mlx.errorPending();
    defer mlx.dropLatchedErrorUnless(had_error);

    const xfm = lm.transformer orelse return .not_found;
    const cfg = lm.config orelse return .not_found;
    const hc = lm.prefix_cache orelse return .not_found;
    const binding = prefix_cache_mod.ModuleBinding.forVariant(xfm, variant) orelse return .not_found;
    const pack_identity = switch (pack_identity_mod.status(lm.pack_identity)) {
        .ready => |id| id,
        .pending => return .identity_pending,
        .unsupported => return .transfer_unsupported,
    };
    const quant_config = kv_quant_override orelse cfg.kv_quant_override orelse sch.kv_quant_config;

    const hit = hc.findRamModule(binding, prompt_ids, has_tools, quant_config) orelse return .not_found;

    var info: plugin_abi.AuxSnapshotInfo = .{};
    if (!plugin_driver.auxDescribe(binding.xfm, hit.payload.snapshot, &info).isOk() or
        info.n != hit.payload.n or
        info.n != hit.tokens.len or
        prefix_cache_mod.variantKey(info.prefix_key, binding.variant) != hit.payload.prefix_key or
        !std.mem.eql(u32, info.ids.slice(), hit.tokens) or
        mlx.errorPending())
    {
        return .not_found;
    }

    const exp = try prefix_cache_mod.ModuleExport.create(allocator, hit.payload);
    defer exp.destroy(allocator);

    var inv_arena = std.heap.ArenaAllocator.init(allocator);
    defer inv_arena.deinit();
    const inv = try exp.inventory(inv_arena.allocator());

    const total_len = try peer_wire.encodedSize(hit.tokens.len, exp.out.header_len, inv);
    if (total_len > max_bytes) {
        return .{ .too_large = @intCast(total_len) };
    }

    const layout_key = modelLayoutKey(cfg, xfm);
    const settings_key = modelSettingsKey(lm, cfg, xfm);

    try materializeExportTensors(exp.tensors[0..exp.out.tensor_len], xfm.s);

    const out = try allocator.alloc(u8, total_len);
    errdefer allocator.free(out);

    const eff_cache_key = if (cache_key != 0) cache_key else hit.cache_key;
    const payload_off = try peer_wire.writeEnvelopePrefix(out, .{
        .layout_key = layout_key,
        .pack_identity = pack_identity,
        .settings = settings_key,
        .prefix_key = info.prefix_key,
        .cache_key = eff_cache_key,
        .variant = binding.variant,
        .has_tools = has_tools,
        .quant_config = hit.quant_config,
        .ids = hit.tokens,
        .plugin_header = exp.header[0..exp.out.header_len],
        .tensors = inv,
    });

    var cursor = payload_off;
    for (exp.tensors[0..exp.out.tensor_len], inv) |t, item| {
        const nb: usize = @intCast(item.bytes);
        const src = rawArrayBytes(.{ .ctx = t.array.ctx }) orelse return error.UnreadableExportTensor;
        @memcpy(out[cursor..][0..nb], src[0..nb]);
        cursor += nb;
    }
    peer_wire.stampChecksum(out);

    const elapsed_ms: u64 = @intCast(@max(@divTrunc(io_util.nowUsMonotonic(io) - t0, 1000), 0));
    log.info("[hot-cache] peer module export: n={d} bytes={d} ({d} ms)\n", .{ hit.tokens.len, out.len, elapsed_ms });
    return .{ .ok = .{ .bytes = out, .n = hit.tokens.len, .export_ms = elapsed_ms } };
}

fn mapDecodeError(err: peer_wire.DecodeError) error{OutOfMemory}!requests.CacheImportRefusal {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.TruncatedEnvelope => .truncated_body,
        error.BadEnvelope, error.BadTensorName => .bad_envelope,
        error.UnsupportedEnvelopeVersion => .unknown_version,
        error.BadQuantConfig => .bad_quant_config,
        error.BadTensorDtype => .bad_dtype,
        error.BadTensorShape => .bad_shape,
        error.BadTensorOffsets => .bad_offsets,
        error.BadChecksum => .bad_checksum,
    };
}

fn logRefusal(refusal: requests.CacheImportRefusal, tokens: usize) void {
    log.info("[hot-cache] peer module import refused ({s}, n={d})\n", .{ @tagName(refusal), tokens });
}

/// The refusals `commitModule` would give, asked on the envelope's own byte counts before anything is evicted.
fn budgetRefusal(hc: *const prefix_cache_mod.HotPrefixCache, tensors: []const module_record.Tensor) !?requests.CacheImportRefusal {
    if (!hc.ram_enabled or hc.max_entries == 0) return .commit_declined;
    if (hc.max_kv_bytes == 0) return null;
    var bytes: u64 = 0;
    for (tensors) |t| bytes = try std.math.add(u64, bytes, t.bytes);
    if (bytes > hc.max_kv_bytes) return .commit_declined;
    return null;
}

/// A failed import on an out-of-memory latch is the receiver's memory, not a bad pair of boxes.
fn importArraysRefusal() requests.CacheImportRefusal {
    return if (mlx.latchedErrorIs(mlx.mlxErrorIsMemory)) .no_memory else .import_arrays;
}

pub fn importWireModule(
    allocator: std.mem.Allocator,
    io: std.Io,
    sch: *Scheduler,
    lm: *LoadedModel,
    body: []const u8,
) !requests.CacheImportOutcome {
    const t0 = io_util.nowUsMonotonic(io);
    const had_error = mlx.errorPending();
    defer mlx.dropLatchedErrorUnless(had_error);

    const local_pack_id = switch (pack_identity_mod.status(lm.pack_identity)) {
        .ready => |id| id,
        .pending => {
            logRefusal(.identity_pending, 0);
            return .{ .refused = .identity_pending };
        },
        .unsupported => {
            logRefusal(.transfer_unsupported, 0);
            return .{ .refused = .transfer_unsupported };
        },
    };

    var env = peer_wire.decode(allocator, body) catch |err| {
        const r = try mapDecodeError(err);
        logRefusal(r, 0);
        return .{ .refused = r };
    };
    defer env.deinit();
    const t_decode = io_util.nowUsMonotonic(io);

    const xfm = lm.transformer orelse {
        logRefusal(.unsupported_model, env.ids.len);
        return .{ .refused = .unsupported_model };
    };
    const cfg = lm.config orelse {
        logRefusal(.unsupported_model, env.ids.len);
        return .{ .refused = .unsupported_model };
    };
    const hc = lm.prefix_cache orelse {
        logRefusal(.unsupported_model, env.ids.len);
        return .{ .refused = .unsupported_model };
    };
    const binding = prefix_cache_mod.ModuleBinding.forVariant(xfm, env.variant) orelse {
        logRefusal(.unsupported_model, env.ids.len);
        return .{ .refused = .unsupported_model };
    };

    if (env.layout_key != modelLayoutKey(cfg, xfm)) {
        logRefusal(.layout_key, env.ids.len);
        return .{ .refused = .layout_key };
    }
    if (env.pack_identity != local_pack_id) {
        logRefusal(.pack_identity, env.ids.len);
        return .{ .refused = .pack_identity };
    }
    if (env.settings != modelSettingsKey(lm, cfg, xfm)) {
        logRefusal(.settings, env.ids.len);
        return .{ .refused = .settings };
    }
    const local_quant = cfg.kv_quant_override orelse sch.kv_quant_config;
    if (!std.meta.eql(env.quant_config, local_quant)) {
        logRefusal(.quant_config, env.ids.len);
        return .{ .refused = .quant_config };
    }

    if (hc.hasModuleCovering(binding, env.ids, env.has_tools, env.quant_config)) {
        logRefusal(.stale, env.ids.len);
        return .{ .refused = .stale };
    }

    if (try budgetRefusal(hc, env.tensors)) |r| {
        logRefusal(r, env.ids.len);
        return .{ .refused = r };
    }

    var planned_peak: u64 = 0;
    if (!plugin_driver.auxImportPlan(binding.xfm, env.plugin_header, &planned_peak).isOk()) {
        logRefusal(.import_plan, env.ids.len);
        return .{ .refused = .import_plan };
    }
    var peak = try prefill_mod.moduleImportPeak(planned_peak, env.tensors);
    peak = try std.math.add(u64, peak, @intCast(body.len));
    if (peak > prefillHeadroomNow(cfg, stagedHostBytes(sch)) +| hc.residentBytes()) {
        logRefusal(.no_memory, env.ids.len);
        return .{ .refused = .no_memory };
    }
    var probe: prefill_mod.ModuleImportProbe = .{ .sch = sch, .cfg = cfg, .peak = peak };
    const admitted = hc.evictLruToAdmit(@intCast(env.ids.len), &probe, prefill_mod.ModuleImportProbe.fits, false).admitted;
    publishHotCacheResidency(sch);
    if (!admitted) {
        logRefusal(.no_memory, env.ids.len);
        return .{ .refused = .no_memory };
    }
    const t_admit = io_util.nowUsMonotonic(io);

    const wrappers = try allocator.alloc(mlx.mlx_array, env.tensors.len);
    for (wrappers) |*w| w.* = .{ .ctx = null };
    defer {
        for (wrappers) |w| {
            if (w.ctx != null) _ = mlx.mlx_array_free(w);
        }
        allocator.free(wrappers);
    }
    const aux_tensors = try allocator.alloc(plugin_abi.AuxTensor, env.tensors.len);
    defer allocator.free(aux_tensors);

    for (env.tensors, env.descriptors, wrappers, aux_tensors) |t, desc, *w, *aux| {
        var c_shape: [8]c_int = undefined;
        for (t.shape, 0..) |dim, k| c_shape[k] = @intCast(dim);
        const slice = env.payload[@intCast(desc.offset)..@intCast(desc.end)];
        w.* = mlx.mlx_array_new_data(slice.ptr, &c_shape, @intCast(t.shape.len), desc.mlx_dtype);
        if (w.ctx == null) {
            const r = importArraysRefusal();
            logRefusal(r, env.ids.len);
            return .{ .refused = r };
        }
        aux.* = .{ .name = .of(t.name), .array = .{ .ctx = w.ctx } };
    }

    var snapshot: plugin_abi.Handle = null;
    if (!plugin_driver.auxImportArrays(binding.xfm, env.ids, env.plugin_header, aux_tensors, &snapshot).isOk() or
        snapshot == null or
        mlx.errorPending())
    {
        if (snapshot != null) plugin_driver.auxFree(binding.xfm, snapshot);
        const r = importArraysRefusal();
        logRefusal(r, env.ids.len);
        return .{ .refused = r };
    }
    const t_import = io_util.nowUsMonotonic(io);

    var info: plugin_abi.AuxSnapshotInfo = .{};
    if (!plugin_driver.auxDescribe(binding.xfm, snapshot, &info).isOk() or
        info.n != env.ids.len or
        info.prefix_key != env.prefix_key or
        prefix_cache_mod.variantKey(info.prefix_key, binding.variant) != prefix_cache_mod.variantKey(env.prefix_key, binding.variant) or
        !std.mem.eql(u32, info.ids.slice(), env.ids) or
        mlx.errorPending())
    {
        plugin_driver.auxFree(binding.xfm, snapshot);
        logRefusal(.describe_mismatch, env.ids.len);
        return .{ .refused = .describe_mismatch };
    }

    const committed = hc.commitModule(
        binding,
        snapshot,
        env.ids,
        env.prefix_key,
        &.{},
        env.has_tools,
        env.quant_config,
        env.cache_key,
    ) catch |err| {
        publishHotCacheResidency(sch);
        return err;
    };
    publishHotCacheResidency(sch);
    const t_commit = io_util.nowUsMonotonic(io);

    switch (committed) {
        .ok => |n| {
            const stats: requests.CacheImportStats = .{
                .n = n,
                .bytes = body.len,
                .decode_ms = @intCast(@max(@divTrunc(t_decode - t0, 1000), 0)),
                .admit_ms = @intCast(@max(@divTrunc(t_admit - t_decode, 1000), 0)),
                .import_ms = @intCast(@max(@divTrunc(t_import - t_admit, 1000), 0)),
                .commit_ms = @intCast(@max(@divTrunc(t_commit - t_import, 1000), 0)),
            };
            log.info("[hot-cache] peer module import: n={d} bytes={d} (decode={d}ms admit={d}ms import={d}ms commit={d}ms)\n", .{
                stats.n, stats.bytes, stats.decode_ms, stats.admit_ms, stats.import_ms, stats.commit_ms,
            });
            return .{ .ok = stats };
        },
        else => {
            logRefusal(.commit_declined, env.ids.len);
            return .{ .refused = .commit_declined };
        },
    }
}

pub fn runCacheExportRequest(sch: *Scheduler, req: *requests.CacheExportRequest) void {
    if (deadlinePassed(sch.io, req.deadline_ms)) {
        req.done_mu.lockUncancelable(sch.io);
        defer req.done_mu.unlock(sch.io);
        req.outcome = .deadline_passed;
        req.done = true;
        req.done_cond.broadcast(sch.io);
        return;
    }
    const out = exportWireModule(
        req.allocator,
        sch.io,
        sch,
        req.model,
        req.prompt_ids,
        req.has_tools,
        req.kv_quant_override,
        req.variant,
        req.cache_key,
        req.max_bytes,
    ) catch |err| {
        log.warn("[hot-cache] peer cache export failed: {s}\n", .{@errorName(err)});
        req.done_mu.lockUncancelable(sch.io);
        defer req.done_mu.unlock(sch.io);
        req.error_name = @errorName(err);
        req.done = true;
        req.done_cond.broadcast(sch.io);
        return;
    };
    req.done_mu.lockUncancelable(sch.io);
    defer req.done_mu.unlock(sch.io);
    req.outcome = out;
    req.done = true;
    req.done_cond.broadcast(sch.io);
}

pub fn runCacheImportRequest(sch: *Scheduler, req: *requests.CacheImportRequest) void {
    if (deadlinePassed(sch.io, req.deadline_ms)) {
        req.done_mu.lockUncancelable(sch.io);
        defer req.done_mu.unlock(sch.io);
        req.outcome = .{ .refused = .deadline_passed };
        req.done = true;
        req.done_cond.broadcast(sch.io);
        return;
    }
    const out = importWireModule(req.allocator, sch.io, sch, req.model, req.body) catch |err| {
        log.warn("[hot-cache] peer cache import failed: {s}\n", .{@errorName(err)});
        req.done_mu.lockUncancelable(sch.io);
        defer req.done_mu.unlock(sch.io);
        req.error_name = @errorName(err);
        req.done = true;
        req.done_cond.broadcast(sch.io);
        return;
    };
    req.done_mu.lockUncancelable(sch.io);
    defer req.done_mu.unlock(sch.io);
    req.outcome = out;
    req.done = true;
    req.done_cond.broadcast(sch.io);
}

fn deadlinePassed(io: std.Io, deadline_ms: ?u64) bool {
    const deadline = deadline_ms orelse return false;
    const now: u64 = @intCast(@max(nowMsMonotonic(io), 0));
    return now > deadline;
}

pub fn exportCache(sch: *Scheduler, req: *requests.CacheExportRequest) !void {
    sch.queue_mu.lockUncancelable(sch.io);
    sch.cache_export_queue.append(sch.allocator, req) catch |err| {
        sch.queue_mu.unlock(sch.io);
        return err;
    };
    sch.queue_cond.broadcast(sch.io);
    sch.queue_mu.unlock(sch.io);

    req.done_mu.lockUncancelable(sch.io);
    defer req.done_mu.unlock(sch.io);
    while (!req.done) {
        req.done_cond.waitUncancelable(sch.io, &req.done_mu);
    }
    if (req.error_name != null) {
        req.error_name = null;
        return error.CacheExportFailed;
    }
}

pub fn importCache(sch: *Scheduler, req: *requests.CacheImportRequest) !void {
    sch.queue_mu.lockUncancelable(sch.io);
    sch.cache_import_queue.append(sch.allocator, req) catch |err| {
        sch.queue_mu.unlock(sch.io);
        return err;
    };
    sch.queue_cond.broadcast(sch.io);
    sch.queue_mu.unlock(sch.io);

    req.done_mu.lockUncancelable(sch.io);
    defer req.done_mu.unlock(sch.io);
    while (!req.done) {
        req.done_cond.waitUncancelable(sch.io, &req.done_mu);
    }
    if (req.error_name != null) {
        req.error_name = null;
        return error.CacheImportFailed;
    }
}
