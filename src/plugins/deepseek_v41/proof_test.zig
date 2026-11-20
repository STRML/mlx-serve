//! Parity proof for the V4.1 plugin: its forward on a tiny random pack equals the deleted legacy seam at every step,
//! its verify block the module's own all-rows extend, and a resumed prompt the legacy resume, as recorded in the tapes
//! in `fixtures/` (`../deepseek_v4/proof_test.zig`).

const std = @import("std");
const log = @import("../../sdk/util/log.zig");
const env = @import("../../env.zig");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const proof = @import("../deepseek_v4/proof_test.zig");
const fixture = @import("../deepseek_v4/fixture.zig");
const v41 = @import("plugin.zig");
const sdk = @import("../../sdk/root.zig");

const testing = std.testing;
const Twin = proof.Twin(v41);

test "dsv41 plugin: 8192 prefill rows run and 8193 are refused before slot activation" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), "");
    defer t.close();
    const ids: [8193]u32 = @splat(3);
    testing.allocator.free(try slotForward(t, .prefill, &.{1}, ids[0..8192], &.{0}));
    const m: *v41.Model = @ptrCast(@alignCast(t.m));
    try testing.expectEqual(@as(usize, 8192), m.mdl.dec_state.?.n);
    const Cases = [_]struct { mode: abi.Mode, logits: abi.Logits }{
        .{ .mode = .prefill, .logits = .last_row },
        .{ .mode = .prefill, .logits = .all_rows },
        .{ .mode = .verify, .logits = .all_rows },
    };
    for (Cases) |c| {
        var out: abi.ForwardOut = .{};
        const verdict = v41.table.forward(t.m, &.{
            .mode = c.mode,
            .rows = ids.len,
            .tokens = .of(&ids),
            .slots = .of(&.{2}),
            .positions = .of(&.{1}),
            .logits = c.logits,
        }, &out);
        try testing.expectEqual(abi.Verdict.Code.too_large, verdict.code);
        try testing.expect(std.mem.indexOf(u8, verdict.why.slice(), "8192") != null);
        try testing.expectEqual(@as(u32, 1), m.owner);
        try testing.expectEqual(@as(usize, 8192), m.mdl.dec_state.?.n);
    }
}

test "deepseek_v41 plugin: prefill, GPU decode, a verify block and a later chunk equal the legacy seam bit for bit" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), @embedFile("fixtures/steps.tape"));
    defer t.close();
    try proof.expectSteps(v41, t);
}

test "deepseek_v41 plugin: a paired instance keeps the legacy resume and continues bit-equal" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const first = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 60 };
    const second = first ++ [_]u32{ 11, 2, 90 };
    const paired: dsv41.Pair = .{ .lo = 2, .hi = 4, .worker = false };
    for ([_]bool{ false, true }) |release_parked| {
        const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), @embedFile("fixtures/proof-resume.tape"));
        defer t.close();
        const model: *v41.Model = @ptrCast(@alignCast(t.m));
        for ([_][]const u32{ &first, &second }, 0..) |prompt, turn| {
            model.mdl.pair = paired;
            const n: u32 = @intCast(v41.resumePrompt(t.m, 7, .of(prompt), .{}));
            model.mdl.pair = null;
            if (turn == 1) try testing.expect(n > 0);
            try t.check(mlx.mlx_array_new_data(&n, &[_]c_int{1}, 1, .uint32));
            try t.check(try t.plugin(.{ .mode = .prefill, .ids = prompt[n..] }, n));
            if (turn == 0 and release_parked)
                testing.allocator.free(try slotForward(t, .prefill, &.{8}, &.{ 4, 12, 6 }, &.{0}));
            model.mdl.pair = paired;
            v41.table.release_slot.?(t.m, 7);
            model.mdl.pair = null;
            if (turn == 0 and !release_parked)
                testing.allocator.free(try slotForward(t, .prefill, &.{8}, &.{ 4, 12, 6 }, &.{0}));
        }
        try t.tape.expectDone();
        testing.allocator.free(try slotForward(t, .decode, &.{8}, &.{13}, &.{3}));
    }
}

test "deepseek_v41 plugin: a decode token on the device equals the legacy decode" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), @embedFile("fixtures/device-decode.tape"));
    defer t.close();
    try proof.expectDeviceDecode(v41, t);
}

test "deepseek_v41 plugin: trunk_weight hands out the loaded embedding and head" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), "");
    defer t.close();
    const m: *v41.Model = @ptrCast(@alignCast(t.m));
    var w: @import("../../abi/quant.zig").Weight = .{};
    try testing.expect(v41.table.spec.?.trunk_weight.?(t.m, .embed, &w).isOk());
    try testing.expectEqual(m.mdl.embed.w.ctx, w.w.ctx);
    try testing.expect(v41.table.spec.?.trunk_weight.?(t.m, .head, &w).isOk());
    try testing.expectEqual(m.mdl.head.w.ctx, w.w.ctx);
    try testing.expectEqual(m.mdl.head.qp.bits, w.bits);
}

fn slotForwardRaw(t: *Twin, mode: abi.Mode, logits: abi.Logits, slots: []const u32, toks: []const u32, positions: []const u32) !abi.ForwardOut {
    var out: abi.ForwardOut = .{};
    const rows: u32 = if (mode == .batched_rows) 1 else @intCast(toks.len);
    const v = v41.table.forward(t.m, &.{ .mode = mode, .logits = logits, .rows = rows, .tokens = .of(toks), .slots = .of(slots), .positions = .of(positions) }, &out);
    if (!v.isOk()) {
        std.debug.print("forward: {s}\n", .{v.why.slice()});
        return error.PluginForward;
    }
    return out;
}

/// Runs a forward requesting no logits and verifies the plugin returns a null logits handle.
fn slotForwardNoLogits(t: *Twin, mode: abi.Mode, slots: []const u32, toks: []const u32, positions: []const u32) !void {
    const out = try slotForwardRaw(t, mode, .none, slots, toks, positions);
    if (out.logits.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = out.logits.ctx });
    try testing.expect(out.logits.ctx == null);
}

/// The plugin's logits `[rows, V]` (host f32, caller frees) for `toks` of `slots` at `positions`.
fn slotForwardLogits(t: *Twin, mode: abi.Mode, logits: abi.Logits, slots: []const u32, toks: []const u32, positions: []const u32) ![]f32 {
    std.debug.assert(logits != .none);
    const out = try slotForwardRaw(t, mode, logits, slots, toks, positions);
    const l: mlx.mlx_array = .{ .ctx = out.logits.ctx };
    defer _ = mlx.mlx_array_free(l);
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, l, .float32, t.s));
    try mlx.check(mlx.mlx_array_eval(f));
    return try testing.allocator.dupe(f32, (mlx.mlx_array_data_float32(f) orelse return error.NoData)[0..mlx.mlx_array_size(f)]);
}

/// The plugin's logits `[rows, V]` (host f32, caller frees) for `toks` of `slots` at `positions`.
fn slotForward(t: *Twin, mode: abi.Mode, slots: []const u32, toks: []const u32, positions: []const u32) ![]f32 {
    return slotForwardLogits(t, mode, .last_row, slots, toks, positions);
}

/// Three batched ticks of two slots (prompts `a` and `b`, decode tokens `da` and `db`), each tick's `[2, V]` logits.
fn batchTicks(files: []const sdk.fixture.File, a: []const u32, b: []const u32, da: [3]u32, db: [3]u32) ![3][]f32 {
    const t = try Twin.open(files, "");
    defer t.close();
    testing.allocator.free(try slotForward(t, .prefill, &.{1}, a, &.{0}));
    testing.allocator.free(try slotForward(t, .prefill, &.{2}, b, &.{0}));
    var out: [3][]f32 = undefined;
    for (0..3) |i| out[i] = try slotForward(t, .batched_rows, &.{ 1, 2 }, &.{ da[i], db[i] }, &.{ @intCast(a.len + i), @intCast(b.len + i) });
    return out;
}

test "deepseek_v41 plugin: a slot's row in a batch depends on its own state only, past the window, a pooling group and the indexer's budget" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const p = [2][]const u32{ &.{ 3, 17, 42, 9, 7, 25, 1, 60, 11, 2, 90 }, &.{ 5, 8, 13, 21, 34, 55, 89, 4, 6, 10, 12, 14, 16, 18 } };
    const d = [2][3]u32{ .{ 44, 2, 71 }, .{ 9, 30, 63 } };
    // Two rows run other kernels than one (the MoE's gather arm), so a row is compared at the same width: beside
    // another slot it equals itself beside a copy of its own conversation.
    const mixed = try batchTicks(files, p[0], p[1], d[0], d[1]);
    defer for (mixed) |x| testing.allocator.free(x);
    const own0 = try batchTicks(files, p[0], p[0], d[0], d[0]);
    defer for (own0) |x| testing.allocator.free(x);
    const own1 = try batchTicks(files, p[1], p[1], d[1], d[1]);
    defer for (own1) |x| testing.allocator.free(x);
    for (0..3) |i| {
        const V = mixed[i].len / 2;
        try testing.expectEqualSlices(f32, own0[i][0..V], mixed[i][0..V]);
        try testing.expectEqualSlices(f32, own1[i][V..], mixed[i][V..]);
    }
}

test "deepseek_v41 plugin: a one-slot batch is the slot's own decode step, bit for bit" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const prompt = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 60, 11, 2, 90 };
    const steps = [_]u32{ 44, 2, 71 };
    var ts: [2]*Twin = undefined;
    for (&ts) |*t| t.* = try Twin.open(files, "");
    defer for (ts) |t| t.close();
    for (ts) |t| testing.allocator.free(try slotForward(t, .prefill, &.{1}, &prompt, &.{0}));
    for (steps, 0..) |tok, i| {
        const pos: u32 = @intCast(prompt.len + i);
        const want = try slotForward(ts[0], .decode, &.{1}, &.{tok}, &.{pos});
        defer testing.allocator.free(want);
        const got = try slotForward(ts[1], .batched_rows, &.{1}, &.{tok}, &.{pos});
        defer testing.allocator.free(got);
        try testing.expectEqualSlices(f32, want, got);
    }
}

test "deepseek_v41 aux: overlapping captures survive either release order and restore independently" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const prompts = [2][]const u32{ &.{ 3, 17, 42, 9, 7, 25, 1, 60 }, &.{ 5, 8, 13, 21, 34, 55, 89, 4, 6 } };
    for ([_][2]u32{ .{ 1, 2 }, .{ 2, 1 } }) |order| {
        const t = try Twin.open(files, "");
        defer t.close();
        const aux = v41.table.cache.aux orelse return error.MissingAuxHooks;
        try testing.expect(aux.enabled(t.m));
        for (prompts, 1..) |p, slot| {
            try testing.expect(aux.begin_prompt(t.m, @intCast(slot), .of(p), .{}, 0, true).isOk());
            testing.allocator.free(try slotForward(t, .prefill, &.{@intCast(slot)}, p[0..2], &.{0}));
        }
        for (prompts, 1..) |p, slot| testing.allocator.free(try slotForward(t, .prefill, &.{@intCast(slot)}, p[2..], &.{2}));
        var snaps: [2]abi.Handle = .{ null, null };
        defer for (snaps) |sn| aux.free(t.m, sn);
        for (order) |slot| {
            const i = slot - 1;
            try testing.expect(aux.snapshot(t.m, slot, &snaps[i]).isOk());
            try testing.expect(snaps[i] != null);
            var empty: abi.Handle = null;
            try testing.expect(aux.snapshot(t.m, slot, &empty).isOk());
            try testing.expect(empty == null);
            v41.table.release_slot.?(t.m, slot);
            v41.table.release_slot.?(t.m, slot);
        }
        try testing.expectEqual(@as(u64, 0), v41.stateBytes(t.m));
        for (snaps, prompts) |sn, p| {
            var info: abi.AuxSnapshotInfo = .{};
            try testing.expect(aux.describe(t.m, sn, &info).isOk());
            try testing.expectEqual(p.len - 1, info.n);
            try testing.expectEqualSlices(u32, p[0 .. p.len - 1], info.ids.slice());
            try testing.expect(aux.bytes(t.m, sn) > 0);
            try testing.expectEqual(abi.Verdict.Code.unsupported, aux.restore_at(t.m, 3, sn, info.n - 1).code);
            var clone: abi.Handle = null;
            try testing.expectEqual(abi.Verdict.Code.unsupported, aux.slice(t.m, sn, info.n + 1, &clone).code);
            try testing.expect(clone == null);
            try testing.expect(aux.slice(t.m, sn, info.n, &clone).isOk());
            defer aux.free(t.m, clone);
            var warm: [2][]f32 = undefined;
            for ([_]u32{ 3, 4 }, 0..) |slot, k| {
                try testing.expect(aux.restore_at(t.m, slot, if (k == 0) sn else clone, info.n).isOk());
                try testing.expectEqual(abi.Verdict.Code.unsupported, aux.restore_at(t.m, slot, sn, info.n + 1).code);
                try testing.expectEqual(abi.Verdict.Code.failed, aux.begin_prompt(t.m, slot, .of(p), .{}, info.n - 1, false).code);
                try testing.expect(aux.begin_prompt(t.m, slot, .of(p), .{}, info.n, false).isOk());
                warm[k] = try slotForward(t, .prefill, &.{slot}, p[info.n..], &.{@intCast(info.n)});
            }
            defer for (warm) |row| testing.allocator.free(row);
            try testing.expectEqualSlices(f32, warm[0], warm[1]);
            try testing.expectEqual(abi.Verdict.Code.busy, aux.handoff(t.m, 3, 4).code);
            try testing.expect(aux.handoff(t.m, 3, 5).isOk());
            v41.table.release_slot.?(t.m, 3);
            v41.table.release_slot.?(t.m, 5);
            try testing.expect(aux.begin_prompt(t.m, 4, .of(p), .{}, 0, false).isOk());
            try testing.expect(aux.commit(t.m, 4, 0).isOk());
            testing.allocator.free(try slotForward(t, .prefill, &.{4}, p[0..2], &.{0}));
            testing.allocator.free(try slotForward(t, .prefill, &.{4}, p[2 .. p.len - 1], &.{2}));
            const cold = try slotForward(t, .prefill, &.{4}, p[p.len - 1 ..], &.{@intCast(p.len - 1)});
            defer testing.allocator.free(cold);
            try testing.expectEqualSlices(f32, cold, warm[0]);
            const reference = try Twin.open(files, "");
            defer reference.close();
            testing.allocator.free(try slotForward(reference, .prefill, &.{1}, p[0..2], &.{0}));
            testing.allocator.free(try slotForward(reference, .prefill, &.{1}, p[2 .. p.len - 1], &.{2}));
            const fresh = try slotForward(reference, .prefill, &.{1}, p[p.len - 1 ..], &.{@intCast(p.len - 1)});
            defer testing.allocator.free(fresh);
            for (fresh) |value| try testing.expect(std.math.isFinite(value));
            try testing.expectEqualSlices(f32, fresh, warm[0]);
            var no_capture: abi.Handle = null;
            try testing.expect(aux.snapshot(t.m, 4, &no_capture).isOk());
            try testing.expect(no_capture == null);
            v41.table.release_slot.?(t.m, 4);
        }
    }
}

test "deepseek_v41 aux: a snapshot sliced short of its end restores there by replaying its window" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var long: [41]u32 = undefined;
    for (&long, 0..) |*x, i| x.* = @intCast(3 + (i * 37 + i * i) % 90);
    const tail = [_]u32{ 11, 29, 5 };
    // The run the snapshot comes from and the cold runs it is judged against are chunked alike and kept on the wide
    // path, so a row is the same bits wherever it was made.
    const chunks = [_]usize{ 0, 8, 24, 40 };
    for ([_]bool{ false, true }) |dspark| {
        const files = if (dspark) try fixture.v41DsparkFiles(arena.allocator(), 7) else try fixture.v41Files(arena.allocator(), 7);
        const wide = try WideTwin.open(files);
        defer wide.close();
        const t = wide.t;
        const model = wide.model();
        const w: usize = model.mdl.cfg.sliding_window;
        const aux = v41.table.cache.aux orelse return error.MissingAuxHooks;
        try testing.expect(aux.begin_prompt(t.m, 1, .of(&long), .{}, 0, true).isOk());
        for (chunks[0..3], chunks[1..]) |lo, hi| testing.allocator.free(try slotForward(t, .prefill, &.{1}, long[lo..hi], &.{@intCast(lo)}));
        var sn: abi.Handle = null;
        try testing.expect(aux.snapshot(t.m, 1, &sn).isOk());
        try testing.expect(sn != null);
        defer aux.free(t.m, sn);
        const src: *const dsv41.Snap = @ptrCast(@alignCast(sn.?));
        v41.table.release_slot.?(t.m, 1);
        var ends: [2][]f32 = .{ &.{}, &.{} };
        defer for (ends) |e| testing.allocator.free(e);
        ends[0] = try restoredTail(t, aux, sn, &long, long[40..]);

        // A cut at one window replays the whole prefix (nothing is truncated); one at 31 stands at the pooling
        // group below it.
        for ([_][2]usize{ .{ 8, 8 }, .{ 24, 24 }, .{ 31, 30 } }) |c| {
            const n = c[1];
            var cut: abi.Handle = null;
            try testing.expect(aux.slice(t.m, sn, c[0], &cut).isOk());
            defer aux.free(t.m, cut);
            var info: abi.AuxSnapshotInfo = .{};
            try testing.expect(aux.describe(t.m, cut, &info).isOk());
            try testing.expectEqual(n, @as(usize, @intCast(info.n)));
            try testing.expectEqualSlices(u32, long[0..n], info.ids.slice());
            var out: abi.AuxExport = .{};
            try testing.expectEqual(abi.Verdict.Code.failed, aux.export_arrays(t.m, cut, &out).code);
            try testing.expectEqual(abi.Verdict.Code.unsupported, aux.restore_at(t.m, 2, cut, info.n + 2).code);
            try testing.expect(aux.restore_at(t.m, 2, cut, info.n).isOk());
            const ws = &model.mdl.dec_state.?;
            try testing.expectEqual(n, ws.n);

            const cold_twin = try WideTwin.open(files);
            defer cold_twin.close();
            const r = cold_twin.t;
            const cold_model = cold_twin.model();
            for (chunks[0..3], chunks[1..]) |lo, hi| {
                if (lo < n) testing.allocator.free(try slotForward(r, .prefill, &.{1}, long[lo..@min(hi, n)], &.{@intCast(lo)}));
            }
            const cs = &cold_model.mdl.dec_state.?;
            // Bars: the global rows and index keys are the snapshot's own, cut; the first layer's window is the cold
            // prompt's (nothing below it was truncated), and so is every layer's when the replay is the whole prefix.
            for (ws.layers, cs.layers, src.layers) |*wl, *cl, sl| {
                try testing.expect(wl.pend_kv.ctx == null and wl.pend_gate.ctx == null);
                inline for (.{ "comp", "idx_k" }) |f| if (@field(wl, f)) |rows| {
                    try testing.expectEqual(@field(cl, f).?.used, rows.used);
                    try expectRowsEqual(rows.buf, @field(sl, f).?.buf, 0, rows.used, rows.used);
                    if (n <= 24) try expectRowsEqual(rows.buf, @field(cl, f).?.buf, 0, rows.used, rows.used);
                };
                if (n == w) try expectRowsEqual(wl.win.buf, cl.win.buf, n - w, n, wl.win.cap) else try expectFinite(wl.win.buf);
            }
            if (n > w) {
                // The replay makes the first layer's window in one call of `w` rows; a row's bits can depend on its
                // call's width, so the cold prompt it is judged against ends with that same call.
                const split = try WideTwin.open(files);
                defer split.close();
                const r2 = split.t;
                const split_model = split.model();
                for (chunks[0..3], chunks[1..]) |lo, hi| {
                    if (lo < n - w) testing.allocator.free(try slotForward(r2, .prefill, &.{1}, long[lo..@min(hi, n - w)], &.{@intCast(lo)}));
                }
                testing.allocator.free(try slotForward(r2, .prefill, &.{1}, long[n - w .. n], &.{@intCast(n - w)}));
                const first = split_model.mdl.dec_state.?.layers[0].win;
                try expectRowsEqual(ws.layers[0].win.buf, first.buf, n - w, n, ws.layers[0].win.cap);
            }
            if (n == w) {
                for (ws.ds, cs.ds) |wr, cr| try expectRowsEqual(wr.buf, cr.buf, 0, n, wr.cap);
                if (ws.mh.ctx != null) try expectRowsEqual(ws.mh, cs.mh, 0, 1, 1);
            }
            try testing.expectEqualSlices(u32, cs.hist.items, ws.hist.items);

            var prompt: [44]u32 = undefined;
            @memcpy(prompt[0..n], long[0..n]);
            @memcpy(prompt[n..][0..tail.len], &tail);
            try testing.expect(aux.begin_prompt(t.m, 2, .of(prompt[0 .. n + tail.len]), .{}, info.n, false).isOk());
            const warm = try slotForward(t, .prefill, &.{2}, &tail, &.{@intCast(n)});
            defer testing.allocator.free(warm);
            v41.table.release_slot.?(t.m, 2);
            const cold = try slotForward(r, .prefill, &.{1}, &tail, &.{@intCast(n)});
            defer testing.allocator.free(cold);
            for (warm) |x| try testing.expect(std.math.isFinite(x));
            if (n == w) try testing.expectEqualSlices(f32, cold, warm);
        }
        // The snapshot the slices were cut from is untouched: it restores at its end as before them.
        ends[1] = try restoredTail(t, aux, sn, &long, long[40..]);
        try testing.expectEqualSlices(f32, ends[0], ends[1]);
    }
}

test "deepseek_v41 CED: a lead chunk that skips the decoder leaves what the chunk after it reads" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var ids: [40]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast(3 + (i * 37 + i * i) % 90);
    {
        const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), "");
        defer t.close();
        const model: *v41.Model = @ptrCast(@alignCast(t.m));
        const mdl = model.mdl;
        try testing.expect(mdl.ced_split != null);
        const w: usize = mdl.cfg.sliding_window;
        const skip = mdl.ced_min_skip;
        defer mdl.ced_min_skip = skip;
        mdl.ced_min_skip = w; // a chunk of two windows replays its last one
        // The same two chunks, the first with its decoder replay and without.
        var sts: [2]dsv41.State = undefined;
        var logits: [2][]f32 = undefined;
        for (&sts, &logits, [_]dsv41.Want{ .last, .encode }) |*st, *out, lead| {
            st.* = try dsv41.initState(mdl, testing.allocator);
            const head = try dsv41.extend(mdl, testing.allocator, st, ids[0..16], lead);
            if (head.ctx != null) _ = mlx.mlx_array_free(head);
            const rest = try dsv41.extend(mdl, testing.allocator, st, ids[16..], .last);
            defer _ = mlx.mlx_array_free(rest);
            out.* = try hostRows(rest);
        }
        defer for (&sts, logits) |*st, l| {
            st.deinit();
            testing.allocator.free(l);
        };
        try testing.expectEqual(@as(u64, 2 + 1), mdl.ced_stats.chunks);
        try testing.expectEqualSlices(f32, logits[0], logits[1]);
        const n = ids.len;
        for (sts[0].layers, sts[1].layers) |*a, *b| {
            inline for (.{ "comp", "idx_k" }) |f| if (@field(a, f)) |rows| {
                try testing.expectEqual(rows.used, @field(b, f).?.used);
                try expectRowsEqual(@field(b, f).?.buf, rows.buf, 0, rows.used, rows.used);
            };
            try expectRowsEqual(b.win.buf, a.win.buf, n - w, n, a.win.cap);
        }
    }
}

test "deepseek_v41 CED encode interior: multi-chunk prefill skips decoder on interior chunks and matches bit for bit" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    var p: [65]u32 = undefined;
    for (&p, 0..) |*x, i| x.* = @intCast(3 + (i * 37 + i * i) % 90);
    const aux = v41.table.cache.aux orelse return error.MissingAuxHooks;

    const cold_t = try Twin.open(files, "");
    defer cold_t.close();
    const cold_m: *v41.Model = @ptrCast(@alignCast(cold_t.m));
    const w: usize = cold_m.mdl.cfg.sliding_window;
    cold_m.mdl.ced_min_skip = w;
    try testing.expect(aux.begin_prompt(cold_t.m, 1, .of(&p), .{}, 0, true).isOk());
    for ([_][2]usize{ .{ 0, 16 }, .{ 16, 32 }, .{ 32, 48 }, .{ 48, 64 } }) |r| {
        testing.allocator.free(try slotForward(cold_t, .prefill, &.{1}, p[r[0]..r[1]], &.{@intCast(r[0])}));
    }
    const cold_logits = try slotForward(cold_t, .prefill, &.{1}, p[64..65], &.{64});
    defer testing.allocator.free(cold_logits);
    try testing.expectEqual(@as(u64, 0), cold_m.mdl.ced_stats.encode_chunks);
    try testing.expectEqual(@as(u64, 4), cold_m.mdl.ced_stats.chunks);

    const enc_t = try Twin.open(files, "");
    defer enc_t.close();
    const enc_m: *v41.Model = @ptrCast(@alignCast(enc_t.m));
    enc_m.mdl.ced_min_skip = w;
    try testing.expect(aux.begin_prompt(enc_t.m, 1, .of(&p), .{}, 0, true).isOk());
    for ([_][2]usize{ .{ 0, 16 }, .{ 16, 32 }, .{ 32, 48 } }) |r| {
        try slotForwardNoLogits(enc_t, .prefill, &.{1}, p[r[0]..r[1]], &.{@intCast(r[0])});
        try testing.expect(enc_m.mdl.dec_state.?.decoder_stale);
    }
    try testing.expectEqual(@as(u64, 3), enc_m.mdl.ced_stats.encode_chunks);
    try testing.expectEqual(@as(u64, 48), enc_m.mdl.ced_stats.encode_rows);
    try testing.expectEqual(@as(u64, 0), enc_m.mdl.ced_stats.chunks);
    testing.allocator.free(try slotForward(enc_t, .prefill, &.{1}, p[48..64], &.{48}));
    try testing.expect(!enc_m.mdl.dec_state.?.decoder_stale);
    const enc_logits = try slotForward(enc_t, .prefill, &.{1}, p[64..65], &.{64});
    defer testing.allocator.free(enc_logits);
    try testing.expectEqual(@as(u64, 1), enc_m.mdl.ced_stats.chunks);
    try testing.expectEqualSlices(f32, cold_logits, enc_logits);

    // Every encoder/decoder KV and window ring at [p.len - w, p.len) matches bit for bit.
    const st_a = &cold_m.mdl.dec_state.?;
    const st_b = &enc_m.mdl.dec_state.?;
    for (st_a.layers, st_b.layers) |*a, *b| {
        inline for (.{ "comp", "idx_k" }) |f| if (@field(a, f)) |rows| {
            try testing.expectEqual(rows.used, @field(b, f).?.used);
            try expectRowsEqual(@field(b, f).?.buf, rows.buf, 0, rows.used, rows.used);
        };
        try expectRowsEqual(b.win.buf, a.win.buf, p.len - w, p.len, a.win.cap);
    }

    // Short-tail boundary: when remaining tokens <= w + ced_min_skip, the chunk before the short tail runs `.last` (cedTail) instead of `.encode`.
    const short_p = p[0..55];
    const short_cold_t = try Twin.open(files, "");
    defer short_cold_t.close();
    const short_cold_m: *v41.Model = @ptrCast(@alignCast(short_cold_t.m));
    short_cold_m.mdl.ced_min_skip = w;
    try testing.expect(aux.begin_prompt(short_cold_t.m, 1, .of(short_p), .{}, 0, true).isOk());
    for ([_][2]usize{ .{ 0, 16 }, .{ 16, 32 }, .{ 32, 48 }, .{ 48, 54 } }) |r| {
        testing.allocator.free(try slotForward(short_cold_t, .prefill, &.{1}, short_p[r[0]..r[1]], &.{@intCast(r[0])}));
    }
    const short_cold = try slotForward(short_cold_t, .prefill, &.{1}, short_p[54..55], &.{54});
    defer testing.allocator.free(short_cold);

    const short_enc_t = try Twin.open(files, "");
    defer short_enc_t.close();
    const short_enc_m: *v41.Model = @ptrCast(@alignCast(short_enc_t.m));
    short_enc_m.mdl.ced_min_skip = w;
    try testing.expect(aux.begin_prompt(short_enc_t.m, 1, .of(short_p), .{}, 0, true).isOk());
    for ([_][2]usize{ .{ 0, 16 }, .{ 16, 32 }, .{ 32, 48 } }) |r| {
        try slotForwardNoLogits(short_enc_t, .prefill, &.{1}, short_p[r[0]..r[1]], &.{@intCast(r[0])});
    }
    // Chunks [0..16) and [16..32) encoded; [32..48) declined `.encode` because remaining=7 <= 2*w=16.
    try testing.expectEqual(@as(u64, 2), short_enc_m.mdl.ced_stats.encode_chunks);
    try testing.expect(!short_enc_m.mdl.dec_state.?.decoder_stale);
    testing.allocator.free(try slotForward(short_enc_t, .prefill, &.{1}, short_p[48..54], &.{48}));
    const short_enc = try slotForward(short_enc_t, .prefill, &.{1}, short_p[54..55], &.{54});
    defer testing.allocator.free(short_enc);
    try testing.expectEqualSlices(f32, short_cold, short_enc);
}

test "deepseek_v41 CED encode interior: mid-prefill snap_at and cancelled prefill resume (long and short suffix)" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    var p: [65]u32 = undefined;
    for (&p, 0..) |*x, i| x.* = @intCast(3 + (i * 37 + i * i) % 90);
    const aux = v41.table.cache.aux orelse return error.MissingAuxHooks;

    // Reference cold run for 65 tokens.
    const ref_t = try Twin.open(files, "");
    defer ref_t.close();
    const ref_m: *v41.Model = @ptrCast(@alignCast(ref_t.m));
    const w: usize = ref_m.mdl.cfg.sliding_window;
    ref_m.mdl.ced_min_skip = w;
    for ([_][2]usize{ .{ 0, 16 }, .{ 16, 32 }, .{ 32, 48 }, .{ 48, 64 } }) |r| {
        testing.allocator.free(try slotForward(ref_t, .prefill, &.{1}, p[r[0]..r[1]], &.{@intCast(r[0])}));
    }
    const cold = try slotForward(ref_t, .prefill, &.{1}, p[64..65], &.{64});
    defer testing.allocator.free(cold);

    // 1) Mid-prefill snap_at: chunks within `min_chunk` of snap_at must NOT leave `decoder_stale == true`,
    // and the split chunk containing snap_at runs `.last` + captures an exact snapshot.
    {
        const t = try Twin.open(files, "");
        defer t.close();
        const m: *v41.Model = @ptrCast(@alignCast(t.m));
        m.mdl.ced_min_skip = w;
        try testing.expect(aux.begin_prompt(t.m, 1, .of(&p), .{}, 0, true).isOk());
        // Override snap_at to land inside chunk [16..32) at position 24 (within min_chunk=16 of chunk 0 end 16).
        m.mdl.dec_state.?.snap_at = 24;
        m.mdl.dec_state.?.gpa.free(m.mdl.dec_state.?.snap_ids);
        m.mdl.dec_state.?.snap_ids = try m.mdl.dec_state.?.gpa.dupe(u32, p[0..24]);
        try slotForwardNoLogits(t, .prefill, &.{1}, p[0..16], &.{0});
        try testing.expectEqual(@as(u64, 0), m.mdl.ced_stats.encode_chunks);
        try testing.expect(!m.mdl.dec_state.?.decoder_stale);
        try slotForwardNoLogits(t, .prefill, &.{1}, p[16..32], &.{16});
        try testing.expectEqual(@as(u64, 0), m.mdl.ced_stats.encode_chunks);
        try testing.expect(!m.mdl.dec_state.?.decoder_stale);
        try testing.expect(m.mdl.dec_state.?.snap != null);
        try testing.expectEqual(@as(usize, 24), m.mdl.dec_state.?.snap.?.n);
        try slotForwardNoLogits(t, .prefill, &.{1}, p[32..48], &.{32});
        testing.allocator.free(try slotForward(t, .prefill, &.{1}, p[48..64], &.{48}));
        const got = try slotForward(t, .prefill, &.{1}, p[64..65], &.{64});
        defer testing.allocator.free(got);
        try testing.expectEqualSlices(f32, cold, got);
    }
    {
        const t = try Twin.open(files, "");
        defer t.close();
        const m: *v41.Model = @ptrCast(@alignCast(t.m));
        m.mdl.ced_min_skip = w;
        try testing.expect(aux.begin_prompt(t.m, 1, .of(&p), .{}, 0, true).isOk());
        // Override snap_at to 40 (inside chunk [32..48)): chunk 0 encodes, chunk 1 repairs via cedTail before split at 40.
        m.mdl.dec_state.?.snap_at = 40;
        m.mdl.dec_state.?.gpa.free(m.mdl.dec_state.?.snap_ids);
        m.mdl.dec_state.?.snap_ids = try m.mdl.dec_state.?.gpa.dupe(u32, p[0..40]);
        try slotForwardNoLogits(t, .prefill, &.{1}, p[0..16], &.{0});
        try testing.expectEqual(@as(u64, 1), m.mdl.ced_stats.encode_chunks);
        try testing.expect(m.mdl.dec_state.?.decoder_stale);
        try slotForwardNoLogits(t, .prefill, &.{1}, p[16..32], &.{16});
        try testing.expectEqual(@as(u64, 1), m.mdl.ced_stats.encode_chunks);
        try testing.expect(!m.mdl.dec_state.?.decoder_stale);
        try slotForwardNoLogits(t, .prefill, &.{1}, p[32..48], &.{32});
        try testing.expectEqual(@as(u64, 1), m.mdl.ced_stats.encode_chunks);
        try testing.expect(!m.mdl.dec_state.?.decoder_stale);
        try testing.expect(m.mdl.dec_state.?.snap != null);
        try testing.expectEqual(@as(usize, 40), m.mdl.dec_state.?.snap.?.n);
        testing.allocator.free(try slotForward(t, .prefill, &.{1}, p[48..64], &.{48}));
        const got = try slotForward(t, .prefill, &.{1}, p[64..65], &.{64});
        defer testing.allocator.free(got);
        try testing.expectEqualSlices(f32, cold, got);
    }

    // 2) Cancelled prefill after `.encode` chunk (`decoder_stale == true`): snapshot is skipped (`snap == null`),
    // and direct decode on stale state without a `cedTail` / `.replay` chunk is refused (`error.Dsv41DecodeOnStaleState`).
    {
        const t = try Twin.open(files, "");
        defer t.close();
        const m: *v41.Model = @ptrCast(@alignCast(t.m));
        m.mdl.ced_min_skip = w;
        try testing.expect(aux.begin_prompt(t.m, 1, .of(&p), .{}, 0, true).isOk());
        try slotForwardNoLogits(t, .prefill, &.{1}, p[0..16], &.{0});
        try slotForwardNoLogits(t, .prefill, &.{1}, p[16..32], &.{16});
        try testing.expect(m.mdl.dec_state.?.decoder_stale);

        var snap_stale: abi.Handle = null;
        try testing.expect(aux.snapshot(t.m, 1, &snap_stale).isOk());
        try testing.expectEqual(@as(abi.Handle, null), snap_stale);
        try testing.expect(m.mdl.dec_state.?.decoder_stale);
        try testing.expectError(
            error.Dsv41DecodeOnStaleState,
            dsv41.extend(m.mdl, testing.allocator, &m.mdl.dec_state.?, p[32..33], .last),
        );
        v41.table.release_slot.?(t.m, 1);
    }

    // 3) Cancelled prefill after `.encode` chunks followed by a `.last` (`cedTail`) chunk (`decoder_stale == false`):
    // snapshot captures an exact `whole` state, and resuming with both >= w + ced_min_skip and < w + ced_min_skip
    // remaining rows produces bit-identical logits to cold prefill.
    {
        const short_prompt = p[0..55]; // suffix 48..55 has 7 rows (< w + ced_min_skip = 16)
        const cold_short_t = try Twin.open(files, "");
        defer cold_short_t.close();
        const cold_short_m: *v41.Model = @ptrCast(@alignCast(cold_short_t.m));
        cold_short_m.mdl.ced_min_skip = w;
        testing.allocator.free(try slotForward(cold_short_t, .prefill, &.{1}, short_prompt[0..16], &.{0}));
        testing.allocator.free(try slotForward(cold_short_t, .prefill, &.{1}, short_prompt[16..32], &.{16}));
        testing.allocator.free(try slotForward(cold_short_t, .prefill, &.{1}, short_prompt[32..48], &.{32}));
        testing.allocator.free(try slotForward(cold_short_t, .prefill, &.{1}, short_prompt[48..54], &.{48}));
        const cold_short = try slotForward(cold_short_t, .prefill, &.{1}, short_prompt[54..55], &.{54});
        defer testing.allocator.free(cold_short);

        const t = try Twin.open(files, "");
        defer t.close();
        const m: *v41.Model = @ptrCast(@alignCast(t.m));
        m.mdl.ced_min_skip = w;
        try testing.expect(aux.begin_prompt(t.m, 1, .of(&p), .{}, 0, true).isOk());
        try slotForwardNoLogits(t, .prefill, &.{1}, p[0..16], &.{0});
        try slotForwardNoLogits(t, .prefill, &.{1}, p[16..32], &.{16});
        try testing.expect(m.mdl.dec_state.?.decoder_stale);
        testing.allocator.free(try slotForward(t, .prefill, &.{1}, p[32..48], &.{32}));
        try testing.expect(!m.mdl.dec_state.?.decoder_stale);

        var snap_48: abi.Handle = null;
        try testing.expect(aux.snapshot(t.m, 1, &snap_48).isOk());
        try testing.expect(snap_48 != null);
        defer aux.free(t.m, snap_48);

        var info: abi.AuxSnapshotInfo = .{};
        try testing.expect(aux.describe(t.m, snap_48, &info).isOk());
        try testing.expectEqual(@as(u64, 48), info.n);
        v41.table.release_slot.?(t.m, 1);

        // Long suffix (>= w + ced_min_skip: 17 rows remaining).
        try testing.expect(aux.restore_at(t.m, 2, snap_48, 48).isOk());
        try testing.expect(aux.begin_prompt(t.m, 2, .of(&p), .{}, 48, true).isOk());
        testing.allocator.free(try slotForward(t, .prefill, &.{2}, p[48..64], &.{48}));
        const warm_long = try slotForward(t, .prefill, &.{2}, p[64..65], &.{64});
        defer testing.allocator.free(warm_long);
        try testing.expectEqualSlices(f32, cold, warm_long);
        v41.table.release_slot.?(t.m, 2);

        // Short suffix (< w + ced_min_skip: 7 rows remaining).
        try testing.expect(aux.restore_at(t.m, 3, snap_48, 48).isOk());
        try testing.expect(aux.begin_prompt(t.m, 3, .of(short_prompt), .{}, 48, true).isOk());
        testing.allocator.free(try slotForward(t, .prefill, &.{3}, short_prompt[48..54], &.{48}));
        const warm_short = try slotForward(t, .prefill, &.{3}, short_prompt[54..55], &.{54});
        defer testing.allocator.free(warm_short);
        try testing.expectEqualSlices(f32, cold_short, warm_short);
    }
}

test "deepseek_v41 CED encode interior: all_rows, pair, TP and disabled CED never encode" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    var p: [40]u32 = undefined;
    for (&p, 0..) |*x, i| x.* = @intCast(3 + (i * 37 + i * i) % 90);
    const aux = v41.table.cache.aux orelse return error.MissingAuxHooks;

    const t = try Twin.open(files, "");
    defer t.close();
    const m: *v41.Model = @ptrCast(@alignCast(t.m));
    const w: usize = m.mdl.cfg.sliding_window;
    m.mdl.ced_min_skip = w;

    // 1) `.all_rows` (prompt logprobs path) never encodes and returns [L, V].
    try testing.expect(aux.begin_prompt(t.m, 1, .of(&p), .{}, 0, true).isOk());
    const all_logits = try slotForwardLogits(t, .prefill, .all_rows, &.{1}, p[0..16], &.{0});
    defer testing.allocator.free(all_logits);
    try testing.expectEqual(16 * m.mdl.vocab, all_logits.len);
    try testing.expectEqual(@as(u64, 0), m.mdl.ced_stats.encode_chunks);
    v41.table.release_slot.?(t.m, 1);

    // 2) `m.pair != null`, `m.tp != null`, and `m.prefill_whole != null` refuse `Want.encode`.
    var st = try dsv41.initState(m.mdl, testing.allocator);
    defer st.deinit();
    m.mdl.pair = .{ .lo = 1, .hi = 2, .worker = false };
    try testing.expectError(error.Dsv41EncodeUnsupported, dsv41.extend(m.mdl, testing.allocator, &st, p[0..16], .encode));
    m.mdl.pair = null;
    m.mdl.tp = .{ .rank = 0, .size = 4 };
    try testing.expectError(error.Dsv41EncodeUnsupported, dsv41.extend(m.mdl, testing.allocator, &st, p[0..16], .encode));
    m.mdl.tp = null;
    m.mdl.prefill_whole = m.mdl;
    try testing.expectError(error.Dsv41EncodeUnsupported, dsv41.extend(m.mdl, testing.allocator, &st, p[0..16], .encode));
    m.mdl.prefill_whole = null;

    // 3) Disabled CED (`MLX_SERVE_DSV41_CED=0`) never encodes even on `.none` forwards.
    const prev_ced = env.get(.MLX_SERVE_DSV41_CED);
    env.set(.MLX_SERVE_DSV41_CED, "0");
    dsv41.resetCedEnvForTest();
    defer {
        if (prev_ced) |v| env.set(.MLX_SERVE_DSV41_CED, v) else env.unset(.MLX_SERVE_DSV41_CED);
        dsv41.resetCedEnvForTest();
    }
    try testing.expect(aux.begin_prompt(t.m, 2, .of(&p), .{}, 0, true).isOk());
    try slotForwardNoLogits(t, .prefill, &.{2}, p[0..16], &.{0});
    try testing.expectEqual(@as(u64, 0), m.mdl.ced_stats.encode_chunks);
    try testing.expect(!m.mdl.dec_state.?.decoder_stale);
}

/// The logits of `tail` on a slot restored from `sn` at its end (the prompt `ids`).
fn restoredTail(t: *Twin, aux: *const abi.AuxHooks, sn: abi.Handle, ids: []const u32, tail: []const u32) ![]f32 {
    const n = ids.len - tail.len;
    try testing.expect(aux.restore_at(t.m, 3, sn, n).isOk());
    try testing.expect(aux.begin_prompt(t.m, 3, .of(ids), .{}, n, false).isOk());
    defer v41.table.release_slot.?(t.m, 3);
    return slotForward(t, .prefill, &.{3}, tail, &.{@intCast(n)});
}

fn hostRows(a: mlx.mlx_array) ![]f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, a, .float32, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(f));
    return testing.allocator.dupe(f32, (mlx.mlx_array_data_float32(f) orelse return error.NoData)[0..mlx.mlx_array_size(f)]);
}

/// A twin kept on the wide prompt path (its compiled regions set aside until close).
const WideTwin = struct {
    t: *Twin,
    regions: @FieldType(std.meta.Child(@FieldType(v41.Model, "mdl")), "regions"),

    fn open(files: anytype) !WideTwin {
        const t = try Twin.open(files, "");
        const m: *v41.Model = @ptrCast(@alignCast(t.m));
        const regions = m.mdl.regions;
        m.mdl.regions = regions[0..0];
        return .{ .t = t, .regions = regions };
    }

    fn model(self: WideTwin) *v41.Model {
        return @ptrCast(@alignCast(self.t.m));
    }

    fn close(self: WideTwin) void {
        self.model().mdl.regions = self.regions;
        self.t.close();
    }
};

/// Rows of positions [lo, hi) of two `[cap, width]` arrays (a ring's slots, or rows in order when `cap` covers them).
fn expectRowsEqual(a: mlx.mlx_array, b: mlx.mlx_array, lo: usize, hi: usize, cap: usize) !void {
    const x = try hostRows(a);
    defer testing.allocator.free(x);
    const y = try hostRows(b);
    defer testing.allocator.free(y);
    const width = x.len / @as(usize, @intCast(mlx.getShape(a)[0]));
    for (lo..hi) |p| {
        const s = if (cap == 0) p else p % @max(cap, 1);
        try testing.expectEqualSlices(f32, y[s * width ..][0..width], x[s * width ..][0..width]);
    }
}

fn expectFinite(a: mlx.mlx_array) !void {
    const x = try hostRows(a);
    defer testing.allocator.free(x);
    for (x) |v| try testing.expect(std.math.isFinite(v));
}

test "deepseek_v41 aux: a prompt abandoned mid-prefill leaves a snapshot at its last chunk, and a restore there continues the cold prefill bit for bit" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    var p: [40]u32 = undefined;
    for (&p, 0..) |*id, i| id.* = @intCast((i * 7 + 3) % 90 + 2);
    const cut = 23; // inside a pooling group and past the window: any chunk end is a restore point
    const t = try Twin.open(files, "");
    defer t.close();
    const aux = v41.table.cache.aux orelse return error.MissingAuxHooks;

    // Armed for its end, forwarded up to `cut`, then abandoned.
    try testing.expect(aux.begin_prompt(t.m, 1, .of(&p), .{}, 0, true).isOk());
    var none: abi.Handle = null;
    try testing.expect(aux.snapshot(t.m, 1, &none).isOk());
    try testing.expect(none == null); // nothing forwarded yet
    testing.allocator.free(try slotForward(t, .prefill, &.{1}, p[0..9], &.{0}));
    testing.allocator.free(try slotForward(t, .prefill, &.{1}, p[9..cut], &.{9}));
    var sn: abi.Handle = null;
    defer aux.free(t.m, sn);
    try testing.expect(aux.snapshot(t.m, 1, &sn).isOk());
    try testing.expect(sn != null);
    var info: abi.AuxSnapshotInfo = .{};
    try testing.expect(aux.describe(t.m, sn, &info).isOk());
    try testing.expectEqual(@as(u64, cut), info.n);
    try testing.expectEqualSlices(u32, p[0..cut], info.ids.slice());
    try testing.expect(aux.snapshot(t.m, 1, &none).isOk());
    try testing.expect(none == null); // handed over once
    v41.table.release_slot.?(t.m, 1);

    // The same prompt resumes from it and still captures its own end.
    try testing.expect(aux.restore_at(t.m, 2, sn, cut).isOk());
    try testing.expect(aux.begin_prompt(t.m, 2, .of(&p), .{}, cut, true).isOk());
    testing.allocator.free(try slotForward(t, .prefill, &.{2}, p[cut .. p.len - 1], &.{cut}));
    const warm = try slotForward(t, .prefill, &.{2}, p[p.len - 1 ..], &.{@intCast(p.len - 1)});
    defer testing.allocator.free(warm);
    var end: abi.Handle = null;
    defer aux.free(t.m, end);
    try testing.expect(aux.snapshot(t.m, 2, &end).isOk());
    try testing.expect(aux.describe(t.m, end, &info).isOk());
    try testing.expectEqual(@as(u64, p.len - 1), info.n);

    const reference = try Twin.open(files, "");
    defer reference.close();
    testing.allocator.free(try slotForward(reference, .prefill, &.{1}, p[0..9], &.{0}));
    testing.allocator.free(try slotForward(reference, .prefill, &.{1}, p[9..cut], &.{9}));
    testing.allocator.free(try slotForward(reference, .prefill, &.{1}, p[cut .. p.len - 1], &.{cut}));
    const cold = try slotForward(reference, .prefill, &.{1}, p[p.len - 1 ..], &.{@intCast(p.len - 1)});
    defer testing.allocator.free(cold);
    for (cold) |value| try testing.expect(std.math.isFinite(value));
    try testing.expectEqualSlices(f32, cold, warm);
}

test "deepseek_v41 aux: disabled capture and short prompts retain no checkpoint" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), "");
    defer t.close();
    const aux = v41.table.cache.aux orelse return error.MissingAuxHooks;
    for ([_][]const u32{ &.{}, &.{3}, &.{ 3, 4, 5 } }) |p| {
        try testing.expect(aux.begin_prompt(t.m, 1, .of(p), .{}, 0, p.len < 2).isOk());
        try testing.expect(aux.reserve(t.m, 1, p.len).isOk());
        if (p.len > 0) testing.allocator.free(try slotForward(t, .prefill, &.{1}, p, &.{0}));
        try testing.expect(aux.commit(t.m, 1, p.len).isOk());
        try testing.expectEqual(abi.Verdict.Code.unsupported, aux.commit(t.m, 1, p.len + 1).code);
        var sn: abi.Handle = null;
        try testing.expect(aux.snapshot(t.m, 1, &sn).isOk());
        try testing.expect(sn == null);
        v41.table.release_slot.?(t.m, 1);
    }
}

test "deepseek_v41 aux: candidate image key ignores later rows and eligibility excludes streaming banks and paired instances" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), "");
    defer t.close();
    const m: *v41.Model = @ptrCast(@alignCast(t.m));
    const aux = v41.table.cache.aux orelse return error.MissingAuxHooks;
    m.mdl.cfg.image_token_id = 17;
    const a = mlx.mlx_array_new_data(&[_]f32{ 1, 2, 3, 4 }, &[_]c_int{ 2, 2 }, 2, .float32);
    defer _ = mlx.mlx_array_free(a);
    const b = mlx.mlx_array_new_data(&[_]f32{ 1, 2, 9, 8 }, &[_]c_int{ 2, 2 }, 2, .float32);
    defer _ = mlx.mlx_array_free(b);
    var key: [4]u64 = undefined;
    try testing.expect(aux.prefix_key(t.m, .of(&.{ 3, 17, 4 }), .{ .ctx = a.ctx }, &key[0]).isOk());
    try testing.expect(aux.prefix_key(t.m, .of(&.{ 3, 17, 4 }), .{ .ctx = b.ctx }, &key[1]).isOk());
    try testing.expect(aux.prefix_key(t.m, .of(&.{ 3, 17, 4, 17 }), .{ .ctx = a.ctx }, &key[2]).isOk());
    try testing.expect(aux.prefix_key(t.m, .of(&.{ 3, 17, 4, 17 }), .{ .ctx = b.ctx }, &key[3]).isOk());
    try testing.expectEqual(key[0], key[1]);
    try testing.expect(key[2] != key[3]);
    try testing.expect(aux.prefix_key(t.m, .of(&.{ 3, 17, 4 }), .{}, &key[1]).isOk());
    try testing.expect(key[0] != key[1]);
    try testing.expect(aux.enabled(t.m));
    m.cfg.dsv41_streamed = true;
    try testing.expect(aux.enabled(t.m));
    m.cfg.dsv41_streamed = false;
    m.bank_streams = true;
    try testing.expect(!aux.enabled(t.m));
    m.bank_streams = false;
    try testing.expect(v41.bankStreams(146, 183, 384));
    try testing.expect(v41.bankStreams(384, 200, 384));
    try testing.expect(!v41.bankStreams(384, 384, 384));
    m.mdl.pair = .{ .lo = 2, .hi = 4, .worker = false };
    defer m.mdl.pair = null;
    try testing.expect(!aux.enabled(t.m));
    try testing.expectEqual(abi.Verdict.Code.unsupported, aux.begin_prompt(t.m, 1, .of(&.{ 3, 4 }), .{}, 0, true).code);
}

const codec = @import("cache_codec.zig");
const cache_pack = @import("cache_pack.zig");
const dsv41 = @import("deepseek_v41.zig");

test "cache pack kernels round trip signed bf16 rows and mark one off-grid group corrupt" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const stream = mlx.gpuStream();
    var kernels = try cache_pack.Kernels.init();
    defer kernels.deinit();
    const gpa = testing.allocator;
    const levels = [_]u16{ 0, 0x3f00, 0x3f80, 0x3fc0, 0x4000, 0x4040, 0x4080, 0x40c0 };
    for ([_]codec.Encoding{ .fp4_e4m3_g16, .fp4_ue8m0_g32, .fp8_ue8m0_g32 }) |encoding| {
        const group: usize = if (encoding == .fp4_e4m3_g16) 16 else 32;
        const width = group * 300;
        const rows = 2;
        const input = try gpa.alloc(u16, rows * width);
        defer gpa.free(input);
        for (input, 0..) |*value, i| {
            const lane = i % 8;
            const sign: u16 = if (i % 3 == 0) 0x8000 else 0;
            value.* = levels[lane] | sign;
        }
        if (encoding == .fp4_e4m3_g16) {
            @memset(input[0..group], 0);
            input[0] = 0x8000;
            input[1] = 0x3b00;
        }
        const spec: codec.Spec = .{ .kind = if (encoding == .fp4_e4m3_g16) .comp else if (encoding == .fp4_ue8m0_g32) .index else .win, .rows = rows, .width = width, .dtype = .bfloat16 };
        const shape = [_]c_int{ @intCast(rows), @intCast(width) };
        const original = mlx.mlx_array_new_data(input.ptr, &shape, 2, .bfloat16);
        defer _ = mlx.mlx_array_free(original);
        var bits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bits);
        try mlx.check(mlx.mlx_view(&bits, original, .uint16, stream));
        const output = try cache_pack.encode(&kernels, stream, bits, encoding);
        defer _ = mlx.mlx_array_free(output);
        try mlx.check(mlx.mlx_array_eval(output));
        const encoded = mlx.mlx_array_data_uint8(output) orelse return error.NoData;
        const row_bytes = codec.packedRowBytes(encoding, width).?;
        if (encoding == .fp4_e4m3_g16) try testing.expectEqual(@as(u8, 1), encoded[width / 2]);
        const zig_decoded = try gpa.alloc(u16, input.len);
        defer gpa.free(zig_decoded);
        try codec.unpackBf16(spec, encoded[0 .. rows * row_bytes], zig_decoded);
        try testing.expectEqualSlices(u16, input, zig_decoded);
        const restored = try cache_pack.decode(&kernels, stream, output, spec, encoding);
        defer _ = mlx.mlx_array_free(restored.bits);
        defer _ = mlx.mlx_array_free(restored.status);
        const evaluated = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{ restored.bits, restored.status }, 2);
        defer _ = mlx.mlx_vector_array_free(evaluated);
        try mlx.check(mlx.mlx_eval(evaluated));
        const status = mlx.mlx_array_data_uint32(restored.status) orelse return error.NoData;
        for (status[0 .. rows * (width / group)]) |flag| try testing.expectEqual(@as(u32, 0), flag);
        var restored_bf16 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(restored_bf16);
        try mlx.check(mlx.mlx_view(&restored_bf16, restored.bits, .bfloat16, stream));
        try mlx.check(mlx.mlx_array_eval(restored_bf16));
        try testing.expectEqualSlices(u16, input, mlx.mlx_array_data_bfloat16(restored_bf16).?[0..input.len]);
    }

    const bad_values = [_]u16{ 0x3f01, 0x4040, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    const bad_shape = [_]c_int{ 1, 16 };
    const bad_input = mlx.mlx_array_new_data(&bad_values, &bad_shape, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(bad_input);
    var bad_bits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bad_bits);
    try mlx.check(mlx.mlx_view(&bad_bits, bad_input, .uint16, stream));
    const bad_spec: codec.Spec = .{ .kind = .comp, .rows = 1, .width = 16, .dtype = .bfloat16 };
    const bad_output = try cache_pack.encode(&kernels, stream, bad_bits, .fp4_e4m3_g16);
    defer _ = mlx.mlx_array_free(bad_output);
    try mlx.check(mlx.mlx_array_eval(bad_output));
    const bad_bytes = mlx.mlx_array_data_uint8(bad_output) orelse return error.NoData;
    try testing.expectEqual(@as(u8, 0), bad_bytes[8]);
    const rejected = try cache_pack.decode(&kernels, stream, bad_output, bad_spec, .fp4_e4m3_g16);
    defer _ = mlx.mlx_array_free(rejected.bits);
    defer _ = mlx.mlx_array_free(rejected.status);
    const rejected_arrays = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{ rejected.bits, rejected.status }, 2);
    defer _ = mlx.mlx_vector_array_free(rejected_arrays);
    try mlx.check(mlx.mlx_eval(rejected_arrays));
    try testing.expectEqual(@as(u32, 1), mlx.mlx_array_data_uint32(rejected.status).?[0]);
}

// Two l20.comp groups from a live 32k export (rows 8 and 71). The writer took the scale from the max before rounding
// (about 4.5-5.0 x 2^-8, E4M3 subnormal 2^-8), so the stored max is 4 x scale, and re-deriving the scale from it gives
// 2^-9, which puts 0.015625 at 8 x scale, off the grid.
test "cache pack keeps comp groups whose stored max rounded down to 4 x a subnormal scale" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const stream = mlx.gpuStream();
    var kernels = try cache_pack.Kernels.init();
    defer kernels.deinit();
    const values = [_]u16{
        0x3c00, 0x3b80, 0x3b00, 0xbb00, 0xbc00, 0x3b00, 0x3c00, 0xbbc0, 0x3b80, 0xbc40, 0x3c40, 0xbbc0, 0x3c40, 0x3c80, 0x3c80, 0xbc40,
        0x3c80, 0x3b00, 0xbb00, 0xbb00, 0x3c40, 0xbc00, 0x3c00, 0x3b00, 0x0000, 0x3c00, 0xbb00, 0x3b80, 0x3bc0, 0xbb00, 0x3b00, 0x3b80,
    };
    const shape = [_]c_int{ 2, 16 };
    const input = mlx.mlx_array_new_data(&values, &shape, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(input);
    var bits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bits);
    try mlx.check(mlx.mlx_view(&bits, input, .uint16, stream));
    const output = try cache_pack.encode(&kernels, stream, bits, .fp4_e4m3_g16);
    defer _ = mlx.mlx_array_free(output);
    try mlx.check(mlx.mlx_array_eval(output));
    const encoded = mlx.mlx_array_data_uint8(output) orelse return error.NoData;
    const spec: codec.Spec = .{ .kind = .comp, .rows = 2, .width = 16, .dtype = .bfloat16 };
    const row_bytes = codec.packedRowBytes(.fp4_e4m3_g16, 16).?;
    // E4M3 subnormal 2 x 2^-9 = 2^-8.
    for (0..2) |row| try testing.expectEqual(@as(u8, 0x02), encoded[row * row_bytes + 8]);
    var decoded: [32]u16 = undefined;
    try codec.unpackBf16(spec, encoded[0 .. 2 * row_bytes], &decoded);
    try testing.expectEqualSlices(u16, &values, &decoded);
}

test "cache pack keeps sentinel scales nonzero after exhaustive bf16 decode" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const stream = mlx.gpuStream();
    var kernels = try cache_pack.Kernels.init();
    defer kernels.deinit();
    const rows: usize = 32640;
    for ([_]codec.Encoding{ .fp4_e4m3_g16, .fp4_ue8m0_g32, .fp8_ue8m0_g32 }) |encoding| {
        const group: usize = if (encoding == .fp4_e4m3_g16) 16 else 32;
        const width = group;
        const input = try testing.allocator.alloc(u16, rows * width);
        defer testing.allocator.free(input);
        @memset(input, 0);
        var row: usize = 0;
        for (0..255) |exponent| {
            if (exponent == 255) continue;
            for (0..128) |mantissa| {
                input[row * width] = @intCast((exponent << 7) | mantissa);
                row += 1;
            }
        }
        try testing.expectEqual(rows, row);
        const shape = [_]c_int{ @intCast(rows), @intCast(width) };
        const original = mlx.mlx_array_new_data(input.ptr, &shape, 2, .bfloat16);
        defer _ = mlx.mlx_array_free(original);
        var bits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bits);
        try mlx.check(mlx.mlx_view(&bits, original, .uint16, stream));
        const packed_bits = try cache_pack.encode(&kernels, stream, bits, encoding);
        defer _ = mlx.mlx_array_free(packed_bits);
        const spec: codec.Spec = .{ .kind = if (encoding == .fp4_e4m3_g16) .comp else if (encoding == .fp4_ue8m0_g32) .index else .win, .rows = rows, .width = width, .dtype = .bfloat16 };
        const decoded = try cache_pack.decode(&kernels, stream, packed_bits, spec, encoding);
        defer _ = mlx.mlx_array_free(decoded.bits);
        defer _ = mlx.mlx_array_free(decoded.status);
        const repacked = try cache_pack.encode(&kernels, stream, decoded.bits, encoding);
        defer _ = mlx.mlx_array_free(repacked);
        try mlx.check(mlx.mlx_array_eval(repacked));
        const bytes = mlx.mlx_array_data_uint8(repacked) orelse return error.NoData;
        const row_bytes = codec.packedRowBytes(encoding, width).?;
        const code_bytes = if (encoding == .fp8_ue8m0_g32) width else width / 2;
        for (0..rows) |i| try testing.expect(bytes[i * row_bytes + code_bytes] != 0);
    }
}

const Exported = struct {
    tensors: [codec.max_tensors]abi.AuxTensor = undefined,
    names: [16 * 1024]u8 = undefined,
    header: [codec.max_header]u8 = undefined,
    out: abi.AuxExport = .{},

    fn capture(self: *Exported, t: *Twin, p: []const u32) !void {
        const aux = v41.table.cache.aux.?;
        try testing.expect(aux.begin_prompt(t.m, 1, .of(p), .{}, 0, true).isOk());
        const first = @min(2, p.len - 1);
        testing.allocator.free(try slotForward(t, .prefill, &.{1}, p[0..first], &.{0}));
        if (first < p.len - 1) testing.allocator.free(try slotForward(t, .prefill, &.{1}, p[first .. p.len - 1], &.{@intCast(first)}));
        var sn: abi.Handle = null;
        try testing.expect(aux.snapshot(t.m, 1, &sn).isOk());
        defer aux.free(t.m, sn);
        try testing.expect(sn != null);
        self.out = .{ .tensors = &self.tensors, .tensor_cap = self.tensors.len, .names = &self.names, .names_cap = self.names.len, .header = &self.header, .header_cap = self.header.len };
        const v = aux.export_arrays(t.m, sn, &self.out);
        if (!v.isOk()) std.debug.print("export: {s}\n", .{v.why.slice()});
        try testing.expect(v.isOk());
        v41.table.release_slot.?(t.m, 1);
    }

    fn deinit(self: *Exported) void {
        for (self.arrays()) |t| _ = mlx.mlx_array_free(.{ .ctx = t.array.ctx });
        self.out.tensor_len = 0;
    }

    fn arrays(self: *Exported) []abi.AuxTensor {
        return self.tensors[0..self.out.tensor_len];
    }

    fn bytes(self: *Exported) abi.Str {
        return .of(self.header[0..self.out.header_len]);
    }
};

fn expectImportRefused(t: *Twin, e: *Exported, ids: []const u32, arrays: []const abi.AuxTensor) !void {
    var sn: abi.Handle = null;
    const v = v41.table.cache.aux.?.import_arrays(t.m, .of(ids), e.bytes(), .of(arrays), &sn);
    defer v41.table.cache.aux.?.free(t.m, sn);
    try testing.expectEqual(abi.Verdict.Code.failed, v.code);
    try testing.expect(v.why.len > 0 and sn == null);
}

fn expectInvalidImport(t: *Twin, e: *Exported, ids: []const u32, arrays: []const abi.AuxTensor) !void {
    try expectImportRefused(t, e, ids, arrays);
    const m: *v41.Model = @ptrCast(@alignCast(t.m));
    var failing = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 0 });
    const unexpected = dsv41.importSnapshot(m.mdl, failing.allocator(), ids, e.bytes().slice(), arrays) catch {
        try testing.expect(!failing.has_induced_failure);
        return;
    };
    unexpected.deinit(failing.allocator());
    failing.allocator().destroy(unexpected);
    return error.InvalidTensorAccepted;
}

test "codec rejects duplicate, missing, unexpected and malformed tensors before construction" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41DsparkFiles(arena.allocator(), 7), "");
    defer t.close();
    const p = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 60 };
    var e: Exported = .{};
    try e.capture(t, &p);
    defer e.deinit();
    const ids = p[0 .. p.len - 1];
    try expectInvalidImport(t, &e, ids, e.arrays()[0 .. e.arrays().len - 1]);
    const last = e.arrays().len - 1;
    const saved = e.tensors[last];
    e.tensors[last] = e.tensors[0];
    try expectInvalidImport(t, &e, ids, e.arrays());
    e.tensors[last] = saved;
    for (e.arrays()) |*tensor| {
        const original = tensor.*;
        tensor.name = .of("unexpected");
        try expectInvalidImport(t, &e, ids, e.arrays());
        tensor.* = original;
        tensor.array = .{};
        try expectInvalidImport(t, &e, ids, e.arrays());
        tensor.* = original;
        tensor.size = 0;
        try expectInvalidImport(t, &e, ids, e.arrays());
        tensor.* = original;
        const x: mlx.mlx_array = .{ .ctx = tensor.array.ctx };
        const shape = mlx.getShape(x);
        var wrong = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wrong);
        try mlx.check(mlx.mlx_zeros(&wrong, shape.ptr, shape.len, .int32, t.s));
        tensor.array = .{ .ctx = wrong.ctx };
        try expectInvalidImport(t, &e, ids, e.arrays());
        tensor.* = original;
        var dims: [2]c_int = .{ 1, 1 };
        @memcpy(dims[0..shape.len], shape);
        dims[0] += 1;
        try mlx.check(mlx.mlx_zeros(&wrong, &dims, 2, mlx.mlx_array_dtype(x), t.s));
        tensor.array = .{ .ctx = wrong.ctx };
        try expectInvalidImport(t, &e, ids, e.arrays());
        tensor.* = original;
        try mlx.check(mlx.mlx_zeros(&wrong, &.{1}, 1, mlx.mlx_array_dtype(x), t.s));
        tensor.array = .{ .ctx = wrong.ctx };
        try expectInvalidImport(t, &e, ids, e.arrays());
        tensor.* = original;
    }
    try expectInvalidImport(t, &e, ids[1..], e.arrays());
    try testing.expectEqual(@as(u64, 0), v41.stateBytes(t.m));
}

test "codec import unwinds every allocation and checked MLX failure" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41DsparkFiles(arena.allocator(), 7), "");
    defer t.close();
    const m: *v41.Model = @ptrCast(@alignCast(t.m));
    const p = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 60 };
    var e: Exported = .{};
    try e.capture(t, &p);
    defer e.deinit();
    const ids = p[0 .. p.len - 1];
    for (e.arrays()) |x| try mlx.check(mlx.mlx_array_eval(.{ .ctx = x.array.ctx }));
    try mlx.check(mlx.mlx_synchronize(t.s));
    var before: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&before));
    for (0..8) |at| {
        var failing = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = at });
        const sn = dsv41.importSnapshot(m.mdl, failing.allocator(), ids, e.bytes().slice(), e.arrays()) catch |err| {
            try testing.expectEqual(error.OutOfMemory, err);
            try testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
            continue;
        };
        sn.deinit(failing.allocator());
        failing.allocator().destroy(sn);
        try testing.expect(!failing.has_induced_failure);
    }
    var completed = false;
    for (1..512) |at| {
        mlx.fault.arm(at);
        var sn: abi.Handle = null;
        const v = v41.table.cache.aux.?.import_arrays(t.m, .of(ids), e.bytes(), .of(e.arrays()), &sn);
        const fired = mlx.fault.didFire();
        mlx.fault.disarm();
        defer v41.table.cache.aux.?.free(t.m, sn);
        if (!fired) {
            try testing.expect(v.isOk() and sn != null);
            completed = true;
            break;
        }
        try testing.expectEqual(abi.Verdict.Code.failed, v.code);
        try testing.expect(sn == null);
    }
    try testing.expect(completed);
    try mlx.check(mlx.mlx_synchronize(t.s));
    var after: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&after));
    try testing.expectEqual(before, after);
    try testing.expectEqual(@as(u64, 0), v41.stateBytes(t.m));
}

test "optional codec failure preserves the original MLX error latch" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), "");
    defer t.close();
    const p = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 60 };
    var e: Exported = .{};
    try e.capture(t, &p);
    defer e.deinit();
    const original = "[matmul] original codec-test failure";
    mlx.latchErrorForTest(original);
    defer {
        var buf: [512]u8 = undefined;
        _ = mlx.takeError(&buf);
    }
    mlx.fault.arm(3);
    defer mlx.fault.disarm();
    try expectImportRefused(t, &e, p[0 .. p.len - 1], e.arrays());
    var buf: [512]u8 = undefined;
    try testing.expectEqualStrings(original, mlx.takeError(&buf).?);
    mlx.fault.disarm();
    mlx.armLatchingFaultForTest(3);
    try expectImportRefused(t, &e, p[0 .. p.len - 1], e.arrays());
    try testing.expect(mlx.latchingFaultFiredForTest());
    try testing.expect(!mlx.errorPending());
}

fn codecRoundTrip(files: []const sdk.fixture.File, p: []const u32, dspark: bool) !void {
    const aux = v41.table.cache.aux.?;
    var e: Exported = .{};
    defer e.deinit();
    {
        const origin = try Twin.open(files, "");
        defer origin.close();
        try e.capture(origin, p);
    }
    try testing.expectEqualStrings("DV41A002", e.bytes().slice()[0..8]);
    for (e.arrays()) |tensor| {
        const name = tensor.name.slice();
        const candidate = std.mem.endsWith(u8, name, ".comp") or std.mem.endsWith(u8, name, ".index") or std.mem.endsWith(u8, name, ".win");
        if (candidate) try testing.expectEqual(mlx.mlx_dtype.uint8, mlx.mlx_array_dtype(.{ .ctx = tensor.array.ctx }));
    }
    const t = try Twin.open(files, "");
    defer t.close();
    const m: *v41.Model = @ptrCast(@alignCast(t.m));
    try testing.expectEqual(@as(usize, if (dspark) 2 else 0), m.mdl.n_mtp);
    var peak: u64 = 0;
    try testing.expect(aux.import_plan(t.m, e.bytes(), &peak).isOk());
    try testing.expect(peak > 0);
    std.mem.reverse(abi.AuxTensor, e.arrays());
    var sn: abi.Handle = null;
    const input_ids = try testing.allocator.dupe(u32, p[0 .. p.len - 1]);
    defer testing.allocator.free(input_ids);
    try testing.expect(aux.import_arrays(t.m, .of(input_ids), e.bytes(), .of(e.arrays()), &sn).isOk());
    defer aux.free(t.m, sn);
    e.deinit();
    @memset(input_ids, 0);
    @memset(&e.header, 0);
    @memset(&e.names, 0);
    var info: abi.AuxSnapshotInfo = .{};
    try testing.expect(aux.describe(t.m, sn, &info).isOk());
    try testing.expectEqual(p.len - 1, info.n);
    try testing.expectEqualSlices(u32, p[0 .. p.len - 1], info.ids.slice());
    try testing.expect(peak > aux.bytes(t.m, sn));
    try testing.expect(aux.restore_at(t.m, 1, sn, info.n).isOk());
    var caps: [64][2]usize = @splat(.{ 0, 0 });
    for (m.mdl.dec_state.?.layers, 0..) |l, i| caps[i] = .{ if (l.comp) |r| r.cap else 0, if (l.idx_k) |r| r.cap else 0 };
    try testing.expect(aux.begin_prompt(t.m, 1, .of(p), .{}, info.n, false).isOk());
    const warm = try slotForward(t, .prefill, &.{1}, p[info.n..], &.{@intCast(info.n)});
    defer testing.allocator.free(warm);
    for (m.mdl.dec_state.?.layers, 0..) |l, i| {
        if (l.comp) |r| try testing.expectEqual(caps[i][0], r.cap);
        if (l.idx_k) |r| try testing.expectEqual(caps[i][1], r.cap);
    }
    const reference = try Twin.open(files, "");
    defer reference.close();
    const first = @min(2, p.len - 1);
    testing.allocator.free(try slotForward(reference, .prefill, &.{1}, p[0..first], &.{0}));
    if (first < p.len - 1) testing.allocator.free(try slotForward(reference, .prefill, &.{1}, p[first .. p.len - 1], &.{@intCast(first)}));
    const cold = try slotForward(reference, .prefill, &.{1}, p[p.len - 1 ..], &.{@intCast(p.len - 1)});
    defer testing.allocator.free(cold);
    for (cold) |x| try testing.expect(std.math.isFinite(x));
    try testing.expectEqualSlices(f32, cold, warm);
}

test "codec round trip survives origin destruction at odd and even compressor boundaries with DSpark on and off" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old = env.get(.DSV41_CACHE_PACK);
    defer if (old) |value| {
        _ = env.setForeign("DSV41_CACHE_PACK", value);
    } else {
        env.unsetForeign("DSV41_CACHE_PACK");
    };
    env.set(.DSV41_CACHE_PACK, "1");
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var p: [27]u32 = undefined;
    for (&p, 0..) |*id, i| id.* = @intCast((i * 13 + 3) % 96);
    for ([_]bool{ false, true }) |dspark| {
        const files = if (dspark) try fixture.v41DsparkFiles(arena.allocator(), 7) else try fixture.v41Files(arena.allocator(), 7);
        for ([_]usize{ 2, 3, 8, 9, 26, 27 }) |len| try codecRoundTrip(files, p[0..len], dspark);
    }
}

test "packed snapshot restore refuses the zero scale corruption sentinel" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41Files(arena.allocator(), 7);
    const origin = try Twin.open(files, "");
    defer origin.close();
    const p = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 60 };
    var e: Exported = .{};
    try e.capture(origin, &p);
    defer e.deinit();
    try testing.expectEqualStrings("DV41A002", e.bytes().slice()[0..8]);
    for (e.arrays()) |tensor| if (mlx.mlx_array_dtype(.{ .ctx = tensor.array.ctx }) == .uint8) {
        var ready = true;
        try mlx.check(mlx._mlx_array_is_available(&ready, .{ .ctx = tensor.array.ctx }));
        try testing.expect(!ready);
    };
    var packed_index: ?usize = null;
    for (e.arrays(), 0..) |tensor, i| {
        if (mlx.mlx_array_dtype(.{ .ctx = tensor.array.ctx }) == .uint8 and std.mem.indexOf(u8, tensor.name.slice(), ".comp") != null) {
            packed_index = i;
            break;
        }
    }
    const i = packed_index orelse return error.MissingPackedComp;
    const original = mlx.mlx_array{ .ctx = e.arrays()[i].array.ctx };
    try mlx.check(mlx.mlx_array_eval(original));
    const shape = mlx.getShape(original);
    const size = mlx.mlx_array_size(original);
    const changed = try testing.allocator.dupe(u8, mlx.mlx_array_data_uint8(original).?[0..size]);
    defer testing.allocator.free(changed);
    changed[@intCast(shape[1] - 1)] = 0;
    const damaged = mlx.mlx_array_new_data(changed.ptr, shape.ptr, @intCast(shape.len), .uint8);
    _ = mlx.mlx_array_free(original);
    e.arrays()[i].array.ctx = damaged.ctx;
    const receiver = try Twin.open(files, "");
    defer receiver.close();
    try expectImportRefused(receiver, &e, p[0 .. p.len - 1], e.arrays());
}

test "cache packing kill lever keeps the v1 header and raw tensor inventory" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const old = env.get(.DSV41_CACHE_PACK);
    defer if (old) |value| {
        _ = env.setForeign("DSV41_CACHE_PACK", value);
    } else {
        env.unsetForeign("DSV41_CACHE_PACK");
    };
    env.set(.DSV41_CACHE_PACK, "0");
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), "");
    defer t.close();
    const p = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 60 };
    var e: Exported = .{};
    try e.capture(t, &p);
    defer e.deinit();
    try testing.expectEqualStrings("DV41A001", e.bytes().slice()[0..8]);
    for (e.arrays()) |tensor| try testing.expect(mlx.mlx_array_dtype(.{ .ctx = tensor.array.ctx }) != .uint8);
    const receiver = try Twin.open(try fixture.v41Files(arena.allocator(), 7), "");
    defer receiver.close();
    var imported: abi.Handle = null;
    try testing.expect(v41.table.cache.aux.?.import_arrays(receiver.m, .of(p[0 .. p.len - 1]), e.bytes(), .of(e.arrays()), &imported).isOk());
    defer v41.table.cache.aux.?.free(receiver.m, imported);

    env.set(.DSV41_CACHE_PACK, "1");
    const qat_off = try Twin.open(try fixture.v41Files(arena.allocator(), 7), "");
    defer qat_off.close();
    const model: *v41.Model = @ptrCast(@alignCast(qat_off.m));
    model.mdl.qat_off = true;
    var raw_v2: Exported = .{};
    try raw_v2.capture(qat_off, &p);
    defer raw_v2.deinit();
    try testing.expectEqualStrings("DV41A002", raw_v2.bytes().slice()[0..8]);
    for (raw_v2.arrays()) |tensor| try testing.expect(mlx.mlx_array_dtype(.{ .ctx = tensor.array.ctx }) != .uint8);
}

test "codec refuses changed loaded settings and invalid IDs before allocating" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41DsparkFiles(arena.allocator(), 7), "");
    defer t.close();
    const m: *v41.Model = @ptrCast(@alignCast(t.m));
    const p = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 60 };
    var e: Exported = .{};
    try e.capture(t, &p);
    defer e.deinit();
    const ids = p[0 .. p.len - 1];
    // A sender loaded with another config differs in both: the model's own cfg and the whole config the cache key hashes.
    const cfg = m.mdl.cfg;
    const whole = m.mdl.whole_cfg;
    defer {
        m.mdl.cfg = cfg;
        m.mdl.whole_cfg = whole;
    }
    for (0..5) |change| {
        inline for (.{ &m.mdl.cfg, &m.mdl.whole_cfg }) |c| switch (change) {
            0 => c.sliding_window += 1,
            1 => c.dsv4_dspark_target_layers[0] += 1,
            2 => c.quant_bits += 1,
            3 => c.dsv41_engram_compressed_vocab += 1,
            4 => c.dsv4_compress_ratios[2] += 1,
            else => unreachable,
        };
        var peak: u64 = 123;
        const verdict = v41.table.cache.aux.?.import_plan(t.m, e.bytes(), &peak);
        try testing.expectEqual(abi.Verdict.Code.failed, verdict.code);
        try testing.expectEqual(@as(u64, 0), peak);
        try expectInvalidImport(t, &e, ids, e.arrays());
        m.mdl.cfg = cfg;
        m.mdl.whole_cfg = whole;
    }
    const stages = m.mdl.n_mtp;
    m.mdl.n_mtp = 0;
    try expectInvalidImport(t, &e, ids, e.arrays());
    m.mdl.n_mtp = stages;
    var invalid = p;
    invalid[0] = @intCast(m.mdl.vocab);
    try expectInvalidImport(t, &e, invalid[0..ids.len], e.arrays());
    try expectInvalidImport(t, &e, ids[1..], e.arrays());
}

test "codec export refuses short buffers and unwinds partially created references" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), "");
    defer t.close();
    const aux = v41.table.cache.aux.?;
    const p = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 60 };
    try testing.expect(aux.begin_prompt(t.m, 1, .of(&p), .{}, 0, true).isOk());
    testing.allocator.free(try slotForward(t, .prefill, &.{1}, &p, &.{0}));
    var sn: abi.Handle = null;
    try testing.expect(aux.snapshot(t.m, 1, &sn).isOk());
    defer aux.free(t.m, sn);
    var e: Exported = .{};
    const storage: abi.AuxExport = .{ .tensors = &e.tensors, .tensor_cap = e.tensors.len, .names = &e.names, .names_cap = e.names.len, .header = &e.header, .header_cap = e.header.len };
    for (0..3) |small| {
        var out = storage;
        switch (small) {
            0 => out.tensor_cap = 1,
            1 => out.names_cap = 1,
            2 => out.header_cap = 1,
            else => unreachable,
        }
        try testing.expectEqual(abi.Verdict.Code.failed, aux.export_arrays(t.m, sn, &out).code);
        try testing.expectEqual(@as(usize, 0), out.tensor_len + out.names_len + out.header_len);
    }
    var completed = false;
    for (1..512) |at| {
        var out = storage;
        mlx.fault.arm(at);
        const v = aux.export_arrays(t.m, sn, &out);
        const fired = mlx.fault.didFire();
        mlx.fault.disarm();
        if (!fired) {
            try testing.expect(v.isOk());
            e.out = out;
            e.deinit();
            completed = true;
            break;
        }
        try testing.expectEqual(abi.Verdict.Code.failed, v.code);
        try testing.expectEqual(@as(usize, 0), out.tensor_len + out.names_len + out.header_len);
    }
    try testing.expect(completed);
}

const engram = @import("engram.zig");

test "codec preserves compressed Engram history including dead sentinels through export, import and restore" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const t = try Twin.open(try fixture.v41Files(arena.allocator(), 7), "");
    defer t.close();
    const pm: *v41.Model = @ptrCast(@alignCast(t.m));
    const m = pm.mdl;
    const ids = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 60 };
    try dsv41.reset(m, gpa);
    try dsv41.armPrompt(m, &ids, null, true, null);
    _ = mlx.mlx_array_free(try dsv41.extendResumable(m, gpa, &ids, null));
    const sn = &m.dec_state.?.snap.?;
    const history = [_]u32{ 9, engram.Hash.DEAD, 2, 4, engram.Hash.DEAD, 7, 1 };
    // The tiny pack has no Engram projections; supply its already-compressed history at the codec boundary.
    m.hash = .{ .n_layers = 0, .max_ngram = 2, .heads = 1, .pad = 0 };
    sn.hist = try gpa.dupe(u32, &history);
    var tensors: [codec.max_tensors]abi.AuxTensor = undefined;
    var names: [16 * 1024]u8 = undefined;
    var header: [codec.max_header]u8 = undefined;
    var out: abi.AuxExport = .{ .tensors = &tensors, .tensor_cap = tensors.len, .names = &names, .names_cap = names.len, .header = &header, .header_cap = header.len };
    try dsv41.exportSnapshot(m, sn, &out);
    defer for (tensors[0..out.tensor_len]) |x| {
        _ = mlx.mlx_array_free(.{ .ctx = x.array.ctx });
    };
    m.dec_state.?.deinit();
    m.dec_state = null;
    try testing.checkAllAllocationFailures(gpa, importForAllocationTest, .{ m, ids[0..history.len], header[0..out.header_len], tensors[0..out.tensor_len] });
    const imported = try dsv41.importSnapshot(m, gpa, ids[0..history.len], header[0..out.header_len], tensors[0..out.tensor_len]);
    defer {
        imported.deinit(gpa);
        gpa.destroy(imported);
    }
    try testing.expectEqualSlices(u32, &history, imported.hist);
    var restored = try dsv41.stateFromSnapshot(m, gpa, imported);
    defer restored.deinit();
    try testing.expectEqualSlices(u32, &history, restored.hist.items);
}

fn importForAllocationTest(gpa: std.mem.Allocator, m: *const dsv41.Dsv41Model, ids: []const u32, header: []const u8, arrays: []const abi.AuxTensor) !void {
    const sn = try dsv41.importSnapshot(m, gpa, ids, header, arrays);
    defer gpa.destroy(sn);
    defer sn.deinit(gpa);
}

const gemv_twin = @import("decode/gemv_twin.zig");

const LoneGroup = v41.mode.LoneGroup;

/// One step's logits for `slot` (host f32, caller frees): all rows for a verify, else the last.
fn stepLogits(t: *Twin, slot: u32, st: proof.Step, pos: u32) ![]f32 {
    var out: abi.ForwardOut = .{};
    const v = v41.table.forward(t.m, &.{ .mode = st.mode, .logits = if (st.mode == .verify) .all_rows else .last_row, .rows = @intCast(st.ids.len), .tokens = .of(st.ids), .slots = .of(&.{slot}), .positions = .of(&.{pos}) }, &out);
    if (!v.isOk()) {
        std.debug.print("forward: {s}\n", .{v.why.slice()});
        return error.PluginForward;
    }
    const l: mlx.mlx_array = .{ .ctx = out.logits.ctx };
    defer _ = mlx.mlx_array_free(l);
    return hostRows(l);
}

/// Today's load (no `tp_mode`): every proof step's logits on slot 1.
fn wholeSteps(files: []const sdk.fixture.File) ![proof.steps.len][]f32 {
    const t = try Twin.open(files, "");
    defer t.close();
    var out: [proof.steps.len][]f32 = undefined;
    var pos: u32 = 0;
    for (proof.steps, &out) |st, *o| {
        o.* = try stepLogits(t, 1, st, pos);
        pos += @intCast(st.ids.len);
    }
    return out;
}

const tp4_setting = "{\"tp_mode\": {\"rank\": 0, \"size\": 4}}";

fn settledLoadBytes() !usize {
    try mlx.check(mlx.mlx_synchronize(mlx.gpuStream()));
    try mlx.check(mlx.mlx_clear_cache());
    var bytes: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&bytes));
    return bytes;
}

test "deepseek_v41 modes: TP4 beside whole allocates only wo_b quarters and small mode constants" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41TpDsparkFiles(arena.allocator(), 7);
    var whole_bytes: usize = 0;
    var wo_b_bytes: usize = 0;
    {
        const before = try settledLoadBytes();
        const t = try Twin.open(files, "");
        defer t.close();
        whole_bytes = (try settledLoadBytes()) - before;
        const m: *v41.Model = @ptrCast(@alignCast(t.m));
        for (m.mdl.layers) |ly| inline for (.{ "w", "s", "b" }) |f| {
            const a = @field(ly.wo_b, f);
            if (a.ctx != null) wo_b_bytes += mlx.mlx_array_size(a) * mlx.mlx_array_itemsize(a) / 4;
        };
    }
    const before = try settledLoadBytes();
    const t = try Twin.openWith(files, "", tp4_setting);
    defer t.close();
    const both_bytes = (try settledLoadBytes()) - before;
    const extra = both_bytes -| whole_bytes;
    const allowance = wo_b_bytes + 64 * 1024;
    std.debug.print("[mode-memory] whole {d} B, both {d} B, extra {d} B, wo_b {d} B, allowance {d} B\n", .{ whole_bytes, both_bytes, extra, wo_b_bytes, allowance });
    try testing.expect(extra <= allowance);
}

test "deepseek_v41 modes: the whole model beside a TP4 rank's model gives today's logits bit for bit, through TP4 forwards, swaps and the TP4 model's state" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41TpFiles(arena.allocator(), 7);
    const want = try wholeSteps(files);
    defer for (want) |w| testing.allocator.free(w);

    LoneGroup.arm(4);
    defer LoneGroup.disarm();
    const t = try Twin.openWith(files, "", tp4_setting);
    defer t.close();
    const m: *v41.Model = @ptrCast(@alignCast(t.m));
    const tp = m.modes.tp orelse return error.NoTpMode;
    try testing.expectEqual(m.modes.tp1, m.mdl);
    try testing.expectEqual(@as(usize, 4), m.modes.tp1.cfg.num_attention_heads / tp.cfg.num_attention_heads);
    try testing.expectEqual(m.modes.tp1.cfg.moe_intermediate_size, 4 * tp.cfg.moe_intermediate_size);

    // Each whole-model step follows a TP4 prompt on the TP4 model's own state.
    var pos: u32 = 0;
    for (proof.steps, want) |st, w| {
        _ = mlx.mlx_array_free(try dsv41.prefill(tp, testing.allocator, &.{ 3, 1, 4, 1, 5, 9, 2, 6 }, null));
        const got = try stepLogits(t, 1, st, pos);
        defer testing.allocator.free(got);
        try testing.expectEqualSlices(f32, w, got);
        pos += @intCast(st.ids.len);
    }
    try testing.expect(LoneGroup.sums > 0);

    // TP4 serves a turn, then TP1 serves the same steps from a fresh state, bit for bit.
    try v41.swapMode(m, .tp);
    try testing.expectEqual(tp, m.mdl);
    testing.allocator.free(try stepLogits(t, 1, proof.steps[0], 0));
    try v41.swapMode(m, .tp1);
    pos = 0;
    for (proof.steps, want) |st, w| {
        const got = try stepLogits(t, 1, st, pos);
        defer testing.allocator.free(got);
        try testing.expectEqualSlices(f32, w, got);
        pos += @intCast(st.ids.len);
    }
}

test "deepseek_v41 modes: a swap drops every slot's state, live and parked, so a slot's next turn cannot continue it" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    LoneGroup.arm(4);
    defer LoneGroup.disarm();
    const t = try Twin.openWith(try fixture.v41TpFiles(arena.allocator(), 7), "", tp4_setting);
    defer t.close();
    const m: *v41.Model = @ptrCast(@alignCast(t.m));
    const a = [_]u32{ 3, 1, 4, 1, 5 };
    const b = [_]u32{ 2, 7, 1, 8 };
    for ([_]v41.mode.Mode{ .tp1, .tp, .tp1 }) |to| {
        try v41.swapMode(m, to);
        testing.allocator.free(try slotForward(t, .prefill, &.{1}, &a, &.{0}));
        testing.allocator.free(try slotForward(t, .prefill, &.{2}, &b, &.{0}));
        try testing.expectEqual(@as(u32, 1), m.parked.count());
        const from = m.mdl;
        try v41.swapMode(m, if (to == .tp1) .tp else .tp1);
        try testing.expectEqual(@as(u32, 0), m.parked.count());
        try testing.expectEqual(@as(u32, 0), m.owner);
        try testing.expect(from.dec_state == null and m.mdl.dec_state == null);
        // Slot 2's next token has no state to follow on either model.
        try testing.expectError(error.PluginForward, slotForward(t, .decode, &.{2}, &.{6}, &.{@intCast(b.len)}));
    }
}

test "deepseek_v41 modes: on the wide prompt path the whole model gives today's logits with a TP4 model's wide prompts between its steps" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    try wideBeside();
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "deepseek_v41 modes: the wide arm again with the flash kernel on, a setting both modes read from one load" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    _ = setenv("MLX_SERVE_DSV41_FLASH_ATTN", "", 1);
    defer _ = unsetenv("MLX_SERVE_DSV41_FLASH_ATTN");
    try wideBeside();
}

/// The whole model's wide steps beside a TP4 model's wide prompts, against today's load on the wide path.
fn wideBeside() !void {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41TpFiles(arena.allocator(), 7);
    var want: [proof.steps.len][]f32 = undefined;
    {
        const wide = try WideTwin.open(files);
        defer wide.close();
        var pos: u32 = 0;
        for (proof.steps, &want) |st, *w| {
            w.* = try stepLogits(wide.t, 1, st, pos);
            pos += @intCast(st.ids.len);
        }
    }
    defer for (want) |w| testing.allocator.free(w);

    LoneGroup.arm(4);
    defer LoneGroup.disarm();
    const t = try Twin.openWith(files, "", tp4_setting);
    defer t.close();
    const m: *v41.Model = @ptrCast(@alignCast(t.m));
    const tp = m.modes.tp orelse return error.NoTpMode;
    // Both models off their compiled regions: every call runs the wide layer and its process-wide scratch rings.
    const kept = .{ m.modes.tp1.regions, tp.regions };
    m.modes.tp1.regions = kept[0][0..0];
    tp.regions = kept[1][0..0];
    defer {
        m.modes.tp1.regions = kept[0];
        tp.regions = kept[1];
    }
    var pos: u32 = 0;
    for (proof.steps, want) |st, w| {
        _ = mlx.mlx_array_free(try dsv41.prefill(tp, testing.allocator, &.{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5 }, null));
        const got = try stepLogits(t, 1, st, pos);
        defer testing.allocator.free(got);
        try testing.expectEqualSlices(f32, w, got);
        pos += @intCast(st.ids.len);
    }
}

test "deepseek_v41 modes: freeing the TP4 model leaves the whole model's launch table and its logits as they were" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41TpFiles(arena.allocator(), 7);
    const want = try wholeSteps(files);
    defer for (want) |w| testing.allocator.free(w);
    LoneGroup.arm(4);
    defer LoneGroup.disarm();
    const t = try Twin.openWith(files, "", tp4_setting);
    defer t.close();
    const m: *v41.Model = @ptrCast(@alignCast(t.m));
    _ = mlx.mlx_array_free(try dsv41.prefill(m.modes.tp.?, testing.allocator, &.{ 3, 1, 4, 1, 5, 9, 2, 6 }, null));
    const held = gemv_twin.shapesHeld();
    try testing.expect(held > 0);
    m.modes.tp.?.deinit();
    m.modes.tp = null;
    try testing.expectEqual(held, gemv_twin.shapesHeld());
    var pos: u32 = 0;
    for (proof.steps, want) |st, w| {
        const got = try stepLogits(t, 1, st, pos);
        defer testing.allocator.free(got);
        try testing.expectEqualSlices(f32, w, got);
        pos += @intCast(st.ids.len);
    }
}

test "deepseek_v41 modes: a DSpark pack's TP4 rank holds its stages as a --dist tensor rank does, sharded or on rank 0 whole" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41TpDsparkFiles(arena.allocator(), 7);
    LoneGroup.arm(4);
    defer LoneGroup.disarm();
    defer _ = unsetenv("DSV41_DRAFT_SHARD");
    for ([_]bool{ true, false }) |shard| {
        _ = setenv("DSV41_DRAFT_SHARD", if (shard) "1" else "0", 1);
        const t = try Twin.openWith(files, "", tp4_setting);
        defer t.close();
        const m: *v41.Model = @ptrCast(@alignCast(t.m));
        const tp = m.modes.tp orelse return error.NoTpMode;
        try testing.expectEqual(@as(usize, 2), m.modes.tp1.n_mtp);
        try testing.expectEqual(@as(usize, 2), tp.n_mtp);
        try testing.expectEqual(shard, tp.draft_shard);
        _ = mlx.mlx_array_free(try dsv41.prefill(tp, testing.allocator, &.{ 3, 1, 4, 1, 5, 9, 2, 6 }, null));
    }
}

test "deepseek_v41 modes: tp_mode is refused by name at a size below 2 or a rank past it" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41TpFiles(arena.allocator(), 7);
    for ([_][]const u8{ "{\"tp_mode\": {\"rank\": 0, \"size\": 1}}", "{\"tp_mode\": {\"rank\": 4, \"size\": 4}}", "{\"tp_mode\": {\"rank\": 0, \"size\": 3}}" }) |bad| {
        try testing.expectError(error.PluginLoad, Twin.openWith(files, "", bad));
    }
}

const TpProofGroup = @import("decode/fixture.zig").Group;
const tp_prompt = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 60, 3, 17, 42, 9, 7, 25, 1, 60, 3, 17, 42, 9, 7, 25, 1, 60, 3, 17, 42, 9, 7, 25, 1, 60 };

const TpTurn = struct {
    prefill: []f32,
    decode: []f32,
    draft: []f32 = &.{},

    fn deinit(self: TpTurn) void {
        testing.allocator.free(self.prefill);
        testing.allocator.free(self.decode);
        testing.allocator.free(self.draft);
    }
};

fn tpTurn(m: *dsv41.Dsv41Model) !TpTurn {
    const p = try dsv41.prefill(m, testing.allocator, &tp_prompt, null);
    defer _ = mlx.mlx_array_free(p);
    const prefill = try dsv41.hostF32(p);
    errdefer testing.allocator.free(prefill);
    if (m.row_cut) |rc| try testing.expectEqual(@as(usize, 0), rc.calls);
    const d = try dsv41.extendResumable(m, testing.allocator, &.{8}, null);
    defer _ = mlx.mlx_array_free(d);
    const decoded = try dsv41.hostF32(d);
    errdefer testing.allocator.free(decoded);
    if (m.row_cut) |rc| try testing.expect(rc.calls > 0);
    var out: TpTurn = .{ .prefill = prefill, .decode = decoded };
    if (m.n_mtp > 0) {
        var sc = dsv41.Scope.init(testing.allocator, m.s);
        defer sc.deinit();
        const g = try dsv41.draftGraph(&sc, m, testing.allocator, &.{&m.dec_state.?}, &.{5}, true);
        out.draft = try dsv41.hostF32(g.logits);
    }
    return out;
}

const tp_logit_tol: f32 = 2.0 / 128.0;
const tp_quality_issue = "https://github.com/STRML/mlx-serve/issues/30";

fn tpCheckLogits(label: []const u8, want: []const f32, got: []const f32, tol: f32, required: bool) !bool {
    if (required) return tpLogitsMatch(label, want, got, tol);
    std.debug.print("[tp-logits] diagnostic {s}: {s}\n", .{ label, tp_quality_issue });
    _ = tpLogitsMatch(label, want, got, tol) catch |err| {
        std.debug.print("[tp-logits] diagnostic {s}: {t}\n", .{ label, err });
        return true;
    };
    return true;
}

fn tpLogitsMatch(label: []const u8, want: []const f32, got: []const f32, tol: f32) !bool {
    try testing.expectEqual(want.len, got.len);
    var peak: f64 = 0;
    var max_abs: f64 = 0;
    var max_rel: f64 = 0;
    var squared_error: f64 = 0;
    var squared_ref: f64 = 0;
    for (want, got) |u, v| {
        if (!std.math.isFinite(u) or !std.math.isFinite(v)) return error.NonFinite;
        const a: f64 = u;
        const e = @abs(a - @as(f64, v));
        peak = @max(peak, @abs(a));
        max_abs = @max(max_abs, e);
        max_rel = @max(max_rel, e / @max(@abs(a), 1e-6));
        squared_error += e * e;
        squared_ref += a * a;
    }
    if (peak == 0 or squared_ref == 0) return error.EmptyReference;
    const rms = @sqrt(squared_error / squared_ref);
    std.debug.print("[tp-logits] {s}: n={d} max-abs={e:.8} max-rel={e:.8} peak-rel={e:.8} rel-rms={e:.8} bar={e:.8}\n", .{ label, want.len, max_abs, max_rel, max_abs / peak, rms, tol });
    return max_abs <= tol * peak and rms <= tol;
}

fn settledTpTurn(ts: [4]*Twin) ![4]TpTurn {
    LoneGroup.arm(4);
    defer LoneGroup.disarm();
    TpProofGroup.join(4);
    defer TpProofGroup.leave();
    var out: [4]?TpTurn = @splat(null);
    errdefer for (out) |o| if (o) |v| v.deinit();
    const lead: *v41.Model = @ptrCast(@alignCast(ts[0].m));
    const bound = 4 * lead.mdl.n_layers + 4 * lead.mdl.n_mtp + lead.mdl.ds_block + 4;
    for (0..bound) |pass| {
        for (ts, 0..) |t, r| {
            TpProofGroup.begin(r);
            const model: *v41.Model = @ptrCast(@alignCast(t.m));
            const tp = model.modes.tp.?;
            model.modes.row_cut.?.calls = 0;
            if (out[r]) |o| o.deinit();
            out[r] = null;
            // Keep the conventional tensor proof independent of the leader-prefill route.
            const whole = tp.prefill_whole;
            tp.prefill_whole = null;
            defer tp.prefill_whole = whole;
            out[r] = try tpTurn(tp);
        }
        if (try TpProofGroup.settled()) {
            try TpProofGroup.expectAgreement();
            std.debug.print("[tp-logits] settled after {d} passes\n", .{pass + 1});
            return .{ out[0].?, out[1].?, out[2].?, out[3].? };
        }
    }
    return error.CollectivesDidNotSettle;
}

fn tpProof(act: mlx.mlx_dtype) !void {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const tol: f32 = if (act == .float32) 1e-3 else tp_logit_tol;
    var parity = true;
    for ([_]bool{ false, true }) |stages| {
        const files = if (stages) try fixture.v41TpDsparkFiles(arena.allocator(), 7) else try fixture.v41TpFiles(arena.allocator(), 7);
        var ts: [4]*Twin = undefined;
        var made: usize = 0;
        defer for (ts[0..made]) |t| t.close();
        for (&ts, 0..) |*t, r| {
            const setting = try std.fmt.allocPrint(arena.allocator(), "{{\"tp_mode\":{{\"rank\":{d},\"size\":4}}}}", .{r});
            t.* = try Twin.openWith(files, "", setting);
            made += 1;
            if (act == .float32) try rebuildTpProofF32(t.*, @intCast(r));
        }
        std.debug.print("[tp-logits] act={s} stages={any}\n", .{ @tagName(act), stages });
        const model: *v41.Model = @ptrCast(@alignCast(ts[0].m));
        try testing.expectEqual(@as(usize, if (stages) 2 else 0), model.mdl.n_mtp);
        const want = try tpTurn(model.mdl);
        defer want.deinit();
        const got = try settledTpTurn(ts);
        defer for (got) |o| o.deinit();
        // The DSpark fixture's head has only two nonzero vocabulary rows; its trunk ratios are diagnostic.
        const require_trunk = act == .float32 and !stages;
        parity = try tpCheckLogits(if (stages) "DSpark trunk prefill" else "wide prefill", want.prefill, got[0].prefill, tol, require_trunk) and parity;
        parity = try tpCheckLogits(if (stages) "DSpark trunk decode" else "row-cut decode", want.decode, got[0].decode, tol, require_trunk) and parity;
        if (stages) {
            const joined = try testing.allocator.alloc(f32, want.draft.len);
            defer testing.allocator.free(joined);
            const width = model.mdl.vocab / 4;
            for (got, 0..) |o, r| {
                try testing.expectEqual(model.mdl.ds_block * width, o.draft.len);
                for (0..model.mdl.ds_block) |b| @memcpy(joined[b * model.mdl.vocab + r * width ..][0..width], o.draft[b * width ..][0..width]);
            }
            parity = try tpCheckLogits("DSpark stages + Markov logits", want.draft, joined, tol, act == .float32) and parity;
            continue;
        }
        // Swap the borrowed halves on every tensor layer without changing TP1 or its banks.
        for (ts) |t| {
            const pm: *v41.Model = @ptrCast(@alignCast(t.m));
            for (pm.modes.tp.?.layers) |*ly| std.mem.swap(dsv41.Bank, &ly.experts.gate, &ly.experts.up);
        }
        defer for (ts) |t| {
            const pm: *v41.Model = @ptrCast(@alignCast(t.m));
            for (pm.modes.tp.?.layers) |*ly| std.mem.swap(dsv41.Bank, &ly.experts.gate, &ly.experts.up);
        };
        const bad = try settledTpTurn(ts);
        defer for (bad) |o| o.deinit();
        try testing.expect(!try tpLogitsMatch("negative swapped gate/up vs intact TP4", got[0].prefill, bad[0].prefill, tol));
        std.debug.print("[tp-logits] negative control rejected (act={s})\n", .{@tagName(act)});
    }
    try testing.expect(parity);
}

test "deepseek_v41 TP logit comparator rejects NaN and infinity" {
    const want = [_]f32{ 1, -2, 3, 0.5 };
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32) }) |bad|
        try testing.expectError(error.NonFinite, tpLogitsMatch("nonfinite", &want, &.{ 1, -2, bad, 0.5 }, tp_logit_tol));
    try testing.expect(try tpLogitsMatch("equal", &want, &want, tp_logit_tol));
}

fn rebuildTpProofF32(t: *Twin, rank: u32) !void {
    const m: *v41.Model = @ptrCast(@alignCast(t.m));
    try testing.expectEqual(@as(u32, 0), m.owner);
    try testing.expectEqual(@as(u32, 0), m.parked.count());
    const io = std.Io.Threaded.global_single_threaded.io();
    var buf: [512]u8 = undefined;
    const dir = buf[0..try t.tmp.dir.realPath(io, &buf)];
    var weights = try dsv41.loadPack(io, testing.allocator, dir, false);
    errdefer weights.deinit();
    const modes = try v41.mode.build(testing.allocator, &m.cfg, &weights, m.s, .{ .act = .float32, .dspark = true, .head_8bit = dsv41.head8Wanted(), .pack_dir = dir }, .{ .rank = rank, .size = 4 });
    m.modes.deinit();
    m.weights.deinit();
    m.weights = weights;
    m.modes = modes;
    m.mdl = modes.tp1;
}

test "deepseek_v41 modes: TP4 logits match TP1 through the one-process group (bf16 diagnostics)" {
    try tpProof(.bfloat16);
}

extern "c" fn msv_test_swap_tf32(enabled: c_int) c_int;

test "deepseek_v41 modes: TP4 logits match TP1 through the one-process group (f32)" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    try mlx.check(mlx.mlx_synchronize(mlx.gpuStream()));
    // MLX caches the environment; its config API updates the value used by the backend's worker threads.
    const previous = msv_test_swap_tf32(0);
    defer {
        _ = mlx.mlx_synchronize(mlx.gpuStream());
        _ = msv_test_swap_tf32(previous);
    }
    try testing.expectEqual(@as(c_int, 0), msv_test_swap_tf32(0));
    std.debug.print("[tp-logits] f32 TF32 disabled via MLX config API (previous={d})\n", .{previous});
    try tpProof(.float32);
}

const LeaderReplay = struct {
    const A = mlx.mlx_array;
    const Chunk = struct { gen: u64, p0: usize, n: usize, capture: bool, xs: []A, stages: usize };
    var chunks: std.ArrayList(Chunk) = .empty;
    var rank: usize = 0;
    var wide_steps: usize = 0;

    fn clear() void {
        for (chunks.items) |c| dsv41.freeArrays(testing.allocator, c.xs);
        chunks.deinit(testing.allocator);
        chunks = .empty;
        wide_steps = 0;
    }

    fn step(ids: []const u32, _: usize, _: bool, _: bool, _: dsv41.VerifyOpts, _: u64, _: ?A) anyerror!void {
        if (ids.len > @import("rowcut.zig").max_rows) wide_steps += 1;
    }

    fn ship(gen: u64, p0: usize, n: usize, capture: bool, xs: []const A, stages: usize) anyerror!void {
        const kept = try testing.allocator.alloc(A, xs.len);
        for (kept, xs) |*o, x| {
            o.* = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(o, x));
        }
        try chunks.append(testing.allocator, .{ .gen = gen, .p0 = p0, .n = n, .capture = capture, .xs = kept, .stages = stages });
    }

    fn install() void {
        dsv41.clusterHooks().tp_step = &step;
        dsv41.clusterHooks().tp_prefill = &ship;
    }

    fn imported(m: *dsv41.Dsv41Model, p0: usize, n: usize) !bool {
        if (rank == 0) return false;
        for (chunks.items) |c| if (c.p0 == p0 and c.n == n) {
            var parked: dsv41.Parked = .init(testing.allocator);
            defer dsv41.dropParked(&parked);
            try dsv41.tpWorkerPrefill(m, testing.allocator, &parked, m.state_gen, p0, n, c.capture, c.xs, c.stages);
            return true;
        };
        return false;
    }
};

fn leaderTurn(m: *dsv41.Dsv41Model, widths: []const usize) !TpTurn {
    try dsv41.reset(m, testing.allocator);
    var p0: usize = 0;
    var prefill: []f32 = &.{};
    errdefer testing.allocator.free(prefill);
    for (widths) |width| {
        testing.allocator.free(prefill);
        prefill = &.{};
        if (m.tp != null and try LeaderReplay.imported(m, p0, p0 + width)) {
            p0 += width;
            continue;
        }
        const p = try dsv41.extendResumable(m, testing.allocator, (tp_prompt ++ tp_prompt)[p0..][0..width], null);
        defer _ = mlx.mlx_array_free(p);
        prefill = try dsv41.hostF32(p);
        p0 += width;
    }
    const d = try dsv41.extendResumable(m, testing.allocator, &.{8}, null);
    defer _ = mlx.mlx_array_free(d);
    const decoded = try dsv41.hostF32(d);
    if (m.hash != null) {
        const prompt = tp_prompt ++ tp_prompt;
        try testing.expectEqualSlices(u32, prompt[0..p0], m.dec_state.?.hist.items[0..p0]);
        try testing.expectEqual(p0 + 1, m.dec_state.?.hist.items.len);
    }
    var out: TpTurn = .{ .prefill = prefill, .decode = decoded };
    errdefer testing.allocator.free(decoded);
    if (m.n_mtp > 0) {
        var sc = dsv41.Scope.init(testing.allocator, m.s);
        defer sc.deinit();
        const g = try dsv41.draftGraph(&sc, m, testing.allocator, &.{&m.dec_state.?}, &.{5}, true);
        out.draft = try dsv41.hostF32(g.logits);
    }
    return out;
}

fn settledLeaderTurn(ts: [4]*Twin, widths: []const usize) ![4]TpTurn {
    LoneGroup.arm(4);
    defer LoneGroup.disarm();
    LeaderReplay.install();
    defer LeaderReplay.clear();
    TpProofGroup.join(4);
    defer TpProofGroup.leave();
    var out: [4]?TpTurn = @splat(null);
    errdefer for (out) |o| if (o) |v| v.deinit();
    for (0..40) |pass| {
        LeaderReplay.clear();
        for (ts, 0..) |t, r| {
            TpProofGroup.begin(r);
            LeaderReplay.rank = r;
            const pm: *v41.Model = @ptrCast(@alignCast(t.m));
            if (out[r]) |o| o.deinit();
            out[r] = null;
            out[r] = try leaderTurn(pm.modes.tp.?, widths);
        }
        if (try TpProofGroup.settled()) {
            try TpProofGroup.expectAgreement();
            try testing.expectEqual(@as(usize, 0), LeaderReplay.wide_steps);
            try testing.expect(LeaderReplay.chunks.items.len > 0);
            std.debug.print("[leader-prefill] settled in {d} passes, {d} leader chunks\n", .{ pass + 1, LeaderReplay.chunks.items.len });
            return .{ out[0].?, out[1].?, out[2].?, out[3].? };
        }
    }
    return error.TpGroupDidNotSettle;
}

fn leaderProofPack(widths: []const usize, stages: bool, shard: bool) !void {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const previous = msv_test_swap_tf32(0);
    defer {
        _ = mlx.mlx_synchronize(mlx.gpuStream());
        _ = msv_test_swap_tf32(previous);
    }
    _ = setenv("DSV41_DRAFT_SHARD", if (shard) "1" else "0", 1);
    defer _ = unsetenv("DSV41_DRAFT_SHARD");
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = if (stages) try fixture.v41TpDsparkFiles(arena.allocator(), 7) else try fixture.v41TpFiles(arena.allocator(), 7);
    var ts: [4]*Twin = undefined;
    var made: usize = 0;
    defer for (ts[0..made]) |t| t.close();
    for (&ts, 0..) |*t, r| {
        const setting = try std.fmt.allocPrint(arena.allocator(), "{{\"tp_mode\":{{\"rank\":{d},\"size\":4}}}}", .{r});
        t.* = try Twin.openWith(files, "", setting);
        made += 1;
        try rebuildTpProofF32(t.*, @intCast(r));
        const model: *v41.Model = @ptrCast(@alignCast(t.*.m));
        for ([_]*dsv41.Dsv41Model{ model.modes.tp1, model.modes.tp.? }) |m| {
            var cfg = m.cfg;
            @import("decode/fixture.zig").engramConfig(&cfg, .{ 1, 3 });
            m.hash = try engram.Hash.init(&cfg, 2);
            m.token_map = try m.gpa.alloc(u32, m.vocab);
            for (m.token_map, 0..) |*v, k| v.* = @intCast(k);
        }
    }
    const pm: *v41.Model = @ptrCast(@alignCast(ts[0].m));
    const want = try leaderTurn(pm.modes.tp1, widths);
    defer want.deinit();
    const got = try settledLeaderTurn(ts, widths);
    defer for (got) |o| o.deinit();
    try testing.expect(try tpLogitsMatch("leader wide prefill", want.prefill, got[0].prefill, 1e-3));
    for (got) |o| try testing.expect(try tpLogitsMatch("decode after leader prefill", want.decode, o.decode, 1e-3));
    if (stages) {
        const joined = try testing.allocator.alloc(f32, want.draft.len);
        defer testing.allocator.free(joined);
        if (shard) {
            const width = pm.mdl.vocab / 4;
            for (got, 0..) |o, r| for (0..pm.mdl.ds_block) |b| {
                @memcpy(joined[b * pm.mdl.vocab + r * width ..][0..width], o.draft[b * width ..][0..width]);
            };
        } else @memcpy(joined, got[0].draft);
        try testing.expect(try tpLogitsMatch("leader DSpark and Markov", want.draft, joined, 1e-3));
        return;
    }
    for (ts) |t| {
        const m: *v41.Model = @ptrCast(@alignCast(t.m));
        for (m.modes.row_cut.?.layers) |*layer| if (layer.*) |*rl| {
            rl.fused = null;
            std.mem.swap(dsv41.Bank, &rl.experts.gate, &rl.experts.up);
        };
    }
    const bad = try settledLeaderTurn(ts, widths);
    defer for (bad) |o| o.deinit();
    try testing.expect(!try tpLogitsMatch("negative after leader prefill", got[0].decode, bad[0].decode, 1e-3));
}

test "deepseek_v41 leader prefill: a wide chunk ships state before TP4 decode" {
    for ([_]bool{ false, true }) |stages| try leaderProofPack(&.{32}, stages, true);
    try leaderProofPack(&.{32}, true, false);
}

test "deepseek_v41 leader prefill: wide chunks and a narrow tail keep TP4 logits" {
    for ([_]bool{ false, true }) |stages| try leaderProofPack(&.{ 25, 32, 7 }, stages, true);
}
const LeaderCacheRig = struct {
    models: [3]*dsv41.Dsv41Model,
    parked: [3]dsv41.Parked,
    snaps: [3]dsv41.Snaps,
    loads: usize = 0,
    fail_ship: bool = false,
    failed_ship_seen: bool = false,
    failure: ?anyerror = null,
    var active: *@This() = undefined;

    fn init(ts: [4]*Twin) @This() {
        var out: @This() = undefined;
        for (ts[1..], 0..) |t, i| {
            const pm: *v41.Model = @ptrCast(@alignCast(t.m));
            out.models[i] = pm.modes.tp.?;
            out.parked[i] = .init(testing.allocator);
            out.snaps[i] = .init(testing.allocator);
        }
        out.loads = 0;
        out.fail_ship = false;
        out.failed_ship_seen = false;
        out.failure = null;
        return out;
    }

    fn deinit(self: *@This()) void {
        for (&self.parked, &self.snaps) |*p, *s| {
            dsv41.dropParked(p);
            dsv41.dropSnaps(s, testing.allocator);
        }
    }

    fn reform(self: *@This(), lead: *v41.Model) void {
        self.deinit();
        for (self.models, &self.parked, &self.snaps) |m, *p, *s| {
            if (m.dec_state) |*st| st.deinit();
            m.dec_state = null;
            p.* = .init(testing.allocator);
            s.* = .init(testing.allocator);
        }
        v41.groupFormed(lead);
    }

    fn ship(gen: u64, p0: usize, n: usize, capture: bool, xs: []const mlx.mlx_array, stages: usize) anyerror!void {
        const self = active;
        for (self.models, &self.parked, 0..) |m, *p, i| {
            if (self.fail_ship and i == 1) {
                self.failed_ship_seen = true;
                return error.TestLeaderShipFailed;
            }
            try dsv41.tpWorkerPrefill(m, testing.allocator, p, gen, p0, n, capture, xs, stages);
        }
    }

    fn load(id: u64, source_gen: u64, ids: []const u32, xs: []const mlx.mlx_array) anyerror!void {
        const self = active;
        for (self.models, &self.snaps) |m, *s| try dsv41.tpWorkerLoad(m, testing.allocator, s, id, source_gen, ids, xs);
        self.loads += 1;
    }

    fn snap(op: dsv41.SnapOp, a: u64, b: u64, evidence: dsv41.SnapEvidence) void {
        const self = active;
        for (self.models, &self.parked, &self.snaps) |m, *p, *s|
            dsv41.tpWorkerSnap(m, testing.allocator, p, s, op, a, b, evidence) catch |err| {
                self.failure = err;
                return;
            };
    }

    fn free(gen: u64, owner: log.RequestTag) void {
        const self = active;
        for (self.models, &self.parked) |m, *p| dsv41.tpWorkerFree(m, p, gen, owner);
    }
};

test "deepseek_v41 leader prefill: aborted chunks restore and reload after group formation, failed shipping cannot snapshot" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try fixture.v41TpFiles(arena.allocator(), 7);
    var ts: [4]*Twin = undefined;
    var made: usize = 0;
    defer for (ts[0..made]) |t| t.close();
    for (&ts, 0..) |*t, r| {
        const setting = try std.fmt.allocPrint(arena.allocator(), "{{\"tp_mode\":{{\"rank\":{d},\"size\":4}}}}", .{r});
        t.* = try Twin.openWith(files, "", setting);
        made += 1;
        try rebuildTpProofF32(t.*, @intCast(r));
        const pm: *v41.Model = @ptrCast(@alignCast(t.*.m));
        try v41.swapMode(pm, .tp);
    }
    const lead: *v41.Model = @ptrCast(@alignCast(ts[0].m));
    v41.groupFormed(lead);
    var rig = LeaderCacheRig.init(ts);
    defer rig.deinit();
    LeaderCacheRig.active = &rig;
    dsv41.clusterHooks().* = .{ .tp_prefill = &LeaderCacheRig.ship, .tp_load = &LeaderCacheRig.load, .tp_snap = &LeaderCacheRig.snap, .tp_free = &LeaderCacheRig.free };
    defer dsv41.clusterHooks().* = .{};
    const aux = v41.table.cache.aux.?;
    const prompt = tp_prompt ++ tp_prompt;
    try testing.expect(aux.begin_prompt(ts[0].m, 1, .of(&prompt), .{}, 0, true).isOk());
    testing.allocator.free(try slotForward(ts[0], .prefill, &.{1}, prompt[0..25], &.{0}));
    var sn: abi.Handle = null;
    try testing.expect(aux.snapshot(ts[0].m, 1, &sn).isOk());
    defer aux.free(ts[0].m, sn);
    const saved: *dsv41.Snap = @ptrCast(@alignCast(sn orelse return error.MissingSnapshot));
    try testing.expectEqual(@as(usize, 25), saved.n);
    try testing.expectEqualSlices(u32, prompt[0..25], saved.ids);
    try testing.expect(rig.failure == null);
    for (&rig.snaps) |*s| try testing.expectEqual(@as(usize, 25), (s.getPtr(saved.tp_id) orelse return error.FollowerSnapshotMissing).n);
    var clone: abi.Handle = null;
    try testing.expect(aux.slice(ts[0].m, sn, saved.n, &clone).isOk());
    defer aux.free(ts[0].m, clone);
    try testing.expect(aux.restore_at(ts[0].m, 2, clone, saved.n).isOk());
    for (rig.models) |m| try testing.expectEqual(saved.n, m.dec_state.?.n);
    try testing.expect(rig.failure == null);
    rig.reform(lead);
    try testing.expect(aux.restore_at(ts[0].m, 3, sn, saved.n).isOk());
    try testing.expectEqual(@as(usize, 1), rig.loads);
    try testing.expect(aux.restore_at(ts[0].m, 3, sn, saved.n).isOk());
    try testing.expectEqual(@as(usize, 1), rig.loads);
    for (rig.models) |m| try testing.expectEqual(saved.n, m.dec_state.?.n);
    try testing.expect(rig.failure == null);
    rig.fail_ship = true;
    try testing.expect(aux.begin_prompt(ts[0].m, 4, .of(&prompt), .{}, 0, true).isOk());
    try testing.expectError(error.TestLeaderShipFailed, dsv41.extendResumable(lead.mdl, testing.allocator, prompt[0..25], null));
    try testing.expect(rig.failed_ship_seen);
    var refused: abi.Handle = null;
    try testing.expectEqual(abi.Verdict.Code.failed, aux.snapshot(ts[0].m, 4, &refused).code);
    try testing.expect(refused == null);
}
test "deepseek_v41 leader prefill: Engram read-ahead stays with the active TP state across wide chunks" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const fxmod = @import("decode/fixture.zig");
    const rcmod = @import("rowcut.zig");
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = std.testing.tmpDir(.{});
    defer td.cleanup();
    try fxmod.writeEngramPack(gpa, td.dir, 1024, .{ 0, 1 });
    var path: [512]u8 = undefined;
    const dir = path[0..try td.dir.realPath(io, &path)];
    const fx = try fxmod.Fixture.initWhole(gpa, .{ .layers = 2, .engram_dir = dir, .engram_layers = .{ 0, 1 } });
    defer fx.deinit();
    fx.m.cfg.dsv41_kv_sources &= 3;
    fx.m.cfg.dsv41_index_sources &= 3;
    fx.m.whole_cfg = fx.m.cfg;
    const rows = try rcmod.RowCut.create(gpa, fx.m.s, fx.m, 0, fx.m.n_layers, 0, 4);
    defer rows.destroy();
    const tp = try dsv41.init(gpa, &fx.m.cfg, &fx.w, fx.m.s, .{ .tp = .{ .rank = 0, .size = 4 }, .resident = .{ .model = fx.m, .rows = rows }, .pack_dir = dir });
    defer tp.deinit();
    tp.prefill_whole = fx.m;
    LoneGroup.arm(4);
    defer LoneGroup.disarm();
    var prompt: [96]u32 = undefined;
    for (&prompt, 0..) |*id, i| id.* = @intCast((i * 13 + 3) % 1024);
    try dsv41.reset(tp, gpa);
    try dsv41.armPrompt(tp, &prompt, null, true, null);
    for ([_]usize{ 0, 32 }) |p0| {
        _ = mlx.mlx_array_free(try dsv41.extendResumable(tp, gpa, prompt[p0..][0..32], null));
        const ahead = tp.ahead orelse return error.LeaderPrefillDroppedEngramReadAhead;
        try testing.expect(ahead.covers(p0 + 32, 32));
        try testing.expect(fx.m.ahead == null);
        try testing.expectEqualSlices(u32, prompt[0 .. p0 + 32], tp.dec_state.?.hist.items);
    }
    try testing.expectEqual(@as(usize, 0), fx.m.prompt.len);
}
const WarmRestore = struct {
    const Step = struct { ids: [16]u32 = undefined, len: usize, p0: usize, snap: bool, all: bool, verify: dsv41.VerifyOpts, gen: u64 };
    const Snap = struct { op: dsv41.SnapOp, a: u64, b: u64 };
    const Op = union(enum) { step: Step, chunk: usize, snap: Snap, free: u64 };
    var events: std.ArrayList(Op) = .empty;
    var failed: bool = false;
    const suffix = tp_prompt ++ [_]u32{ 9, 7, 25, 1, 60, 3, 17, 42 };

    fn clear() void {
        events.deinit(testing.allocator);
        events = .empty;
        LeaderReplay.clear();
        failed = false;
    }

    fn install() void {
        dsv41.clusterHooks().tp_step = &step;
        dsv41.clusterHooks().tp_prefill = &ship;
        dsv41.clusterHooks().tp_snap = &snap;
        dsv41.clusterHooks().tp_free = &free;
    }

    fn step(ids: []const u32, p0: usize, capture: bool, all: bool, verify: dsv41.VerifyOpts, gen: u64, images: ?mlx.mlx_array) anyerror!void {
        if (images != null or ids.len > 16) return error.TestUnexpectedResult;
        var op: Step = .{ .len = ids.len, .p0 = p0, .snap = capture, .all = all, .verify = verify, .gen = gen };
        @memcpy(op.ids[0..ids.len], ids);
        try events.append(testing.allocator, .{ .step = op });
    }

    fn ship(gen: u64, p0: usize, n: usize, capture: bool, xs: []const mlx.mlx_array, stages: usize) anyerror!void {
        const index = LeaderReplay.chunks.items.len;
        try LeaderReplay.ship(gen, p0, n, capture, xs, stages);
        try events.append(testing.allocator, .{ .chunk = index });
    }

    fn snap(op: dsv41.SnapOp, a: u64, b: u64, _: dsv41.SnapEvidence) void {
        events.append(testing.allocator, .{ .snap = .{ .op = op, .a = a, .b = b } }) catch {
            failed = true;
        };
    }

    fn free(gen: u64, _: log.RequestTag) void {
        events.append(testing.allocator, .{ .free = gen }) catch {
            failed = true;
        };
    }

    fn leader(t: *Twin, keep: usize) !void {
        const aux = v41.table.cache.aux.?;
        const pm: *v41.Model = @ptrCast(@alignCast(t.m));
        try testing.expect(aux.begin_prompt(t.m, 1, .of(&tp_prompt), .{}, 0, false).isOk());
        testing.allocator.free(try stepLogits(t, 1, .{ .mode = .prefill, .ids = &tp_prompt }, 0));
        testing.allocator.free(try stepLogits(t, 1, .{ .mode = .decode, .ids = &.{8} }, 32));
        try dsv41.armVerify(&pm.mdl.dec_state.?, pm.mdl.n_layers);
        const verified = try dsv41.extend(pm.mdl, testing.allocator, &pm.mdl.dec_state.?, &.{ 3, 17, 42, 9, 7, 25 }, .all);
        _ = mlx.mlx_array_free(verified);
        try dsv41.finishVerify(pm.mdl, testing.allocator, &pm.mdl.dec_state.?, keep);
        const n = 33 + keep;
        var held: [40]u32 = undefined;
        const accepted = tp_prompt ++ [_]u32{ 8, 3, 17, 42, 9, 7, 25 };
        @memcpy(held[0..n], accepted[0..n]);
        held[n] = 0;
        try testing.expect(aux.begin_prompt(t.m, 1, .of(held[0 .. n + 1]), .{}, n, true).isOk());
        var sn: abi.Handle = null;
        try testing.expect(aux.snapshot(t.m, 1, &sn).isOk());
        defer aux.free(t.m, sn);
        const saved: *dsv41.Snap = @ptrCast(@alignCast(sn orelse return error.MissingSnapshot));
        try testing.expectEqual(n, saved.n);
        v41.table.release_slot.?(t.m, 1);
        try testing.expect(aux.restore_at(t.m, 1, sn, n).isOk());
        var next: [80]u32 = undefined;
        @memcpy(next[0..n], accepted[0..n]);
        @memcpy(next[n..][0..suffix.len], &suffix);
        next[n + suffix.len] = 11;
        try testing.expect(aux.begin_prompt(t.m, 1, .of(next[0 .. n + suffix.len + 1]), .{}, n, true).isOk());
        const skipped = pm.mdl.prefill_whole orelse pm.mdl;
        const before = skipped.ced_stats.skipped;
        testing.allocator.free(try stepLogits(t, 1, .{ .mode = .prefill, .ids = &suffix }, @intCast(n)));
        try testing.expectEqual(@as(u64, suffix.len - skipped.cfg.sliding_window), skipped.ced_stats.skipped - before);
        try testing.expect(suffix.len - skipped.cfg.sliding_window > @import("rowcut.zig").max_rows);
        try testing.expect(!failed);
    }

    fn replay(m: *dsv41.Dsv41Model, end: usize, keep: usize) !void {
        var parked: dsv41.Parked = .init(testing.allocator);
        defer dsv41.dropParked(&parked);
        var snaps: dsv41.Snaps = .init(testing.allocator);
        defer dsv41.dropSnaps(&snaps, testing.allocator);
        var restored: usize = 0;
        for (events.items[0..end]) |op| switch (op) {
            .step => |s| try dsv41.tpWorkerStep(m, testing.allocator, &parked, s.gen, s.ids[0..s.len], s.p0, s.snap, if (s.all) .all else .last, s.verify, null),
            .chunk => |index| {
                const c = LeaderReplay.chunks.items[index];
                if (m.dec_state != null and m.dec_state.?.n != c.p0)
                    std.debug.print("[warm-restore] rank={d} import base={d} live={d} end={d}\n", .{ m.tp.?.rank, c.p0, m.dec_state.?.n, c.n });
                try dsv41.tpWorkerPrefill(m, testing.allocator, &parked, c.gen, c.p0, c.n, c.capture, c.xs, c.stages);
            },
            .snap => |s| {
                try dsv41.tpWorkerSnap(m, testing.allocator, &parked, &snaps, s.op, s.a, s.b, .{});
                if (s.op == .restore) restored = m.dec_state.?.n;
            },
            .free => |gen| dsv41.tpWorkerFree(m, &parked, gen, .{}),
        };
        try testing.expectEqual(33 + keep, restored);
    }

    fn tail(m: *dsv41.Dsv41Model) ![]f32 {
        var out: std.ArrayList(f32) = .empty;
        errdefer out.deinit(testing.allocator);
        for ([_]u32{ 11, 5, 8 }) |id| {
            const logits = try dsv41.extendResumable(m, testing.allocator, &.{id}, null);
            defer _ = mlx.mlx_array_free(logits);
            const vals = try dsv41.hostF32(logits);
            defer testing.allocator.free(vals);
            try out.appendSlice(testing.allocator, vals);
        }
        return out.toOwnedSlice(testing.allocator);
    }
};

test "deepseek_v41 leader prefill: RAM decode checkpoint settles pending verify before CED suffix and TP4 decode" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const previous = msv_test_swap_tf32(0);
    defer {
        _ = mlx.mlx_synchronize(mlx.gpuStream());
        _ = msv_test_swap_tf32(previous);
    }
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const files = try arena.allocator().dupe(sdk.fixture.File, try fixture.v41TpDsparkFiles(arena.allocator(), 7));
    files[0].bytes = try std.mem.replaceOwned(u8, arena.allocator(), files[0].bytes, "\"dspark_target_layer_ids\": [2, 3]", "\"dspark_target_layer_ids\": [5, 7]");
    var ts: [4]*Twin = undefined;
    var made: usize = 0;
    defer for (ts[0..made]) |t| t.close();
    for (&ts, 0..) |*t, r| {
        const setting = try std.fmt.allocPrint(arena.allocator(), "{{\"tp_mode\":{{\"rank\":{d},\"size\":4}}}}", .{r});
        t.* = try Twin.openWith(files, "", setting);
        made += 1;
        try rebuildTpProofF32(t.*, @intCast(r));
        const pm: *v41.Model = @ptrCast(@alignCast(t.*.m));
        for ([_]*dsv41.Dsv41Model{ pm.modes.tp1, pm.modes.tp.? }) |m| {
            try testing.expect(m.ced_split != null);
            m.ced_min_skip = m.cfg.sliding_window;
        }
        try v41.swapMode(pm, .tp);
    }
    const lead: *v41.Model = @ptrCast(@alignCast(ts[0].m));
    for ([_]usize{ 3, 6 }) |keep| {
        try v41.swapMode(lead, .tp1);
        try WarmRestore.leader(ts[0], keep);
        const want = try WarmRestore.tail(lead.mdl);
        defer testing.allocator.free(want);
        try v41.swapMode(lead, .tp);
        LoneGroup.arm(4);
        TpProofGroup.join(4);
        defer {
            TpProofGroup.leave();
            LoneGroup.disarm();
            WarmRestore.clear();
        }
        var settled = false;
        for (0..40) |pass| {
            WarmRestore.clear();
            WarmRestore.install();
            TpProofGroup.begin(0);
            try WarmRestore.leader(ts[0], keep);
            const end = WarmRestore.events.items.len;
            dsv41.clusterHooks().tp_step = &LeaderReplay.step;
            var decoded: [4]?[]f32 = @splat(null);
            defer for (decoded) |values| if (values) |v| testing.allocator.free(v);
            decoded[0] = try WarmRestore.tail(lead.mdl);
            for (ts[1..], 1..) |t, rank| {
                TpProofGroup.begin(rank);
                const pm: *v41.Model = @ptrCast(@alignCast(t.m));
                try WarmRestore.replay(pm.mdl, end, keep);
                decoded[rank] = try WarmRestore.tail(pm.mdl);
            }
            if (try TpProofGroup.settled()) {
                try TpProofGroup.expectAgreement();
                for (decoded) |values| try testing.expect(try tpLogitsMatch("RAM restore TP4 decode", want, values.?, 1e-3));
                std.debug.print("[warm-restore] keep={d}/6 settled in {d} passes\n", .{ keep, pass + 1 });
                settled = true;
                break;
            }
        }
        try testing.expect(settled);
    }
}
