//! Synchronized PP4 bound measurements and their JSONL receipts.
const std = @import("std");

pub const Phase = enum { encoder, decoder };

pub const Trace = struct {
    p0: usize,
    rows: usize,
    want: []const u8,
    start_us: i64,
    epoch_us: i64,
    enc_hi: usize = 0,
    tail_rows: usize = 0,
    encoder_us: i64 = 0,
    encoder_wait_us: i64 = 0,
    compressor_us: i64 = 0,
    decoder_us: i64 = 0,
    dspark_us: i64 = 0,
    head_us: i64 = 0,
    layer_wait_us: i64 = 0,
    dspark_enabled: bool = false,

    pub fn xUs(self: Trace) i64 {
        return self.compressor_us + self.decoder_us + self.dspark_us;
    }

    pub fn formatLayer(self: Trace, buf: []u8, layer: usize, layer_rows: usize, start_us: i64, end_us: i64) ![]const u8 {
        const phase: Phase = if (layer < self.enc_hi) .encoder else .decoder;
        const elapsed_us = end_us - start_us;
        return std.fmt.bufPrint(buf, "{{\"kind\":\"pp4_layer\",\"p0\":{d},\"rows\":{d},\"layer_rows\":{d},\"layer\":{d},\"phase\":\"{s}\",\"start_us\":{d},\"epoch_us\":{d},\"end_us\":{d},\"elapsed_us\":{d},\"engram_wait_us\":{d},\"compute_us\":{d}}}\n", .{ self.p0, self.rows, layer_rows, layer, @tagName(phase), start_us, self.epoch_us + start_us - self.start_us, end_us, elapsed_us, self.layer_wait_us, elapsed_us - self.layer_wait_us });
    }

    pub fn noteLayer(self: *Trace, layer: usize, elapsed_us: i64) void {
        if (layer < self.enc_hi) {
            self.encoder_us += elapsed_us;
            self.encoder_wait_us += self.layer_wait_us;
        } else self.decoder_us += elapsed_us;
    }

    pub fn formatChunk(self: Trace, buf: []u8, end_us: i64) ![]const u8 {
        const elapsed_us = end_us - self.start_us;
        const accounted_us = self.encoder_us + self.xUs() + self.head_us;
        return std.fmt.bufPrint(buf, "{{\"kind\":\"pp4_chunk\",\"p0\":{d},\"rows\":{d},\"want\":\"{s}\",\"enc_hi\":{d},\"tail_rows\":{d},\"start_us\":{d},\"epoch_us\":{d},\"end_us\":{d},\"elapsed_us\":{d},\"encoder_us\":{d},\"encoder_wait_us\":{d},\"encoder_compute_us\":{d},\"compressor_us\":{d},\"decoder_us\":{d},\"dspark_us\":{d},\"head_us\":{d},\"x_us\":{d},\"unattributed_us\":{d},\"dspark_enabled\":{s}}}\n", .{ self.p0, self.rows, self.want, self.enc_hi, self.tail_rows, self.start_us, self.epoch_us, end_us, elapsed_us, self.encoder_us, self.encoder_wait_us, self.encoder_us - self.encoder_wait_us, self.compressor_us, self.decoder_us, self.dspark_us, self.head_us, self.xUs(), elapsed_us - accounted_us, if (self.dspark_enabled) "true" else "false" });
    }

    pub fn formatEngram(self: Trace, buf: []u8, table: usize, span: usize, read_start_us: i64, ready_us: i64, needed_us: i64, taken_us: i64) ![]const u8 {
        return std.fmt.bufPrint(buf, "{{\"kind\":\"pp4_engram\",\"p0\":{d},\"rows\":{d},\"table\":{d},\"span\":{d},\"read_start_us\":{d},\"ready_us\":{d},\"needed_us\":{d},\"taken_us\":{d},\"wait_us\":{d},\"epoch_us\":{d}}}\n", .{ self.p0, self.rows, table, span, read_start_us, ready_us, needed_us, taken_us, taken_us - needed_us, self.epoch_us + needed_us - self.start_us });
    }
};

test "pp4 step0: X includes evaluated DSpark and excludes head and Engram wait" {
    const trace: Trace = .{
        .p0 = 2048,
        .rows = 2048,
        .want = "last",
        .start_us = 100,
        .epoch_us = 900100,
        .enc_hi = 20,
        .tail_rows = 128,
        .encoder_us = 1100,
        .encoder_wait_us = 100,
        .compressor_us = 20,
        .decoder_us = 70,
        .dspark_us = 30,
        .head_us = 400,
    };
    try std.testing.expectEqual(@as(i64, 120), trace.xUs());
    var buf: [2048]u8 = undefined;
    const raw = try trace.formatChunk(&buf, 2100);
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, raw, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try std.testing.expectEqual(@as(i64, 1000), o.get("encoder_compute_us").?.integer);
    try std.testing.expectEqual(@as(i64, 120), o.get("x_us").?.integer);
    try std.testing.expectEqual(@as(i64, 380), o.get("unattributed_us").?.integer);
    try std.testing.expectEqual(@as(i64, 2048), o.get("p0").?.integer);
    try std.testing.expectEqual(@as(i64, 128), o.get("tail_rows").?.integer);
    try std.testing.expectEqual(@as(i64, 900100), o.get("epoch_us").?.integer);
}

test "pp4 step0: layer wait and delivery timestamps retain the actual demand schedule" {
    var trace: Trace = .{ .p0 = 4096, .rows = 2048, .want = "last", .start_us = 100, .epoch_us = 900100, .enc_hi = 20 };
    trace.layer_wait_us = 70;
    trace.noteLayer(1, 300);
    trace.layer_wait_us = 0;
    trace.noteLayer(20, 40);
    try std.testing.expectEqual(@as(i64, 300), trace.encoder_us);
    try std.testing.expectEqual(@as(i64, 70), trace.encoder_wait_us);
    try std.testing.expectEqual(@as(i64, 40), trace.decoder_us);
    trace.layer_wait_us = 70;
    var buf: [1024]u8 = undefined;
    const layer = try trace.formatLayer(&buf, 1, 2048, 200, 500);
    var lp = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, layer, .{});
    defer lp.deinit();
    try std.testing.expectEqual(@as(i64, 230), lp.value.object.get("compute_us").?.integer);
    try std.testing.expectEqual(@as(i64, 900200), lp.value.object.get("epoch_us").?.integer);
    const rows = try trace.formatEngram(&buf, 1, 3, 120, 300, 230, 300);
    var rp = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, rows, .{});
    defer rp.deinit();
    try std.testing.expectEqual(@as(i64, 70), rp.value.object.get("wait_us").?.integer);
    try std.testing.expectEqual(@as(i64, 300), rp.value.object.get("ready_us").?.integer);
    try std.testing.expectEqual(@as(i64, 230), rp.value.object.get("needed_us").?.integer);
    try std.testing.expectEqual(@as(i64, 900230), rp.value.object.get("epoch_us").?.integer);
}
