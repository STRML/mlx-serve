//! HTTP handlers for `X-Cache-Export: 1` on `/v1/chat/completions` and `POST /v1/cache/import`.

const std = @import("std");
const kv_quant = @import("../../sdk/blocks/attention/kv_quant.zig");
const log = @import("../../sdk/util/log.zig");
const http_server = @import("../http/server.zig");
const Conn = @import("../http/conn.zig").Conn;
const router = @import("../http/router.zig");
const sendResponse = @import("../http/respond.zig").sendResponse;
const sendResponseQuiet = @import("../http/respond.zig").sendResponseQuiet;
const sendErrorResponse = @import("../http/errors.zig").sendErrorResponse;
const sendErrorResponseWithReason = @import("../http/errors.zig").sendErrorResponseWithReason;
const requests = @import("../gpu_thread/requests.zig");
const LoadedModel = @import("../registry/registry.zig").LoadedModel;

pub var max_export_bytes_override: ?usize = null;

/// Each export's buffer lives until the client has read it; past this many the next export is refused.
pub const max_exports_in_flight = 8;
pub var exports_in_flight = std.atomic.Value(u32).init(0);

fn effectiveMaxExportBytes() usize {
    return max_export_bytes_override orelse router.max_cache_export_bytes;
}

pub fn speechAsked(root: std.json.ObjectMap) bool {
    if (root.get("audio")) |a| {
        if (a != .null) return true;
    }
    const m = root.get("modalities") orelse return false;
    if (m == .null) return false;
    if (m != .array) return true;
    for (m.array.items) |v| {
        if (v != .string or !std.mem.eql(u8, v.string, "text")) return true;
    }
    return false;
}

pub fn handleCacheExport(
    allocator: std.mem.Allocator,
    stream: *Conn,
    lm: *LoadedModel,
    prompt_ids: []const u32,
    has_tools: bool,
    kv_quant_override: ?kv_quant.KVQuantConfig,
    variant: u64,
    cache_key: u64,
    deadline_ms: ?u64,
) !void {
    const scheduler = http_server.launch().global_scheduler orelse {
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Scheduler not ready", 503);
        return;
    };
    defer _ = exports_in_flight.fetchSub(1, .acq_rel);
    if (exports_in_flight.fetchAdd(1, .acq_rel) >= max_exports_in_flight) {
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "insufficient_memory", "cache export refused: exports busy, too many export buffers in flight", 503);
        return;
    }
    var req: requests.CacheExportRequest = .{
        .model = lm,
        .prompt_ids = prompt_ids,
        .has_tools = has_tools,
        .kv_quant_override = kv_quant_override,
        .variant = variant,
        .cache_key = cache_key,
        .deadline_ms = deadline_ms,
        .max_bytes = effectiveMaxExportBytes(),
        .allocator = allocator,
    };
    const outcome = scheduler.exportCache(&req) catch |err| {
        log.warn("POST /v1/chat/completions (cache export) -> 500 ({s})\n", .{@errorName(err)});
        try sendErrorResponse(allocator, stream, "500 Internal Server Error", "internal_error", "Cache export failed", 500);
        return;
    };
    switch (outcome) {
        .deadline_passed => {
            try sendErrorResponse(allocator, stream, "408 Request Timeout", "timeout_error", "cache export refused: deadline passed before it started", 408);
        },
        .not_found => {
            try sendErrorResponse(allocator, stream, "404 Not Found", "not_found", "No matching RAM prefix-cache entry", 404);
        },
        .transfer_unsupported => {
            try sendErrorResponseWithReason(allocator, stream, "409 Conflict", "conflict", transfer_unsupported_message, 409, "transfer_unsupported");
        },
        .identity_pending => {
            try sendErrorResponseWithReason(allocator, stream, "409 Conflict", "conflict", identity_pending_message, 409, "identity_pending");
        },
        .too_large => |got| {
            var msg_buf: [128]u8 = undefined;
            const msg = router.payloadTooLargeMessage(&msg_buf, @intCast(@min(got, std.math.maxInt(usize))), effectiveMaxExportBytes());
            try sendErrorResponse(allocator, stream, "413 Payload Too Large", "invalid_request_error", msg, 413);
        },
        .ok => |res| {
            defer allocator.free(res.bytes);
            try sendResponseQuiet(stream, "200 OK", "application/octet-stream", res.bytes);
        },
    }
}

const ImportHttpError = struct {
    status: []const u8,
    code: u32,
    kind: []const u8,
    message: []const u8,
    reason: ?[]const u8 = null,
};

const transfer_unsupported_message = "cache transfer unsupported: the model path is neither a file nor a folder, or could not be read";
const identity_pending_message = "cache transfer refused: the model's pack identity is still being hashed, retry later";

fn refusalHttp(r: requests.CacheImportRefusal) ImportHttpError {
    return switch (r) {
        .deadline_passed => .{ .status = "408 Request Timeout", .code = 408, .kind = "timeout_error", .message = "cache import refused: deadline passed before it started" },
        .bad_envelope => .{ .status = "400 Bad Request", .code = 400, .kind = "invalid_request_error", .message = "cache import refused: invalid envelope" },
        .unknown_version => .{ .status = "400 Bad Request", .code = 400, .kind = "invalid_request_error", .message = "cache import refused: unknown envelope version" },
        .truncated_body => .{ .status = "400 Bad Request", .code = 400, .kind = "invalid_request_error", .message = "cache import refused: truncated body" },
        .bad_dtype => .{ .status = "400 Bad Request", .code = 400, .kind = "invalid_request_error", .message = "cache import refused: invalid tensor dtype" },
        .bad_shape => .{ .status = "400 Bad Request", .code = 400, .kind = "invalid_request_error", .message = "cache import refused: invalid tensor shape" },
        .bad_offsets => .{ .status = "400 Bad Request", .code = 400, .kind = "invalid_request_error", .message = "cache import refused: invalid payload offsets" },
        .bad_checksum => .{ .status = "400 Bad Request", .code = 400, .kind = "invalid_request_error", .message = "cache import refused: payload checksum mismatch" },
        .bad_quant_config => .{ .status = "400 Bad Request", .code = 400, .kind = "invalid_request_error", .message = "cache import refused: invalid quant config" },
        .unsupported_model => .{ .status = "422 Unprocessable Entity", .code = 422, .kind = "invalid_request_error", .message = "cache import refused: model has no module prefix cache" },
        .layout_key => .{ .status = "422 Unprocessable Entity", .code = 422, .kind = "invalid_request_error", .message = "cache import refused: layout_key mismatch" },
        .pack_identity => .{ .status = "422 Unprocessable Entity", .code = 422, .kind = "invalid_request_error", .message = "cache import refused: pack_identity mismatch" },
        .settings => .{ .status = "422 Unprocessable Entity", .code = 422, .kind = "invalid_request_error", .message = "cache import refused: settings mismatch" },
        .quant_config => .{ .status = "422 Unprocessable Entity", .code = 422, .kind = "invalid_request_error", .message = "cache import refused: quant_config mismatch" },
        .import_plan => .{ .status = "422 Unprocessable Entity", .code = 422, .kind = "invalid_request_error", .message = "cache import refused: plugin import_plan rejected header" },
        .import_arrays => .{ .status = "422 Unprocessable Entity", .code = 422, .kind = "invalid_request_error", .message = "cache import refused: plugin import_arrays failed" },
        .describe_mismatch => .{ .status = "422 Unprocessable Entity", .code = 422, .kind = "invalid_request_error", .message = "cache import refused: snapshot identity mismatch" },
        .stale => .{ .status = "409 Conflict", .code = 409, .kind = "conflict", .message = "cache import refused: stale or duplicate entry" },
        .commit_declined => .{ .status = "409 Conflict", .code = 409, .kind = "conflict", .message = "cache import refused: commit declined" },
        .no_memory => .{ .status = "503 Service Unavailable", .code = 503, .kind = "insufficient_memory", .message = "cache import refused: insufficient memory headroom" },
        .transfer_unsupported => .{ .status = "409 Conflict", .code = 409, .kind = "conflict", .message = transfer_unsupported_message, .reason = "transfer_unsupported" },
        .identity_pending => .{ .status = "409 Conflict", .code = 409, .kind = "conflict", .message = identity_pending_message, .reason = "identity_pending" },
    };
}

pub fn handleCacheImport(
    allocator: std.mem.Allocator,
    stream: *Conn,
    body: []const u8,
    lm: *LoadedModel,
) !void {
    const scheduler = http_server.launch().global_scheduler orelse {
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Scheduler not ready", 503);
        return;
    };
    var req: requests.CacheImportRequest = .{
        .model = lm,
        .body = body,
        .deadline_ms = stream.cache_deadline_ms,
        .allocator = allocator,
    };
    const outcome = scheduler.importCache(&req) catch |err| {
        log.warn("POST /v1/cache/import -> 500 ({s})\n", .{@errorName(err)});
        try sendErrorResponse(allocator, stream, "500 Internal Server Error", "internal_error", "Cache import failed", 500);
        return;
    };
    switch (outcome) {
        .ok => |s| {
            var json_buf: [256]u8 = undefined;
            const json = try std.fmt.bufPrint(
                &json_buf,
                "{{\"n\":{d},\"bytes\":{d},\"decode_ms\":{d},\"admit_ms\":{d},\"import_ms\":{d},\"commit_ms\":{d}}}",
                .{ s.n, s.bytes, s.decode_ms, s.admit_ms, s.import_ms, s.commit_ms },
            );
            try sendResponse(stream, "200 OK", "application/json", json);
        },
        .refused => |r| {
            const e = refusalHttp(r);
            try sendErrorResponseWithReason(allocator, stream, e.status, e.kind, e.message, e.code, e.reason);
        },
    }
}
