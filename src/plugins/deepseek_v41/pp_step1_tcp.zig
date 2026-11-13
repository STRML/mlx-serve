//! POSIX TCP transport for Plan 75 Step 1: getifaddrs subnet bind, MTU check, solo/pipelined hops, and KV streaming.
const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("../../mlx/ffi.zig");
const dsv41 = @import("deepseek_v41.zig");
const pp_step1 = @import("pp_step1.zig");

const is_macos = builtin.os.tag == .macos;
const is_windows = builtin.os.tag == .windows;

pub const SOCKET_BUF_BYTES: c_int = 4 * 1024 * 1024;

const Ifaddrs = extern struct {
    next: ?*Ifaddrs,
    name: [*:0]u8,
    flags: c_uint,
    addr: ?*std.posix.sockaddr,
    netmask: ?*std.posix.sockaddr,
    dstaddr: ?*std.posix.sockaddr,
    data: ?*anyopaque,
};

const IfreqMtu = extern struct {
    name: [16]u8 = @splat(0),
    mtu: c_int = 0,
    pad: [12]u8 = @splat(0),
};

extern "c" fn getifaddrs(ifap: *?*Ifaddrs) c_int;
extern "c" fn freeifaddrs(ifa: ?*Ifaddrs) void;
extern "c" fn if_nametoindex(ifname: [*:0]const u8) c_uint;
extern "c" fn ioctl(fd: c_int, request: c_ulong, ...) c_int;

const IFF_UP: c_uint = 0x1;
const DARWIN_IP_BOUND_IF: c_int = 25;
const DARWIN_SO_NOSIGPIPE: c_int = 0x1022;
const SIOCGIFMTU_CODE: c_ulong = if (is_macos) 0xc0206933 else 0x8921;

pub const BoundSocketInfo = struct {
    fd: std.c.fd_t,
    iface_buf: [16]u8 = @splat(0),
    iface_len: usize = 0,
    local_ip: [4]u8 = .{ 0, 0, 0, 0 },
    peer_ip: [4]u8 = .{ 0, 0, 0, 0 },
    mtu: i32 = 0,
    sndbuf: i32 = 0,
    rcvbuf: i32 = 0,

    pub fn iface(self: *const BoundSocketInfo) []const u8 {
        return self.iface_buf[0..self.iface_len];
    }
};

fn monoUs() i64 {
    const io = std.Io.Threaded.global_single_threaded.io();
    return @intCast(@divTrunc(std.Io.Timestamp.now(io, .awake).toNanoseconds(), 1000));
}

pub fn parseIp4Port(raw: []const u8) !struct { ip4: [4]u8, port: u16 } {
    const colon = std.mem.lastIndexOfScalar(u8, raw, ':') orelse return error.InvalidEndpoint;
    const host = raw[0..colon];
    const port = std.fmt.parseInt(u16, raw[colon + 1 ..], 10) catch return error.InvalidEndpoint;
    var ip4: [4]u8 = undefined;
    var it = std.mem.splitScalar(u8, host, '.');
    for (0..4) |i| {
        const part = it.next() orelse return error.InvalidEndpoint;
        ip4[i] = std.fmt.parseInt(u8, part, 10) catch return error.InvalidEndpoint;
    }
    if (it.next() != null) return error.InvalidEndpoint;
    return .{ .ip4 = ip4, .port = port };
}

pub fn queryInterfaceMtu(ifname: []const u8) i32 {
    if (is_windows or ifname.len == 0 or ifname.len >= 16) return 0;
    const s = std.c.socket(std.posix.AF.INET, std.posix.SOCK.DGRAM, 0);
    if (s < 0) return 0;
    defer _ = std.c.close(s);
    var ifr: IfreqMtu = .{};
    @memcpy(ifr.name[0..ifname.len], ifname);
    if (ioctl(s, SIOCGIFMTU_CODE, &ifr) != 0) return 0;
    return ifr.mtu;
}

pub fn findSubnetInterface(peer_ip: [4]u8, out_name: *[16]u8) ?struct { len: usize, local_ip: [4]u8 } {
    if (is_windows) return null;
    var head: ?*Ifaddrs = null;
    if (getifaddrs(&head) != 0) return null;
    defer freeifaddrs(head);

    const peer_u32: u32 = @bitCast(peer_ip);
    var cur = head;
    while (cur) |ifa| : (cur = ifa.next) {
        if ((ifa.flags & IFF_UP) == 0) continue;
        const addr = ifa.addr orelse continue;
        const mask_sa = ifa.netmask orelse continue;
        if (addr.family != std.posix.AF.INET) continue;
        const sin: *const std.posix.sockaddr.in = @ptrCast(@alignCast(addr));
        const msin: *const std.posix.sockaddr.in = @ptrCast(@alignCast(mask_sa));
        const mask = msin.addr;
        if (mask == 0) continue;
        if ((sin.addr & mask) != (peer_u32 & mask)) continue;
        const name_span = std.mem.span(ifa.name);
        const copy_len = @min(name_span.len, out_name.len);
        @memset(out_name, 0);
        @memcpy(out_name[0..copy_len], name_span[0..copy_len]);
        return .{ .len = copy_len, .local_ip = @bitCast(sin.addr) };
    }
    return null;
}

pub fn configureTcpSocket(fd: std.c.fd_t) struct { sndbuf: i32, rcvbuf: i32 } {
    const one: c_int = 1;
    _ = std.c.setsockopt(fd, std.posix.IPPROTO.TCP, std.posix.TCP.NODELAY, &one, @sizeOf(c_int));
    if (is_macos) _ = std.c.setsockopt(fd, std.posix.SOL.SOCKET, DARWIN_SO_NOSIGPIPE, &one, @sizeOf(c_int));
    const want: c_int = SOCKET_BUF_BYTES;
    _ = std.c.setsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.SNDBUF, &want, @sizeOf(c_int));
    _ = std.c.setsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.RCVBUF, &want, @sizeOf(c_int));
    var snd: c_int = 0;
    var rcv: c_int = 0;
    var slen: std.posix.socklen_t = @sizeOf(c_int);
    _ = std.c.getsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.SNDBUF, &snd, &slen);
    slen = @sizeOf(c_int);
    _ = std.c.getsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.RCVBUF, &rcv, &slen);
    return .{ .sndbuf = snd, .rcvbuf = rcv };
}

pub fn bindAndConnectPeer(peer_ip: [4]u8, port: u16, timeout_ms: i32) !BoundSocketInfo {
    const fd = std.c.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, 0);
    if (fd < 0) return error.SocketCreateFailed;
    errdefer _ = std.c.close(fd);

    const bufs = configureTcpSocket(fd);
    var info: BoundSocketInfo = .{
        .fd = fd,
        .peer_ip = peer_ip,
        .sndbuf = bufs.sndbuf,
        .rcvbuf = bufs.rcvbuf,
    };

    if (findSubnetInterface(peer_ip, &info.iface_buf)) |found| {
        info.iface_len = found.len;
        info.local_ip = found.local_ip;
        if (is_macos) {
            var zname: [17]u8 = @splat(0);
            @memcpy(zname[0..found.len], info.iface_buf[0..found.len]);
            const idx: c_uint = if_nametoindex(@ptrCast(&zname));
            if (idx != 0) _ = std.c.setsockopt(fd, std.posix.IPPROTO.IP, DARWIN_IP_BOUND_IF, &idx, @sizeOf(c_uint));
        }
        var local_sa: std.posix.sockaddr.in = .{ .port = 0, .addr = @bitCast(found.local_ip) };
        if (std.c.bind(fd, @ptrCast(&local_sa), @sizeOf(std.posix.sockaddr.in)) != 0) return error.BindToSubnetFailed;
        info.mtu = queryInterfaceMtu(info.iface());
    } else if (peer_ip[0] != 127) {
        return error.SubnetInterfaceNotFound;
    }

    const nonblock: c_int = @bitCast(std.c.O{ .NONBLOCK = true });
    const flags = std.c.fcntl(fd, std.c.F.GETFL);
    _ = std.c.fcntl(fd, std.c.F.SETFL, flags | nonblock);
    var remote_sa: std.posix.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, port), .addr = @bitCast(peer_ip) };
    if (std.c.connect(fd, @ptrCast(&remote_sa), @sizeOf(std.posix.sockaddr.in)) != 0) {
        const eno = std.c._errno().*;
        if (eno != @intFromEnum(std.c.E.INPROGRESS)) return error.ConnectFailed;
        var pfd = [_]std.posix.pollfd{.{ .fd = fd, .events = std.posix.POLL.OUT, .revents = 0 }};
        if ((std.posix.poll(&pfd, timeout_ms) catch 0) == 0) return error.ConnectTimeout;
        var so_err: c_int = 0;
        var so_len: std.posix.socklen_t = @sizeOf(c_int);
        if (std.c.getsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.ERROR, &so_err, &so_len) != 0 or so_err != 0) return error.ConnectFailed;
    }
    _ = std.c.fcntl(fd, std.c.F.SETFL, flags);
    return info;
}

pub fn listenTcp(bind_ip: [4]u8, port: u16) !struct { fd: std.c.fd_t, port: u16 } {
    const fd = std.c.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, 0);
    if (fd < 0) return error.SocketCreateFailed;
    errdefer _ = std.c.close(fd);
    const one: c_int = 1;
    _ = std.c.setsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.REUSEADDR, &one, @sizeOf(c_int));
    _ = configureTcpSocket(fd);
    var sa: std.posix.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, port), .addr = @bitCast(bind_ip) };
    if (std.c.bind(fd, @ptrCast(&sa), @sizeOf(std.posix.sockaddr.in)) != 0) return error.BindFailed;
    if (std.c.listen(fd, 8) != 0) return error.ListenFailed;
    var bound_sa: std.posix.sockaddr.in = undefined;
    var slen: std.posix.socklen_t = @sizeOf(std.posix.sockaddr.in);
    if (std.c.getsockname(fd, @ptrCast(&bound_sa), &slen) != 0) return error.GetsocknameFailed;
    return .{ .fd = fd, .port = std.mem.bigToNative(u16, bound_sa.port) };
}

pub fn writeFull(fd: std.c.fd_t, buf: []const u8) !void {
    var off: usize = 0;
    while (off < buf.len) {
        const chunk = @min(buf.len - off, 16 * 1024 * 1024);
        const n = std.c.write(fd, buf.ptr + off, chunk);
        if (n < 0 and std.c._errno().* == @backingInt(std.c.E.INTR)) continue;
        if (n <= 0) return error.TcpWriteFailed;
        off += @intCast(n);
    }
}

pub fn readFull(fd: std.c.fd_t, buf: []u8, timeout_ms: i32) !void {
    var off: usize = 0;
    while (off < buf.len) {
        if (timeout_ms > 0) {
            var pfd = [_]std.posix.pollfd{.{ .fd = fd, .events = std.posix.POLL.IN, .revents = 0 }};
            if ((std.posix.poll(&pfd, timeout_ms) catch 0) == 0) return error.TcpReadTimeout;
        }
        const chunk = @min(buf.len - off, 16 * 1024 * 1024);
        const n = std.c.read(fd, buf.ptr + off, chunk);
        if (n < 0 and std.c._errno().* == @backingInt(std.c.E.INTR)) continue;
        if (n <= 0) return error.TcpReadFailed;
        off += @intCast(n);
    }
}

const WIRE_MAGIC: u32 = 0x5050_3735;
const CMD_PING: u32 = 1;
pub const CMD_ONE_WAY: u32 = 2;
const CMD_PIPELINED: u32 = 3;
const CMD_KV_PREPARE: u32 = 4;
const CMD_KV_STREAM: u32 = 5;
const CMD_DONE: u32 = 9;

pub const FrameHdr = extern struct {
    magic: u32 = WIRE_MAGIC,
    cmd: u32,
    bytes: u64 = 0,
    count: u64 = 0,
    arg0: u64 = 0,
    arg1: u64 = 0,
};

pub const FrameAck = extern struct {
    magic: u32 = WIRE_MAGIC,
    status: u32 = 0,
    t0_us: u64 = 0,
    t1_us: u64 = 0,
};

const KvPrepareSpec = extern struct {
    n_snap: u64,
    chunk_rows: u64,
    is_drain: u64,
};

const RecvKvSlots = struct {
    allocated: bool = false,
    l8_comp: dsv41.Rows = .{ .width = 512, .dtype = .bfloat16 },
    l8_idx: dsv41.Rows = .{ .width = 128, .dtype = .bfloat16 },
    l14_comp: dsv41.Rows = .{ .width = 512, .dtype = .bfloat16 },
    l14_idx: dsv41.Rows = .{ .width = 128, .dtype = .bfloat16 },
    l20_comp: dsv41.Rows = .{ .width = 512, .dtype = .bfloat16 },
    l20_idx: dsv41.Rows = .{ .width = 128, .dtype = .bfloat16 },
    r2_rows: usize = 0,
    r1_rows: usize = 0,

    fn deinit(self: *RecvKvSlots) void {
        if (!self.allocated) return;
        dsv41.deinitRowsForBench(&self.l8_comp);
        dsv41.deinitRowsForBench(&self.l8_idx);
        dsv41.deinitRowsForBench(&self.l14_comp);
        dsv41.deinitRowsForBench(&self.l14_idx);
        dsv41.deinitRowsForBench(&self.l20_comp);
        dsv41.deinitRowsForBench(&self.l20_idx);
        self.* = .{};
    }

    fn prepare(self: *RecvKvSlots, r2_rows: usize, r1_rows: usize, s: mlx.mlx_stream) !void {
        self.deinit();
        if (r2_rows == 0 or r1_rows == 0) return error.InvalidKvRows;
        self.l8_comp = try dsv41.importedRowsForBench(r2_rows, 512, .bfloat16, s);
        errdefer self.deinit();
        self.l8_idx = try dsv41.importedRowsForBench(r2_rows, 128, .bfloat16, s);
        self.l14_comp = try dsv41.importedRowsForBench(r2_rows, 512, .bfloat16, s);
        self.l14_idx = try dsv41.importedRowsForBench(r2_rows, 128, .bfloat16, s);
        self.l20_comp = try dsv41.importedRowsForBench(r1_rows, 512, .bfloat16, s);
        self.l20_idx = try dsv41.importedRowsForBench(r1_rows, 128, .bfloat16, s);
        self.r2_rows = r2_rows;
        self.r1_rows = r1_rows;
        self.allocated = true;
    }

    fn importFromStaging(self: *RecvKvSlots, stage: []const u8, s: mlx.mlx_stream) !void {
        if (!self.allocated) return error.KvSlotsNotPrepared;
        const r2 = self.r2_rows;
        const r1 = self.r1_rows;
        const specs = [_]struct { rows: *dsv41.Rows, n: usize, w: usize }{
            .{ .rows = &self.l8_comp, .n = r2, .w = 512 },
            .{ .rows = &self.l8_idx, .n = r2, .w = 128 },
            .{ .rows = &self.l14_comp, .n = r2, .w = 512 },
            .{ .rows = &self.l14_idx, .n = r2, .w = 128 },
            .{ .rows = &self.l20_comp, .n = r1, .w = 512 },
            .{ .rows = &self.l20_idx, .n = r1, .w = 128 },
        };
        var off: usize = 0;
        for (specs) |sp| {
            const b = sp.n * sp.w * 2;
            if (off + b > stage.len) return error.StagingShort;
            const x = mlx.mlx_array_new_data(stage.ptr + off, &[_]c_int{ @intCast(sp.n), @intCast(sp.w) }, 2, .bfloat16);
            defer _ = mlx.mlx_array_free(x);
            sp.rows.used = sp.n;
            try dsv41.fillImportedRowsForBench(sp.rows, x, s);
            off += b;
        }
    }
};

pub fn serveProbeConn(conn_fd: std.c.fd_t, gpa: std.mem.Allocator) !void {
    _ = configureTcpSocket(conn_fd);
    const s = mlx.gpuStream();
    var staging: []u8 = &.{};
    defer if (staging.len > 0) gpa.free(staging);
    var slots_in: [2][]u8 = .{ &.{}, &.{} };
    defer for (&slots_in) |*sl| if (sl.len > 0) gpa.free(sl.*);
    var kv_slots: RecvKvSlots = .{};
    defer kv_slots.deinit();

    while (true) {
        var hdr: FrameHdr = undefined;
        try readFull(conn_fd, std.mem.asBytes(&hdr), 300_000);
        if (hdr.magic != WIRE_MAGIC) return error.BadMagic;
        switch (hdr.cmd) {
            CMD_PING => {
                const ack: FrameAck = .{};
                try writeFull(conn_fd, std.mem.asBytes(&ack));
            },
            CMD_ONE_WAY => {
                const nbytes: usize = @intCast(hdr.bytes);
                if (staging.len < nbytes) {
                    if (staging.len > 0) gpa.free(staging);
                    staging = try gpa.alloc(u8, nbytes);
                }
                const ready: [1]u8 = .{0xA5};
                try writeFull(conn_fd, &ready);
                const t_rx0 = monoUs();
                try readFull(conn_fd, staging[0..nbytes], 60_000);
                const rx_us = monoUs() - t_rx0;
                var uma_us: i64 = 0;
                if (hdr.arg0 != 0 and nbytes >= 2) {
                    const t_u0 = monoUs();
                    const elems: c_int = @intCast(nbytes / 2);
                    const arr = mlx.mlx_array_new_data(staging.ptr, &[_]c_int{elems}, 1, .bfloat16);
                    _ = mlx.mlx_array_free(arr);
                    uma_us = monoUs() - t_u0;
                }
                const ack: FrameAck = .{ .t0_us = @intCast(@max(rx_us, 0)), .t1_us = @intCast(@max(uma_us, 0)) };
                try writeFull(conn_fd, std.mem.asBytes(&ack));
            },
            CMD_PIPELINED => {
                const nbytes: usize = @intCast(hdr.bytes);
                const count: usize = @intCast(hdr.count);
                for (&slots_in) |*sl| {
                    if (sl.len < nbytes) {
                        if (sl.len > 0) gpa.free(sl.*);
                        sl.* = try gpa.alloc(u8, nbytes);
                    }
                }
                const ready: [1]u8 = .{0xA5};
                try writeFull(conn_fd, &ready);
                const t0 = monoUs();
                var uma_total: i64 = 0;
                for (0..count) |c| {
                    const dst = slots_in[c % 2][0..nbytes];
                    try readFull(conn_fd, dst, 60_000);
                    if (hdr.arg0 != 0 and nbytes >= 2) {
                        const tu = monoUs();
                        const elems: c_int = @intCast(nbytes / 2);
                        const arr = mlx.mlx_array_new_data(dst.ptr, &[_]c_int{elems}, 1, .bfloat16);
                        _ = mlx.mlx_array_free(arr);
                        uma_total += monoUs() - tu;
                    }
                }
                const wall_us = monoUs() - t0;
                const ack: FrameAck = .{ .t0_us = @intCast(@max(wall_us, 0)), .t1_us = @intCast(@max(uma_total, 0)) };
                try writeFull(conn_fd, std.mem.asBytes(&ack));
            },
            CMD_KV_PREPARE => {
                var spec: KvPrepareSpec = undefined;
                try readFull(conn_fd, std.mem.asBytes(&spec), 30_000);
                const r2: usize = if (spec.is_drain != 0) @intCast(spec.chunk_rows / 2) else @intCast(spec.n_snap / 2);
                const r1: usize = if (spec.is_drain != 0) @intCast(spec.chunk_rows) else @intCast(spec.n_snap);
                try kv_slots.prepare(r2, r1, s);
                const ack: FrameAck = .{};
                try writeFull(conn_fd, std.mem.asBytes(&ack));
            },
            CMD_KV_STREAM => {
                const nbytes: usize = @intCast(hdr.bytes);
                if (staging.len < nbytes) {
                    if (staging.len > 0) gpa.free(staging);
                    staging = try gpa.alloc(u8, nbytes);
                }
                const ready: [1]u8 = .{0xA5};
                try writeFull(conn_fd, &ready);
                const t_rx0 = monoUs();
                try readFull(conn_fd, staging[0..nbytes], 60_000);
                const tcp_us = monoUs() - t_rx0;
                const t_fill0 = monoUs();
                try kv_slots.importFromStaging(staging[0..nbytes], s);
                const fill_us = monoUs() - t_fill0;
                const ack: FrameAck = .{ .t0_us = @intCast(@max(tcp_us, 0)), .t1_us = @intCast(@max(fill_us, 0)) };
                try writeFull(conn_fd, std.mem.asBytes(&ack));
            },
            CMD_DONE => {
                const ack: FrameAck = .{};
                try writeFull(conn_fd, std.mem.asBytes(&ack));
                return;
            },
            else => return error.UnknownCommand,
        }
    }
}

pub fn sendDone(fd: std.c.fd_t) void {
    const done_hdr: FrameHdr = .{ .cmd = CMD_DONE };
    writeFull(fd, std.mem.asBytes(&done_hdr)) catch return;
    var ack: FrameAck = undefined;
    readFull(fd, std.mem.asBytes(&ack), 5_000) catch {};
}

pub fn measureHalfPingUs(fd: std.c.fd_t) !f64 {
    var samples: [11]f64 = undefined;
    for (0..samples.len + 2) |i| {
        const hdr: FrameHdr = .{ .cmd = CMD_PING };
        var ack: FrameAck = undefined;
        const t0 = monoUs();
        try writeFull(fd, std.mem.asBytes(&hdr));
        try readFull(fd, std.mem.asBytes(&ack), 10_000);
        const dt = @as(f64, @floatFromInt(@max(monoUs() - t0, 0))) * 0.5;
        if (i >= 2) samples[i - 2] = dt;
    }
    std.mem.sort(f64, &samples, {}, std.sort.asc(f64));
    return samples[samples.len / 2];
}

pub fn statsOf(samples: []f64) struct { mean: f64, median: f64, best: f64, p95: f64 } {
    std.mem.sort(f64, samples, {}, std.sort.asc(f64));
    var sum: f64 = 0;
    for (samples) |v| sum += v;
    const n = samples.len;
    const p95_idx = @min(n - 1, (n * 95) / 100);
    return .{
        .mean = sum / @as(f64, @floatFromInt(n)),
        .median = samples[n / 2],
        .best = samples[0],
        .p95 = samples[p95_idx],
    };
}

pub fn runTcpHopSuite(
    sock: *const BoundSocketInfo,
    gpa: std.mem.Allocator,
    json_out: *const pp_step1.JsonOut,
    widths: []const usize,
    warmup: usize,
    iters: usize,
) !void {
    const s = mlx.gpuStream();
    const half_ping = try measureHalfPingUs(sock.fd);
    var line_buf: [768]u8 = undefined;

    for (widths) |c_rows| {
        for ([_]pp_step1.SeamKind{ .pair_cut, .mid_group }) |seam| {
            const mid = seam == .mid_group;
            const nbytes_u64 = pp_step1.seamPayloadBytes(c_rows, mid);
            const nbytes: usize = @intCast(nbytes_u64);

            const elems: c_int = @intCast(nbytes / 2);
            var gpu_src = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_ones(&gpu_src, &[_]c_int{elems}, 1, .bfloat16, s));
            defer _ = mlx.mlx_array_free(gpu_src);
            try mlx.check(mlx.mlx_array_eval(gpu_src));
            const raw_ptr: [*]const u8 = @ptrCast(mlx.mlx_array_data_bfloat16(gpu_src) orelse return error.NoArrayData);

            const out_slots: [2][]u8 = .{ try gpa.alloc(u8, nbytes), try gpa.alloc(u8, nbytes) };
            defer for (out_slots) |sl| gpa.free(sl);

            const samples = try gpa.alloc(f64, iters);
            defer gpa.free(samples);
            var memcpy_sum: f64 = 0;
            var mlx_new_sum: f64 = 0;

            for (0..warmup + iters) |rep| {
                const hdr: FrameHdr = .{ .cmd = CMD_ONE_WAY, .bytes = nbytes_u64, .arg0 = 1 };
                try writeFull(sock.fd, std.mem.asBytes(&hdr));
                var ready: [1]u8 = undefined;
                try readFull(sock.fd, &ready, 10_000);

                const t_start = monoUs();
                const dst = out_slots[rep % 2];
                @memcpy(dst, raw_ptr[0..nbytes]);
                const t_copied = monoUs();
                try writeFull(sock.fd, dst);
                var ack: FrameAck = undefined;
                try readFull(sock.fd, std.mem.asBytes(&ack), 60_000);
                const t_done = monoUs();

                if (rep >= warmup) {
                    samples[rep - warmup] = @max(0.0, @as(f64, @floatFromInt(t_done - t_start)) - half_ping);
                    memcpy_sum += @floatFromInt(@max(t_copied - t_start, 0));
                    mlx_new_sum += @floatFromInt(ack.t1_us);
                }
            }

            const st_one = statsOf(samples);
            const n_f = @as(f64, @floatFromInt(iters));
            json_out.writeLine(try pp_step1.formatTcpHopJson(
                &line_buf, "one_way", c_rows, seam, nbytes_u64, iters,
                st_one.mean, st_one.median, st_one.best, st_one.p95,
                memcpy_sum / n_f, mlx_new_sum / n_f, sock.iface(),
                sock.local_ip, sock.peer_ip, sock.mtu, sock.sndbuf, sock.rcvbuf,
            ));

            @memcpy(out_slots[0], raw_ptr[0..nbytes]);
            @memcpy(out_slots[1], raw_ptr[0..nbytes]);
            {
                const w_hdr: FrameHdr = .{ .cmd = CMD_PIPELINED, .bytes = nbytes_u64, .count = warmup, .arg0 = 1 };
                try writeFull(sock.fd, std.mem.asBytes(&w_hdr));
                var ready: [1]u8 = undefined;
                try readFull(sock.fd, &ready, 10_000);
                for (0..warmup) |k| try writeFull(sock.fd, out_slots[k % 2]);
                var w_ack: FrameAck = undefined;
                try readFull(sock.fd, std.mem.asBytes(&w_ack), 120_000);
            }

            const p_hdr: FrameHdr = .{ .cmd = CMD_PIPELINED, .bytes = nbytes_u64, .count = iters, .arg0 = 1 };
            try writeFull(sock.fd, std.mem.asBytes(&p_hdr));
            var p_ready: [1]u8 = undefined;
            try readFull(sock.fd, &p_ready, 10_000);
            const tp0 = monoUs();
            for (0..iters) |k| try writeFull(sock.fd, out_slots[k % 2]);
            var p_ack: FrameAck = undefined;
            try readFull(sock.fd, std.mem.asBytes(&p_ack), 300_000);
            const pipe_wall = @max(0.0, @as(f64, @floatFromInt(monoUs() - tp0)) - half_ping);
            const pipe_per_hop = pipe_wall / n_f;
            const pipe_uma = @as(f64, @floatFromInt(p_ack.t1_us)) / n_f;
            json_out.writeLine(try pp_step1.formatTcpHopJson(
                &line_buf, "pipelined", c_rows, seam, nbytes_u64, iters,
                pipe_per_hop, pipe_per_hop, pipe_per_hop, pipe_per_hop,
                memcpy_sum / n_f, pipe_uma, sock.iface(),
                sock.local_ip, sock.peer_ip, sock.mtu, sock.sndbuf, sock.rcvbuf,
            ));
        }
    }
}

pub fn runKvStreamSuite(
    sock: *const BoundSocketInfo,
    gpa: std.mem.Allocator,
    json_out: *const pp_step1.JsonOut,
    prompt_ns: []const usize,
    chunk_widths: []const usize,
    warmup: usize,
    iters: usize,
) !void {
    var line_buf: [512]u8 = undefined;

    for (prompt_ns) |prompt_n| {
        const n_snap = prompt_n - 1;
        const non_s0 = pp_step1.nonStage0KvBytes(n_snap);
        const wire_bytes: usize = @intCast(non_s0.total_bytes);
        const payload = try gpa.alloc(u8, wire_bytes);
        defer gpa.free(payload);
        @memset(payload, 0x2A);

        const prep_hdr: FrameHdr = .{ .cmd = CMD_KV_PREPARE, .bytes = @sizeOf(KvPrepareSpec) };
        const prep: KvPrepareSpec = .{ .n_snap = n_snap, .chunk_rows = 0, .is_drain = 0 };
        try writeFull(sock.fd, std.mem.asBytes(&prep_hdr));
        try writeFull(sock.fd, std.mem.asBytes(&prep));
        var prep_ack: FrameAck = undefined;
        try readFull(sock.fd, std.mem.asBytes(&prep_ack), 30_000);

        const tcp_samples = try gpa.alloc(f64, iters);
        defer gpa.free(tcp_samples);
        const fill_samples = try gpa.alloc(f64, iters);
        defer gpa.free(fill_samples);
        const tot_samples = try gpa.alloc(f64, iters);
        defer gpa.free(tot_samples);

        for (0..warmup + iters) |rep| {
            const hdr: FrameHdr = .{ .cmd = CMD_KV_STREAM, .bytes = non_s0.total_bytes };
            try writeFull(sock.fd, std.mem.asBytes(&hdr));
            var ready: [1]u8 = undefined;
            try readFull(sock.fd, &ready, 10_000);
            const t0 = monoUs();
            try writeFull(sock.fd, payload);
            var ack: FrameAck = undefined;
            try readFull(sock.fd, std.mem.asBytes(&ack), 60_000);
            const tot = @as(f64, @floatFromInt(@max(monoUs() - t0, 0)));
            if (rep >= warmup) {
                tcp_samples[rep - warmup] = @floatFromInt(ack.t0_us);
                fill_samples[rep - warmup] = @floatFromInt(ack.t1_us);
                tot_samples[rep - warmup] = tot;
            }
        }

        json_out.writeLine(try pp_step1.formatKvStreamJson(
            &line_buf, "full_non_stage0", prompt_n, 0,
            non_s0.total_bytes, non_s0.rows_bytes, iters,
            statsOf(tcp_samples).median, statsOf(fill_samples).median, statsOf(tot_samples).median,
        ));
    }

    for (chunk_widths) |c_rows| {
        const delta_rows: u64 = 2 * 1280 * @as(u64, c_rows / 2) + 1280 * @as(u64, c_rows);
        const drain_wire = delta_rows + 15 * pp_step1.WIN_RING_BYTES + 2 * 2 * 512 * 4 + pp_step1.TAIL_128_WIRE_BYTES;
        const payload = try gpa.alloc(u8, @intCast(drain_wire));
        defer gpa.free(payload);
        @memset(payload, 0x1F);

        const prep_hdr: FrameHdr = .{ .cmd = CMD_KV_PREPARE, .bytes = @sizeOf(KvPrepareSpec) };
        const prep: KvPrepareSpec = .{ .n_snap = 0, .chunk_rows = c_rows, .is_drain = 1 };
        try writeFull(sock.fd, std.mem.asBytes(&prep_hdr));
        try writeFull(sock.fd, std.mem.asBytes(&prep));
        var prep_ack: FrameAck = undefined;
        try readFull(sock.fd, std.mem.asBytes(&prep_ack), 30_000);

        const tcp_samples = try gpa.alloc(f64, iters);
        defer gpa.free(tcp_samples);
        const fill_samples = try gpa.alloc(f64, iters);
        defer gpa.free(fill_samples);
        const tot_samples = try gpa.alloc(f64, iters);
        defer gpa.free(tot_samples);

        for (0..warmup + iters) |rep| {
            const hdr: FrameHdr = .{ .cmd = CMD_KV_STREAM, .bytes = drain_wire };
            try writeFull(sock.fd, std.mem.asBytes(&hdr));
            var ready: [1]u8 = undefined;
            try readFull(sock.fd, &ready, 10_000);
            const t0 = monoUs();
            try writeFull(sock.fd, payload);
            var ack: FrameAck = undefined;
            try readFull(sock.fd, std.mem.asBytes(&ack), 60_000);
            const tot = @as(f64, @floatFromInt(@max(monoUs() - t0, 0)));
            if (rep >= warmup) {
                tcp_samples[rep - warmup] = @floatFromInt(ack.t0_us);
                fill_samples[rep - warmup] = @floatFromInt(ack.t1_us);
                tot_samples[rep - warmup] = tot;
            }
        }

        json_out.writeLine(try pp_step1.formatKvStreamJson(
            &line_buf, "final_chunk_drain", 0, c_rows,
            drain_wire, delta_rows, iters,
            statsOf(tcp_samples).median, statsOf(fill_samples).median, statsOf(tot_samples).median,
        ));
    }
}

// ── Tests ──

const testing = std.testing;

test "pp_step1_tcp: loopback getifaddrs bindToSubnet, MTU check, TCP hop and KV fillImportedRows roundtrip" {
    if (is_windows or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;

    var ifname: [16]u8 = @splat(0);
    const found = findSubnetInterface(.{ 127, 0, 0, 1 }, &ifname) orelse return error.LoopbackNotFound;
    try testing.expect(found.len > 0);
    try testing.expectEqual([4]u8{ 127, 0, 0, 1 }, found.local_ip);
    try testing.expect(queryInterfaceMtu(ifname[0..found.len]) > 0);

    const srv = try listenTcp(.{ 127, 0, 0, 1 }, 0);
    defer _ = std.c.close(srv.fd);

    const ServerCtx = struct {
        listen_fd: std.c.fd_t,
        err: ?anyerror = null,
        fn run(ctx: *@This()) void {
            const cfd = std.c.accept(ctx.listen_fd, null, null);
            if (cfd < 0) {
                ctx.err = error.AcceptFailed;
                return;
            }
            defer _ = std.c.close(cfd);
            serveProbeConn(cfd, std.heap.c_allocator) catch |e| {
                ctx.err = e;
            };
        }
    };

    var sctx: ServerCtx = .{ .listen_fd = srv.fd };
    const sth = try std.Thread.spawn(.{ .stack_size = 1 << 20 }, ServerCtx.run, .{&sctx});
    defer sth.join();

    const sock = try bindAndConnectPeer(.{ 127, 0, 0, 1 }, srv.port, 3000);
    defer _ = std.c.close(sock.fd);
    try testing.expect(sock.mtu > 0);

    const null_out: pp_step1.JsonOut = .{ .fd = -1 };
    try runTcpHopSuite(&sock, gpa, &null_out, &.{64}, 1, 2);
    try runKvStreamSuite(&sock, gpa, &null_out, &.{128}, &.{64}, 1, 2);
    sendDone(sock.fd);
    if (sctx.err) |e| return e;
}
