//! Configure-time facts every graph reads: the version stamps and engine pins checked into the tree, the macOS SDK and
//! webp the macOS graph links, and the checks that a staged dependency is there before anything compiles.

const std = @import("std");

/// Where the macOS graph finds the SDK's frameworks and webp.
pub const Host = struct {
    /// The macOS SDK's frameworks dir: setting any non-default target field disables Zig's native SDK detection.
    frameworks: ?[]const u8,
    /// `-Dwebp-prefix`: a copy of webp's include/ and lib/ (a box without Homebrew), else null.
    webp_prefix: ?[]const u8,
    /// webp's prefix: the copy, else Homebrew's.
    brew_prefix: []const u8,

    /// Resolves the SDK and webp, and on a macOS target checks webp's version and the staged mlx before any step runs.
    pub fn init(b: *std.Build, target: std.Build.ResolvedTarget) Host {
        const frameworks = macosFrameworks(b, target);
        // Unset: Homebrew's prefix, its version read from `brew`. Set: a directory holding a copy of the formula's
        // include/ and lib/, its version read from the copy's lib/pkgconfig.
        const webp_prefix = b.option([]const u8, "webp-prefix", "Directory holding a copy of webp's include/ and lib/ (unset: Homebrew)");
        if (target.result.os.tag == .macos) {
            if (webp_prefix) |dir| verifyCopiedDeps(b, dir) else verifyBrewDeps(b);
            verifyMlxStage(b);
        }
        return .{ .frameworks = frameworks, .webp_prefix = webp_prefix, .brew_prefix = webp_prefix orelse "/opt/homebrew" };
    }
};

fn macosFrameworks(b: *std.Build, target: std.Build.ResolvedTarget) ?[]const u8 {
    if (target.result.os.tag != .macos) return null;
    var code: u8 = undefined;
    const stdout = b.runAllowFail(
        &.{ "xcrun", "--sdk", "macosx", "--show-sdk-path" },
        &code,
        .inherit,
    ) catch return null;
    const sdk = std.mem.trim(u8, stdout, " \n\r\t");
    if (sdk.len == 0) return null;
    return b.fmt("{s}/System/Library/Frameworks", .{sdk});
}

/// Build identity (`build_options.build_id`): HEAD SHA, `<sha>-dirty` on uncommitted tracked/untracked
/// repo changes, staged changes (including staged submodule pointers), dirty non-MLX submodules, moved
/// submodule HEAD pointers, or unpatched/extra edits inside `lib/mlx-src` / `lib/mlxc-src`; or `"unknown"`
/// when git metadata is unavailable. Excludes known checkout build artifacts (`cuda-keyring_*.deb`, `dist/`)
/// and exact working-tree patches applied by `scripts/build-mlx.sh` (`scripts/build-mlx-patches.txt`).
pub fn readBuildId(b: *std.Build, git_sha: []const u8) []const u8 {
    b.graph.poisonCache();
    return resolveBuildId(b.allocator, b.graph.io, buildRootHandle(b), git_sha) catch "unknown";
}

fn runGitQuery(allocator: std.mem.Allocator, io: std.Io, root: std.Io.Dir, argv: []const []const u8, stdout_limit: usize) ?[]u8 {
    const res = std.process.run(allocator, io, .{
        .argv = argv,
        .cwd = .{ .dir = root },
        .stdout_limit = .limited(stdout_limit),
        .stderr_limit = .limited(4096),
    }) catch return null;
    allocator.free(res.stderr);
    switch (res.term) {
        .exited => |code| if (code != 0) {
            allocator.free(res.stdout);
            return null;
        },
        else => {
            allocator.free(res.stdout);
            return null;
        },
    }
    return res.stdout;
}

fn runGitOk(allocator: std.mem.Allocator, io: std.Io, root: std.Io.Dir, argv: []const []const u8) bool {
    const res = std.process.run(allocator, io, .{
        .argv = argv,
        .cwd = .{ .dir = root },
        .stdout_limit = .limited(4096),
        .stderr_limit = .limited(4096),
    }) catch return false;
    allocator.free(res.stdout);
    allocator.free(res.stderr);
    return switch (res.term) {
        .exited => |code| code == 0,
        else => false,
    };
}

fn isMlxSubmodule(path: []const u8) bool {
    return std.mem.eql(u8, path, "lib/mlx-src") or std.mem.eql(u8, path, "lib/mlxc-src");
}

fn mlxSubmoduleMatchesPatches(
    allocator: std.mem.Allocator,
    io: std.Io,
    root: std.Io.Dir,
    sub_dir: std.Io.Dir,
    subpath: []const u8,
    changed_names: []const u8,
) bool {
    const list_bytes = root.readFileAlloc(io, "scripts/build-mlx-patches.txt", allocator, .limited(64 * 1024)) catch return false;
    defer allocator.free(list_bytes);

    var patches: std.ArrayList([]const u8) = .empty;
    defer patches.deinit(allocator);
    var lines = std.mem.tokenizeScalar(u8, list_bytes, '\n');
    while (lines.next()) |line| {
        var fields = std.mem.tokenizeAny(u8, line, " \t\r");
        const target_sub = fields.next() orelse continue;
        if (target_sub[0] == '#') continue;
        const patch_rel = fields.next() orelse return false;
        if (std.mem.eql(u8, target_sub, subpath)) patches.append(allocator, patch_rel) catch return false;
    }
    if (patches.items.len == 0) return false;

    const S = struct {
        var seq: std.atomic.Value(u64) = .init(0);
    };
    const id = S.seq.fetchAdd(1, .monotonic);
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_abs = root_buf[0 .. root.realPath(io, &root_buf) catch return false];
    const tmp_name = std.fmt.allocPrint(allocator, "msv-bid-{x}-{d}", .{ std.hash.Wyhash.hash(0, root_abs), id }) catch return false;
    defer allocator.free(tmp_name);
    const tmp_base = std.c.getenv("TMPDIR");
    const tmp_parent = if (tmp_base) |ptr| std.mem.trimEnd(u8, std.mem.span(ptr), "/") else "/tmp";

    var parent_dir = std.Io.Dir.openDirAbsolute(io, tmp_parent, .{}) catch return false;
    defer parent_dir.close(io);
    parent_dir.createDirPath(io, tmp_name) catch return false;
    defer parent_dir.deleteTree(io, tmp_name) catch {};
    var tmp_dir = parent_dir.openDir(io, tmp_name, .{}) catch return false;
    defer tmp_dir.close(io);

    if (!runGitOk(allocator, io, tmp_dir, &.{ "git", "init" })) return false;

    var files: std.ArrayList([]const u8) = .empty;
    defer files.deinit(allocator);
    var file_lines = std.mem.tokenizeScalar(u8, changed_names, '\n');
    while (file_lines.next()) |raw| {
        const rel = std.mem.trim(u8, raw, " \t\r");
        if (rel.len == 0) continue;
        files.append(allocator, rel) catch return false;
        const spec = std.fmt.allocPrint(allocator, "HEAD:{s}", .{rel}) catch return false;
        defer allocator.free(spec);
        const orig = runGitQuery(allocator, io, sub_dir, &.{ "git", "--no-optional-locks", "show", spec }, 16 * 1024 * 1024) orelse return false;
        defer allocator.free(orig);
        if (std.fs.path.dirname(rel)) |dir_part| tmp_dir.createDirPath(io, dir_part) catch return false;
        tmp_dir.writeFile(io, .{ .sub_path = rel, .data = orig }) catch return false;
    }

    for (patches.items) |patch_rel| {
        const patch_abs = std.fmt.allocPrint(allocator, "{s}/{s}", .{ root_abs, patch_rel }) catch return false;
        defer allocator.free(patch_abs);
        if (!runGitOk(allocator, io, tmp_dir, &.{ "git", "apply", "-p1", patch_abs })) return false;
    }

    for (files.items) |rel| {
        const want = tmp_dir.readFileAlloc(io, rel, allocator, .limited(16 * 1024 * 1024)) catch return false;
        defer allocator.free(want);
        const have = sub_dir.readFileAlloc(io, rel, allocator, .limited(16 * 1024 * 1024)) catch return false;
        defer allocator.free(have);
        if (!std.mem.eql(u8, want, have)) return false;
    }
    return true;
}

fn submodulesClean(allocator: std.mem.Allocator, io: std.Io, root: std.Io.Dir) bool {
    const sub_status = runGitQuery(allocator, io, root, &.{
        "git", "--no-optional-locks", "submodule", "status",
    }, 64 * 1024) orelse return false;
    defer allocator.free(sub_status);

    var lines = std.mem.tokenizeScalar(u8, sub_status, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        if (line[0] == '+' or line[0] == 'U') return false;
        if (line[0] == '-') continue;
        const rest = std.mem.trimStart(u8, line[1..], " ");
        var toks = std.mem.tokenizeScalar(u8, rest, ' ');
        _ = toks.next() orelse continue; // sha
        const subpath = toks.next() orelse continue;

        var sub_dir = root.openDir(io, subpath, .{}) catch return false;
        defer sub_dir.close(io);

        if (!isMlxSubmodule(subpath)) {
            const st = runGitQuery(allocator, io, sub_dir, &.{ "git", "--no-optional-locks", "status", "--porcelain" }, 64 * 1024) orelse return false;
            defer allocator.free(st);
            if (st.len != 0) return false;
            continue;
        }

        const untracked = runGitQuery(allocator, io, sub_dir, &.{ "git", "--no-optional-locks", "ls-files", "--others", "--exclude-standard" }, 64 * 1024) orelse return false;
        defer allocator.free(untracked);
        if (untracked.len != 0) return false;

        const staged = runGitQuery(allocator, io, sub_dir, &.{ "git", "--no-optional-locks", "diff", "--cached", "--name-only" }, 64 * 1024) orelse return false;
        defer allocator.free(staged);
        if (staged.len != 0) return false;

        const changed = runGitQuery(allocator, io, sub_dir, &.{ "git", "--no-optional-locks", "diff", "--name-only" }, 64 * 1024) orelse return false;
        defer allocator.free(changed);
        if (changed.len != 0 and !mlxSubmoduleMatchesPatches(allocator, io, root, sub_dir, subpath, changed)) return false;
    }
    return true;
}

fn resolveBuildId(allocator: std.mem.Allocator, io: std.Io, root: std.Io.Dir, git_sha: []const u8) ![]u8 {
    root.access(io, ".git", .{}) catch return allocator.dupe(u8, "unknown");
    const sha: []const u8 = if (isGitSha(git_sha))
        try allocator.dupe(u8, git_sha)
    else blk: {
        const out = runGitQuery(allocator, io, root, &.{ "git", "rev-parse", "--verify", "HEAD" }, 256) orelse return allocator.dupe(u8, "unknown");
        defer allocator.free(out);
        break :blk try allocator.dupe(u8, std.mem.trim(u8, out, " \t\r\n"));
    };
    defer allocator.free(sha);
    if (!isGitSha(sha)) return allocator.dupe(u8, "unknown");

    const status = runGitQuery(allocator, io, root, &.{
        "git", "--no-optional-locks", "status", "--porcelain", "--ignore-submodules", "--", ".", ":(exclude)cuda-keyring_*.deb", ":(exclude)dist",
    }, 1024 * 1024) orelse return allocator.dupe(u8, "unknown");
    defer allocator.free(status);
    if (status.len != 0) return std.fmt.allocPrint(allocator, "{s}-dirty", .{sha});

    if (!runGitOk(allocator, io, root, &.{ "git", "--no-optional-locks", "diff-index", "--cached", "--quiet", "HEAD", "--" })) {
        return std.fmt.allocPrint(allocator, "{s}-dirty", .{sha});
    }

    if (!submodulesClean(allocator, io, root)) return std.fmt.allocPrint(allocator, "{s}-dirty", .{sha});
    return allocator.dupe(u8, sha);
}

fn isGitSha(sha: []const u8) bool {
    if (sha.len != 40 and sha.len != 64) return false;
    for (sha) |c| if (!std.ascii.isHex(c)) return false;
    return true;
}

pub fn buildRootHandle(b: *std.Build) std.Io.Dir {
    return b.root.root_dir.handle;
}

/// The llama.cpp tag staged by scripts/fetch-llama.sh (it writes LLAMA_TAG to
/// `lib/llama/.version`). Read at configure time so a plain `zig build` reports
/// the real tag without app/build.sh having to pass `--llama-tag`. Returns null
/// (→ "unknown") when llama hasn't been fetched yet.
pub fn readLlamaTag(b: *std.Build) ?[]const u8 {
    const bytes = buildRootHandle(b).readFileAlloc(
        b.graph.io,
        "lib/llama/.version",
        b.allocator,
        .limited(256),
    ) catch return null;
    const trimmed = std.mem.trim(u8, bytes, " \t\r\n");
    return if (trimmed.len == 0) null else b.dupe(trimmed);
}

/// `CFBundleShortVersionString` out of the checked-in app/Info.plist — the one
/// place the current CalVer is committed (app/build.sh stamps it on every real
/// build). Read at configure time so a plain `zig build` reports the same
/// version the last shipped build did, instead of a made-up literal.
pub fn readAppVersion(b: *std.Build) ?[]const u8 {
    const bytes = buildRootHandle(b).readFileAlloc(
        b.graph.io,
        "app/Info.plist",
        b.allocator,
        .limited(64 * 1024),
    ) catch return null;
    const key = "<key>CFBundleShortVersionString</key>";
    const at = std.mem.indexOf(u8, bytes, key) orelse return null;
    const open = std.mem.indexOfPos(u8, bytes, at + key.len, "<string>") orelse return null;
    const start = open + "<string>".len;
    const end = std.mem.indexOfPos(u8, bytes, start, "</string>") orelse return null;
    const v = std.mem.trim(u8, bytes[start..end], " \t\r\n");
    return if (v.len > 0) b.dupe(v) else null;
}

/// The ds4 submodule's checked-out commit, short, read without git: its `.git` file names the
/// git dir, whose detached HEAD holds the commit.
pub fn readDs4Commit(b: *std.Build) ?[]const u8 {
    const root = buildRootHandle(b);
    const link = root.readFileAlloc(b.graph.io, "lib/ds4/.git", b.allocator, .limited(512)) catch return null;
    const line = std.mem.trim(u8, link, " \t\r\n");
    const prefix = "gitdir: ";
    if (!std.mem.startsWith(u8, line, prefix)) return null;
    const git_dir = line[prefix.len..];
    const head_path = b.pathJoin(&.{ if (std.fs.path.isAbsolute(git_dir)) git_dir else b.pathJoin(&.{ "lib/ds4", git_dir }), "HEAD" });
    const head = root.readFileAlloc(b.graph.io, head_path, b.allocator, .limited(256)) catch return null;
    const sha = std.mem.trim(u8, head, " \t\r\n");
    if (sha.len < 7 or std.mem.startsWith(u8, sha, "ref:")) return null;
    return b.dupe(sha[0..7]);
}

/// The pinned mlx-c revision from lib/mlx/.version (written by
/// scripts/build-mlx.sh as "mlx=<sha> mlxc=<sha> target=<ver>"), surfaced in
/// `mlx-serve --version`. Returns null (→ "unknown") when not staged yet.
pub fn readMlxcPin(b: *std.Build) ?[]const u8 {
    const bytes = buildRootHandle(b).readFileAlloc(
        b.graph.io,
        "lib/mlx/.version",
        b.allocator,
        .limited(256),
    ) catch return null;
    var it = std.mem.tokenizeScalar(u8, std.mem.trim(u8, bytes, " \t\r\n"), ' ');
    while (it.next()) |tok| {
        if (std.mem.startsWith(u8, tok, "mlxc=")) return b.dupe(tok["mlxc=".len..]);
    }
    return null;
}

/// Module `mlx_stamp`: the staged MLX build's stamp (commits, patch inputs, CUDA arch) as
/// `bytes`, keying the CUDA kernel cache (`mlx.exportPtxCacheDir`); "" when nothing is staged.
/// Copied through the build graph: a configure-time read is cached, so a rebuilt MLX kept the old stamp.
pub fn mlxStampModule(b: *std.Build, stamp_path: []const u8) *std.Build.Module {
    const wf = b.addWriteFiles();
    const src = if (buildRootHandle(b).access(b.graph.io, stamp_path, .{})) |_| blk: {
        _ = wf.addCopyFile(b.path(stamp_path), "version");
        break :blk "pub const bytes = @embedFile(\"version\");\n";
    } else |_| "pub const bytes = \"\";\n";
    return b.createModule(.{ .root_source_file = wf.add("mlx_stamp.zig", src) });
}

/// Configure-time check that scripts/build-mlx.sh has staged the pinned
/// mlx/mlx-c build. Mirrors verifyBrewDeps: fail loudly with the fix, never
/// let the linker produce a confusing -lmlxc error (or silently pick up a
/// leftover brew copy from /opt/homebrew/lib).
fn verifyMlxStage(b: *std.Build) void {
    const stage_ok = blk: {
        buildRootHandle(b).access(b.graph.io, "lib/mlx/lib/libmlxc.dylib", .{}) catch break :blk false;
        buildRootHandle(b).access(b.graph.io, "lib/mlx/lib/mlx.metallib", .{}) catch break :blk false;
        buildRootHandle(b).access(b.graph.io, "lib/mlx/.version", .{}) catch break :blk false;
        break :blk true;
    };
    if (!stage_ok) {
        std.debug.print(
            "\n[mlx-serve] lib/mlx is not staged (self-built mlx + mlx-c). Run:\n" ++
                "  git submodule update --init lib/mlx-src lib/mlxc-src && ./scripts/build-mlx.sh\n\n",
            .{},
        );
        std.process.exit(1);
    }
    const stamp = buildRootHandle(b).readFileAlloc(b.graph.io, "lib/mlx/.version", b.allocator, .limited(256)) catch return;
    b.dependOnFileContents(b.path("lib/mlx/.version"));
    const patch_list = "scripts/build-mlx-patches.txt";
    const inputs = [_]StageInput{
        .{ .key = "mlx=", .path = "lib/mlx-src", .have = submoduleCommit(b, "lib/mlx-src") },
        .{ .key = "mlxc=", .path = "lib/mlxc-src", .have = submoduleCommit(b, "lib/mlxc-src") },
        .{ .key = "patch=", .path = patch_list, .have = patchesSha1(b.allocator, RootReader{ .b = b }, patch_list) },
    };
    if (staleStageInput(stamp, &inputs)) |s| {
        std.debug.print(
            "\n[mlx-serve] lib/mlx was staged for {s}{s} but {s} is at {s}. Run:\n  ./scripts/build-mlx.sh\n\n",
            .{ s.input.key, s.staged, s.input.path, s.input.have.? },
        );
        std.process.exit(1);
    }
}

/// A submodule's checked-out commit, short as scripts/build-mlx.sh stamps it, read without git: its `.git` file names
/// the git dir, whose detached HEAD holds the commit. Null without a checkout or on a branch.
fn submoduleCommit(b: *std.Build, dir: []const u8) ?[]const u8 {
    const root = buildRootHandle(b);
    const link = root.readFileAlloc(b.graph.io, b.pathJoin(&.{ dir, ".git" }), b.allocator, .limited(512)) catch return null;
    const git_dir = std.mem.cutPrefix(u8, std.mem.trim(u8, link, " \t\r\n"), "gitdir: ") orelse return null;
    const head_path = std.fs.path.resolveAlloc(b.allocator, &.{ dir, git_dir, "HEAD" }) catch return null;
    const head = root.readFileAlloc(b.graph.io, head_path, b.allocator, .limited(256)) catch return null;
    b.dependOnFileContents(if (std.fs.path.isAbsolute(head_path)) .{ .cwd_relative = head_path } else b.path(head_path));
    const sha = std.mem.trim(u8, head, " \t\r\n");
    if (sha.len < 12 or std.mem.startsWith(u8, sha, "ref:")) return null;
    return sha[0..12];
}

/// The patches `list` names (`<submodule> <patch>` lines; blank and `#` lines skipped), concatenated in order, as SHA-1
/// short as `cat <patches> | shasum | cut -c1-12` prints it. Null when the list or a patch cannot be read.
fn patchesSha1(a: std.mem.Allocator, reader: anytype, list: []const u8) ?[]const u8 {
    const text = reader.read(list) catch return null;
    var sha1 = std.crypto.hash.Sha1.init(.{});
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| {
        var fields = std.mem.tokenizeAny(u8, line, " \t\r");
        const submodule = fields.next() orelse continue;
        if (submodule[0] == '#') continue;
        sha1.update(reader.read(fields.next() orelse return null) catch return null);
    }
    return a.dupe(u8, std.fmt.bytesToHex(sha1.finalResult(), .lower)[0..12]) catch @panic("OOM");
}

/// Reads a file under the build root and declares it a configure input.
const RootReader = struct {
    b: *std.Build,
    fn read(self: RootReader, path: []const u8) ![]const u8 {
        self.b.dependOnFileContents(self.b.path(path));
        return buildRootHandle(self.b).readFileAlloc(self.b.graph.io, path, self.b.allocator, .limited(1 << 20));
    }
};

/// The build's own tests of this file, on the host.
pub fn addTests(b: *std.Build, step: *std.Build.Step) void {
    const t = b.addTest(.{ .name = "build-config-test", .root_module = b.createModule(.{
        .root_source_file = b.path("build/config.zig"),
        .target = b.graph.host,
    }) });
    step.dependOn(&b.addRunArtifact(t).step);
}

/// One input scripts/build-mlx.sh stamps into lib/mlx/.version: its key, where it lives, and its value now (12 hex),
/// null when this tree cannot tell (no submodule checkout), which leaves it unchecked.
const StageInput = struct { key: []const u8, path: []const u8, have: ?[]const u8 };

/// The first input whose value differs from the one `stamp` records, with the recorded value ("none" when absent).
fn staleStageInput(stamp: []const u8, inputs: []const StageInput) ?struct { input: StageInput, staged: []const u8 } {
    for (inputs) |in| {
        const have = in.have orelse continue;
        var staged: []const u8 = "none";
        var it = std.mem.tokenizeAny(u8, stamp, " \t\r\n");
        while (it.next()) |tok| {
            if (std.mem.cutPrefix(u8, tok, in.key)) |v| staged = v;
        }
        if (!std.mem.eql(u8, staged, have)) return .{ .input = in, .staged = staged };
    }
    return null;
}

/// A Homebrew formula the macOS graph links: its name, the pkg-config file and header a copied prefix must hold, and
/// the version floor.
const BrewDep = struct { name: []const u8, pc: []const u8, header: []const u8, min: std.SemanticVersion };

const required_brew_deps = [_]BrewDep{
    // mlx + mlx-c are NOT brew deps anymore: they are pinned submodules built
    // by scripts/build-mlx.sh (see addMlxLib) so the NAX kernels ship enabled.
    .{ .name = "webp", .pc = "libwebp", .header = "webp/decode.h", .min = .{ .major = 1, .minor = 6, .patch = 0 } },
};

/// `-Dwebp-prefix=<dir>`: the copy must hold each dependency's header and its pkg-config file, whose `Version:` line
/// meets the same floor the Homebrew check applies.
fn verifyCopiedDeps(b: *std.Build, dir: []const u8) void {
    for (required_brew_deps) |dep| {
        const header = b.fmt("{s}/include/{s}", .{ dir, dep.header });
        std.Io.Dir.cwd().access(b.graph.io, header, .{}) catch {
            std.debug.print("\n[mlx-serve] -Dwebp-prefix={s}: {s} not found\n\n", .{ dir, header });
            std.process.exit(1);
        };
        const pc_path = b.fmt("{s}/lib/pkgconfig/{s}.pc", .{ dir, dep.pc });
        const pc = std.Io.Dir.cwd().readFileAlloc(b.graph.io, pc_path, b.allocator, .limited(16 * 1024)) catch {
            std.debug.print("\n[mlx-serve] -Dwebp-prefix={s}: {s} not found (the copy must include lib/pkgconfig)\n\n", .{ dir, pc_path });
            std.process.exit(1);
        };
        var ver_str: ?[]const u8 = null;
        var lines = std.mem.splitScalar(u8, pc, '\n');
        while (lines.next()) |line| {
            if (std.mem.startsWith(u8, line, "Version:")) ver_str = std.mem.trim(u8, line["Version:".len..], " \t\r");
        }
        const have = std.SemanticVersion.parse(ver_str orelse "") catch {
            std.debug.print("[mlx-serve] cannot read '{s}' version from {s}\n", .{ dep.name, pc_path });
            std.process.exit(1);
        };
        if (have.order(dep.min) == .lt) {
            std.debug.print(
                "\n[mlx-serve] -Dwebp-prefix={s}: '{s}' is {d}.{d}.{d}; need >= {d}.{d}.{d}\n\n",
                .{ dir, dep.name, have.major, have.minor, have.patch, dep.min.major, dep.min.minor, dep.min.patch },
            );
            std.process.exit(1);
        }
    }
}

fn verifyBrewDeps(b: *std.Build) void {
    for (required_brew_deps) |dep| {
        var code: u8 = undefined;
        const stdout = b.runAllowFail(
            &.{ "brew", "list", "--versions", dep.name },
            &code,
            .inherit,
        ) catch {
            std.debug.print(
                "\n[mlx-serve] missing Homebrew dependency '{s}' (>= {d}.{d}.{d}). Install with: brew install webp\n\n",
                .{ dep.name, dep.min.major, dep.min.minor, dep.min.patch },
            );
            std.process.exit(1);
        };
        const trimmed = std.mem.trim(u8, stdout, " \n\r\t");
        const space = std.mem.indexOfScalar(u8, trimmed, ' ') orelse {
            std.debug.print("[mlx-serve] cannot parse `brew list --versions {s}` output: {s}\n", .{ dep.name, trimmed });
            std.process.exit(1);
        };
        var ver_str = trimmed[space + 1 ..];
        // Strip Homebrew revision suffix (e.g., "0.6.0_2" -> "0.6.0").
        if (std.mem.indexOfScalar(u8, ver_str, '_')) |us| ver_str = ver_str[0..us];
        const have = std.SemanticVersion.parse(ver_str) catch {
            std.debug.print("[mlx-serve] cannot parse '{s}' version '{s}'\n", .{ dep.name, ver_str });
            std.process.exit(1);
        };
        if (have.order(dep.min) == .lt) {
            std.debug.print(
                "\n[mlx-serve] Homebrew '{s}' is {d}.{d}.{d}; need >= {d}.{d}.{d}. Run: brew upgrade {s}\n\n",
                .{ dep.name, have.major, have.minor, have.patch, dep.min.major, dep.min.minor, dep.min.patch, dep.name },
            );
            std.process.exit(1);
        }
    }
}

test "build config: a stage stamped for other pins names the first stale input; an input it cannot read is skipped" {
    const stamp = "mlx=64ea011cb65f mlxc=56b2d39fc831 patch=613a4be4d66a target=26.2\n";
    const mlxc: StageInput = .{ .key = "mlxc=", .path = "lib/mlxc-src", .have = "56b2d39fc831" };
    const patch: StageInput = .{ .key = "patch=", .path = "patches/mlxc.patch", .have = "613a4be4d66a" };
    const mlx_staged: StageInput = .{ .key = "mlx=", .path = "lib/mlx-src", .have = "64ea011cb65f" };
    const mlx_bumped: StageInput = .{ .key = "mlx=", .path = "lib/mlx-src", .have = "c69303b00f5f" };
    try std.testing.expectEqual(null, staleStageInput(stamp, &.{ mlx_staged, mlxc, patch }));
    const stale = staleStageInput(stamp, &.{ mlxc, mlx_bumped, patch }).?;
    try std.testing.expectEqualStrings("lib/mlx-src", stale.input.path);
    try std.testing.expectEqualStrings("64ea011cb65f", stale.staged);
    try std.testing.expectEqualStrings("none", staleStageInput("mlx=64ea011cb65f target=26.2", &.{patch}).?.staged);
    try std.testing.expectEqual(null, staleStageInput(stamp, &.{.{ .key = "mlx=", .path = "lib/mlx-src", .have = null }}));
}

const MapReader = struct {
    files: []const [2][]const u8,
    fn read(self: MapReader, path: []const u8) ![]const u8 {
        for (self.files) |f| if (std.mem.eql(u8, f[0], path)) return f[1];
        return error.FileNotFound;
    }
};

test "build config: patch= covers each patch the list names, hashed as `cat <patches> | shasum` prints" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const list = "# <submodule> <patch>\nlib/mlx-src patches/a.patch\n\nlib/mlxc-src patches/b.patch\n";
    const staged: MapReader = .{ .files = &.{ .{ "list", list }, .{ "patches/a.patch", "a\n" }, .{ "patches/b.patch", "b\n" } } };
    try std.testing.expectEqualStrings("05dec960e24d", patchesSha1(a, staged, "list").?);
    const stamp = "mlx=c69303b00f5f mlxc=56b2d39fc831 patch=05dec960e24d target=26.2";
    const edited_a: MapReader = .{ .files = &.{ .{ "list", list }, .{ "patches/a.patch", "a2\n" }, .{ "patches/b.patch", "b\n" } } };
    const edited_b: MapReader = .{ .files = &.{ .{ "list", list }, .{ "patches/a.patch", "a\n" }, .{ "patches/b.patch", "b2\n" } } };
    const dropped_b: MapReader = .{ .files = &.{ .{ "list", "lib/mlx-src patches/a.patch\n" }, .{ "patches/a.patch", "a\n" } } };
    // A list with no patch left is SHA-1 of nothing, as `cat /dev/null | shasum` prints.
    const none: MapReader = .{ .files = &.{.{ "list", "# <submodule> <patch>\n" }} };
    try std.testing.expectEqualStrings("da39a3ee5e6b", patchesSha1(a, none, "list").?);
    for ([_]MapReader{ staged, edited_a, edited_b, dropped_b }, [_]bool{ false, true, true, true }) |r, stale| {
        const in: StageInput = .{ .key = "patch=", .path = "list", .have = patchesSha1(a, r, "list") };
        try std.testing.expectEqual(stale, staleStageInput(stamp, &.{in}) != null);
    }
}

fn runGit(allocator: std.mem.Allocator, io: std.Io, cwd: std.Io.Dir, argv: []const []const u8) ![]u8 {
    const res = try std.process.run(allocator, io, .{
        .argv = argv,
        .cwd = .{ .dir = cwd },
        .stdout_limit = .limited(64 * 1024),
        .stderr_limit = .limited(64 * 1024),
    });
    defer allocator.free(res.stderr);
    switch (res.term) {
        .exited => |code| if (code != 0) {
            std.debug.print("git {s} failed ({d}): {s}\n", .{ argv[1], code, res.stderr });
            allocator.free(res.stdout);
            return error.GitFailed;
        },
        else => {
            allocator.free(res.stdout);
            return error.GitFailed;
        },
    }
    return res.stdout;
}

fn initTestRepo(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, file: []const u8, msg: []const u8) !void {
    allocator.free(try runGit(allocator, io, dir, &.{ "git", "init" }));
    allocator.free(try runGit(allocator, io, dir, &.{ "git", "config", "user.name", "Test" }));
    allocator.free(try runGit(allocator, io, dir, &.{ "git", "config", "user.email", "test@example.com" }));
    try dir.writeFile(io, .{ .sub_path = file, .data = "base\n" });
    allocator.free(try runGit(allocator, io, dir, &.{ "git", "add", file }));
    allocator.free(try runGit(allocator, io, dir, &.{ "git", "-c", "commit.gpgsign=false", "commit", "-m", msg }));
}

test "build config: resolveBuildId allows exact MLX patch diffs and build artifacts while flagging arbitrary MLX edits, staged pointers, untracked source, and embedded submodules" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();

    const no_git = try resolveBuildId(t.allocator, t.io, tmp.dir, "");
    defer t.allocator.free(no_git);
    try t.expectEqualStrings("unknown", no_git);

    try tmp.dir.createDir(t.io, "sub_mlx_origin", .default_dir);
    var sub_mlx_origin = try tmp.dir.openDir(t.io, "sub_mlx_origin", .{});
    defer sub_mlx_origin.close(t.io);
    try initTestRepo(t.allocator, t.io, sub_mlx_origin, "patch_target.txt", "mlx init");
    try sub_mlx_origin.writeFile(t.io, .{ .sub_path = "nax.h", .data = "// nax base\n" });
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx_origin, &.{ "git", "add", "nax.h" }));
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx_origin, &.{ "git", "-c", "commit.gpgsign=false", "commit", "-m", "add nax.h" }));

    try tmp.dir.createDir(t.io, "sub_ds4_origin", .default_dir);
    var sub_ds4_origin = try tmp.dir.openDir(t.io, "sub_ds4_origin", .{});
    defer sub_ds4_origin.close(t.io);
    try initTestRepo(t.allocator, t.io, sub_ds4_origin, "ds4.c", "ds4 init");

    try tmp.dir.createDir(t.io, "repo", .default_dir);
    var repo = try tmp.dir.openDir(t.io, "repo", .{});
    defer repo.close(t.io);
    try initTestRepo(t.allocator, t.io, repo, "main.zig", "init");
    try repo.createDirPath(t.io, "scripts");
    try repo.createDirPath(t.io, "patches");
    try repo.writeFile(t.io, .{ .sub_path = "scripts/build-mlx-patches.txt", .data = "lib/mlx-src patches/mlx-test.patch\n" });
    try repo.writeFile(t.io, .{
        .sub_path = "patches/mlx-test.patch",
        .data = "--- a/patch_target.txt\n+++ b/patch_target.txt\n@@ -1 +1 @@\n-base\n+patched\n",
    });
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "add", "scripts/build-mlx-patches.txt", "patches/mlx-test.patch" }));
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "-c", "commit.gpgsign=false", "commit", "-m", "add patch manifest" }));
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "-c", "protocol.file.allow=always", "submodule", "add", "../sub_mlx_origin", "lib/mlx-src" }));
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "-c", "protocol.file.allow=always", "submodule", "add", "../sub_ds4_origin", "lib/ds4" }));
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "-c", "commit.gpgsign=false", "commit", "-m", "add submodules" }));

    const head_raw = try runGit(t.allocator, t.io, repo, &.{ "git", "rev-parse", "--verify", "HEAD" });
    defer t.allocator.free(head_raw);
    const head_sha = std.mem.trim(u8, head_raw, " \t\r\n");
    const expected_dirty = try std.fmt.allocPrint(t.allocator, "{s}-dirty", .{head_sha});
    defer t.allocator.free(expected_dirty);

    const clean = try resolveBuildId(t.allocator, t.io, repo, "");
    defer t.allocator.free(clean);
    try t.expectEqualStrings(head_sha, clean);

    const override_sha = "0123456789abcdef0123456789abcdef01234567";
    const overridden = try resolveBuildId(t.allocator, t.io, repo, override_sha);
    defer t.allocator.free(overridden);
    try t.expectEqualStrings(override_sha, overridden);

    // Known release-script artifacts + exact patch from scripts/build-mlx-patches.txt stay clean.
    try repo.writeFile(t.io, .{ .sub_path = "cuda-keyring_1.1-1_all.deb", .data = "deb\n" });
    try repo.createDirPath(t.io, "dist");
    try repo.writeFile(t.io, .{ .sub_path = "dist/out.zip", .data = "zip\n" });
    var sub_mlx = try repo.openDir(t.io, "lib/mlx-src", .{});
    defer sub_mlx.close(t.io);
    try sub_mlx.writeFile(t.io, .{ .sub_path = "patch_target.txt", .data = "patched\n" });
    const patched_mlx = try resolveBuildId(t.allocator, t.io, repo, "");
    defer t.allocator.free(patched_mlx);
    try t.expectEqualStrings(head_sha, patched_mlx);

    // An unrelated edit inside lib/mlx-src (like nax.h) MUST flag -dirty.
    try sub_mlx.writeFile(t.io, .{ .sub_path = "nax.h", .data = "// edited nax\n" });
    const dirty_nax = try resolveBuildId(t.allocator, t.io, repo, "");
    defer t.allocator.free(dirty_nax);
    try t.expectEqualStrings(expected_dirty, dirty_nax);
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx, &.{ "git", "checkout", "--", "nax.h", "patch_target.txt" }));

    // Dirty embedded submodule code (lib/ds4) must flag -dirty.
    var sub_ds4 = try repo.openDir(t.io, "lib/ds4", .{});
    defer sub_ds4.close(t.io);
    try sub_ds4.writeFile(t.io, .{ .sub_path = "ds4.c", .data = "edited\n" });
    const dirty_ds4 = try resolveBuildId(t.allocator, t.io, repo, "");
    defer t.allocator.free(dirty_ds4);
    try t.expectEqualStrings(expected_dirty, dirty_ds4);
    t.allocator.free(try runGit(t.allocator, t.io, sub_ds4, &.{ "git", "checkout", "--", "ds4.c" }));

    // Moving lib/mlx-src HEAD AND staging its gitlink (`git add lib/mlx-src`) MUST flag -dirty.
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx, &.{ "git", "config", "user.name", "Test" }));
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx, &.{ "git", "config", "user.email", "test@example.com" }));
    try sub_mlx.writeFile(t.io, .{ .sub_path = "nax.h", .data = "// bumped\n" });
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx, &.{ "git", "add", "nax.h" }));
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx, &.{ "git", "-c", "commit.gpgsign=false", "commit", "-m", "move sub head" }));
    const dirty_ptr = try resolveBuildId(t.allocator, t.io, repo, "");
    defer t.allocator.free(dirty_ptr);
    try t.expectEqualStrings(expected_dirty, dirty_ptr);

    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "add", "lib/mlx-src" }));
    const dirty_staged_ptr = try resolveBuildId(t.allocator, t.io, repo, "");
    defer t.allocator.free(dirty_staged_ptr);
    try t.expectEqualStrings(expected_dirty, dirty_staged_ptr);

    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "reset", "HEAD", "--", "lib/mlx-src" }));
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx, &.{ "git", "reset", "--hard", "HEAD~1" }));
    const restored = try resolveBuildId(t.allocator, t.io, repo, "");
    defer t.allocator.free(restored);
    try t.expectEqualStrings(head_sha, restored);

    // Untracked plugin/source file MUST flag -dirty.
    try repo.createDirPath(t.io, "src/plugins/new_plugin");
    try repo.writeFile(t.io, .{ .sub_path = "src/plugins/new_plugin/plugin.zig", .data = "//! new\n" });
    const dirty_untracked = try resolveBuildId(t.allocator, t.io, repo, "");
    defer t.allocator.free(dirty_untracked);
    try t.expectEqualStrings(expected_dirty, dirty_untracked);
}
