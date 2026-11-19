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
/// or populated-uninitialized submodule dirs, or partial/mixed/unapproved edits inside `lib/mlx-src` /
/// `lib/mlxc-src`; or `"unknown"` only when `.git` / `HEAD` metadata is unavailable. Excludes known
/// root-anchored checkout build artifacts (`cuda-keyring_*.deb`, `dist/`, `.work-mlx-windows/`, etc.)
/// and complete matched platform patch sets (`scripts/build-mlx.sh`, `scripts/build-mlx-linux.sh`,
/// `scripts/build-mlx-windows.ps1`).
pub fn readBuildId(b: *std.Build, git_sha: []const u8) []const u8 {
    b.graph.poisonCache();
    return resolveBuildId(b.allocator, b.graph.io, &b.graph.environ_map, buildRootHandle(b), git_sha) catch "unknown";
}

fn runGitQueryEnv(
    allocator: std.mem.Allocator,
    io: std.Io,
    env: ?*const std.process.Environ.Map,
    root: std.Io.Dir,
    argv: []const []const u8,
    stdout_limit: usize,
) ?[]u8 {
    const res = std.process.run(allocator, io, .{
        .argv = argv,
        .cwd = .{ .dir = root },
        .environ_map = env,
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

fn runGitQuery(allocator: std.mem.Allocator, io: std.Io, root: std.Io.Dir, argv: []const []const u8, stdout_limit: usize) ?[]u8 {
    return runGitQueryEnv(allocator, io, null, root, argv, stdout_limit);
}

fn runGitOkEnv(
    allocator: std.mem.Allocator,
    io: std.Io,
    env: ?*const std.process.Environ.Map,
    root: std.Io.Dir,
    argv: []const []const u8,
) bool {
    const res = std.process.run(allocator, io, .{
        .argv = argv,
        .cwd = .{ .dir = root },
        .environ_map = env,
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

fn runGitOk(allocator: std.mem.Allocator, io: std.Io, root: std.Io.Dir, argv: []const []const u8) bool {
    return runGitOkEnv(allocator, io, null, root, argv);
}

fn runGitRefreshIndex(
    allocator: std.mem.Allocator,
    io: std.Io,
    env: *const std.process.Environ.Map,
    root: std.Io.Dir,
) bool {
    const res = std.process.run(allocator, io, .{
        .argv = &.{ "git", "--no-optional-locks", "update-index", "-q", "--refresh" },
        .cwd = .{ .dir = root },
        .environ_map = env,
        .stdout_limit = .limited(4096),
        .stderr_limit = .limited(4096),
    }) catch return false;
    allocator.free(res.stdout);
    allocator.free(res.stderr);
    return switch (res.term) {
        .exited => |code| code == 0 or code == 1,
        else => false,
    };
}

fn isMlxSubmodule(path: []const u8) bool {
    return std.mem.eql(u8, path, "lib/mlx-src") or std.mem.eql(u8, path, "lib/mlxc-src");
}

const HeadFile = struct {
    rel: []const u8,
    bytes: []const u8,
};

const PatchMode = enum {
    exact,
    windows,
};

const linux_mlx_cuda_patches = [_][]const u8{
    "patches/mlx-cuda-sdpa-head-dim-256-512.patch",
    "patches/mlx-cuda-qmm-read-weights-once.patch",
    "patches/mlx-cuda-qmv-rows.patch",
    "patches/mlx-cuda-row-dot-matvec.patch",
    "patches/mlx-cuda-sorted-gather-experts.patch",
    "patches/mlx-cuda-graphs-switch.patch",
    "patches/mlx-cuda-sorted-gather-matvec.patch",
    "patches/mlx-cuda-custom-kernel-disk-cache.patch",
};

const linux_mlxc_patches = [_][]const u8{
    "patches/mlxc-gather-qmm-global-scale.patch",
    "patches/mlxc-x86-f16-data-alias.patch",
    "patches/mlxc-clear-backends.patch",
};

const HeadTree = struct {
    paths: std.StringHashMapUnmanaged(void),
    submodules: std.ArrayList([]const u8),
    raw_buf: []u8,

    fn deinit(self: *HeadTree, allocator: std.mem.Allocator) void {
        self.paths.deinit(allocator);
        self.submodules.deinit(allocator);
        allocator.free(self.raw_buf);
    }
};

fn queryHeadTree(allocator: std.mem.Allocator, io: std.Io, repo_dir: std.Io.Dir) ?HeadTree {
    const raw = runGitQuery(allocator, io, repo_dir, &.{
        "git", "--no-optional-locks", "-c", "core.quotePath=false", "ls-tree", "-r", "-z", "HEAD",
    }, 16 * 1024 * 1024) orelse return null;

    var paths: std.StringHashMapUnmanaged(void) = .empty;
    var submodules: std.ArrayList([]const u8) = .empty;
    errdefer {
        paths.deinit(allocator);
        submodules.deinit(allocator);
        allocator.free(raw);
    }

    var recs = std.mem.tokenizeScalar(u8, raw, 0);
    while (recs.next()) |rec| {
        if (rec.len == 0) continue;
        const tab = std.mem.indexOfScalar(u8, rec, '\t') orelse {
            paths.deinit(allocator);
            submodules.deinit(allocator);
            allocator.free(raw);
            return null;
        };
        const meta = rec[0..tab];
        const path = rec[tab + 1 ..];
        var toks = std.mem.tokenizeScalar(u8, meta, ' ');
        _ = toks.next() orelse {
            paths.deinit(allocator);
            submodules.deinit(allocator);
            allocator.free(raw);
            return null;
        };
        const obj_type = toks.next() orelse {
            paths.deinit(allocator);
            submodules.deinit(allocator);
            allocator.free(raw);
            return null;
        };
        if (std.mem.eql(u8, obj_type, "commit")) {
            submodules.append(allocator, path) catch {
                paths.deinit(allocator);
                submodules.deinit(allocator);
                allocator.free(raw);
                return null;
            };
        } else {
            paths.put(allocator, path, {}) catch {
                paths.deinit(allocator);
                submodules.deinit(allocator);
                allocator.free(raw);
                return null;
            };
        }
    }

    return .{ .paths = paths, .submodules = submodules, .raw_buf = raw };
}

const HeadDiff = struct {
    changed_buf: []u8,
    changed: std.ArrayList([]const u8),
    mode_changed: bool,

    fn deinit(self: *HeadDiff, allocator: std.mem.Allocator) void {
        self.changed.deinit(allocator);
        allocator.free(self.changed_buf);
    }
};

fn queryHeadDiff(
    allocator: std.mem.Allocator,
    io: std.Io,
    parent_env: *const std.process.Environ.Map,
    repo_dir: std.Io.Dir,
    stage_abs: []const u8,
    idx_name: []const u8,
    ignore_submodules: bool,
) ?HeadDiff {
    const idx_abs = std.fmt.allocPrint(allocator, "{s}/{s}", .{ stage_abs, idx_name }) catch return null;
    defer allocator.free(idx_abs);

    var child_env = parent_env.clone(allocator) catch return null;
    defer child_env.deinit();
    child_env.put("GIT_INDEX_FILE", idx_abs) catch return null;

    if (!runGitOkEnv(allocator, io, &child_env, repo_dir, &.{ "git", "--no-optional-locks", "read-tree", "HEAD" })) return null;
    if (!runGitRefreshIndex(allocator, io, &child_env, repo_dir)) return null;

    const names_raw = if (ignore_submodules)
        runGitQueryEnv(allocator, io, &child_env, repo_dir, &.{
            "git", "--no-optional-locks", "-c", "core.quotePath=false", "diff-index", "--name-only", "--ignore-submodules", "HEAD", "--",
        }, 1024 * 1024) orelse return null
    else
        runGitQueryEnv(allocator, io, &child_env, repo_dir, &.{
            "git", "--no-optional-locks", "-c", "core.quotePath=false", "diff-index", "--name-only", "HEAD", "--",
        }, 1024 * 1024) orelse return null;

    var changed: std.ArrayList([]const u8) = .empty;
    var lines = std.mem.tokenizeScalar(u8, names_raw, '\n');
    while (lines.next()) |raw| {
        const rel = std.mem.trim(u8, raw, " \t\r");
        if (rel.len == 0) continue;
        changed.append(allocator, rel) catch {
            changed.deinit(allocator);
            allocator.free(names_raw);
            return null;
        };
    }

    const summary = if (ignore_submodules)
        runGitQueryEnv(allocator, io, &child_env, repo_dir, &.{
            "git", "--no-optional-locks", "diff-index", "--summary", "--ignore-submodules", "HEAD", "--",
        }, 1024 * 1024) orelse {
            changed.deinit(allocator);
            allocator.free(names_raw);
            return null;
        }
    else
        runGitQueryEnv(allocator, io, &child_env, repo_dir, &.{
            "git", "--no-optional-locks", "diff-index", "--summary", "HEAD", "--",
        }, 1024 * 1024) orelse {
            changed.deinit(allocator);
            allocator.free(names_raw);
            return null;
        };
    defer allocator.free(summary);

    const mode_changed = std.mem.indexOf(u8, summary, "mode change ") != null;
    return .{ .changed_buf = names_raw, .changed = changed, .mode_changed = mode_changed };
}

fn isDigitsOnly(s: []const u8) bool {
    if (s.len == 0) return false;
    for (s) |c| if (!std.ascii.isDigit(c)) return false;
    return true;
}

fn isAllowedSuperprojectUntrackedPath(rel: []const u8, entry_name: []const u8) bool {
    if (std.mem.eql(u8, entry_name, ".DS_Store")) return true;

    const exact_paths = [_][]const u8{
        ".zig-cache",
        "zig-out",
        ".zig-toolchain",
        "lib/mlx",
        "lib/mlx-win",
        "lib/.mlx-build",
        "lib/llama",
        "lib/llama-win",
        "lib/ds4-linux",
        "lib/guest",
        "lib/jinja_cpp/libjinja-linux.a",
        ".work-mlx-linux",
        ".work-mlx-linux-cuda",
        ".work-mlx-linux-cpu",
        ".work-mlx-windows",
        "dist",
        "scripts/__pycache__",
        ".claude",
        ".serena",
        ".cursor",
        ".vscode",
        ".build",
        "app/.build",
        "app/MLXCore-MAS.pkg",
        "app/.swiftpm",
        "app/.claude",
        "app/MLXCore.xcodeproj",
        "app/MLX Core.app",
        "app/MLX-Serve.app",
        "app/MLX Serve.app",
        "app/MLX-Serve.dmg",
        "app-web/node_modules",
        "models",
        "runs",
        "data",
        "docs/tmp",
        "docs/perf",
        "tests/pi-results",
        "tests/validator-results",
        "tests/llmprobe-smoke-results",
        "tests/agent_eval/out",
        "tests/fixtures",
        "missionlog.md",
        ".mc.json",
        "src/plugins/registry.zig",
    };
    for (exact_paths) |p| {
        if (std.mem.eql(u8, rel, p)) return true;
    }

    // Root-anchored `cuda-keyring_*.deb` (never in subdirectories).
    if (std.mem.indexOfScalar(u8, rel, '/') == null and
        std.mem.startsWith(u8, rel, "cuda-keyring_") and
        std.mem.endsWith(u8, rel, ".deb"))
    {
        return true;
    }

    // Generated `src/unit_<digits>.zig` from build.zig.
    if (std.mem.startsWith(u8, rel, "src/unit_") and std.mem.endsWith(u8, rel, ".zig")) {
        const inner = rel["src/unit_".len .. rel.len - ".zig".len];
        if (isDigitsOnly(inner)) return true;
    }

    // Release archives in app/*.zip.
    if (std.mem.startsWith(u8, rel, "app/") and
        std.mem.indexOfScalar(u8, rel["app/".len..], '/') == null and
        std.mem.endsWith(u8, rel, ".zip"))
    {
        return true;
    }

    return false;
}

fn scanSuperprojectDisallowedUntracked(
    allocator: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    prefix: []const u8,
    head_paths: *const std.StringHashMapUnmanaged(void),
    head_submodules: []const []const u8,
) bool {
    var iter = dir.iterate();
    while (iter.next(io) catch return true) |entry| {
        if (prefix.len == 0 and std.mem.eql(u8, entry.name, ".git")) continue;
        if (std.mem.eql(u8, entry.name, ".git")) return true; // nested .git outside registered submodules

        const rel = if (prefix.len == 0)
            allocator.dupe(u8, entry.name) catch return true
        else
            std.fmt.allocPrint(allocator, "{s}/{s}", .{ prefix, entry.name }) catch return true;
        defer allocator.free(rel);

        if (containsPath(head_submodules, rel)) continue;
        if (isAllowedSuperprojectUntrackedPath(rel, entry.name)) continue;

        if (entry.kind == .directory) {
            var sub = dir.openDir(io, entry.name, .{ .iterate = true }) catch return true;
            defer sub.close(io);
            if (scanSuperprojectDisallowedUntracked(allocator, io, sub, rel, head_paths, head_submodules)) return true;
        } else {
            if (!head_paths.contains(rel)) return true;
        }
    }
    return false;
}

fn scanSubmoduleUntracked(
    allocator: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    prefix: []const u8,
    subpath: []const u8,
    head_paths: *const std.StringHashMapUnmanaged(void),
    out_untracked: *std.ArrayList([]const u8),
) bool {
    var iter = dir.iterate();
    while (iter.next(io) catch return false) |entry| {
        if (prefix.len == 0 and std.mem.eql(u8, entry.name, ".git")) continue;
        if (std.mem.eql(u8, entry.name, ".git")) return false; // nested git repo inside submodule

        const rel = if (prefix.len == 0)
            allocator.dupe(u8, entry.name) catch return false
        else
            std.fmt.allocPrint(allocator, "{s}/{s}", .{ prefix, entry.name }) catch return false;

        if (entry.kind == .directory) {
            defer allocator.free(rel);
            var sub = dir.openDir(io, entry.name, .{ .iterate = true }) catch return false;
            defer sub.close(io);
            if (!scanSubmoduleUntracked(allocator, io, sub, rel, subpath, head_paths, out_untracked)) return false;
            continue;
        }

        if (head_paths.contains(rel)) {
            allocator.free(rel);
            continue;
        }
        // scripts/build-ds4-linux.sh compiles *.o in-tree inside lib/ds4.
        if (std.mem.eql(u8, subpath, "lib/ds4") and std.mem.endsWith(u8, rel, ".o")) {
            allocator.free(rel);
            continue;
        }
        if (std.mem.eql(u8, entry.name, ".DS_Store")) {
            allocator.free(rel);
            continue;
        }

        out_untracked.append(allocator, rel) catch {
            allocator.free(rel);
            return false;
        };
    }
    return true;
}

fn collectMacosPatchSet(
    allocator: std.mem.Allocator,
    io: std.Io,
    root: std.Io.Dir,
    subpath: []const u8,
    out: *std.ArrayList([]const u8),
) bool {
    const list_bytes = root.readFileAlloc(io, "scripts/build-mlx-patches.txt", allocator, .limited(64 * 1024)) catch return false;
    defer allocator.free(list_bytes);

    var lines = std.mem.tokenizeScalar(u8, list_bytes, '\n');
    while (lines.next()) |line| {
        var fields = std.mem.tokenizeAny(u8, line, " \t\r");
        const sub_col = fields.next() orelse continue;
        if (sub_col[0] == '#') continue;
        const patch_rel = fields.next() orelse return false;
        if (std.mem.eql(u8, sub_col, subpath)) {
            const duped = allocator.dupe(u8, patch_rel) catch return false;
            out.append(allocator, duped) catch {
                allocator.free(duped);
                return false;
            };
        }
    }
    return out.items.len != 0;
}

fn collectLinuxPatchSet(
    allocator: std.mem.Allocator,
    io: std.Io,
    root: std.Io.Dir,
    subpath: []const u8,
    out: *std.ArrayList([]const u8),
) bool {
    const list: []const []const u8 = if (std.mem.eql(u8, subpath, "lib/mlx-src"))
        &linux_mlx_cuda_patches
    else if (std.mem.eql(u8, subpath, "lib/mlxc-src"))
        &linux_mlxc_patches
    else
        return false;

    for (list) |patch_rel| {
        root.access(io, patch_rel, .{}) catch return false;
        const duped = allocator.dupe(u8, patch_rel) catch return false;
        out.append(allocator, duped) catch {
            allocator.free(duped);
            return false;
        };
    }
    return out.items.len != 0;
}

fn collectWindowsPatchSet(
    allocator: std.mem.Allocator,
    io: std.Io,
    root: std.Io.Dir,
    subpath: []const u8,
    out: *std.ArrayList([]const u8),
) bool {
    const want_mlxc = std.mem.eql(u8, subpath, "lib/mlxc-src");
    if (!want_mlxc and !std.mem.eql(u8, subpath, "lib/mlx-src")) return false;

    var patch_dir = root.openDir(io, "patches", .{ .iterate = true }) catch return false;
    defer patch_dir.close(io);

    var names: std.ArrayList([]const u8) = .empty;
    defer {
        for (names.items) |n| allocator.free(n);
        names.deinit(allocator);
    }

    var iter = patch_dir.iterate();
    while (iter.next(io) catch return false) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.startsWith(u8, entry.name, "mlx")) continue;
        if (!std.mem.endsWith(u8, entry.name, ".patch")) continue;
        const is_mlxc = std.mem.startsWith(u8, entry.name, "mlxc-");
        if (want_mlxc != is_mlxc) continue;
        const duped = allocator.dupe(u8, entry.name) catch return false;
        names.append(allocator, duped) catch {
            allocator.free(duped);
            return false;
        };
    }
    if (names.items.len == 0) return false;

    std.mem.sort([]const u8, names.items, {}, struct {
        fn lt(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lt);

    for (names.items) |name| {
        const patch_rel = std.fmt.allocPrint(allocator, "patches/{s}", .{name}) catch return false;
        out.append(allocator, patch_rel) catch {
            allocator.free(patch_rel);
            return false;
        };
    }
    return true;
}

fn freePatchList(allocator: std.mem.Allocator, patches: *std.ArrayList([]const u8)) void {
    for (patches.items) |p| allocator.free(p);
    patches.deinit(allocator);
}

const BlobHash = struct {
    rel: []const u8,
    sha: []const u8,
};

fn freeBlobHashes(allocator: std.mem.Allocator, list: *std.ArrayList(BlobHash), free_paths: bool) void {
    for (list.items) |item| {
        if (free_paths) allocator.free(item.rel);
        allocator.free(item.sha);
    }
    list.deinit(allocator);
}

fn hashSubmoduleFiles(
    allocator: std.mem.Allocator,
    io: std.Io,
    sub_dir: std.Io.Dir,
    dirty_files: []const []const u8,
    existing_hashes: *std.ArrayList(BlobHash),
    deleted_files: *std.ArrayList([]const u8),
) bool {
    var existing_paths: std.ArrayList([]const u8) = .empty;
    defer existing_paths.deinit(allocator);

    for (dirty_files) |rel| {
        if (rel.len == 0 or rel[rel.len - 1] == '/') return false;
        const st = sub_dir.statFile(io, rel, .{}) catch |err| switch (err) {
            error.FileNotFound => {
                deleted_files.append(allocator, rel) catch return false;
                continue;
            },
            else => return false,
        };
        if (st.kind != .file) return false;
        existing_paths.append(allocator, rel) catch return false;
    }

    var idx: usize = 0;
    while (idx < existing_paths.items.len) {
        const end = @min(idx + 32, existing_paths.items.len);
        const chunk = existing_paths.items[idx..end];
        var argv: std.ArrayList([]const u8) = .empty;
        defer argv.deinit(allocator);
        argv.appendSlice(allocator, &.{
            "git", "--no-optional-locks", "-c", "core.quotePath=false", "-c", "core.safecrlf=false", "hash-object", "--",
        }) catch return false;
        argv.appendSlice(allocator, chunk) catch return false;

        const out = runGitQuery(allocator, io, sub_dir, argv.items, 64 * 1024) orelse return false;
        defer allocator.free(out);

        var lines = std.mem.tokenizeScalar(u8, out, '\n');
        for (chunk) |rel| {
            const raw_sha = lines.next() orelse return false;
            const sha = std.mem.trim(u8, raw_sha, " \t\r");
            if (!isGitSha(sha)) return false;
            const duped = allocator.dupe(u8, sha) catch return false;
            existing_hashes.append(allocator, .{ .rel = rel, .sha = duped }) catch {
                allocator.free(duped);
                return false;
            };
        }
        if (lines.next() != null) return false;
        idx = end;
    }
    return true;
}

fn findBlobSha(list: []const BlobHash, path: []const u8) ?[]const u8 {
    for (list) |item| {
        if (std.mem.eql(u8, item.rel, path)) return item.sha;
    }
    return null;
}

fn containsPath(list: []const []const u8, path: []const u8) bool {
    for (list) |item| {
        if (std.mem.eql(u8, item, path)) return true;
    }
    return false;
}

fn openUniqueStageDir(allocator: std.mem.Allocator, io: std.Io, parent_dir: std.Io.Dir) ?[]u8 {
    var attempt: usize = 0;
    while (attempt < 8) : (attempt += 1) {
        var rnd: [8]u8 = undefined;
        io.random(&rnd);
        const dir_name = std.fmt.allocPrint(allocator, "mlx-patch-verify-{s}", .{std.fmt.bytesToHex(rnd, .lower)}) catch return null;
        parent_dir.createDir(io, dir_name, .default_dir) catch |err| {
            allocator.free(dir_name);
            if (err == error.PathAlreadyExists) continue;
            return null;
        };
        return dir_name;
    }
    return null;
}

fn tempParentPath() []const u8 {
    for ([_][*:0]const u8{ "TMPDIR", "TEMP", "TMP" }) |env_name| {
        if (std.c.getenv(env_name)) |ptr| {
            const trimmed = std.mem.trimEnd(u8, std.mem.span(ptr), "/\\");
            if (trimmed.len != 0) return trimmed;
        }
    }
    return "/tmp";
}

const MlxSubSnapshot = struct {
    present: bool = false,
    subpath: []const u8,
    sub_dir: ?std.Io.Dir = null,
    changed_buf: []u8 = &.{},
    changed: std.ArrayList([]const u8) = .empty,
    untracked: std.ArrayList([]const u8) = .empty,
    dirty_files: std.ArrayList([]const u8) = .empty,
    sub_hashes: std.ArrayList(BlobHash) = .empty,
    deleted_files: std.ArrayList([]const u8) = .empty,
    head_files: std.ArrayList(HeadFile) = .empty,
    autocrlf: []u8 = &.{},
    attributes: ?[]u8 = null,
    prepared: bool = false,

    fn deinit(self: *MlxSubSnapshot, allocator: std.mem.Allocator, io: std.Io) void {
        if (self.sub_dir) |*d| d.close(io);
        self.changed.deinit(allocator);
        if (self.changed_buf.len != 0) allocator.free(self.changed_buf);
        for (self.untracked.items) |u| allocator.free(u);
        self.untracked.deinit(allocator);
        self.dirty_files.deinit(allocator);
        freeBlobHashes(allocator, &self.sub_hashes, false);
        self.deleted_files.deinit(allocator);
        for (self.head_files.items) |hf| allocator.free(hf.bytes);
        self.head_files.deinit(allocator);
        if (self.autocrlf.len != 0) allocator.free(self.autocrlf);
        if (self.attributes) |a| allocator.free(a);
    }

    fn isClean(self: *const MlxSubSnapshot) bool {
        return self.changed.items.len == 0 and self.untracked.items.len == 0;
    }

    fn prepare(self: *MlxSubSnapshot, allocator: std.mem.Allocator, io: std.Io) bool {
        if (self.prepared) return true;
        if (!self.present) return false;
        const sub_dir = self.sub_dir orelse return false;

        for (self.changed.items) |rel| self.dirty_files.append(allocator, rel) catch return false;
        for (self.untracked.items) |rel| {
            if (!containsPath(self.dirty_files.items, rel)) self.dirty_files.append(allocator, rel) catch return false;
        }
        if (!hashSubmoduleFiles(allocator, io, sub_dir, self.dirty_files.items, &self.sub_hashes, &self.deleted_files)) return false;

        for (self.changed.items) |rel| {
            const spec = std.fmt.allocPrint(allocator, "HEAD:{s}", .{rel}) catch return false;
            defer allocator.free(spec);
            const orig = runGitQuery(allocator, io, sub_dir, &.{ "git", "--no-optional-locks", "show", spec }, 16 * 1024 * 1024) orelse return false;
            self.head_files.append(allocator, .{ .rel = rel, .bytes = orig }) catch {
                allocator.free(orig);
                return false;
            };
        }

        const autocrlf_raw = runGitQuery(allocator, io, sub_dir, &.{ "git", "--no-optional-locks", "config", "--get", "core.autocrlf" }, 64);
        if (autocrlf_raw) |buf| {
            defer allocator.free(buf);
            const trimmed = std.mem.trim(u8, buf, " \t\r\n");
            self.autocrlf = allocator.dupe(u8, if (trimmed.len != 0) trimmed else "false") catch return false;
        } else {
            self.autocrlf = allocator.dupe(u8, "false") catch return false;
        }

        self.attributes = sub_dir.readFileAlloc(io, ".gitattributes", allocator, .limited(256 * 1024)) catch null;
        self.prepared = true;
        return true;
    }
};

fn verifySnapshotAgainstPatches(
    allocator: std.mem.Allocator,
    io: std.Io,
    root_abs: []const u8,
    stage_handle: std.Io.Dir,
    cand_name: []const u8,
    snap: *MlxSubSnapshot,
    patches: []const []const u8,
    mode: PatchMode,
) bool {
    if (patches.len == 0) return !snap.present or snap.isClean();
    if (!snap.present or snap.isClean()) return false;
    if (!snap.prepare(allocator, io)) return false;

    stage_handle.createDir(io, cand_name, .default_dir) catch return false;
    var cand_dir = stage_handle.openDir(io, cand_name, .{}) catch return false;
    defer cand_dir.close(io);

    if (!runGitOk(allocator, io, cand_dir, &.{ "git", "init" })) return false;
    if (!runGitOk(allocator, io, cand_dir, &.{ "git", "config", "core.autocrlf", snap.autocrlf })) return false;
    if (snap.attributes) |attrs| {
        cand_dir.createDirPath(io, ".git/info") catch return false;
        cand_dir.writeFile(io, .{ .sub_path = ".git/info/attributes", .data = attrs }) catch return false;
    }

    for (snap.head_files.items) |hf| {
        if (std.fs.path.dirname(hf.rel)) |dir_part| cand_dir.createDirPath(io, dir_part) catch return false;
        cand_dir.writeFile(io, .{ .sub_path = hf.rel, .data = hf.bytes }) catch return false;
    }

    for (patches) |patch_rel| {
        const patch_abs = std.fmt.allocPrint(allocator, "{s}/{s}", .{ root_abs, patch_rel }) catch return false;
        defer allocator.free(patch_abs);
        switch (mode) {
            .exact => {
                if (!runGitOk(allocator, io, cand_dir, &.{ "git", "apply", "-p1", patch_abs })) return false;
            },
            .windows => {
                const check_args = &.{ "git", "apply", "-p1", "--ignore-whitespace", "--exclude=tests/*", "--exclude=docs/*", "--check", patch_abs };
                if (runGitOk(allocator, io, cand_dir, check_args)) {
                    if (!runGitOk(allocator, io, cand_dir, &.{ "git", "apply", "-p1", "--ignore-whitespace", "--exclude=tests/*", "--exclude=docs/*", patch_abs })) return false;
                } else {
                    const rev_args = &.{ "git", "apply", "-p1", "--ignore-whitespace", "--exclude=tests/*", "--exclude=docs/*", "--reverse", "--check", patch_abs };
                    if (!runGitOk(allocator, io, cand_dir, rev_args)) return false;
                }
            },
        }
    }

    if (!runGitOk(allocator, io, cand_dir, &.{ "git", "-c", "core.safecrlf=false", "add", "-A" })) return false;
    const ls_out = runGitQuery(allocator, io, cand_dir, &.{
        "git", "--no-optional-locks", "-c", "core.quotePath=false", "ls-files", "-s",
    }, 256 * 1024) orelse return false;
    defer allocator.free(ls_out);

    var cand_hashes: std.ArrayList(BlobHash) = .empty;
    defer freeBlobHashes(allocator, &cand_hashes, true);

    var ls_lines = std.mem.tokenizeScalar(u8, ls_out, '\n');
    while (ls_lines.next()) |raw_line| {
        const line = std.mem.trimEnd(u8, raw_line, "\r");
        if (line.len == 0) continue;
        const tab = std.mem.indexOfScalar(u8, line, '\t') orelse return false;
        var meta = std.mem.tokenizeScalar(u8, line[0..tab], ' ');
        _ = meta.next() orelse return false; // mode
        const sha = meta.next() orelse return false;
        if (!isGitSha(sha)) return false;
        const path = line[tab + 1 ..];
        const path_dup = allocator.dupe(u8, path) catch return false;
        const sha_dup = allocator.dupe(u8, sha) catch {
            allocator.free(path_dup);
            return false;
        };
        cand_hashes.append(allocator, .{ .rel = path_dup, .sha = sha_dup }) catch {
            allocator.free(path_dup);
            allocator.free(sha_dup);
            return false;
        };
    }

    if (cand_hashes.items.len != snap.sub_hashes.items.len) return false;
    for (snap.sub_hashes.items) |sh| {
        const cand_sha = findBlobSha(cand_hashes.items, sh.rel) orelse return false;
        if (!std.mem.eql(u8, cand_sha, sh.sha)) return false;
    }
    for (snap.deleted_files.items) |del_path| {
        if (findBlobSha(cand_hashes.items, del_path) != null) return false;
    }
    for (cand_hashes.items) |ch| {
        if (containsPath(snap.deleted_files.items, ch.rel)) return false;
    }
    return true;
}

fn verifyMlxPairJoint(
    allocator: std.mem.Allocator,
    io: std.Io,
    root: std.Io.Dir,
    stage_handle: std.Io.Dir,
    mlx_snap: *MlxSubSnapshot,
    mlxc_snap: *MlxSubSnapshot,
) bool {
    if (mlx_snap.isClean() and mlxc_snap.isClean()) return true;

    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_abs = root_buf[0 .. root.realPath(io, &root_buf) catch return false];

    // 1. macOS joint manifest (`scripts/build-mlx-patches.txt`).
    var mac_mlx: std.ArrayList([]const u8) = .empty;
    defer freePatchList(allocator, &mac_mlx);
    const has_mac_mlx = collectMacosPatchSet(allocator, io, root, "lib/mlx-src", &mac_mlx);

    var mac_mlxc: std.ArrayList([]const u8) = .empty;
    defer freePatchList(allocator, &mac_mlxc);
    const has_mac_mlxc = collectMacosPatchSet(allocator, io, root, "lib/mlxc-src", &mac_mlxc);

    if (has_mac_mlx or has_mac_mlxc) {
        if (verifySnapshotAgainstPatches(allocator, io, root_abs, stage_handle, "mac_mlx", mlx_snap, mac_mlx.items, .exact) and
            verifySnapshotAgainstPatches(allocator, io, root_abs, stage_handle, "mac_mlxc", mlxc_snap, mac_mlxc.items, .exact))
        {
            return true;
        }
    }

    // 2. Linux CUDA (`8` mlx-cuda patches + `3` mlxc patches) and Linux CPU (`clean` mlx + `3` mlxc patches).
    var lin_mlx: std.ArrayList([]const u8) = .empty;
    defer freePatchList(allocator, &lin_mlx);
    const has_lin_mlx = collectLinuxPatchSet(allocator, io, root, "lib/mlx-src", &lin_mlx);

    var lin_mlxc: std.ArrayList([]const u8) = .empty;
    defer freePatchList(allocator, &lin_mlxc);
    const has_lin_mlxc = collectLinuxPatchSet(allocator, io, root, "lib/mlxc-src", &lin_mlxc);

    if (has_lin_mlx and has_lin_mlxc) {
        if (verifySnapshotAgainstPatches(allocator, io, root_abs, stage_handle, "lin_mlx", mlx_snap, lin_mlx.items, .exact) and
            verifySnapshotAgainstPatches(allocator, io, root_abs, stage_handle, "lin_mlxc", mlxc_snap, lin_mlxc.items, .exact))
        {
            return true;
        }
    }
    if (has_lin_mlxc and mlx_snap.isClean()) {
        if (verifySnapshotAgainstPatches(allocator, io, root_abs, stage_handle, "lin_cpu_mlxc", mlxc_snap, lin_mlxc.items, .exact)) {
            return true;
        }
    }

    // 3. Windows joint set (`patches/mlx-*.patch` + `patches/mlxc-*.patch`).
    var win_mlx: std.ArrayList([]const u8) = .empty;
    defer freePatchList(allocator, &win_mlx);
    const has_win_mlx = collectWindowsPatchSet(allocator, io, root, "lib/mlx-src", &win_mlx);

    var win_mlxc: std.ArrayList([]const u8) = .empty;
    defer freePatchList(allocator, &win_mlxc);
    const has_win_mlxc = collectWindowsPatchSet(allocator, io, root, "lib/mlxc-src", &win_mlxc);

    if (has_win_mlx and has_win_mlxc) {
        if (verifySnapshotAgainstPatches(allocator, io, root_abs, stage_handle, "win_mlx", mlx_snap, win_mlx.items, .windows) and
            verifySnapshotAgainstPatches(allocator, io, root_abs, stage_handle, "win_mlxc", mlxc_snap, win_mlxc.items, .windows))
        {
            return true;
        }
    }

    return false;
}

fn dirHasAnyEntry(io: std.Io, dir: std.Io.Dir) bool {
    var iter = dir.iterate();
    return (iter.next(io) catch return true) != null;
}

fn submodulesClean(
    allocator: std.mem.Allocator,
    io: std.Io,
    parent_env: *const std.process.Environ.Map,
    root: std.Io.Dir,
    stage_handle: std.Io.Dir,
    stage_abs: []const u8,
    head_submodules: []const []const u8,
) bool {
    const sub_status = runGitQuery(allocator, io, root, &.{
        "git", "--no-optional-locks", "submodule", "status",
    }, 64 * 1024) orelse return false;
    defer allocator.free(sub_status);

    var all_subpaths: std.ArrayList([]const u8) = .empty;
    defer all_subpaths.deinit(allocator);
    for (head_submodules) |sp| all_subpaths.append(allocator, sp) catch return false;

    var lines = std.mem.tokenizeScalar(u8, sub_status, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        if (line[0] == '+' or line[0] == 'U') return false;
        const rest = std.mem.trimStart(u8, line[1..], " ");
        var toks = std.mem.tokenizeScalar(u8, rest, ' ');
        _ = toks.next() orelse continue;
        const subpath = toks.next() orelse continue;
        if (!containsPath(all_subpaths.items, subpath)) {
            all_subpaths.append(allocator, subpath) catch return false;
        }
    }

    var mlx_snap: MlxSubSnapshot = .{ .subpath = "lib/mlx-src" };
    defer mlx_snap.deinit(allocator, io);
    var mlxc_snap: MlxSubSnapshot = .{ .subpath = "lib/mlxc-src" };
    defer mlxc_snap.deinit(allocator, io);

    for (all_subpaths.items, 0..) |subpath, idx| {
        var sub_dir = root.openDir(io, subpath, .{ .iterate = true }) catch |err| switch (err) {
            error.FileNotFound => continue,
            else => return false,
        };
        var keep_open = false;
        defer if (!keep_open) sub_dir.close(io);

        // If `.git` is missing inside the submodule directory, ONLY an empty placeholder directory is clean.
        if (sub_dir.access(io, ".git", .{})) |_| {} else |_| {
            if (dirHasAnyEntry(io, sub_dir)) return false;
            continue;
        }

        if (!runGitOk(allocator, io, sub_dir, &.{ "git", "--no-optional-locks", "diff-index", "--cached", "--quiet", "HEAD", "--" })) {
            return false;
        }

        var sub_head = queryHeadTree(allocator, io, sub_dir) orelse return false;
        defer sub_head.deinit(allocator);

        const idx_name = std.fmt.allocPrint(allocator, "idx_sub_{d}", .{idx}) catch return false;
        defer allocator.free(idx_name);
        var sub_diff = queryHeadDiff(allocator, io, parent_env, sub_dir, stage_abs, idx_name, false) orelse return false;
        var owned_diff = true;
        defer if (owned_diff) sub_diff.deinit(allocator);
        if (sub_diff.mode_changed) return false;

        var sub_untracked: std.ArrayList([]const u8) = .empty;
        var owned_untracked = true;
        defer if (owned_untracked) {
            for (sub_untracked.items) |u| allocator.free(u);
            sub_untracked.deinit(allocator);
        };
        if (!scanSubmoduleUntracked(allocator, io, sub_dir, "", subpath, &sub_head.paths, &sub_untracked)) return false;

        if (!isMlxSubmodule(subpath)) {
            if (sub_diff.changed.items.len != 0 or sub_untracked.items.len != 0) return false;
            continue;
        }

        const dst = if (std.mem.eql(u8, subpath, "lib/mlx-src")) &mlx_snap else &mlxc_snap;
        dst.present = true;
        dst.sub_dir = sub_dir;
        dst.changed_buf = sub_diff.changed_buf;
        dst.changed = sub_diff.changed;
        dst.untracked = sub_untracked;
        keep_open = true;
        owned_diff = false;
        owned_untracked = false;
    }

    return verifyMlxPairJoint(allocator, io, root, stage_handle, &mlx_snap, &mlxc_snap);
}

fn dirtyId(allocator: std.mem.Allocator, sha: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}-dirty", .{sha});
}

fn resolveBuildId(
    allocator: std.mem.Allocator,
    io: std.Io,
    parent_env: *const std.process.Environ.Map,
    root: std.Io.Dir,
    git_sha: []const u8,
) ![]u8 {
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

    if (!runGitOk(allocator, io, root, &.{ "git", "--no-optional-locks", "diff-index", "--cached", "--quiet", "HEAD", "--" })) {
        return dirtyId(allocator, sha);
    }

    var parent_dir = std.Io.Dir.openDirAbsolute(io, tempParentPath(), .{}) catch return dirtyId(allocator, sha);
    defer parent_dir.close(io);
    const stage_dir = openUniqueStageDir(allocator, io, parent_dir) orelse return dirtyId(allocator, sha);
    defer {
        parent_dir.deleteTree(io, stage_dir) catch {};
        allocator.free(stage_dir);
    }
    var stage_handle = parent_dir.openDir(io, stage_dir, .{}) catch return dirtyId(allocator, sha);
    defer stage_handle.close(io);
    var stage_buf: [std.fs.max_path_bytes]u8 = undefined;
    const stage_abs = stage_buf[0 .. stage_handle.realPath(io, &stage_buf) catch return dirtyId(allocator, sha)];

    var head_tree = queryHeadTree(allocator, io, root) orelse return dirtyId(allocator, sha);
    defer head_tree.deinit(allocator);

    var root_diff = queryHeadDiff(allocator, io, parent_env, root, stage_abs, "idx_root", true) orelse return dirtyId(allocator, sha);
    defer root_diff.deinit(allocator);
    if (root_diff.mode_changed or root_diff.changed.items.len != 0) return dirtyId(allocator, sha);

    var root_iter_dir = root.openDir(io, ".", .{ .iterate = true }) catch return dirtyId(allocator, sha);
    defer root_iter_dir.close(io);
    if (scanSuperprojectDisallowedUntracked(allocator, io, root_iter_dir, "", &head_tree.paths, head_tree.submodules.items)) {
        return dirtyId(allocator, sha);
    }

    if (!submodulesClean(allocator, io, parent_env, root, stage_handle, stage_abs, head_tree.submodules.items)) {
        return dirtyId(allocator, sha);
    }
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
pub fn addTests(b: *std.Build, step: *std.Build.Step, test_filter: ?[]const u8) void {
    const t = b.addTest(.{
        .name = "build-config-test",
        .root_module = b.createModule(.{
            .root_source_file = b.path("build/config.zig"),
            .target = b.graph.host,
        }),
        .filters = if (test_filter) |f| &.{f} else &.{},
    });
    const run = b.addRunArtifact(t);
    if (test_filter != null) run.has_side_effects = true;
    step.dependOn(&run.step);
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

test "build config: build_id allows joint MLX patch pairs and root artifacts while catching unanchored dist, excludes, index flags, uninitialized archives, and submodules" {
    const t = std.testing;
    var test_env = try t.environ.createMap(t.allocator);
    defer test_env.deinit();

    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();

    const no_git = try resolveBuildId(t.allocator, t.io, &test_env, tmp.dir, "");
    defer t.allocator.free(no_git);
    try t.expectEqualStrings("unknown", no_git);

    try tmp.dir.createDir(t.io, "sub_mlx_origin", .default_dir);
    var sub_mlx_origin = try tmp.dir.openDir(t.io, "sub_mlx_origin", .{});
    defer sub_mlx_origin.close(t.io);
    try initTestRepo(t.allocator, t.io, sub_mlx_origin, "patch_target.txt", "mlx init");
    try sub_mlx_origin.writeFile(t.io, .{ .sub_path = "nax.h", .data = "// nax\n" });
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx_origin, &.{ "git", "add", "nax.h" }));
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx_origin, &.{ "git", "-c", "commit.gpgsign=false", "commit", "-m", "add nax" }));

    try tmp.dir.createDir(t.io, "sub_mlxc_origin", .default_dir);
    var sub_mlxc_origin = try tmp.dir.openDir(t.io, "sub_mlxc_origin", .{});
    defer sub_mlxc_origin.close(t.io);
    try initTestRepo(t.allocator, t.io, sub_mlxc_origin, "mlxc_target.txt", "mlxc init");

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
    try repo.writeFile(t.io, .{
        .sub_path = "scripts/build-mlx-patches.txt",
        .data = "lib/mlx-src patches/mlx-test.patch\nlib/mlxc-src patches/mlxc-test.patch\n",
    });
    try repo.writeFile(t.io, .{
        .sub_path = "patches/mlx-test.patch",
        .data = "--- a/patch_target.txt\n+++ b/patch_target.txt\n@@ -1 +1 @@\n-base\n+patched\n",
    });
    try repo.writeFile(t.io, .{
        .sub_path = "patches/mlxc-test.patch",
        .data = "--- a/mlxc_target.txt\n+++ b/mlxc_target.txt\n@@ -1 +1 @@\n-base\n+patched_c\n",
    });
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "add", "scripts/build-mlx-patches.txt", "patches/mlx-test.patch", "patches/mlxc-test.patch" }));
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "-c", "commit.gpgsign=false", "commit", "-m", "add patch manifest" }));
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "-c", "protocol.file.allow=always", "submodule", "add", "../sub_mlx_origin", "lib/mlx-src" }));
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "-c", "protocol.file.allow=always", "submodule", "add", "../sub_mlxc_origin", "lib/mlxc-src" }));
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "-c", "protocol.file.allow=always", "submodule", "add", "../sub_ds4_origin", "lib/ds4" }));
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "-c", "commit.gpgsign=false", "commit", "-m", "add submodules" }));

    const head_raw = try runGit(t.allocator, t.io, repo, &.{ "git", "rev-parse", "--verify", "HEAD" });
    defer t.allocator.free(head_raw);
    const head_sha = std.mem.trim(u8, head_raw, " \t\r\n");
    const expected_dirty = try std.fmt.allocPrint(t.allocator, "{s}-dirty", .{head_sha});
    defer t.allocator.free(expected_dirty);

    const clean = try resolveBuildId(t.allocator, t.io, &test_env, repo, "");
    defer t.allocator.free(clean);
    try t.expectEqualStrings(head_sha, clean);

    const override_sha = "0123456789abcdef0123456789abcdef01234567";
    const overridden = try resolveBuildId(t.allocator, t.io, &test_env, repo, override_sha);
    defer t.allocator.free(overridden);
    try t.expectEqualStrings(override_sha, overridden);

    // Root-anchored release artifacts + complete matched macOS pair stay clean.
    try repo.writeFile(t.io, .{ .sub_path = "cuda-keyring_1.1-1_all.deb", .data = "deb\n" });
    try repo.createDirPath(t.io, "dist");
    try repo.writeFile(t.io, .{ .sub_path = "dist/out.zip", .data = "zip\n" });
    try repo.createDirPath(t.io, ".build");
    try repo.writeFile(t.io, .{ .sub_path = ".build/build.db", .data = "db\n" });
    try repo.createDirPath(t.io, "app/.build");
    try repo.writeFile(t.io, .{ .sub_path = "app/.build/zig-build.log", .data = "log\n" });
    var sub_mlx = try repo.openDir(t.io, "lib/mlx-src", .{});
    defer sub_mlx.close(t.io);
    var sub_mlxc = try repo.openDir(t.io, "lib/mlxc-src", .{});
    defer sub_mlxc.close(t.io);
    try sub_mlx.writeFile(t.io, .{ .sub_path = "patch_target.txt", .data = "patched\n" });
    try sub_mlxc.writeFile(t.io, .{ .sub_path = "mlxc_target.txt", .data = "patched_c\n" });
    const patched_both = try resolveBuildId(t.allocator, t.io, &test_env, repo, "");
    defer t.allocator.free(patched_both);
    try t.expectEqualStrings(head_sha, patched_both);

    // Partial platform patch set (macOS MLX patched + untouched lib/mlxc-src) MUST flag -dirty!
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlxc, &.{ "git", "checkout", "--", "mlxc_target.txt" }));
    const dirty_partial = try resolveBuildId(t.allocator, t.io, &test_env, repo, "");
    defer t.allocator.free(dirty_partial);
    try t.expectEqualStrings(expected_dirty, dirty_partial);
    try sub_mlxc.writeFile(t.io, .{ .sub_path = "mlxc_target.txt", .data = "patched_c\n" });

    // An unrelated edit inside lib/mlx-src (like nax.h) MUST flag -dirty.
    try sub_mlx.writeFile(t.io, .{ .sub_path = "nax.h", .data = "// edited nax\n" });
    const dirty_nax = try resolveBuildId(t.allocator, t.io, &test_env, repo, "");
    defer t.allocator.free(dirty_nax);
    try t.expectEqualStrings(expected_dirty, dirty_nax);
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx, &.{ "git", "checkout", "--", "nax.h", "patch_target.txt" }));
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlxc, &.{ "git", "checkout", "--", "mlxc_target.txt" }));

    // --assume-unchanged and --skip-worktree in superproject and submodules MUST still flag -dirty!
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "update-index", "--assume-unchanged", "--skip-worktree", "main.zig" }));
    try repo.writeFile(t.io, .{ .sub_path = "main.zig", .data = "// sneaky edit\n" });
    const dirty_skip = try resolveBuildId(t.allocator, t.io, &test_env, repo, "");
    defer t.allocator.free(dirty_skip);
    try t.expectEqualStrings(expected_dirty, dirty_skip);
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "update-index", "--no-assume-unchanged", "--no-skip-worktree", "main.zig" }));
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "checkout", "--", "main.zig" }));

    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx, &.{ "git", "update-index", "--assume-unchanged", "--skip-worktree", "nax.h" }));
    try sub_mlx.writeFile(t.io, .{ .sub_path = "nax.h", .data = "// sneaky nax\n" });
    const dirty_sub_skip = try resolveBuildId(t.allocator, t.io, &test_env, repo, "");
    defer t.allocator.free(dirty_sub_skip);
    try t.expectEqualStrings(expected_dirty, dirty_sub_skip);
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx, &.{ "git", "update-index", "--no-assume-unchanged", "--no-skip-worktree", "nax.h" }));
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx, &.{ "git", "checkout", "--", "nax.h" }));

    // Dirty embedded submodule code (lib/ds4) must flag -dirty.
    var sub_ds4 = try repo.openDir(t.io, "lib/ds4", .{});
    defer sub_ds4.close(t.io);
    try sub_ds4.writeFile(t.io, .{ .sub_path = "ds4.c", .data = "edited\n" });
    const dirty_ds4 = try resolveBuildId(t.allocator, t.io, &test_env, repo, "");
    defer t.allocator.free(dirty_ds4);
    try t.expectEqualStrings(expected_dirty, dirty_ds4);
    t.allocator.free(try runGit(t.allocator, t.io, sub_ds4, &.{ "git", "checkout", "--", "ds4.c" }));

    // Populated uninitialized submodule dir without `.git` MUST flag -dirty; empty uninitialized dir stays clean.
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "submodule", "deinit", "-f", "lib/ds4" }));
    const clean_empty_sub = try resolveBuildId(t.allocator, t.io, &test_env, repo, "");
    defer t.allocator.free(clean_empty_sub);
    try t.expectEqualStrings(head_sha, clean_empty_sub);

    try repo.createDirPath(t.io, "lib/ds4");
    try repo.writeFile(t.io, .{ .sub_path = "lib/ds4/ds4.c", .data = "// archive extract\n" });
    const dirty_archive_sub = try resolveBuildId(t.allocator, t.io, &test_env, repo, "");
    defer t.allocator.free(dirty_archive_sub);
    try t.expectEqualStrings(expected_dirty, dirty_archive_sub);
    try repo.deleteFile(t.io, "lib/ds4/ds4.c");
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "-c", "protocol.file.allow=always", "submodule", "update", "--init", "lib/ds4" }));

    // Moving lib/mlx-src HEAD AND staging its gitlink MUST flag -dirty.
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx, &.{ "git", "config", "user.name", "Test" }));
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx, &.{ "git", "config", "user.email", "test@example.com" }));
    try sub_mlx.writeFile(t.io, .{ .sub_path = "nax.h", .data = "// bumped\n" });
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx, &.{ "git", "add", "nax.h" }));
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx, &.{ "git", "-c", "commit.gpgsign=false", "commit", "-m", "move sub head" }));
    const dirty_ptr = try resolveBuildId(t.allocator, t.io, &test_env, repo, "");
    defer t.allocator.free(dirty_ptr);
    try t.expectEqualStrings(expected_dirty, dirty_ptr);

    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "add", "lib/mlx-src" }));
    const dirty_staged_ptr = try resolveBuildId(t.allocator, t.io, &test_env, repo, "");
    defer t.allocator.free(dirty_staged_ptr);
    try t.expectEqualStrings(expected_dirty, dirty_staged_ptr);

    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "reset", "HEAD", "--", "lib/mlx-src" }));
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx, &.{ "git", "reset", "--hard", "HEAD~1" }));
    const restored = try resolveBuildId(t.allocator, t.io, &test_env, repo, "");
    defer t.allocator.free(restored);
    try t.expectEqualStrings(head_sha, restored);

    // Unanchored `dist` plugin (`src/plugins/dist/plugin.zig`) MUST flag -dirty!
    try repo.createDirPath(t.io, "src/plugins/dist");
    try repo.writeFile(t.io, .{ .sub_path = "src/plugins/dist/plugin.zig", .data = "//! dist plugin\n" });
    const dirty_dist_plugin = try resolveBuildId(t.allocator, t.io, &test_env, repo, "");
    defer t.allocator.free(dirty_dist_plugin);
    try t.expectEqualStrings(expected_dirty, dirty_dist_plugin);
    try repo.deleteTree(t.io, "src/plugins/dist");

    // Untracked plugin hidden by status.showUntrackedFiles=no + .git/info/exclude + core.excludesFile MUST flag -dirty!
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "config", "status.showUntrackedFiles", "no" }));
    try repo.writeFile(t.io, .{ .sub_path = ".git/info/exclude", .data = "src/plugins/new_plugin/\n" });
    var repo_buf: [std.fs.max_path_bytes]u8 = undefined;
    const repo_abs = repo_buf[0..try repo.realPath(t.io, &repo_buf)];
    const user_excl_abs = try std.fmt.allocPrint(t.allocator, "{s}/user_excludes", .{repo_abs});
    defer t.allocator.free(user_excl_abs);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "user_excludes", .data = "src/plugins/\nrogue.cpp\n" });
    var tmp_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_abs = tmp_buf[0..try tmp.dir.realPath(t.io, &tmp_buf)];
    const tmp_excl_abs = try std.fmt.allocPrint(t.allocator, "{s}/user_excludes", .{tmp_abs});
    defer t.allocator.free(tmp_excl_abs);
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "config", "core.excludesFile", tmp_excl_abs }));

    try repo.createDirPath(t.io, "src/plugins/new_plugin");
    try repo.writeFile(t.io, .{ .sub_path = "src/plugins/new_plugin/plugin.zig", .data = "//! new\n" });
    const dirty_untracked = try resolveBuildId(t.allocator, t.io, &test_env, repo, "");
    defer t.allocator.free(dirty_untracked);
    try t.expectEqualStrings(expected_dirty, dirty_untracked);
    try repo.deleteTree(t.io, "src/plugins");

    // Non-MLX submodule untracked file MUST flag -dirty even with status.showUntrackedFiles=no in the submodule.
    var sub_ds4_reopen = try repo.openDir(t.io, "lib/ds4", .{});
    defer sub_ds4_reopen.close(t.io);
    t.allocator.free(try runGit(t.allocator, t.io, sub_ds4_reopen, &.{ "git", "config", "status.showUntrackedFiles", "no" }));
    try sub_ds4_reopen.writeFile(t.io, .{ .sub_path = "untracked_ds4.c", .data = "// untracked\n" });
    const dirty_sub_untracked = try resolveBuildId(t.allocator, t.io, &test_env, repo, "");
    defer t.allocator.free(dirty_sub_untracked);
    try t.expectEqualStrings(expected_dirty, dirty_sub_untracked);
}

test "build config: build_id accepts Linux CUDA joint pair (incl. new files) and rejects mixed/rogue/deleted/nested" {
    const t = std.testing;
    var test_env = try t.environ.createMap(t.allocator);
    defer test_env.deinit();

    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.createDir(t.io, "sub_mlx_origin", .default_dir);
    var sub_mlx_orig = try tmp.dir.openDir(t.io, "sub_mlx_origin", .{});
    defer sub_mlx_orig.close(t.io);
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx_orig, &.{ "git", "init" }));
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx_orig, &.{ "git", "config", "user.name", "Test" }));
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx_orig, &.{ "git", "config", "user.email", "test@example.com" }));
    const mlx_targets = [_][]const u8{
        "mlx/backend/cuda/scaled_dot_product_attention.cpp",
        "mlx/backend/cuda/quantized/quantized.cpp",
        "mlx/backend/cuda/quantized/qmm/qmv.cu",
        "mlx/backend/cuda/device_info.cpp",
        "mlx/backend/cuda/device/qmm_sm80.cuh",
        "mlx/backend/cuda/device.cpp",
        "mlx/backend/cuda/quantized/qmm/qmm.h",
        "mlx/backend/cuda/custom_kernel.cpp",
        "mlx/distributed/jaccl/lib/jaccl/tcp.cpp",
        "nax.h",
    };
    for (mlx_targets) |rel| {
        if (std.fs.path.dirname(rel)) |dir_part| try sub_mlx_orig.createDirPath(t.io, dir_part);
        try sub_mlx_orig.writeFile(t.io, .{ .sub_path = rel, .data = "orig\n" });
    }
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx_orig, &.{ "git", "add", "." }));
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx_orig, &.{ "git", "-c", "commit.gpgsign=false", "commit", "-m", "init" }));

    try tmp.dir.createDir(t.io, "sub_mlxc_origin", .default_dir);
    var sub_mlxc_orig = try tmp.dir.openDir(t.io, "sub_mlxc_origin", .{});
    defer sub_mlxc_orig.close(t.io);
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlxc_orig, &.{ "git", "init" }));
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlxc_orig, &.{ "git", "config", "user.name", "Test" }));
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlxc_orig, &.{ "git", "config", "user.email", "test@example.com" }));
    for ([_][]const u8{ "mlx/c/ops.cpp", "mlx/c/half.h", "mlx/c/distributed_group.h" }) |rel| {
        if (std.fs.path.dirname(rel)) |dir_part| try sub_mlxc_orig.createDirPath(t.io, dir_part);
        try sub_mlxc_orig.writeFile(t.io, .{ .sub_path = rel, .data = "orig\n" });
    }
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlxc_orig, &.{ "git", "add", "." }));
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlxc_orig, &.{ "git", "-c", "commit.gpgsign=false", "commit", "-m", "init" }));

    try tmp.dir.createDir(t.io, "repo", .default_dir);
    var repo = try tmp.dir.openDir(t.io, "repo", .{});
    defer repo.close(t.io);
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "init" }));
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "config", "user.name", "Test" }));
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "config", "user.email", "test@example.com" }));
    try repo.createDirPath(t.io, "scripts");
    try repo.createDirPath(t.io, "patches");
    try repo.writeFile(t.io, .{
        .sub_path = "scripts/build-mlx-patches.txt",
        .data = "lib/mlx-src patches/mlx-jaccl-tcp-retry.patch\nlib/mlxc-src patches/mlxc-clear-backends.patch\n",
    });
    try repo.writeFile(t.io, .{
        .sub_path = "patches/mlx-jaccl-tcp-retry.patch",
        .data = "--- a/mlx/distributed/jaccl/lib/jaccl/tcp.cpp\n+++ b/mlx/distributed/jaccl/lib/jaccl/tcp.cpp\n@@ -1 +1 @@\n-orig\n+mac\n",
    });

    const cuda_patch_specs = [_]struct { patch: []const u8, file: []const u8, new_file: bool }{
        .{ .patch = "patches/mlx-cuda-sdpa-head-dim-256-512.patch", .file = "mlx/backend/cuda/scaled_dot_product_attention.cpp", .new_file = false },
        .{ .patch = "patches/mlx-cuda-qmm-read-weights-once.patch", .file = "mlx/backend/cuda/quantized/quantized.cpp", .new_file = false },
        .{ .patch = "patches/mlx-cuda-qmv-rows.patch", .file = "mlx/backend/cuda/quantized/qmm/qmv.cu", .new_file = false },
        .{ .patch = "patches/mlx-cuda-row-dot-matvec.patch", .file = "mlx/backend/cuda/gemms/row_dot.cuh", .new_file = true },
        .{ .patch = "patches/mlx-cuda-sorted-gather-experts.patch", .file = "mlx/backend/cuda/device/qmm_sm80.cuh", .new_file = false },
        .{ .patch = "patches/mlx-cuda-graphs-switch.patch", .file = "mlx/backend/cuda/device.cpp", .new_file = false },
        .{ .patch = "patches/mlx-cuda-sorted-gather-matvec.patch", .file = "mlx/backend/cuda/quantized/qmm/qmm.h", .new_file = false },
        .{ .patch = "patches/mlx-cuda-custom-kernel-disk-cache.patch", .file = "mlx/backend/cuda/custom_kernel.cpp", .new_file = false },
    };
    for (cuda_patch_specs) |spec| {
        const diff = if (spec.new_file)
            try std.fmt.allocPrint(
                t.allocator,
                "diff --git a/{0s} b/{0s}\nnew file mode 100644\nindex 0000000..1111111\n--- /dev/null\n+++ b/{0s}\n@@ -0,0 +1 @@\n+created\n",
                .{spec.file},
            )
        else
            try std.fmt.allocPrint(
                t.allocator,
                "--- a/{0s}\n+++ b/{0s}\n@@ -1 +1 @@\n-orig\n+patched\n",
                .{spec.file},
            );
        defer t.allocator.free(diff);
        try repo.writeFile(t.io, .{ .sub_path = spec.patch, .data = diff });
    }
    const mlxc_patch_specs = [_]struct { patch: []const u8, file: []const u8 }{
        .{ .patch = "patches/mlxc-gather-qmm-global-scale.patch", .file = "mlx/c/ops.cpp" },
        .{ .patch = "patches/mlxc-x86-f16-data-alias.patch", .file = "mlx/c/half.h" },
        .{ .patch = "patches/mlxc-clear-backends.patch", .file = "mlx/c/distributed_group.h" },
    };
    for (mlxc_patch_specs) |spec| {
        const diff = try std.fmt.allocPrint(t.allocator, "--- a/{0s}\n+++ b/{0s}\n@@ -1 +1 @@\n-orig\n+patched\n", .{spec.file});
        defer t.allocator.free(diff);
        try repo.writeFile(t.io, .{ .sub_path = spec.patch, .data = diff });
    }

    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_abs = root_buf[0..try tmp.dir.realPath(t.io, &root_buf)];
    const mlx_url = try std.fmt.allocPrint(t.allocator, "{s}/sub_mlx_origin", .{root_abs});
    defer t.allocator.free(mlx_url);
    const mlxc_url = try std.fmt.allocPrint(t.allocator, "{s}/sub_mlxc_origin", .{root_abs});
    defer t.allocator.free(mlxc_url);

    try repo.createDirPath(t.io, "lib");
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "-c", "protocol.file.allow=always", "submodule", "add", mlx_url, "lib/mlx-src" }));
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "-c", "protocol.file.allow=always", "submodule", "add", mlxc_url, "lib/mlxc-src" }));
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "add", "." }));
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "-c", "commit.gpgsign=false", "commit", "-m", "init" }));

    const head_raw = try runGit(t.allocator, t.io, repo, &.{ "git", "rev-parse", "--verify", "HEAD" });
    defer t.allocator.free(head_raw);
    const head_sha = std.mem.trim(u8, head_raw, " \t\r\n");
    const expected_dirty = try std.fmt.allocPrint(t.allocator, "{s}-dirty", .{head_sha});
    defer t.allocator.free(expected_dirty);

    var sub_mlx = try repo.openDir(t.io, "lib/mlx-src", .{});
    defer sub_mlx.close(t.io);
    var sub_mlxc = try repo.openDir(t.io, "lib/mlxc-src", .{});
    defer sub_mlxc.close(t.io);

    var repo_buf: [std.fs.max_path_bytes]u8 = undefined;
    const repo_abs = repo_buf[0..try repo.realPath(t.io, &repo_buf)];
    for (cuda_patch_specs) |spec| {
        const p_abs = try std.fmt.allocPrint(t.allocator, "{s}/{s}", .{ repo_abs, spec.patch });
        defer t.allocator.free(p_abs);
        t.allocator.free(try runGit(t.allocator, t.io, sub_mlx, &.{ "git", "apply", "-p1", p_abs }));
    }
    for (mlxc_patch_specs) |spec| {
        const p_abs = try std.fmt.allocPrint(t.allocator, "{s}/{s}", .{ repo_abs, spec.patch });
        defer t.allocator.free(p_abs);
        t.allocator.free(try runGit(t.allocator, t.io, sub_mlxc, &.{ "git", "apply", "-p1", p_abs }));
    }

    const clean_linux = try resolveBuildId(t.allocator, t.io, &test_env, repo, "");
    defer t.allocator.free(clean_linux);
    try t.expectEqualStrings(head_sha, clean_linux);

    // Extra tracked edit inside lib/mlx-src MUST still report -dirty.
    try sub_mlx.writeFile(t.io, .{ .sub_path = "nax.h", .data = "// edited\n" });
    const dirty_nax = try resolveBuildId(t.allocator, t.io, &test_env, repo, "");
    defer t.allocator.free(dirty_nax);
    try t.expectEqualStrings(expected_dirty, dirty_nax);
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx, &.{ "git", "checkout", "--", "nax.h" }));

    // Extra untracked file hidden via user core.excludesFile inside lib/mlx-src MUST still report -dirty.
    try tmp.dir.writeFile(t.io, .{ .sub_path = "mlx_excludes", .data = "rogue.cpp\n" });
    const mlx_excl_abs = try std.fmt.allocPrint(t.allocator, "{s}/mlx_excludes", .{root_abs});
    defer t.allocator.free(mlx_excl_abs);
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx, &.{ "git", "config", "core.excludesFile", mlx_excl_abs }));
    try sub_mlx.writeFile(t.io, .{ .sub_path = "rogue.cpp", .data = "// rogue\n" });
    const dirty_rogue = try resolveBuildId(t.allocator, t.io, &test_env, repo, "");
    defer t.allocator.free(dirty_rogue);
    try t.expectEqualStrings(expected_dirty, dirty_rogue);
    try sub_mlx.deleteFile(t.io, "rogue.cpp");

    // Deleted tracked file inside lib/mlx-src MUST report -dirty (even alongside valid patches).
    try sub_mlx.deleteFile(t.io, "nax.h");
    const dirty_del = try resolveBuildId(t.allocator, t.io, &test_env, repo, "");
    defer t.allocator.free(dirty_del);
    try t.expectEqualStrings(expected_dirty, dirty_del);
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx, &.{ "git", "checkout", "--", "nax.h" }));

    // Mixed platform pair (macOS MLX + Linux mlxc) MUST report -dirty!
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx, &.{ "git", "reset", "--hard", "HEAD" }));
    try sub_mlx.deleteFile(t.io, "mlx/backend/cuda/gemms/row_dot.cuh");
    const mac_p_abs = try std.fmt.allocPrint(t.allocator, "{s}/patches/mlx-jaccl-tcp-retry.patch", .{repo_abs});
    defer t.allocator.free(mac_p_abs);
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx, &.{ "git", "apply", "-p1", mac_p_abs }));
    const dirty_mixed = try resolveBuildId(t.allocator, t.io, &test_env, repo, "");
    defer t.allocator.free(dirty_mixed);
    try t.expectEqualStrings(expected_dirty, dirty_mixed);
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx, &.{ "git", "reset", "--hard", "HEAD" }));
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlxc, &.{ "git", "reset", "--hard", "HEAD" }));

    // Untracked nested git repo / directory inside lib/mlx-src MUST fail closed as -dirty.
    try sub_mlx.createDirPath(t.io, "nested_repo");
    var nested = try sub_mlx.openDir(t.io, "nested_repo", .{});
    defer nested.close(t.io);
    t.allocator.free(try runGit(t.allocator, t.io, nested, &.{ "git", "init" }));
    const dirty_nested = try resolveBuildId(t.allocator, t.io, &test_env, repo, "");
    defer t.allocator.free(dirty_nested);
    try t.expectEqualStrings(expected_dirty, dirty_nested);
}

test "build config: build_id accepts Windows CRLF patches under autocrlf=true and safecrlf=true, and ignores .work-mlx-windows" {
    const t = std.testing;
    var test_env = try t.environ.createMap(t.allocator);
    defer test_env.deinit();

    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.createDir(t.io, "sub_mlx_origin", .default_dir);
    var sub_mlx_orig = try tmp.dir.openDir(t.io, "sub_mlx_origin", .{});
    defer sub_mlx_orig.close(t.io);
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx_orig, &.{ "git", "init" }));
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx_orig, &.{ "git", "config", "user.name", "Test" }));
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx_orig, &.{ "git", "config", "user.email", "test@example.com" }));
    for ([_][]const u8{
        "mlx/backend/cuda/delayload.cpp",
        "tests/cuda_tests.cpp",
        "docs/env.rst",
        "nax.h",
    }) |rel| {
        if (std.fs.path.dirname(rel)) |dir_part| try sub_mlx_orig.createDirPath(t.io, dir_part);
        try sub_mlx_orig.writeFile(t.io, .{ .sub_path = rel, .data = "line1\nline2\n" });
    }
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx_orig, &.{ "git", "add", "." }));
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx_orig, &.{ "git", "-c", "commit.gpgsign=false", "commit", "-m", "init" }));

    try tmp.dir.createDir(t.io, "sub_mlxc_origin", .default_dir);
    var sub_mlxc_orig = try tmp.dir.openDir(t.io, "sub_mlxc_origin", .{});
    defer sub_mlxc_orig.close(t.io);
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlxc_orig, &.{ "git", "init" }));
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlxc_orig, &.{ "git", "config", "user.name", "Test" }));
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlxc_orig, &.{ "git", "config", "user.email", "test@example.com" }));
    try sub_mlxc_orig.createDirPath(t.io, "mlx/c");
    try sub_mlxc_orig.writeFile(t.io, .{ .sub_path = "mlx/c/ops.cpp", .data = "orig1\norig2\n" });
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlxc_orig, &.{ "git", "add", "." }));
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlxc_orig, &.{ "git", "-c", "commit.gpgsign=false", "commit", "-m", "init" }));

    try tmp.dir.createDir(t.io, "repo", .default_dir);
    var repo = try tmp.dir.openDir(t.io, "repo", .{});
    defer repo.close(t.io);
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "init" }));
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "config", "user.name", "Test" }));
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "config", "user.email", "test@example.com" }));
    try repo.createDirPath(t.io, "scripts");
    try repo.writeFile(t.io, .{ .sub_path = "scripts/build-mlx-patches.txt", .data = "# macos\n" });
    try repo.createDirPath(t.io, "patches");

    try repo.writeFile(t.io, .{
        .sub_path = "patches/mlx-cuda-windows-toolkit-dirs.patch",
        .data = "--- a/mlx/backend/cuda/delayload.cpp\r\n" ++
            "+++ b/mlx/backend/cuda/delayload.cpp\r\n" ++
            "@@ -1,2 +1,2 @@\r\n" ++
            " line1\r\n" ++
            "-line2\r\n" ++
            "+line2_win\r\n" ++
            "--- a/tests/cuda_tests.cpp\r\n" ++
            "+++ b/tests/cuda_tests.cpp\r\n" ++
            "@@ -1,2 +1,2 @@\r\n" ++
            " line1\r\n" ++
            "-line2\r\n" ++
            "+line2_test\r\n" ++
            "--- a/docs/env.rst\r\n" ++
            "+++ b/docs/env.rst\r\n" ++
            "@@ -1,2 +1,2 @@\r\n" ++
            " line1\r\n" ++
            "-line2\r\n" ++
            "+line2_doc\r\n",
    });
    try repo.writeFile(t.io, .{
        .sub_path = "patches/mlxc-win.patch",
        .data = "--- a/mlx/c/ops.cpp\r\n" ++
            "+++ b/mlx/c/ops.cpp\r\n" ++
            "@@ -1,2 +1,2 @@\r\n" ++
            " orig1\r\n" ++
            "-orig2\r\n" ++
            "+orig2_win\r\n",
    });

    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_abs = root_buf[0..try tmp.dir.realPath(t.io, &root_buf)];
    const mlx_url = try std.fmt.allocPrint(t.allocator, "{s}/sub_mlx_origin", .{root_abs});
    defer t.allocator.free(mlx_url);
    const mlxc_url = try std.fmt.allocPrint(t.allocator, "{s}/sub_mlxc_origin", .{root_abs});
    defer t.allocator.free(mlxc_url);

    try repo.createDirPath(t.io, "lib");
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "-c", "protocol.file.allow=always", "submodule", "add", mlx_url, "lib/mlx-src" }));
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "-c", "protocol.file.allow=always", "submodule", "add", mlxc_url, "lib/mlxc-src" }));
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "add", "." }));
    t.allocator.free(try runGit(t.allocator, t.io, repo, &.{ "git", "-c", "commit.gpgsign=false", "commit", "-m", "init" }));

    const head_raw = try runGit(t.allocator, t.io, repo, &.{ "git", "rev-parse", "--verify", "HEAD" });
    defer t.allocator.free(head_raw);
    const head_sha = std.mem.trim(u8, head_raw, " \t\r\n");
    const expected_dirty = try std.fmt.allocPrint(t.allocator, "{s}-dirty", .{head_sha});
    defer t.allocator.free(expected_dirty);

    var sub_mlx = try repo.openDir(t.io, "lib/mlx-src", .{});
    defer sub_mlx.close(t.io);
    var sub_mlxc = try repo.openDir(t.io, "lib/mlxc-src", .{});
    defer sub_mlxc.close(t.io);

    for ([_]std.Io.Dir{ sub_mlx, sub_mlxc }) |sd| {
        t.allocator.free(try runGit(t.allocator, t.io, sd, &.{ "git", "config", "core.autocrlf", "true" }));
        t.allocator.free(try runGit(t.allocator, t.io, sd, &.{ "git", "config", "core.safecrlf", "true" }));
        t.allocator.free(try runGit(t.allocator, t.io, sd, &.{ "git", "rm", "-rf", "." }));
        t.allocator.free(try runGit(t.allocator, t.io, sd, &.{ "git", "reset", "--hard", "HEAD" }));
    }
    const disk_nax = try sub_mlx.readFileAlloc(t.io, "nax.h", t.allocator, .limited(4096));
    defer t.allocator.free(disk_nax);
    try t.expect(std.mem.indexOf(u8, disk_nax, "\r\n") != null);

    var repo_buf: [std.fs.max_path_bytes]u8 = undefined;
    const repo_abs = repo_buf[0..try repo.realPath(t.io, &repo_buf)];
    const mlx_p = try std.fmt.allocPrint(t.allocator, "{s}/patches/mlx-cuda-windows-toolkit-dirs.patch", .{repo_abs});
    defer t.allocator.free(mlx_p);
    const mlxc_p = try std.fmt.allocPrint(t.allocator, "{s}/patches/mlxc-win.patch", .{repo_abs});
    defer t.allocator.free(mlxc_p);
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx, &.{ "git", "apply", "-p1", "--ignore-whitespace", "--exclude=tests/*", "--exclude=docs/*", mlx_p }));
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlxc, &.{ "git", "apply", "-p1", "--ignore-whitespace", "--exclude=tests/*", "--exclude=docs/*", mlxc_p }));

    // Windows build staging dir (.work-mlx-windows/) MUST stay clean.
    try repo.createDirPath(t.io, ".work-mlx-windows/prefix");
    try repo.writeFile(t.io, .{ .sub_path = ".work-mlx-windows/prefix/foo.obj", .data = "obj\n" });

    const clean_win = try resolveBuildId(t.allocator, t.io, &test_env, repo, "");
    defer t.allocator.free(clean_win);
    try t.expectEqualStrings(head_sha, clean_win);

    // Hand-modifying a patched line under CRLF + autocrlf=true MUST report -dirty.
    try sub_mlx.writeFile(t.io, .{ .sub_path = "mlx/backend/cuda/delayload.cpp", .data = "line1\r\nline2_hand_edit\r\n" });
    const dirty_hand = try resolveBuildId(t.allocator, t.io, &test_env, repo, "");
    defer t.allocator.free(dirty_hand);
    try t.expectEqualStrings(expected_dirty, dirty_hand);
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx, &.{ "git", "checkout", "--", "mlx/backend/cuda/delayload.cpp" }));
    t.allocator.free(try runGit(t.allocator, t.io, sub_mlx, &.{ "git", "apply", "-p1", "--ignore-whitespace", "--exclude=tests/*", "--exclude=docs/*", mlx_p }));

    // Unrelated CRLF edit inside nax.h MUST report -dirty.
    try sub_mlx.writeFile(t.io, .{ .sub_path = "nax.h", .data = "// edited\r\nline2\r\n" });
    const dirty_nax_crlf = try resolveBuildId(t.allocator, t.io, &test_env, repo, "");
    defer t.allocator.free(dirty_nax_crlf);
    try t.expectEqualStrings(expected_dirty, dirty_nax_crlf);
}
