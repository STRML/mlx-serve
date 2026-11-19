# Redesign Plan: Build Identity (`feat/report-build-id`, PR #53)

## Context

`build_options.build_id` (computed by `readBuildId` / `resolveBuildId` in `build/config.zig` and exposed on `/v1/internal/mode` and `/props`) must uniquely identify the exact source code compiled into `mlx-serve`:

- `<40/64-char HEAD SHA>` when the checkout and all embedded/staged submodules match committed `HEAD` plus at most the expected release/CI build artifacts (`cuda-keyring_*.deb`, `dist/`) and the exact platform MLX/mlx-c patch set (`scripts/build-mlx.sh`, `scripts/build-mlx-linux.sh`, `scripts/build-mlx-windows.ps1`).
- `<HEAD SHA>-dirty` on any uncommitted superproject edit, staged change (including staged submodule gitlinks), untracked source/plugin file, moved/conflicted submodule HEAD, dirty non-MLX submodule (`lib/ds4`, `lib/opencode2-mlx-serve`), or unapproved/partial/extra edit or untracked file inside `lib/mlx-src` or `lib/mlxc-src`.
- `"unknown"` only when `.git` / git metadata is genuinely unavailable.

Rounds 1–4 surfaced recurring regressions because earlier incremental patches tweaked `git status` flags/pathspecs ad hoc and later reimplemented Git's path traversal and content comparison manually in Zig. This redesign eliminates those defect classes structurally.

---

## Findings Inventory (Rounds 1–4) & Root-Cause Grouping

### Group A — Ad-hoc `git status` / submodule flag exclusions
1. **Round 1 `[P1]` — Linux release builds report clean source as dirty (`cuda-keyring_*.deb`, `dist/`)**:
   `.github/workflows/linux.yml` downloads `cuda-keyring_1.1-1_all.deb` and `scripts/build-windows.ps1` creates `dist/` in the checkout; bare `--untracked-files=normal` flagged them `-dirty`.
2. **Round 1 `[P1]` — Modified embedded submodule code reports a clean build (`lib/ds4`, `lib/opencode2-mlx-serve`)**:
   `--ignore-submodules=dirty` hid working-tree edits inside embedded C/Zig submodules (`build/libs.zig`, `lib/opencode2_plugin.zig`).
3. **Round 2 `[P1]` — Modified embedded MLX submodule code reports a clean build (`nax.h`)**:
   Switching to `--ignore-submodules=none` + pathspec exclusion hid all edits inside `lib/mlx-src` and `lib/mlxc-src`, including `nax.h` embedded via `lib/mlx_steel_sources.zig`.
4. **Round 2 `[P1]` — Staged MLX submodule updates report a clean build**:
   Excluding MLX submodules from superproject status caused staged gitlink updates (`git add lib/mlx-src`) to match `git submodule status` (which compares working HEAD to index) and miss the staged pointer change vs `HEAD`.
5. **Round 2 `[P1]` — Untracked plugin source reports a clean build**:
   Using `--untracked-files=no` hid untracked plugins under `src/plugins/`, which `build/plugins.zig` discovers automatically.
6. **Round 3 `[P1]` — Untracked plugin source reports a clean build under `status.showUntrackedFiles=no`**:
   User/global Git config `status.showUntrackedFiles=no` caused bare `git status --porcelain` in both superproject and non-MLX submodules to omit untracked files.

**Design change eliminating Group A**:
- Always pass `-c status.showUntrackedFiles=normal -c core.quotePath=false status --porcelain=v1` in both superproject (`--ignore-submodules` with explicit `:(exclude)cuda-keyring_*.deb` and `:(exclude)dist` pathspecs) and non-MLX submodules.
- Explicitly check staged changes (including staged gitlink changes) via `git --no-optional-locks diff-index --cached --quiet HEAD --`.
- Inspect every initialized entry from `git --no-optional-locks submodule status` directly: fail on `+` (moved HEAD) or `U` (conflict); validate non-MLX submodules cleanly; validate MLX submodules (`lib/mlx-src`, `lib/mlxc-src`) via canonical Git verification below.

### Group B — Incomplete platform patch manifest coverage
7. **Round 3 `[P1]` — Linux CUDA builds report expected patches as dirty**:
   Only `scripts/build-mlx-patches.txt` (macOS) was checked, rejecting `scripts/build-mlx-linux.sh` patches (`8` `mlx-cuda-*.patch` + `3` `mlxc-*.patch`, including `new file mode` untracked files `row_dot.cuh` and `row_dot_ops.cuh` created by `mlx-cuda-row-dot-matvec.patch`) and `scripts/build-mlx-windows.ps1` (`patches/mlx*.patch` with `--ignore-whitespace --exclude=tests/* --exclude=docs/*`).

**Design change eliminating Group B**:
- Collect all three official platform patch sets cleanly in `matchesExpectedMlxPatches`:
  1. **macOS**: exact manifest from `scripts/build-mlx-patches.txt` (`git apply -p1`).
  2. **Linux**: exact `build-mlx-linux.sh` ordered arrays (`linux_mlx_cuda_patches` for `lib/mlx-src`, `linux_mlxc_patches` for `lib/mlxc-src`, applied with `git apply -p1`), active when those patch files exist in `root`.
  3. **Windows**: sorted `patches/mlx*.patch` split by `mlxc-` prefix (`collectWindowsPatchSet`), applied with `git apply -p1 --ignore-whitespace --exclude=tests/* --exclude=docs/*` (honoring `build-mlx-windows.ps1`'s `--check` / `--reverse --check` skip-if-already-applied semantics).
- Inspect BOTH `diff --name-only` (tracked edits/deletions) AND `ls-files --others --exclude-standard` (patch-created untracked files vs rogue untracked files).

### Group C — Manual filesystem traversal & raw-byte comparison fighting OS separators and CRLF
8. **Round 4 `[P1]` — Windows patch verification walks the temporary Git directory**:
   `std.Io.Dir.walk` returns `.git\config` on Windows, bypassing `startsWith(entry.path, ".git/")` and attempting to open `sub_dir/.git\config`.
9. **Round 4 `[P1]` — Windows line endings make expected patches appear dirty**:
   `std.mem.eql(u8, want, have)` compared raw LF `git show HEAD:<rel>` / LF-patched bytes against CRLF working-tree files produced when `core.autocrlf=true`.

**Design change eliminating Group C**:
- **Delete `Dir.walk` and delete raw-byte file comparisons (`std.mem.eql(u8, want, have)`) completely.** Delegate canonical path enumeration and canonical line-ending normalization to **Git itself**:
  1. In the temporary candidate repo (`cand_dir`), read `sub_dir`'s `core.autocrlf` (`git -C sub_dir config --get core.autocrlf`, defaulting to `false`) and apply that exact setting to `cand_dir`; if `sub_dir/.gitattributes` exists, copy it to `cand_dir/.git/info/attributes` (never polluting `cand_dir`'s working tree or index).
  2. Seed `HEAD:<rel>` blobs for `changed_files` into `cand_dir`, apply the candidate's patches, run `git -C cand_dir add -A`, and query `git -C cand_dir -c core.quotePath=false ls-files -s`.
     - Git **never** lists `.git/` or `.git\`.
     - Git **always** emits normalized forward-slash (`/`) paths on macOS, Linux, and Windows.
     - Git computes the canonical Git blob SHA-1 for every file in `cand_dir` under `sub_dir`'s exact line-ending rules.
  3. In `sub_dir` (100% read-only, without `-w`), batch-hash all existing `dirty_files` (`changed ∪ untracked`) via `git -C sub_dir -c core.quotePath=false hash-object -- <chunk...>` (chunks of 32 paths to stay well within Windows `CreateProcess` command-line limits).
     - Git runs `sub_dir`'s `core.autocrlf` / `.gitattributes` clean filters, hashing CRLF Windows checkouts and LF checkouts to the exact same canonical blob SHA-1 when content matches, while any real content change produces a different SHA-1.
  4. Verify bidirectional exact match:
     - Every `rel` in `dirty_files`: if present in `sub_dir` (`sub_dir.access` succeeds), it must appear in `cand_dir`'s `ls-files -s` output with identical blob SHA-1; if absent in `sub_dir` (deleted), it must also be absent from `cand_dir`'s `ls-files -s` output.
     - Every `(path, sha)` in `cand_dir`'s `ls-files -s` output must appear in `dirty_files` (so a file created/kept by a patch cannot be missing from `sub_dir`).

### Group D — Test config leakage & concurrent build directory races
10. **Round 1 `[P2]` — Test fixture inherits global `commit.gpgsign=true`**:
    All test commits explicitly pass `-c commit.gpgsign=false`.
11. **Round 3 `[P2]` — Concurrent builds collide on deterministic `msv-bid-<hash>-0`**:
    `openUniqueStageDir` generates a random hex suffix via `io.random` and creates the directory atomically via `parent_dir.createDir` with `PathAlreadyExists` retry, cleaned up via `defer deleteTree`.

---

## Architecture & Control Flow

```
readBuildId(b, git_sha)
  ├─ b.graph.poisonCache()
  └─ resolveBuildId(allocator, io, root, git_sha)
       ├─ check .git exists (else "unknown")
       ├─ resolve HEAD SHA (use valid 40/64 hex git_sha override or `git rev-parse --verify HEAD`)
       ├─ superproject working tree + untracked (override showUntrackedFiles=normal, porcelain=v1,
       │  --ignore-submodules, excluding `cuda-keyring_*.deb` and `dist`) -> non-empty => "<sha>-dirty"
       ├─ superproject staged index (`diff-index --cached --quiet HEAD --`) -> non-zero => "<sha>-dirty"
       └─ submodulesClean(allocator, io, root)
            ├─ `git submodule status`: `+` or `U` => dirty; `-` => skip uninitialized
            ├─ non-MLX submodule (`lib/ds4`, `lib/opencode2-mlx-serve`, ...):
            │    `git -c status.showUntrackedFiles=normal status --porcelain=v1` -> non-empty => dirty
            └─ MLX submodule (`lib/mlx-src`, `lib/mlxc-src`):
                 ├─ `git diff --cached --name-only` -> non-empty => dirty
                 ├─ `changed = git -c core.quotePath=false diff --name-only`
                 ├─ `untracked = git -c core.quotePath=false ls-files --others --exclude-standard`
                 └─ if (changed != empty or untracked != empty):
                      matchesExpectedMlxPatches(...) -> false => dirty
```

---

## Failure Matrix

| # | Scenario | Expected `build_id` | Mechanism enforcing expectation |
|---|---|---|---|
| 1 | Clean checkout matching `HEAD` | `<sha>` | Superproject status empty, staged empty, submodules clean |
| 2 | Explicit valid `git_sha` override (`-Dgit-sha=...`) on clean tree | `<override_sha>` | `isGitSha(git_sha)` honored after cleanliness checks |
| 3 | Missing `.git` metadata (tarball / non-git export) | `"unknown"` | `root.access(io, ".git", .{})` fails |
| 4 | CI artifacts `cuda-keyring_*.deb` or `dist/...` present | `<sha>` | Pathspec `:(exclude)cuda-keyring_*.deb` and `:(exclude)dist` |
| 5 | Superproject tracked edit or staged change | `<sha>-dirty` | `status --porcelain=v1` and `diff-index --cached --quiet HEAD --` |
| 6 | Untracked plugin `src/plugins/new_plugin/plugin.zig` with `status.showUntrackedFiles=no` | `<sha>-dirty` | `-c status.showUntrackedFiles=normal` overrides repo/global config |
| 7 | Non-MLX submodule (`lib/ds4`) tracked edit or untracked file with `status.showUntrackedFiles=no` | `<sha>-dirty` | Per-submodule `-c status.showUntrackedFiles=normal status --porcelain=v1` |
| 8 | Moved submodule HEAD (`+`) or conflict (`U`) | `<sha>-dirty` | `git submodule status` prefix check (`+`, `U`) |
| 9 | Staged MLX submodule gitlink update (`git add lib/mlx-src`) | `<sha>-dirty` | `git diff-index --cached --quiet HEAD --` |
| 10 | Official macOS patches (`scripts/build-mlx-patches.txt`) applied to `lib/mlx-src` + `lib/mlxc-src` | `<sha>` | `collectMacosPatchSet` -> `verifyPatchCandidate` |
| 11 | Official Linux CUDA patches (`scripts/build-mlx-linux.sh`), incl. newly created `row_dot.cuh` / `row_dot_ops.cuh` | `<sha>` | `collectLinuxPatchSet` + `untracked` included in `dirty_files` & `cand_dir` `ls-files -s` |
| 12 | Official Windows patches (`scripts/build-mlx-windows.ps1`, `--ignore-whitespace --exclude=tests/* --exclude=docs/*`) | `<sha>` | `collectWindowsPatchSet` + `.windows` mode |
| 13 | Windows `core.autocrlf=true` CRLF checkout + CRLF patches + `\` path separators | `<sha>` | No `Dir.walk`, no raw byte `mem.eql`; Git `ls-files -s` (forward `/`, no `.git\`) + `hash-object` clean filters |
| 14 | Unrelated edit inside `lib/mlx-src` (`nax.h`) alone or alongside official patches | `<sha>-dirty` | `nax.h` `hash-object` != `HEAD:nax.h` blob SHA in `cand_dir` |
| 15 | Extra untracked file (`rogue.cpp`) inside `lib/mlx-src` alongside official patches | `<sha>-dirty` | `rogue.cpp` in `dirty_files` absent from `cand_dir` `ls-files -s` |
| 16 | Partial patch set or hand-modified patched line (incl. CRLF/content edit under `autocrlf=true`) | `<sha>-dirty` | Candidate patch set fails `git apply` or `hash-object` SHA mismatches |
| 17 | Deleted tracked file in `lib/mlx-src` not deleted by any official patch | `<sha>-dirty` | Missing in `sub_dir` (`access` fails) while present in `cand_dir` `ls-files -s` |
| 18 | Concurrent `zig build` invocations checking MLX patches | `<sha>` (no race) | `openUniqueStageDir` (`io.random` + exclusive `createDir`) |
| 19 | Global `commit.gpgsign=true` when running `zig build test` | All tests pass | Fixture commits use `-c commit.gpgsign=false` |

---

## Verification Plan

1. Unit tests in `build/config.zig` (run via `zig test build/config.zig` and `zig build test -Dtest-filter="resolveBuildId"` / `build_id` under `/private/tmp/claude/zig-build.lock`):
   - Existing assertions: clean HEAD, override SHA, `cuda-keyring_*.deb` + `dist/`, macOS patches, `nax.h` edit, `lib/ds4` edit, moved & staged submodule gitlinks, `status.showUntrackedFiles=no` (superproject plugin + `lib/ds4` untracked), Linux CUDA patch set + `new file mode` (`row_dot.cuh`) + `nax.h` + `rogue.cpp`.
   - New assertions for the redesign:
     - **Windows `core.autocrlf=true` + CRLF patches + `--exclude=tests/* --exclude=docs/*`**: clone `lib/mlx-src` & `lib/mlxc-src` with `core.autocrlf=true` (CRLF bytes on disk), apply CRLF-encoded `patches/mlx-*.patch` and `patches/mlxc-*.patch` (leaving `tests/` and `docs/` untouched), verify `resolveBuildId` returns `<head_sha>`, and verify editing `nax.h` (even with CRLF) returns `<head_sha>-dirty`.
     - **Deleted file inside `lib/mlx-src`**: deleting a tracked file flags `<head_sha>-dirty`.
2. `python3 scripts/lint.py --strict` exits 0.
