# Redesign Plan: Build Identity (`feat/report-build-id`, PR #53)

## Context

`build_options.build_id` (computed by `readBuildId` / `resolveBuildId` in `build/config.zig` and exposed on `/v1/internal/mode` and `/props`) must uniquely identify the exact source code compiled into `mlx-serve`:

- `<40/64-char HEAD SHA>` when the checkout and all embedded/staged submodules match committed `HEAD` plus at most the expected root-anchored build/CI artifacts (`cuda-keyring_*.deb`, `dist/`, `.work-mlx-windows/`, `.zig-cache/`, `zig-out/`, etc.) and a complete, matched platform MLX/mlx-c patch pair (`scripts/build-mlx.sh`, `scripts/build-mlx-linux.sh`, `scripts/build-mlx-windows.ps1`).
- `<HEAD SHA>-dirty` on any uncommitted superproject edit, staged change (including staged submodule gitlinks), untracked source/plugin file (even under `status.showUntrackedFiles=no`, `core.excludesFile`, `.git/info/exclude`, or `src/plugins/dist/`), index-flagged edit (`assume-unchanged` / `skip-worktree`), moved/conflicted submodule HEAD, populated submodule directory missing `.git`, dirty non-MLX submodule (`lib/ds4`, `lib/opencode2-mlx-serve`), or unapproved/partial/mixed/extra edit, mode change, deletion, or untracked file inside `lib/mlx-src` or `lib/mlxc-src`, as well as any post-HEAD query/stat failure.
- `"unknown"` only when `.git` / `HEAD` metadata itself is unavailable.

Rounds 1–5 surfaced recurring loopholes because `git status`, `git diff --name-only`, and `git ls-files --others` honor `.git/index` skip flags (`assume-unchanged`, `skip-worktree`), ignore files (`core.excludesFile`, `.git/info/exclude`, `.gitignore`), and unanchored pathspecs, while validating `lib/mlx-src` and `lib/mlxc-src` independently allowed partial/mixed patch sets. This redesign eliminates those defect classes structurally.

---

## Findings Inventory (Rounds 1–5) & Root-Cause Grouping

### Group A — Relying on `git status` / `.git/index` / ignore rules / unanchored exclusions
1. **Round 1 `[P1]` — Linux/Windows release builds report clean source as dirty (`cuda-keyring_*.deb`, `dist/`, `.work-mlx-windows/`)** + **Round 5 `[P1]` — Overbroad `:(exclude)dist` hides `src/plugins/dist/plugin.zig`**:
   Unanchored `:(exclude)dist` matched `src/plugins/dist/` as well as root `dist/`.
2. **Round 1 `[P1]` — Modified embedded submodule code reports a clean build (`lib/ds4`, `lib/opencode2-mlx-serve`)**:
   `--ignore-submodules=dirty` hid working-tree edits inside embedded submodules.
3. **Round 2 `[P1]` — Modified embedded MLX submodule code reports a clean build (`nax.h`)**:
   Pathspec exclusions hid `lib/mlx-src` edits (`lib/mlx_steel_sources.zig`).
4. **Round 2 `[P1]` — Staged MLX submodule updates report a clean build**:
   Excluding MLX submodules missed staged gitlink changes vs `HEAD`.
5. **Round 2 `[P1]` / Round 3 `[P1]` / Round 5 `[P1]` — Untracked plugin/source hidden by `--untracked-files=no`, `status.showUntrackedFiles=no`, `.gitignore`, `.git/info/exclude`, or global `core.excludesFile`**:
   `git status` and `git ls-files --others --exclude-standard` both obey ignore rules and user config, hiding `src/plugins/new_plugin/` and `lib/mlx-src/rogue.cpp`.
6. **Round 5 `[P1]` — `assume-unchanged` / `skip-worktree` index flags hide edits to compiled files**:
   `.git/index` flags cause `git status` and `git diff` to skip modified tracked files in the superproject and submodules.
7. **Round 5 `[P1]` — Populated uninitialized submodules skipped (`git submodule status` `-`)**:
   An extracted archive directory in `lib/ds4/` or `lib/mlx-src/` without `.git` reports `-` in `git submodule status`, yet `build/libs.zig` and `lib/mlx_steel_sources.zig` still compile/embed it.

**Design change eliminating Group A**:
- **Zero `git status`, zero `git diff --name-only`, zero `git ls-files --others`, zero `:(exclude)` pathspecs.**
- **Staged changes**: checked directly via `git --no-optional-locks diff-index --cached --quiet HEAD --` in superproject and initialized submodules.
- **Tracked working-tree changes, deletions, and mode changes**: checked against a **fresh temporary `GIT_INDEX_FILE`** populated via `git read-tree HEAD` + `git update-index -q --refresh` + `git diff-index --name-only` / `--summary HEAD --`. Completely ignores `.git/index` (`assume-unchanged` / `skip-worktree`), honors `HEAD` CRLF (`legacy_http.golden`), `core.autocrlf`, `.gitattributes`, symlinks, and `core.filemode`, and never mutates the repo's `.git/`.
- **Untracked files**: checked directly on disk via root-relative directory recursion (`Dir.iterate`) compared against `git ls-tree -r -z HEAD` (`head_paths` + `head_submodules`):
  - Superproject skips only top-level `.git`, `head_submodules`, and an exact **root-anchored** list of build/CI/tooling paths (root `dist`, root `.work-mlx-windows`, root `.work-mlx-linux*`, root `cuda-keyring_*.deb`, `.zig-cache`, `zig-out`, `lib/mlx`, `lib/llama`, `src/plugins/registry.zig`, `src/unit_*.zig`, etc.) — `src/plugins/dist/` and `core.excludesFile` / `.git/info/exclude` entries NEVER bypass it.
  - MLX submodules allow NO untracked files outside verified patch outputs (`rogue.cpp` under `core.excludesFile` is always caught). Non-MLX submodules allow only `*.o` inside `lib/ds4` (for `scripts/build-ds4-linux.sh`).
- **Submodules**: iterate `head_submodules` + `git submodule status`; fail on `+` / `U`; if a submodule directory exists without `.git`, allow ONLY if completely empty (`iter.next == null`); otherwise validate `.git` submodules normally even if `submodule status` marked `-`.

### Group B — Incomplete / uncoupled platform patch verification
8. **Round 3 `[P1]` — Linux CUDA builds report expected patches as dirty**:
   Supported only `scripts/build-mlx-patches.txt`, omitting `scripts/build-mlx-linux.sh` and `scripts/build-mlx-windows.ps1` + newly created `row_dot.cuh` / `row_dot_ops.cuh`.
9. **Round 5 `[P1]` — Partial and mixed platform patch sets report clean**:
   Validating `lib/mlx-src` and `lib/mlxc-src` independently allowed macOS MLX + untouched `lib/mlxc-src` or macOS MLX + Linux `lib/mlxc-src` to pass clean.

**Design change eliminating Group B**:
- Validate `(lib/mlx-src, lib/mlxc-src)` **jointly** against whole-platform candidates:
  1. **Both clean** (0 changed, 0 untracked in both).
  2. **macOS** (`scripts/build-mlx-patches.txt`): every submodule listed in the manifest must be present and match its listed patches (no short-circuit if a listed submodule has 0 changes), and any unlisted MLX submodule must be clean.
  3. **Linux CUDA** (`scripts/build-mlx-linux.sh`): BOTH `lib/mlx-src` (`8` `mlx-cuda-*.patch`) AND `lib/mlxc-src` (`3` `mlxc-*.patch`) must be present and match.
  4. **Linux CPU** (`MLX_BACKEND=cpu ./scripts/build-mlx-linux.sh`): `lib/mlx-src` must be clean AND `lib/mlxc-src` must match all `3` `mlxc-*.patch`.
  5. **Windows** (`scripts/build-mlx-windows.ps1`): BOTH `lib/mlx-src` (sorted non-`mlxc` `mlx*.patch`) AND `lib/mlxc-src` (sorted `mlxc-*.patch`) must match with `--ignore-whitespace --exclude=tests/* --exclude=docs/*` (assumes patches remain un-upstreamed against pinned `HEAD`).

### Group C — Manual filesystem traversal & raw-byte comparison fighting OS separators and CRLF
10. **Round 4 `[P1]` — Windows patch verification walks temporary `.git\`** & **CRLF mismatch under `core.autocrlf=true`**:
    - Inside `cand_dir`, configure `sub_dir`'s `core.autocrlf` + `.git/info/attributes` (`.gitattributes`), seed `HEAD:<rel>`, apply candidate patches, run `git -c core.safecrlf=false add -A`, and compare `git -c core.quotePath=false ls-files -s` (forward `/`, no `.git`) against read-only `git -c core.quotePath=false -c core.safecrlf=false hash-object --` on `sub_dir` (partitioning `dirty_files` into `existing` vs `deleted` via `statFile` before chunked hashing).

### Group D — Test coverage, config isolation, fail-closed errors & concurrent build races
11. **Round 1 `[P2]` — `commit.gpgsign=true`** -> test fixtures pass `-c commit.gpgsign=false`.
12. **Round 3 `[P2]` — Shared temp dir** -> `openUniqueStageDir` via `io.random` + exclusive `createDir` retry.
13. **Round 5 `[P1]` — Build-ID tests absent from Linux/Windows `zig build test`**:
    Wire `config.addTests(b, test_step, test_filter)` into BOTH `build/macos.zig` and `build/linux.zig` (which builds both Linux and Windows) and use `build_id` test names so `zig build test -Dtest-filter="build_id"` executes them on every platform.
14. **Round 5 `[P2]` — Status/query errors returned `"unknown"` after `HEAD` was read**:
    Once `HEAD` SHA is resolved, ALL subsequent command, IO, buffer-limit, or stat failures fail closed as `<sha>-dirty`.

---

## Failure Matrix

| # | Scenario | Expected `build_id` | Mechanism enforcing expectation |
|---|---|---|---|
| 1 | Clean checkout matching `HEAD` (incl. CRLF `legacy_http.golden` under `autocrlf=input`) | `<sha>` | Temp `GIT_INDEX_FILE` `diff-index` empty, root walk clean, submodules clean |
| 2 | Explicit valid `git_sha` override on clean tree | `<override_sha>` | `isGitSha(git_sha)` honored after cleanliness checks |
| 3 | Missing `.git` or invalid/missing `HEAD` | `"unknown"` | Initial `root.access(".git")` / `rev-parse HEAD` fails |
| 4 | Post-HEAD query failure (buffer > limit, unreadable dir, failed temp index) | `<sha>-dirty` | Fail-closed `dirtyId(allocator, sha)` after `HEAD` SHA is known |
| 5 | Root CI/build artifacts (`cuda-keyring_*.deb`, root `dist/`, root `.work-mlx-windows/`, `.zig-cache/`, `src/plugins/registry.zig`) | `<sha>` | Root-anchored exact allowlist in `isAllowedSuperprojectUntracked` |
| 6 | Unanchored path collision (`src/plugins/dist/plugin.zig` or `sub/cuda-keyring_1.deb`) | `<sha>-dirty` | Allowlist matches ONLY root `dist/` and root `cuda-keyring_*.deb` |
| 7 | Untracked plugin `src/plugins/new_plugin/plugin.zig` under `showUntrackedFiles=no`, `.git/info/exclude`, or `core.excludesFile` | `<sha>-dirty` | Direct `Dir` recursion vs `ls-tree -r -z HEAD` ignores all exclude files & status config |
| 8 | Tracked edit marked `--assume-unchanged` or `--skip-worktree` in superproject or any submodule | `<sha>-dirty` | Fresh `GIT_INDEX_FILE` (`read-tree HEAD` + `update-index --refresh` + `diff-index`) ignores `.git/index` flags |
| 9 | Staged edit or staged MLX submodule gitlink (`git add lib/mlx-src`) | `<sha>-dirty` | `git diff-index --cached --quiet HEAD --` |
| 10 | Moved submodule HEAD (`+`), conflict (`U`), or populated submodule dir missing `.git` | `<sha>-dirty` | `submodule status` check + non-empty check when `.git` is absent |
| 11 | Non-MLX submodule (`lib/ds4`) tracked edit or untracked `.c` under `showUntrackedFiles=no` / `core.excludesFile` (while allowing `build-ds4-linux.sh` `*.o`) | `<sha>-dirty` | Submodule fresh `GIT_INDEX_FILE` + `Dir` walk |
| 12 | Complete matched platform patches: macOS (`build-mlx-patches.txt`), Linux CUDA (`8` + `3` incl. `row_dot.cuh`), Linux CPU (`clean` + `3`), Windows (`core.autocrlf=true`, `safecrlf=true`, CRLF patches, `--exclude=tests/* --exclude=docs/*`) | `<sha>` | Joint `(lib/mlx-src, lib/mlxc-src)` candidate validation + Git `ls-files -s` / `hash-object` |
| 13 | Partial or mixed MLX patch sets (e.g., macOS `lib/mlx-src` + untouched or Linux `lib/mlxc-src`) | `<sha>-dirty` | Joint platform-pair validator requires full platform agreement |
| 14 | Unrelated edit (`nax.h`), `rogue.cpp` (even in `core.excludesFile`), partial edit under `autocrlf=true`, mode-only `chmod +x`, deleted tracked file, or nested `.git` in `lib/mlx-src` | `<sha>-dirty` | Fresh `GIT_INDEX_FILE` `diff-index`, `Dir` walk, `statFile` partitioning, `ls-files -s` vs `hash-object` |
| 15 | Concurrent `zig build` invocations & `commit.gpgsign=true` | `<sha>` / pass | `io.random` + `createDir` retry; `-c commit.gpgsign=false` in test commits |
| 16 | Running `zig build test -Dtest-filter=build_id` on macOS, Linux, or Windows | Runs & passes | `config.addTests(b, test_step, test_filter)` wired in BOTH `build/macos.zig` and `build/linux.zig` |

---

## Verification Plan

1. Run `zig test build/config.zig` and `zig build test -Dtest-filter="build_id" --summary all` (under `/private/tmp/claude/zig-build.lock`), verifying `build-config-test` runs on both `macos` and `linux`/`windows` build graphs and passes all cases covering matrix rows 1–16.
2. Run `python3 scripts/lint.py --strict` -> exit 0.
3. Verify all Sweeps from rounds 1–5 return 0 matches on `build/config.zig`.
