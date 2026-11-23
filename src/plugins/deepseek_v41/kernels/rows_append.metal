// rows_append: writes `n` rows of `src` into `dst` at row offset `off` in place and emits a 1-word carrier for event gating.
const uint tid = thread_position_in_grid.x;
const uint off = meta[0];
const uint w = meta[1];
const uint n = meta[2];
const size_t total = size_t(n) * size_t(w);
if (size_t(tid) < total) {
  ((device T*)dst)[size_t(off) * size_t(w) + size_t(tid)] = src[tid];
}
if (tid == 0) carrier[0] = 0u;
