// dec_moe_fp4_combine_rg: dec_moe_fp4_down_bw's combine over the partials dec_moe_fp4_down_rg wrote. One thread per
// output (row r, hc copy h, column d): the column's slots added in today's order (picks 0 .. TOPK - 1, then the shared
// expert), rounded to T where `hcExpand` reads y, then the same fma chain into the next stream.
const uint i = thread_position_in_grid.x;
if (i >= uint(R * HC * N)) return;
const uint d = i % uint(N);
const uint h = (i / uint(N)) % uint(HC);
const uint r = i / uint(N * HC);
float s = 0.0f;
for (int j = 0; j <= TOPK; ++j) s += part[(r * (TOPK + 1) + j) * N + d];
const float y = float(T(s));
float a = 0.0f;
for (int j = 0; j < HC; ++j) a = metal::fma(comb[(r * HC + j) * HC + h], float(res[(r * HC + j) * N + d]), a);
out[(r * HC + h) * N + d] = T(metal::fma(post[r * HC + h], y, a));
