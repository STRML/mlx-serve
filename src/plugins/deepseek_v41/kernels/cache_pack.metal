const uint group = thread_position_in_grid.x;
const uint rows = uint(x_shape[0]);
const uint width = uint(x_shape[1]);
const uint groups_per_row = width / G;
if (group >= rows * groups_per_row) return;
const uint row = group / groups_per_row;
const uint group_in_row = group - row * groups_per_row;
const uint first = row * width + group_in_row * G;
const uint code_bytes = ENC == 3 ? width : width / 2;
const uint row_bytes = code_bytes + groups_per_row;
const uint output = row * row_bytes;
float amax = 0.0f;
bool valid = true;
for (uint i = 0; i < G; ++i) {
  const float value = as_type<float>(uint(x[first + i]) << 16);
  valid = valid && metal::isfinite(value);
  amax = metal::max(amax, metal::abs(value));
}
float scale;
uchar scale_byte;
if (ENC == 1) {
  scale = dsv41_fp8_roundtrip(metal::max(amax, 6.0f * 0.001953125f) / 6.0f);
  // The writer scales by the max before rounding; under a subnormal scale that max can round down to 4 x scale.
  if (amax > 6.0f * scale && dsv41_fp8_roundtrip(amax * 0.25f) == amax * 0.25f) scale = amax * 0.25f;
  scale_byte = dsv41_cache_e4m3_bits(scale);
} else if (ENC == 2) {
  scale = dsv41_pow2ceil(metal::max(amax, 6.0f * 1.17549435e-38f) * (1.0f / 6.0f));
  scale_byte = uchar(as_type<uint>(scale) >> 23);
} else {
  scale = dsv41_pow2ceil(metal::max(amax, 1e-4f) * (1.0f / 448.0f));
  scale_byte = uchar(as_type<uint>(scale) >> 23);
}
uint mismatch = 0;
for (uint i = 0; i < G; i += (ENC == 3 ? 1 : 2)) {
  const uint ix = first + i;
  if (ENC == 3) {
    const uint sign = uint(x[ix] >> 15);
    const float value = as_type<float>(uint(x[ix]) << 16);
    const uchar code = dsv41_cache_e4m3_bits(metal::clamp(value / scale, -448.0f, 448.0f));
    out[output + group_in_row * G + i] = code;
    const float restored = dsv41_cache_e4m3_value(code) * scale;
    mismatch |= uint(dsv41_cache_bf16(restored, sign) != x[ix]);
  } else {
    const uint i0 = ix;
    const uint i1 = ix + 1;
    const uint sign0 = uint(x[i0] >> 15);
    const uint sign1 = uint(x[i1] >> 15);
    const float value0 = as_type<float>(uint(x[i0]) << 16);
    const float value1 = as_type<float>(uint(x[i1]) << 16);
    const uchar level0 = dsv41_cache_e2m1_index(metal::clamp(value0 / scale, -6.0f, 6.0f));
    const uchar level1 = dsv41_cache_e2m1_index(metal::clamp(value1 / scale, -6.0f, 6.0f));
    const uchar code0 = uchar((sign0 << 3) | level0);
    const uchar code1 = uchar((sign1 << 3) | level1);
    out[output + (group_in_row * G + i) / 2] = code0 | uchar(code1 << 4);
    mismatch |= uint(dsv41_cache_bf16(dsv41_cache_e2m1_value(level0) * scale, sign0) != x[i0]);
    mismatch |= uint(dsv41_cache_bf16(dsv41_cache_e2m1_value(level1) * scale, sign1) != x[i1]);
  }
}
out[output + code_bytes + group_in_row] = valid && mismatch == 0 ? scale_byte : 0;
