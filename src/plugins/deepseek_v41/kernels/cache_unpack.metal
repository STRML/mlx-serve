const uint group = thread_position_in_grid.x;
const uint rows = uint(packed_shape[0]);
const uint row_bytes = uint(packed_shape[1]);
const uint width = ENC == 1 ? row_bytes * 16u / 9u : ENC == 2 ? row_bytes * 32u / 17u : row_bytes * 32u / 33u;
const uint groups_per_row = width / G;
if (group >= rows * groups_per_row) return;
const uint row = group / groups_per_row;
const uint group_in_row = group - row * groups_per_row;
const uint code_bytes = ENC == 3 ? width : width / 2;
const uint input = row * row_bytes;
const uchar scale_byte = packed[input + code_bytes + group_in_row];
bool valid = scale_byte != 0;
float scale = 0.0f;
if (ENC == 1) {
  valid = valid && (scale_byte & 0x80u) == 0 && scale_byte <= 0x7eu;
  scale = dsv41_cache_e4m3_value(scale_byte);
} else {
  valid = valid && scale_byte != 0xffu;
  scale = as_type<float>(uint(scale_byte) << 23);
}
if (valid && ENC == 3) {
  for (uint i = 0; i < G; ++i) {
    const uchar code = packed[input + group_in_row * G + i];
    valid = valid && (code & 0x7fu) != 0x7fu;
  }
}
for (uint i = 0; i < G; ++i) {
  const uint offset = row * width + group_in_row * G + i;
  if (!valid) {
    out[offset] = 0;
  } else if (ENC == 3) {
    const uchar code = packed[input + group_in_row * G + i];
    out[offset] = dsv41_cache_bf16(dsv41_cache_e4m3_value(code) * scale, uint(code >> 7));
  } else {
    const uchar byte = packed[input + (group_in_row * G + i) / 2];
    const uchar code = (i & 1u) == 0 ? byte & 0x0fu : byte >> 4;
    const float value = dsv41_cache_e2m1_value(code) * scale;
    out[offset] = dsv41_cache_bf16(value, uint(code >> 3));
  }
}
status[group] = valid ? 0u : 1u;
