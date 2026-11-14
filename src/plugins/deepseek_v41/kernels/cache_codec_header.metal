inline uchar dsv41_cache_e4m3_bits(float f) {
  uint fb = as_type<uint>(f);
  const uint sign = fb & 0x80000000u;
  fb ^= sign;
  uchar bits;
  if (fb >= (543u << 21)) bits = 0x7e;
  else if (fb < (121u << 23)) {
    const uint dm = 141u << 23;
    bits = uchar(as_type<uint>(as_type<float>(fb) + as_type<float>(dm)) - dm);
  } else {
    const uint odd = (fb >> 20) & 1u;
    fb += (uint(7 - 127) << 23) + 0x7ffffu;
    fb += odd;
    bits = uchar(fb >> 20);
  }
  return bits | uchar(sign >> 24);
}

inline float dsv41_cache_e4m3_value(uchar bits) {
  const float sign = (bits & 0x80u) == 0 ? 1.0f : -1.0f;
  const uint exponent = (bits >> 3) & 0x0fu;
  const uint mantissa = bits & 7u;
  if (exponent == 0) return sign * float(mantissa) * 0x1p-9f;
  return sign * (1.0f + float(mantissa) * 0.125f) * exp2(float(exponent) - 7.0f);
}

inline uchar dsv41_cache_e2m1_index(float value) {
  const float m = metal::abs(value);
  return uchar(int(m > 0.25f) + int(m > 1.25f) + int(m > 2.5f) + int(m > 5.0f) +
               int(m >= 0.75f) + int(m >= 1.75f) + int(m >= 3.5f));
}

inline float dsv41_cache_e2m1_value(uchar level) {
  const float values[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  return values[level & 7u];
}

inline ushort dsv41_cache_bf16(float value, uint sign) {
  uint bits = as_type<uint>(metal::abs(value));
  return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16) | ushort(sign << 15);
}
