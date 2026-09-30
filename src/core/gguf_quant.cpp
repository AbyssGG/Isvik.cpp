#include "isvik/core/gguf_quant.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>

#include "gguf_iq3s_grid.h"

namespace isvik {
namespace {

constexpr std::array<int8_t, 16> kIq4NlValues{
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113};
constexpr std::array<int8_t, 16> kMxFp4Values{
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

uint16_t ReadU16(const uint8_t* bytes) {
  return static_cast<uint16_t>(bytes[0]) |
         static_cast<uint16_t>(static_cast<uint16_t>(bytes[1]) << 8U);
}

uint32_t ReadU32(const uint8_t* bytes) {
  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8U) |
         (static_cast<uint32_t>(bytes[2]) << 16U) |
         (static_cast<uint32_t>(bytes[3]) << 24U);
}

float HalfToFloat(uint16_t half) {
  const bool negative = (half & 0x8000U) != 0U;
  const uint16_t exponent = static_cast<uint16_t>((half >> 10U) & 0x1fU);
  const uint16_t mantissa = static_cast<uint16_t>(half & 0x03ffU);
  float value = 0.0F;
  if (exponent == 0U) {
    value = std::ldexp(static_cast<float>(mantissa), -24);
  } else if (exponent == 0x1fU) {
    value = mantissa == 0U ? std::numeric_limits<float>::infinity()
                           : std::numeric_limits<float>::quiet_NaN();
  } else {
    value = std::ldexp(1.0F + static_cast<float>(mantissa) / 1024.0F,
                       static_cast<int>(exponent) - 15);
  }
  return negative ? -value : value;
}

float E8M0HalfToFloat(uint8_t exponent) {
  // MXFP4's nibble lookup values are doubled E2M1 values, so the scale is
  // E8M0/2. Preserve the two exponent encodings below the normal range.
  uint32_t bits = 0;
  if (exponent < 2U) {
    bits = 0x00200000U << exponent;
  } else {
    bits = static_cast<uint32_t>(exponent - 1U) << 23U;
  }
  return std::bit_cast<float>(bits);
}

void DecodeQ5_1(const uint8_t* block, float* output) {
  const float scale = HalfToFloat(ReadU16(block));
  const float minimum = HalfToFloat(ReadU16(block + 2));
  const uint32_t high_bits = ReadU32(block + 4);
  const uint8_t* quantized = block + 8;
  for (uint32_t index = 0; index < 32U; ++index) {
    const uint32_t packed_index = index < 16U ? index : index - 16U;
    const uint32_t shift = index < 16U ? 0U : 4U;
    const uint32_t low = (quantized[packed_index] >> shift) & 0x0fU;
    const uint32_t high = (high_bits >> index) & 1U;
    output[index] = static_cast<float>(low | (high << 4U)) * scale + minimum;
  }
}

void DecodeQ8_0(const uint8_t* block, float* output) {
  const float scale = HalfToFloat(ReadU16(block));
  for (uint32_t index = 0; index < 32U; ++index) {
    const int8_t quantized = std::bit_cast<int8_t>(block[2U + index]);
    output[index] = static_cast<float>(quantized) * scale;
  }
}

void GetQ5KScaleMin(const uint8_t* scales, uint32_t index,
                    uint8_t* scale, uint8_t* minimum) {
  if (index < 4U) {
    *scale = static_cast<uint8_t>(scales[index] & 0x3fU);
    *minimum = static_cast<uint8_t>(scales[index + 4U] & 0x3fU);
    return;
  }
  *scale = static_cast<uint8_t>((scales[index + 4U] & 0x0fU) |
                                ((scales[index - 4U] >> 6U) << 4U));
  *minimum = static_cast<uint8_t>((scales[index + 4U] >> 4U) |
                                  ((scales[index] >> 6U) << 4U));
}

void DecodeQ5K(const uint8_t* block, float* output) {
  const float scale = HalfToFloat(ReadU16(block));
  const float minimum = HalfToFloat(ReadU16(block + 2));
  const uint8_t* scales = block + 4;
  const uint8_t* high_bits = block + 16;
  const uint8_t* low_bits = block + 48;
  for (uint32_t group = 0; group < 4U; ++group) {
    uint8_t sub_scale_0 = 0;
    uint8_t sub_minimum_0 = 0;
    uint8_t sub_scale_1 = 0;
    uint8_t sub_minimum_1 = 0;
    GetQ5KScaleMin(scales, group * 2U, &sub_scale_0, &sub_minimum_0);
    GetQ5KScaleMin(scales, group * 2U + 1U, &sub_scale_1, &sub_minimum_1);
    const float d0 = scale * static_cast<float>(sub_scale_0);
    const float m0 = minimum * static_cast<float>(sub_minimum_0);
    const float d1 = scale * static_cast<float>(sub_scale_1);
    const float m1 = minimum * static_cast<float>(sub_minimum_1);
    const uint32_t low_bit_shift = group * 2U;
    const uint32_t high_bit_shift = low_bit_shift + 1U;
    const uint8_t low_mask = static_cast<uint8_t>(1U << low_bit_shift);
    const uint8_t high_mask = static_cast<uint8_t>(1U << high_bit_shift);
    const uint8_t* packed = low_bits + group * 32U;
    for (uint32_t index = 0; index < 32U; ++index) {
      const float q0 = static_cast<float>((packed[index] & 0x0fU) +
          ((high_bits[index] & low_mask) != 0U ? 16U : 0U));
      const float q1 = static_cast<float>((packed[index] >> 4U) +
          ((high_bits[index] & high_mask) != 0U ? 16U : 0U));
      output[group * 64U + index] = d0 * q0 - m0;
      output[group * 64U + 32U + index] = d1 * q1 - m1;
    }
  }
}

void DecodeIq4Nl(const uint8_t* block, float* output) {
  const float scale = HalfToFloat(ReadU16(block));
  const uint8_t* quantized = block + 2;
  for (uint32_t index = 0; index < 16U; ++index) {
    output[index] = scale * static_cast<float>(kIq4NlValues[quantized[index] & 0x0fU]);
    output[index + 16U] = scale * static_cast<float>(kIq4NlValues[quantized[index] >> 4U]);
  }
}

uint8_t Iq3SGridValue(uint32_t grid_index, uint32_t byte_index) {
  const uint32_t packed = detail::kIq3SGrid[grid_index];
  return static_cast<uint8_t>((packed >> (byte_index * 8U)) & 0xffU);
}

void DecodeIq3S(const uint8_t* block, float* output) {
  const float block_scale = HalfToFloat(ReadU16(block));
  const uint8_t* quants = block + 2;
  const uint8_t* high_bits = block + 66;
  const uint8_t* signs = block + 74;
  const uint8_t* scales = block + 106;

  for (uint32_t sub_block = 0; sub_block < 8U; ++sub_block) {
    const uint8_t packed_scales = scales[sub_block / 2U];
    const uint32_t scale_shift = (sub_block & 1U) * 4U;
    const float scale = block_scale *
        static_cast<float>(1U + 2U * ((packed_scales >> scale_shift) & 0x0fU));
    const uint8_t high = high_bits[sub_block];
    for (uint32_t lane = 0; lane < 4U; ++lane) {
      const uint32_t base = sub_block * 8U;
      const uint32_t grid_0 = quants[base + lane * 2U] |
          ((static_cast<uint32_t>(high) << (8U - lane * 2U)) & 0x100U);
      const uint32_t grid_1 = quants[base + lane * 2U + 1U] |
          ((static_cast<uint32_t>(high) << (7U - lane * 2U)) & 0x100U);
      const uint8_t sign_bits = signs[sub_block * 4U + lane];
      for (uint32_t value = 0; value < 4U; ++value) {
        const uint32_t sign_mask = 1U << value;
        const float sign_0 = (sign_bits & sign_mask) != 0U ? -1.0F : 1.0F;
        const float sign_1 = (sign_bits & (sign_mask << 4U)) != 0U ? -1.0F : 1.0F;
        const uint32_t output_offset = sub_block * 32U + lane * 8U;
        output[output_offset + value] = scale *
            static_cast<float>(Iq3SGridValue(grid_0, value)) * sign_0;
        output[output_offset + value + 4U] = scale *
            static_cast<float>(Iq3SGridValue(grid_1, value)) * sign_1;
      }
    }
  }
}

void DecodeMxFp4(const uint8_t* block, float* output) {
  const float scale = E8M0HalfToFloat(block[0]);
  const uint8_t* packed = block + 1;
  for (uint32_t index = 0; index < 16U; ++index) {
    output[index] = scale * static_cast<float>(kMxFp4Values[packed[index] & 0x0fU]);
    output[index + 16U] = scale * static_cast<float>(kMxFp4Values[packed[index] >> 4U]);
  }
}

}  // namespace

std::optional<GgufQuantBlockInfo> GgufQuantBlockInfoForType(uint32_t type) {
  switch (type) {
    case 0: return GgufQuantBlockInfo{1U, 4U};    // F32
    case 1: return GgufQuantBlockInfo{1U, 2U};    // F16
    case 2: return GgufQuantBlockInfo{32U, 18U};  // Q4_0
    case 3: return GgufQuantBlockInfo{32U, 20U};  // Q4_1
    case 6: return GgufQuantBlockInfo{32U, 22U};  // Q5_0
    case 7: return GgufQuantBlockInfo{32U, 24U};  // Q5_1
    case 8: return GgufQuantBlockInfo{32U, 34U};  // Q8_0
    case 12: return GgufQuantBlockInfo{256U, 144U}; // Q4_K
    case 13: return GgufQuantBlockInfo{256U, 176U}; // Q5_K
    case 14: return GgufQuantBlockInfo{256U, 210U}; // Q6_K
    case 20: return GgufQuantBlockInfo{32U, 18U}; // IQ4_NL
    case 21: return GgufQuantBlockInfo{256U, 110U}; // IQ3_S
    case 39: return GgufQuantBlockInfo{32U, 17U}; // MXFP4
    default: return std::nullopt;
  }
}

bool IsGgufTensorEncodingDecodable(uint32_t type) {
  switch (type) {
    case 0:
    case 1:
    case 7:
    case 8:
    case 13:
    case 20:
    case 21:
    case 39:
      return true;
    default:
      return false;
  }
}

StatusOr<std::vector<float>> DecodeGgufTensorBlocks(
    uint32_t type, std::span<const uint8_t> encoded_blocks) {
  const auto info = GgufQuantBlockInfoForType(type);
  if (!info.has_value()) {
    return Status::Unsupported("GGUF tensor encoding is not supported by the native block decoder: " +
                               std::to_string(type));
  }
  if (!IsGgufTensorEncodingDecodable(type)) {
    return Status::Unsupported("GGUF tensor encoding has a known layout but no native decoder: " +
                               std::to_string(type));
  }
  if (encoded_blocks.size() % info->bytes != 0U) {
    return Status::InvalidArgument("GGUF tensor bytes do not contain complete quantization blocks");
  }
  const uint64_t block_count = encoded_blocks.size() / info->bytes;
  if (block_count > std::numeric_limits<size_t>::max() / info->elements) {
    return Status::InvalidArgument("decoded GGUF tensor range is too large");
  }
  if (block_count * info->elements > kMaxGgufCpuDecodeElements) {
    return Status::InvalidArgument("GGUF CPU decoder range exceeds its bounded output size");
  }
  std::vector<float> output(static_cast<size_t>(block_count * info->elements));
  for (uint64_t block_index = 0; block_index < block_count; ++block_index) {
    const uint8_t* source = encoded_blocks.data() +
        static_cast<size_t>(block_index * info->bytes);
    float* destination = output.data() +
        static_cast<size_t>(block_index * info->elements);
    switch (type) {
      case 0:
        destination[0] = std::bit_cast<float>(ReadU32(source));
        break;
      case 1:
        destination[0] = HalfToFloat(ReadU16(source));
        break;
      case 7: DecodeQ5_1(source, destination); break;
      case 8: DecodeQ8_0(source, destination); break;
      case 13: DecodeQ5K(source, destination); break;
      case 20: DecodeIq4Nl(source, destination); break;
      case 21: DecodeIq3S(source, destination); break;
      case 39: DecodeMxFp4(source, destination); break;
      default:
        return Status::Internal("GGUF decoder block type dispatch is inconsistent");
    }
  }
  return output;
}

}  // namespace isvik
