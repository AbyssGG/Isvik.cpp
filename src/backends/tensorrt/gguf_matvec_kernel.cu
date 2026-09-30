#include "gguf_matvec_kernel.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <set>

#include "../../core/gguf_iq3s_grid.h"
#include "isvik/core/gguf_quant.h"

namespace isvik::tensorrt_backend::detail {
namespace {

__device__ __constant__ uint32_t kIq3SGridDevice[512];

__device__ uint16_t ReadU16Device(const uint8_t* bytes) {
  return static_cast<uint16_t>(bytes[0]) |
         static_cast<uint16_t>(static_cast<uint16_t>(bytes[1]) << 8U);
}

__device__ uint32_t ReadU32Device(const uint8_t* bytes) {
  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8U) |
         (static_cast<uint32_t>(bytes[2]) << 16U) |
         (static_cast<uint32_t>(bytes[3]) << 24U);
}

__device__ float HalfToFloatDevice(uint16_t half) {
  const bool negative = (half & 0x8000U) != 0U;
  const uint16_t exponent = static_cast<uint16_t>((half >> 10U) & 0x1fU);
  const uint16_t mantissa = static_cast<uint16_t>(half & 0x03ffU);
  float value = 0.0F;
  if (exponent == 0U) {
    value = ldexpf(static_cast<float>(mantissa), -24);
  } else if (exponent == 0x1fU) {
    value = mantissa == 0U ? __int_as_float(0x7f800000) : __int_as_float(0x7fc00000);
  } else {
    value = ldexpf(1.0F + static_cast<float>(mantissa) / 1024.0F,
                   static_cast<int>(exponent) - 15);
  }
  return negative ? -value : value;
}

__device__ float E8M0HalfToFloatDevice(uint8_t exponent) {
  uint32_t bits = 0U;
  if (exponent < 2U) {
    bits = 0x00200000U << exponent;
  } else {
    bits = static_cast<uint32_t>(exponent - 1U) << 23U;
  }
  return __uint_as_float(bits);
}

__device__ void GetQ5KScaleMinDevice(const uint8_t* scales, uint32_t index,
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

__device__ float DecodeWeight(uint32_t type, const uint8_t* row,
                              uint64_t element) {
  if (type == 0U) {
    return __uint_as_float(ReadU32Device(row + element * 4U));
  }
  if (type == 1U) {
    return HalfToFloatDevice(ReadU16Device(row + element * 2U));
  }
  if (type == 7U) {
    const uint8_t* block = row + (element / 32U) * 24U;
    const uint32_t within = static_cast<uint32_t>(element % 32U);
    const uint32_t index = within < 16U ? within : within - 16U;
    const uint32_t shift = within < 16U ? 0U : 4U;
    const uint32_t low = (block[8U + index] >> shift) & 0x0fU;
    const uint32_t high = (ReadU32Device(block + 4U) >> within) & 1U;
    return static_cast<float>(low | (high << 4U)) * HalfToFloatDevice(ReadU16Device(block)) +
           HalfToFloatDevice(ReadU16Device(block + 2U));
  }
  if (type == 8U) {
    const uint8_t* block = row + (element / 32U) * 34U;
    const int8_t quantized = static_cast<int8_t>(block[2U + element % 32U]);
    return static_cast<float>(quantized) * HalfToFloatDevice(ReadU16Device(block));
  }
  if (type == 13U) {
    const uint32_t within = static_cast<uint32_t>(element % 256U);
    const uint32_t group = within / 64U;
    const uint32_t lane = within % 32U;
    const uint8_t* block = row + (element / 256U) * 176U;
    const uint8_t* scales = block + 4U;
    const uint8_t* high_bits = block + 16U;
    const uint8_t* low_bits = block + 48U;
    const uint32_t scale_index = group * 2U + (within >= group * 64U + 32U ? 1U : 0U);
    uint8_t sub_scale = 0U;
    uint8_t sub_minimum = 0U;
    GetQ5KScaleMinDevice(scales, scale_index, &sub_scale, &sub_minimum);
    const uint32_t high_bit_index = group * 2U + (within >= group * 64U + 32U ? 1U : 0U);
    const uint8_t high_mask = static_cast<uint8_t>(1U << high_bit_index);
    const uint8_t packed = low_bits[group * 32U + lane];
    const uint32_t quantized = within < group * 64U + 32U
        ? (packed & 0x0fU) + ((high_bits[lane] & high_mask) != 0U ? 16U : 0U)
        : (packed >> 4U) + ((high_bits[lane] & high_mask) != 0U ? 16U : 0U);
    return HalfToFloatDevice(ReadU16Device(block)) * static_cast<float>(sub_scale) *
               static_cast<float>(quantized) -
           HalfToFloatDevice(ReadU16Device(block + 2U)) * static_cast<float>(sub_minimum);
  }
  if (type == 20U) {
    const uint8_t* block = row + (element / 32U) * 18U;
    const uint32_t within = static_cast<uint32_t>(element % 32U);
    const uint8_t packed = block[2U + within % 16U];
    const uint32_t quant_index = within < 16U ? packed & 0x0fU : packed >> 4U;
    constexpr int8_t values[16]{-127, -104, -83, -65, -49, -35, -22, -10,
                                1, 13, 25, 38, 53, 69, 89, 113};
    return HalfToFloatDevice(ReadU16Device(block)) * static_cast<float>(values[quant_index]);
  }
  if (type == 21U) {
    const uint32_t within = static_cast<uint32_t>(element % 256U);
    const uint32_t sub_block = within / 32U;
    const uint32_t lane = within % 32U;
    const uint32_t group = lane / 8U;
    const uint32_t position = lane % 8U;
    const uint8_t* block = row + (element / 256U) * 110U;
    const uint8_t* quants = block + 2U;
    const uint8_t* high_bits = block + 66U;
    const uint8_t* signs = block + 74U;
    const uint8_t* scales = block + 106U;
    const uint32_t packed_scale = scales[sub_block / 2U];
    const uint32_t scale_shift = (sub_block & 1U) * 4U;
    const float scale = HalfToFloatDevice(ReadU16Device(block)) *
        static_cast<float>(1U + 2U * ((packed_scale >> scale_shift) & 0x0fU));
    const uint8_t high = high_bits[sub_block];
    const uint32_t grid = quants[sub_block * 8U + group * 2U + (position >= 4U ? 1U : 0U)] |
        ((static_cast<uint32_t>(high) <<
          ((position < 4U ? 8U : 7U) - group * 2U)) & 0x100U);
    const uint32_t code = (kIq3SGridDevice[grid] >> ((position % 4U) * 8U)) & 0xffU;
    const uint8_t sign_bits = signs[sub_block * 4U + group];
    const uint32_t sign_index = position < 4U ? position : position;
    const float sign = (sign_bits & (1U << sign_index)) != 0U ? -1.0F : 1.0F;
    return scale * static_cast<float>(code) * sign;
  }
  if (type == 39U) {
    const uint8_t* block = row + (element / 32U) * 17U;
    const uint32_t within = static_cast<uint32_t>(element % 32U);
    const uint8_t packed = block[1U + within % 16U];
    const uint32_t quant_index = within < 16U ? packed & 0x0fU : packed >> 4U;
    constexpr int8_t values[16]{0, 1, 2, 3, 4, 6, 8, 12,
                                0, -1, -2, -3, -4, -6, -8, -12};
    return E8M0HalfToFloatDevice(block[0]) * static_cast<float>(values[quant_index]);
  }
  return 0.0F;
}

__global__ void GgufQuantizedMatVecKernel(
    uint32_t type, uint64_t input_features, uint64_t output_features,
    uint64_t row_bytes, const float* activations,
    const uint8_t* packed_weights, float* output) {
  const uint64_t result_index = blockIdx.x;
  const uint64_t batch_index = result_index / output_features;
  const uint64_t output_index = result_index % output_features;
  const uint8_t* weight_row = packed_weights + output_index * row_bytes;
  float partial = 0.0F;
  for (uint64_t index = threadIdx.x; index < input_features; index += blockDim.x) {
    const float weight = DecodeWeight(type, weight_row, index);
    partial = fmaf(weight, activations[batch_index * input_features + index], partial);
  }
  __shared__ float sums[256];
  sums[threadIdx.x] = partial;
  __syncthreads();
  for (uint32_t stride = 128U; stride > 0U; stride >>= 1U) {
    if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0U) output[result_index] = sums[0];
}

cudaError_t EnsureIq3SGridInitialized() {
  static std::mutex mutex;
  static std::set<int> initialized_devices;
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  std::lock_guard lock(mutex);
  if (initialized_devices.contains(device)) return cudaSuccess;
  status = cudaMemcpyToSymbol(kIq3SGridDevice, isvik::detail::kIq3SGrid.data(),
                              sizeof(isvik::detail::kIq3SGrid));
  if (status != cudaSuccess) return status;
  initialized_devices.insert(device);
  return cudaSuccess;
}

}  // namespace

cudaError_t LaunchGgufQuantizedMatVec(
    uint32_t tensor_type, uint64_t input_features, uint64_t output_features,
    uint64_t batch_size, const float* activations,
    const uint8_t* packed_weights, float* output, cudaStream_t stream) {
  const auto block = isvik::GgufQuantBlockInfoForType(tensor_type);
  if (!isvik::IsGgufTensorEncodingDecodable(tensor_type) || !block.has_value() ||
      activations == nullptr || packed_weights == nullptr || output == nullptr ||
      input_features == 0U || output_features == 0U || batch_size == 0U ||
      input_features % block->elements != 0U ||
      output_features > std::numeric_limits<uint64_t>::max() / batch_size ||
      output_features * batch_size > std::numeric_limits<uint32_t>::max()) {
    return cudaErrorInvalidValue;
  }
  const uint64_t row_bytes = (input_features / block->elements) * block->bytes;
  const uint64_t total_results = batch_size * output_features;
  const cudaError_t init_status = EnsureIq3SGridInitialized();
  if (init_status != cudaSuccess) return init_status;
  GgufQuantizedMatVecKernel<<<static_cast<uint32_t>(total_results), 256U, 0U, stream>>>(
      tensor_type, input_features, output_features, row_bytes,
      activations, packed_weights, output);
  return cudaGetLastError();
}

}  // namespace isvik::tensorrt_backend::detail
