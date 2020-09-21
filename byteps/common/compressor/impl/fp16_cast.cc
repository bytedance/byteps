// Copyright 2020 Amazon Inc. or its affiliates. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "fp16_cast.h"

#include "../compressor_registry.h"

namespace byteps {
namespace common {
namespace compressor {
namespace {
CompressorRegistry::Register reg(
    "fp16_cast",
    [](const kwargs_t& kwargs, size_t size, DataType dtype,
       std::unique_ptr<Compressor> cptr) -> std::unique_ptr<Compressor> {
      // register cptr
      BPS_CHECK_NE(cptr, nullptr);
      BPS_CHECK_EQ(dtype, BYTEPS_FLOAT16);

      BPS_LOG(INFO) << "fp16 cast is registered.";
      return std::unique_ptr<FP16CastCompressor>(
          new FP16CastCompressor(size, dtype, std::move(cptr)));
    });
}
#if __F16C__
tensor_t FP16CastCompressor::CastToFP32(tensor_t grad) {
  auto src = reinterpret_cast<half_t*>(grad.data);
  auto dst = reinterpret_cast<float*>(_fp32_buf.get());
  size_t len = grad.size / sizeof(half_t);

#pragma omp parallel for simd
  for (size_t i = 0; i < len; ++i) {
    dst[i] = src[i];
  }

  // use BYTEPS_FLOAT32 as dtype in the following compression
  return {dst, len * sizeof(float), BYTEPS_FLOAT32};
}
#else
tensor_t FP16CastCompressor::CastToFP32(tensor_t grad) { return grad; }
#endif
}  // namespace compressor
}  // namespace common
}  // namespace byteps