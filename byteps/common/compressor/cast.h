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

#ifndef BYTEPS_COMPRESSOR_CAST_H
#define BYTEPS_COMPRESSOR_CAST_H

#include "../cpu_reducer.h"
#include "compressor.h"

namespace byteps {
namespace common {
namespace compressor {
/*!
 * \brief Wrapper of Compressor to deal with low-precision types
 *
 * During summation, data in low-precision suffers from overflow problem.
 * To solve the issue, we instead use float32 as our data type for inter-
 * mediate buffers. When intra-node all-reduce is over, locally aggregated
 * gradients are first transformed into float32 via the wrapper.
 *
 * The wrapper has an internel fp32 buffer to store the transformed data.
 *
 * \sa Compressor
 */
class Cast : public Compressor {
 public:
  Cast(size_t size, DataType dtype, std::unique_ptr<Compressor> cptr)
      : Compressor(size, dtype),
        _fp32_buf(new byte_t[size]()),
        _cptr(std::move(cptr)){};
  ~Cast() override = default;

  tensor_t Compress(tensor_t grad) final;

  tensor_t Decompress(tensor_t compressed) final;

 protected:
  /*!
   * \brief Cast low-precision gradients into float32
   *
   * \param grad input gradient
   * \return tensor_t fp32 gradient
   */
  virtual tensor_t CastToFP32(tensor_t grad) = 0;

 protected:
  /*! \brief buffer of Cast */
  std::unique_ptr<byte_t[]> _fp32_buf;

 private:
  /*! \brief compressor pointer */
  std::unique_ptr<Compressor> _cptr;
};
}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESSOR_CAST_H