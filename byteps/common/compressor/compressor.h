// Copyright 2019 Amazon Inc. or its affiliates. All Rights Reserved.
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

#ifndef BYTEPS_COMPRESSOR_COMPRESSOR_H
#define BYTEPS_COMPRESSOR_COMPRESSOR_H

#include <memory>

#include "../cpu_reducer.h"
#include "common.h"

namespace byteps {
namespace common {
namespace compressor {
/*!
 *  \brief Compressor interface used in BytePS core.
 */
class Compressor {
 public:
  Compressor(size_t size)
      : _size(size),
        _buf(new byte_t[size]),
        _cpu_reducer(new CpuReducer(nullptr)){};
  virtual ~Compressor() = default;

  /*!
   * \brief Compress function
   *
   * \param grad gradient tensor
   * \param compressed compressed tensor
   */
  virtual void Compress(tensor_t grad, tensor_t& compressed) = 0;

  /*!
   * \brief Decompress function
   *
   * \param compressed compressed tensor
   * \param decompressed decompressed tensor
   */
  virtual void Decompress(tensor_t compressed, tensor_t& decompressed) = 0;

  /*!
   * \brief help function for error feedback `UpdateError`
   *
   * \param corrected gradient corrected with error
   * \param error error
   * \param compressed compressed gradient
   */
  virtual void FastUpdateError(tensor_t error, tensor_t corrected,
                               tensor_t compressed) {
    BPS_LOG(FATAL) << "FastUpdateError is not implemented";
  };

  size_t size() const { return _size; }

 protected:
  /*!
   * \brief buffer
   */
  std::unique_ptr<byte_t[]> _buf;

  /*!
   * \brief original size
   */
  size_t _size;

  /*!
   * \brief CPU reducer
   */
  std::unique_ptr<CpuReducer> _cpu_reducer;
};

}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESSOR_COMPRESSOR_H