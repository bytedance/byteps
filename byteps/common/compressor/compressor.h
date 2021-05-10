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

#include "../common.h"
#include "../logging.h"
#include "common.h"

namespace byteps {
namespace common {
namespace compressor {
/*!
 * \brief Compressor interface
 * Compressor defines two universal API - Compress & Decompress
 *
 * \par
 * The caller do not need to allocate additional memory to store compressed data
 * because there is an internal buffer to store the compressed data and the
 * pointer will be returned to the caller. Then the caller can send the returned
 * compressed data as normal.
 *
 * \par
 * There are two optional features of the compressor - error-feedback &
 * momentum. These two features can be added to any common compressors like 1bit
 * and topk. To be generic, these two features are also compressors, exposing
 * the same API as Compressor. More details can be found in their own files.
 *
 * \par
 * To add a new compressor, developers need to inherit this class in 'impl'
 * directory. If a new optional feature like error-feedback is needed,
 * developers need to use decorator pattern and add new files in the current
 * directory. The existing implementation can be used as a reference.
 *
 *
 * \sa ErrorFeedback, Momentum
 */
class Compressor {
 public:
  Compressor(size_t size, DataType dtype) : _size(size), _dtype(dtype) {
    auto buf = new byte_t[size]();
    BPS_CHECK(buf) << "failed to allocate " << size << " bytes memory.";
    _buf.reset(buf);
  };
  virtual ~Compressor() = default;

  virtual void Compress(tensor_t grad, tensor_t& output) = 0;

  virtual void Decompress(tensor_t compressed, tensor_t& output) = 0;

  virtual void FusedCompress(tensor_t grad, tensor_t& output, tensor_t error) {
    BPS_CHECK(0) << "not implemented error.";
  };

 protected:
  size_t _size;
  DataType _dtype;
  std::unique_ptr<byte_t[]> _buf;
};

}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESSOR_COMPRESSOR_H