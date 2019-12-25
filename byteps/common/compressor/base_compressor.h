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

#ifndef BYTEPS_COMPRESS_BASE_COMPRESSOR_H
#define BYTEPS_COMPRESS_BASE_COMPRESSOR_H

#include <cstring>
#include <memory>
#include <unordered_map>

#include "common.h"

namespace byteps {
namespace common {
namespace compressor {

struct TensorType {
  char* data;
  size_t len;
  int dtype;
};

/**
 *  \brief Compressor interface used in BytePS core.
 */
class BaseCompressor {
 public:
  BaseCompressor();
  virtual ~BaseCompressor();

  /**
   * \brief allocate buffer for compression process.
   * \param len the size of buffer (bytes)
   */
  virtual void InitBuff(size_t len);

  /**
   * \brief Compress function, which must be overrided by children
   * \param grad the original gradient to be compressed
   * \return compressed gradient
   */
  virtual TensorType Compress(const TensorType& grad) = 0;

  /**
   * \brief Decompress funciton, which must be overrided by children
   * \param compressed_grad compressed gradient to be decompressed
   * \return decompressed gradient
   */
  virtual TensorType Decompress(const TensorType& compressed_grad) = 0;

 private:
  /**
   * \brief buffer for compression process
   */
  char* _compress_buff;
};

using CompressorPtr = std::unique_ptr<BaseCompressor>;

class CompressorRegistry {
 public:
  CompressorRegistry();
  ~CompressorRegistry();

  using CompressorFactory =
      std::function<CompressorPtr(const CompressorParam& param)>;
  using map_t = std::unordered_map<std::string, CompressorFactory>;

  struct Register {
    explicit Register(std::string name, CompressorFactory factory);
  };

  CompressorPtr create(std::string name, const CompressorParam& param) const;

  static CompressorRegistry& instance();

 private:
  map_t _compressors;
};
}
}
}

#endif  // BYTEPS_COMPRESS_BASE_COMPRESSOR_H