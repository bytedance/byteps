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

#ifndef BYTEPS_COMPRESS_MOMENTUM_H
#define BYTEPS_COMPRESS_MOMENTUM_H

#include "base_compressor.h"

namespace byteps {
namespace common {
namespace compressor {
/*!
 * \brief Momentum
 *
 * Stochastic gradient descent with momentum
 *
 * NOTE: This should not be used at the same time with the momentum implemented
 * in the framework such as MXNet, Tensorflow or PyTorch etc.
 */
class Momentum : public BaseCompressor {
 public:
  Momentum(std::unique_ptr<BaseCompressor> compressor_ptr, float mu);
  virtual ~Momentum();
  /*!
   * \brief Allocate encoding buffer for compression.
   * \param aligned_size aligned size
   */
  virtual void Init(size_t aligned_size) final;

  /*!
   * \brief Compress function
   *
   * \param grad gradient tensor
   * \param compressed compressed tensor
   */
  virtual void Compress(tensor_t grad, tensor_t& compressed) final;

  /*!
   * \brief Decompress function
   *
   * \param compressed compressed tensor
   * \param decompressed decompressed tensor
   */
  virtual void Decompress(tensor_t compressed, tensor_t& decompressed) final;

 protected:
  /*!
   * \brief Update momentum
   *
   * m_t = \mu * m_{t-1} + g_t
   *
   * \param grad refers to gradient
   */
  virtual void UpdateMom(tensor_t grad) = 0;

  /*!
   * \brief Update gradient
   *
   * p_t = \mu m_t + g_t
   *
   * \param grad refers to gradient which adds momentum in place.
   */
  virtual void UpdateGradient(tensor_t grad) = 0;

 protected:
  std::unique_ptr<char[]> _mom;

  float _mu;

 private:
  /*!
   * \brief compressor
   */
  std::unique_ptr<BaseCompressor> _compressor_ptr;
};
}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif