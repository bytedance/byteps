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

#ifndef BYTEPS_COMPRESSOR_MOMENTUM_H
#define BYTEPS_COMPRESSOR_MOMENTUM_H

#include "../cpu_reducer.h"
#include "compressor.h"

namespace byteps {
namespace common {
namespace compressor {
/*!
 * \brief Momentum
 *
 * Stochastic gradient descent with momentum
 *
 * \note
 * The momentum is added to gradient before compression. This should not be used
 * at the same time with the momentum implemented in the framework such as
 * MXNet, Tensorflow or PyTorch etc. The key difference between the two is the
 * position where they are added to the gradients. For this one, it is added
 * before push_pull. But for framework's momentum, it is added after push_pull.
 *
 * \note
 * The framework's momentum is disabled when using this momentum. User do not
 * need to disable it manully.
 *
 * \sa Compressor, NesterovMomentumCompressor
 */
class Momentum : public Compressor {
 public:
  // momentum should be cleared to zeros
  Momentum(size_t size, DataType dtype, std::unique_ptr<Compressor> cptr,
           float mu)
      : Compressor(size, dtype),
        _mom(new byte_t[size]()),
        _mu(mu),
        _cpu_reducer(new CpuReducer(nullptr)),
        _cptr(std::move(cptr)){};
  virtual ~Momentum() = default;

  virtual tensor_t Compress(tensor_t grad) final;

  virtual tensor_t Decompress(tensor_t compressed) final;

 protected:
  /*!
   * \brief Update momentum
   *
   * e.g. m_t = \mu * m_{t-1} + g_t
   *
   * \param grad refers to gradient
   */
  virtual void UpdateMom(tensor_t grad) = 0;

  /*!
   * \brief Update gradient with momentum
   *
   * e.g. g_t = \mu m_t + g_t
   *
   * \param grad refers to gradient which adds momentum in place.
   */
  virtual void UpdateGradient(tensor_t grad) = 0;

 protected:
  /*! \brief buffer of momentum */
  std::unique_ptr<byte_t[]> _mom;

  /*! \brief momentum factor */
  float _mu;

  std::unique_ptr<CpuReducer> _cpu_reducer;

 private:
  /*! \brief compressor pointer */
  std::unique_ptr<Compressor> _cptr;
};
}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESSOR_MOMENTUM_H