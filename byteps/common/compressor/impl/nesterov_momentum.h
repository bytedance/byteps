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

#ifndef BYTEPS_COMPRESSOR_IMPL_NESTEROV_MOMENTUM_H
#define BYTEPS_COMPRESSOR_IMPL_NESTEROV_MOMENTUM_H

#include "../momentum.h"

namespace byteps {
namespace common {
namespace compressor {

/*!
 * \brief Nesterov Momentum Compressor
 *
 * paper: A method for solving the convex programming problem with convergence
 * rate $O (1/k^2)$
 *
 * m_t <- \mu m_{t-1} + g_t
 * g_t <- \mu m_t + g_t
 *
 */
class NesterovMomentumCompressor : public Momentum {
 public:
  NesterovMomentumCompressor(size_t size, DataType dtype,
                             std::unique_ptr<Compressor> cptr, float mu)
      : Momentum(size, dtype, std::move(cptr), mu){};
  virtual ~NesterovMomentumCompressor() = default;

 protected:
  void UpdateMom(tensor_t grad) override;
  void UpdateGradient(tensor_t grad) override;
};

}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESSOR_IMPL_NESTEROV_MOMENTUM_H