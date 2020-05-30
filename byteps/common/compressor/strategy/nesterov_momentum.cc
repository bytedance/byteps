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

#include "nesterov_momentum.h"

#include "../../logging.h"

namespace byteps {
namespace common {
namespace compressor {
namespace {
CompressorRegistry::Register reg(
    "nesterov_momentum",
    [](const kwargs_t& kwargs) -> std::unique_ptr<BaseCompressor> {
      // register cpr
      auto kwargs_clone = kwargs;
      kwargs_clone.erase("momentum_type");
      auto compressor_ptr = CompressorRegistry::Create(kwargs_clone);
      BPS_CHECK_NE(compressor_ptr, nullptr);
      // find \mu
      auto iter = kwargs.find("momentum_mu");
      BPS_CHECK_NE(iter, kwargs.end()) << "momentum \mu is not defined";
      float mu = std::stof(iter->second);
      BPS_LOG(DEBUG) << "with momentum";
      return std::unique_ptr<NesterovMomentumCompressor>(
          new NesterovMomentumCompressor(std::move(compressor_ptr), mu));
    });
}

NesterovMomentumCompressor::NesterovMomentumCompressor(
    std::unique_ptr<BaseCompressor> compressor_ptr, float mu)
    : Momentum(std::move(compressor_ptr), mu){};

NesterovMomentumCompressor::~NesterovMomentumCompressor() = default;

void NesterovMomentumCompressor::UpdateMom(tensor_t grad) {
  // m_t = \mu * m_{t-1} + g_t
  this->_cpu_reducer->sum(_mom.get(), grad.data, _mom.get(), grad.size,
                          static_cast<DataType>(grad.dtype), _mu);
}

void NesterovMomentumCompressor::UpdateGradient(tensor_t grad) {
  // p_t = \mu m_t + g_t
  this->_cpu_reducer->sum(grad.data, _mom.get(), grad.size,
                          static_cast<DataType>(grad.dtype), _mu);
}

}  // namespace compressor
}  // namespace common
}  // namespace byteps