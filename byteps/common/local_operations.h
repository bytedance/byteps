// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
// Copyright 2021 Bytedance Inc. All Rights Reserved.
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

#ifndef BYTEPS_LOCAL_OPERATIONS_H
#define BYTEPS_LOCAL_OPERATIONS_H

#include "common.h"

namespace byteps {
namespace common {

#if BYTEPS_BUILDING_CUDA == 1
void MemcpyInFusionBuffer(std::vector<std::shared_ptr<Tensor>>& src, char *dst, cudaStream_t stream);
void MemcpyOutFusionBuffer(char *src, std::vector<std::shared_ptr<Tensor>>& dst, cudaStream_t stream);
void ZeroOutTensors(std::vector<std::shared_ptr<Tensor>>& dst, cudaStream_t stream);
void CompensateGrads(std::vector<std::shared_ptr<Tensor>>& params, 
                     std::vector<std::shared_ptr<Tensor>>& grads,
                     std::vector<std::shared_ptr<Tensor>>& prev_params,
                     float lambda,
                     cudaStream_t stream);
void DCAdamLocalOp(std::vector<std::shared_ptr<Tensor>>& params,
                   std::vector<std::shared_ptr<Tensor>>& grads,
                   std::vector<std::shared_ptr<Tensor>>& prev_params,
                   float lambda,
                   std::vector<std::shared_ptr<Tensor>>& exp_avgs,
                   std::vector<std::shared_ptr<Tensor>>& exp_avg_sqs,
                   std::vector<int64_t>& steps,
                   float lr,
                   float eps,
                   float weight_decay,
                   float beta1,
                   float beta2,
                   cudaStream_t stream);
#endif

}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_LOCAL_OPERATIONS_H
