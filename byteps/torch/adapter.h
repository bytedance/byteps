// Copyright 2019 ByteDance, Inc. All Rights Reserved.
// Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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

#ifndef BYTEPS_TORCH_ADAPTER_H
#define BYTEPS_TORCH_ADAPTER_H

#include <torch/extension.h>
#include <torch/torch.h>

#include "../common/common.h"

namespace byteps {
namespace torch {

using namespace byteps::common;

class TorchTensor : public Tensor {
 public:
  TorchTensor(::torch::Tensor tensor);
  virtual const DataType dtype() const override;
  virtual const TensorShape shape() const override;
  virtual const void* data() const override;
  virtual int64_t size() const override;

 protected:
  ::torch::Tensor tensor_;
};

void ThrowIfError(Status status);

}  // namespace torch
}  // namespace byteps

#endif  // BYTEPS_TORCH_ADAPTER_H
