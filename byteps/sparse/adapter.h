// Copyright 2020 ByteDance, Inc. All Rights Reserved.
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

#ifndef BYTEPS_SPARSE_ADAPTER_H
#define BYTEPS_SPARSE_ADAPTER_H

#include "../common/common.h"
#include "nccl.h"

namespace byteps {
namespace sparse {

using namespace byteps::common;

template <class T>
class GeneralTensor : public Tensor {
 public:
  GeneralTensor(T* tensor, ncclDataType_t datatype, size_t size);
  virtual const DataType dtype() const override;
  virtual const TensorShape shape() const override;
  virtual const void* data() const override;
  virtual int64_t size() const override;

 protected:
  T* tensor_;
  ncclDataType_t nccl_datatype_;
  size_t size_;
};

void ThrowIfError(Status status);

}  // namespace sparse
}  // namespace byteps

#endif  // BYTEPS_SPARSE_ADAPTER_H
