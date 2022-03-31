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

#include "adapter.h"
#include "cuda_util.h"

namespace byteps {
namespace torch {

TorchTensor::TorchTensor(::torch::Tensor tensor) : tensor_(tensor) {}

const DataType TorchTensor::dtype() const {
  switch (tensor_.scalar_type()) {
    case ::torch::kByte:
      return DataType::BYTEPS_UINT8;
    case ::torch::kChar:
      return DataType::BYTEPS_INT8;
    // case ::torch::kShort:
    //   return DataType::BYTEPS_INT16;
    case ::torch::kInt:
      return DataType::BYTEPS_INT32;
    case ::torch::kLong:
      return DataType::BYTEPS_INT64;
    case ::torch::kHalf:
      return DataType::BYTEPS_FLOAT16;
    case ::torch::kFloat:
      return DataType::BYTEPS_FLOAT32;
    case ::torch::kDouble:
      return DataType::BYTEPS_FLOAT64;
    default:
      throw std::logic_error("Invalid or unsupported tensor type.");
  }
}

const TensorShape TorchTensor::shape() const {
  TensorShape shape;
  for (int idx = 0; idx < tensor_.dim(); ++idx) {
    shape.AddDim(tensor_.size(idx));
  }
  return shape;
}

const void* TorchTensor::data() const { return tensor_.data_ptr(); }

int64_t TorchTensor::size() const {
#if TORCH_VERSION >= 1001000000
  return tensor_.element_size() * tensor_.numel();
#else
  return tensor_.type().elementSizeInBytes() * tensor_.numel();
#endif
}

void ThrowIfError(Status status) {
  switch (status.type()) {
    case StatusType::OK:
      return;
    case StatusType::PRECONDITION_ERROR:
      throw std::logic_error(status.reason());
    case StatusType::ABORTED:
      throw std::runtime_error(status.reason());
    case StatusType::INVALID_ARGUMENT:
      throw std::invalid_argument(status.reason());
    default:  // Includes UNKNOWN_ERROR
      throw std::runtime_error(status.reason());
  }
}

}  // namespace torch
}  // namespace byteps
