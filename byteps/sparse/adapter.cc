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

#include "adapter.h"
#include "cuda_util.h"

namespace byteps {
namespace sparse {

template <class T>
GeneralTensor::GeneralTensor(T* tensor, ncclDataType_t datatype, size_t size) : tensor_(tensor), nccl_datatype_(datatype), size_(size) {}

const DataType GeneralTensor::dtype() const {
  switch (nccl_datatype_) {
    case ncclUint8:
      return DataType::BYTEPS_UINT8;
    case ncclChar:
      return DataType::BYTEPS_INT8;
    // case ::torch::kShort:
    //   return DataType::BYTEPS_INT16;
    case ncclInt32:
      return DataType::BYTEPS_INT32;
    case ncclInt64:
      return DataType::BYTEPS_INT64;
    case ncclFloat16:
      return DataType::BYTEPS_FLOAT16;
    case ncclFloat32:
      return DataType::BYTEPS_FLOAT32;
    case ncclFloat64:
      return DataType::BYTEPS_FLOAT64;
    default:
      throw std::logic_error("Invalid or unsupported tensor type.");
  }
}

const TensorShape GeneralTensor::shape() const {
  // TBA
  TensorShape shape;
  return shape;
}

const void* GeneralTensor::data() const { return tensor_; }

int64_t GeneralTensor::size() const {
  return size_;
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

}  // namespace sparse
}  // namespace byteps
