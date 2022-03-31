// Copyright 2019 Bytedance Inc. or its affiliates. All Rights Reserved.
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

#include <cassert>
#include <sstream>

#include "common.h"
#include "logging.h"

namespace byteps {
namespace common {

Status::Status() = default;

Status::Status(StatusType type, std::string reason) {
  type_ = type;
  reason_ = reason;
}

Status Status::OK() { return Status(); }

Status Status::UnknownError(std::string message) {
  return Status(StatusType::UNKNOWN_ERROR, message);
}

Status Status::PreconditionError(std::string message) {
  return Status(StatusType::PRECONDITION_ERROR, message);
}

Status Status::Aborted(std::string message) {
  return Status(StatusType::ABORTED, message);
}

Status Status::InvalidArgument(std::string message) {
  return Status(StatusType::INVALID_ARGUMENT, message);
}

Status Status::InProgress() { return Status(StatusType::IN_PROGRESS, ""); }

bool Status::ok() const { return type_ == StatusType::OK; }

bool Status::in_progress() const { return type_ == StatusType::IN_PROGRESS; }

StatusType Status::type() const { return type_; }

const std::string& Status::reason() const { return reason_; }

void TensorShape::AddDim(int64_t dim) { shape_.push_back(dim); }

void TensorShape::AppendShape(TensorShape& other) {
  for (auto dim : other.shape_) {
    shape_.push_back(dim);
  }
}

const std::string TensorShape::DebugString() const {
  std::stringstream args;
  args << "[";
  for (auto it = shape_.begin(); it != shape_.end(); ++it) {
    if (it != shape_.begin()) {
      args << ", ";
    }
    args << *it;
  }
  args << "]";
  return args.str();
}

int TensorShape::dims() const { return (int)shape_.size(); }

int64_t TensorShape::dim_size(int idx) const {
  assert(idx >= 0);
  assert(idx < shape_.size());
  return shape_[idx];
}

int64_t TensorShape::num_elements() const {
  int64_t result = 1;
  for (auto dim : shape_) {
    result *= dim;
  }
  return result;
}

int GetCommandType(RequestType requestType, int d) {
  int m = static_cast<int>(requestType);
  return (((m + d) * (m + d + 1)) / 2) + d;
}

#ifndef BYTEPS_BUILDING_SERVER
ncclDataType_t getNcclDataType(DataType dtype) {
  switch (dtype) {
    case BYTEPS_FLOAT32:
      return ncclFloat32;
    case BYTEPS_FLOAT64:
      return ncclFloat64;
    case BYTEPS_FLOAT16:
      return ncclFloat16;
    case BYTEPS_UINT8:
      return ncclUint8;
    case BYTEPS_INT32:
      return ncclInt32;
    case BYTEPS_INT8:
      return ncclInt8;
    case BYTEPS_INT64:
      return ncclUint64;
    default:
      BPS_CHECK(0) << "Unsupported data type: " << dtype;
  }
  return ncclFloat32;
}
#endif

int getDataTypeLength(int dtype) {
  switch (dtype) {
    case BYTEPS_INT8:
    case BYTEPS_UINT8:
      return 1;
    case BYTEPS_FLOAT16:
      return 2;
    case BYTEPS_INT32:
    case BYTEPS_FLOAT32:
      return 4;
    case BYTEPS_INT64:
    case BYTEPS_FLOAT64:
      return 8;
    default:
      BPS_CHECK(0) << "Unsupported data type: " << dtype;
  }
  return 4;
}

}  // namespace common
}  // namespace byteps
