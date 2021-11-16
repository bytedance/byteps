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

const std::string& DataType_Name(DataType value) {
  switch (value) {
    case BYTEPS_FLOAT32:
      static const std::string float32("float32");
      return float32;
    case BYTEPS_FLOAT64:
      static const std::string float64("float64");
      return float64;
    case BYTEPS_FLOAT16:
      static const std::string float16("float16");
      return float16;
    case BYTEPS_UINT8:
      static const std::string uint8("uint8");
      return uint8;
    case BYTEPS_INT32:
      static const std::string int32("int32");
      return int32;
    case BYTEPS_INT8:
      static const std::string int8("int8");
      return int8;
    case BYTEPS_INT64:
      static const std::string int64("int64");
      return int64;
    // case BYTEPS_UINT16:
    //   static const std::string uint16("uint16");
    //   return uint16;
    // case BYTEPS_INT16:
    //   static const std::string int16("int16");
    //   return int16;
    // case BYTEPS_BOOL:
    //   static const std::string bool_("bool");
    //   return bool_;
    // case BYTEPS_BYTE:
    //   static const std::string byte_("byte");
    //   return byte_;
    default:
      static const std::string unknown("<unknown>");
      return unknown;
  }
}

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

Status Status::DataLoss(std::string message) {
  return Status(StatusType::DATA_LOSS, message);
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

#if BYTEPS_BUILDING_CUDA == 1
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
const char* P2PTensorTableEntry::tensor_data(int index) const {
  if (group_tensors.size() > 0) {
    return (const char*) group_tensors.at(index)->data();
  } else {
    return (const char*) tensor->data();
  }
}

char* P2PTensorTableEntry::output_data(int index) const {
  if (group_outputs.size() > 0) {
    return (char*) group_outputs.at(index)->data();
  } else {
    return (char*) output->data();
  }
}

DataType P2PTensorTableEntry::tensor_dtype() const {
  if (group_tensors.size() > 0) {
    return group_tensors.at(0)->dtype();
  } else {
    return tensor->dtype();
  }
}

DataType P2PTensorTableEntry::output_dtype() const {
  if (group_outputs.size() > 0) {
    return group_outputs.at(0)->dtype();
  } else {
    return output->dtype();
  }
}

}  // namespace common
}  // namespace byteps
