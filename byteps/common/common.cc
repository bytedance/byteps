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
#include <fstream>

#include "common.h"
#include "logging.h"

namespace byteps {
namespace common {

void BytePSContext::set_profile_flag() {
  // Set the profile flag
  auto is_trace = getenv("BYTEPS_TRACE_ON");
  auto start_step = getenv("BYTEPS_TRACE_START_STEP");
  auto end_step = getenv("BYTEPS_TRACE_END_STEP");
  if (is_trace && atoi(is_trace) == 1) {
    // Enable trace, check the start and end step
    BPS_CHECK(start_step != NULL && end_step != NULL)
                << "BYTEPS_TRACE_START_STEP and BYTEPS_TRACE_END_STEP must be given "
                << "if BYTEPS_TRACE_ON is set.";
    BPS_CHECK(atoi(start_step) >= 1 && atoi(end_step) > atoi(start_step)) 
                << "BYTEPS_TRACE_START_STEP must be larger than 1, "
                << "BYTEPS_TRACE_END_STEP must be larger than BYTEPS_TRACE_START_STEP.";
    if(step_cnt == atoi(start_step)-1){
      profile_flag = true;
    } else if(step_cnt == atoi(end_step)){
      profile_flag = false;
      this->output_traces();
    } 
  } else {
    profile_flag = false;
  }
}
void BytePSContext::emit_trace(std::ostream *os, const BPSCommTime *ret){
    std::string tid = (ret->key == -1) ? "total" : std::to_string(ret->key);
    std::string para_name = "Comm." + tensor_name;
    std::string para_name_type = (ret->key == -1) ? para_name : para_name + "." + LogStrings[ret->type];
    (*os) << "        {\n"
          << "            \"ph\": \"X\",\n"
          << "            \"args\": {\n"
          << "                \"name\": \"" << para_name << "\"\n"
          << "            },\n"
          << "            \"pid\": \"" << para_name << "\",\n"
          << "            \"name\": \"" << para_name_type << "\",\n"
          << "            \"ts\": " << ret->start_t << ",\n"
          << "            \"dur\": " << ret->dur << ",\n"
          << "            \"tid\": \"" << tid << "\",\n"
          << "        }";
}

void BytePSContext::output_traces(){
  auto trace_dir = std::string(getenv("BYTEPS_TRACE_DIR"));
  auto trace_path = trace_dir + "/" + std::to_string(local_rank) 
                  + "/Comm/" + tensor_name + ".json";
  // Output these traces
  std::ofstream file;
  file.open(trace_path);
  file << "{" << std::endl;
  file << "    \"traceEvents\": [" << std::endl;
  auto first = true;
  while (comm_time.size() > 0) {
    BPSCommTime *ret = comm_time.front();
    if (!first) file << ",\n";
    else first = false;
    this->emit_trace(&file, ret);
    comm_time.pop();
  }
  while (!part_comm_time.empty()){
    auto part_id = part_comm_time.begin()->first;
    auto& type2part_comm_time = part_comm_time.begin()->second;
    BPS_CHECK(!type2part_comm_time.empty()) << "type2part_comm_time should not be empty";
    while (!type2part_comm_time.empty()){
      auto type = type2part_comm_time.begin()->first;
      auto& _part_comm_time_queue = type2part_comm_time.begin()->second;
      BPS_CHECK(_part_comm_time_queue.size() > 0) << "_part_comm_time_queue should not be empty";
      while (_part_comm_time_queue.size() > 0){
        BPSCommTime *ret = _part_comm_time_queue.front();
        _part_comm_time_queue.pop();
        if (!first) file << ",\n";
        this->emit_trace(&file, ret); // todo
      }
      type2part_comm_time.erase(type);
    }
    // if the unordered_map becomes empty, all the traces of this part_id has been read, delete this part_id
    part_comm_time.erase(part_id);
  }
  file << "\n" << std::endl;
  file << "    ]," << std::endl;
  file << "    \"displayTimeUnit\": \"ms\"" << std::endl;
  file << "}" << std::endl;
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
