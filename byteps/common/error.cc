// Copyright 2021 Bytedance Inc. or its affiliates. All Rights Reserved.
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

#include "error.h"

namespace byteps {
namespace common {

std::unordered_map<uint64_t, StatusCallback> BytePSError::callbacks_;
std::mutex BytePSError::mu_;

Status BytePSError::Error2Status(ps::ErrorCode code, std::string reason) {
  switch (code) {
    case ps::PS_OK:
      return Status::OK();
    case ps::PS_ERR_TIMED_OUT:
    case ps::PS_ERR_NOT_CONNECTED:
      return Status::DataLoss(reason);
    case ps::PS_ERR_CONNECTION_RESET:
      return Status::Aborted(reason);
    case ps::PS_ERR_OTHER:
      return Status::UnknownError(reason);
    default:
      return Status::UnknownError(reason);
  }
  return Status::UnknownError(reason);
}

void BytePSError::RecordCallback(uint64_t key, StatusCallback cb) {
  std::lock_guard<std::mutex> lk(mu_);
  callbacks_[key] = cb;
}

void BytePSError::RemoveCallback(uint64_t key) {
  std::lock_guard<std::mutex> lk(mu_);
  callbacks_.erase(key);
}

void BytePSError::ErrHandle(void* data, ps::ErrorCode code, std::string reason) {
  BPS_LOG(INFO) << "BytePS error handler invoked. status: " << code << ". reason: " << reason;
  std::lock_guard<std::mutex> lk(mu_);
  BPS_LOG(INFO) << "Processing " << callbacks_.size() << " pending callbacks";
  if (callbacks_.size() == 0) {
    return;
  }
  for (auto key_cb : callbacks_) {
    auto cb = key_cb.second;
    cb(Error2Status(code, reason));
  }
  callbacks_.clear();
}

}  // namespace common
}  // namespace byteps
