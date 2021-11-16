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

#ifndef BYTEPS_ERROR_H
#define BYTEPS_ERROR_H

#include <mutex>
#include <unordered_map>

#include "common.h"
#include "logging.h"
#include "utils.h"
#include "ps/ps.h"

namespace byteps {
namespace common {

class BytePSError {
 public:
  static Status Error2Status(ps::ErrorCode code, std::string reason);
  static void RecordCallback(uint64_t key, StatusCallback cb);
  static void RemoveCallback(uint64_t key);
  static void ErrHandle(void* data, ps::ErrorCode code, std::string reason);

 private:
  static std::unordered_map<uint64_t, StatusCallback> callbacks_;
  static std::mutex mu_;
};


}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_ERROR_H
