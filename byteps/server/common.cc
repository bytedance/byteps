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

#include "common.h"
#include <cmath>

namespace byteps {
namespace server {

int GetCommandType(RequestType requestType, int dtype, int device) {
  // command is 32 bit. from higher positions to lower ones:
  // 16 bits unused
  // 6 bits for request type
  // 6 bits for dtype
  // 4 bit for device
  int m = static_cast<int>(requestType);
  m = (m << 6) + dtype;
  m = (m << 4) + device;
  return m;
}

DataHandleType DepairDataHandleType(int cmd) {
  DataHandleType type;
  type.requestType = static_cast<RequestType>((cmd << 16) >> (32 - 6));
  type.dtype = (cmd << (16 + 6)) >> (32 - 6);
  type.device = (cmd << (16 + 6 + 6)) >> (32 - 4);
  return type;
}

uint32_t GetAlltoallTensorId(uint64_t key) {
  return (key << 32) >> 48;
}

}  // namespace server
}  // namespace byteps
