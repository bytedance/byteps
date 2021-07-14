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
#include "../common/common.h"
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

int GetAlltoallSender(uint64_t key) {
  return (key << 16) >> 48;
}

uint32_t GetAlltoallTensorId(uint64_t key) {
  return (key << 32) >> 48;
}

uint64_t ComposeAlltoallKey(int32_t declared_key, int request_rank) {
  // Total key space is [0, 2^64 - 1]
  // It will be divided to N PS servers, for now we assume N <= 2^16
  // Then we have 2^48 key space left.
  // Top 16 bits out of the 48 bits encodes the sender rank
  // Mid 16 bits out of the 48 bits encodes the tensor id
  // The next 6 bits encodes request types (pushpull, send, etc)
  // The last 10 bits encodes the partition id
  // Therefore, we support up to 2^16 tensors, and up to 2^10 partitions per tensor
  uint64_t request_key = ((uint64_t) request_rank) << 32;
  request_key += ((uint64_t) declared_key) << 16;
  request_key += ((uint64_t) common::P2P_OP) << 10;
  return request_key;
}

}  // namespace server
}  // namespace byteps
