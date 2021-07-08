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
#ifndef BYTEPS_SERVER_COMMON_H
#define BYTEPS_SERVER_COMMON_H
#include "../common/logging.h"

namespace byteps {
namespace server {

enum class RequestType {
  kDefaultPushPull,
  kLeaderPushPull,
  kLeaderPushPullAvg,
  kRowSparsePushPull,
  kCompressedPushPull,
  kDefaultSend,
  kDefaultPull,
  kAckSignal,
  // wait for a group of send before copying them to output.
  // used when the total output size is unknown
  kGroupSend,
  // a special case for group send, where the send does not contain any
  // valid data (i.e. empty)
  kEmptyGroupSend
};

struct DataHandleType {
  RequestType requestType;
  int dtype;
  // the type of the device when handling data requests.
  // usually this only matters for push requests,
  // since server may need to register gpu buffers for GDR
  int device;
};

int GetCommandType(RequestType requestType, int dtype, int device);

DataHandleType DepairDataHandleType(int cmd);

uint32_t GetAlltoallTensorId(uint64_t key);

}  // namespace server
}  // namespace byteps

#endif  // BYTEPS_SERVER_COMMON_H
