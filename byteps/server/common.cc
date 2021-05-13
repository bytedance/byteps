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
#include "../common/logging.h"
#include <cmath>

namespace byteps {
namespace server {

int GetCommandType(RequestType requestType, int d) {
  int m = static_cast<int>(requestType);
  return (((m + d) * (m + d + 1)) / 2) + d;
}

DataHandleType DepairDataHandleType(int cmd) {
  int w = std::floor((std::sqrt(8 * cmd + 1) - 1)/2);
  int t = ((w * w) + w) / 2;
  int y = cmd - t;
  int x = w - y;
  BPS_CHECK_GE(x, 0);
  BPS_CHECK_GE(y, 0);
  DataHandleType type;
  type.requestType = static_cast<RequestType>(x);
  type.dtype = y;
  return type;
}

}  // namespace server
}  // namespace byteps
