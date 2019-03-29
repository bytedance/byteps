// Copyright 2019 ByteDance Inc. or its affiliates. All Rights Reserved.
// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
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

#include <atomic>
#include <cassert>
#include <cstring>
#include <queue>
#include <sstream>
#include <thread>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>

#include "operations.h"

namespace byteps {
namespace common {


Status EnqueueTensorPush(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> input,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              StatusCallback callback) {
  return Status::OK();
}

Status EnqueueTensorPull(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> output,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              StatusCallback callback) {
  return Status::OK();
}


} // namespace common
} // namespace byteps
