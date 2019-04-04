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

#include <cstring>
#include <memory>
#include <thread>
#include <chrono>

#include "global.h"
#include "operations.h"

namespace byteps {
namespace common {


bool RunPushLoopOnce() {
    auto q = BytePSGlobal::GetScheduledQueue(PUSH);
    if (q->pendingSize() > 0) {
        q->getTask()->callback(Status::OK());
    }
    return true;
}

bool RunPullLoopOnce() {
    auto q = BytePSGlobal::GetScheduledQueue(PULL);
    if (q->pendingSize() > 0) {
        q->getTask()->callback(Status::OK());
    }
    return true;
}

void PushLoop() {
    while (RunPushLoopOnce()) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }
}

void PullLoop() {
    while (RunPullLoopOnce()) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }
}

void byteps_init(const int *ranks, int nranks) {
    if (BytePSGlobal::StartInit()) {
        std::thread(PushLoop);
        std::thread(PullLoop);
        BytePSGlobal::FinishInit();
    }
    return;
}

void byteps_shutdown() {
    return;
}

int byteps_rank() {
    return 0;
}

int byteps_local_rank() {
    return 0;
}

int byteps_size() {
    return 0;
}

int byteps_local_size() {
    return 0;
}

Status EnqueueTensorPush(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> input,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              const int priority, const int version,
                              StatusCallback callback) {
    std::shared_ptr<TensorTableEntry> e(new TensorTableEntry);
    e->tensor_name = name;
    e->context = context;
    e->tensor = input;
    e->output = NULL;
    e->ready_event = ready_event;
    e->device = device;
    e->priority = priority;
    e->version = version;
    e->callback = callback;

    BytePSGlobal::GetScheduledQueue(PUSH)->addTask(e);
    return Status::OK();
}

Status EnqueueTensorPull(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> output,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              const int priority, const int version,
                              StatusCallback callback) {
    std::shared_ptr<TensorTableEntry> e(new TensorTableEntry);
    e->tensor_name = name;
    e->context = context;
    e->tensor = NULL;
    e->output = output;
    e->ready_event = ready_event;
    e->device = device;
    e->priority = priority;
    e->version = version;
    e->callback = callback;

    BytePSGlobal::GetScheduledQueue(PULL)->addTask(e);
    return Status::OK();
}


} // namespace common
} // namespace byteps
