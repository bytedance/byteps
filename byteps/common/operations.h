// Copyright 2019 Bytedance Inc. All Rights Reserved.
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

#ifndef BYTEPS_OPERATIONS_H
#define BYTEPS_OPERATIONS_H

#include <functional>
#include "common.h"

namespace byteps {
namespace common {

// Check that byteps is initialized.
Status CheckInitialized();

extern "C" {

// C interface to initialize byteps.
void byteps_init();

// C interface to initialize byteps (without initializing ps-lite).
void byteps_lazy_init();

// C interface to shut down byteps.
void byteps_shutdown();

// C interface to restart byteps.
void byteps_resume(int num_workers, int num_servers);

// C interface to suspend byteps.
void byteps_suspend();

// C interface to get index of current byteps process.
// Returns -1 if byteps is not initialized.
int byteps_rank();

// C interface to get index of current byteps process in the node it is on.
// Returns -1 if byteps is not initialized.
int byteps_local_rank();

// C interface to return number of byteps processes.
// Returns -1 if byteps is not initialized.
int byteps_size();

// C interface to return number of byteps processes in the node it is on.
// Returns -1 if byteps is not initialized.
int byteps_local_size();
}

// Below are all for Framework plugins
Status EnqueueTensor(BPSContext &context, std::shared_ptr<Tensor> input,
                     std::shared_ptr<Tensor> output,
                     std::shared_ptr<ReadyEvent> ready_event, const int device,
                     const int priority, const int version,
                     StatusCallback callback,
                     std::shared_ptr<std::vector<QueueType>> queue_list);

void InitTensor(BPSContext &context, size_t size, int dtype, void *cpubuff);

// Only call these in Framework plugins for the best performance
bool IsTensorDeclared(const std::string &name);

void RegisterCompressor(const std::string &name,
                        std::unordered_map<std::string, std::string> &kwargs);

BPSContext &GetContextFromName(const std::string &name);

std::shared_ptr<std::vector<QueueType>> GetPushQueueList(int device);

std::shared_ptr<std::vector<QueueType>> GetPullQueueList(int device);

}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_OPERATIONS_H
