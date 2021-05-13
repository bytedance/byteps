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

uint64_t byteps_session_id(const char* name);

uint32_t byteps_session_size();

void byteps_mark_done(const char* name);
}

extern "C" PyObject* byteps_get_pushpull_speed();

// Below are all for Framework plugins
Status EnqueueTensor(BPSContext &context, std::shared_ptr<Tensor> input,
                     std::shared_ptr<Tensor> output,
                     std::shared_ptr<ReadyEvent> ready_event, const int device,
                     const int priority, const int version,
                     StatusCallback callback,
                     std::shared_ptr<std::vector<QueueType>> queue_list);

// size_output: an auxiliary output tensor. In all-to-all, when recv_split is not provided,
// we also store the value of recv_split in `size_output`. When recv_split is provided,
// size_output is not used.
// send_begin: the begin index (number of elements) for each rank for `input` tensor. It's length
// is (num_ranks + 1) and always starts with 0
// recv_begin: the begin index (number of elements) for each rank for `output` tensor. It's length
// is (num_ranks + 1) and always starts with 0
Status EnqueueAlltoAllTensor(std::string& name,
                             std::shared_ptr<Tensor> input,
                             std::shared_ptr<Tensor> output,
                             std::shared_ptr<Tensor> size_output,
                             std::shared_ptr<ReadyEvent> ready_event,
                             const int device,
                             const int priority, const int version,
                             StatusCallback callback,
                             const std::vector<int>& send_begin, // begin offsets for send
                             const std::vector<int>& recv_begin, // begin offsets for recv
                             std::atomic_int* counter_ptr,
                             bool output_size_unknown);

void InitTensor(BPSContext &context, size_t size, int dtype, void *cpubuff);

void InitTensorP2P(BPSContext &context, size_t size, int dtype, void *cpubuff,
                   int sender, int receiver);

// Only call these in Framework plugins for the best performance
bool IsTensorDeclared(const std::string &name);
bool IsTensorDeclaredP2P(const std::string &name, int sender, int receiver);

void RegisterCompressor(const std::string &name,
                        std::unordered_map<std::string, std::string> &kwargs);

BPSContext &GetContextFromName(const std::string &name);

std::shared_ptr<std::vector<QueueType>> GetSendOneShotQueueList();

std::shared_ptr<std::vector<QueueType>> GetSendQueueList();

std::shared_ptr<std::vector<QueueType>> GetRecvQueueList();
 
std::shared_ptr<std::vector<QueueType>> GetPushQueueList(int device);

std::shared_ptr<std::vector<QueueType>> GetPullQueueList(int device);

void print_queue_list(std::shared_ptr<std::vector<QueueType>> queue_list,
                      std::string &name, bool is_dist_reduce_root_node);

}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_OPERATIONS_H
