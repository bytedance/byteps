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

// TODO: C interface not needed.
uint64_t byteps_session_id(const char* name);

uint32_t byteps_session_size();

void byteps_mark_done(const char* name);

void byteps_get_telemetry_size(int32_t* size);

void byteps_get_telemetry_data(const char** names, float* mean, float* stdev,
                               int* count, int* actual_size, int max_size);

}

// Below are all for Framework plugins
Status EnqueueTensor(BPSContext &context, std::shared_ptr<Tensor> input,
                     std::shared_ptr<Tensor> output,
                     std::shared_ptr<ReadyEvent> ready_event, const int device,
                     const int priority, const int version,
                     StatusCallback callback,
                     std::shared_ptr<std::vector<QueueType>> queue_list,
                     ReduceOp op = REDUCE_OP_SUM);

// size_output: an auxiliary output tensor. In all-to-all, when recv_split is not provided,
//              we also store the value of recv_split in `size_output`. When recv_split is
//              provided, size_output is not used.
// send_begin: the begin index (number of elements) for each rank for `input` tensor. It's length
//             is (num_ranks + 1) and always starts with 0
// recv_begin: the begin index (number of elements) for each rank for `output` tensor. It's length
//             is (num_ranks + 1) and always starts with 0
Status EnqueueAlltoAllTensor(std::string& name,
                             std::shared_ptr<Tensor> input,
                             std::vector<std::shared_ptr<Tensor>>& group_inputs,
                             std::shared_ptr<Tensor> output,
                             std::vector<std::shared_ptr<Tensor>>& group_outputs,
                             std::shared_ptr<Tensor> size_output,
                             std::shared_ptr<ReadyEvent> ready_event,
                             const int input_device,
                             const int output_device,
                             const int priority, const int version,
                             StatusCallback callback,
                             const std::vector<int>& send_begin, // begin offsets for send
                             const std::vector<int>& recv_begin, // begin offsets for recv
                             bool output_size_unknown);

Status EnqueueAllgatherTensor(BPSContext &context, std::shared_ptr<Tensor> input,
                              std::shared_ptr<Tensor> output,
                              std::shared_ptr<ReadyEvent> ready_event, const int device,
                              const int priority, const int version,
                              const std::vector<int>& shape_list, StatusCallback callback);

// shape: input tensor shape
// tensor_key: the 32-bit tensor_key returned from declare_alltoall_tensor
// split_list: the split list for alltoall send
// recv_split_list: the recv split list for alltoall recv
// name: the provided name for the operation
// split_indices_list (output): the split indices based on strides
// recv_split_indices_list (output): the recv split indices based on strides
// dim0_in (output): the size of dimension 0 of the input
// dim0_out (output): the size of dimension 0 of the output
// session_name (output): the op name with session prefix
// initialized (output): whether the byteps context is already initialized
Status PrepareAlltoallTensor(TensorShape shape,
  const std::vector<int32_t>& tensor_key,
  const std::vector<int32_t>& split_list,
  const std::vector<int32_t>& recv_split_list, std::string& name,
  std::vector<int32_t>* split_indices_list,
  std::vector<int32_t>* recv_split_indices_list,
  int32_t* dim0_in, int32_t* dim0_out,
  std::string* session_name, bool* initialized);

void InitTensor(BPSContext &context, size_t size, int dtype, void *cpubuff);

// initializes BPSContext with key list and buffer list for the request and response
// tasks. It will fill context.key_list first with all request task keys, followed by
// response task keys. It will also populate context.cpubuff_list with buffer addresses
// for all request tasks, followed by all response tasks.
void InitTensorAlltoall(BPSContext &context, std::vector<int> &request_size_list,
                        std::vector<int> &resp_size_list, int dtype,
                        bool recv_on_gpu, bool output_size_unknown, bool use_pull);

void InitTensorP2P(BPSContext &context, size_t size, int dtype, void *cpubuff,
                   int sender, int receiver, bool recv_on_gpu = false);

void InitTensorAllgather(BPSContext &context, size_t input_size, size_t output_size, int dtype, void *cpubuff);

// Only call these in Framework plugins for the best performance
// declare the operation name with a provided key. -1 means no key is provided.
int32_t DeclareTensor(const std::string &name, int32_t provided_key);
// declare the operation name with a provided key and session id.
// provided_key = -1 means no key is provided
// session = -1 means no session is provided
int32_t DeclareAlltoallTensor(const std::string &name, int32_t provided_key, int32_t session);
int32_t DeclareP2PTensor(const std::string &name, int sender, int receiver);
int32_t DeclareAllgatherTensor(const std::string &name, int32_t provided_key);

void RegisterCompressor(const std::string &name,
                        std::unordered_map<std::string, std::string> &kwargs);

void PinMemory(void* ptr, int numa_or_gpu_index, size_t bytes, bool gpu);

BPSContext &GetContextFromName(const std::string &name);

std::shared_ptr<std::vector<QueueType>> GetSendOneShotQueueList();

std::shared_ptr<std::vector<QueueType>> GetSendQueueList();

std::shared_ptr<std::vector<QueueType>> GetRecvQueueList();
 
std::shared_ptr<std::vector<QueueType>> GetPushQueueList(int device);

std::shared_ptr<std::vector<QueueType>> GetPullQueueList(int device);

std::vector<QueueType> GetAlltoallRequestQueueList(bool use_pull);

std::vector<QueueType> GetAlltoallResponseQueueList(bool use_pull, bool output_size_unknown);

std::shared_ptr<std::vector<QueueType>> GetAllgatherRequestQueueList();

std::shared_ptr<std::vector<QueueType>> GetAllgatherResponseQueueList();

void print_queue_list(std::shared_ptr<std::vector<QueueType>> queue_list,
                      std::string &name, bool is_dist_reduce_root_node);

}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_OPERATIONS_H
