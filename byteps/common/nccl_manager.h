// Copyright 2019 Bytedance Inc. or its affiliates. All Rights Reserved.
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

#ifndef BYTEPS_NCCL_MANAGER_H
#define BYTEPS_NCCL_MANAGER_H

#include <memory>
#include <queue>
#include <vector>
#include "common.h"
#include "communicator.h"
#include "scheduled_queue.h"

namespace byteps {
namespace common {

class NcclGroupEntry {
 public:
  void RecordEvents();
  void SynchronizeEvents();
  void DestroyEvents();

  std::vector<std::shared_ptr<TensorTableEntry>> tasks;
  std::vector<BytePSScheduledQueue*> queues;

 private:
  std::vector<cudaEvent_t> _events;
};

class NcclManager {
 public:
  NcclManager(std::shared_ptr<BytePSComm> comm);
  ~NcclManager() {
    if (_nccl_stream) {
      CUDA_CALL(cudaStreamDestroy(*_nccl_stream));
    }
    if (_nccl_id) {
      free(_nccl_id);
    }
    if (_nccl_comm) {
      free(_nccl_comm);
    }
    if (_signal_comm) {
      _signal_comm.reset();
    }
    if (_global_comm) {
      _global_comm.reset();
    }
    while (!_nccl_pipeline.empty()) _nccl_pipeline.pop();

    BPS_LOG(DEBUG) << "Clear NcclManager";
  }

  int GetGroupSize() { return _nccl_group_size; }
  void EnqueueGroup(std::shared_ptr<NcclGroupEntry> e);
  std::shared_ptr<NcclGroupEntry> DequeueGroup();

  virtual cudaStream_t GetStream(uint64_t key, QueueType op);
  virtual ncclComm_t GetComm(uint64_t key, QueueType op);
  virtual int GetRoot(uint64_t key, QueueType op);
  virtual int GetRank(uint64_t key, QueueType op);

  int GetSize() { return _nccl_size; }
  std::shared_ptr<BytePSComm> GetSignalComm() { return _signal_comm; }
  bool IsSignalRoot();

 protected:
  void InitGlobalEnv();
  virtual void ConstructRings();

  cudaStream_t* _nccl_stream;
  ncclUniqueId* _nccl_id;
  ncclComm_t* _nccl_comm;

  // global user-defined env
  size_t _nccl_group_size;
  size_t _nccl_pcie_size;
  size_t _nccl_pcie_num;
  size_t _nccl_num_rings;

  int _nccl_size;

  // for pipelining nccl
  std::mutex _nccl_mutex;
  std::queue<std::shared_ptr<NcclGroupEntry>> _nccl_pipeline;

  std::shared_ptr<BytePSComm> _signal_comm;
  std::shared_ptr<BytePSComm> _global_comm;
};

class NcclManagerExpr : public NcclManager {
 public:
  cudaStream_t GetStream(uint64_t key, QueueType op);
  ncclComm_t GetComm(uint64_t key, QueueType op);
  int GetRoot(uint64_t key, QueueType op);
  int GetRank(uint64_t key, QueueType op);

 protected:
  void ConstructRings();

  // for multi-ring
  std::vector<std::vector<int>> _rings;
};

}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_NCCL_MANAGER_H
