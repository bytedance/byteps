// Copyright 2020 Bytedance Inc. or its affiliates. All Rights Reserved.
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

#include "sparse.h"

namespace byteps {
namespace sparse {

/**
  embedBuffers: the addresses of all embedding buffers (could have variable length)
  denseBuffers: the addresses of all dense buffers (the length should be identical)
  embedBufferLens: the length of the embedding buffers (could have variable length)
  size: the length of a dense buffer (in bytes), it is equivalent for all GPUs
 */
void bytepsSparseInit(std::vector<void*>& embedBuffers, 
                      std::vector<void*>& denseBuffers, 
                      std::vector<int>& embedBufferLens, 
                      int size) {
  BytePSSparseCommon::Init();
  CHECK_EQ(embedBuffers.size(), denseBuffers.size());
  CHECK_EQ(embedBufferLens.size(), denseBuffers.size());
  
  // Init IPC stuff
  sharedMemoryInfo info;
  CHECK_EQ(sharedMemoryCreate(bpsShmName, sizeof(shmStruct), &info), 0);
  auto shm = (volatile shmStruct *)info.addr;
  memset((void *)shm, 0, sizeof(*shm));

  auto localSize = BytePSSparseCommon::GetLocalSize();
  auto workerNum = BytePSSparseCommon::GetNumWorker();
  auto workerID = BytePSSparseCommon::GetWorkerID();

  for (int i = 0; i < localSize; i++) {
    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, i));

    // CUDA IPC is only supported on devices with unified addressing
    if (!prop.unifiedAddressing) {
      // BPS_LOG(INFO) << "Device " << i << " does not support unified addressing, skipping...";
      continue;
    }
    // We require two processes accessing each device, so we need
    // to ensure exclusive or prohibited mode is not set
    if (prop.computeMode != cudaComputeModeDefault) {
      // BPS_LOG(INFO) << "Device " << i << "is in an unsupported compute mode for this sample";
      continue;
    }

    shm->devices[shm->nprocesses++] = i;
    CHECK_GT(MAX_CUDA_DEVICES, shm->nprocesses);
  }
  CHECK(shm->nprocesses > 0) 
      << "No cuda device suppported";
  CHECK_EQ(shm->nprocesses, embedBuffers.size())
      << "Shared memory processes: " << shm->nprocesses 
      << ", send buffers: " << embedBuffers.size();

  // We need to manually we need to clear the containers because
  // bytepsSparseInit() might be (unexpectedly) invoked multiple times
  _embedBuffers.clear();
  _denseBuffers.clear();

  _embedBufferLens.clear();
  _embedBufferLens.resize(workerNum);
  for (int i = 0; i < workerNum; i++) {
    _embedBufferLens[i].resize(localSize);
  }

  // Allocate memory and an event for each process and fill 
  // the shared memory buffer with the IPC handles 
  for (size_t i = 0; i < shm->nprocesses; i++) {
    cudaEvent_t event;
    CUDA_CALL(cudaSetDevice(
        shm->devices[i]));
    CUDA_CALL(cudaIpcGetMemHandle(
        (cudaIpcMemHandle_t *)&shm->embedMemHandle[i], embedBuffers[i]));
    CUDA_CALL(cudaIpcGetMemHandle(
        (cudaIpcMemHandle_t *)&shm->denseMemHandle[i], denseBuffers[i]));
    CUDA_CALL(cudaEventCreate(
        &event, cudaEventDisableTiming | cudaEventInterprocess));
    CUDA_CALL(cudaIpcGetEventHandle(
        (cudaIpcEventHandle_t *)&shm->eventHandle[i], event));
    
    // Store the buffers 
    _embedBuffers.push_back(embedBuffers[i]); 
    _denseBuffers.push_back(denseBuffers[i]);
    _embedBufferLens[workerID][i] = embedBufferLens[i]; // local buffer length
  }
  _denseBufferLength = size;

  // Check buffer length
  int accuml = 0;
  for (int i = 0; i < localSize; i++) {
    accuml += _embedBufferLens[workerID][i] / localSize;
  }
  CHECK_EQ(accuml, _denseBufferLength) 
      << accuml << " " << _denseBufferLength;

  // Need a continous CPU buffer for each GPU
  _cpuBuffers.clear();
  for (int i = 0; i < localSize; i++) {
    void* _cpuBuffer;
    CUDA_CALL(cudaHostAlloc(
        &_cpuBuffer, size, cudaHostAllocMapped | cudaHostAllocPortable));
    _cpuBuffers.push_back(_cpuBuffer);
  }
  
  // Start the pslite instance
  if (!_ps && BytePSSparseCommon::IsDistributed()) {
    _ps = new ps::KVWorker<char>(0, 0);
    ps::StartAsync(0, "byteps_sparse\0");
    ps::Postoffice::Get()->Barrier(
        0, ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);

    // Global coordination of the _embedBufferLens 
    // which is equivalent to all-gather 
    std::vector<ps::SArray<char>> bufferLenSarrays;
    for (int i = 0; i < workerNum; i++) {
      ps::SArray<char> tmp((char*)_embedBufferLens[i].data(), 
                        workerNum * sizeof(int), 
                        false);
      bufferLenSarrays.push_back(tmp);
    }

    std::vector<ps::SArray<ps::Key>> tmpKeys;
    std::vector<ps::SArray<int>> tmpLens;
    auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
    for (int key = 0; key < workerNum; key++) {
      int server = key;

      std::vector<ps::Key> tmp1(1, krs[server].begin() + key);
      ps::SArray<ps::Key> keys(tmp1);
      tmpKeys.push_back(keys);

      std::vector<int> tmp2(1, workerNum * sizeof(int));
      ps::SArray<int> lens(tmp2);
      tmpLens.push_back(lens);
    }

    // Push once to the associated server
    {
      int server = workerID;
      auto keys = tmpKeys[server];
      auto vals = bufferLenSarrays[server];
      auto lens = tmpLens[server];
      _ps->Wait(_ps->ZPush(keys, vals, lens));
    }

    // Call a barrier to sync across multiple workers. 
    // In case that some workers finish push too fast, 
    // and then pull from other workers too early
    ps::Postoffice::Get()->Barrier(
        0, ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);

    // Gather from other workers 
    for (int key = 0; key < workerNum; key++) {
      int server = key;
      if (server == workerID) continue; // skip myself
      auto keys = tmpKeys[server];
      auto vals = bufferLenSarrays[server];
      auto lens = tmpLens[server];
      _ps->Wait(_ps->ZPull(keys, &vals, &lens));
    }
  }

  // Prepare gossip communication
  _local_gather_comms.resize(localSize);
  for (int i = 0; i < localSize; i++) {
    if (i != 0) continue;
    std::vector<float*> srcs(localSize);
    std::vector<size_t> srcs_lens(localSize);
    std::vector<size_t> send_counts(localSize);
    for (int j = 0; j < localSize; j++) {
      srcs[j] = (float*)_embedBuffers[j] + (i * _embedBufferLens[workerID][j] / localSize);
      srcs_lens[j] = (localSize - i) * _embedBufferLens[workerID][j] / localSize;
      send_counts[j] = _embedBufferLens[workerID][j] / localSize;
    }
    float* dst = (float *)_denseBuffers[i];
    size_t dst_len = _denseBufferLength;

    std::string planfile_name("gather_plan_");
    planfile_name += std::to_string(i) + std::string(".json");
    _local_gather_comms[i] = std::make_unique<LocalGatherComm>(
        planfile_name, localSize, srcs, srcs_lens, send_counts, dst, dst_len);
  }

} 

void bytepsSparseShutdown() {
}


void bytepsGather(int local_rank, cudaStream_t stream) {
  // Gather from local peer GPUs on the same worker
  auto localSize = BytePSSparseCommon::GetLocalSize();
  auto workerID = BytePSSparseCommon::GetWorkerID();
  auto workerNum = BytePSSparseCommon::GetNumWorker();

  if (local_rank > 0) return; // debug
  _local_gather_comms[local_rank]->ExecAsync();
  _local_gather_comms[local_rank]->Sync();

  CUDA_CALL(cudaStreamSynchronize(stream));
}


void bytepsScatter(int local_rank, cudaStream_t stream) {
  auto localSize = BytePSSparseCommon::GetLocalSize();
  auto workerID = BytePSSparseCommon::GetWorkerID();
  void* baseSrcPtr = (void*)_denseBuffers[local_rank];
  CUDA_CALL(cudaStreamSynchronize(stream));
}

} // namespace sparse
} // namespace byteps 