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
                      std::vector<size_t>& embedBufferLens, 
                      size_t denseBufferLen) {
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
  auto globalSize = localSize * workerNum;

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
  _embedBuffers.assign(embedBuffers.begin(), embedBuffers.end());
  _denseBuffers.assign(denseBuffers.begin(), denseBuffers.end());

  _localEmbedBufLens.clear();
  _localEmbedBufLens.resize(localSize);
  _globalTotalEmbedBufLens.resize(workerNum, 0);

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
    _localEmbedBufLens[i] = embedBufferLens[i]; // local buffer length
  }
  _denseBufferLen = denseBufferLen;

  for (int i = 0; i < localSize; i++) {
    _globalTotalEmbedBufLens[workerID] += _localEmbedBufLens[i];
  }

  // Need a continous CPU buffer for each GPU
  _cpuBuffers.clear();
  for (int i = 0; i < localSize; i++) {
    void* _cpuBuffer;
    CUDA_CALL(cudaHostAlloc(
        &_cpuBuffer, _denseBufferLen, cudaHostAllocMapped | cudaHostAllocPortable));
    _cpuBuffers.push_back(_cpuBuffer);
  }
  
  // The followings are for the global coordination of 
  // the embedding buffer length, which is equivalent to all-gather 
  auto ps = BytePSSparseCommon::GetPS();
  if (BytePSSparseCommon::IsDistributed()) {
    CHECK(ps); // must init the pslite instance before
    std::vector<ps::SArray<ps::Key>> tmpKeys;
    std::vector<ps::SArray<int>> tmpLens;
    std::vector<ps::SArray<char>> bufferLenSarrays;
    auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
    for (int i = 0; i < workerNum; i++) {
      ps::Key key = i;
      int server = i;

      // vals
      ps::SArray<char> tmp(
          (char*)&_globalTotalEmbedBufLens[i], sizeof(int), false);
      bufferLenSarrays.push_back(tmp);
      
      // keys
      std::vector<ps::Key> tmp1(1, krs[server].begin() + key);
      ps::SArray<ps::Key> keys(tmp1);
      tmpKeys.push_back(keys);
      
      // lens
      std::vector<int> tmp2(1, localSize * sizeof(int));
      ps::SArray<int> lens(tmp2);
      tmpLens.push_back(lens);
    }

    // Push once to the associated server
    {
      int server = workerID;
      auto keys = tmpKeys[server];
      auto vals = bufferLenSarrays[server];
      auto lens = tmpLens[server];
      ps->Wait(ps->ZPush(keys, vals, lens));
    }

    // Call a barrier to sync across multiple workers. 
    // In case that some workers finish push too fast, 
    // and then pull from other workers too early
    ps::Postoffice::Get()->Barrier(
        0, ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);

    // Pull the embedding buffer length of other workers
    for (int key = 0; key < workerNum; key++) {
      int server = key;
      if (server == workerID) continue; // skip myself
      auto keys = tmpKeys[server];
      auto vals = bufferLenSarrays[server];
      auto lens = tmpLens[server];
      ps->Wait(ps->ZPull(keys, &vals, &lens));
    }
  }

  // For debug: print _localEmbedBufLens
  std::cout << "_localEmbedBufLens:" << std::endl;
  for (auto len : _localEmbedBufLens) 
    std::cout << len << " ";
  std::cout << std::endl;

  // For debug: print _globalTotalEmbedBufLens
  std::cout << "_globalTotalEmbedBufLens:" << std::endl;
  for (auto len : _globalTotalEmbedBufLens) 
    std::cout << len << " ";
  std::cout << std::endl;

  // Check the buffer size 
  size_t accmul = 0;
  for (auto len : _globalTotalEmbedBufLens) accmul += len / globalSize;
  CHECK_EQ(accmul, _denseBufferLen) << accmul << " " << _denseBufferLen;

  // Calc the global offset for the communication buffers
  size_t global_offset = 0;
  for (int id = 0; id < workerID; id++) {
    global_offset += _globalTotalEmbedBufLens[id] / globalSize;
  }

  // Prepare gossip-gather communication
  _local_gather_comms.resize(localSize);
  for (int i = 0; i < localSize; i++) {
    std::vector<float*> srcs(localSize);
    std::vector<size_t> srcs_lens(localSize);
    std::vector<size_t> send_counts(localSize);

    for (int j = 0; j < localSize; j++) {
      srcs[j] = 
          (float*)_embedBuffers[j] + 
          _localEmbedBufLens[j] / globalSize * (i + localSize * workerID);

      srcs_lens[j] = 
          _localEmbedBufLens[j] / globalSize * 
          (globalSize - (i + localSize * workerID));
          
      send_counts[j] = 
          _localEmbedBufLens[j] / globalSize;
    }
    float* dst = (float *)_denseBuffers[i] + global_offset;
    size_t dst_len = _globalTotalEmbedBufLens[workerID] / globalSize;

    std::string planfile_name("gather_plan_");
    planfile_name += std::to_string(i) + std::string(".json");
    _local_gather_comms[i] = std::make_unique<LocalGatherComm>(
        planfile_name, localSize, srcs, srcs_lens, send_counts, dst, dst_len);
  }

  // Prepare gossip-scatter communication
  _local_scatter_comms.resize(localSize);
  for (int i = 0; i < localSize; i++) {
    float* src = (float *)_denseBuffers[i] + global_offset;
    size_t src_len = _globalTotalEmbedBufLens[workerID] / globalSize;
    std::vector<float*> dsts(localSize);
    std::vector<size_t> dsts_lens(localSize);
    std::vector<size_t> send_counts(localSize);
    for (int j = 0; j < localSize; j++) {
      dsts[j] = 
          (float*)_embedBuffers[j] + 
          _localEmbedBufLens[j] / globalSize * (i + localSize * workerID);

      dsts_lens[j] = 
          _localEmbedBufLens[j] / globalSize * 
          (globalSize - (i + localSize * workerID));

      send_counts[j] = 
          _localEmbedBufLens[j] / globalSize;
    }

    std::string planfile_name("scatter_plan_");
    planfile_name += std::to_string(i) + std::string(".json");
    _local_scatter_comms[i] = std::make_unique<LocalScatterComm>(
        planfile_name, localSize, src, src_len, send_counts, dsts, dsts_lens);
  }

} 

void bytepsSparseShutdown() {
}


void bytepsGatherExecAsync(int local_rank, cudaStream_t stream) {
  // Gather from local peer GPUs on the same worker
  auto localSize = BytePSSparseCommon::GetLocalSize();
  auto workerID = BytePSSparseCommon::GetWorkerID();
  auto workerNum = BytePSSparseCommon::GetNumWorker();

  _local_gather_comms[local_rank]->ExecAsync();
}

void bytepsSynchronize(int local_rank, cudaStream_t stream, OP op) { 
  switch (op) {
    case GATHER:
      _local_gather_comms[local_rank]->Sync();
      break;
    case SCATTER:
      _local_scatter_comms[local_rank]->Sync();
      break;
    default:
      CHECK(0) << "unrecognized operation";
  }
  CUDA_CALL(cudaStreamSynchronize(stream));
}

void bytepsScatterExecAsync(int local_rank, cudaStream_t stream) {
  auto localSize = BytePSSparseCommon::GetLocalSize();
  auto workerID = BytePSSparseCommon::GetWorkerID();
  void* baseSrcPtr = (void*)_denseBuffers[local_rank];

  _local_scatter_comms[local_rank]->ExecAsync();
}


} // namespace sparse
} // namespace byteps 