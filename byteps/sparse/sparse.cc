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
#include "../common/global.h"

namespace byteps {
namespace sparse {

/**
  embedBuffers: the addresses of all embedding buffers (could have variable length)
  denseBuffers: the addresses of all dense buffers (the length should be identical)
  bufferLengths: the length of the embedding buffers (could have variable length)
  size: the length of a dense buffer (in bytes), it is equivalent for all GPUs
 */
void bytepsSparseInit(std::vector<void*>& embedBuffers, 
                      std::vector<void*>& denseBuffers, 
                      std::vector<int>& bufferLengths, 
                      int size) {
  BytePSSparseComm::InitComm();
  BPS_CHECK_EQ(embedBuffers.size(), denseBuffers.size());
  BPS_CHECK_EQ(bufferLengths.size(), denseBuffers.size());
  
  // Init IPC stuff
  sharedMemoryInfo info;
  BPS_CHECK_EQ(sharedMemoryCreate(bpsShmName, sizeof(shmStruct), &info), 0);
  auto shm = (volatile shmStruct *)info.addr;
  memset((void *)shm, 0, sizeof(*shm));

  auto localSize = BytePSSparseComm::GetLocalSize();
  auto workerNum = BytePSSparseComm::GetNumWorker();
  auto workerID = BytePSSparseComm::GetWorkerID();

  for (int i = 0; i < localSize; i++) {
    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, i));

    // CUDA IPC is only supported on devices with unified addressing
    if (!prop.unifiedAddressing) {
      BPS_LOG(INFO) << "Device " << i << " does not support unified addressing, skipping...";
      continue;
    }
    // We require two processes accessing each device, so we need
    // to ensure exclusive or prohibited mode is not set
    if (prop.computeMode != cudaComputeModeDefault) {
      BPS_LOG(INFO) << "Device " << i << "is in an unsupported compute mode for this sample";
      continue;
    }

    shm->devices[shm->nprocesses++] = i;
    BPS_CHECK_GT(MAX_CUDA_DEVICES, shm->nprocesses);
  }
  BPS_CHECK(shm->nprocesses > 0) 
      << "No cuda device suppported";
  BPS_CHECK_EQ(shm->nprocesses, embedBuffers.size())
      << "Shared memory processes: " << shm->nprocesses 
      << ", send buffers: " << embedBuffers.size();

  // We need to manually we need to clear the containers because
  // bytepsSparseInit() might be (unexpectedly) invoked multiple times
  _embedBuffers.clear();
  _denseBuffers.clear();
  _offsets.clear();

  _bufferLengths.clear();
  _bufferLengths.resize(workerNum);
  for (int i = 0; i < workerNum; i++) {
    _bufferLengths[i].resize(localSize);
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
    _bufferLengths[workerID][i] = bufferLengths[i]; // local buffer length

    // int offset = 0;
    // for (size_t j = 0; j < i; j++) {
    //   offset += _bufferLengths[workerID][j] / (localSize * workerNum);
    // }
    // _offsets[workerID].push_back(offset);
  }
  _denseBufferLength = size;

  // Check buffer length
  int accuml = 0;
  for (int i = 0; i < localSize; i++) {
    accuml += _bufferLengths[workerID][i] / localSize;
  }
  BPS_CHECK_EQ(accuml, _denseBufferLength) 
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
  if (!_ps) {
    _ps = new ps::KVWorker<char>(0, 0);
    ps::StartAsync(0, "byteps_sparse\0");
    ps::Postoffice::Get()->Barrier(
        0, ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
  }

  // Global coordination of the bufferLengths 
  // which is equivalent to all-gather 
  std::vector<ps::SArray<char>> bufferLenSarrays;
  for (int i = 0; i < workerNum; i++) {
    ps::SArray<char> tmp((char*)_bufferLengths[i].data(), 
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

  // Complete the _offsets table
  _offsets.resize(workerNum);
  for (int i = 0; i < workerNum; i++) {
    int offset = 0;
    for (size_t j = 0; j < i; j++) {
      offset += _bufferLengths[i][j] / (localSize * workerNum);
    }
    _offsets[i].push_back(offset);
  }
} 

void bytepsSparseShutdown() {
}


void bytepsGather(int local_rank, cudaStream_t stream) {
  // Gather from local peer GPUs on the same worker
  auto localSize = BytePSSparseComm::GetLocalSize();
  auto workerID = BytePSSparseComm::GetWorkerID();
  auto workerNum = BytePSSparseComm::GetNumWorker();
  void* baseDstPtr = (void*) _denseBuffers[local_rank];
  for (int i = 0; i < localSize; i++) {
    void* baseSrcPtr = (void*) ((char*)_embedBuffers[i]);
    void* srcPtr = (void*) ((char*)baseSrcPtr + _bufferLengths[workerID][i] / localSize * local_rank);
    void* dstPtr = (void*) ((char*)baseDstPtr + _offsets[workerID][i]);
    CUDA_CALL(cudaMemcpyPeerAsync(dstPtr, local_rank, srcPtr, i, _bufferLengths[workerID][i] / localSize, stream));
  }
  CUDA_CALL(cudaStreamSynchronize(stream));

  // // Gather from other distributed workers
  // if (BytePSSparseComm::IsDistributed()) {
  //   std::vector<int> ts; // store the zpush timestamp
  //   for (int i = 0; i < workerNum; i++) {
  //     if (i == workerID) continue;
  //     void* baseSrcPtr = (void*) ((char*)_embedBuffers[i]);
  //     ps::Key key = i;
  //     int len = _bufferLengths[i][j] / (localSize * workerNum);
  //     auto &pskv = EncodeDefaultKey(key, len);
  //     for (int j = 0; j < localSize; j++) {
  //       ps::SArray<char> vals((char*)_cpuBuffers[i] + _offsets[i][j], len, false);
  //       ts.push_back(_ps->ZPull(pskv.keys, &vals, &pskv.lens));
  //     }
  //   }
  //   for (auto t : ts) _ps->Wait(t);

  //   for (int i = 0; i < workerNum; i++) {
  //     if (i == workerID) continue; 
  //     for (int j = 0; j < localSize; j++) {
  //       int len = _bufferLengths[i][j] / (localSize * workerNum);
  //       CUDA_CALL(cudaMemcpyAsync(
  //           (void*)((char*)_denseBuffers[local_rank] + _offsets[i][j]), 
  //           (void*)((char*)_cpuBuffers[i] + _offsets[i][j]), 
  //           len, cudaMemcpyHostToDevice, stream));
  //     }
  //   }
  // }
  CUDA_CALL(cudaStreamSynchronize(stream));
}


void bytepsScatter(int local_rank, cudaStream_t stream) {
  auto localSize = BytePSSparseComm::GetLocalSize();
  auto workerID = BytePSSparseComm::GetWorkerID();
  void* baseSrcPtr = (void*)_denseBuffers[local_rank];
  for (int i = 0; i < localSize; i++) {
    void* baseDstPtr = (void*) ((char*)_embedBuffers[i]);
    void* srcPtr = (void*) ((char*)baseSrcPtr + _offsets[workerID][i]);
    void* dstPtr = (void*) ((char*)baseDstPtr + _bufferLengths[workerID][i] / localSize * local_rank);
    CUDA_CALL(cudaMemcpyPeerAsync(dstPtr, i, srcPtr, local_rank, _bufferLengths[workerID][i] / localSize, stream));
  }
  CUDA_CALL(cudaStreamSynchronize(stream));

  // // Scatter to other distributed workers
  // if (BytePSSparseComm::IsDistributed()) {
  //   // Copy to host memory
  //   auto valLen = len / workerNum;
  //   for (int i = 0; i < workerNum; i++) {
  //     if (i == workerID) continue; 
  //     CUDA_CALL(cudaMemcpyAsync(
  //         (void*)((char*)_cpuBuffers + valLen * i), (void*)((char*)_denseBuffers[rank] + valLen * i),
  //         valLen, cudaMemcpyDeviceToHost, stream));
  //   }
  //   CUDA_CALL(cudaStreamSynchronize(stream));

  //   std::vector<int> ts;
  //   for (int i = 0; i < workerNum; i++) {
  //     if (i == workerID) continue;
  //     auto key = i;
  //     auto &pskv = BytePSGlobal::EncodeDefaultKey(key, valLen);
  //     // need to use cpu buffer
  //     ps::SArray<char> vals((char*)_cpuBuffers + valLen * i, valLen, false);
  //     ts.push_back(_ps->ZPush(pskv.keys, vals, pskv.lens));
  //   }
  //   for (auto t : ts) _ps->Wait(t);
  // } else {
  //   CUDA_CALL(cudaStreamSynchronize(stream));
  // }

}

} // namespace sparse
} // namespace byteps 