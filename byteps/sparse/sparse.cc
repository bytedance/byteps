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

void InitBytepsSparse(std::vector<void*>& embedBuffers, std::vector<void*>& denseBuffers, int size) {
  BytePSGlobal::Init();
  BPS_CHECK_EQ(embedBuffers.size(), denseBuffers.size());
  // Init IPC stuff
  auto shm_obj = BytePSGlobal::GetSharedMemoryObj();
  auto shm = (volatile shmStruct*) 
              shm_obj->openSharedMemory(
              std::string("BytePS_Sparse_ShM_"), 
              0, // key
              sizeof(shmStruct)
              );  
  memset((void *)shm, 0, sizeof(*shm));

  auto localSize = BytePSGlobal::GetLocalSize();
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
    
    // store the buffers 
    _embedBuffers.push_back(embedBuffers[i]); 
    _denseBuffers.push_back(denseBuffers[i]);
  }

  // Need a continous CPU buffer for all GPUs
  CUDA_CALL(cudaHostAlloc(
      &_cpuBuffer, size, cudaHostAllocMapped | cudaHostAllocPortable));

  // Launch ps-lite if needs distributed training
  if (BytePSGlobal::IsDistributed()) {
    // Init worker
    _ps = new ps::KVWorker<char>(0, 0);
    ps::StartAsync(0, "byteps\0");
    ps::Postoffice::Get()->Barrier(0, 
        ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
  }
} 

void ShutdownBytepsSparse() {
  BytePSGlobal::Shutdown();
}


void BytepsGather(int rank, int len, ncclDataType_t datatype, cudaStream_t stream) {
  // Gather from local peer GPUs on the same worker
  auto localSize = BytePSGlobal::GetLocalSize();
  auto workerID = BytePSGlobal::GetWorkerID();
  void* baseDstPtr = (void*) ((char*)_denseBuffers[rank] + len * workerID);
  for (int i = 0; i < localSize; i++) {
    if (rank == i) continue; // skip memcpy from myself to myself
    void* baseSrcPtr = (void*)((char*)_embedBuffers[i] + len * workerID);
    void* srcPtr = (void*)((char*)baseSrcPtr + len * i);
    void* dstPtr = (void*)((char*)baseDstPtr + len * i);
    CUDA_CALL(cudaMemcpyPeerAsync(dstPtr, rank, srcPtr, i, len, stream));
  }

  // Gather from other distributed workers
  if (BytePSGlobal::IsDistributed()) {
    std::vector<int> ts;
    auto valLen = len * localSize;
    auto workerNum = BytePSGlobal::GetNumWorker();
    for (int i = 0; i < workerNum; i++) {
      if (i == workerID) continue;
      auto key = i;
      auto &pskv = BytePSGlobal::EncodeDefaultKey(key, valLen);
      // need to use cpu buffer
      ps::SArray<char> vals((char*)_cpuBuffer + valLen * i, valLen, false);
      ts.push_back(_ps->ZPull(pskv.keys, &vals, &pskv.lens));
    }
    for (auto t : ts) _ps->Wait(t);
    for (int i = 0; i < workerNum; i++) {
      if (i == workerID) continue; 
      CUDA_CALL(cudaMemcpyAsync(
          (void*)((char*)_denseBuffers[rank] + valLen * i), (void*)((char*)_cpuBuffer + valLen * i), 
          valLen, cudaMemcpyHostToDevice, stream));
    }
  }
  CUDA_CALL(cudaStreamSynchronize(stream));
}


void BytepsScatter(int rank, int len, ncclDataType_t datatype, cudaStream_t stream) {
  auto workerID = BytePSGlobal::GetWorkerID();

  void* baseSrcPtr = _denseBuffers[rank];
  
  // Scatter to local peer GPUs on the same worker
  auto localSize = BytePSGlobal::GetLocalSize();
  auto workerNum = BytePSGlobal::GetNumWorker();
  auto globalSize = workerNum * localSize;
  // Assume the len is partitionable 
  auto unitLen = len / globalSize;
  for (int i = 0; i < localSize; i++) {
    if (rank == i) continue; // skip memcpy from myself to myself
    void* baseDstPtr = _embedBuffers[i];
    void* srcPtr = (void*)((char*)baseSrcPtr + unitLen * i);
    void* dstPtr = (void*)((char*)baseDstPtr + unitLen * i);
    CUDA_CALL(cudaMemcpyPeerAsync(dstPtr, rank, srcPtr, i, unitLen, stream));
  }

  // Scatter to other distributed workers
  if (BytePSGlobal::IsDistributed()) {
    // Copy to host memory
    auto valLen = len / workerNum;
    for (int i = 0; i < workerNum; i++) {
      if (i == workerID) continue; 
      CUDA_CALL(cudaMemcpyAsync(
          (void*)((char*)_cpuBuffer + valLen * i), (void*)((char*)_denseBuffers[rank] + valLen * i),
          valLen, cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CALL(cudaStreamSynchronize(stream));

    std::vector<int> ts;
    for (int i = 0; i < workerNum; i++) {
      if (i == workerID) continue;
      auto key = i;
      auto &pskv = BytePSGlobal::EncodeDefaultKey(key, valLen);
      // need to use cpu buffer
      ps::SArray<char> vals((char*)_cpuBuffer + valLen * i, valLen, false);
      ts.push_back(_ps->ZPush(pskv.keys, vals, pskv.lens));
    }
    for (auto t : ts) _ps->Wait(t);
  } else {
    CUDA_CALL(cudaStreamSynchronize(stream));
  }
}

} // namespace sparse
} // namespace byteps 