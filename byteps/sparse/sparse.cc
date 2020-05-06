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

void InitBytepsSparse(std::vector<void*>& cudaBuffer, int size) {
  BytePSGlobal::Init();

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
  for (i = 0; i < localSize; i++) {
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
  BPS_CHECK_EQ(shm->nprocesses, cudaBuffer.size())
      << "Shared memory processes: " << shm->nprocesses 
      << ", cuda buffers: " << cudaBuffer.size();

  // Allocate memory and an event for each process and fill 
  // the shared memory buffer with the IPC handles 
  for (i = 0; i < shm->nprocesses; i++) {
    cudaEvent_t event;
    CUDA_CALL(cudaSetDevice(
        shm->devices[i]));
    CUDA_CALL(cudaIpcGetMemHandle(
        (cudaIpcMemHandle_t *)&shm->memHandle[i], cudaBuffer[i]));
    CUDA_CALL(cudaEventCreate(
        &event, cudaEventDisableTiming | cudaEventInterprocess));
    CUDA_CALL(cudaIpcGetEventHandle(
        (cudaIpcEventHandle_t *)&shm->eventHandle[i], event));
    
    _cudaBuffers.push_back(cudaBuffer[i]); // store the buffers 
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
  auto input_tensor = std::make_shared<GeneralTensor>(tensor, datatype, len);
  auto dtype = input_tensor->dtype();
  
  // Gather from local peer GPUs on the same worker
  auto localSize = BytePSGlobal::GetLocalSize();
  auto workerID = BytePSGlobal::GetWorkerID();
  void* baseDstPtr = _cudaBuffers[rank] + len * workerID;
  for (int i = 0; i < localSize; i++) {
    if (rank == i) continue; // skip memcpy from myself to myself
    void* baseSrcPtr = _cudaBuffers[i] + len * workerID;
    void* srcPtr = baseSrcPtr + len * i;
    void* dstPtr = baseDstPtr + len * i;
    CUDA_CALL(cudaMemcpyPeerAsync(dstPtr, rank, src, i, len, stream));
  }

  // Gather from other distributed workers
  if (BytePSGlobal::IsDistributed()) {
    std::vector<int> ts;
    auto valLen = len * localSize;
    auto workerNum = BytePSGlobal::GetNumWorker();
    auto cmd = GetCommandType(RequestType::kDefaultPushPull, dtype);
    for (int i = 0; i < workerNum; i++) {
      if (i == workerID) continue;
      auto key = i;
      auto &pskv = BytePSGlobal::EncodeDefaultKey(key, valLen);
      // need to use cpu buffer
      ps::SArray<char> vals(_cpuBuffer + valLen * i, valLen, false);
      ts.push_back(_ps->ZPull(pskv.keys, vals, pskv.lens, cmd));
    }
    for (auto t : ts) _ps->Wait(t);
    for (int i = 0; i < workerNum; i++) {
      if (i == workerID) continue; 
      CUDA_CALL(cudaMemcpyAsync(
          _cudaBuffers[rank] + valLen * i, _cpuBuffer + valLen * i, 
          valLen, cudaMemcpyHostToDevice, stream));
    }
  }
  CUDA_CALL(cudaStreamSynchronize(stream));
}


void BytepsScatter(int rank, int len, ncclDataType_t datatype, cudaStream_t stream) {
  auto input_tensor = std::make_shared<GeneralTensor>(tensor, datatype, len);
  auto dtype = input_tensor->dtype();
  auto workerID = BytePSGlobal::GetWorkerID();

  void* baseSrcPtr = _cudaBuffers[rank];
  
  // Scatter to local peer GPUs on the same worker
  auto localSize = BytePSGlobal::GetLocalSize();
  auto globalSize = BytePSGlobal::GetGlobalSize();
  // Assume the len is partitionable 
  auto unitLen = len / globalSize;
  for (int i = 0; i < localSize; i++) {
    if (rank == i) continue; // skip memcpy from myself to myself
    void* baseDstPtr = _cudaBuffers[i];
    void* srcPtr = baseSrcPtr + unitLen * i;
    void* dstPtr = baseDstPtr + unitLen * i;
    CUDA_CALL(cudaMemcpyPeerAsync(dstPtr, rank, src, i, unitLen, stream));
  }

  // Scatter to other distributed workers
  if (BytePSGlobal::IsDistributed()) {
    // Copy to host memory
    auto workerNum = BytePSGlobal::GetNumWorker();
    auto valLen = len / workerNum;
    for (int i = 0; i < workerNum; i++) {
      if (i == workerID) continue; 
      CUDA_CALL(cudaMemcpyAsync(
          _cpuBuffer + valLen * i, _cudaBuffers[rank] + valLen * i,
          valLen, cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CALL(cudaStreamSynchronize(stream));

    std::vector<int> ts;
    auto cmd = GetCommandType(RequestType::kDefaultPushPull, dtype);
    for (int i = 0; i < workerNum; i++) {
      if (i == workerID) continue;
      auto key = i;
      auto &pskv = BytePSGlobal::EncodeDefaultKey(key, valLen);
      // need to use cpu buffer
      ps::SArray<char> vals(_cpuBuffer + valLen * i, valLen, false);
      ts.push_back(_ps->ZPush(pskv.keys, vals, pskv.lens, cmd));
    }
    for (auto t : ts) _ps->Wait(t);
  } else {
    CUDA_CALL(cudaStreamSynchronize(stream));
  }
}

} // namespace sparse
} // namespace byteps 