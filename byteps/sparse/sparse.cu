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
#include "sparse_dense.h"
#include "sparse.cuh"

namespace byteps {
namespace sparse {

/**
  embedBuffers: the addresses of all embedding buffers (could have variable length)
  denseBuffers: the addresses of all dense buffers (the length should be identical)
  embedBufferLens: the length of the embedding buffers (could have variable length)
  denseDeltaBeforeReduceBuffers: the addresses of all dense layer delta before reduce buffers
                                 (the length should be identical)
  denseDeltaReducedBuffers: the addresses of all dense layer delta after reduce buffers
                            (the length should be identical)  
  size: the length of a dense buffer (in bytes), it is equivalent for all GPUs
  sizeDenseDelta: the length of a dense layer Delta buffer for reduceasync (in bytes), 
                  it is equivalent for all GPUs
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
  _embedBuffers.assign(embedBuffers.begin(), embedBuffers.end());
  _denseBuffers.assign(denseBuffers.begin(), denseBuffers.end());

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
    _embedBufferLens[workerID][i] = embedBufferLens[i]; // local buffer length
  }
  _denseBufferLen = size;

  // Check buffer length
  int accuml = 0;
  for (int i = 0; i < localSize; i++) {
    accuml += _embedBufferLens[workerID][i] / localSize;
  }
  CHECK_EQ(accuml, _denseBufferLen) 
      << accuml << " " << _denseBufferLen;

  // Need a continous CPU buffer for each GPU
  _cpuBuffers.clear();
  for (int i = 0; i < localSize; i++) {
    void* _cpuBuffer;
    CUDA_CALL(cudaHostAlloc(
        &_cpuBuffer, size, cudaHostAllocMapped | cudaHostAllocPortable));
    _cpuBuffers.push_back(_cpuBuffer);
  }

  // Start the pslite instance
  auto ps = BytePSSparseCommon::GetPS();
  if (BytePSSparseCommon::IsDistributed() && !ps) {
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
      ps->Wait(ps->ZPush(keys, vals, lens));
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
      ps->Wait(ps->ZPull(keys, &vals, &lens));
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
    size_t dst_len = _denseBufferLen;

    std::string planfile_name("gather_plan_");
    planfile_name += std::to_string(i) + std::string(".json");
    _local_gather_comms[i] = std::make_unique<LocalGatherComm>(
        planfile_name, localSize, srcs, srcs_lens, send_counts, dst, dst_len);
  }

  _local_scatter_comms.resize(localSize);
  for (int i = 0; i < localSize; i++) {
    float* src = (float *)_denseBuffers[i];
    size_t src_len = _denseBufferLen;
    std::vector<float*> scatter_dsts(localSize);
    std::vector<size_t> scatter_dsts_lens(localSize);
    std::vector<size_t> scatter_send_counts(localSize);
    for (int j = 0; j < localSize; j++) {
      scatter_dsts[j] = (float*)_embedBuffers[j] + (i * _embedBufferLens[workerID][j] / localSize);
      scatter_dsts_lens[j] = (localSize - i) * _embedBufferLens[workerID][j] / localSize;
      scatter_send_counts[j] = _embedBufferLens[workerID][j] / localSize;
    }
  }
}

void bytepsSparseInitDense(std::vector<void*>& denseDeltaBeforeReduceBuffers,
                           std::vector<void*>& denseDeltaAfterReduceBuffers,
                           int sizeDenseDelta) {

  CHECK_EQ(denseDeltaBeforeReduceBuffers.size(), denseDeltaAfterReduceBuffers.size());
  _denseDeltaBufferLength = sizeDenseDelta;

  // // Init IPC stuff
  // sharedMemoryInfo info;
  // CHECK_EQ(sharedMemoryCreate(bpsShmName, sizeof(shmStruct), &info), 0);
  // auto shm = (volatile shmStruct *)info.addr;
  // memset((void *)shm, 0, sizeof(*shm));

  auto localSize = BytePSSparseCommon::GetLocalSize();
  auto workerNum = BytePSSparseCommon::GetNumWorker();
  auto workerID = BytePSSparseCommon::GetWorkerID();

  // for (int i = 0; i < localSize; i++) {
  //   cudaDeviceProp prop;
  //   CUDA_CALL(cudaGetDeviceProperties(&prop, i));

  //   // CUDA IPC is only supported on devices with unified addressing
  //   if (!prop.unifiedAddressing) {
  //     // BPS_LOG(INFO) << "Device " << i << " does not support unified addressing, skipping...";
  //     continue;
  //   }
  //   // We require two processes accessing each device, so we need
  //   // to ensure exclusive or prohibited mode is not set
  //   if (prop.computeMode != cudaComputeModeDefault) {
  //     // BPS_LOG(INFO) << "Device " << i << "is in an unsupported compute mode for this sample";
  //     continue;
  //   }

  //   shm->devices[shm->nprocesses++] = i;
  //   CHECK_GT(MAX_CUDA_DEVICES, shm->nprocesses);
  // }

  // CHECK(shm->nprocesses > 0) 
  //     << "No cuda device suppported";
  // CHECK_EQ(shm->nprocesses, embedBuffers.size())
  //     << "Shared memory processes: " << shm->nprocesses 
  //     << ", send buffers: " << embedBuffers.size();

  // We need to manually we need to clear the containers because
  // bytepsSparseInit() might be (unexpectedly) invoked multiple times
  _denseDeltaBeforeReduceBuffers.clear();
  _denseDeltaAfterReduceBuffers.clear();
  for (size_t i = 0; i < localSize; i++) {
    _denseDeltaBeforeReduceBuffers.push_back(denseDeltaBeforeReduceBuffers[i]); 
    _denseDeltaAfterReduceBuffers.push_back(denseDeltaAfterReduceBuffers[i]);
  }

  // Allocate memory and an event for each process and fill 
  // the shared memory buffer with the IPC handles 
  // for (size_t i = 0; i < shm->nprocesses; i++) {
  //   cudaEvent_t event;
  //   CUDA_CALL(cudaSetDevice(
  //       shm->devices[i]));

  //   CUDA_CALL(cudaIpcGetMemHandle(
  //     (cudaIpcMemHandle_t *)&shm->denseDeltaBeforeReduceMemHandle[i], denseDeltaBeforeReduceBuffers[i]));
  //   CUDA_CALL(cudaIpcGetMemHandle(
  //     (cudaIpcMemHandle_t *)&shm->denseDeltaAfterReduceMemHandle[i], denseDeltaAfterReduceBuffers[i]));

  //   // Store the buffers 
  //   _denseDeltaBeforeReduceBuffers.push_back(denseDeltaBeforeReduceBuffers[i]); 
  //   _denseDeltaAfterReduceBuffers.push_back(denseDeltaAfterReduceBuffers[i]);
  // }

  // Get CPU buffer for dense layer reduceasync
  {
    CUDA_CALL(cudaHostAlloc(
        &_cpuDenseDeltaBuffers, sizeDenseDelta, cudaHostAllocMapped | cudaHostAllocPortable));
  }

  // Start the DenseReduce loop
  runDenseReduceLoop(_denseReduceLoop);
  _denseReducer = new ::byteps::common::CpuReducer(nullptr);
}

extern "C" void bytepsSparseInitDensePerGPU(int device_id /* starts with 0 */,
                                            void* denseDeltaBeforeReduceBuffer,
                                            void* denseDeltaAfterReduceBuffer,
                                            int sizeDenseDelta) {
  auto localSize = BytePSSparseCommon::GetLocalSize();
  auto workerNum = BytePSSparseCommon::GetNumWorker();
  auto workerID = BytePSSparseCommon::GetWorkerID();
  assert((device_id < localSize) && "Device id must be within local gpu size.");

  if (device_id == 0){
    _denseDeltaBufferLength = sizeDenseDelta;
    // Get CPU buffer for dense layer reduceasync
    CUDA_CALL(cudaHostAlloc(
        &_cpuDenseDeltaBuffers, sizeDenseDelta, cudaHostAllocMapped | cudaHostAllocPortable));
    // Start the DenseReduce loop
    runDenseReduceLoop(_denseReduceLoop);
    _denseReducer = new ::byteps::common::CpuReducer(nullptr);
  } else{
    CHECK_EQ(_denseDeltaBufferLength, sizeDenseDelta);
  }

  _denseDeltaBeforeReduceBuffers.push_back(denseDeltaBeforeReduceBuffer); 
  _denseDeltaAfterReduceBuffers.push_back(denseDeltaAfterReduceBuffer);
}


void bytepsSparseShutdown() {
}


void bytepsGatherExecAsync(int local_rank, cudaStream_t stream) {
  // Gather from local peer GPUs on the same worker
  auto localSize = BytePSSparseCommon::GetLocalSize();
  auto workerID = BytePSSparseCommon::GetWorkerID();
  auto workerNum = BytePSSparseCommon::GetNumWorker();

  if (local_rank > 0) return; // debug
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


// Currently the implementation is blocking. Async means that API is used in the async
// training of Wide and Deep model such as HugeCTR.


// TODO (chengyu.dai): Add Broadcast for initializing the latestBuffer.
extern "C" void bytepsDenseReduceExecAsync(int local_rank, cudaStream_t stream) {
  auto localSize = BytePSSparseCommon::GetLocalSize();
  auto workerID = BytePSSparseCommon::GetWorkerID();
  void* baseSrcPtr = (void*) (_denseDeltaBeforeReduceBuffers[local_rank]);
  void* baseResultPtr = (void*) (_denseDeltaAfterReduceBuffers[local_rank]);

  size_t buffer_size = _denseDeltaBufferLength;

  // Create a local thread and related mutex to synchronnize.
  std::mutex signal_mtx;
  std::condition_variable signal_cv;
  bool is_ready = false;

  auto reduce_async_job = [& signal_mtx, & signal_cv, & is_ready, baseSrcPtr, baseResultPtr,
                           buffer_size, stream]() {
    // Copy dense layer's param delta D2H.
    CUDA_CALL(cudaMemcpyAsync((void *)_cpuDenseDeltaBuffers, baseSrcPtr, buffer_size, cudaMemcpyDeviceToHost, stream));
    CUDA_CALL(cudaStreamSynchronize(stream));

    // CPU Work to reduce.
    _denseReducer->sum(_cpuDenseLatestBuffers, _cpuDenseDeltaBuffers, _denseDeltaBufferLength /* in bytes*/, DataType::BYTEPS_FLOAT32);

    // Copy dense layer's latest param H2D.
    CUDA_CALL(cudaMemcpyAsync(baseResultPtr, _cpuDenseLatestBuffers, buffer_size, cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaStreamSynchronize(stream));

    std::unique_lock<std::mutex> lck(signal_mtx);
    is_ready = true;
    signal_cv.notify_one();
  };
  _denseReduceLoop->add_worker(reduce_async_job);

  std::unique_lock<std::mutex> lck(signal_mtx);
  while (!is_ready)
    signal_cv.wait(lck);
}

} // namespace sparse
} // namespace byteps 