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
#include <iostream>

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

  _embedBuffers.assign(embedBuffers.begin(), embedBuffers.end());
  _denseBuffers.assign(denseBuffers.begin(), denseBuffers.end());

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
          (char*)&_globalTotalEmbedBufLens[i], sizeof(size_t), false);
      bufferLenSarrays.push_back(tmp);
      
      // keys
      std::vector<ps::Key> tmp1(1, krs[server].begin() + key);
      ps::SArray<ps::Key> keys(tmp1);
      tmpKeys.push_back(keys);
      
      // lens
      std::vector<int> tmp2(1, sizeof(size_t));
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

/*
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
*/

extern "C" void bytepsSparseInitDensePerGPU(int device_id /* starts with 0 */,
                                            void* denseDeltaBeforeReduceBuffer,
                                            void* denseDeltaAfterReduceBuffer,
                                            int sizeDenseDelta) {
  auto localSize = BytePSSparseCommon::GetLocalSize();
  auto workerNum = BytePSSparseCommon::GetNumWorker();
  auto workerID = BytePSSparseCommon::GetWorkerID();
  assert((device_id < localSize) && "Device id must be within local gpu size.");

  std::cout << "Init BytePS Sparse for dense layers: Device" << device_id << std::endl;

  if (device_id == 0){
    _denseDeltaBufferLength = sizeDenseDelta;
    _mtx_DenseLatestBuffers = new std::mutex();

    // Allocate latest parameter buffer.
    CUDA_CALL(cudaHostAlloc(
      &_cpuDenseLatestBuffers, sizeDenseDelta, cudaHostAllocMapped | cudaHostAllocPortable));

    // Start the DenseReduce loop
    // _denseReducer = new ::byteps::common::CpuReducer(nullptr);
    // runDenseReduceLoop(_denseReduceLoop);

    // Start the 3-stage pipeline: D2H -> CpuReduce -> H2D
    _denseReducer = new ::byteps::common::CpuReducer(nullptr);
    runDenseReducePipeline(_denseD2HLoop, _denseReduceLoop, _denseH2DLoop, _denseReducer);
  } else{
    CHECK_EQ(_denseDeltaBufferLength, sizeDenseDelta);
  }

  // Get CPU buffer for dense layer reduceasync
  void * _cpuDenseDeltaBuffers_per_gpu;
  CUDA_CALL(cudaHostAlloc(
    &_cpuDenseDeltaBuffers_per_gpu, sizeDenseDelta, cudaHostAllocMapped | cudaHostAllocPortable));
  _cpuDenseDeltaBuffers.push_back(_cpuDenseDeltaBuffers_per_gpu);
  _denseDeltaBeforeReduceBuffers.push_back(denseDeltaBeforeReduceBuffer); 
  _denseDeltaAfterReduceBuffers.push_back(denseDeltaAfterReduceBuffer);

  bool is_ready = false;
  std::mutex * mtx = new std::mutex();
  std::condition_variable * signal_cv = new std::condition_variable();
  _is_ready_per_gpu.push_back(is_ready);
  _signal_mtx_per_gpu.push_back(mtx);
  _signal_cv_per_gpu.push_back(signal_cv);
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

// void dense_ready_callback(int local_rank) {
//   // std::mutex signal_mtx = _signal_mtx_per_gpu.at(local_rank);
//   // std::condition_variable signal_cv = _signal_cv_per_gpu.at(local_rank);

//   std::unique_lock<std::mutex> lck(* _signal_mtx_per_gpu.at(local_rank));
//   _is_ready_per_gpu.at(local_rank) = true;
//   _signal_cv_per_gpu.at(local_rank)->notify_one();
// }

// TODO (chengyu.dai): Add Broadcast for initializing the latestBuffer.
void bytepsDenseReduceExecAsync(int local_rank, cudaStream_t stream) {
  auto localSize = BytePSSparseCommon::GetLocalSize();
  auto workerID = BytePSSparseCommon::GetWorkerID();
  void* baseSrcPtr = (void*) (_denseDeltaBeforeReduceBuffers.at(local_rank));
  void* baseResultPtr = (void*) (_denseDeltaAfterReduceBuffers.at(local_rank));

  size_t buffer_size = _denseDeltaBufferLength;

  // Create a local thread and related mutex to synchronnize.
  _is_ready_per_gpu.at(local_rank) = false;

  // auto reduce_async_job = [//& signal_mtx, & signal_cv, & is_ready, 
  //                          local_rank, baseSrcPtr, baseResultPtr,
  //                          buffer_size, stream]() {
  //   // Copy dense layer's param delta D2H.
  //   CUDA_CALL(cudaMemcpyAsync((void *)_cpuDenseDeltaBuffers, baseSrcPtr, buffer_size, cudaMemcpyDeviceToHost, stream));
  //   CUDA_CALL(cudaStreamSynchronize(stream));

  //   // CPU Work to reduce.
  //   _denseReducer->sum(_cpuDenseLatestBuffers, _cpuDenseDeltaBuffers, _denseDeltaBufferLength /* in bytes*/, DataType::BYTEPS_FLOAT32);

  //   // Copy dense layer's latest param H2D.
  //   CUDA_CALL(cudaMemcpyAsync(baseResultPtr, _cpuDenseLatestBuffers, buffer_size, cudaMemcpyHostToDevice, stream));
  //   CUDA_CALL(cudaStreamSynchronize(stream));

  //   dense_ready_callback(local_rank);
  // };
  // _denseReduceLoop->add_worker(reduce_async_job);

  auto dense_ready_callback = 
    [] (int local_rank) {
    std::unique_lock<std::mutex> lck(* _signal_mtx_per_gpu.at(local_rank));
    _is_ready_per_gpu.at(local_rank) = true;
    _signal_cv_per_gpu.at(local_rank)->notify_one();
  };

  DenseTask task;
  {
    task.workerID = workerID;
    task.local_rank = local_rank;
    task.buffer_size = buffer_size; // In bytes.
    task.streamH2D = stream;
    task.streamD2H = stream; // TODO(chengyu.dai): separate the streams for two directions.

    task.baseSrcPtr = baseSrcPtr;
    task.cpuDenseDeltaPtr = (void *) (_cpuDenseDeltaBuffers.at(local_rank));
    task.cpuDenseLatestPtr = _cpuDenseLatestBuffers;
    task.baseResultPtr = baseResultPtr;

    task.allFinishCallback = dense_ready_callback;
  }
  _denseD2HLoop->add_predefined_worker(task);
}

void bytepsDenseSynchronize(int local_rank, cudaStream_t stream) {
  // auto signal_mtx = _signal_mtx_per_gpu.at(local_rank);
  // std::condition_variable signal_cv = _signal_cv_per_gpu.at(local_rank);

  std::unique_lock<std::mutex> lck(* _signal_mtx_per_gpu.at(local_rank));
  while (!_is_ready_per_gpu.at(local_rank))
    _signal_cv_per_gpu.at(local_rank)->wait(lck);
}


} // namespace sparse
} // namespace byteps 