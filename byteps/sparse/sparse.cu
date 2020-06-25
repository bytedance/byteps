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

#define BYTEPS_DEBUG

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
  volatile shmStruct *shm = NULL;
  sharedMemoryInfo info;
  CHECK_EQ(createCudaIpcSharedMemory(bpsCudaIpcShmName, sizeof(*shm), &info), 0);
  shm = (volatile shmStruct *)info.addr;
  memset((void *)shm, 0, sizeof(*shm));

  auto localSize = BytePSSparseCommon::GetLocalSize();
  auto workerNum = BytePSSparseCommon::GetNumWorker();
  auto workerID = BytePSSparseCommon::GetWorkerID();
  auto globalSize = localSize * workerNum;

  for (int i = 0; i < localSize; i++) {
    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, i));

    // CUDA IPC is only supported on devices with unified addressing
    CHECK(prop.unifiedAddressing)
        << "Device " << i << " does not support unified addressing.";

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
  _globalEmbedBufLens.resize(workerNum, std::vector<size_t>(localSize));
  _globalTotalEmbedBufLens.resize(workerNum, 0);

  // Allocate memory for each process and fill 
  // the shared memory buffer with the IPC handles 
  for (size_t i = 0; i < shm->nprocesses; i++) {
    CUDA_CALL(cudaSetDevice(
        shm->devices[i]));
    CUDA_CALL(cudaIpcGetMemHandle(
        (cudaIpcMemHandle_t *)&shm->embedMemHandle[i], embedBuffers[i]));
    
    shm->embedBufferLength[i] = embedBufferLens[i];
    // Store the buffers 
    _localEmbedBufLens[i] = embedBufferLens[i]; // local buffer length
  }
  _denseBufferLen = denseBufferLen;
  shm->denseBufferLength = denseBufferLen;

#ifdef BYTEPS_DEBUG
  // For debug: print _localEmbedBufLens
  std::cout << "_localEmbedBufLens:" << std::endl;
  for (auto len : _localEmbedBufLens) 
    std::cout << len << " ";
  std::cout << std::endl;
#endif

  for (int i = 0; i < localSize; i++) {
    _globalEmbedBufLens[workerID][i] = _localEmbedBufLens[i];
  }
  
  // The followings are for the global coordination of 
  // the embedding buffer length, which is equivalent to all-gather 
  auto ps = BytePSSparseCommon::GetPS();
  if (BytePSSparseCommon::IsDistributed()) {
    CHECK(ps); // must init the pslite instance before
    
    // keys
    std::vector<ps::Key> pskeys(workerNum);
    std::vector<ps::SArray<ps::Key>> keys_array; 

    // lens
    std::vector<int> pslens(workerNum);
    std::vector<ps::SArray<int>> lens_array; 

    // vals
    std::vector<ps::SArray<char>> vals_array; 

    auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
    for (int i = 0; i < workerNum; i++) {
      ps::Key key = i;
      int server = i;
      
      // keys 
      pskeys[i] = krs[server].begin() + key;
      ps::SArray<ps::Key> keys;
      keys.reset(&pskeys[i], 1, [](void *){});
      keys_array.push_back(keys);
      
      // lens 
      pslens[i] = sizeof(size_t) * localSize;
      ps::SArray<int> lens;
      lens.reset(&pslens[i], 1, [](void *){});
      lens_array.push_back(lens);

      // vals 
      ps::SArray<char> vals;
      vals.reset((char*)_globalEmbedBufLens[i].data(), localSize * sizeof(size_t), [](void *){});
      vals_array.push_back(vals);
    }

    // Push once to the associated server
    {
      int server = workerID;
      auto keys = keys_array[server];
      auto vals = vals_array[server];
      auto lens = lens_array[server];
      ps->Wait(ps->ZPush(keys, vals, lens));
    }

    ps::Postoffice::Get()->Barrier(
        0, ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);

    // Pull the embedding buffer length of other workers
    for (int i = 0; i < workerNum; i++) {
      if (i == workerID) continue; // skip myself
      int server = i;
      auto keys = keys_array[server];
      auto vals = vals_array[server];
      auto lens = lens_array[server];
      ps->Wait(ps->ZPull(keys, &vals, &lens));
    }
  } // BytePSSparseCommon::IsDistributed()

  for (int wid = 0; wid < workerNum; wid++) {
    for (int gpu = 0; gpu < localSize; gpu++) {
      _globalTotalEmbedBufLens[wid] += _globalEmbedBufLens[wid][gpu];
    }
  }

#ifdef BYTEPS_DEBUG
  // For debug: print _globalEmbedBufLens
  std::cout << "_globalEmbedBufLens:" << std::endl;
  for (auto vec : _globalEmbedBufLens) {
    for (auto len : vec) {
      std::cout << len << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  // For debug: print _globalTotalEmbedBufLens
  std::cout << "_globalTotalEmbedBufLens:" << std::endl;
  for (auto len : _globalTotalEmbedBufLens) {
    std::cout << len << " ";
  }
  std::cout << std::endl;
#endif 

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
  
  if (BytePSSparseCommon::IsDistributed()) {
    // Prepare distributed gather communication
    _dist_gather_comms.resize(localSize);
    for (int i = 0; i < localSize; i++) {
      auto ps = BytePSSparseCommon::GetPS();
      _dist_gather_comms[i] = std::make_unique<DistGatherComm>(ps, _globalEmbedBufLens, 
        _denseBuffers[i], _denseBufferLen, i, localSize, workerID, workerNum);
    }
    // Prepare distributed scatter communication
    _dist_scatter_comms.resize(localSize);
    for (int i = 0; i < localSize; i++) {
      auto ps = BytePSSparseCommon::GetPS();
      _dist_scatter_comms[i] = std::make_unique<DistScatterComm>(ps, _globalEmbedBufLens, 
        _denseBuffers[i], _denseBufferLen, i, localSize, workerID, workerNum);
    }
  } 
}

extern "C" void bytepsSparseInitDensePerGPU(int device_id /* starts with 0 */,
                                            void* denseDeltaBeforeReduceBuffer,
                                            void* denseDeltaAfterReduceBuffer,
                                            int sizeDenseDelta) {
  auto localSize = BytePSSparseCommon::GetLocalSize();
  auto workerNum = BytePSSparseCommon::GetNumWorker();
  auto workerID = BytePSSparseCommon::GetWorkerID();
  CHECK_LT(device_id, localSize) << "Device id must be within local gpu size.";

  LOG(INFO) << "Init BytePS Sparse for dense layers: Device " << device_id;

  _denseDeltaBufferLength = sizeDenseDelta;
  
  auto ps = BytePSSparseCommon::GetPS();
  _dense_reduce_comms.push_back(
      std::make_unique<DenseReduceComm>(
        ps, 
        sizeDenseDelta, 
        denseDeltaBeforeReduceBuffer,
        denseDeltaAfterReduceBuffer,
        device_id,
        localSize, 
        workerID, 
        workerNum
      )
  );
}

void bytepsSparseShutdown() {
}


void bytepsGatherExecAsync(int local_rank, cudaStream_t stream) {
  // Gather from local peer GPUs on the same worker
  _local_gather_comms[local_rank]->ExecAsync();
  
  // Gather from distributed peer GPUs on other workers
  if (BytePSSparseCommon::IsDistributed()) {
    _dist_gather_comms[local_rank]->ExecAsync();
  }
}


void bytepsScatterExecAsync(int local_rank, cudaStream_t stream) {
  // Scatter to local peer GPUs on the same worker
  _local_scatter_comms[local_rank]->ExecAsync();
  
  // Scatter to distributed peer GPUs on other workers
  if (BytePSSparseCommon::IsDistributed()) {
    _dist_scatter_comms[local_rank]->ExecAsync();
  }
}


// TODO (chengyu.dai): Add Broadcast for initializing the latestBuffer.
void bytepsDenseReduceExecAsync(int local_rank, cudaStream_t stream) {
  _dense_reduce_comms[local_rank]->ExecAsync();
}

void bytepsSynchronize(int local_rank, cudaStream_t stream, OP op) { 
  switch (op) {
    case GATHER: {
      _local_gather_comms[local_rank]->Sync();
      if (BytePSSparseCommon::IsDistributed()) {
        _dist_gather_comms[local_rank]->Sync();
      }
    } break;
    case SCATTER: {
      _local_scatter_comms[local_rank]->Sync();
      if (BytePSSparseCommon::IsDistributed()) {
        _dist_scatter_comms[local_rank]->Sync();
      }
    } break;
    default:
      CHECK(0) << "unrecognized operation: " << op;
  }
  CUDA_CALL(cudaStreamSynchronize(stream));
}


// TODO: should merge this with bytepsSynchronize
void bytepsDenseSynchronize(int local_rank, cudaStream_t stream) {
  _dense_reduce_comms[local_rank]->Sync();
}

} // namespace sparse
} // namespace byteps 