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

#include "sparse_server.h"

namespace byteps {
namespace sparse {

using namespace ps;

void InitGatherCudaIpc() {
  sharedMemoryInfo info;
  volatile shmStruct *shm = NULL;
  CHECK_EQ(sharedMemoryOpen(bpsShmName, sizeof(shmStruct), &info), 0)
      << "shared memory open failed";
  shm = (volatile shmStruct *)info.addr;

  CHECK_EQ(shm->nprocesses, local_size_) 
      << shm->nprocesses << " " << local_size_;

  streams_d2h_.resize(local_size_);
  embed_ipc_handlers_.resize(local_size_);
  embed_bufs_.resize(local_size_);

  for (int gid = 0; gid < local_size_; ++gid) {
    CUDA_CALL(cudaSetDevice(shm->devices[gid]));

    CUDA_CALL(cudaIpcOpenMemHandle(&embed_bufs_[gid], 
              *(cudaIpcMemHandle_t *)&shm->embedMemHandle[gid],
              cudaIpcMemLazyEnablePeerAccess));

    CUDA_CALL(cudaStreamCreateWithFlags(&streams_d2h_[gid], 
              cudaStreamNonBlocking));

    CUDA_CALL(cudaStreamSynchronize(streams_d2h_[gid]));
  }
}

template <typename Val>
void BytepsSparseHandler(const ps::KVMeta &req_meta, 
                         const ps::KVPairs<Val> &req_data, 
                         ps::KVServer<Val> *server) {
  uint64_t key = DecodeKey(req_data.keys[0]);

  if (debug_) {
    LOG(INFO) << "receive " << (req_meta.push ? "push" : "pull")
              << "\t key=" << key 
              << "\t len=" << req_meta.val_len
              << "\t sender=" << req_meta.sender
              << "\t cmd=" << req_meta.cmd;
  }

  if ((key & 0xffff) == 0xffff) { // scatter or gather
    if (req_meta.push) { // TODO: scatter request 


    } else { // gather request
    
      // as it receives the first gather key, the IPC 
      // handler should have been inited already
      if (gather_init_stage_ == 0) {
        InitGatherCudaIpc();
        ++gather_init_stage_;
      }

      int target_local_gid = (key >> 16) % local_size_;
      int len = req_meta.val_len;
      if (map_.find(key) == map_.end()) {
        AllocMemoryAndCreateSarray(map_[key].keys, 1, (ps::Key*)&req_data.keys[0]);
        AllocMemoryAndCreateSarray(map_[key].vals, len);
        AllocMemoryAndCreateSarray(map_[key].lens, 1, (int*)&len);
      }

      int from_gid = req_meta.cmd; // from which gpu (global id)
      int global_num_gpu = local_size_ * ps::NumServers();
      CHECK_LT(from_gid, global_num_gpu) << from_gid;

      void* src = (void*) ((char*)embed_bufs_[target_local_gid] + 
                  embed_buflens_[target_local_gid] / global_num_gpu * from_gid);
      void* dst = (void*) map_[key].vals.data();

      CUDA_CALL(cudaMemcpyAsync(
          (void*) dst, 
          (const void *)src, 
          (size_t) len, 
          (cudaMemcpyKind) cudaMemcpyDeviceToHost, 
          (cudaStream_t) streams_d2h_[target_local_gid]));
      CUDA_CALL(cudaStreamSynchronize(streams_d2h_[target_local_gid]));

      server->Response(req_meta, map_[key]);
    }
  } else { 
    // init global buffer length
    if (req_meta.push) {
      CHECK(req_data.lens.size());
      CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[0]) 
          << "key=" << key << ", " 
          << req_data.vals.size() << ", " 
          << req_data.lens[0];

      auto recved = reinterpret_cast<char*>(req_data.vals.data());

      int len = (int) req_data.vals.size();
      if (map_.find(key) == map_.end()) {
        AllocMemoryAndCreateSarray(map_[key].keys, 1, (ps::Key*)&req_data.keys[0]);
        AllocMemoryAndCreateSarray(map_[key].vals, len, recved);
        AllocMemoryAndCreateSarray(map_[key].lens, 1, (int*)&len);
      }

      for (int i = 0; i < local_size_; ++i) {
        embed_buflens_.push_back(*((size_t*)recved + i));
      }
      
      // send push response (empty payload)
      ps::KVPairs<char> res;
      server->Response(req_meta, res);
    } else { // pull 
      CHECK(map_.find(key) != map_.end()) << key;
      server->Response(req_meta, map_[key]);
    }
  }
}

void InitEnv() {
  if (ps::IsScheduler()) return; // skip this init for the scheduler
  CHECK(getenv("BYTEPS_LOCAL_SIZE")) << "Should init BYTEPS_LOCAL_SIZE";
  local_size_ = atoi(getenv("BYTEPS_LOCAL_SIZE"));

  debug_ = getenv("BYTEPS_SPARSE_SERVER_DEBUG") ? true : false;
}

void ReleaseServerResources() {
  if (ps::IsScheduler()) return; 

  if (byteps_server_) {
    delete byteps_server_;
    byteps_server_ = nullptr;
  }

  for (int gid = 0; gid < local_size_; ++gid) {
    CUDA_CALL(cudaIpcCloseMemHandle(embed_bufs_[gid]));
    CUDA_CALL(cudaIpcCloseMemHandle(dense_bufs_[gid]));
    CUDA_CALL(cudaStreamDestroy(streams_d2h_[gid]));
  }

  LOG(INFO) << "Succesfully release all server resources.";
}

extern "C" void bytepsSparseServer() {
  LOG(INFO) << "Launch BytePS Server process for sparse training";

  // should init ps-lite instance before anything else
  ps::Start(0, "byteps_server\0");
  byteps_server_ = new ps::KVServer<char>(0);
  byteps_server_->set_request_handle(BytepsSparseHandler<char>);

  InitEnv();

  // post a barrier to sync the global buffer length
  ps::Postoffice::Get()->Barrier(
      0, ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);

  // this Finalize will also post a barrier
  ps::Finalize(0, true); 

  ReleaseServerResources();
}


} // namespace sparse
} // namespace byteps