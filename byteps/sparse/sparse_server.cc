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

void BytePSSparseEngineThread(int i) {
  auto& q = engine_queues_[i];
  while (true) {
    BytePSSparseEngineMessage msg;
    q->WaitAndPop(&msg);
    if (msg.type == TERMINATE) break;
    switch (msg.type) {
      case GATHER: {
        CUDA_CALL(cudaMemcpyAsync(
            (void*) msg.dst, 
            (const void *) msg.src, 
            (size_t) msg.len, 
            (cudaMemcpyKind) cudaMemcpyDeviceToHost, 
            (cudaStream_t) *msg.stream));
        CUDA_CALL(cudaStreamSynchronize(*msg.stream));

        byteps_server_->Response(msg.req_meta, msg.kvpairs);
      } break;

      case SCATTER: {
        CUDA_CALL(cudaMemcpyAsync(
            (void*) msg.dst, 
            (const void *) msg.src, 
            (size_t) msg.len, 
            (cudaMemcpyKind) cudaMemcpyHostToDevice, 
            (cudaStream_t) *msg.stream));
        CUDA_CALL(cudaStreamSynchronize(*msg.stream));

        ps::KVPairs<char> res; // send push response (empty payload)
        byteps_server_->Response(msg.req_meta, res); 
      } break;

      case REDUCE: {

      } break;

      default: 
        CHECK(0) << "Invalid msg type: " << msg.type;
    }
  }
}

void InitCudaIpc() {
  sharedMemoryInfo info;
  volatile shmStruct *shm = NULL;
  CHECK_EQ(openCudaIpcSharedMemory(bpsCudaIpcShmName, sizeof(shmStruct), &info), 0)
      << "shared memory open failed";
  shm = (volatile shmStruct *)info.addr;

  CHECK_EQ(shm->nprocesses, local_size_) 
      << shm->nprocesses << " " << local_size_;

  streams_d2h_.resize(local_size_);
  streams_h2d_.resize(local_size_);
  embed_ipc_handlers_.resize(local_size_);
  embed_bufs_.resize(local_size_);
  embed_buflens_.resize(local_size_);

  for (int gid = 0; gid < local_size_; ++gid) {
    CUDA_CALL(cudaSetDevice(shm->devices[gid]));

    CUDA_CALL(cudaIpcOpenMemHandle(&embed_bufs_[gid], 
              *(cudaIpcMemHandle_t *)&shm->embedMemHandle[gid],
              cudaIpcMemLazyEnablePeerAccess));

    CUDA_CALL(cudaStreamCreateWithFlags(&streams_d2h_[gid], 
              cudaStreamNonBlocking));
    CUDA_CALL(cudaStreamCreateWithFlags(&streams_h2d_[gid], 
              cudaStreamNonBlocking));

    CUDA_CALL(cudaStreamSynchronize(streams_d2h_[gid]));
    CUDA_CALL(cudaStreamSynchronize(streams_h2d_[gid]));

    embed_buflens_[gid] = shm->embedBufferLength[gid];
  }

  if (dense_buflen_ == 0) dense_buflen_ = shm->denseBufferLength;
  else CHECK_EQ(dense_buflen_, shm->denseBufferLength);
}

void InitDenseReduce() {
  sharedMemoryInfo info;
  volatile shmStruct *shm = NULL;
  CHECK_EQ(openCudaIpcSharedMemory(bpsCudaIpcShmName, sizeof(shmStruct), &info), 0)
      << "shared memory open failed";
  shm = (volatile shmStruct *)info.addr;

  if (dense_buflen_ == 0) dense_buflen_ = shm->denseBufferLength;
  else CHECK_EQ(dense_buflen_, shm->denseBufferLength);

  local_dense_bufs_.resize(local_size_);
  for (int gid = 0; gid < local_size_; ++gid) {
    std::string shm_name = std::string(bpsShmName) + std::to_string(gid);
    CHECK_EQ(openSharedMemory(
        shm_name.c_str(), dense_buflen_, &local_dense_bufs_[gid]), 0);
  }

  // init the latest params buffer
  mallocAligned(&lastest_params_buf_, dense_buflen_);
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
  
  if (IsDenseKey(key)) { // dense reduce
    if (!is_dense_inited_) {
      InitDenseReduce();
      is_dense_inited_ = true;
    }

    if (req_meta.push) { // push
      if (IsSenderLocalWorker(req_meta.sender)) {
        CHECK_EQ(local_dense_bufs_.size(), local_size_);

        int src_gpu = (key >> 32) % local_size_;
        void* shm_addr = CHECK_NOTNULL(local_dense_bufs_[src_gpu]);

        size_t len = dense_buflen_ / ps::NumServers();
        size_t offset = len * ps::MyRank();

        void* src = reinterpret_cast<void*>((char*)shm_addr + offset);
        void* dst = reinterpret_cast<void*>((char*)lastest_params_buf_ + offset);

        bps_reducer_->sum(dst, src, len, DataType::BYTEPS_FLOAT32);

        // send push response (empty payload)
        ps::KVPairs<char> res;
        server->Response(req_meta, res);
      } else {

      }
    } else { // pull
      if (IsSenderLocalWorker(req_meta.sender)) {
        CHECK_EQ(local_dense_bufs_.size(), local_size_);

        int src_gpu = (key >> 32) % local_size_;
        void* shm_addr = CHECK_NOTNULL(local_dense_bufs_[src_gpu]);

        size_t len = dense_buflen_ / ps::NumServers();
        size_t offset = len * ps::MyRank();

        void* src = reinterpret_cast<void*>((char*)lastest_params_buf_ + offset);
        void* dst = reinterpret_cast<void*>((char*)shm_addr + offset);
        
        bps_reducer_->copy(dst, src, len);

        // send dummy pull response
        int tmplen = req_meta.val_len;
        if (dense_map_.find(key) == dense_map_.end()) {
          AllocMemoryAndCreateSarray(dense_map_[key].keys, 1, (ps::Key*)&req_data.keys[0]);
          AllocMemoryAndCreateSarray(dense_map_[key].vals, tmplen);
          AllocMemoryAndCreateSarray(dense_map_[key].lens, 1, (int*)&tmplen);
        }
        server->Response(req_meta, dense_map_[key]);
      } else {

      }
    }
  } else if (IsScatterOrGatherKey(key)) {  // gather or scatter
    // as it receives the first gather key, the IPC 
    // handler should have been inited already
    if (!is_ipc_inited_) {
      InitCudaIpc();
      is_ipc_inited_ = true;
    }

    if (req_meta.push) { // scatter request 
      int target_local_gid = (key >> 16) % local_size_;
      int len = req_meta.val_len;

      auto recved = reinterpret_cast<char*>(req_data.vals.data());

      int from_gid = req_meta.cmd; // from which gpu (global id)
      int global_num_gpu = local_size_ * ps::NumServers();
      CHECK_LT(from_gid, global_num_gpu) << from_gid;

      void* src = (void*) recved;
      void* dst = (void*) ((char*)embed_bufs_[target_local_gid] + 
                  embed_buflens_[target_local_gid] / global_num_gpu * from_gid);

      // init engine message
      BytePSSparseEngineMessage msg;
      msg.dst = dst;
      msg.src = src;
      msg.len = len;
      msg.type = SCATTER;
      msg.kvpairs = req_data; // hold the sarrays to prevent it from being released too early
      msg.req_meta = req_meta;
      msg.stream = &streams_h2d_[target_local_gid];

      int qid = (request_id_++) % engine_nthreads_; // load balanced
      engine_queues_[qid]->Push(msg);

    } else { // gather request
      int target_local_gid = (key >> 16) % local_size_;
      int len = req_meta.val_len;
      if (gather_map_.find(key) == gather_map_.end()) {
        AllocMemoryAndCreateSarray(gather_map_[key].keys, 1, (ps::Key*)&req_data.keys[0]);
        AllocMemoryAndCreateSarray(gather_map_[key].vals, len);
        AllocMemoryAndCreateSarray(gather_map_[key].lens, 1, (int*)&len);
      }

      int from_gid = req_meta.cmd; // from which gpu (global id)
      int global_num_gpu = local_size_ * ps::NumServers();
      CHECK_LT(from_gid, global_num_gpu) << from_gid;

      void* src = (void*) ((char*)embed_bufs_[target_local_gid] + 
                  embed_buflens_[target_local_gid] / global_num_gpu * from_gid);
      void* dst = (void*) gather_map_[key].vals.data();

      // init engine message
      BytePSSparseEngineMessage msg;
      msg.dst = dst;
      msg.src = src;
      msg.len = len;
      msg.type = GATHER;
      msg.kvpairs = gather_map_[key]; 
      msg.req_meta = req_meta;
      msg.stream = &streams_d2h_[target_local_gid];

      int qid = (request_id_++) % engine_nthreads_; // load balanced
      engine_queues_[qid]->Push(msg);

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
      if (init_map_.find(key) == init_map_.end()) {
        AllocMemoryAndCreateSarray(init_map_[key].keys, 1, (ps::Key*)&req_data.keys[0]);
        AllocMemoryAndCreateSarray(init_map_[key].vals, len, recved);
        AllocMemoryAndCreateSarray(init_map_[key].lens, 1, (int*)&len);
      }
      
      // send push response (empty payload)
      ps::KVPairs<char> res;
      server->Response(req_meta, res);
    } else { // pull 
      CHECK(init_map_.find(key) != init_map_.end()) << key;
      server->Response(req_meta, init_map_[key]);
    }
  }
}

void InitEnv() {
  if (ps::IsScheduler()) return; // skip this init for the scheduler
  CHECK(getenv("BYTEPS_LOCAL_SIZE")) << "Should init BYTEPS_LOCAL_SIZE";
  local_size_ = atoi(getenv("BYTEPS_LOCAL_SIZE"));

  debug_ = getenv("BYTEPS_SPARSE_SERVER_DEBUG") ? true : false;

  engine_nthreads_ = getenv("BYTEPS_SPARSE_ENGINE_NTHREADS") ?
      atoi(getenv("BYTEPS_SPARSE_ENGINE_NTHREADS")) : 8; 
  LOG(INFO) << "BYTEPS_SPARSE_ENGINE_NTHREADS set to " << engine_nthreads_;
  for (size_t i = 0; i < engine_nthreads_; ++i) {  
    auto q = new TsQueue();
    engine_queues_.push_back(q);
  }

  for (size_t i = 0; i < engine_nthreads_; ++i) {
    auto t = new std::thread(&BytePSSparseEngineThread, i);
    threads_.push_back(t);
  }

  bps_reducer_ = new ::byteps::sparse::CpuReducer(nullptr);
}

void ReleaseServerResources() {
  if (ps::IsScheduler()) return; 

  BytePSSparseEngineMessage msg;
  msg.type = TERMINATE;
  for (auto q : engine_queues_) q->Push(msg);
  for (auto t : threads_) t->join();

  if (byteps_server_) {
    delete byteps_server_;
    byteps_server_ = nullptr;
  }

  if (bps_reducer_) {
    delete bps_reducer_;
    bps_reducer_ = nullptr;
  }

  for (int gid = 0; gid < local_size_; ++gid) {
    CUDA_CALL(cudaIpcCloseMemHandle(embed_bufs_[gid]));
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