// Copyright 2019 Bytedance Inc. or its affiliates. All Rights Reserved.
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

#ifndef BYTEPS_SERVER_H
#define BYTEPS_SERVER_H

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <set>
#include <unistd.h>
#include "ps/ps.h"
#include "common.h"
#include "../common/ready_table.h"
#include "../common/cpu_reducer.h"
#include "../common/scheduled_queue.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "../common/compressor/compressor.h"
#include "../common/compressor/compressor_registry.h"
#include "../common/global.h"
#if HAVE_CUDA == 1
#include <cuda_runtime.h>
#endif

namespace byteps {
namespace server {

#define SERVER_KEY_TYPE uint64_t
#define SERVER_DATA_TYPE char
#define DEBUG_PRINT_TENSOR_VALUE(X) (*((float *)(X) + 0))
#define DEBUG_PRINT_TENSOR_ADDRESS(X) (reinterpret_cast<uint64_t>(X))

using ps::SArray;
using ps::Key;
using ps::KVServer;
using ps::KVWorker;
using common::ReadyTable;
using BytePSGlobal=byteps::common::BytePSGlobal;

enum BytePSEngineOperation {
  SUM_RECV, SUM_LOCAL_COPY, COPY_FIRST, ALL_RECV, COPY_D2H, COPY_H2D, TERMINATE,
  GPU_SUM_RECV, GPU_SUM_RECV_LOCAL
};

struct PSKV {
  SArray<Key> keys;  // n keys
  SArray<int> lens;  // the length of the i-th value
};

struct BytePSArray {
  char* tensor;
  size_t len;
  int dtype;
  ps::KVPairs<char> tmp_sarray;
  common::DeviceType device = common::CPU;
  // whether the tensor data is managed by the server
  bool managed = true;
  bool registered = false;
};

struct RecvArray {
  int len = -1;
  SArray<char> val;
};

struct UpdateBuf {
  std::vector<ps::KVMeta> request;
  BytePSArray merged;
};

struct BytePSEngineMessage {
  uint64_t id;
  DataHandleType type;
  uint64_t key;
  void* dst;
  void* src;
  size_t len;
  BytePSEngineOperation ops;
  ps::KVPairs<char> sarray; // to temporarily hold it and auto release
  ps::KVMeta req_meta;
};

class GDRCopyManager {
 public: 
  GDRCopyManager() {
#if HAVE_CUDA == 1
    CUDA_CALL(cudaSetDevice(BytePSGlobal::GetVisibleDevice()));
    InitCudaStream();
#endif
  }

  ~GDRCopyManager() {
#if HAVE_CUDA == 1
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& it : map_) {
      cudaError_t e = cudaFreeHost(it.second);
      CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading);
    }
    if (d2h_stream_) {
      CUDA_CALL(cudaStreamSynchronize(*d2h_stream_));
      free(d2h_stream_);
      d2h_stream_ = nullptr;
    }
    if (h2d_stream_) {
      CUDA_CALL(cudaStreamSynchronize(*h2d_stream_));
      free(h2d_stream_);
      h2d_stream_ = nullptr;
    }
#endif
  }

  char* GetBuffer(uint64_t key, size_t len) {
    std::lock_guard<std::mutex> lk(mu_);
    if (map_.find(key) == map_.end()) {
#if HAVE_CUDA == 1
      CUDA_CALL(cudaHostAlloc((void**)&map_[key], len, cudaHostAllocDefault));
#endif
    }
    return map_[key];
  }

  void StorePullAddr(uint64_t key, char* pull_addr) {
    std::lock_guard<std::mutex> lk(mu_);
    pull_addr_map_[key] = pull_addr;
  }

  char* GetPullAddr(uint64_t key) {
    std::lock_guard<std::mutex> lk(mu_);
    CHECK_NE(pull_addr_map_.find(key), pull_addr_map_.end());
    return pull_addr_map_[key];
  }

  void CopyD2H(uint64_t key, void* dst, void* src, size_t len) {
    CHECK(!IsD2HFinished(key)) << key;
#if HAVE_CUDA == 1
    CUDA_CALL(cudaMemcpyAsync(dst, src, len, 
              (cudaMemcpyKind)cudaMemcpyDeviceToHost,
              (cudaStream_t)*d2h_stream_));
    CUDA_CALL(cudaStreamSynchronize(*d2h_stream_));
#endif
  }

  void CopyH2D(void* dst, void* src, size_t len) {
#if HAVE_CUDA == 1
    CUDA_CALL(cudaMemcpyAsync(dst, src, len, 
              (cudaMemcpyKind)cudaMemcpyHostToDevice,
              (cudaStream_t)*h2d_stream_));
    CUDA_CALL(cudaStreamSynchronize(*h2d_stream_));
#endif
  }

  bool IsD2HFinished(uint64_t key) {
    std::lock_guard<std::mutex> lk(state_mu_);
    return is_finished_[key];
  }
  
  void MarkFinish(uint64_t key) {
    std::lock_guard<std::mutex> lk(state_mu_);
    is_finished_[key] = true;
  }

  void ResetFinishState(uint64_t key) {
    std::lock_guard<std::mutex> lk(state_mu_);
    is_finished_[key] = false;
  }

 private:
  void InitCudaStream() {
#if HAVE_CUDA == 1
    CUDA_CALL(cudaSetDevice(BytePSGlobal::GetVisibleDevice()));
    d2h_stream_ = (cudaStream_t*) malloc(sizeof(cudaStream_t));
    h2d_stream_ = (cudaStream_t*) malloc(sizeof(cudaStream_t));
    int greatest_priority;
    cudaError_t e = cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority);
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading);
    CUDA_CALL(cudaStreamCreateWithPriority(
              d2h_stream_, cudaStreamNonBlocking, greatest_priority));
    CUDA_CALL(cudaStreamCreateWithPriority(
              h2d_stream_, cudaStreamNonBlocking, greatest_priority));
    CUDA_CALL(cudaStreamSynchronize(*d2h_stream_));
    CUDA_CALL(cudaStreamSynchronize(*h2d_stream_));
#endif
  }

  std::mutex mu_;
  std::unordered_map<uint64_t, char*> map_;
  std::unordered_map<uint64_t, char*> pull_addr_map_;
  std::mutex state_mu_;
  std::unordered_map<uint64_t, bool> is_finished_;
#if HAVE_CUDA == 1
  cudaStream_t* d2h_stream_ = nullptr;
  cudaStream_t* h2d_stream_ = nullptr;
#endif
};

class BytePSServer {
  public: 
    // Init should be called before Init(), since it
    // uses `getenv` under the hood, calling it from the
    // main thread is recommended to avoid race conditions
    static void InitEnv();
    // initialize the byteps server
    static void Init(int rank);
    static void SendPushResponse(uint64_t key);

    static void BytePSHandler(const ps::KVMeta& req_meta,
                              const ps::KVPairs<char> &req_data,
                              ps::KVServer<char>* server);
    static void BytePSGDRHandler(const ps::KVMeta& req_meta,
                                 const ps::KVPairs<char> &req_data,
                                 ps::KVServer<char>* server);
    static void BytePSGDRv2Handler(const ps::KVMeta& req_meta,
                                   const ps::KVPairs<char> &req_data,
                                   ps::KVServer<char>* server);

    static std::vector<RecvArray> GetRecvPartitions(std::vector<uint64_t> keys) {
      std::vector<RecvArray> arrs;
      std::lock_guard<std::mutex> lk(recved_partitions_mu_);
      for (auto key : keys) {
        auto iter = recved_partitions_.find(key);
        CHECK(iter != recved_partitions_.end()) << key;
        arrs.emplace_back(iter->second);
      }
      return arrs;
    }

    // received tensor info
    static RecvArray GetRecvPartition(uint64_t key) {
      std::lock_guard<std::mutex> lk(recved_partitions_mu_);
      auto iter = recved_partitions_.find(key);
      CHECK(iter != recved_partitions_.end()) << key;
      return iter->second;
    }

    static void SetRecvPartition(uint64_t key, const RecvArray& arr) {
      std::lock_guard<std::mutex> lk(recved_partitions_mu_);
      recved_partitions_[key] = arr;
    }

    static ReadyTable* GetP2PCopyTable() { return p2p_copy_table_; }
    static ReadyTable* GetP2PGroupCopyTable() { return p2p_group_copy_table_; }
    static ReadyTable* GetP2PPullResponseTable() { return p2p_pull_response_table_; }
    static ReadyTable* GetP2PAckTable() { return p2p_ack_table_; }
    static ReadyTable* GetGDRPushPullTable() { return gdr_push_pull_table_; } 
    static ReadyTable* GetGDRAckTable() { return gdr_ack_table_; } 

    static void InitP2PCopyTable();
    static int IsP2PDirectResponse() { return p2p_direct_response_; }
    static void SendPullResponse(uint64_t key, char* data, int len);

    static void InitGDRReadyTable();
    static void LocalPushPull(uint64_t key, char* push_addr, char* pull_addr, size_t len, int dtype);
    static void EnqueueLocalGpuSumTask(uint64_t key, char* input, char* output, size_t len,
                                       int dtype, bool do_copy);


  private:
    // functions
    static void P2PHandler(const ps::KVMeta& req_meta,
                           const ps::KVPairs<char> &req_data,
                           ps::KVServer<char>* server);
    static size_t GetThreadID(uint64_t key, size_t len);
    static BytePSArray* GetStore(uint64_t key);
    static UpdateBuf* GetUpdateBuf(uint64_t key);
    static size_t GetUpdateNumRequest(uint64_t key);
    static void CleanUpdateRequest(uint64_t key);
    static void AddUpdateRequest(uint64_t key, const ps::KVMeta& req_meta);

    static BytePSArray* GetGDRPushBuffer(uint64_t key, int sender);

    // routines
    static void SendPushResponse(uint64_t key, const ps::KVMeta& req,
                                 ps::KVServer<char>* server);
    static void SendPullResponse(const DataHandleType type,
                                const uint64_t key,
                                const ps::KVMeta& req_meta,
                                ps::KVServer<char>* server);
    static void BytePSServerEngineThread(int i);
    static void BytePSServerGDREngineThread(int i);
    static void BytePSServerGDRIndependentThread(int i);

    static size_t GetNumExpectedWorker(uint64_t key);
    static void SetNumExpectedWorker(uint64_t key, size_t num);

    static void InitStoreAndUpdateBuf(uint64_t key, size_t len, int dtype);

    static void PrintServerRecvMessageLog(
        const ps::KVMeta& req_meta, const ps::KVPairs<char> &req_data);

    inline static void SendGDRBufferedPullResponse(int i, BytePSEngineMessage& msg);
    static std::atomic<uint64_t> independent_cnt_;

    // knobs
    static std::atomic<uint64_t> timestamp_;
    static size_t engine_thread_num_;
    static volatile bool is_engine_blocking_;
    static volatile bool log_key_info_;
    static volatile bool sync_mode_;
    static volatile bool debug_mode_;
    static volatile bool enable_schedule_;
    static volatile bool is_server_;
    static int ps_instance_size_;

    // cluster info
    static ps::Node::Role role_;
    static int rank_;
    static size_t num_phy_node_;
    static size_t local_size_;
    static size_t num_byteps_workers_;

    // received tensor info
    static std::unordered_map<uint64_t, RecvArray> recved_partitions_;
    static std::mutex recved_partitions_mu_;

    // debug
    static uint64_t debug_key_;
    static std::mutex debug_mu_;

    // server operations
    static byteps::common::CpuReducer* bps_reducer_;

    // ==== p2p related ====
    static volatile bool should_stop_;
    static int p2p_direct_response_;
    static bool alltoall_batch_recv_;
    // ==== p2p related ====

    static std::vector<KVServer<char>*> byteps_server_;

    static std::mutex pullresp_mu_;
    static std::unordered_map<uint64_t, ps::KVPairs<char> > push_response_map_;
    static std::unordered_map<uint64_t, ps::KVPairs<char> > pull_response_map_;

    // push & pull flag
    static std::vector<std::mutex> flag_mu_;
    static std::vector<std::unordered_map<uint64_t, bool> > is_push_finished_;
    static std::vector<std::unordered_map<uint64_t, std::vector<ps::KVMeta> > > q_pull_reqmeta_;
    static std::vector<std::unordered_map<uint64_t, std::set<int> > > seen_sender_;
    static std::vector<std::unordered_map<uint64_t, size_t> > pull_cnt_;

    // GDR allreduce 
    static GDRCopyManager* gdr_copy_mngr_;
    static std::mutex gdr_push_buffer_mu_;
    static std::unordered_map<uint64_t, std::unordered_map<int, BytePSArray>> gdr_push_buffer_;

    // byteps handler
    static std::mutex handle_mu_;
    static std::unordered_map<uint64_t, UpdateBuf> update_buf_;
    static std::mutex update_buf_mu_;

    // address map
    static std::mutex store_mu_;
    static std::unordered_map<uint64_t, BytePSArray> store_;

    // req_meta
    static std::mutex req_meta_mu_;
    static std::unordered_map<uint64_t, std::pair<ps::KVMeta, ps::KVServer<char>*>> response_meta_;

    // hash function
    static std::mutex hash_mu_;
    static std::unordered_map<uint64_t, size_t> hash_cache_;
    // accumulated tensor size for an engine thread
    static std::vector<uint64_t> acc_load_;

    // ready tables
    static ReadyTable* p2p_copy_table_;
    static ReadyTable* p2p_group_copy_table_;
    static ReadyTable* p2p_pull_response_table_;
    static ReadyTable* p2p_ack_table_;
    static ReadyTable* gdr_push_pull_table_;
    static ReadyTable* gdr_ack_table_;
    static std::unordered_map<uint64_t, std::unique_ptr<common::compressor::Compressor>> compressor_map_;
    
    static std::unordered_map<uint64_t, ps::KVMeta> p2p_pull_reqmetas_;

    static std::mutex expected_worker_mu_;
    static std::unordered_map<uint64_t, size_t> num_expected_workers_;
};

extern "C" void byteps_server();

}  // namespace server
}  // namespace byteps

#endif  // BYTEPS_SERVER_H
