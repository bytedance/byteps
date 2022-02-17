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

#include <chrono>
#include <thread>

#include "server.h"
#include "../common/compressor/utils.h"
#include "queue.h"
#include "../common/error.h"

namespace byteps {
namespace server {

using ps::SArray;
using ps::Key;
using ps::KVServer;
using ps::KVWorker;
using ps::GetEnv;

// engine related
std::vector<PriorityQueue*> engine_queues_;
std::vector<std::thread*> engine_threads_;
std::vector<PriorityQueue*> gdr_engine_queues_;
std::vector<std::thread*> gdr_engine_threads_;
std::vector<PriorityQueue*> independent_queues_;
std::vector<std::thread*> independent_threads_;

// definition
// knobs
std::atomic<uint64_t> BytePSServer::timestamp_;
std::atomic<uint64_t> BytePSServer::independent_cnt_;
size_t BytePSServer::engine_thread_num_ = 8;
volatile bool BytePSServer::is_engine_blocking_ = false;
volatile bool BytePSServer::log_key_info_ = false;
volatile bool BytePSServer::sync_mode_ = true;
volatile bool BytePSServer::debug_mode_ = false;
volatile bool BytePSServer::enable_schedule_ = false;
volatile bool BytePSServer::is_server_ = false;
// received tensors
std::unordered_map<uint64_t, RecvArray> BytePSServer::recved_partitions_;
std::mutex BytePSServer::recved_partitions_mu_;

size_t BytePSServer::num_phy_node_;
size_t BytePSServer::local_size_;
size_t BytePSServer::num_byteps_workers_;
std::mutex BytePSServer::expected_worker_mu_;
std::unordered_map<uint64_t, size_t> BytePSServer::num_expected_workers_;
// number of ps-lite instances. typically 1.
int BytePSServer::ps_instance_size_;

// debug
uint64_t BytePSServer::debug_key_;
std::mutex BytePSServer::debug_mu_;
// server operations
byteps::common::CpuReducer* BytePSServer::bps_reducer_ = nullptr;

GDRCopyManager* BytePSServer::gdr_copy_mngr_ = nullptr;

// ========= for p2p ==========
volatile bool BytePSServer::should_stop_ = false;
int BytePSServer::rank_ = -1;
ps::Node::Role BytePSServer::role_;
int BytePSServer::p2p_direct_response_ = 0;
std::unordered_map<uint64_t, ps::KVMeta> BytePSServer::p2p_pull_reqmetas_;
ReadyTable* BytePSServer::p2p_pull_response_table_;
ReadyTable* BytePSServer::p2p_ack_table_;
// ========= for p2p ==========

std::vector<KVServer<char>*> BytePSServer::byteps_server_;

std::mutex BytePSServer::pullresp_mu_;
std::unordered_map<uint64_t, ps::KVPairs<char> > BytePSServer::push_response_map_;
std::unordered_map<uint64_t, ps::KVPairs<char> > BytePSServer::pull_response_map_;


// push & pull flag
std::vector<std::mutex> BytePSServer::flag_mu_;
std::vector<std::unordered_map<uint64_t, bool> > BytePSServer::is_push_finished_;
std::vector<std::unordered_map<uint64_t, std::vector<ps::KVMeta> > > BytePSServer::q_pull_reqmeta_;
std::vector<std::unordered_map<uint64_t, std::set<int> > > BytePSServer::seen_sender_;
std::vector<std::unordered_map<uint64_t, size_t> > BytePSServer::pull_cnt_;

// ========= for allgather ==========
std::unordered_map<uint64_t, ps::KVMeta> BytePSServer::allgather_pull_reqmetas_;
std::mutex BytePSServer::allgather_pull_resp_mu_;
ReadyTable* BytePSServer::allgather_pull_resp_table_;
ReadyTable* BytePSServer::allgather_pull_ack_table_;

std::unordered_map<uint64_t, ps::KVMeta> BytePSServer::allgather_pull_worker_local_root_reqmetas_;
std::mutex BytePSServer::allgather_pull_worker_local_root_resp_mu_;
ReadyTable* BytePSServer::allgather_pull_worker_local_root_resp_table_;
ReadyTable* BytePSServer::allgather_pull_worker_local_root_ack_table_;

// byteps handler
std::mutex BytePSServer::handle_mu_;
std::unordered_map<uint64_t, UpdateBuf> BytePSServer::update_buf_;
std::mutex BytePSServer::update_buf_mu_;

// address map
std::mutex BytePSServer::store_mu_;
std::unordered_map<uint64_t, BytePSArray> BytePSServer::store_;

std::mutex BytePSServer::gdr_push_buffer_mu_;
std::unordered_map<uint64_t, std::unordered_map<int, BytePSArray>> BytePSServer::gdr_push_buffer_;

// req_meta
std::mutex BytePSServer::req_meta_mu_;
std::unordered_map<uint64_t, std::pair<ps::KVMeta, ps::KVServer<char>*>> BytePSServer::response_meta_;

// hash function
std::mutex BytePSServer::hash_mu_;
std::unordered_map<uint64_t, size_t> BytePSServer::hash_cache_;
std::vector<uint64_t> BytePSServer::acc_load_;

// ready table
ReadyTable* BytePSServer::p2p_copy_table_ = nullptr;
ReadyTable* BytePSServer::p2p_group_copy_table_ = nullptr;
ReadyTable* BytePSServer::gdr_push_pull_table_ = nullptr;

std::mutex SmallTensorMngr::small_tensor_mu_;
std::unordered_map<uint64_t, bool> SmallTensorMngr::small_tensor_map_;

// compression
std::unordered_map<uint64_t, std::unique_ptr<common::compressor::Compressor>> BytePSServer::compressor_map_;

bool BytePSServer::gdr_lazy_sync_;

// TODO: remove Postoffice API calls
uint64_t DecodeKey(ps::Key key) {
  auto kr = ps::Postoffice::Get()->GetServerKeyRanges()[ps::MyRank()];
  return key - kr.begin();
}

uint64_t EncodeKey(ps::Key key) {
  auto kr = ps::Postoffice::Get()->GetServerKeyRanges()[ps::MyRank()];
  return key + kr.begin();
}

UpdateBuf* BytePSServer::GetUpdateBuf(uint64_t key) {
  std::lock_guard<std::mutex> lock(update_buf_mu_);
  return &update_buf_[key];
}

size_t BytePSServer::GetUpdateNumRequest(uint64_t key) {
  std::lock_guard<std::mutex> lock(update_buf_mu_);
  return update_buf_[key].request.size();
}

void BytePSServer::CleanUpdateRequest(uint64_t key) {
  std::lock_guard<std::mutex> lock(update_buf_mu_);
  update_buf_[key].request.clear();
}

void BytePSServer::AddUpdateRequest(uint64_t key, const ps::KVMeta& req_meta) {
  std::lock_guard<std::mutex> lock(update_buf_mu_);
  update_buf_[key].request.push_back(req_meta);
}

size_t BytePSServer::GetThreadID(uint64_t key, size_t len) {
  std::lock_guard<std::mutex> lock(hash_mu_);
  if (len == 0) { // pull
    CHECK_NE(hash_cache_.find(key), hash_cache_.end());
    return hash_cache_[key];
  }
  if (hash_cache_.find(key) != hash_cache_.end()) {
    return hash_cache_[key];
  }
  CHECK_GT(len, 0);
  CHECK_EQ(acc_load_.size(), engine_thread_num_);
  auto min_index = -1;
  auto min_load = std::numeric_limits<uint64_t>::max();
  for (size_t i = 0; i < engine_thread_num_; ++i) {
    if (acc_load_[i] < min_load) {
      min_load = acc_load_[i];
      min_index = i;
    }
  }
  CHECK_GE(min_index, 0);
  CHECK_LT(min_index, (int) engine_thread_num_);
  acc_load_[min_index] += len;
  hash_cache_[key] = min_index;
  return hash_cache_[key];
}

// Opens a file for shared memory.
void* PageAlignedSharedMemory(const std::string& shm_name, size_t size) {
  size_t page_size = sysconf(_SC_PAGESIZE);
  size = RoundUp(size, page_size);
  int shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
  CHECK_GE(shm_fd, 0) << "shm_open failed for " << shm_name;
  CHECK_GE(ftruncate(shm_fd, size), 0) << strerror(errno);
  void* ptr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  CHECK_NE(ptr, (void*)-1) << strerror(errno);
  return ptr;
}

BytePSArray* BytePSServer::GetStore(uint64_t key) {
  std::lock_guard<std::mutex> lock(store_mu_);
  return &store_[key];
}

BytePSArray* BytePSServer::GetGDRPushBuffer(uint64_t key, int sender) {
  std::lock_guard<std::mutex> lock(gdr_push_buffer_mu_);
  return &gdr_push_buffer_[key][sender];
}

void BytePSServer::SendPushResponse(uint64_t key, const ps::KVMeta& req,
                                    ps::KVServer<char>* server){
  auto iterator = push_response_map_.find(key);
  if (iterator == push_response_map_.end()) {  // new key
    ps::KVPairs<char> response;
    push_response_map_[key] = response;  // add to the map
    server->Response(req, response);
  } else {  // not new key, then reuse the memory address to avoid ibv_reg_mr on
            // RDMA data path
    ps::KVPairs<char>* response = &iterator->second;
    server->Response(req, *response);
  }
}

// For alltoall use only 
void BytePSServer::SendPullResponse(uint64_t key, char* data, int len) {
  ps::KVMeta req_meta;
  {
    std::lock_guard<std::mutex> lk(pullresp_mu_);
    req_meta = p2p_pull_reqmetas_[key];
  }
  ps::KVPairs<char> response;
  response.keys = {key};
  response.lens = {len};
  response.vals = ps::SArray<char>(data, len, false); // zero copy
  byteps_server_[0]->Response(req_meta, response);
}

// For allgather use only 
void BytePSServer::SendAllgatherPullResponse(uint64_t key, char* data, int len) {
  ps::KVMeta req_meta;
  {
    std::lock_guard<std::mutex> lk(allgather_pull_resp_mu_);
    req_meta = allgather_pull_reqmetas_[key];
  }
  ps::KVPairs<char> response;
  response.keys = {key};
  response.lens = {len};
  response.vals = ps::SArray<char>(data, len, false); // zero copy
  byteps_server_[0]->Response(req_meta, response);
}

// For allgather use only 
void BytePSServer::SendAllgatherPullWorkerLocalRootResp(uint64_t key, char* data, int len) {
  ps::KVMeta req_meta;
  {
    std::lock_guard<std::mutex> lk(allgather_pull_worker_local_root_resp_mu_);
    req_meta = allgather_pull_worker_local_root_reqmetas_[key];
  }
  ps::KVPairs<char> response;
  response.keys = {key};
  response.lens = {len};
  response.vals = ps::SArray<char>(data, len, false); // zero copy
  byteps_server_[0]->Response(req_meta, response);
}

void BytePSServer::SendPullResponse(const DataHandleType type,
                                    const uint64_t key,
                                    const ps::KVMeta& req_meta,
                                    ps::KVServer<char>* server) {
  std::lock_guard<std::mutex> lock(pullresp_mu_);
  auto updates = GetUpdateBuf(key);
  CHECK(updates->merged.tensor) << "init " << key << " first";
  char* data = updates->merged.tensor;
  auto len = updates->merged.len;

  // send pull response
  auto it = pull_response_map_.find(key);
  if (it == pull_response_map_.end()) {  // new key
    ps::KVPairs<char> response;
    response.keys = {EncodeKey(key)};
    response.lens = {len};
    response.vals = ps::SArray<char>(data, len, false);  // zero copy
    pull_response_map_[key] = response;                  // add to the map
    server->Response(req_meta, response);
  } else {  
    ps::KVPairs<char>* response = &it->second;
    auto p = static_cast<char*>(data);
    CHECK(p);
    response->lens = {len};
    response->vals = ps::SArray<char>(p, len, false);
    server->Response(req_meta, *response);
  }
}

void BytePSServer::SendGDRPullResponse(const DataHandleType type,
                                       const uint64_t key,
                                       const ps::KVMeta& req_meta,
                                       ps::KVServer<char>* server) {
  auto& m = GetUpdateBuf(key)->merged;
  char* data = m.tensor;
  auto len = m.len;
  CHECK(data) << "UpdateBuf of key " << key << " not inited";
  ps::KVPairs<char> response;
  response.keys = {EncodeKey(key)};
  response.lens = {len};
  response.vals = ps::SArray<char>(data, len, false); 
  server->Response(req_meta, response);
}

void BytePSServer::BytePSServerEngineThread(int i) {
  auto& q = engine_queues_[i];
  while (true) {
    BytePSEngineMessage msg;
    q->WaitAndPop(&msg);
    if (msg.ops == TERMINATE) break;
    // do some check
    CHECK(msg.dst);
    CHECK(msg.src);

    auto iter = compressor_map_.find(msg.key);
    if (iter != compressor_map_.end()) {
      // compress
      if (msg.ops == ALL_RECV) {
        common::compressor::tensor_t grad(reinterpret_cast<char*>(msg.src),
                                          msg.len, msg.type.dtype);
        auto compressed = iter->second->Compress(grad);
        // 1. compress
        auto updates = GetUpdateBuf(msg.key);
        updates->merged.tensor = compressed.data;
        updates->merged.len = compressed.size;
      } else {  // decompress
        auto compressed_len = msg.sarray.lens[0];
        CHECK_LE(compressed_len, (int)msg.len);
        common::compressor::tensor_t compressed(
            reinterpret_cast<char*>(msg.src), compressed_len, msg.type.dtype);
        auto decompressed = iter->second->Decompress(compressed);
        msg.src = decompressed.data;
      }
    } else {
      if (msg.ops == ALL_RECV) {
        // 2. no compress
        auto updates = GetUpdateBuf(msg.key);
        updates->merged.len = msg.len;
        // add a copy to avoid push/pull data pollution
        bps_reducer_->copy(updates->merged.tensor, msg.dst, msg.len);
      }
    }

    bool is_debug = (debug_mode_ && (debug_key_ == msg.key));
    switch (msg.ops) {
      case COPY_FIRST: {
        if (is_debug) {
          std::lock_guard<std::mutex> lock(debug_mu_);
          LOG(INFO) << "stage: ENGINE_COPY_MERGED_TO_STORE_BEFORE \t"
                    << "dst: " << DEBUG_PRINT_TENSOR_VALUE(msg.dst) << "\t"
                    << "src: " << DEBUG_PRINT_TENSOR_VALUE(msg.src) << "\t"
                    << "dst_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.dst)
                    << "\t"
                    << "src_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.src)
                    << "\t";
        }
        bps_reducer_->copy(msg.dst, msg.src, msg.len);
        if (is_debug) {
          std::lock_guard<std::mutex> lock(debug_mu_);
          LOG(INFO) << "stage: ENGINE_COPY_MERGED_TO_STORE_AFTER \t"
                    << "dst: " << DEBUG_PRINT_TENSOR_VALUE(msg.dst) << "\t"
                    << "src: " << DEBUG_PRINT_TENSOR_VALUE(msg.src) << "\t"
                    << "dst_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.dst)
                    << "\t"
                    << "src_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.src)
                    << "\t";
        }
      } break;

      case ALL_RECV: {
        auto bps_type = bps_reducer_->GetDataType(msg.type.dtype);
        if (msg.type.requestType == RequestType::kLeaderPushPullAvg) {
          auto updates = GetUpdateBuf(msg.key);
          CHECK_GE(bps_reducer_->div(updates->merged.tensor, msg.len, bps_type,
                                     num_byteps_workers_ * 1.0), 0);
        }
        std::lock_guard<std::mutex> lock(flag_mu_[i]);
        if (is_push_finished_[i].find(msg.key) == is_push_finished_[i].end()) {
          is_push_finished_[i][msg.key] = false;
          pull_cnt_[i][msg.key] = 0;
          seen_sender_[i][msg.key].clear();
        }
        is_push_finished_[i][msg.key] = true;

        auto it = q_pull_reqmeta_[i][msg.key].begin();
        while (it != q_pull_reqmeta_[i][msg.key].end()) {
          if (seen_sender_[i][msg.key].find(it->sender) ==
              seen_sender_[i][msg.key].end()) {
            // TODO: support multi-instance for push-pull
            SendPullResponse(msg.type, msg.key, *it, byteps_server_.at(0));
            pull_cnt_[i][msg.key] += 1;
            seen_sender_[i][msg.key].insert(it->sender);
            it = q_pull_reqmeta_[i][msg.key].erase(it);
          } else {
            ++it;
          }
          if (pull_cnt_[i][msg.key] == GetNumExpectedWorker(msg.key)) {
            is_push_finished_[i][msg.key] = false;
            pull_cnt_[i][msg.key] = 0;
            seen_sender_[i][msg.key].clear();
            break;
          }
        }
      } break;

      case SUM_RECV: {
        auto bps_type = bps_reducer_->GetDataType(msg.type.dtype);
        if (is_debug) {
          std::lock_guard<std::mutex> lock(debug_mu_);
          LOG(INFO) << "stage: ENGINE_SUM_RECV_BEFORE \t"
                    << "dst: " << DEBUG_PRINT_TENSOR_VALUE(msg.dst) << "\t"
                    << "src: " << DEBUG_PRINT_TENSOR_VALUE(msg.src) << "\t"
                    << "dst_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.dst)
                    << "\t"
                    << "src_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.src)
                    << "\t";
        }
        CHECK_GE(bps_reducer_->sum(msg.dst, msg.src, msg.len, bps_type), 0);
        if (is_debug) {
          std::lock_guard<std::mutex> lock(debug_mu_);
          LOG(INFO) << "stage: ENGINE_SUM_RECV_AFTER \t"
                    << "dst: " << DEBUG_PRINT_TENSOR_VALUE(msg.dst) << "\t"
                    << "src: " << DEBUG_PRINT_TENSOR_VALUE(msg.src) << "\t"
                    << "dst_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.dst)
                    << "\t"
                    << "src_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.src)
                    << "\t";
        }
      } break;
      default:
        CHECK(0);
    }
  }
}

void BytePSServer::BytePSServerGDRIndependentThread(int i) {
  auto& q = independent_queues_[i];
  while (true) {
    BytePSEngineMessage msg;
    q->WaitAndPop(&msg);
    if (msg.ops == TERMINATE) break;

    auto stored = GetStore(msg.key);
    switch (msg.ops) {
      case COPY_D2H: {
        gdr_copy_mngr_->CopyD2H(msg.key, 
            (void*)msg.dst, (void*)msg.src, msg.len);
        gdr_copy_mngr_->MarkFinish(msg.key);

        // stored = stored + buff
        BytePSEngineMessage new_msg;
        new_msg.id = timestamp_++;
        new_msg.key = msg.key;
        new_msg.dst = stored->tensor;
        new_msg.src = msg.dst;
        new_msg.len = msg.len;
        new_msg.ops = SUM_LOCAL_COPY;
        int tid = GetThreadID(msg.key, msg.len);
        gdr_engine_queues_[tid]->Push(std::move(new_msg));
      } // case COPY_D2H
      break;

      case COPY_H2D: {
        gdr_copy_mngr_->CopyH2D(
            (void*)msg.dst,(void*)msg.src, msg.len);
        GetGDRPushPullTable()->AddReadyCount(msg.key);
      } // case COPY_H2D
      break;

      default: CHECK(false) << "Unknown engine msg type: " << msg.ops;
    } // switch
  }   // while
}

inline void BytePSServer::SendGDRBufferedPullResponse(int i, BytePSEngineMessage& msg) {
  std::lock_guard<std::mutex> lock(flag_mu_[i]);
  CHECK(BytePSGlobal::IsGDR());
  if (is_push_finished_[i].find(msg.key) == is_push_finished_[i].end()) {
    is_push_finished_[i][msg.key] = false;
    pull_cnt_[i][msg.key] = 0;
    seen_sender_[i][msg.key].clear();
  }
  is_push_finished_[i][msg.key] = true;
  auto it = q_pull_reqmeta_[i][msg.key].begin();
  while (it != q_pull_reqmeta_[i][msg.key].end()) {
    if (seen_sender_[i][msg.key].find(it->sender) ==
        seen_sender_[i][msg.key].end()) {
      if (BytePSGlobal::IsGDRGpu2Gpu()) {
        SendGDRPullResponse(msg.type, msg.key, *it, byteps_server_.at(0));
      } else {
        SendPullResponse(msg.type, msg.key, *it, byteps_server_.at(0));
      }
      pull_cnt_[i][msg.key] += 1;
      seen_sender_[i][msg.key].insert(it->sender);
      it = q_pull_reqmeta_[i][msg.key].erase(it);
    } else {
      ++it;
    }
    size_t expected = SmallTensorMngr::IsRegistered(msg.key) 
                    ? BytePSGlobal::GetSize() : (num_phy_node_ - 1);
    if (pull_cnt_[i][msg.key] == expected) {
      is_push_finished_[i][msg.key] = false;
      pull_cnt_[i][msg.key] = 0;
      seen_sender_[i][msg.key].clear();
      break;
    }
  }
}

void BytePSServer::BytePSServerGDREngineThread(int i) {
  auto& q = gdr_engine_queues_[i];
  while (true) {
    BytePSEngineMessage msg;
    q->WaitAndPop(&msg);
    if (msg.ops == TERMINATE) break;

    switch (msg.ops) {
#if HAVE_CUDA == 1
      case GPU_SUM_RECV_LOCAL: {
        gdr_copy_mngr_->StoreLocalAddress(msg.key, (char*)msg.src);
      }
      // intentionally go to next without break
      case GPU_SUM_RECV: {
        auto stored = GetStore(msg.key);
        auto cuda_reducer = BytePSGlobal::GetCudaReducer(i);
        cudaStream_t* stream = BytePSGlobal::GetCudaReducerStream(i);
        CHECK_EQ(stored->len, msg.len) << stored->len << " " << msg.len;
        CHECK(msg.src);
        CHECK(stored->tensor);
        int num_req = GetUpdateNumRequest(msg.key);
        if (num_req == 0) {
          cuda_reducer->CopyD2DAsync(stored->tensor, msg.src, msg.len, stream);
          if (!gdr_lazy_sync_) CUDA_CALL(cudaStreamSynchronize(*stream));
        } else {
          auto bps_dtype = cuda_reducer->GetDataType(msg.type.dtype);
          cuda_reducer->SumAsync(stored->tensor, msg.src, msg.len, bps_dtype, stream);
          if (!gdr_lazy_sync_) CUDA_CALL(cudaStreamSynchronize(*stream));
        }
        AddUpdateRequest(msg.key, msg.req_meta);
        size_t expected = SmallTensorMngr::IsRegistered(msg.key) 
                        ? BytePSGlobal::GetSize() : GetNumExpectedWorker(msg.key);
        size_t num_req_new = GetUpdateNumRequest(msg.key);
        CHECK(num_req_new <= expected);
        if (num_req_new == expected) {
          CHECK(GetUpdateBuf(msg.key)->merged.tensor);
          cuda_reducer->CopyD2DAsync(GetUpdateBuf(msg.key)->merged.tensor, stored->tensor, msg.len, stream);
          CUDA_CALL(cudaStreamSynchronize(*stream));
          CleanUpdateRequest(msg.key);
          SendGDRBufferedPullResponse(i, msg);
          if (!SmallTensorMngr::IsRegistered(msg.key)) {
            char* local_addr = gdr_copy_mngr_->GetLocalAddress(msg.key);
            if (local_addr) {
              cuda_reducer->CopyD2DAsync(local_addr, GetUpdateBuf(msg.key)->merged.tensor, msg.len, stream);
              CUDA_CALL(cudaStreamSynchronize(*stream));
            }
            // add count only when copy_d2d is done
            GetGDRPushPullTable()->AddReadyCount(msg.key);
          }
        }
      } 
      break; // case GPU_SUM_RECV
#endif
      case SUM_LOCAL_COPY:
        // go to SUM_RECV
      case SUM_RECV: {
        auto stored = GetStore(msg.key);
        // stored = stored + recved
        if (!stored->tensor) { // first 
          stored->tensor = (char*) msg.src;
        } else {
          auto bps_dtype = bps_reducer_->GetDataType(stored->dtype);
          bps_reducer_->sum(stored->tensor, msg.src, msg.len, bps_dtype);
        }

        if (msg.ops == SUM_RECV) { // might be SUM_LOCAL_COPY
          // buffer the request meta
          AddUpdateRequest(msg.key, msg.req_meta);
        }

        // check if all SUM_RECV done (including potential local push)
        if (GetUpdateNumRequest(msg.key) == GetNumExpectedWorker(msg.key)) {
          if (!gdr_copy_mngr_->IsD2HFinished(msg.key)) break;
          q->ClearCounter(msg.key);
          CleanUpdateRequest(msg.key);
          gdr_copy_mngr_->ResetFinishState(msg.key);
          // do not break, just let it jump to case ALL_RECV
        } else {
          break;
        }
      } // case SUM_RECV 

      case ALL_RECV: {
        /* Overview:
         Step 1: copy to pull buffer 
         Step 2: local H2D copy 
         Step 3: call asynchronous SendPullResponse
        */
        auto stored = GetStore(msg.key);

        // Step 1: copy to pull buffer, and set stored to null
        bps_reducer_->copy(GetUpdateBuf(msg.key)->merged.tensor, stored->tensor, msg.len);
        stored->tensor = NULL;
        
        // Step 2: local H2D copy 
        BytePSEngineMessage new_msg;
        new_msg.key = msg.key;
        new_msg.dst = gdr_copy_mngr_->GetPullAddr(msg.key);
        new_msg.src = GetUpdateBuf(msg.key)->merged.tensor;
        new_msg.len = msg.len;
        new_msg.ops = COPY_H2D;
        independent_queues_[independent_cnt_++ % engine_thread_num_]->Push(std::move(new_msg));

        // Step 3: call asynchronous SendPullResponse
        SendGDRBufferedPullResponse(i, msg);
      } 
      break; // case ALL_RECV

      default: CHECK(false) << "Unknown engine msg type: " << msg.ops;
    }
  }
}

void BytePSServer::SendPushResponse(uint64_t key) {
  std::lock_guard<std::mutex> lk(req_meta_mu_);
  CHECK_NE(response_meta_.find(key), response_meta_.end()) << key;
  auto response_meta = response_meta_[key];
  SendPushResponse(key, response_meta.first, response_meta.second);
}

void BytePSServer::P2PHandler(const ps::KVMeta& req_meta,
                              const ps::KVPairs<char> &req_data,
                              ps::KVServer<char>* server) {
  std::lock_guard<std::mutex> lock(handle_mu_); // push & pull may have racing
  DataHandleType type = DepairDataHandleType(req_meta.cmd);
  auto req_type = type.requestType;
  CHECK(req_type == RequestType::kDefaultSend
        || req_type == RequestType::kGroupSend
        || req_type == RequestType::kEmptyGroupSend
        || req_type == RequestType::kDefaultPull
        || req_type == RequestType::kAckSignal
        || req_type == RequestType::kAllgatherPull
        || req_type == RequestType::kAllgatherPullAck
        || req_type == RequestType::kAllgatherPullWorkerLocalRoot
        || req_type == RequestType::kAllgatherPullWorkerLocalRootAck);
  // do some check
  CHECK_EQ(req_data.keys.size(), (size_t)1);
  if (req_meta.push) {
    CHECK_EQ(req_data.lens.size(), (size_t)1) << req_data.lens.size();
    CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[0])
      << req_data.vals.size() << " v.s. " << req_data.lens[0];
  }
  uint64_t key = DecodeKey(req_data.keys[0]);
  if (debug_mode_) {
    LOG(INFO) << "server receive key=" << key
              << "\t msgtype=" << (req_meta.push ? "push" : "pull")
              << "\t sender=" << req_meta.sender;
  }

  if (req_type == RequestType::kAckSignal || req_type == RequestType::kAllgatherPullAck ||
      req_type == RequestType::kAllgatherPullWorkerLocalRootAck) {
    ReadyTable* table = nullptr;
    switch (req_type) {
      case RequestType::kAckSignal:
        table = GetP2PAckTable();
        break;
      case RequestType::kAllgatherPullAck:
        table = GetAllgatherPullAckTable();
        break;
      case RequestType::kAllgatherPullWorkerLocalRootAck:
        table = GetAllgatherPullWorkerLocalRootAckTable();
        break;
      default:
        BPS_CHECK(false) << "unknown ack signal";
    }

    table->AddReadyCount(key);
    return;
  }

  // not ack message
  size_t len;
  if (req_meta.push) {
    len = (size_t) req_data.lens[0];
  } else { // pull 
    if (req_type == RequestType::kDefaultPull) {
      {
        std::lock_guard<std::mutex> lk(pullresp_mu_);
        p2p_pull_reqmetas_[key] = req_meta;
      }
      GetP2PPullResponseTable()->AddReadyCount(key);
    } else if (req_type == RequestType::kAllgatherPull) {
      {
        std::lock_guard<std::mutex> lk(allgather_pull_resp_mu_);
        allgather_pull_reqmetas_[key] = req_meta;
      }
      GetAllgatherPullRespTable()->AddReadyCount(key);
    } else if (req_type == RequestType::kAllgatherPullWorkerLocalRoot) {
      {
        std::lock_guard<std::mutex> lk(allgather_pull_worker_local_root_resp_mu_);
        allgather_pull_worker_local_root_reqmetas_[key] = req_meta;
      }
      GetAllgatherPullWorkerLocalRootRespTable()->AddReadyCount(key);
    }

    return;
  }
  auto stored = GetStore(key);
  // initialization
  if (!stored->tensor) {
    if (type.device == common::GPU) {
      int worker_rank = key >> 32;
      // prepare buffer
      // CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank()));
      ps::SArray<ps::Key> keys;
      ps::SArray<char> vals;
      ps::SArray<int> lens;
      CUDA_CALL(cudaMalloc(&(stored->tensor), len));
      keys.push_back(req_data.keys[0]);
      vals.reset((char*) stored->tensor, len * sizeof(char), [](void *){});
      lens.push_back(len);
      stored->device = common::GPU;
      // perform registration
      server->RegisterRecvBufferWithRank(worker_rank, keys, vals, lens);
    } else {
      // we directly use the data pointer allocated in ps-lite
      // note that this pointer may change. we store the value
      // here simply to indicate that this key was seen before
      stored->tensor = req_data.vals.data();
      // the tensor is managed by pslite, not us
      stored->managed = false;
    }
    if (log_key_info_) {
      LOG(INFO) << "stored tensor for key=" << key << "\tcreated, rank="
                << rank_ << " len=" << len << " device=" << stored->device;
    }
    // the max length
    stored->len = len;
    stored->dtype = type.dtype;
    CHECK(stored->tensor);
    SendPushResponse(key, req_meta, server);
    return;
  }
  // buffer the request meta
  {
    std::lock_guard<std::mutex> lk(req_meta_mu_);
    response_meta_[key] = std::make_pair(req_meta, server);
  }
  RecvArray arr;
  arr.val = req_data.vals;
  arr.len = len;
  // group send request, where the output size is unknown
  bool is_empty = type.requestType == RequestType::kEmptyGroupSend;
  if (type.requestType == RequestType::kGroupSend || is_empty) {
    // Note: the task is created using tensor_id as the key
    uint64_t tensor_key = GetAlltoallTensorId(key);
    // set recv len
    arr.len = is_empty ? 0 : len;
    if (debug_mode_) {
      LOG(INFO) << "GROUP SEND: set recv val addr = " << (long long) req_data.vals.data()
                << ", tensor_id = " << tensor_key << ", key = " << key;
    }
    SetRecvPartition(key, arr);
    GetP2PGroupCopyTable()->AddReadyCount(tensor_key);
    if (p2p_direct_response_ == 1) {
      BytePSServer::SendPushResponse(key);
    }
  } else {
    if (debug_mode_) {
      LOG(INFO) << "set recv val addr = " << (long long) req_data.vals.data()
                << " len =  " << len;
    }
    SetRecvPartition(key, arr);
    GetP2PCopyTable()->AddReadyCount(key);
    if (p2p_direct_response_ == 1) {
      BytePSServer::SendPushResponse(key);
    }
  }
}

void BytePSServer::BytePSHandler(const ps::KVMeta& req_meta,
                                 const ps::KVPairs<char> &req_data,
                                 ps::KVServer<char>* server) {
  DataHandleType type = DepairDataHandleType(req_meta.cmd);
  auto req_type = type.requestType;
  if (req_type == RequestType::kDefaultSend
      || req_type == RequestType::kGroupSend
      || req_type == RequestType::kEmptyGroupSend
      || req_type == RequestType::kDefaultPull
      || req_type == RequestType::kAckSignal
      || req_type == RequestType::kAllgatherPull
      || req_type == RequestType::kAllgatherPullAck
      || req_type == RequestType::kAllgatherPullWorkerLocalRoot
      || req_type == RequestType::kAllgatherPullWorkerLocalRootAck) {
    P2PHandler(req_meta, req_data, server);
    return;
  } else if (type.requestType == RequestType::kGDRPushPull) {
    BytePSGDRHandler(req_meta, req_data, server);
    return;
  } else if (type.requestType == RequestType::kGDRv2PushPull
             || type.requestType == RequestType::kGDRv2PushPullSmall) {
    BytePSGDRv2Handler(req_meta, req_data, server);
    return;
  }
  CHECK_EQ(req_data.keys.size(), (size_t)1);
  uint64_t key = DecodeKey(req_data.keys[0]);
  if (type.requestType == RequestType::kLeaderPushPull
      || type.requestType == RequestType::kLeaderPushPullAvg) {
    SetNumExpectedWorker(key, num_phy_node_);
  } else {
    SetNumExpectedWorker(key, (size_t) ps::NumWorkers());
  }
  std::lock_guard<std::mutex> lock(handle_mu_); // push & pull may have racing
  if (log_key_info_) {
    if (req_meta.push) {
      CHECK_EQ(req_data.lens.size(), (size_t)1);
      CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[0]);
      LOG(INFO) << "push key=" << DecodeKey(req_data.keys[0])
                << "\t sender=" << req_meta.sender
                << "\t size=" << (size_t)req_data.lens[0];
    } else {
      LOG(INFO) << "pull key=" << (uint64_t)DecodeKey(req_data.keys[0])
                << "\t sender=" << req_meta.sender;
    }
  }

  // register compressor
  if (type.requestType == RequestType::kCompressedPushPull) {
    if (compressor_map_.find(key) == compressor_map_.end()) {
      std::string content{reinterpret_cast<char*>(req_data.vals.data()),
                          static_cast<size_t>(req_data.lens[0])};
      auto kwargs = byteps::common::compressor::Deserialize(content);
      auto stored = GetStore(key);
      size_t aligned_size = byteps::common::Align(stored->len, stored->dtype);
      auto compressor_ptr =
          byteps::common::compressor::CompressorRegistry::Create(
              kwargs, aligned_size,
              static_cast<byteps::common::DataType>(stored->dtype));
      CHECK_NE(compressor_ptr, nullptr);
      compressor_map_[key] = std::move(compressor_ptr);
      if (log_key_info_) {
        LOG(INFO) << "register compressor for key=" << key;
      }
    }

    // buffer the request meta
    auto updates = GetUpdateBuf(key);
    updates->request.push_back(req_meta);
    // should send response after collecting all init push
    if (updates->request.size() < GetNumExpectedWorker(key)) return;

    for (const auto& req : updates->request) {
      SendPushResponse(key, req, server);
    }
    updates->request.clear();
    return;
  }

  if (req_meta.push) {  // push request
    CHECK_EQ(req_data.lens.size(), (size_t)1);
    CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[0]);
    auto stored = GetStore(key);
    auto len = (size_t)req_data.lens[0];
    auto recved = reinterpret_cast<char*>(req_data.vals.data());

    if (!stored->tensor) {
      if (sync_mode_) {
        std::lock_guard<std::mutex> lock(update_buf_mu_);
        if (update_buf_.find(key) == update_buf_.end()) {
          update_buf_[key].merged.len = len;
          update_buf_[key].merged.dtype = type.dtype;
          PageAlignedMalloc((void**)&update_buf_[key].merged.tensor, len);
        }
      }
      // buffer the request meta
      auto updates = GetUpdateBuf(key);
      updates->request.push_back(req_meta);
      // should send response after collecting all init push
      if (updates->request.size() < GetNumExpectedWorker(key)) return;
      if (log_key_info_) {
        LOG(INFO) << "Collected all " << updates->request.size()
                  << " requests for key=" << key
                  << ", init the store buffer size="
                  << (size_t)req_data.lens[0];
      }
      // init stored buffer, use page aligned memory
      size_t aligned_size = common::Align(len, type.dtype);
      PageAlignedMalloc((void**)&stored->tensor, aligned_size);
      stored->len = len;
      stored->dtype = type.dtype;
      CHECK(stored->tensor);

      for (const auto& req : updates->request) {
        SendPushResponse(key, req, server);
      }
      updates->request.clear();
    } else {
      auto updates = GetUpdateBuf(key);
      auto tid = GetThreadID(key, len);
      if (updates->request.empty()) {  // from the first incoming worker
        if (sync_mode_) {
          if (debug_mode_ && (debug_key_ == key)) {
            std::lock_guard<std::mutex> lock(debug_mu_);
            LOG(INFO) << "stage: FIRST_WORKER_RECV \t"
                      << "stored: " << DEBUG_PRINT_TENSOR_VALUE(stored->tensor)
                      << "\t"
                      << "recved: " << DEBUG_PRINT_TENSOR_VALUE(recved) << "\t"
                      << "len: " << len << "\t"
                      << "addr: " << DEBUG_PRINT_TENSOR_ADDRESS(recved);
          }
          updates->merged.tmp_sarray = req_data;
          // copy
          BytePSEngineMessage msg = {timestamp_++,   type,     key,
                                     stored->tensor, recved,   stored->len,
                                     COPY_FIRST,     req_data, req_meta};
          engine_queues_[tid]->Push(msg);
        } else {  // async mode, directly add to the buffer
          CHECK_GE(bps_reducer_->sum((void*)stored->tensor, (void*)recved, len,
                                     bps_reducer_->GetDataType(stored->dtype)),
                   0);
        }
      } else {  // from other workers
        CHECK(sync_mode_);
        // CHECK(updates->merged.tensor);
        if (debug_mode_ && (debug_key_ == key)) {
          std::lock_guard<std::mutex> lock(debug_mu_);
          LOG(INFO) << "stage: OTHER_WORKER_SUM \t"
                    << "stored: " << DEBUG_PRINT_TENSOR_VALUE(stored->tensor)
                    << "\t"
                    << "recved: " << DEBUG_PRINT_TENSOR_VALUE(recved) << "\t"
                    << "len: " << len << "\t"
                    << "addr: " << DEBUG_PRINT_TENSOR_ADDRESS(recved);
        }
        if (is_engine_blocking_) {
          // TODO: decompress
          CHECK_GE(bps_reducer_->sum(
                       (void*)updates->merged.tensor, (void*)recved, len,
                       bps_reducer_->GetDataType(updates->merged.dtype)),
                   0);
        } else {  // non-blocking
          BytePSEngineMessage msg = {timestamp_++,   type,     key,
                                     stored->tensor, recved,   stored->len,
                                     SUM_RECV,       req_data, req_meta};
          engine_queues_[tid]->Push(msg);
        }
      }
      // add a worker information (request.size() is the # workers received)
      updates->request.push_back(req_meta);
      SendPushResponse(key, req_meta, server);
      if (sync_mode_ && updates->request.size() == GetNumExpectedWorker(key)) {
        auto stored = GetStore(key);
        if (debug_mode_ && (debug_key_ == key)) {
          std::lock_guard<std::mutex> lock(debug_mu_);
          LOG(INFO) << "stage: COPY_MERGED_TO_STORE \t"
                    << "stored: " << DEBUG_PRINT_TENSOR_VALUE(stored->tensor)
                    << "\t"
                    // << "merged: "
                    // << DEBUG_PRINT_TENSOR_VALUE(updates->merged.tensor) << "\t"
                    << "recved: " << DEBUG_PRINT_TENSOR_VALUE(recved);
        }
        if (is_engine_blocking_) {
          // TODO: compress
          bps_reducer_->copy(stored->tensor, updates->merged.tensor, len);
        } else {
          BytePSEngineMessage msg = {
              timestamp_++,   type,        key,     stored->tensor,
              stored->tensor, stored->len, ALL_RECV};
          engine_queues_[tid]->Push(msg);
          engine_queues_[tid]->ClearCounter(key);
        }
        updates->request.clear();
      } else if (!sync_mode_) {
        // async: clean the request buffer
        updates->request.clear();
      }
    }
  } else {  // pull request
    auto stored = GetStore(key);
    CHECK(stored->tensor) << "Should init the buffer for key=" << key
                          << " first";
    if (is_engine_blocking_ || !sync_mode_) {
      SendPullResponse(type, key, req_meta, server);
    } else {
      auto tid = GetThreadID(key, 0);
      std::lock_guard<std::mutex> lock(flag_mu_[tid]);
      if (is_push_finished_[tid].find(key) == is_push_finished_[tid].end()) {
        is_push_finished_[tid][key] = false;
        pull_cnt_[tid][key] = 0;
        seen_sender_[tid][key].clear();
      }

      auto it = seen_sender_[tid][key].find(req_meta.sender);
      if (is_push_finished_[tid][key] && (it == seen_sender_[tid][key].end())) {
        // push already finished && not received the associated pull response
        // yet
        SendPullResponse(type, key, req_meta, server);
        pull_cnt_[tid][key] += 1;
        seen_sender_[tid][key].insert(req_meta.sender);

        if (pull_cnt_[tid][key] == GetNumExpectedWorker(key)) {
          is_push_finished_[tid][key] = false;
          pull_cnt_[tid][key] = 0;
          seen_sender_[tid][key].clear();
        }
      } else {
        // push not finished, put into the queue, and wait for the engine
        q_pull_reqmeta_[tid][key].push_back(req_meta);
      }
    }
  }
}

void BytePSServer::InitStoreAndUpdateBuf(uint64_t key, size_t len, int dtype) {
  std::lock_guard<std::mutex> lock(handle_mu_); 
  auto stored = GetStore(key);
  auto updates = GetUpdateBuf(key);
  if (updates->merged.tensor) return; // return if inited before
  stored->len = len;
  stored->dtype = dtype;
  updates->merged.len = len;
  updates->merged.dtype = dtype;
  auto align_size = common::Align(len, dtype);
  PageAlignedMalloc((void**)&updates->merged.tensor, align_size);
  CHECK(updates->merged.tensor);
#if HAVE_CUDA == 1
  // cuda host register
  cudaError_t e = cudaHostRegister(updates->merged.tensor, align_size, cudaHostRegisterDefault);
  CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)
      << "CUDA: " << cudaGetErrorString(e);
  if (e == cudaSuccess) {
    updates->merged.registered = true;
  }
#endif
}

void BytePSServer::PrintServerRecvMessageLog(const ps::KVMeta& req_meta,
                                   const ps::KVPairs<char> &req_data) {
  if (!log_key_info_) return;
  std::string ack_string("");
  if (req_meta.push) {
    CHECK_EQ(req_data.lens.size(), (size_t)1);
    CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[0]);
    LOG(INFO) << "push key=" << DecodeKey(req_data.keys[0])
              << "\t sender=" << req_meta.sender
              << "\t size=" << (size_t)req_data.lens[0]
              << "\t " << ack_string
              << "\t (rank=" << rank_ << ")";
  } else {
    LOG(INFO) << "pull key=" << (uint64_t)DecodeKey(req_data.keys[0])
              << "\t sender=" << req_meta.sender
              << "\t " << ack_string
              << "\t (rank=" << rank_ << ")";
  }
}

void BytePSServer::BytePSGDRHandler(const ps::KVMeta& req_meta,
                                 const ps::KVPairs<char> &req_data,
                                 ps::KVServer<char>* server) {
  DataHandleType type = DepairDataHandleType(req_meta.cmd);
  auto req_type = type.requestType;
  CHECK_EQ(req_type, RequestType::kGDRPushPull); 
  CHECK_EQ(req_data.keys.size(), (size_t)1);
  uint64_t key = DecodeKey(req_data.keys[0]);

  PrintServerRecvMessageLog(req_meta, req_data);

  // get expected number of phy node for each key 
  SetNumExpectedWorker(key, num_phy_node_ - 1);

  if (req_meta.push) {  // push request
    CHECK_EQ(req_data.lens.size(), (size_t)1);
    CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[0]);
    auto stored = GetStore(key);
    auto len = (size_t)req_data.lens[0];
    auto recved = reinterpret_cast<char*>(req_data.vals.data());

    // init stored and update buffer, use page aligned memory
    InitStoreAndUpdateBuf(key, len, type.dtype);
    
    std::lock_guard<std::mutex> lock(handle_mu_); 
    auto tid = GetThreadID(key, len);
    BytePSEngineMessage msg = {timestamp_++,   type,     key,
                                stored->tensor, recved,   stored->len,
                                SUM_RECV,       req_data, req_meta};
    gdr_engine_queues_[tid]->Push(std::move(msg));
    SendPushResponse(key, req_meta, server);

  } else {  // pull request
    auto tid = GetThreadID(key, 0);
    std::lock_guard<std::mutex> lock(flag_mu_[tid]);
    if (is_push_finished_[tid].find(key) == is_push_finished_[tid].end()) {
      is_push_finished_[tid][key] = false;
      pull_cnt_[tid][key] = 0;
      seen_sender_[tid][key].clear();
    }

    auto it = seen_sender_[tid][key].find(req_meta.sender);
    if (is_push_finished_[tid][key] && (it == seen_sender_[tid][key].end())) {
      // push already finished && not received the associated pull response yet
      SendPullResponse(type, key, req_meta, server);
      pull_cnt_[tid][key] += 1;
      seen_sender_[tid][key].insert(req_meta.sender);
      if (pull_cnt_[tid][key] == GetNumExpectedWorker(key)) {
        is_push_finished_[tid][key] = false;
        pull_cnt_[tid][key] = 0;
        seen_sender_[tid][key].clear();
      }
    } else {
      // push not finished, put into the queue, and wait for the engine
      q_pull_reqmeta_[tid][key].push_back(req_meta);
    }
  } // end of pull request 
}

void BytePSServer::ThreadSafeInitCudaBuffer(uint64_t key, size_t len) {
#if HAVE_CUDA == 1
  std::lock_guard<std::mutex> lk(handle_mu_);
  auto& merged = GetUpdateBuf(key)->merged;
  auto stored = GetStore(key);
  if (!merged.tensor) {
    CUDA_CALL(cudaMalloc(&(merged.tensor), len));
    CUDA_CALL(cudaMalloc(&(stored->tensor), len));
    merged.len = len;
    merged.device = common::GPU;
    stored->len = len;
    stored->device = common::GPU;
  }
  CHECK_EQ(len, merged.len) << len << ", " << merged.len;
#endif
}

void BytePSServer::BytePSGDRv2Handler(const ps::KVMeta& req_meta,
                                      const ps::KVPairs<char> &req_data,
                                      ps::KVServer<char>* server) {
  PrintServerRecvMessageLog(req_meta, req_data);
  DataHandleType type = DepairDataHandleType(req_meta.cmd);
  auto req_type = type.requestType;
  CHECK_EQ(req_data.keys.size(), (size_t)1);
  uint64_t key = DecodeKey(req_data.keys[0]);
  
  if (req_type == RequestType::kGDRv2PushPullSmall) {
    SmallTensorMngr::Register(key);
  } else {
    SetNumExpectedWorker(key, num_phy_node_);
  }

  if (req_meta.push) {  // push request
    CHECK_EQ(req_data.lens.size(), (size_t)1);
    CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[0]);
    size_t len = (size_t) req_data.lens[0];
    ThreadSafeInitCudaBuffer(key, len);

    int worker_rank = (req_meta.sender - 9) / 2;
    auto pbuff = GetGDRPushBuffer(key, worker_rank);
    // initialization
    if (!pbuff->tensor) {
      CHECK_EQ(type.device, common::GPU);
      ps::SArray<ps::Key> keys;
      ps::SArray<char> vals;
      ps::SArray<int> lens;
      CUDA_CALL(cudaMalloc(&(pbuff->tensor), len));
      keys.push_back(req_data.keys[0]);
      vals.reset((char*) pbuff->tensor, len * sizeof(char), [](void *){});
      lens.push_back(len);
      pbuff->device = common::GPU;
      // perform registration
      server->RegisterRecvBufferWithRank(worker_rank, keys, vals, lens);
      if (log_key_info_) {
        LOG(INFO) << "init cuda buffer for key=" << key 
                  << ", len=" << len << ", worker_rank=" << worker_rank
                  <<  "(rank=" << rank_ << ")";
      }
      // the max length
      pbuff->len = len;
      pbuff->dtype = type.dtype;
      CHECK(pbuff->tensor);
      SendPushResponse(key, req_meta, server);
      return;
    } 
    
    std::lock_guard<std::mutex> lock(handle_mu_); 
    auto tid = GetThreadID(key, len);
    void* addr = (void*)req_data.vals.data();
    CHECK_EQ(addr, pbuff->tensor) << key;
    BytePSEngineMessage msg = {timestamp_++, type, key,
                               addr, addr, pbuff->len,
                               GPU_SUM_RECV, req_data, req_meta};
    gdr_engine_queues_[tid]->Push(std::move(msg));
    SendPushResponse(key, req_meta, server);

  } else {  // pull request
    auto tid = GetThreadID(key, 0);
    std::lock_guard<std::mutex> lock(flag_mu_[tid]);
    if (is_push_finished_[tid].find(key) == is_push_finished_[tid].end()) {
      is_push_finished_[tid][key] = false;
      pull_cnt_[tid][key] = 0;
      seen_sender_[tid][key].clear();
    }
    auto it = seen_sender_[tid][key].find(req_meta.sender);
    if (is_push_finished_[tid][key] && (it == seen_sender_[tid][key].end())) {
      // push already finished && not received the associated pull response yet
      if (log_key_info_) {
        LOG(INFO) << "pull request of key " << key 
                  << " received and processed immediately";
      }
      SendGDRPullResponse(type, key, req_meta, server);
      pull_cnt_[tid][key] += 1;
      seen_sender_[tid][key].insert(req_meta.sender);
      // For small tensor, we receive from all GPUs
      size_t expected = SmallTensorMngr::IsRegistered(key) 
                      ? BytePSGlobal::GetSize() : (num_phy_node_ - 1);
      if (pull_cnt_[tid][key] == expected) {
        is_push_finished_[tid][key] = false;
        pull_cnt_[tid][key] = 0;
        seen_sender_[tid][key].clear();
      }
    } else {
      // push not finished, put into the queue, and wait for the engine
      if (log_key_info_) {
        LOG(INFO) << "pull request of key " << key 
                  << " buffered because the push flag is not ready";
      }
      q_pull_reqmeta_[tid][key].push_back(req_meta);
    }
  } // end of pull request 
}

void BytePSServer::LocalPushPull(uint64_t key, char* push_addr, char* pull_addr, size_t len, int dtype) {
  InitStoreAndUpdateBuf(key, len, dtype);
  BytePSEngineMessage msg;
  msg.id = timestamp_++;
  msg.key = key;
  msg.dst = gdr_copy_mngr_->GetBuffer(key, len);
  msg.src = push_addr;
  msg.len = len;
  msg.ops = COPY_D2H;
  gdr_copy_mngr_->StorePullAddr(key, pull_addr);
  independent_queues_[independent_cnt_++ % engine_thread_num_]->Push(std::move(msg));
}

void BytePSServer::EnqueueLocalGpuSumTask(uint64_t key, char* input, 
                                          char* output, size_t len, 
                                          int dtype, bool do_copy) {
#if HAVE_CUDA == 1
  auto tid = GetThreadID(key, len);
  auto cuda_reducer = BytePSGlobal::GetCudaReducer(tid);
  if (do_copy && input != output) {
    // TODO: remove this copy 
    cuda_reducer->CopyD2D(output, input, len, /*sync*/false);
    cuda_reducer->Sync();
  } 
  ThreadSafeInitCudaBuffer(key, len);
  BytePSEngineMessage msg;
  msg.id = timestamp_++;
  msg.key = key;
  msg.dst = output;
  msg.src = output;
  msg.len = len;
  msg.ops = GPU_SUM_RECV_LOCAL;
  msg.type.dtype = dtype;
  gdr_engine_queues_[tid]->Push(std::move(msg));
#endif
}

size_t BytePSServer::GetNumExpectedWorker(uint64_t key) {
  std::lock_guard<std::mutex> lk(expected_worker_mu_);
  if (BytePSGlobal::IsGDR()) {
    size_t num_expected = num_phy_node_; 
    if (!BytePSGlobal::IsGDRGpu2Gpu()) num_expected--;
    if (num_expected_workers_.find(key) == num_expected_workers_.end()) {
      num_expected_workers_[key] = num_expected;
    } 
    // for small tensors, it is guaranteed that SetNumExpectedWorker has already been called
  } else {
    CHECK_NE(num_expected_workers_.find(key), num_expected_workers_.end()) << key;
  }
  return num_expected_workers_[key];
}

void BytePSServer::SetNumExpectedWorker(uint64_t key, size_t num) {
  std::lock_guard<std::mutex> lk(expected_worker_mu_);
  if (num_expected_workers_.find(key) == num_expected_workers_.end()) {
    num_expected_workers_[key] = num;
  } else {
    CHECK_EQ(num_expected_workers_[key], num);
  }
}

void BytePSServer::InitP2PCopyTable() {
  p2p_copy_table_ = new ReadyTable(1, "RECV");
  p2p_pull_response_table_ = new ReadyTable(1, "P2P_PULL_RESPONSE");
  p2p_ack_table_ = new ReadyTable(1, "P2P_WAIT_ACK");
  auto num_worker_str = getenv("DMLC_NUM_WORKER");
  CHECK(num_worker_str);
  int num_workers = atoi(num_worker_str);
  p2p_group_copy_table_ = new ReadyTable(num_workers - 1, "P2P_GROUP_COPYH2D");
}

void BytePSServer::InitGDRReadyTable() {
  int wait_count;
  if (BytePSGlobal::IsGDRGpu2Gpu()) {
    wait_count = BytePSGlobal::GetPhyNodeNum();
  } else {
    wait_count = 1;
  }
  gdr_push_pull_table_ = new ReadyTable(wait_count, "GDR_PUSH_PULL");
}

void BytePSServer::InitAllgatherTable() {
  // 1 + 1, the first 1 indicates allgather is ready, the second 1 indicates the pull request has already come
  allgather_pull_resp_table_ = new ReadyTable(2, "ALLGATHER_PULL_RESP");
  allgather_pull_ack_table_ = new ReadyTable(1, "ALLGATHER_PULL_ACK");
  allgather_pull_worker_local_root_resp_table_ = new ReadyTable(1, "ALLGATHER_PULL_WORKER_LOCAL_ROOT_RESP");
  allgather_pull_worker_local_root_ack_table_ = new ReadyTable(1, "ALLGATHER_PULL_WORKER_LOCAL_ROOT_ACK");
}

void BytePSServer::InitEnv() {
  // enable to print key profile
  log_key_info_ = GetEnv("BYTEPS_PS_KEY_LOG", false);
  std::string role_str = GetEnv("DMLC_ROLE", "unk");
  // Just compare with "scheduler" here, the env var may be "joint", "server", "scheduler"
  role_ = ps::GetRole(role_str);
  if (role_str != std::string("scheduler")) {
    is_server_ = true;
  }

  // Group size
  auto ps_instance_size_str = getenv("DMLC_GROUP_SIZE");
  ps_instance_size_ = ps_instance_size_str ? atoi(ps_instance_size_str) : 1;

  // enable engine block mode (default: non-blocking)
  is_engine_blocking_ = GetEnv("BYTEPS_SERVER_ENGINE_BLOCKING", false);
  if (is_engine_blocking_)
    LOG(INFO) << "Enable blocking mode of the server engine";

  // sync or async training
  sync_mode_ = !GetEnv("BYTEPS_ENABLE_ASYNC", false);
  if (!sync_mode_)
    LOG(INFO) << "BytePS server is enabled asynchronous training";

  // debug mode
  auto debug_mode_str = getenv("BYTEPS_SERVER_DEBUG");
  debug_mode_ = debug_mode_str ? atoi(debug_mode_str) : false;
  debug_key_ = GetEnv("BYTEPS_SERVER_DEBUG_KEY", 0);
  if (debug_mode_) LOG(INFO) << "Debug mode enabled! Printing key " << debug_key_;

  // number of engine thread
  // invalid if is_engine_blocking = true
  engine_thread_num_ = GetEnv("BYTEPS_SERVER_ENGINE_THREAD", 8);
  CHECK_GE(engine_thread_num_, 1);

  // enable scheduling for server engine
  enable_schedule_ = GetEnv("BYTEPS_SERVER_ENABLE_SCHEDULE", false);
  if (enable_schedule_) LOG(INFO) << "Enable engine scheduling for BytePS server";

  auto direct_response_str = getenv("BYTEPS_SERVER_DIRECT_RESPONSE");
  if (direct_response_str) p2p_direct_response_ = atoi(direct_response_str);

  auto gdr_sync_str = getenv("BYTEPS_SERVER_GDR_LAZY_SYNC");
  gdr_lazy_sync_ = gdr_sync_str ? atoi(gdr_sync_str) : false;
  if (gdr_lazy_sync_) {
    LOG(INFO) << "BYTEPS_SERVER_GDR_LAZY_SYNC set to enabled.";
  }

  auto num_worker_str = getenv("DMLC_NUM_WORKER");
  CHECK(num_worker_str);
  auto local_size_str = getenv("BYTEPS_LOCAL_SIZE");
  local_size_ = atoi(local_size_str);
  CHECK_GE(local_size_, 1);
  if (role_str == "joint") {
    // In joint mode, each worker process will also contain its own server thread.
    CHECK(local_size_str);
    num_phy_node_ = atoi(num_worker_str) / local_size_;
    num_byteps_workers_ = atoi(num_worker_str);
  } else if (role_str != "scheduler") {
    // In non-joint mode, each physical node will only contain one server process shared by
    // all workers on the same node.
    auto local_size_str = getenv("BYTEPS_LOCAL_SIZE");
    CHECK(local_size_str);
    num_phy_node_ = atoi(num_worker_str);
    num_byteps_workers_ = atoi(num_worker_str) * local_size_;
  }
  if (is_server_) {
    std::string msg = " with " + std::to_string(engine_thread_num_) + " engine threads.";
    if (engine_thread_num_ < 4) {
      msg += " Consider increasing BYTEPS_SERVER_ENGINE_THREAD for better performance";
    }
    LOG(INFO) << "Using num_phy_node=" << num_phy_node_ << " num_byteps_workers_=" <<
              num_byteps_workers_ << " direct_response = " << p2p_direct_response_ << msg;
  } else {
    LOG(INFO) << "This is a scheduler.";
  }
}

void BytePSServer::Init(int rank) {
  LOG(INFO) << "byteps server is starting. rank=" << rank;

  // cpu reducer
  bps_reducer_ = new byteps::common::CpuReducer(nullptr);

  // flag mu and its protected map
  std::vector<std::mutex> tmp_flagmu(engine_thread_num_);
  std::vector<std::unordered_map<uint64_t, bool> > tmp_ispushfinished(
      engine_thread_num_);
  std::vector<std::mutex> tmp_copymu(engine_thread_num_);
  std::vector<std::unordered_map<uint64_t, std::vector<ps::KVMeta> > >
      tmp_qpullreqmeta(engine_thread_num_);
  std::vector<std::unordered_map<uint64_t, std::set<int> > > tmp_seensender(
      engine_thread_num_);
  std::vector<std::unordered_map<uint64_t, size_t> > tmp_pullcnt(
      engine_thread_num_);
  flag_mu_.swap(tmp_flagmu);
  is_push_finished_.swap(tmp_ispushfinished);
  q_pull_reqmeta_.swap(tmp_qpullreqmeta);
  seen_sender_.swap(tmp_seensender);
  pull_cnt_.swap(tmp_pullcnt);
  CHECK_EQ(flag_mu_.size(), engine_thread_num_);
  CHECK_EQ(is_push_finished_.size(), engine_thread_num_);
  CHECK_EQ(q_pull_reqmeta_.size(), engine_thread_num_);
  CHECK_EQ(pull_cnt_.size(), engine_thread_num_);

  // init the engine
  for (size_t i = 0; i < engine_thread_num_; ++i) {
    acc_load_.push_back(0);
  }
  if (sync_mode_) {
    // these engine threads are not used by p2p
    for (size_t i = 0; i < engine_thread_num_; ++i) {
      auto q = new PriorityQueue(enable_schedule_);
      engine_queues_.push_back(q);
    }
    for (size_t i = 0; i < engine_thread_num_; ++i) {
      auto t = new std::thread(&BytePSServerEngineThread, i);
      engine_threads_.push_back(t);
    }
    if (BytePSGlobal::IsGDR()) {
      CUDA_CALL(cudaSetDevice(BytePSGlobal::GetVisibleDevice()));
      for (size_t i = 0; i < engine_thread_num_; ++i) {
        auto q = new PriorityQueue(enable_schedule_);
        gdr_engine_queues_.push_back(q);
      }
      for (size_t i = 0; i < engine_thread_num_; ++i) {
        auto q = new PriorityQueue(enable_schedule_);
        independent_queues_.push_back(q);
      }
      for (size_t i = 0; i < engine_thread_num_; ++i) {
        auto t = new std::thread(&BytePSServerGDRIndependentThread, i);
        independent_threads_.push_back(t);
      }
      for (size_t i = 0; i < engine_thread_num_; ++i) {
        auto t = new std::thread(&BytePSServerGDREngineThread, i);
        gdr_engine_threads_.push_back(t);
      }
    }
  }
  rank_ = rank;
  auto role = is_server_ ? ps::Node::SERVER : ps::Node::SCHEDULER;
  if (!is_server_) {
    rank_ = 0;
  }
  // When set to joint, the worker has already started the PS
  if (role_ != ps::Node::JOINT) {
    ps::StartPS(0, role, rank_, false, "byteps\0");
  }

  if (is_server_ && role_ == ps::Node::JOINT) {
    // GDR copy manager
    gdr_copy_mngr_ = new GDRCopyManager();
  } 

  // init server instance
  if (is_server_) {
    for (int instance_idx = 0; instance_idx < ps_instance_size_; instance_idx++) {
      auto server = new KVServer<char>(0, false, instance_idx);
      server->set_request_handle(BytePSHandler);
      byteps_server_.push_back(server);
    }
  }
  int barrier_group = ps::kScheduler + ps::kWorkerGroup + ps::kServerGroup;
  ps::Postoffice::GetServer()->Barrier(0, barrier_group);
  // error handling
  if (BytePSGlobal::EnableErrHandling()) {
    ps::Postoffice::GetServer()->van()->set_err_handle(common::BytePSError::ErrHandle);
  }

  // clean the server resource
  ps::Finalize(0, role, true);
  should_stop_ = true;
  for (auto server : byteps_server_) {
    delete server;
  }
  byteps_server_.clear();
  if (bps_reducer_) {
    delete bps_reducer_;
    bps_reducer_ = nullptr;
  }
  BytePSEngineMessage msg;
  msg.ops = TERMINATE;
  for (auto q : engine_queues_) q->Push(msg);
  for (auto q : gdr_engine_queues_) q->Push(msg);
  for (auto t : engine_threads_) t->join();
  for (auto t : gdr_engine_threads_) t->join();
  for (auto q : independent_queues_) q->Push(msg);
  for (auto t : independent_threads_) t->join();

  for (auto& it : store_) {
    if (it.second.tensor) {
      if (it.second.device == common::CPU) {
        // only free the tensors managed by us
        if (it.second.managed) free(it.second.tensor);
      } else {
        CUDA_CALL(cudaFree(it.second.tensor));
      }
    }
  }

  for (auto& it1 : gdr_push_buffer_) {
    for (auto& it2 : it1.second) {
      if (it2.second.tensor) {
        CUDA_CALL(cudaFree(it2.second.tensor));
      }
    }
  }

#if HAVE_CUDA == 1
  for (auto &it : update_buf_) {
    char* buf = it.second.merged.tensor;
    if (buf && it.second.merged.registered) {
      CUDA_CALL(cudaHostUnregister(buf));
      free(buf);
    }
    if (buf && it.second.merged.device == common::GPU) {
      CUDA_CALL(cudaFree(buf));
    }
  }
#endif

  if (gdr_push_pull_table_) {
    delete gdr_push_pull_table_;
    gdr_push_pull_table_ = nullptr;
  }

  if (gdr_copy_mngr_) {
    delete gdr_copy_mngr_;
    gdr_copy_mngr_ = nullptr;
  }

  LOG(INFO) << "BytePS Server has been shutdown";
  return;
}

extern "C" void byteps_server() {
  int rank = getenv("DMLC_RANK") ? atoi(getenv("DMLC_RANK")) : -1;
  BytePSServer::InitEnv();
  BytePSServer::Init(rank);
}

}  // namespace server
}  // namespace byteps
