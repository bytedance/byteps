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

// definition
// knobs
uint64_t BytePSServer::timestamp_ = 0;
size_t BytePSServer::engine_thread_num_ = 4;
volatile bool BytePSServer::is_engine_blocking_ = false;
volatile bool BytePSServer::log_key_info_ = false;
volatile bool BytePSServer::sync_mode_ = true;
volatile bool BytePSServer::debug_mode_ = false;
volatile bool BytePSServer::enable_schedule_ = false;
volatile bool BytePSServer::is_server_ = false;
// p2p knobs
std::string BytePSServer::p2p_shm_prefix_;
// received tensors
std::unordered_map<uint64_t, RecvArray> BytePSServer::recved_partitions_;
std::mutex BytePSServer::recved_partitions_mu_;

size_t BytePSServer::num_phy_node_;
size_t BytePSServer::num_expected_workers_;
size_t BytePSServer::num_byteps_workers_;

// debug
uint64_t BytePSServer::debug_key_;
std::mutex BytePSServer::debug_mu_;
// server operations
byteps::common::CpuReducer* BytePSServer::bps_reducer_ = nullptr;

// ========= for p2p ==========
volatile bool BytePSServer::should_stop_ = false;
int BytePSServer::preferred_rank_ = -1;
bool BytePSServer::enable_preferred_rank_ = false;
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

// byteps handler
std::mutex BytePSServer::handle_mu_;
std::unordered_map<uint64_t, UpdateBuf> BytePSServer::update_buf_;

// address map
std::mutex BytePSServer::store_mu_;
std::unordered_map<uint64_t, BytePSArray> BytePSServer::store_;

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

// compression
std::unordered_map<uint64_t, std::unique_ptr<common::compressor::Compressor>> BytePSServer::compressor_map_;

int DivUp(int x, int y) { return (x + y - 1) / y; }
int RoundUp(int x, int y) { return DivUp(x, y) * y; }

// TODO: remove Postoffice API calls
uint64_t DecodeKey(ps::Key key) {
  auto kr = ps::Postoffice::Get()->GetServerKeyRanges()[ps::MyRank()];
  return key - kr.begin();
}

uint64_t EncodeKey(ps::Key key) {
  auto kr = ps::Postoffice::Get()->GetServerKeyRanges()[ps::MyRank()];
  return key + kr.begin();
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
  CHECK_LT(min_index, engine_thread_num_);
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

void PageAlignedMalloc(void** ptr, size_t size) {
  size_t page_size = sysconf(_SC_PAGESIZE);
  void* p;
  int size_aligned = RoundUp(size, page_size);
  int ret = posix_memalign(&p, page_size, size_aligned);
  CHECK_EQ(ret, 0) << "posix_memalign error: " << strerror(ret);
  CHECK(p);
  memset(p, 0, size);
  *ptr = p;
}

BytePSArray* BytePSServer::GetStore(uint64_t key) {
  std::lock_guard<std::mutex> lock(store_mu_);
  return &store_[key];
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

void BytePSServer::SendPullResponse(const DataHandleType type,
                                    const uint64_t key,
                                    const ps::KVMeta& req_meta,
                                    ps::KVServer<char>* server) {
  std::lock_guard<std::mutex> lock(pullresp_mu_);
  auto& updates = update_buf_[key];
  CHECK(updates.merged.tensor) << "init " << key << " first";
  char* data = updates.merged.tensor;
  auto len = updates.merged.len;

  // send pull response
  auto iterator = pull_response_map_.find(key);
  if (iterator == pull_response_map_.end()) {  // new key
    ps::KVPairs<char> response;
    response.keys = {EncodeKey(key)};
    response.lens = {len};
    response.vals = ps::SArray<char>(data, len, false);  // zero copy
    pull_response_map_[key] = response;                  // add to the map
    server->Response(req_meta, response);
  } else {  // not new key, then reuse the memory address to avoid ibv_reg_mr on
            // RDMA data path
    ps::KVPairs<char>* response = &iterator->second;

    auto p = static_cast<char*>(data);
    CHECK(p);
    response->lens = {len};
    response->vals = ps::SArray<char>(p, len, false);
    server->Response(req_meta, *response);
  }
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
        auto& updates = update_buf_[msg.key];
        updates.merged.tensor = compressed.data;
        updates.merged.len = compressed.size;
      } else {  // decompress
        auto compressed_len = msg.sarray.lens[0];
        CHECK_LE(compressed_len, msg.len);
        common::compressor::tensor_t compressed(
            reinterpret_cast<char*>(msg.src), compressed_len, msg.type.dtype);
        auto decompressed = iter->second->Decompress(compressed);
        msg.src = decompressed.data;
      }
    } else {
      if (msg.ops == ALL_RECV) {
        // 2. no compress
        auto& updates = update_buf_[msg.key];
        updates.merged.tensor = reinterpret_cast<char*>(msg.src);
        updates.merged.len = msg.len;
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
          CHECK_GE(bps_reducer_->div(msg.dst, msg.len, bps_type,
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
          if (pull_cnt_[i][msg.key] == num_expected_workers_) {
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
        || req_type == RequestType::kAckSignal);
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
  if (req_type == RequestType::kAckSignal) {
    GetP2PAckTable()->AddReadyCount(key);
    return;
  }
  // not ack message
  size_t len;
  if (req_meta.push) {
    len = (size_t) req_data.lens[0];
  } else { // pull 
    CHECK_EQ(req_type, RequestType::kDefaultPull);
    {
      std::lock_guard<std::mutex> lk(pullresp_mu_);
      p2p_pull_reqmetas_[key] = req_meta;
    }
    GetP2PPullResponseTable()->AddReadyCount(key);
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
      // init stored buffer, use page aligned memory
      std::string prefix = p2p_shm_prefix_ + std::to_string(preferred_rank_);
      std::string shm_name = prefix + "_" + std::to_string(req_data.keys[0]);
      PageAlignedMalloc((void**)&stored->tensor, len);
    }
    if (log_key_info_) {
      LOG(INFO) << "stored tensor for key=" << key << "\tcreated, rank="
                << preferred_rank_ << " len=" << len << " device=" << stored->device;
    }
    // TODO: actually there's no need to store this tensor
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
    uint64_t tensor_key = (key << 32) >> 48;
    // set recv len
    arr.len = is_empty ? 0 : len;
    if (debug_mode_) {
      LOG(INFO) << "GROUP SEND: set recv val addr = " << (long long) req_data.vals.data()
                << ", tensor_id = " << tensor_key << ", key = " << key;
    }
    SetRecvPartition(key, arr);
    GetP2PGroupCopyTable()->AddReadyCount(tensor_key);
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
      || req_type == RequestType::kAckSignal) {
    P2PHandler(req_meta, req_data, server);
    return;
  }
  if (type.requestType == RequestType::kLeaderPushPull
      || type.requestType == RequestType::kLeaderPushPullAvg) {
    num_expected_workers_ = num_phy_node_;
  } else {
    num_expected_workers_ = (size_t) ps::NumWorkers();
  }
  // CHECK_EQ(type.requestType, RequestType::kDefaultPushPull);
  // do some check
  CHECK_EQ(req_data.keys.size(), (size_t)1);
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
  uint64_t key = DecodeKey(req_data.keys[0]);

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
    auto& updates = update_buf_[key];
    updates.request.push_back(req_meta);
    // should send response after collecting all init push
    if (updates.request.size() < num_expected_workers_) return;

    for (const auto& req : updates.request) {
      SendPushResponse(key, req, server);
    }
    updates.request.clear();
    return;
  }

  if (req_meta.push) {  // push request
    CHECK_EQ(req_data.lens.size(), (size_t)1);
    CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[0]);
    auto stored = GetStore(key);
    auto len = (size_t)req_data.lens[0];
    auto recved = reinterpret_cast<char*>(req_data.vals.data());

    if (!stored->tensor) {
      if (sync_mode_ && (update_buf_.find(key) == update_buf_.end())) {
        update_buf_[key].merged.len = len;
        update_buf_[key].merged.dtype = type.dtype;
      }
      // buffer the request meta
      auto& updates = update_buf_[key];
      updates.request.push_back(req_meta);
      // should send response after collecting all init push
      if (updates.request.size() < num_expected_workers_) return;
      if (log_key_info_) {
        LOG(INFO) << "Collected all " << updates.request.size()
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

      bps_reducer_->copy(stored->tensor, recved,
                         len);  // we may not need this copy
      for (const auto& req : updates.request) {
        SendPushResponse(key, req, server);
      }
      updates.request.clear();
    } else {
      auto& updates = update_buf_[key];
      auto tid = GetThreadID(key, len);
      if (updates.request.empty()) {  // from the first incoming worker
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
          updates.merged.tmp_sarray = req_data;
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
        // CHECK(updates.merged.tensor);
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
                       (void*)updates.merged.tensor, (void*)recved, len,
                       bps_reducer_->GetDataType(updates.merged.dtype)),
                   0);
        } else {  // non-blocking
          BytePSEngineMessage msg = {timestamp_++,   type,     key,
                                     stored->tensor, recved,   stored->len,
                                     SUM_RECV,       req_data, req_meta};
          engine_queues_[tid]->Push(msg);
        }
      }
      // add a worker information (request.size() is the # workers received)
      updates.request.push_back(req_meta);
      SendPushResponse(key, req_meta, server);
      if (sync_mode_ && updates.request.size() == num_expected_workers_) {
        auto stored = GetStore(key);
        if (debug_mode_ && (debug_key_ == key)) {
          std::lock_guard<std::mutex> lock(debug_mu_);
          LOG(INFO) << "stage: COPY_MERGED_TO_STORE \t"
                    << "stored: " << DEBUG_PRINT_TENSOR_VALUE(stored->tensor)
                    << "\t"
                    // << "merged: "
                    // << DEBUG_PRINT_TENSOR_VALUE(updates.merged.tensor) << "\t"
                    << "recved: " << DEBUG_PRINT_TENSOR_VALUE(recved);
        }
        if (is_engine_blocking_) {
          // TODO: compress
          bps_reducer_->copy(stored->tensor, updates.merged.tensor, len);
        } else {
          BytePSEngineMessage msg = {
              timestamp_++,   type,        key,     stored->tensor,
              stored->tensor, stored->len, ALL_RECV};
          engine_queues_[tid]->Push(msg);
          engine_queues_[tid]->ClearCounter(key);
        }
        updates.request.clear();
      } else if (!sync_mode_) {
        // async: clean the request buffer
        updates.request.clear();
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

        if (pull_cnt_[tid][key] == num_expected_workers_) {
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

void BytePSServer::InitP2PCopyTable() {
  p2p_copy_table_ = new ReadyTable(1, "RECV");
  p2p_pull_response_table_ = new ReadyTable(1, "P2P_PULL_RESPONSE");
  p2p_ack_table_ = new ReadyTable(1, "P2P_WAIT_ACK");
  auto num_worker_str = getenv("DMLC_NUM_WORKER");
  CHECK(num_worker_str);
  int num_workers = atoi(num_worker_str);
  p2p_group_copy_table_ = new ReadyTable(num_workers - 1, "P2P_GROUP_COPYH2D");
  LOG(INFO) << "init p2p_group_copy_table_ with count = " << num_workers - 1;
}

void BytePSServer::init_global_env() {
  // enable to print key profile
  log_key_info_ = GetEnv("BYTEPS_PS_KEY_LOG", false);
  std::string role_str = GetEnv("DMLC_ROLE", "unk");
  // Just compare with "scheduler" here, the env var may be "joint", "server", "scheduler"
  role_ = ps::GetRole(role_str);
  if (role_str != std::string("scheduler")) {
    is_server_ = true;
  }
  LOG(INFO) << "This is a " << role_str << " is_server=" << is_server_;

  // enable engine block mode (default enabled)
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
  engine_thread_num_ = GetEnv("BYTEPS_SERVER_ENGINE_THREAD", 4);
  LOG(INFO) << "BytePS server engine uses " << engine_thread_num_ << " threads"
            << ", consider increasing BYTEPS_SERVER_ENGINE_THREAD for higher "
               "performance";
  CHECK_GE(engine_thread_num_, 1);

  // enable scheduling for server engine
  enable_schedule_ = GetEnv("BYTEPS_SERVER_ENABLE_SCHEDULE", false);
  if (enable_schedule_) LOG(INFO) << "Enable engine scheduling for BytePS server";

  p2p_shm_prefix_ = GetEnv("BYTEPS_P2P_SHM_PREFIX", "BytePS_P2P_ShM_");
  LOG(INFO) << "Using p2p shm prefix " << p2p_shm_prefix_;

  auto direct_response_str = getenv("BYTEPS_SERVER_DIRECT_RESPONSE");
  if (direct_response_str) p2p_direct_response_ = atoi(direct_response_str); 
  LOG(INFO) << "BytePS server direct response = " << p2p_direct_response_;

  auto num_worker_str = getenv("DMLC_NUM_WORKER");
  CHECK(num_worker_str);
  if (role_str == "joint") {
    auto local_size_str = getenv("BYTEPS_LOCAL_SIZE");
    CHECK(local_size_str);
    num_phy_node_ = atoi(num_worker_str) / atoi(local_size_str);
    num_byteps_workers_ = atoi(num_worker_str);
  } else if (role_str != "scheduler") {
    auto local_size_str = getenv("BYTEPS_LOCAL_SIZE");
    CHECK(local_size_str);
    num_phy_node_ = atoi(num_worker_str);
    num_byteps_workers_ = atoi(num_worker_str) * atoi(local_size_str);
  }
  LOG(INFO) << "Using num_phy_node " << num_phy_node_;
  LOG(INFO) << "Using num_byteps_workers_ " << num_byteps_workers_;
}


void BytePSServer::Init(int rank) {
  LOG(INFO) << "byteps server is starting";
  BytePSServer::init_global_env();

  // cpu reducer
  bps_reducer_ = new byteps::common::CpuReducer(nullptr);

  // flag mu and its protected map
  std::vector<std::mutex> tmp_flagmu(engine_thread_num_);
  std::vector<std::unordered_map<uint64_t, bool> > tmp_ispushfinished(
      engine_thread_num_);
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
  }
  preferred_rank_ = rank;
  auto role = is_server_ ? ps::Node::SERVER : ps::Node::SCHEDULER;
  if (!is_server_) {
    preferred_rank_ = 0;
  }
  // When set to joint, the worker has already started the PS
  if (role_ != ps::Node::JOINT) {
    ps::StartPS(0, role, preferred_rank_, true, "byteps\0");
  }

  // init server instance
  if (is_server_) {
    auto val = getenv("DMLC_GROUP_SIZE");
    int group_size = val ? atoi(val) : 1;
    for (int instance_idx = 0; instance_idx < group_size; instance_idx++) {
      auto server = new KVServer<char>(0, false, instance_idx);
      server->set_request_handle(BytePSHandler);
      byteps_server_.push_back(server);
    }
    LOG(INFO) << "BytePS Server with rank=" << preferred_rank_
              << ". group_size=" << group_size;
  }

  // clean the server resource
  LOG(INFO) << "Waiting for all ranks to finalize.";
  Finalize(0, role, true);
  LOG(INFO) << "BytePS server is stopping ...";
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
  for (auto t : engine_threads_) t->join();

  for (auto& it : store_) {
    if (it.second.tensor) {
      if (it.second.device == common::CPU) {
        free(it.second.tensor);
      } else {
        CUDA_CALL(cudaFree(it.second.tensor));
      }
    }
  }

  LOG(INFO) << "byteps has been shutdown";
  return;
}

extern "C" void byteps_server() {
  int preferred_rank =
    getenv("DMLC_RANK") ?  atoi(getenv("DMLC_RANK")) : -1;
  LOG(INFO) << "BytePS Server with preferred_rank=" << preferred_rank;
  BytePSServer::Init(preferred_rank);
}

}  // namespace server
}  // namespace byteps
