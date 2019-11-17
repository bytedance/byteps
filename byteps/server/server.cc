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

#include "server.h"

namespace byteps {
namespace server {

using namespace ps;

void SendPushResponse(uint64_t key, const ps::KVMeta& req, ps::KVServer<char>* server){
  auto iterator = push_response_map_.find(key);
  if (iterator == push_response_map_.end()) { // new key
    ps::KVPairs<char> response;
    response.keys.push_back(key);
    push_response_map_[key] = response; // add to the map
    server->Response(req, response);
  } else { // not new key, then reuse the memory address to avoid ibv_reg_mr on RDMA data path
    ps::KVPairs<char> *response = &iterator->second;
    response->keys[0] = key;
    server->Response(req, *response);
  }
}

void SendPullResponse(const DataHandleType type,
                      const uint64_t key,
                      const ps::KVMeta& req_meta,
                      ps::KVServer<char>* server) {
  auto& stored = store_[key];
  CHECK(stored.tensor) << "init " << key << " first";
  // as server returns when store_realt is ready in this case
  auto len = stored.len;
  // send pull response
  auto iterator = pull_response_map_.find(key);
  if (iterator == pull_response_map_.end()) { // new key
    ps::KVPairs<char> response;
    response.keys = {EncodeKey(key)};
    response.lens = {len};
    response.vals = ps::SArray<char>(stored.tensor, len, false); // zero copy
    pull_response_map_[key] = response; // add to the map
    server->Response(req_meta, response);
  } else { // not new key, then reuse the memory address to avoid ibv_reg_mr on RDMA data path
    ps::KVPairs<char> *response = &iterator->second;
    // keys and lens remain unchanged, just update vals
    auto p = static_cast<char*>(stored.tensor);
    CHECK(p);
    response->vals = ps::SArray<char>(p, len, false); 
    server->Response(req_meta, *response);
  }
}

void BytePSServerEngineThread(int i) {
  auto& q = engine_queues_[i];
  while (true) {
    BytePSEngineMessage msg;
    q->WaitAndPop(&msg);
    if (msg.ops == TERMINATE) break;
    // do some check
    CHECK(msg.dst);
    CHECK(msg.src);

    bool is_debug = (debug_mode_ && (debug_key_ == msg.key));
    switch (msg.ops) {
      case COPY_RECV: {
        if (is_debug) {
          std::lock_guard<std::mutex> lock(debug_mu_);
          LOG(INFO) << "stage: ENGINE_COPY_RECV_BEFORE \t" 
                    << "dst: " << DEBUG_PRINT_TENSOR_VALUE(msg.dst) << "\t"
                    << "src: " << DEBUG_PRINT_TENSOR_VALUE(msg.src) << "\t"
                    << "dst_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.dst) << "\t"
                    << "src_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.src) << "\t";
        }
        memcpy(msg.dst, msg.src, msg.len);
        if (is_debug) {
          std::lock_guard<std::mutex> lock(debug_mu_);
          LOG(INFO) << "stage: ENGINE_COPY_RECV_AFTER \t" 
                    << "dst: " << DEBUG_PRINT_TENSOR_VALUE(msg.dst) << "\t"
                    << "src: " << DEBUG_PRINT_TENSOR_VALUE(msg.src) << "\t"
                    << "dst_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.dst) << "\t"
                    << "src_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.src) << "\t";
        }
        std::lock_guard<std::mutex> lock(recvmap_mu_);
        CHECK_NE(recv_map_.find(msg.src), recv_map_.end());
        recv_map_.erase(msg.src); // release the SArray
        break;
      }
      case COPY_MERGED: {
        if (is_debug) {
          std::lock_guard<std::mutex> lock(debug_mu_);
          LOG(INFO) << "stage: ENGINE_COPY_MERGED_TO_STORE_BEFORE \t" 
                    << "dst: " << DEBUG_PRINT_TENSOR_VALUE(msg.dst) << "\t"
                    << "src: " << DEBUG_PRINT_TENSOR_VALUE(msg.src) << "\t"
                    << "dst_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.dst) << "\t"
                    << "src_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.src) << "\t";
        }
        memcpy(msg.dst, msg.src, msg.len);
        if (is_debug) {
          std::lock_guard<std::mutex> lock(debug_mu_);
          LOG(INFO) << "stage: ENGINE_COPY_MERGED_TO_STORE_AFTER \t" 
                    << "dst: " << DEBUG_PRINT_TENSOR_VALUE(msg.dst) << "\t"
                    << "src: " << DEBUG_PRINT_TENSOR_VALUE(msg.src) << "\t"
                    << "dst_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.dst) << "\t"
                    << "src_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.src) << "\t";
        }
        std::lock_guard<std::mutex> lock(flag_mu_);
        if (is_push_finished_.find(msg.key) == is_push_finished_.end()) {
          is_push_finished_[msg.key] = false;
          pull_cnt_[msg.key] = 0;
        }
        is_push_finished_[msg.key] = true;
        for (auto& req_meta : q_pull_reqmeta_[msg.key]) {
          SendPullResponse(msg.type, msg.key, req_meta, byteps_server_); 
          pull_cnt_[msg.key] += 1;
          if (pull_cnt_[msg.key] == (size_t) ps::NumWorkers()) {
            is_push_finished_[msg.key] = false;
            pull_cnt_[msg.key] = 0;
          }
        }
        q_pull_reqmeta_[msg.key].clear();
        break;
      }
      case SUM_RECV: {
        auto bps_type = bps_reducer_->GetDataType(msg.type.dtype);
        if (is_debug) {
          std::lock_guard<std::mutex> lock(debug_mu_);
          LOG(INFO) << "stage: ENGINE_SUM_RECV_BEFORE \t" 
                    << "dst: " << DEBUG_PRINT_TENSOR_VALUE(msg.dst) << "\t"
                    << "src: " << DEBUG_PRINT_TENSOR_VALUE(msg.src) << "\t"
                    << "dst_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.dst) << "\t"
                    << "src_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.src) << "\t";
        }
        CHECK_GE(bps_reducer_->sum(msg.dst, 
                                  msg.src, 
                                  msg.len, 
                                  bps_type), 0);
        if (is_debug) {
          std::lock_guard<std::mutex> lock(debug_mu_);
          LOG(INFO) << "stage: ENGINE_SUM_RECV_AFTER \t" 
                    << "dst: " << DEBUG_PRINT_TENSOR_VALUE(msg.dst) << "\t"
                    << "src: " << DEBUG_PRINT_TENSOR_VALUE(msg.src) << "\t"
                    << "dst_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.dst) << "\t"
                    << "src_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.src) << "\t";
        }
        std::lock_guard<std::mutex> lock(recvmap_mu_);
        CHECK_NE(recv_map_.find(msg.src), recv_map_.end());
        recv_map_.erase(msg.src); // release the SArray
        break;
      }
      default:
        CHECK(0);
    }
  }
}

void BytePSHandler(const ps::KVMeta& req_meta,
                   const ps::KVPairs<char> &req_data, ps::KVServer<char>* server) {
  std::lock_guard<std::mutex> lock(handle_mu_); // push & pull may have racing
  DataHandleType type = DepairDataHandleType(req_meta.cmd);
  CHECK_EQ(type.requestType, RequestType::kDefaultPushPull); 
  // do some check
  CHECK_EQ(req_data.keys.size(), (size_t)1);
  if (log_key_info_) {
    if (req_meta.push) {
      CHECK_EQ(req_data.lens.size(), (size_t)1);
      CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[0]);
      LOG(INFO) << "push key=" 
                << DecodeKey(req_data.keys[0])
                << "\t sender=" << req_meta.sender
                << "\t size=" << (size_t) req_data.lens[0];
    } else {
      LOG(INFO) << "pull key=" 
                << (uint64_t) DecodeKey(req_data.keys[0])
                << "\t sender=" << req_meta.sender;
    }
  }
  uint64_t key = DecodeKey(req_data.keys[0]);
  if (req_meta.push) { // push request
    CHECK_EQ(req_data.lens.size(), (size_t)1);
    CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[0]);
    auto& stored = store_[key];
    auto len = (size_t) req_data.lens[0];
    auto recved = reinterpret_cast<char*>(req_data.vals.data());
    if (!stored.tensor) {
      // buffer the request meta
      auto &updates = update_buf_[key];
      updates.request.push_back(req_meta);
      // should send response after collecting all init push
      if (updates.request.size() < (size_t) ps::NumWorkers()) return;
      if (log_key_info_) {
        LOG(INFO) << "Collected all " << updates.request.size()
                  << " requests for key=" << key
                  << ", init the store buffer size=" << (size_t) req_data.lens[0];
      }
      // initialization
      stored.tensor = (char*) malloc(len); 
      stored.len = len;
      stored.dtype = type.dtype;
      memcpy(stored.tensor, recved, len); // we may not need this copy
      for (const auto& req : updates.request) {
        SendPushResponse(key, req, server);
      }
      updates.request.clear();
    } else {
      auto &updates = update_buf_[key];
      if (sync_mode_ && !updates.merged.tensor) {
        updates.merged.tensor = (char*) malloc(len); 
        updates.merged.len = len;
        updates.merged.dtype = type.dtype;
      }
      auto tid = GetThreadID(key);
      if (updates.request.empty()) { // from the first incoming worker
        if (sync_mode_) {
          if (is_engine_blocking_) {
            memcpy(updates.merged.tensor, recved, len);
          } else { // non-blocking
            if (debug_mode_ && (debug_key_ == key)) {
              std::lock_guard<std::mutex> lock(debug_mu_);  
              LOG(INFO) << "stage: FIRST_WORKER_COPY \t" 
                        << "stored: " << DEBUG_PRINT_TENSOR_VALUE(stored.tensor) << "\t"
                        << "merged: " << DEBUG_PRINT_TENSOR_VALUE(updates.merged.tensor) << "\t"
                        << "recved: " << DEBUG_PRINT_TENSOR_VALUE(recved) << "\t"
                        << "len: " << len << "\t"
                        << "addr: " << DEBUG_PRINT_TENSOR_ADDRESS(recved);
            }
            recvmap_mu_.lock();
            CHECK_EQ(recv_map_.find(recved), recv_map_.end());
            recv_map_[recved] = req_data.vals;
            recvmap_mu_.unlock();
            
            BytePSEngineMessage msg = {type, key, updates.merged.tensor, recved, len, COPY_RECV};
            engine_queues_[tid]->Push(msg);
          }
        } else { // async mode, directly add to the buffer
          if (is_engine_blocking_) {
            CHECK_GE(bps_reducer_->sum((void *) stored.tensor, 
                                      (void *) recved, 
                                      len, 
                                      bps_reducer_->GetDataType(stored.dtype)), 0);
          } else {
            recvmap_mu_.lock();
            CHECK_EQ(recv_map_.find(recved), recv_map_.end());
            recv_map_[recved] = req_data.vals;
            recvmap_mu_.unlock();

            BytePSEngineMessage msg = {type, key, stored.tensor, recved, len, SUM_RECV};
            engine_queues_[tid]->Push(msg);
          }
        }
      } else { // from other workers
        CHECK(sync_mode_); 
        if (is_engine_blocking_) {
          CHECK_GE(bps_reducer_->sum((void *) updates.merged.tensor, 
                                    (void *) recved, 
                                    len, 
                                    bps_reducer_->GetDataType(updates.merged.dtype)), 0);
        } else { // non-blocking
          if (debug_mode_ && (debug_key_ == key)) {
            std::lock_guard<std::mutex> lock(debug_mu_);
            LOG(INFO) << "stage: OTHER_WORKER_SUM \t" 
                      << "stored: " << DEBUG_PRINT_TENSOR_VALUE(stored.tensor) << "\t"
                      << "merged: " << DEBUG_PRINT_TENSOR_VALUE(updates.merged.tensor) << "\t"
                      << "recved: " << DEBUG_PRINT_TENSOR_VALUE(recved) << "\t"
                      << "len: " << len << "\t"
                      << "addr: " << DEBUG_PRINT_TENSOR_ADDRESS(recved);
          }
          recvmap_mu_.lock();
          CHECK_EQ(recv_map_.find(recved), recv_map_.end());
          recv_map_[recved] = req_data.vals;
          recvmap_mu_.unlock();

          BytePSEngineMessage msg = {type, key, updates.merged.tensor, recved, len, SUM_RECV, req_meta};
          engine_queues_[tid]->Push(msg);
        }
      }
      // add a worker information (request.size() is the # workers received)
      updates.request.push_back(req_meta);
      SendPushResponse(key, req_meta, server);
      if (sync_mode_ && updates.request.size() == (size_t) ps::NumWorkers()) {
        auto& stored = store_[key];
        auto& update = updates.merged;
        auto tid = GetThreadID(key);
        if (is_engine_blocking_) {
          memcpy(stored.tensor, updates.merged.tensor, len);
        } else {
          if (debug_mode_ && (debug_key_ == key)) {
            std::lock_guard<std::mutex> lock(debug_mu_);
            LOG(INFO) << "stage: COPY_MERGED_TO_STORE \t" 
                      << "stored: " << DEBUG_PRINT_TENSOR_VALUE(stored.tensor) << "\t"
                      << "merged: " << DEBUG_PRINT_TENSOR_VALUE(updates.merged.tensor) << "\t"
                      << "recved: " << DEBUG_PRINT_TENSOR_VALUE(recved);
          }
          BytePSEngineMessage msg = {type, key, stored.tensor, update.tensor, len, COPY_MERGED};
          engine_queues_[tid]->Push(msg);
        }
        updates.request.clear();
      } else if (!sync_mode_) { 
        // async: clean the request buffer immediatedly
        updates.request.clear();
      }
    }
  } else { // pull request
    auto& stored = store_[key];
    CHECK(stored.tensor) << "Processing pull request when the NDArray of key " 
               << key << " has not been inited yet, which is not expected.";
    if (is_engine_blocking_) {
      SendPullResponse(type, key, req_meta, server);
    } else {
      std::lock_guard<std::mutex> lock(flag_mu_);
      if (is_push_finished_.find(key) == is_push_finished_.end()) {
        is_push_finished_[key] = false;
        pull_cnt_[key] = 0;
      }
      if (is_push_finished_[key]) { // push already finished
        SendPullResponse(type, key, req_meta, server); 
        pull_cnt_[key] += 1;
        if (pull_cnt_[key] == (size_t) ps::NumWorkers()) {
          is_push_finished_[key] = false;
          pull_cnt_[key] = 0;
          // check: remain should be 0
          auto remain = q_pull_reqmeta_[key].size();
          CHECK_EQ(remain, 0) << remain;
        }
      } else { // push not finished, put into the queue, and wait for the engine 
        q_pull_reqmeta_[key].push_back(req_meta);
      }
    }
  }
}

void init_global_env() {
  // enable to print key profile
  log_key_info_ = GetEnv("PS_KEY_LOG", false);

  // sync or async training
  sync_mode_ = GetEnv("BYTEPS_ENABLE_ASYNC", true);
  if (!sync_mode_) LOG(INFO) << "BytePS server is enabled asynchronous training";

  // debug mode
  debug_mode_ = GetEnv("BYTEPS_SERVER_DEBUG", false);
  debug_key_ = GetEnv("BYTEPS_SERVER_DEBUG_KEY", 0);
  if (debug_mode_) LOG(INFO) << "Debug mode enabled! Printing key " << debug_key_;

  // enable engine block mode (default disabled)
  is_engine_blocking_ = GetEnv("BYTEPS_SERVER_ENGINE_BLOCKING", false);
  if (is_engine_blocking_) LOG(INFO) << "Enable blocking mode of the server engine";

  // number of engine thread
  // invalid if is_engine_blocking = true
  engine_thread_num_ = GetEnv("BYTEPS_SERVER_ENGINE_THREAD", 4);
  LOG(INFO) << "BytePS server engine uses " << engine_thread_num_ << " queues"
            << ", consider increasing BYTEPS_SERVER_ENGINE_THREAD for higher performance";
  CHECK_GE(engine_thread_num_, 1);
}

extern "C" void byteps_server() {
  init_global_env();
  bps_reducer_ = new byteps::common::CpuReducer(nullptr);
  // init the engine
  for (size_t i = 0; i < engine_thread_num_; ++i) {
    auto q = new ThreadsafeQueue<BytePSEngineMessage>();
    engine_queues_.push_back(q);
  }
  for (size_t i = 0; i < engine_thread_num_; ++i) {
    auto t = new std::thread(&BytePSServerEngineThread, i);
    engine_threads_.push_back(t);
  }

  // init server instance
  byteps_server_ = new KVServer<SERVER_DATA_TYPE>(0);
  byteps_server_->set_request_handle(BytePSHandler);
  StartAsync(0, "byteps_server\0");
  if (!Postoffice::Get()->is_recovery()) {
    Postoffice::Get()->Barrier(0,
      ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
  }
  LOG(INFO) << "byteps server runs";
  Finalize(0, true);
  if (byteps_server_) {
    delete byteps_server_;
    byteps_server_ = nullptr;
  }
  if (bps_reducer_) {
    delete bps_reducer_;
    bps_reducer_ = nullptr;
  }

  // join the threads
  BytePSEngineMessage msg;
  msg.ops = TERMINATE;
  for (auto q : engine_queues_) q->Push(msg);
  for (auto t : engine_threads_) t->join();
  for (auto& it : store_) free(it.second.tensor);
  for (auto& it : update_buf_) free(it.second.merged.tensor);
  LOG(INFO) << "byteps has been shutdown";

  return;
}

}  // namespace server
}  // namespace byteps
