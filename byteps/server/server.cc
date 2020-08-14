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
#include "../common/compressor/utils.h"
#include "queue.h"

namespace byteps {
namespace server {

using namespace ps;

// engine related
std::vector<PriorityQueue*> engine_queues_;
std::vector<std::thread*> engine_threads_;

BytePSArray* GetStore(uint64_t key) {
  std::lock_guard<std::mutex> lock(store_mu_);
  return &store_[key];
}

void SendPushResponse(uint64_t key, const ps::KVMeta& req,
                      ps::KVServer<char>* server) {
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

void SendPullResponse(const DataHandleType type, const uint64_t key,
                      const ps::KVMeta& req_meta, ps::KVServer<char>* server) {
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

void BytePSServerEngineThread(int i) {
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
            SendPullResponse(msg.type, msg.key, *it, byteps_server_);
            pull_cnt_[i][msg.key] += 1;
            seen_sender_[i][msg.key].insert(it->sender);
            it = q_pull_reqmeta_[i][msg.key].erase(it);
          } else {
            ++it;
          }
          if (pull_cnt_[i][msg.key] == (size_t)ps::NumWorkers()) {
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
}  // namespace server

void BytePSHandler(const ps::KVMeta& req_meta,
                   const ps::KVPairs<char>& req_data,
                   ps::KVServer<char>* server) {
  std::lock_guard<std::mutex> lock(handle_mu_);  // push & pull may have racing
  DataHandleType type = DepairDataHandleType(req_meta.cmd);
  // CHECK_EQ(type.requestType, RequestType::kDefaultPushPull);
  // do some check
  CHECK_EQ(req_data.keys.size(), (size_t)1);
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
    if (updates.request.size() < (size_t)ps::NumWorkers()) return;

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
      if (updates.request.size() < (size_t)ps::NumWorkers()) return;
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
      if (sync_mode_ && updates.request.size() == (size_t)ps::NumWorkers()) {
        auto stored = GetStore(key);
        auto& update = updates.merged;
        if (debug_mode_ && (debug_key_ == key)) {
          std::lock_guard<std::mutex> lock(debug_mu_);
          LOG(INFO) << "stage: COPY_MERGED_TO_STORE \t"
                    << "stored: " << DEBUG_PRINT_TENSOR_VALUE(stored->tensor)
                    << "\t"
                    << "merged: "
                    << DEBUG_PRINT_TENSOR_VALUE(updates.merged.tensor) << "\t"
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

        if (pull_cnt_[tid][key] == (size_t)ps::NumWorkers()) {
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

void init_global_env() {
  // enable to print key profile
  log_key_info_ = GetEnv("PS_KEY_LOG", false);

  // enable engine block mode (default disabled)
  is_engine_blocking_ = GetEnv("BYTEPS_SERVER_ENGINE_BLOCKING", false);
  if (is_engine_blocking_)
    LOG(INFO) << "Enable blocking mode of the server engine";

  // sync or async training
  sync_mode_ = !GetEnv("BYTEPS_ENABLE_ASYNC", false);
  if (!sync_mode_)
    LOG(INFO) << "BytePS server is enabled asynchronous training";

  // debug mode
  debug_mode_ = GetEnv("BYTEPS_SERVER_DEBUG", false);
  debug_key_ = GetEnv("BYTEPS_SERVER_DEBUG_KEY", 0);
  if (debug_mode_)
    LOG(INFO) << "Debug mode enabled! Printing key " << debug_key_;

  // number of engine thread
  // invalid if is_engine_blocking = true
  engine_thread_num_ = GetEnv("BYTEPS_SERVER_ENGINE_THREAD", 4);
  LOG(INFO) << "BytePS server engine uses " << engine_thread_num_ << " threads"
            << ", consider increasing BYTEPS_SERVER_ENGINE_THREAD for higher "
               "performance";
  CHECK_GE(engine_thread_num_, 1);

  // enable scheduling for server engine
  enable_schedule_ = GetEnv("BYTEPS_SERVER_ENABLE_SCHEDULE", false);
  if (enable_schedule_)
    LOG(INFO) << "Enable engine scheduling for BytePS server";
}

extern "C" void byteps_server() {
  init_global_env();

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
    for (size_t i = 0; i < engine_thread_num_; ++i) {
      auto q = new PriorityQueue(enable_schedule_);
      engine_queues_.push_back(q);
    }
    for (size_t i = 0; i < engine_thread_num_; ++i) {
      auto t = new std::thread(&BytePSServerEngineThread, i);
      engine_threads_.push_back(t);
    }
  }

  // init server instance
  byteps_server_ = new KVServer<SERVER_DATA_TYPE>(0);
  byteps_server_->set_request_handle(BytePSHandler);
  StartAsync(0, "byteps_server\0");
  if (!Postoffice::Get()->is_recovery()) {
    Postoffice::Get()->Barrier(
        0, ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
  }

  // clean the server resource
  Finalize(0, true);
  if (byteps_server_) {
    delete byteps_server_;
    byteps_server_ = nullptr;
  }
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
      free(it.second.tensor);
    }
  }
  
  LOG(INFO) << "byteps has been shutdown";
  return;
}

}  // namespace server
}  // namespace byteps
