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
#include "../common/cpu_reducer.h"
#include "../common/compressor/compressor.h"
#include "../common/compressor/compressor_registry.h"

namespace byteps {
namespace server {

#define SERVER_KEY_TYPE uint64_t
#define SERVER_DATA_TYPE char
#define DEBUG_PRINT_TENSOR_VALUE(X) (*((float *)(X) + 0))
#define DEBUG_PRINT_TENSOR_ADDRESS(X) (reinterpret_cast<uint64_t>(X))

using namespace ps;

enum class RequestType {
  kDefaultPushPull, kRowSparsePushPull, kCompressedPushPull
};

enum BytePSEngineOperation {
  SUM_RECV, COPY_FIRST, ALL_RECV, TERMINATE
};

struct PSKV {
  SArray<Key> keys;  // n keys
  SArray<int> lens;  // the length of the i-th value
};

struct DataHandleType {
  RequestType requestType;
  int dtype;
};

struct BytePSArray {
  char* tensor;
  size_t len;
  int dtype;
  ps::KVPairs<char> tmp_sarray;
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

static DataHandleType DepairDataHandleType(int cmd) {
  int w = std::floor((std::sqrt(8 * cmd + 1) - 1)/2);
  int t = ((w * w) + w) / 2;
  int y = cmd - t;
  int x = w - y;
  CHECK_GE(x, 0);
  CHECK_GE(y, 0);
  DataHandleType type;
  type.requestType = static_cast<RequestType>(x);
  type.dtype = y;
  return type;
}


KVServer<SERVER_DATA_TYPE>* byteps_server_;
byteps::common::CpuReducer* bps_reducer_;

std::mutex pullresp_mu_;
std::unordered_map<uint64_t, ps::KVPairs<char> > push_response_map_;
std::unordered_map<uint64_t, ps::KVPairs<char> > pull_response_map_;

// push & pull flag
std::vector<std::mutex> flag_mu_;
std::vector<std::unordered_map<uint64_t, bool> > is_push_finished_;
std::vector<std::unordered_map<uint64_t, std::vector<ps::KVMeta> > > q_pull_reqmeta_;
std::vector<std::unordered_map<uint64_t, std::set<int> > > seen_sender_;
std::vector<std::unordered_map<uint64_t, size_t> > pull_cnt_;

// byteps handler
std::mutex handle_mu_;
std::unordered_map<uint64_t, UpdateBuf> update_buf_;
std::unordered_map<uint64_t, std::unique_ptr<common::compressor::Compressor>> compressor_map_;

// address map
std::mutex store_mu_;
std::unordered_map<uint64_t, BytePSArray> store_;

// hash function
std::mutex hash_mu_;
std::unordered_map<uint64_t, size_t> hash_cache_;
std::vector<uint64_t> acc_load_; // accumulated tensor size for an engine thread

// global knob
uint64_t timestamp_ = 0;
size_t engine_thread_num_ = 4;
volatile bool is_engine_blocking_ = false;
volatile bool log_key_info_ = false;
volatile bool sync_mode_ = true;
volatile bool debug_mode_ = false;
volatile bool enable_schedule_ = false;

// debug
uint64_t debug_key_;
std::mutex debug_mu_;

int DivUp(int x, int y) { return (x + y - 1) / y; }
int RoundUp(int x, int y) { return DivUp(x, y) * y; }

uint64_t DecodeKey(ps::Key key) {
  auto kr = ps::Postoffice::Get()->GetServerKeyRanges()[ps::MyRank()];
  return key - kr.begin();
}

uint64_t EncodeKey(ps::Key key) {
  auto kr = ps::Postoffice::Get()->GetServerKeyRanges()[ps::MyRank()];
  return key + kr.begin();
}

size_t GetThreadID(uint64_t key, size_t len) {
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

extern "C" void byteps_server();

}  // namespace server
}  // namespace byteps

#endif  // BYTEPS_SERVER_H
