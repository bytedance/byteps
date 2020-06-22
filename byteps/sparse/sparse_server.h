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

#ifndef BYTEPS_SPARSE_SERVER_H
#define BYTEPS_SPARSE_SERVER_H

#include <cuda_runtime.h>
#include <cstdlib>
#include <unistd.h>
#include "ps/ps.h"
#include "ps/internal/threadsafe_queue.h"
#include "util.h"

namespace byteps {
namespace sparse {

extern "C" void bytepsSparseServer();

enum MessageType {
  GATHER, SCATTER, REDUCE, TERMINATE
};

struct BytePSSparseEngineMessage {
  void* dst;
  void* src;
  size_t len;
  MessageType type;
  ps::KVPairs<char> kvpairs; 
  ps::KVMeta req_meta;
  cudaStream_t* stream;
};

static bool debug_ = false;
static ps::KVServer<char>* byteps_server_;
static std::unordered_map<uint64_t, ps::KVPairs<char>> init_map_;
static std::unordered_map<uint64_t, ps::KVPairs<char>> gather_map_;
static int local_size_; // local gpu number
static bool is_inited_ = false;

static std::vector<cudaStream_t> streams_d2h_;
static std::vector<cudaStream_t> streams_h2d_;
static std::vector<cudaIpcMemHandle_t*> embed_ipc_handlers_;
static std::vector<void*> embed_bufs_;
static std::vector<size_t> embed_buflens_;
static size_t dense_buflen_;

using TsQueue = ps::ThreadsafeQueue<BytePSSparseEngineMessage>;
static size_t engine_nthreads_;
static std::vector<TsQueue*> engine_queues_;
static std::vector<std::thread *> threads_;
static uint64_t request_id_ = 0;

uint64_t DecodeKey(ps::Key key) {
  auto kr = ps::Postoffice::Get()->GetServerKeyRanges()[ps::MyRank()];
  return key - kr.begin();
}

template <typename T>
void AllocMemoryAndCreateSarray(ps::SArray<T>& sarr, int count, T* addr = nullptr) {
  void* ptr;
  mallocAligned(&ptr, count * sizeof(T));
  sarr.reset((T*)ptr, count, [](void *){});
  if (addr != nullptr) {
    memcpy(ptr, (void*)addr, count * sizeof(T));
  }
}

} // namespace sparse
} // namespace byteps

#endif  // BYTEPS_SPARSE_SERVER_H
