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

template <typename Val>
void BytepsSparseHandler(const ps::KVMeta &req_meta, 
                         const ps::KVPairs<Val> &req_data, 
                         ps::KVServer<Val> *server) {
  uint64_t key = DecodeKey(req_data.keys[0]);

  if (debug_) {
    LOG(INFO) << "receive " << (req_meta.push ? "push" : "pull")
              << "\t key=" << key 
              << "\t len=" << req_meta.val_len
              << "\t sender=" << req_meta.sender;
  }

  if ((key & 0xffff) == 0xffff) { 
    // scatter or gather, see key encode logic in dist_comm.h
    if (req_meta.push) { 
      // scatter request

    } else { 
      // gather request
      int gpu = (key >> 16) % local_size_;
      int len = req_meta.val_len;
      if (map_.find(key) == map_.end()) {
        AllocMemoryAndCreateSarray(map_[key].keys, 1, (ps::Key*)&req_data.keys[0]);
        AllocMemoryAndCreateSarray(map_[key].vals, len);
        AllocMemoryAndCreateSarray(map_[key].lens, 1, (int*)&len);
      }
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
  if (byteps_server_) {
    delete byteps_server_;
    byteps_server_ = nullptr;
  }
}


} // namespace sparse
} // namespace byteps