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

#ifndef BYTEPS_SPARSE_COMMUNICATOR_H
#define BYTEPS_SPARSE_COMMUNICATOR_H

#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>
#include <memory>
#include <string>
#include "gossip/include/gossip.cuh"
#include "gossip/include/cudahelpers/cuda_helpers.cuh"
#include "gossip/include/plan_parser.hpp"
#include "ps/ps.h"

namespace byteps {
namespace sparse {

class SparseComm {
 public:
  virtual void ExecAsync() = 0;
  virtual void Sync() = 0;
}; // class SparseComm

class LocalGatherComm : public SparseComm {
  using data_t = float;

 public:
  LocalGatherComm(const char* planfile_name, const size_t num_gpu, const std::vector<data_t*>& srcs, 
                  const std::vector<size_t>& srcs_lens, const std::vector<size_t>& send_counts,
                  data_t* dst, const size_t dst_len) : srcs_(srcs), srcs_lens_(srcs_lens),
                  send_counts_(send_counts), dst_(dst), dst_len_(dst_len) {
    auto transfer_plan = parse_plan(planfile_name);
    gossip::gather::verify_plan(transfer_plan);
    CHECK(transfer_plan.valid());
    CHECK_EQ(transfer_plan.num_gpus(), num_gpu);

    context_ = std::make_unique<gossip::context_t>(num_gpu);
    gather_ = std::make_unique<gossip::gather_t>(*context_, transfer_plan);
  }
  
  void ExecAsync() {
    gather_->execAsync(srcs_, srcs_lens_, send_counts_, dst_, dst_len_);
  }

  void Sync() {
    gather_->sync();
  }

 private:
  std::unique_ptr<gossip::context_t> context_;
  std::unique_ptr<gossip::gather_t> gather_;
  std::vector<data_t*> srcs_;
  std::vector<size_t> srcs_lens_;
  std::vector<size_t> send_counts_;
  data_t* dst_;
  size_t dst_len_; 

}; // class LocalGatherComm 

} // namespace sparse
} // namespace byteps

#endif  // BYTEPS_SPARSE_COMMUNICATOR_H
