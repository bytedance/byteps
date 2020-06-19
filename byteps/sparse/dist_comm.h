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

#ifndef BYTEPS_DISTRIBUTED_COMMUNICATOR_H
#define BYTEPS_DISTRIBUTED_COMMUNICATOR_H

#include "communicator.h"

namespace byteps {
namespace sparse {

class DistGatherComm : public SparseComm {
  using data_t = float;
  using K = ps::SArray<ps::Key>;  // keys
  using V = ps::SArray<char>;     // vals
  using L = ps::SArray<int>;      // lens

 public:
  DistGatherComm(ps::KVWorker<char>* ps, std::vector<std::vector<size_t>> globalBufLens,
                 void* dst, size_t dst_len, int local_rank, int local_size, int worker_id, int num_worker) : 
                 ps_(ps), dst_(dst), dst_len_(dst_len), local_rank_(local_rank), local_size_(local_size), 
                 worker_id_(worker_id), num_worker_(num_worker), globalBufLens_(globalBufLens) {
    CHECK_LT(worker_id_, num_worker_);
    CHECK_LT(local_rank_, local_size_);
    
    pskeys_.resize(num_worker_, std::vector<K>(local_size_));
    psvals_.resize(num_worker_, std::vector<V>(local_size_));
    pslens_.resize(num_worker_, std::vector<L>(local_size_));
    cpubuffs_.resize(num_worker_, std::vector<void*>(local_size_));
    cpubuffs_lens_.resize(num_worker_, std::vector<size_t>(local_size_));
    dst_offsets_.resize(num_worker_, std::vector<size_t>(local_size_));

    // prepare dst_offsets_
    size_t offset = 0;
    for (int wid = 0; wid < num_worker_; ++wid) {
      for (int gid = 0; gid < local_size_; ++gid) {
        dst_offsets_[wid][gid] = offset;
        offset += globalBufLens_[wid][gid] / (num_worker_ * local_size_);
      }
    }
    CHECK_EQ(offset, dst_len_) << offset << " " << dst_len_;
    
    auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
    CHECK_EQ(krs.size(), num_worker_);
    for (int wid = 0; wid < num_worker_; ++wid) {
      if (wid == worker_id_) continue;

      for (int gid = 0; gid < local_size_; ++gid) {
        // allocate cuda-aware cpu buffer
        size_t cpubuff_len = 
            globalBufLens_[wid][gid] / (num_worker_ * local_size_);
        void* cpubuff;
        mallocAlignedCudaAwareCpubuff(&cpubuff, cpubuff_len);
        cpubuffs_[wid][gid] = cpubuff;
        cpubuffs_lens_[wid][gid] = cpubuff_len;

        // encode: the last 16 bits are all "1"s 
        // so the server can know this is a gather key
        uint64_t key = ((wid * local_size_ + gid) << 16) + 0xffff;

        // assigned to the server that is colocated with the worker with wid
        ps::Key pskey = krs[wid].begin() + key;
        CHECK_LT(pskey, krs[wid].end());

        // prepare keys
        pskeys_[wid][gid].CopyFrom(&pskey, 1);

        // prepare vals
        psvals_[wid][gid].reset((char*)cpubuff, cpubuff_len, [](void*){});

        // prepare lens
        int tmplen = cpubuff_len;
        pslens_[wid][gid].CopyFrom(&tmplen, 1);
      }
    }

    // init cuda stream for copyH2D
    CUDA_CALL(cudaSetDevice(local_rank_));
    CUDA_CALL(cudaStreamCreateWithFlags(
      copy_stream_, cudaStreamNonBlocking));
    CUDA_CALL(cudaStreamSynchronize(*copy_stream_));
  }

  void ExecAsync() {
    for (int wid = 0; wid < num_worker_; ++wid) {
      if (wid == worker_id_) continue; // skip pull myself
      for (int gid = 0; gid < local_size_; ++gid) {
        auto& keys = pskeys_[wid][gid];
        auto& vals = psvals_[wid][gid]; 
        auto& lens = pslens_[wid][gid]; 
        auto ts = ps_->ZPull(keys, &vals, &lens);
        ts_.push_back(ts);
      }
    }
  }

  void Sync() {
    // sync all gather timestamps and clear 
    for (auto ts : ts_) {
      ps_->Wait(ts);
    }
    ts_.clear();

    // copy from host to gpu
    for (int wid = 0; wid < num_worker_; ++wid) {
      if (wid == worker_id_) continue;
      for (int gid = 0; gid < local_size_; ++gid) {
        void* dst = (void*)((char*)dst_ + dst_offsets_[wid][gid]);
        void* src = cpubuffs_[wid][gid];
        CUDA_CALL(cudaMemcpyAsync(
          (void*) dst, 
          (const void *)src, 
          (size_t) cpubuffs_lens_[wid][gid], 
          (cudaMemcpyKind) cudaMemcpyHostToDevice, 
          (cudaStream_t) *copy_stream_));
      }
    }
    CUDA_CALL(cudaStreamSynchronize(*copy_stream_));
  }

  ~DistGatherComm() {
    pskeys_.clear();
    psvals_.clear();
    pslens_.clear();

    for (auto bufs : cpubuffs_) {
      for (auto buf : bufs) {
        CUDA_CALL(cudaHostUnregister(buf));
        free(buf);
      }
    }
  }
 
 private:
  ps::KVWorker<char>* ps_;
  void* dst_; // the dst buffers
  size_t dst_len_; // length of the dst buffer
  int local_rank_;
  int local_size_;
  int worker_id_;
  int num_worker_;
  std::vector<std::vector<size_t>> globalBufLens_; 
  std::vector<std::vector<size_t>> dst_offsets_; // calculate the offset for copyH2D

  std::vector<std::vector<void *>> cpubuffs_;
  std::vector<std::vector<size_t>> cpubuffs_lens_;
  std::vector<std::vector<K>> pskeys_;
  std::vector<std::vector<V>> psvals_;
  std::vector<std::vector<L>> pslens_;

  std::vector<int> ts_;

  cudaStream_t* copy_stream_;

}; // class DistGatherComm 



class DistScatterComm : public SparseComm {
  using data_t = float;

 public:

  void ExecAsync() {

  }

  void Sync() {

  }
  
}; // class DistScatterComm 

} // namespace sparse
} // namespace byteps

#endif  // BYTEPS_DISTRIBUTED_COMMUNICATOR_H
