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

#ifndef BYTEPS_SPARSE_DENSE_COMM_H
#define BYTEPS_SPARSE_DENSE_COMM_H

#include "communicator.h"
#include "util.h"

namespace byteps {
namespace sparse {

class DenseReduceComm : public SparseComm {
  using data_t = float;
  using K = ps::SArray<ps::Key>;  // keys
  using V = ps::SArray<char>;     // vals
  using L = ps::SArray<int>;      // lens

 public:
  DenseReduceComm(ps::KVWorker<char>* ps, size_t buflen, void* src, void* dst, 
                  int local_rank, int local_size, int worker_id, int num_worker) :
                  ps_(ps), buflen_(buflen), src_(src), dst_(dst), local_rank_(local_rank), 
                  local_size_(local_size), worker_id_(worker_id), num_worker_(num_worker) {
    CHECK_LT(worker_id_, num_worker_);
    CHECK_LT(local_rank_, local_size_);

    mallocAlignedCudaAwareCpubuff(&cpubuff_, buflen_);
    mallocAlignedCudaAwareCpubuff(&dummy_buff_, dummy_buff_len_);

    // prepare pslite communication
    pskeys_.resize(num_worker_);
    psvals_.resize(num_worker_);
    pslens_.resize(num_worker_);
    auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
    CHECK_EQ(krs.size(), num_worker_);
    for (int wid = 0; wid < num_worker_; ++wid) {
      uint64_t key = ((uint64_t)(wid * local_size_ + local_rank_) << 32) + 0xffffffff;
      ps::Key pskey = krs[wid].begin() + key;
      CHECK_LT(pskey, krs[wid].end());

      // keys
      pskeys_[wid].CopyFrom(&pskey, 1);

      if (wid == worker_id_) { 
        // send to local server, use the dummy buffer 
        // (the real data is transferred using shared memory)
        psvals_[wid].reset( 
            (char*) dummy_buff_, (size_t) dummy_buff_len_, [](void*){}); // vals

        pslens_[wid].CopyFrom(&dummy_buff_len_, 1); // lens
      } else { 
        // send to other workers, use normal buffers
        psvals_[wid].reset(
            (char*) cpubuff_, (size_t) buflen_, [](void*){}); // vals
        
        int intlen = buflen_;
        pslens_[wid].CopyFrom(&intlen, 1); // lens
      }
    }
    
    // init cuda stream for cuda memcpy
    CUDA_CALL(cudaSetDevice(local_rank_));
    copy_d2h_stream_ = (cudaStream_t*) malloc(sizeof(cudaStream_t) * 1);
    copy_h2d_stream_ = (cudaStream_t*) malloc(sizeof(cudaStream_t) * 1);
    CUDA_CALL(cudaStreamCreateWithFlags(copy_d2h_stream_, cudaStreamNonBlocking));
    CUDA_CALL(cudaStreamCreateWithFlags(copy_h2d_stream_, cudaStreamNonBlocking));
    CUDA_CALL(cudaStreamSynchronize(*copy_d2h_stream_));
    CUDA_CALL(cudaStreamSynchronize(*copy_h2d_stream_));
  }
  
  void ExecAsync() {
    CUDA_CALL(cudaMemcpyAsync(
      (void*) cpubuff_, 
      (const void *) src_, 
      (size_t) buflen_, 
      (cudaMemcpyKind) cudaMemcpyDeviceToHost, 
      (cudaStream_t) *copy_d2h_stream_));
    CUDA_CALL(cudaStreamSynchronize(*copy_d2h_stream_));

  }

  void Sync() {

  }

  void FreeBuffer(void* buf) {
    CUDA_CALL(cudaHostUnregister(buf));
    free(buf);
  }

  ~DenseReduceComm() {
    CUDA_CALL(cudaStreamSynchronize(*copy_d2h_stream_));
    CUDA_CALL(cudaStreamSynchronize(*copy_h2d_stream_));
    CUDA_CALL(cudaStreamDestroy(*copy_d2h_stream_));
    CUDA_CALL(cudaStreamDestroy(*copy_h2d_stream_));
    FreeBuffer(cpubuff_);
    FreeBuffer(dummy_buff_);
  }
  
 private:
  ps::KVWorker<char>* ps_; 
  size_t buflen_;
  void* src_; // on gpu
  void* dst_; // on gpu
  int local_rank_;
  int local_size_; 
  int worker_id_; 
  int num_worker_;

  void* cpubuff_;
  cudaStream_t* copy_d2h_stream_;
  cudaStream_t* copy_h2d_stream_;

  std::vector<K> pskeys_;
  std::vector<V> psvals_;
  std::vector<L> pslens_;

  // a dummy buffer for local push,
  // this buffer length should be very small
  // in order to prevent loop-back RDMA incast
  void* dummy_buff_;
  const int dummy_buff_len_ = 8; // bytes
};

} // namespace sparse
} // namespace byteps

#endif // BYTEPS_SPARSE_DENSE_COMM_H