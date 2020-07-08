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

#ifndef BYTEPS_TENSORFLOW_OPS_H
#define BYTEPS_TENSORFLOW_OPS_H

#include <string>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#define EIGEN_USE_THREADS
#include "tensorflow/stream_executor/stream.h"

#include "../common/operations.h"

namespace byteps {
namespace tensorflow {

class TFReadyEvent : public common::ReadyEvent {
 public:
  TFReadyEvent(::tensorflow::DeviceContext* device_context);
  bool Ready() const override;

 private:
  std::shared_ptr<perftools::gputools::Event> event_;
};

class XlaReadyEvent : public common::ReadyEvent {
 public:
  XlaReadyEvent(cudaStream_t stream);
  bool Ready() const override;

 private:
  cudaEvent_t cuda_event_ = nullptr;
};

class TFTensor : public common::Tensor {
 public:
  TFTensor(::tensorflow::Tensor& tensor);
  virtual const common::DataType dtype() const override;
  virtual const common::TensorShape shape() const override;
  virtual const void* data() const override;
  virtual int64_t size() const override;

 protected:
  ::tensorflow::Tensor tensor_;
};

class XlaTensor : public common::Tensor {
 public:
  XlaTensor(void *data, int64_t num_elem, ::tensorflow::DataType tf_dtype, int64_t size);
  virtual const common::DataType dtype() const override;
  virtual const common::TensorShape shape() const override;
  virtual const void* data() const override;
  virtual int64_t size() const override;

  protected:
    void *_data;
    int64_t _num_elem;
    ::tensorflow::DataType _tf_dtype;
    int64_t _size;
};

extern "C" void byteps_tensorflow_declare_tensor(char* name);
struct Xla_done_cb_args{
  std::mutex mtx;
  std::condition_variable cv;
  bool is_done;
  void *bps_out_buf;
  void *bps_in_buf;
  std::shared_ptr<::tensorflow::Tensor> output_tensor;
  int bps_buf_size;
  int num_waiting;
};

extern std::unordered_map<std::string, std::shared_ptr<Xla_done_cb_args>> _name_to_done_args;
extern std::mutex _name_to_done_args_mtx;
extern std::condition_variable _name_to_done_args_cv;

}  // namespace tensorflow
}  // namespace byteps

#endif  // BYTEPS_TENSORFLOW_OPS_H
