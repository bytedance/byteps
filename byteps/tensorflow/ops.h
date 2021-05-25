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
#include "../common/common.h"

namespace byteps {
namespace tensorflow {

class TFReadyEvent : public common::ReadyEvent {
 public:
  TFReadyEvent(::tensorflow::DeviceContext* device_context);
  bool Ready() const override;

 private:
  std::shared_ptr<perftools::gputools::Event> event_;
};

class TFTensor : public common::Tensor {
 public:
  TFTensor(::tensorflow::Tensor& tensor, int device);
  // constructor for TF outputs whose memory may be allocated later
  TFTensor(::tensorflow::OpKernelContext* context,
           ::tensorflow::AsyncOpKernel::DoneCallback done, int output_idx,
           int device);
  virtual const common::DataType dtype() const override;
  virtual const common::TensorShape shape() const override;
  virtual const void* data() const override;
  virtual int64_t size() const override;
  virtual void resize(const common::TensorShape&) override;
  virtual int device() const override;

 protected:
  ::tensorflow::Tensor tensor_;
  ::tensorflow::OpKernelContext* context_;
  ::tensorflow::AsyncOpKernel::DoneCallback done_;
  // the output index
  int idx_;
  // allocated
  bool allocated_ = true;
  // the device id
  int device_;
};

extern "C" void byteps_tensorflow_declare_tensor(char* name);
extern "C" void byteps_tensorflow_declare_tensor_p2p(char* name, int sender, int receiver);

}  // namespace tensorflow
}  // namespace byteps

#endif  // BYTEPS_TENSORFLOW_OPS_H
