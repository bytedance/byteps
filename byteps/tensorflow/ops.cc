// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2018 Uber Technologies, Inc.
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

#include <memory>
#include <queue>
#include <thread>
#include <unordered_map>

#include "ops.h"

using namespace byteps;

namespace byteps {
namespace tensorflow {

namespace {

::tensorflow::Status ConvertStatus(const common::Status& status) {
  switch (status.type()) {
    case common::OK:
      return ::tensorflow::Status::OK();
    case common::UNKNOWN_ERROR:
      return ::tensorflow::errors::Unknown(status.reason());
    case common::PRECONDITION_ERROR:
      return ::tensorflow::errors::FailedPrecondition(status.reason());
    case common::ABORTED:
      return ::tensorflow::errors::Aborted(status.reason());
    case common::INVALID_ARGUMENT:
      return ::tensorflow::errors::InvalidArgument(status.reason());
    default:
      return ::tensorflow::errors::Unknown("Unknown error.");
  }
}

int GetDeviceID(::tensorflow::OpKernelContext* context) {
  int device = CPU_DEVICE_ID;
  if (context->device() != nullptr &&
      context->device()->tensorflow_gpu_device_info() != nullptr) {
    device = context->device()->tensorflow_gpu_device_info()->gpu_id;
  }
  return device;
}

// Define all types for TensorUtil.
const common::DataType ConvertDType(int dtype) {
  switch (dtype) {
    case ::tensorflow::DT_UINT8:
      return common::BYTEPS_UINT8;
    case ::tensorflow::DT_INT8:
      return common::BYTEPS_INT8;
    // case ::tensorflow::DT_UINT16:
    //   return common::BYTEPS_UINT16;
    // case ::tensorflow::DT_INT16:
    //   return common::BYTEPS_INT16;
    case ::tensorflow::DT_INT32:
      return common::BYTEPS_INT32;
    case ::tensorflow::DT_INT64:
      return common::BYTEPS_INT64;
    case ::tensorflow::DT_HALF:
      return common::BYTEPS_FLOAT16;
    case ::tensorflow::DT_FLOAT:
      return common::BYTEPS_FLOAT32;
    case ::tensorflow::DT_DOUBLE:
      return common::BYTEPS_FLOAT64;
    // case ::tensorflow::DT_BOOL:
    //   return common::BYTEPS_BOOL;
    default:
      throw std::logic_error("Invalid tensor type.");
  }
}

}  // namespace

TFReadyEvent::TFReadyEvent(::tensorflow::DeviceContext* device_context) {
  auto executor = device_context->stream()->parent();
  auto ready_event = new perftools::gputools::Event(executor);
  ready_event->Init();
  device_context->stream()->ThenRecordEvent(ready_event);
  event_ = std::shared_ptr<perftools::gputools::Event>(ready_event);
}

bool TFReadyEvent::Ready() const {
  return event_->PollForStatus() !=
         perftools::gputools::Event::Status::kPending;
}

TFTensor::TFTensor(::tensorflow::Tensor& tensor) : tensor_(tensor) {}

const common::DataType TFTensor::dtype() const {
  return ConvertDType(tensor_.dtype());
}

const common::TensorShape TFTensor::shape() const {
  common::TensorShape shape;
  for (auto dim : tensor_.shape()) {
    shape.AddDim(dim.size);
  }
  return shape;
}

const void* TFTensor::data() const {
  return (const void*)tensor_.tensor_data().data();
}

int64_t TFTensor::size() const { return (int64_t)tensor_.tensor_data().size(); }

// On GPU this event will signal that data is ready, and tensors are
// allocated.
common::ReadyEvent* RecordReadyEvent(::tensorflow::OpKernelContext* context) {
  auto device_context = context->op_device_context();
  if (device_context != nullptr) {
    return new TFReadyEvent(device_context);
  }
  return nullptr;
}

extern "C" void byteps_tensorflow_declare_tensor(char* name) {
  std::string tensor_name(name);
  common::IsTensorDeclared(tensor_name);
  return;
}

void StartTask(::tensorflow::OpKernelContext* context,
               ::tensorflow::AsyncOpKernel::DoneCallback done,
               std::string node_name, std::shared_ptr<TFTensor> byteps_input,
               std::shared_ptr<TFTensor> byteps_output,
               std::shared_ptr<common::ReadyEvent> ready_event) {
  auto& byteps_context = common::GetContextFromName(node_name);
  auto device = GetDeviceID(context);
  auto size = byteps_input->size();
  auto dtype = byteps_input->dtype();
  void* cpubuff = (device == CPU_DEVICE_ID)
                      ? const_cast<void*>(byteps_input->data())
                      : nullptr;
  common::InitTensor(byteps_context, size, dtype, cpubuff);

  auto queue_list = common::GetPushQueueList(device);
  auto queue_list_pull = common::GetPullQueueList(device);
  queue_list->insert(queue_list->end(), queue_list_pull->begin(),
                     queue_list_pull->end());

  // TODO: assign priority based on topological sort
  auto enqueue_result =
      EnqueueTensor(byteps_context, byteps_input, byteps_output, ready_event,
                    device, -byteps_context.declared_key, 0,
                    [context, done](const common::Status& status) {
                      context->SetStatus(ConvertStatus(status));
                      done();
                    },
                    queue_list);
  OP_REQUIRES_OK_ASYNC(context, ConvertStatus(enqueue_result), done);
}

class BytePSPushPullOp : public ::tensorflow::AsyncOpKernel {
  private:
     std::string input_tensor_name;
 public:
  explicit BytePSPushPullOp(::tensorflow::OpKernelConstruction* context)
      : AsyncOpKernel(context) {
          context->GetAttr("input_name", &input_tensor_name);
      }

  void ComputeAsync(::tensorflow::OpKernelContext* context,
                    DoneCallback done) override {
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(common::CheckInitialized()),
                         done);

    auto tensor = context->input(0);
    ::tensorflow::Tensor* output;
    OP_REQUIRES_OK_ASYNC(
        context, context->allocate_output(0, tensor.shape(), &output), done);
    // ReadyEvent makes sure input tensor is ready, and output is allocated.
    auto ready_event =
        std::shared_ptr<common::ReadyEvent>(RecordReadyEvent(context));
    auto bps_input = std::make_shared<TFTensor>(tensor);
    auto bps_output = std::make_shared<TFTensor>(*output);
    auto node_name = name();
    std::string tmp_name;
    if (input_tensor_name == "default_tensor_name") {
        tmp_name = node_name;
    } else {
        tmp_name = input_tensor_name;
    }
    auto& bps_context = common::GetContextFromName(tmp_name);
    if (bps_context.initialized) {
      StartTask(context, done, tmp_name, bps_input, bps_output, ready_event);
    } else {
      std::thread t(StartTask, context, done, tmp_name, bps_input, bps_output,
                    ready_event);
      t.detach();
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("BytepsPushPull").Device(::tensorflow::DEVICE_CPU),
                        BytePSPushPullOp);
REGISTER_KERNEL_BUILDER(Name("BytepsPushPull").Device(::tensorflow::DEVICE_GPU),
                        BytePSPushPullOp);

REGISTER_OP("BytepsPushPull")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("input_name: string = 'default_tensor_name'")
    .Input("tensor: T")
    .Output("sum: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(
Perform an PushPull on a tensor. All other processes that do a reduction
on a tensor with the same name must have the same dimension for that tensor.
Tensors are reduced with other tensors that have the same node name for the
push_pull.
Arguments
    tensor:     A tensor to reduce.
Output
    sum:    A tensor with the same shape as `tensor`, summed across all processes.
)doc");

}  // namespace tensorflow
}  // namespace byteps
