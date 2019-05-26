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

using namespace tensorflow;
using namespace byteps;

namespace byteps {
namespace tensorflow {

namespace {

std::atomic_int tensor_count;

Status ConvertStatus(const common::Status& status) {
  switch (status.type()) {
  case common::OK:
    return Status::OK();
  case common::UNKNOWN_ERROR:
    return errors::Unknown(status.reason());
  case common::PRECONDITION_ERROR:
    return errors::FailedPrecondition(status.reason());
  case common::ABORTED:
    return errors::Aborted(status.reason());
  case common::INVALID_ARGUMENT:
    return errors::InvalidArgument(status.reason());
  default:
    return errors::Unknown("Unknown error.");
  }
}

common::Status ConvertStatus(const Status& status) {
  switch (status.code()) {
  case error::Code::OK:
    return common::Status::OK();
  case error::Code::UNKNOWN:
    return common::Status::UnknownError(status.error_message());
  case error::Code::FAILED_PRECONDITION:
    return common::Status::PreconditionError(status.error_message());
  case error::Code::ABORTED:
    return common::Status::Aborted(status.error_message());
  case error::Code::INVALID_ARGUMENT:
    return common::Status::InvalidArgument(status.error_message());
  default:
    return common::Status::UnknownError("Unknown error.");
  }
}

int GetDeviceID(OpKernelContext* context) {
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
  case DT_UINT8:
    return common::BYTEPS_UINT8;
  case DT_INT8:
    return common::BYTEPS_INT8;
  // case DT_UINT16:
  //   return common::BYTEPS_UINT16;
  // case DT_INT16:
  //   return common::BYTEPS_INT16;
  case DT_INT32:
    return common::BYTEPS_INT32;
  case DT_INT64:
    return common::BYTEPS_INT64;
  case DT_HALF:
    return common::BYTEPS_FLOAT16;
  case DT_FLOAT:
    return common::BYTEPS_FLOAT32;
  case DT_DOUBLE:
    return common::BYTEPS_FLOAT64;
  // case DT_BOOL:
  //   return common::BYTEPS_BOOL;
  default:
    throw std::logic_error("Invalid tensor type.");
  }
}

} // namespace

#if HAVE_CUDA
TFReadyEvent::TFReadyEvent(DeviceContext* device_context) {
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
#endif

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

const void* TFTensor::data() const { return (const void*)tensor_.tensor_data().data(); }

int64_t TFTensor::size() const { return (int64_t)tensor_.tensor_data().size(); }

// On GPU this event will signal that data is ready, and tensors are
// allocated.
common::ReadyEvent* RecordReadyEvent(OpKernelContext* context) {
#if HAVE_CUDA
  auto device_context = context->op_device_context();
  if (device_context != nullptr) {
    return new TFReadyEvent(device_context);
  }
#endif
  return nullptr;
}

extern "C" void byteps_tensorflow_declare_tensor(char* name, int size, int dtype) {
    // For now, we always allocate cpu buffer
    // TODO: get the device and decide whether to allocate cpu buffer
    std::string tensor_name(name);

    if (!common::IsTensorInitialized(tensor_name, size)) {
        auto& byteps_context = common::GetContextFromName(tensor_name);
        byteps_context.priority = - tensor_count.fetch_add(1);
        common::InitTensor(byteps_context, tensor_name, ConvertDType(dtype), nullptr);
    }
    return;
}

class BytePSPushPullOp : public AsyncOpKernel {
public:
  explicit BytePSPushPullOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(common::CheckInitialized()),
                         done);

    auto node_name = name();
    auto device = GetDeviceID(context);
    auto tensor = context->input(0);
    Tensor* output;
    OP_REQUIRES_OK_ASYNC(
        context, context->allocate_output(0, tensor.shape(), &output), done);
    // ReadyEvent makes sure input tensor is ready, and output is allocated.
    auto ready_event = std::shared_ptr<common::ReadyEvent>(RecordReadyEvent(context));

    auto byteps_input = std::make_shared<TFTensor>(tensor);
    auto byteps_output = std::make_shared<TFTensor>(*output);
    auto& byteps_context = common::GetContextFromName(node_name);

    auto queue_list = common::GetPushQueueList(device);
    auto queue_list_pull = common::GetPullQueueList(device);
    queue_list->insert(queue_list->end(),
                       queue_list_pull->begin(), queue_list_pull->end());

    auto enqueue_result = EnqueueTensor(
        byteps_context, byteps_input, byteps_output, ready_event,
        node_name, device, byteps_context.priority, 0,
        [context, done](const common::Status& status) {
          context->SetStatus(ConvertStatus(status));
          done();
        }, queue_list);
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(enqueue_result), done);
  }
};

REGISTER_KERNEL_BUILDER(Name("BytepsPushPull").Device(DEVICE_CPU),
                        BytePSPushPullOp);
REGISTER_KERNEL_BUILDER(Name("BytepsPushPull").Device(DEVICE_GPU),
                        BytePSPushPullOp);

REGISTER_OP("BytepsPushPull")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Input("tensor: T")
    .Output("sum: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
Perform an PushPull on a tensor. All other processes that do a reduction
on a tensor with the same name must have the same dimension for that tensor.
Tensors are reduced with other tensors that have the same node name for the
push_pull.
Arguments
    tensor:     A tensor to reduce.
Output
    sum:    A tensor with the same shape as `tensor`, summed across all MPI processes.
)doc");

} // namespace tensorflow
} // namespace byteps