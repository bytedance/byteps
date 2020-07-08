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
#include <sstream>

#include <cuda_runtime.h>
#include <cuda.h>

#include "ops.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"

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

// common::ReadyEvent* RecordReadyEvent(::tensorflow::XlaOpKernelContext* context) {
//   auto device_context = context->op_kernel_context()->op_device_context();
//   if (device_context != nullptr) {
//     return new TFReadyEvent(device_context);
//   }
//   return nullptr;
// }

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


class BytepsPushPullXlaOp : public ::tensorflow::XlaOpKernel {
  public:
    explicit BytepsPushPullXlaOp(::tensorflow::OpKernelConstruction* context) : ::tensorflow::XlaOpKernel(context) {
      context->GetAttr("input_name", &input_tensor_name);
    }
    ~BytepsPushPullXlaOp() override = default;

    void Compile(::tensorflow::XlaOpKernelContext* context) override {
      OP_REQUIRES_OK(context, ConvertStatus(common::CheckInitialized()));

      xla::XlaOp tensor = context->Input(0);
      auto shape_or = context->InputXlaShape(0);
      std::cout << " xxx pos 1 " << std::endl;
      // OP_REQUIRES_OK(context, shape_or.status());

      // OP_REQUIRES_OK(
      //   context, context->allocate_output(0, tensor.shape(), &output));
      auto node_name = name();
      std::string tmp_name;
      if (input_tensor_name == "default_tensor_name") {
        tmp_name = node_name;
      } else {
        tmp_name = input_tensor_name;
      }

      std::stringstream ss;
      ss << tmp_name << " " << context->op_kernel_context() << std::endl;
      std::cout << " xxx pos 2 " << std::endl;
      context->SetOutput(
        0, xla::CustomCall(context->builder(),
          /*call_target_name=*/"StartTaskWrapper",
          {tensor}, shape_or.ValueOrDie(), ss.str()));
      // private:
      //   TF_DISALLOW_COPY_AND_ASSIGN(BytepsPushPullXlaOp);
      std::cout << " xxx pos 3 " << std::endl;
    }
  private:
     std::string input_tensor_name;
};

REGISTER_XLA_OP(Name("BytepsPushPull"), BytepsPushPullXlaOp);

void StartTaskXla(::tensorflow::OpKernelContext* context,
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

  bool is_done = false;
  // TODO: assign priority based on topological sort
  auto enqueue_result =
      EnqueueTensor(byteps_context, byteps_input, byteps_output, ready_event,
                    device, -byteps_context.declared_key, 0,
                    [context, &is_done](const common::Status& status) {
                      context->SetStatus(ConvertStatus(status));
                      is_done = true;
                    },
                    queue_list);
  // OP_REQUIRES_OK_ASYNC(context, ConvertStatus(enqueue_result), done);
  // https://en.cppreference.com/w/cpp/thread/condition_variable
  while (!is_done)
    std::this_thread::yield();
}
// void StartTaskWrapper(void* out, const void** in) {
void StartTaskWrapper(CUstream stream, void** buffers,
                      const char* opaque, size_t opaque_len) {
    std::cout << " xxx pos 4 " << std::endl;
    void *a = buffers[0];
    std::cout << " xxx pos 5 " << std::endl;
    void *b = buffers[1];
    std::cout << " xxx pos 6 " << std::endl;
    // auto bps_input = std::make_shared<TFTensor>(*reinterpret_cast<TFTensor *>(buffers[0]));
    auto bps_input = nullptr;
    std::cout << " xxx pos 7 " << std::endl;
    auto bps_output = std::make_shared<TFTensor>(*reinterpret_cast<TFTensor *>(buffers[1]));
    std::cout << " xxx pos 8 " << std::endl;

    std::stringstream ss(opaque);
    std::string tmp_name;
    std::string ptr_str;
    ::tensorflow::OpKernelContext* context;

    ss >> tmp_name;
    ss >> std::hex >> ptr_str;
    context = (::tensorflow::OpKernelContext *) stoul(ptr_str, nullptr, 0);

    auto ready_event =
        std::shared_ptr<common::ReadyEvent>(RecordReadyEvent(context));
    std::cout << " xxx pos 9 " << std::endl;
    auto& bps_context = common::GetContextFromName(tmp_name);
    if (bps_context.initialized) {
      StartTaskXla(context, tmp_name, bps_input, bps_output, ready_event);
    } else {
      std::thread t(StartTaskXla, context, tmp_name, bps_input, bps_output,
                    ready_event);
      t.detach();
    }
}

XLA_REGISTER_CUSTOM_CALL_TARGET(StartTaskWrapper, "CUDA");

}  // namespace tensorflow
}  // namespace byteps

#if 0
// exampel to serialize and deserialize strings  and pointers
#include <iostream>
#include <sstream>

using namespace std;

int main()
{
    cout<<"Hello World";
    int a = 5678;
    int *a_ptr = &a;
//    string mystr = to_string(static_cast<unsigned long>(a_ptr));
//    cout<< "mystr" << mystr << endl;
    string opaque;
    stringstream ss(opaque);
    
    ss << "aabbcc" << " " << a << " " << a_ptr << endl;
    cout << "string is " << ss.str() <<endl;
    
    string tmp_str;
    int b = 0;
    int *b_ptr = NULL;
    string  ptr_str;
    
    ss >> tmp_str;;
    ss >> b;
    ss >> hex >> ptr_str;
    b_ptr = (int *) stoul(ptr_str, nullptr, 0);
    cout << "val of b is " << *b_ptr << endl;
    
    

    return 0;
}
#endif
