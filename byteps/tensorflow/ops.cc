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
#include <atomic>

#include "ops.h"
#include "../common/logging.h"

using namespace byteps;

namespace byteps {
namespace common {
class TensorTableEntry;
}
namespace tensorflow {

namespace {

template <typename T>
std::vector<int32_t> AsInt32(const ::tensorflow::Tensor* tensor, int64_t num_elements) {
  std::vector<int32_t> ret(num_elements);
  auto data = tensor->vec<T>();
  for (int32_t i = 0; i < num_elements; ++i) {
    ret[i] = data(i);
  }
  return ret;
}

void GetIntList(const ::tensorflow::Tensor& tensor, std::vector<int32_t>* results) {
  if (tensor.dtype() == ::tensorflow::DT_INT32) {
    *results = AsInt32<int32_t>(&tensor, tensor.shape().num_elements());
  } else if (tensor.dtype() == ::tensorflow::DT_INT64) {
    *results = AsInt32<int64_t>(&tensor, tensor.shape().num_elements());
  } else {
    CHECK(false) << "unexpected dtype";
  }
}

std::string GetOpName(const std::string& prefix, const std::string& name,
                      int handle) {
  if (!name.empty()) {
    return prefix + "." + std::string(name);
  }
  return prefix + ".noname." + std::to_string(handle);
}


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
#if BYTEPS_BUILDING_CUDA == 1
  if (context->device() != nullptr &&
      context->device()->tensorflow_gpu_device_info() != nullptr) {
    device = context->device()->tensorflow_gpu_device_info()->gpu_id;
  }
#endif
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

TFTensor::TFTensor(::tensorflow::Tensor& tensor, int device) : tensor_(tensor), device_(device) {}
TFTensor::TFTensor(::tensorflow::Tensor* tensor, int device) : tensor_(*tensor), device_(device) {}

TFTensor::TFTensor(::tensorflow::OpKernelContext* context,
                   ::tensorflow::AsyncOpKernel::DoneCallback done, int output_idx, int device) :
                   context_(context), done_(done), idx_(output_idx), allocated_(false), device_(device) {}

void TFTensor::resize(const common::TensorShape& shape) {
  CHECK(!allocated_);
  CHECK(context_ != nullptr);
  ::tensorflow::TensorShape tf_shape;
  for (auto dim : shape.shape_) {
    tf_shape.AddDim(dim);
  }
  ::tensorflow::Tensor* tf_tensor;
  OP_REQUIRES_OK_ASYNC(context_, context_->allocate_output(idx_, tf_shape, &tf_tensor), done_);
  allocated_ = true;
  tensor_ = *tf_tensor;
}

const common::DataType TFTensor::dtype() const {
  CHECK(allocated_);
  return ConvertDType(tensor_.dtype());
}

const common::TensorShape TFTensor::shape() const {
  CHECK(allocated_);
  common::TensorShape shape;
  for (auto dim : tensor_.shape()) {
    shape.AddDim(dim.size);
  }
  return shape;
}

const void* TFTensor::data() const {
  if (!allocated_) return nullptr;
  return (const void*)tensor_.tensor_data().data();
}

int64_t TFTensor::size() const {
  CHECK(allocated_);
  return (int64_t)tensor_.tensor_data().size();
}

int TFTensor::device() const {
  return device_;
}

// On GPU this event will signal that data is ready, and tensors are allocated.
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

extern "C" void byteps_tensorflow_declare_tensor_p2p(char* name, int sender, int receiver) {
  std::string prefix = "byteps_p2p_send_";
  if (sender == -1) sender = common::byteps_rank();
  if (receiver == -1) receiver = common::byteps_rank();
  prefix += std::to_string(sender) + "_recv_" + std::to_string(receiver);
  std::string tensor_name = GetOpName(prefix, name, 0);
  common::IsTensorDeclaredP2P(tensor_name, sender, receiver);
}

extern "C" void byteps_tensorflow_declare_tensor_alltoall(char* name) {
  const int my_rank = common::byteps_rank();
  const int size = common::byteps_size();
  // Declare tensors for alltoall
  std::string name_prefix = std::string(name);
  std::string my_rank_str = std::to_string(my_rank);
  uint32_t session_size = common::byteps_session_size();
  for (uint32_t session_id = 0; session_id < session_size; session_id++) {
    std::string session_prefix = "session_" + std::to_string(session_id) + "_" + name_prefix;
    for (size_t i = 0; i < size; ++i) {
      std::string target_rank = std::to_string(i);
      std::string name_send = session_prefix + "_alltoall_send_" + my_rank_str + "_recv_" + target_rank;
      std::string name_recv = session_prefix + "_alltoall_send_" + target_rank + "_recv_" + my_rank_str;
      common::IsTensorDeclaredP2P(name_send, my_rank, i);
      common::IsTensorDeclaredP2P(name_recv, i, my_rank);
    }
  }
}

enum TaskType {
  kSend,
  kRecv,
  kPushPull,
  kAlltoAll,
};

void StartTask(::tensorflow::OpKernelContext* context,
               ::tensorflow::AsyncOpKernel::DoneCallback done,
               std::string node_name, std::shared_ptr<TFTensor> byteps_input,
               std::shared_ptr<TFTensor> byteps_output,
               std::shared_ptr<common::ReadyEvent> ready_event,
               common::ReduceOp op) {
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
  // TODO: why -byteps_context.declared_key 
  auto enqueue_result =
      EnqueueTensor(byteps_context, byteps_input, byteps_output, ready_event,
                    device, -byteps_context.declared_key, 0,
                    [context, done](const common::Status& status) {
                      context->SetStatus(ConvertStatus(status));
                      done();
                    },
                    queue_list, op);
  OP_REQUIRES_OK_ASYNC(context, ConvertStatus(enqueue_result), done);
}

class BytePSPushPullOp : public ::tensorflow::AsyncOpKernel {
  private:
     std::string input_tensor_name;
     std::string op;
     common::ReduceOp reduce_op;
 public:
  explicit BytePSPushPullOp(::tensorflow::OpKernelConstruction* context)
      : AsyncOpKernel(context) {
          context->GetAttr("input_name", &input_tensor_name);
          context->GetAttr("op", &op);
          if (op == std::string("average")) {
            reduce_op = common::REDUCE_OP_AVERAGE;
          } else if (op == std::string("sum")) {
            reduce_op = common::REDUCE_OP_SUM;
          } else {
            throw std::logic_error("operation type.");
          }
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
    auto bps_input = std::make_shared<TFTensor>(tensor, GetDeviceID(context));
    auto bps_output = std::make_shared<TFTensor>(*output, GetDeviceID(context));
    auto node_name = name();
    std::string tmp_name;
    if (input_tensor_name == "default_tensor_name") {
        tmp_name = node_name;
    } else {
        tmp_name = input_tensor_name;
    }

    auto& bps_context = common::GetContextFromName(tmp_name);
    if (bps_context.initialized) {
      StartTask(context, done, tmp_name, bps_input, bps_output, ready_event,
                reduce_op);
    } else {
      std::thread t(StartTask, context, done, tmp_name, bps_input, bps_output,
                    ready_event, reduce_op);
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
    .Attr("op: string = 'average'")
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

// ================== SEND / RECV ======================== // 

void StartP2PTask(::tensorflow::OpKernelContext* context,
                  ::tensorflow::AsyncOpKernel::DoneCallback done,
                  int sender, int receiver,
                  std::string node_name, std::shared_ptr<TFTensor> tensor,
                  std::shared_ptr<common::ReadyEvent> ready_event, int version,
                  int priority, TaskType task) {
  auto& byteps_context = common::GetContextFromName(node_name);
  auto device = GetDeviceID(context);
  auto size = tensor->size();
  auto dtype = tensor->dtype();
  auto byteps_input = task == kSend ? tensor : nullptr;
  auto byteps_output = task == kRecv ? tensor : nullptr;
  if (task == kSend && receiver == common::byteps_rank()) {
    byteps_output = tensor;
  }
  if (task == kRecv && sender == common::byteps_rank()) {
    byteps_input = tensor;
  }

  void* cpubuff = (device == CPU_DEVICE_ID)
                      ? const_cast<void*>(tensor->data())
                      : nullptr;
  common::InitTensorP2P(byteps_context, size, dtype, cpubuff,
                        sender, receiver, false);

  std::shared_ptr<std::vector<common::QueueType>> queue_list; 

  if (task == kSend) {
    queue_list = common::GetSendQueueList();
  } else if (task == kRecv) {
    queue_list = common::GetRecvQueueList();
  } else {
    throw std::logic_error("Invalid task type");
  }

  auto enqueue_result =
      EnqueueTensor(byteps_context, byteps_input, byteps_output, ready_event,
                    device, priority, version,
                    [context, done](const common::Status& status) {
                      context->SetStatus(ConvertStatus(status));
                      done();
                    },
                    queue_list);
  OP_REQUIRES_OK_ASYNC(context, ConvertStatus(enqueue_result), done);
}


class BytePSSendOp : public ::tensorflow::AsyncOpKernel {
  private:
     std::string input_tensor_name;
     int sender;
     int receiver;
     int version;
     int priority;
 public:
  explicit BytePSSendOp(::tensorflow::OpKernelConstruction* context)
      : AsyncOpKernel(context) {
          context->GetAttr("input_name", &input_tensor_name);
          context->GetAttr("sender", &sender);
          context->GetAttr("receiver", &receiver);
          context->GetAttr("version", &version);
          context->GetAttr("priority", &priority);
      }

  void ComputeAsync(::tensorflow::OpKernelContext* context,
                    DoneCallback done) override {
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(common::CheckInitialized()),
                         done); 
    auto tensor = context->input(0);
    auto ready_event =
        std::shared_ptr<common::ReadyEvent>(RecordReadyEvent(context));
    auto bps_input = std::make_shared<TFTensor>(tensor, GetDeviceID(context));
    auto node_name = name();
    if (sender == -1) sender = common::byteps_rank();
    if (receiver == -1) receiver = common::byteps_rank();
    std::string prefix = "byteps_p2p_send_" + std::to_string(sender) + "_recv_" + std::to_string(receiver);
    std::string tmp_name;
    if (input_tensor_name == "default_tensor_name") {
      tmp_name = GetOpName(prefix, node_name.c_str(), 0);
    } else {
      tmp_name = GetOpName(prefix, input_tensor_name.c_str(), 0);
    }
    auto& bps_context = common::GetContextFromName(tmp_name);
    if (bps_context.initialized) {
      StartP2PTask(context, done, sender, receiver, tmp_name, bps_input, ready_event,
                    version, priority, kSend);
    } else {
      std::thread t(StartP2PTask, context, done, sender, receiver, tmp_name, bps_input,
                    ready_event, version, priority, kSend);
      t.detach();
    }
  }
};

class BytePSRecvOp : public ::tensorflow::AsyncOpKernel {
  private:
     std::string input_tensor_name;
     int sender;
     int receiver;
     int version;
     int priority;
 public:
  explicit BytePSRecvOp(::tensorflow::OpKernelConstruction* context)
      : AsyncOpKernel(context) {
          context->GetAttr("input_name", &input_tensor_name);
          context->GetAttr("sender", &sender);
          context->GetAttr("receiver", &receiver);
          context->GetAttr("version", &version);
          context->GetAttr("priority", &priority);
      }

  void ComputeAsync(::tensorflow::OpKernelContext* context,
                    DoneCallback done) override {
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(common::CheckInitialized()),
                         done);

    auto tensor = context->input(0);
    auto ready_event =
        std::shared_ptr<common::ReadyEvent>(RecordReadyEvent(context));
    auto bps_input = std::make_shared<TFTensor>(tensor, GetDeviceID(context));
    auto node_name = name();
    if (sender == -1) sender = common::byteps_rank();
    if (receiver == -1) receiver = common::byteps_rank();

    std::string prefix = "byteps_p2p_send_" + std::to_string(sender) + "_recv_" + std::to_string(receiver);
    std::string tmp_name;
    if (input_tensor_name == "default_tensor_name") {
      tmp_name = GetOpName(prefix, node_name.c_str(), 0);
    } else {
      tmp_name = GetOpName(prefix, input_tensor_name.c_str(), 0);
    }
    auto& bps_context = common::GetContextFromName(tmp_name);
    if (bps_context.initialized) {
      StartP2PTask(context, done, sender, receiver, tmp_name, bps_input, ready_event,
                    version, priority, kRecv);
    } else {
      std::thread t(StartP2PTask, context, done, sender, receiver, tmp_name, bps_input,
                    ready_event, version, priority, kRecv);
      t.detach();
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("BytepsSend").Device(::tensorflow::DEVICE_CPU),
                        BytePSSendOp);
REGISTER_KERNEL_BUILDER(Name("BytepsSend").Device(::tensorflow::DEVICE_GPU),
                        BytePSSendOp);

REGISTER_KERNEL_BUILDER(Name("BytepsRecv").Device(::tensorflow::DEVICE_CPU),
                        BytePSRecvOp);
REGISTER_KERNEL_BUILDER(Name("BytepsRecv").Device(::tensorflow::DEVICE_GPU),
                        BytePSRecvOp);



// ======================= ALL_TO_ALL OPERATIONS =====================
void StartAlltoAllTask(::tensorflow::OpKernelContext* context,
                        ::tensorflow::AsyncOpKernel::DoneCallback done,
                        std::string node_name,
                        std::shared_ptr<TFTensor> byteps_input,
                        std::vector<std::shared_ptr<TFTensor>> byteps_group_input,
                        std::vector<int32_t> split_count,
                        std::vector<int32_t> recv_split_count,
                        std::shared_ptr<TFTensor> byteps_output,
                        std::vector<std::shared_ptr<TFTensor>> byteps_group_output,
                        std::shared_ptr<TFTensor> byteps_aux_output,
                        std::shared_ptr<common::ReadyEvent> ready_event,
                        TaskType task, bool recv_split_unknown,
                        std::string name_wo_session) {
  // the counter for all send/recv tasks generated for alltoall
  int num_ranks = split_count.size();
  // one recv per rank (including memcpy). one d2h_send
  int num_tasks = recv_split_unknown ? 2 : num_ranks + 1;
  std::shared_ptr<std::atomic_int> counter_ptr(new std::atomic_int(num_tasks));

  // common fields
  auto device = GetDeviceID(context);
  int priority = 0;
  auto callback = [context, done, counter_ptr, name_wo_session](const common::Status& status) {
    // we use a counter to track each subtask
    // the alltoall operator is completed after all subtasks are done
    if (--(*counter_ptr) <= 0) {
      common::byteps_mark_done(name_wo_session.c_str());
      context->SetStatus(ConvertStatus(status));
      done();
    }
  };

  common::Status enqueue_result;
  // the begin index of tensor views for send/recv
  std::vector<int> send_begin;
  std::vector<int> recv_begin;
  send_begin.push_back(0);
  recv_begin.push_back(0);
  for (size_t i = 0; i < num_ranks; ++i) {
    send_begin.push_back(send_begin.back() + split_count.at(i));
    recv_begin.push_back(recv_begin.back() + recv_split_count.at(i));
  }
  std::vector<std::shared_ptr<common::Tensor>> group_input;
  for (auto tensor : byteps_group_input) {
    group_input.push_back(tensor);
  }
  std::vector<std::shared_ptr<common::Tensor>> group_output;
  for (auto tensor : byteps_group_output) {
    group_output.push_back(tensor);
  }
  enqueue_result = common::EnqueueAlltoAllTensor(node_name, 
                                                 byteps_input, group_input,
                                                 byteps_output, group_output, byteps_aux_output,
                                                 ready_event, device,
                                                 priority, 0, callback,
                                                 send_begin, recv_begin,
                                                 counter_ptr.get(), recv_split_unknown);
  OP_REQUIRES_OK_ASYNC(context, ConvertStatus(enqueue_result), done);
}

template <bool cross_device>
class BytepsAllToAllOp : public ::tensorflow::AsyncOpKernel {
 private:
     std::string input_tensor_name;
     bool recv_split_unknown;
     int partition_bytes;

 public:
  explicit BytepsAllToAllOp(::tensorflow::OpKernelConstruction* context)
      : AsyncOpKernel(context) {
      context->GetAttr("input_name", &input_tensor_name);
      context->GetAttr("recv_split_unknown", &recv_split_unknown);
      if (recv_split_unknown) {
        CHECK(getenv("BYTEPS_PARTITION_BYTES"))
          << "BYTEPS_PARTITION_BYTES is required if recv_split_unknown=True";
        partition_bytes = atoi(getenv("BYTEPS_PARTITION_BYTES"));
        if (getenv("BYTEPS_P2P_PARTITION_BYTES")) {
          partition_bytes = atoi(getenv("BYTEPS_P2P_PARTITION_BYTES"));
        }
      }
    }

  void ComputeAsync(::tensorflow::OpKernelContext* context,
                    DoneCallback done) override {
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(common::CheckInitialized()),
                         done);
    const int num_outputs = context->num_outputs();
    CHECK(num_outputs == 2) << num_outputs;
    const int num_inputs = context->num_inputs();
    CHECK(num_inputs == 3) << num_inputs;

    // naming
    auto node_name = name();
    std::string tmp_name;
    if (input_tensor_name == "default_tensor_name") {
      tmp_name = node_name;
    } else {
      tmp_name = input_tensor_name;
    }

    auto tensor = context->input(0);
    ::tensorflow::Tensor* output_data;
    ::tensorflow::Tensor* output_sizes;
    // TODO: move some of these logics to BytePS core
    // calculate the stride based on axis[1:]
    int64_t stride = 1;
    for (int i = 1; i < tensor.shape().dims(); ++i) {
      stride *= tensor.shape().dim_size(i);
    }
    // calculate counts based on the stride
    const auto split_tensor = context->input(1);
    // Note: if recv_is_approximate=True, recv_split_tensor reuses split_tensor
    const auto recv_split_tensor = context->input(2);
    std::vector<int32_t> dim0_split_count;
    std::vector<int32_t> split_count;
    std::vector<int32_t> recv_split_count;
    ::tensorflow::TensorShape result_shape;
    GetIntList(split_tensor, &dim0_split_count);
    int dim0 = 0;
    int dim0_aggregate = 0;
    // the split tensor is based on axis 0, hence scale it by stride
    for (int i = 0; i < dim0_split_count.size(); ++i) {
      dim0_aggregate += dim0_split_count[i];
      split_count.push_back(dim0_split_count[i] * stride);
      if (dim0_split_count[i] < 0) {
        std::string err_msg = "invalid split for " + tmp_name + " at idx " + std::to_string(i) + ": " + std::to_string(dim0_split_count[i]);
        auto status = ::tensorflow::errors::InvalidArgument(err_msg);
        OP_REQUIRES_OK_ASYNC(context, status, done);
      }
    }
    // sanity checks
    if (dim0_aggregate != tensor.shape().dim_size(0)) {
      std::string count_str;
      for (int i = 0; i < dim0_split_count.size(); ++i) {
        count_str += std::to_string(dim0_split_count[i]) + ",";
      }
      count_str = "invalid split for " + tmp_name + ". dim0 = " + std::to_string(tensor.shape().dim_size(0)) + ". split = " + count_str;
      auto status = ::tensorflow::errors::InvalidArgument(count_str);
      OP_REQUIRES_OK_ASYNC(context, status, done);
    }
    CHECK(split_tensor.shape().dim_size(0) == recv_split_tensor.shape().dim_size(0));
    GetIntList(recv_split_tensor, &recv_split_count);
    // translate the split value (for dim0) with stride (considering all dimensions)
    for (int i = 0; i < recv_split_count.size(); ++i) {
      if (recv_split_count[i] < 0) {
        std::string err_msg = "invalid recv_split for " + tmp_name + " at index " + std::to_string(i) + ": " + std::to_string(recv_split_count[i]);
        auto status = ::tensorflow::errors::InvalidArgument(err_msg);
        OP_REQUIRES_OK_ASYNC(context, status, done);
      }
      dim0 += recv_split_count[i];
      recv_split_count[i] *= stride;
    }
    // Allocate output
    result_shape.AddDim(dim0);
    for (int i = 1; i < tensor.shape().dims(); ++i) {
      result_shape.AddDim(tensor.shape().dim_size(i));
    }
    auto device_id = GetDeviceID(context);
    int output_device = device_id;
    int input_device = device_id;
    // adjust device_id in cross device case
    if (cross_device) {
      if (context->input_memory_type(0) == ::tensorflow::HOST_MEMORY) {
        input_device = CPU_DEVICE_ID;
      }
      if (context->output_memory_type(0) == ::tensorflow::HOST_MEMORY) {
        output_device = CPU_DEVICE_ID;
      }
    }
    std::shared_ptr<TFTensor> bps_output;
    if (recv_split_unknown) {
      bps_output = std::make_shared<TFTensor>(context, done, 0, output_device);
    } else {
      OP_REQUIRES_OK_ASYNC(
        context, context->allocate_output(0, result_shape, &output_data), done);
      bps_output = std::make_shared<TFTensor>(*output_data, output_device);
    }
    OP_REQUIRES_OK_ASYNC(
        context, context->allocate_output(1, split_tensor.shape(), &output_sizes), done);

    // naming and declarations
    auto ready_event = std::shared_ptr<common::ReadyEvent>(RecordReadyEvent(context));
    auto bps_input = std::make_shared<TFTensor>(tensor, input_device);
    // TODO: pass the correct aux_output device id
    auto bps_aux_output = std::make_shared<TFTensor>(*output_sizes, CPU_DEVICE_ID);
    const int my_rank = common::byteps_rank();
    // Add session_id prefix to node_name
    std::string name_send = tmp_name + "_alltoall_send_" + std::to_string(my_rank) + "_recv_0";
    int session_id = common::byteps_session_id(name_send.c_str());
    int session_size = common::byteps_session_size();
    std::string session_prefix = "session_" + std::to_string(session_id % session_size) + "_";
    std::string session_name_send = session_prefix + name_send;
    std::string session_tmp_name = session_prefix + tmp_name;
    if (!recv_split_unknown && dim0 == 0 && dim0_aggregate == 0) {
      // TODO: fill in size tensor
      done();
      return;
    }
    auto& bps_context = common::GetContextFromName(session_name_send);
    std::vector<std::shared_ptr<TFTensor>> empty_group_inputs;
    std::vector<std::shared_ptr<TFTensor>> empty_group_outputs;
    if (bps_context.initialized) {
      StartAlltoAllTask(context, done, session_tmp_name, bps_input, empty_group_inputs, split_count,
                        recv_split_count, bps_output, empty_group_outputs, bps_aux_output, ready_event,
                        kAlltoAll, recv_split_unknown, name_send);
    } else {
      std::thread t(StartAlltoAllTask, context, done, session_tmp_name, bps_input, empty_group_inputs, split_count,
                    recv_split_count, bps_output, empty_group_outputs, bps_aux_output, ready_event,
                    kAlltoAll, recv_split_unknown, name_send);
      t.detach();
    }
  }
};


template <bool cross_device>
class BytepsAllToAllGroupOp : public ::tensorflow::AsyncOpKernel {
 private:
     std::string input_tensor_name;
     bool recv_split_unknown;

 public:
  explicit BytepsAllToAllGroupOp(::tensorflow::OpKernelConstruction* context)
      : AsyncOpKernel(context) {
      context->GetAttr("input_name", &input_tensor_name);
      context->GetAttr("recv_split_unknown", &recv_split_unknown);
    }

  void ComputeAsync(::tensorflow::OpKernelContext* context,
                    DoneCallback done) override {
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(common::CheckInitialized()),
                         done);
    const int num_outputs = context->num_outputs();
    const int num_inputs = context->num_inputs();
    CHECK(!recv_split_unknown) << "recv_split_unknown=true is not supported now";

    // naming
    auto node_name = name();
    std::string tmp_name;
    if (input_tensor_name == "default_tensor_name") {
      tmp_name = node_name;
    } else {
      tmp_name = input_tensor_name;
    }

    const int n = num_inputs - 2;
    std::vector<const ::tensorflow::Tensor*> tensors(n);
    for (int i = 0; i < n; ++i) {
      tensors[i] = &context->input(i);
    }

    int64_t stride = 1;
    for (auto& tensor : tensors) {
      for (int i = 1; i < tensor->shape().dims(); ++i) { 
        stride *= tensor->shape().dim_size(i);
      }
    }
    // calculate counts based on the stride
    const auto split_tensor = context->input(num_inputs - 2);
    // Note: if recv_is_approximate=True, recv_split_tensor reuses split_tensor
    const auto recv_split_tensor = context->input(num_inputs - 1);
    std::vector<int32_t> dim0_split_count;
    std::vector<int32_t> split_count;
    std::vector<int32_t> recv_split_count;
    std::vector<::tensorflow::TensorShape> result_shape(num_outputs / 2);
    GetIntList(split_tensor, &dim0_split_count);
    int dim0_aggregate = 0;
    // the split tensor is based on axis 0, hence scale it by stride
    for (int i = 0; i < dim0_split_count.size(); ++i) {
      dim0_aggregate += dim0_split_count[i];
      split_count.push_back(dim0_split_count[i] * stride);
      if (dim0_split_count[i] < 0) {
        std::string err_msg = "invalid split for " + tmp_name + " at idx " 
                            + std::to_string(i) + ": " + std::to_string(dim0_split_count[i]);
        auto status = ::tensorflow::errors::InvalidArgument(err_msg);
        OP_REQUIRES_OK_ASYNC(context, status, done);
      }
    }
    // sanity checks
    unsigned int aggregated_len = 0;
    for (auto tensor : tensors) {
      for (int i = 1; i < tensor->shape().dims(); ++i) {
        aggregated_len += tensor->shape().dim_size(i);
      }
    }
    CHECK(split_tensor.shape().dim_size(0) == recv_split_tensor.shape().dim_size(0));
    GetIntList(recv_split_tensor, &recv_split_count);
    // translate the split value (for dim0) with stride (considering all dimensions)
    int dim0 = 0;
    for (int i = 0; i < recv_split_count.size(); ++i) {
      if (recv_split_count[i] < 0) {
        std::string err_msg = "invalid recv_split for " + tmp_name + " at index " 
                            + std::to_string(i) + ": " + std::to_string(recv_split_count[i]);
        auto status = ::tensorflow::errors::InvalidArgument(err_msg);
        OP_REQUIRES_OK_ASYNC(context, status, done);
      }
      dim0 += recv_split_count[i];
      recv_split_count[i] *= stride;
    }
    // Allocate output
    CHECK(recv_split_count.size() == result_shape.size());
    CHECK(recv_split_count.size() == tensors.size());
    for (int i = 0; i < recv_split_count.size(); ++i) {
      result_shape[i].AddDim(recv_split_count[i]);
      for (int j = 1; j < tensors[i]->shape().dims(); ++j) { 
        result_shape[i].AddDim(tensors[i]->shape().dim_size(j));
      }
    }
    // TODO: fix the shape 
    auto device_id = GetDeviceID(context);
    int output_device = device_id;
    int input_device = device_id;
    // adjust device_id in cross device case
    if (cross_device) {
      if (context->input_memory_type(0) == ::tensorflow::HOST_MEMORY) {
        input_device = CPU_DEVICE_ID;
      }
      if (context->output_memory_type(0) == ::tensorflow::HOST_MEMORY) {
        output_device = CPU_DEVICE_ID;
      }
    }

    int n_tensor = num_outputs / 2;
    std::vector<::tensorflow::Tensor*> output_data(n_tensor);
    std::vector<::tensorflow::Tensor*> output_sizes(n_tensor);
    std::vector<std::shared_ptr<TFTensor>> bps_group_output(n_tensor);

    for (int i = 0; i < n_tensor; ++i) {
      OP_REQUIRES_OK_ASYNC(
          context, context->allocate_output(i, result_shape[i], &output_data[i]), done);
      OP_REQUIRES_OK_ASYNC(
          context, context->allocate_output(i + n_tensor, split_tensor.shape(), &output_sizes[i]), done);
      bps_group_output[i] = std::make_shared<TFTensor>(*output_data[i], output_device);
    }

    // naming and declarations
    auto ready_event = std::shared_ptr<common::ReadyEvent>(RecordReadyEvent(context));
    
    std::vector<std::shared_ptr<TFTensor>> bps_group_input;
    for (auto& tensor : tensors) {
      bps_group_input.push_back(std::make_shared<TFTensor>((::tensorflow::Tensor*)tensor, input_device));
    }
    // TODO: pass the correct aux_output device id
    auto bps_aux_output = std::make_shared<TFTensor>(*output_sizes[0], CPU_DEVICE_ID);
    const int my_rank = common::byteps_rank();
    // Add session_id prefix to node_name
    std::string name_send = tmp_name + "_alltoall_send_" + std::to_string(my_rank) + "_recv_0";
    int session_id = common::byteps_session_id(name_send.c_str());
    int session_size = common::byteps_session_size();
    std::string session_prefix = "session_" + std::to_string(session_id % session_size) + "_";
    std::string session_name_send = session_prefix + name_send;
    std::string session_tmp_name = session_prefix + tmp_name;
    if (!recv_split_unknown && dim0 == 0 && dim0_aggregate == 0) {
      // TODO: fill in size tensor
      done();
      return;
    }
    auto& bps_context = common::GetContextFromName(session_name_send);
    if (bps_context.initialized) {
      StartAlltoAllTask(context, done, session_tmp_name, nullptr, bps_group_input, 
                        split_count, recv_split_count, nullptr, bps_group_output, bps_aux_output, 
                        ready_event, kAlltoAll, recv_split_unknown, name_send);
    } else {
      std::thread t(StartAlltoAllTask, context, done, session_tmp_name, nullptr, bps_group_input, 
                    split_count, recv_split_count, nullptr, bps_group_output, bps_aux_output, 
                    ready_event, kAlltoAll, recv_split_unknown, name_send);
      t.detach();
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("BytepsAlltoall").Device(::tensorflow::DEVICE_CPU),
                        BytepsAllToAllOp<false>);

REGISTER_KERNEL_BUILDER(Name("BytepsAlltoall").Device(::tensorflow::DEVICE_GPU)
                                                  .HostMemory("splits")
                                                  .HostMemory("recv_splits")
                                                  .HostMemory("recv_bytes"),
                        BytepsAllToAllOp<false>);

REGISTER_OP("BytepsAlltoall")
    .Attr(
        "T: {uint8, int8, uint16, int16, int32, int64, float16, float32, float64, bool}")
    .Attr("input_name: string = 'default_tensor_name'") 
    .Input("tensor: T")
    .Input("splits: int32")
    .Input("recv_splits: int32")
    .Attr("recv_split_unknown: bool = False")
    .Output("output: T")
    .Output("recv_bytes: int32") // TODO: rename this output
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle output;
      TF_RETURN_IF_ERROR(c->ReplaceDim(c->input(0), 0, c->UnknownDim(), &output));
      c->set_output(0, output);
      c->set_output(1, c->input(1));
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(

Perform an MPI Alltoall on a tensor.
Arguments
    tensor:     A tensor to be distributed with all to all  // for send counts (dim0)
    splits:     A list of integers in rank order describing how many elements
                in `tensor` to send to each worker.  // for recv counts (dim0)
    recv_split_unknown: A bool to indicate whether recv splits is unknown  
Output
    output:    The collected tensor data from all workers.
)doc");

REGISTER_KERNEL_BUILDER(Name("BytepsAlltoallGroup").Device(::tensorflow::DEVICE_CPU),
                        BytepsAllToAllGroupOp<false>);

REGISTER_KERNEL_BUILDER(Name("BytepsAlltoallGroup").Device(::tensorflow::DEVICE_GPU)
                                                  .HostMemory("splits")
                                                  .HostMemory("recv_splits")
                                                  .HostMemory("recv_bytes"),
                        BytepsAllToAllGroupOp<false>);

REGISTER_OP("BytepsAlltoallGroup")
    .Attr(
        "T: {uint8, int8, uint16, int16, int32, int64, float16, float32, float64, bool}")
    .Attr("input_name: string = 'default_tensor_name'") 
    .Attr("N: int >=1")
    .Input("tensors: N * T")
    .Input("splits: int32")
    .Input("recv_splits: int32")
    .Attr("recv_split_unknown: bool = False")
    .Output("output: N * T")
    .Output("recv_bytes: N * int32") // TODO: rename this output
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      int n = c->num_outputs() / 2;
      for (int i = 0; i < n; ++i) {
        ::tensorflow::shape_inference::ShapeHandle output;
        TF_RETURN_IF_ERROR(c->ReplaceDim(c->input(i), 0, c->UnknownDim(), &output));
        c->set_output(i, output);
        c->set_output(i + n, c->input(n));
      }
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(

Perform an MPI Alltoall on a group of tensors.
Arguments
    tensors:    A group of tensors to be distributed with all to all  // for send counts (dim0)
    splits:     A list of integers in rank order describing how many elements
                in `tensor` to send to each worker.  // for recv counts (dim0)
    recv_split_unknown: A bool to indicate whether recv splits is unknown  
Output
    output:    The collected tensor data from all workers.
)doc");

REGISTER_KERNEL_BUILDER(Name("BytepsAlltoallCputogpu").Device(::tensorflow::DEVICE_GPU)
                                                      .HostMemory("tensor")
                                                      .HostMemory("splits")
                                                      .HostMemory("recv_splits")
                                                      .HostMemory("recv_bytes"),
                        BytepsAllToAllOp<true>);

REGISTER_OP("BytepsAlltoallCputogpu")
    .Attr(
        "T: {uint8, int8, uint16, int16, int32, int64, float16, float32, float64, bool}")
    .Attr("input_name: string = 'default_tensor_name'") 
    .Input("tensor: T")
    .Input("splits: int32")
    .Input("recv_splits: int32")
    .Attr("recv_split_unknown: bool = False")
    .Output("output: T")
    .Output("recv_bytes: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle output;
      TF_RETURN_IF_ERROR(c->ReplaceDim(c->input(0), 0, c->UnknownDim(), &output));
      c->set_output(0, output);
      c->set_output(1, c->input(1));
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(

Perform an MPI Alltoall on a tensor from CPU to GPU.
Arguments
    tensor:     A tensor to be distributed with all to all  // for send counts (dim0)
    splits:     A list of integers in rank order describing how many elements
                in `tensor` to send to each worker.  // for recv counts (dim0)
    recv_split_unknown: A bool to indicate whether recv splits is unknown  
Output
    output:    The collected tensor data from all workers.
)doc");


REGISTER_KERNEL_BUILDER(Name("BytepsAlltoallGputocpu").Device(::tensorflow::DEVICE_CPU)
                                                      .HostMemory("splits")
                                                      .HostMemory("recv_splits")
                                                      .HostMemory("recv_bytes")
                                                      .HostMemory("output"),
                        BytepsAllToAllOp<true>);

REGISTER_OP("BytepsAlltoallGputocpu")
    .Attr(
        "T: {uint8, int8, uint16, int16, int32, int64, float16, float32, float64, bool}")
    .Attr("input_name: string = 'default_tensor_name'") 
    .Input("tensor: T")
    .Input("splits: int32")
    .Input("recv_splits: int32")
    .Attr("recv_split_unknown: bool = False")
    .Output("output: T")
    .Output("recv_bytes: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle output;
      TF_RETURN_IF_ERROR(c->ReplaceDim(c->input(0), 0, c->UnknownDim(), &output));
      c->set_output(0, output);
      c->set_output(1, c->input(1));
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(

Perform an MPI Alltoall on a tensor from GPU to CPU.
Arguments
    tensor:     A tensor to be distributed with all to all  // for send counts (dim0)
    splits:     A list of integers in rank order describing how many elements
                in `tensor` to send to each worker.  // for recv counts (dim0)
    recv_split_unknown: A bool to indicate whether recv splits is unknown  
Output
    output:    The collected tensor data from all workers.
)doc");


REGISTER_OP("BytepsSend")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Input("tensor: T")
    .Attr("sender: int")
    .Attr("receiver: int")
    .Attr("input_name: string = 'default_tensor_name'")
    .Attr("version: int")
    .Attr("priority: int")
    .Doc(R"doc(
Perform an send on a tensor. 
Arguments
    tensor:     A tensor to reduce.
Output
    sum:    A tensor with the same shape as `tensor`, summed across all processes.
)doc");

REGISTER_OP("BytepsRecv")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Input("tensor: T")
    .Attr("sender: int")
    .Attr("receiver: int")
    .Attr("input_name: string = 'default_tensor_name'")
    .Attr("version: int")
    .Attr("priority: int")
    .Doc(R"doc(
Perform an recv on a tensor. 
Arguments
    tensor:     A tensor to reduce.
Output
    sum:    A tensor with the same shape as `tensor`, summed across all processes.
)doc");

}  // namespace tensorflow
}  // namespace byteps
