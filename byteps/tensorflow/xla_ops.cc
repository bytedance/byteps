// Copyright 2021 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2021, NVIDIA CORPORATION. All rights reserved.
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

#if TENSORFLOW_VERSION >= 2006000000

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/human_readable_json.h"

#if BYTEPS_BUILDING_CUDA
#if HAVE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

#include "../common/operations.h"
#include "../common/utils.h"
#include "./custom_call_config_generated.h"

using namespace tensorflow;

namespace byteps {
namespace xla {
namespace {
using gpuEvent_t = cudaEvent_t;

common::DataType GetBPSType(::xla::PrimitiveType type) {
  switch (type) {
  case ::xla::U8:
    return common::BYTEPS_UINT8;
  case ::xla::S8:
    return common::BYTEPS_INT8;
  case ::xla::U16:
    return common::BYTEPS_UINT16;
  case ::xla::S16:
    return common::BYTEPS_INT16;
  case ::xla::S32:
    return common::BYTEPS_INT32;
  case ::xla::S64:
    return common::BYTEPS_INT64;
  case ::xla::F16:
    return common::BYTEPS_FLOAT16;
  case ::xla::F32:
    return common::BYTEPS_FLOAT32;
  case ::xla::F64:
    return common::BYTEPS_FLOAT64;
  case ::xla::PRED:
    return common::BYTEPS_BOOL;
  default:
    throw std::logic_error("Invalid XLA tensor type.");
  }
}

// CustomCallConfig stores configurations of byteps ops. We pass this config
// to ::xla::CustomCall so that the XLA CustomCall can represent various byteps
// ops. Flatbuffer is used to serialize the config into string to conform to the
// XLA CustomCall interface.
class CustomCallConfig {
public:
  std::string SerializeToString();
  void ParseFromString(std::string);

public:
  std::string tensor_name_;
  common::DataType tensor_type_;
  std::vector<std::vector<int64_t>> input_shapes_;
  std::vector<std::vector<int64_t>> output_shapes_;
  int reduce_op_;
};

std::string CustomCallConfig::SerializeToString() {
  flatbuffers::FlatBufferBuilder fbb(1024);

  std::vector<flatbuffers::Offset<wire::TensorShape>> input_shapes_obj;
  absl::c_for_each(input_shapes_, [&](const std::vector<int64_t>& dims) {
    input_shapes_obj.push_back(wire::CreateTensorShapeDirect(fbb, &dims));
  });
  std::vector<flatbuffers::Offset<wire::TensorShape>> output_shapes_obj;
  absl::c_for_each(output_shapes_, [&](const std::vector<int64_t>& dims) {
    output_shapes_obj.push_back(wire::CreateTensorShapeDirect(fbb, &dims));
  });
  auto wire = wire::CreateCustomCallConfigDirect(
      fbb, tensor_name_.c_str(), (common::wire::DataType)tensor_type_,
      &input_shapes_obj, &output_shapes_obj, reduce_op_);
  fbb.Finish(wire);

  uint8_t* buf = fbb.GetBufferPointer();
  auto size = fbb.GetSize();
  return std::string((char*)buf, size);
}

void CustomCallConfig::ParseFromString(std::string input) {
  const wire::CustomCallConfig* obj =
      flatbuffers::GetRoot<wire::CustomCallConfig>(
          (const uint8_t*)input.data());

  tensor_name_ = obj->tensor_name()->str();
  tensor_type_ = (common::DataType)obj->tensor_type();
  for (auto it = obj->input_shapes()->begin(); it != obj->input_shapes()->end();
       it++) {
    auto shape_obj = *it;
    input_shapes_.push_back(std::vector<int64_t>(shape_obj->dims()->begin(),
                                                 shape_obj->dims()->end()));
  }
  for (auto it = obj->output_shapes()->begin();
       it != obj->output_shapes()->end(); it++) {
    auto shape_obj = *it;
    output_shapes_.push_back(std::vector<int64_t>(shape_obj->dims()->begin(),
                                                  shape_obj->dims()->end()));
  }
  reduce_op_ = obj->reduce_op();

  if (VLOG_IS_ON(2)) {
    VLOG(2) << "tensor_name " << tensor_name_;
    VLOG(2) << "tensor_type " << tensor_type_;
    VLOG(2) << "reduce_op = " << reduce_op_;
  }
}

// BPSPushPullOp is an XLAOpKernel that lowers the Tensorflow BytepsPushPull
// op into XLA HLOs. The overall idea is to lower an Tensorflow op into two
// corresponding HLO custom-calls, `start` and `end` calls, so that the XLA can
// asynchronously interact with the Byteps runtime. The `start` call is always
// non-blocking for latency hiding and the `end` call could be blocking. For
// example, as shown in BPSPushPullOp::Compile() below, the "BytepsPushPull"
// op is lowered into the "CallbackBPSPushPull" and "CallbackBPSPushPullDone"
// HLO custom-calls, whose implementations are also provided through dynamic
// registration in this file.
class BPSPushPullOp : public XlaOpKernel {
public:
  explicit BPSPushPullOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("input_name", &input_tensor_name));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("op", &op));
    if (op == std::string("average")) {
      reduce_op_ = common::REDUCE_OP_AVERAGE;
    } else if (op == std::string("sum")) {
      reduce_op_ = common::REDUCE_OP_SUM;
    } else {
      throw std::logic_error("operation type.");
    }
  }

  void Compile(XlaOpKernelContext* ctx) override {
    node_name_ = name();
    if (ignore_name_scope_) {
      auto pos = node_name_.find_last_of('/');
      if (pos != std::string::npos) {
        node_name_ = node_name_.substr(pos + 1);
      }
    }

    if (input_tensor_name == "default_tensor_name") {
      tmp_name = node_name_;
    } else {
      tmp_name = input_tensor_name;
    }

    // Generate below HLOs:
    //     start = custom-call(in), custom_call_target="CallbackBPSPushPull"
    //     end = custom-call(start),
    //         custom_call_target="CallbackBPSPushPullDone"
    // Note that tensors `in`, `start`, and `end'` are aliased, as we want the
    // all-reduce operation to be in-place.
    ::xla::XlaBuilder* const b = ctx->builder();
    // First, generate BPSPushPull.
    std::vector<
        std::pair<::xla::ShapeIndex, std::pair<int64, ::xla::ShapeIndex>>>
        output_operand_aliasing = {
            {::xla::ShapeIndex{}, {0, ::xla::ShapeIndex{}}}};
    ::xla::XlaOp input = ctx->Input(0);
    ::xla::XlaOp pushpull_start = b->ReportErrorOrReturn(
        BuildPushPullCustomCall(b, {input}, /*is_start=*/true));
    // Then, generate BPSPushPullDone.
    ::xla::XlaOp pushpull_end = b->ReportErrorOrReturn(
        BuildPushPullCustomCall(b, {pushpull_start},
                                 /*is_start=*/false, output_operand_aliasing));
    ctx->SetOutput(0, pushpull_end);
    return;
  }

private:
  ::xla::StatusOr<::xla::XlaOp> BuildPushPullCustomCall(
      ::xla::XlaBuilder* b, absl::Span<const ::xla::XlaOp> operands,
      bool is_start,
      absl::Span<const std::pair<::xla::ShapeIndex,
                                 std::pair<int64, ::xla::ShapeIndex>>>
          output_operand_aliasing = {});

private:
  std::string node_name_;
  std::string tmp_name;
  std::string input_tensor_name;
  int reduce_op_;
  std::string op;
  // Using float since TF does not support double OP attributes
  bool ignore_name_scope_ = false;
};

// Implements a customized registrar so that the registration is an opt-in,
// controlled by BYTEPS_ENABLE_XLA_OPS.
#define BPS_REGISTER_XLA_OP(NAME, OP)                                          \
  BPS_REGISTER_XLA_OP_UNIQ_HELPER(__COUNTER__, NAME, OP)

#define BPS_REGISTER_XLA_OP_UNIQ_HELPER(COUNTER, OP_NAME, OP)                  \
  BPS_REGISTER_XLA_OP_UNIQ(COUNTER, OP_NAME, OP)

#define BPS_REGISTER_XLA_OP_UNIQ(CTR, OP_NAME, OP)                             \
  static BPSXlaOpRegistrar xla_op_registrar__body__##CTR##__object(            \
      OP_NAME, [](::tensorflow::OpKernelConstruction* context)                 \
                   -> ::tensorflow::OpKernel* { return new OP(context); });

class BPSXlaOpRegistrar {
public:
  BPSXlaOpRegistrar(string op_name,
                    ::tensorflow::XlaOpRegistry::Factory factory) {
    bool enable_xla_ops = !!common::ParseEnv(BYTEPS_ENABLE_XLA_OPS, 1);
    if (enable_xla_ops) {
      xla_op_registrar_ = new XlaOpRegistrar(
          ::tensorflow::XlaOpRegistrationBuilder::Name(op_name).Build(factory));
      std::cout << "Registered XLA OP: BytepsPushPull" << std::endl;
    }
  }

private:
  XlaOpRegistrar* xla_op_registrar_;
};

BPS_REGISTER_XLA_OP("BytepsPushPull", BPSPushPullOp);

// A helper function to build HLOs for all-reduce.
::xla::StatusOr<::xla::XlaOp> BPSPushPullOp::BuildPushPullCustomCall(
    ::xla::XlaBuilder* b, absl::Span<const ::xla::XlaOp> operands,
    bool is_start,
    absl::Span<
        const std::pair<::xla::ShapeIndex, std::pair<int64, ::xla::ShapeIndex>>>
        output_operand_aliasing) {
  string call_target_name =
      is_start ? "CallbackBPSPushPull" : "CallbackBPSPushPullDone";
  CustomCallConfig config;
  config.tensor_name_ = tmp_name;
  for (const ::xla::XlaOp& opnd : operands) {
    TF_ASSIGN_OR_RETURN(::xla::Shape shape, b->GetShape(opnd));
    config.input_shapes_.push_back(std::vector<int64_t>(
        shape.dimensions().begin(), shape.dimensions().end()));
  }
  TF_ASSIGN_OR_RETURN(::xla::Shape output_shape, b->GetShape(operands.at(0)));
  config.output_shapes_.push_back(std::vector<int64_t>(
      output_shape.dimensions().begin(), output_shape.dimensions().end()));
  config.tensor_type_ = GetBPSType(output_shape.element_type());
  config.reduce_op_ = reduce_op_;

  return ::xla::CustomCall(
      b, call_target_name, operands, output_shape, config.SerializeToString(),
      /*has_side_effect=*/false, output_operand_aliasing, /*literal=*/nullptr,
      // Special schedule hints are given so that XLA knows how to schedule
      // the opague custom-calls for performance.
      is_start ? ::xla::CustomCallSchedule::SCHEDULE_EARLIEST
               : ::xla::CustomCallSchedule::SCHEDULE_LATEST);
}

// Returns a hash for rendezvous.
uint64 GetRendezvousKeyHash(const string& key) {
  string k = strings::StrCat(key);
  return Hash64(k.data(), k.size());
}

// Implements a rendezvous to coordinate the `start` and `end` HLO callbacks.
class BPSCustomCallRendezvous {
public:
  struct Payload {
    std::shared_ptr<gpuEvent_t> event;
  };

  // This `Signal` method places payload to be consumed by Wait().
  //
  // Requirement: tensor_name shall be unique in a graph.
  void Signal(string tensor_name, common::Event bps_event) {
    // Use `tensor_name` to generate a hash value to retrieve the queue.
    uint64 key_hash = GetRendezvousKeyHash(tensor_name);
    mutex_lock l(mu_);
    InitQueue(key_hash);

    Queue& queue = *table_[key_hash];
    if (queue.empty() || queue.front() != nullptr) {
      // No earlier waiters are waiting, so simply push a payload in the back.
      queue.push_back(new Payload{bps_event.event});
      return;
    }

    // There is an earlier waiter to consume this signal. Place payload
    // at the front of the queue where the waiter is polling.
    CHECK(nullptr == queue.front());
    queue.front() = new Payload{bps_event.event};
  }

  // The `Wait` method consumes Payloads. We assume there is at most one
  // outstanding `Wait` call due to its blocking nature to simplify the
  // implementation. Consequently, this method always operates on the very
  // first item in the queue.
  void Wait(string tensor_name, CUstream stream) {
    uint64 key_hash = GetRendezvousKeyHash(tensor_name);

    {
      mutex_lock l(mu_);
      InitQueue(key_hash);
      Queue& queue = *table_[key_hash];
      if (queue.empty()) {
        // So long as the queue is empty, place a NULL payload. Then waiting for
        // Signal() to place the payload below.
        queue.push_back(nullptr);
      }
    }

    auto has_available_signal = [&]() {
      mutex_lock l(mu_);
      Queue& queue = *table_[key_hash];
      return nullptr != queue.front();
    };
    while (!has_available_signal()) {
      // Busy waiting. As we don't anticipate the blocking occurs frequently,
      // this busy waiting should be fine. If this creates any performance
      // overhead, we may implement conditional var wait.
      std::this_thread::sleep_for(std::chrono::nanoseconds(100));
    }

    mutex_lock l(mu_);
    Queue* queue = table_[key_hash];
    Payload* payload = queue->front();
    std::shared_ptr<gpuEvent_t> event = payload->event;
    queue->pop_front();
    if (queue->empty()) {
      table_.erase(key_hash);
      delete queue;
    }
    if (event) {
      CUDA_CALL(cudaStreamWaitEvent(stream, *event, /*flags=*/0));
    }
    delete payload;
  }

private:
  // This method is not thread-safe.
  void InitQueue(uint64 key_hash) {
    auto it = table_.find(key_hash);
    if (it == table_.end()) {
      table_[key_hash] = new Queue();
    }
  }

private:
  // `nullptr` denotes non-readiness of the payload.
  typedef std::deque<Payload*> Queue;
  // maps a hash value to queue. We will use tensor_names to generate the hash
  // values.
  typedef absl::flat_hash_map<uint64, Queue*> Table;

  mutex mu_;
  Table table_ GUARDED_BY(mu_);
};

/*static*/ BPSCustomCallRendezvous* GetBPSCustomCallRendezvous() {
  static BPSCustomCallRendezvous* self = new BPSCustomCallRendezvous();
  return self;
}

class XLAReadyEvent : public common::ReadyEvent {
public:
  XLAReadyEvent(cudaStream_t stream) : stream_(stream) {
    CUDA_CALL(cudaEventCreate(&event_));
    CUDA_CALL(cudaEventRecord(event_, stream));
  }
  ~XLAReadyEvent() { CUDA_CALL(cudaEventDestroy(event_)); }

  bool Ready() const override {
    cudaError_t result = cudaEventQuery(event_);
    return cudaErrorNotReady != result;
  }
  gpuEvent_t event() { return event_; }

private:
  cudaStream_t stream_; // Not Owned.
  cudaEvent_t event_;   // Owned.
};

class XLATensor : public common::Tensor {
public:
  XLATensor(common::DataType type, common::TensorShape shape, void* buffer)
      : type_(type), shape_(std::move(shape)), buffer_(buffer) {}

  virtual common::DataType dtype() const override { return type_; }
  virtual const common::TensorShape shape() const override { return shape_; }
  virtual const void* data() const override { return buffer_; }
  virtual int64_t size() const override {
    return shape_.num_elements() * common::DataType_Size(type_);
  }
  virtual void resize(const common::TensorShape&) override;
  virtual int device() const override;

protected:
  common::DataType type_;
  common::TensorShape shape_;
  void* buffer_; // Not owned.
};

void XLATensor::resize(const common::TensorShape& shape) {
}
int XLATensor::device() const {
  return -5;
}

common::ReadyEvent* RecordReadyEvent(cudaStream_t stream) {
  return new XLAReadyEvent(stream);
}

int GetDeviceOrdinal(void* ptr) {
  cudaPointerAttributes attrs;
  CUDA_CALL(cudaPointerGetAttributes(&attrs, ptr));
  return attrs.device;
}

// Implements for the `BPSPushPull` HLO CustomCall.
void CallbackBPSPushPull(CUstream stream, void** buffers, const char* opaque,
                          size_t opaque_len) {
  CHECK(common::CheckInitialized().ok());
  CustomCallConfig config;
  config.ParseFromString(std::string(opaque, opaque_len));

  // Enqueue requests to the Byteps runtime.
  auto ready_event =
        std::shared_ptr<common::ReadyEvent>(RecordReadyEvent(stream));
  int dev_ordinal = GetDeviceOrdinal(buffers[0]);
  auto node_name = config.tensor_name_;
  auto& byteps_context = common::GetContextFromName(node_name);
  auto bps_input = std::make_shared<XLATensor>(
      config.tensor_type_, common::TensorShape(config.input_shapes_[0]),
      buffers[0]);
  auto bps_output = std::make_shared<XLATensor>(
      config.tensor_type_, common::TensorShape(config.input_shapes_[0]),
      buffers[1]);

  auto size = bps_input->size();
  auto dtype = bps_input->dtype();
  void* cpubuff = nullptr;
  common::InitTensor(byteps_context, size, dtype, cpubuff);
  auto queue_list = common::GetPushQueueList(dev_ordinal);
  auto queue_list_pull = common::GetPullQueueList(dev_ordinal);
  queue_list->insert(queue_list->end(), queue_list_pull->begin(),
                     queue_list_pull->end());
  common::Status enqueue_result = EnqueueTensor(
      byteps_context, bps_input, bps_output, ready_event,
      dev_ordinal, -byteps_context.declared_key, 0,
      [=](const common::Status& status) {
        // When request is done processing, signal `BPSPushPullDone`.
        CHECK(status.ok()) << status.reason();
        GetBPSCustomCallRendezvous()->Signal(config.tensor_name_, status.event);
      },
      queue_list,
      (byteps::common::ReduceOp)config.reduce_op_
      );
  CHECK(enqueue_result.ok()) << enqueue_result.reason();
}

// Implements for the `BPSPushPullDone` HLO CustomCall.
void CallbackBPSPushPullDone(CUstream stream, void** /*buffers*/,
                              const char* opaque, size_t opaque_len) {
  // Blocking until the request is done processing by the BytePS runtime.
  VLOG(2) << "bps-pushpull-done - Start";
  CustomCallConfig config;
  config.ParseFromString(std::string(opaque, opaque_len));
  GetBPSCustomCallRendezvous()->Wait(config.tensor_name_, stream);
  VLOG(2) << "bps-pushpull-done - End";
}

XLA_REGISTER_CUSTOM_CALL_TARGET(CallbackBPSPushPull, "CUDA");
XLA_REGISTER_CUSTOM_CALL_TARGET(CallbackBPSPushPullDone, "CUDA");

} // namespace
} // namespace xla
} // namespace byteps

#endif // TENSORFLOW_VERSION >= 2006000000
#endif // HAVE_CUDA
#endif // BYTEPS_BUILDING_CUDA