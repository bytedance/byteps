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
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>

#include "ops.h"
#include "../common/logging.h"
#include "../common/global.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"

#include "tensorflow/compiler/tf2xla/kernels/tensor_list_utils.h"

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_cudamalloc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"

using namespace byteps;

namespace byteps {
namespace tensorflow {

std::unordered_map<std::string, std::shared_ptr<Xla_done_cb_args>> _name_to_done_args;
std::mutex _name_to_done_args_mtx;
std::condition_variable _name_to_done_args_cv;

namespace {

int printMatOnGPU(std::string &name, const void *buffer, int num_elem) {
  std::ofstream outfile;
  int my_rank = common::byteps_rank();
  outfile.open("output-" + std::to_string(my_rank), std::ios_base::out | std::ios_base::app);
  outfile << "tensor_name: " << name << std::endl;
  outfile << "tensor_ptr: " << buffer << std::endl;

  float *host_buf = nullptr;
  host_buf = (float *)malloc(num_elem * sizeof(float));
  cudaMemcpy((void *) host_buf, buffer, sizeof(float) * num_elem, cudaMemcpyDeviceToHost);
  for (int i = 0; i < num_elem; i++) {
      outfile << *(host_buf + i) << ", ";
  }
  outfile << std::endl;
  free(host_buf);

  return 0;
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

XlaReadyEvent::XlaReadyEvent(cudaStream_t stream) {
  cudaEventCreateWithFlags(
          &cuda_event_, cudaEventBlockingSync | cudaEventDisableTiming);
  cudaEventRecord(cuda_event_, stream);
}

bool XlaReadyEvent::Ready() const {
  auto status = cudaEventQuery(cuda_event_);
  if (status == cudaErrorNotReady) {
    return false;
  }
  return true;
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

XlaTensor::XlaTensor(void *data, int64_t num_elem,
                         ::tensorflow::DataType tf_dtype, int64_t size) {
  _data = data;
  _num_elem = num_elem;
  _tf_dtype = tf_dtype;
  _size = size;
}

const common::DataType XlaTensor::dtype() const {
  return ConvertDType(_tf_dtype);
}

const common::TensorShape XlaTensor::shape() const {
  common::TensorShape shape;
  shape.AddDim(_num_elem);
  return shape;
}

const void* XlaTensor::data() const {
  return (const void*)_data;
}

int64_t XlaTensor::size() const { return _size; }


// On GPU this event will signal that data is ready, and tensors are
// allocated.
common::ReadyEvent* RecordReadyEvent(::tensorflow::OpKernelContext* context) {
  auto device_context = context->op_device_context();
  if (device_context != nullptr) {
    return new TFReadyEvent(device_context);
  }
  return nullptr;
}

std::shared_ptr<common::ReadyEvent> RecordReadyEvent(cudaStream_t stream) {
  return std::make_shared<XlaReadyEvent>(stream);
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

////////////////////////////////////////////////////////////////////////////////
class BarrierHandleOutXlaOp : public ::tensorflow::XlaOpKernel {
 public:
  explicit BarrierHandleOutXlaOp(::tensorflow::OpKernelConstruction* ctx) : ::tensorflow::XlaOpKernel(ctx) { }
  void Compile(::tensorflow::XlaOpKernelContext* ctx) override {
    std::vector<xla::XlaOp> values;
    std::vector<::tensorflow::TensorShape> shapes;
    OP_REQUIRES_OK(ctx, ctx->InputList("values", &values, &shapes));

    // std::cout << "my_name is " << name() << std::endl;
    // S32 is int32
    auto out_shapes = xla::ShapeUtil::MakeTupleShape({
      xla::ShapeUtil::MakeShape(xla::S32, {2})
    });
    std::string opaque("xxdummyxx");
    auto results = xla::CustomCall(ctx->builder(),
                           /*call_target_name=*/"customBarrierHandleOutXlaOp",
                           values, out_shapes, opaque);
    auto out_0 = xla::GetTupleElement(results, 0);
    ctx->SetOutput(0, out_0);
  }
 private:
  std::vector<std::string> token_input_nodes_;
};

void
customBarrierHandleOutXlaOp(CUstream stream, void** buffers,
                  const char* opaque, size_t opaque_len) {
  int my_rank = common::byteps_rank();
  int* out_buf = reinterpret_cast<int*>(buffers[1]);
  const int* in_buf = reinterpret_cast<const int*>(buffers[0]);
}

REGISTER_OP("MyBarrierHandleOut")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("N: int >= 1")
    .Input("values: N * T")
    .Output("output: int32");

REGISTER_XLA_OP(Name("MyBarrierHandleOut").Device(::tensorflow::DEVICE_GPU_XLA_JIT),
                BarrierHandleOutXlaOp);
XLA_REGISTER_CUSTOM_CALL_TARGET(customBarrierHandleOutXlaOp, "CUDA");

////////////////////////////////////////////////////////////////////////////////
/**
 * get the buffer size of the i-th input tensor
 */
int get_buf_size(::tensorflow::XlaOpKernelContext* context, int index) {
    const ::tensorflow::TensorShape input_tensor_shape = context->InputShape(index);
    auto xla_tensor_shape_or =
          TensorShapeToXLAShape(context->input_xla_type(index), input_tensor_shape);
    xla::Shape tf_tensor_shape = xla_tensor_shape_or;
    int ret;

    ret = xla::ShapeUtil::ByteSizeOfPrimitiveType(tf_tensor_shape.element_type());
    for (int i = 0; i < tf_tensor_shape.rank(); i++) {
        ret *= tf_tensor_shape.dimensions(i) ;
    }

    return ret;
}

//for debugging

void PrintTensorsCustomOp(CUstream stream, void** buffers,
  const char* opaque, size_t opaque_len)
{
  int num;
  int count = 0;
  std::vector<int> buf_sizes;
  std::string tmp_name;
  std::stringstream ss(opaque);
  int my_rank = common::byteps_rank();

  ss >> num;
  while (ss >> tmp_name) {
    int buf_size;
    ss >> buf_size;
    cudaMemcpyAsync(buffers[count + num], buffers[count], buf_size, cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);
    printMatOnGPU(tmp_name, buffers[count], buf_size/4);
    count++;
  }
}

class BytePSPrintTensorsXlaOp : public ::tensorflow::XlaOpKernel
{
  public:
    explicit BytePSPrintTensorsXlaOp(::tensorflow::OpKernelConstruction* context) : ::tensorflow::XlaOpKernel(context) {
      context->GetAttr("tensor_names", &tensor_names_to_print);
    }
    ~BytePSPrintTensorsXlaOp() override = default;

    void Compile(::tensorflow::XlaOpKernelContext* ctx) override {
      std::vector<xla::XlaOp> values;
      std::vector<xla::XlaOp> valid_values;
      std::vector<::tensorflow::TensorShape> shapes;
      OP_REQUIRES_OK(ctx, ctx->InputList("values", &values, &shapes));
      const int N = values.size();
      std::vector<xla::Shape> tmp_output_shapes;
      for (int i = 0; i < N; i++) {
        const xla::Shape* shape = (ctx->builder()->GetShapePtr(values[i])).ValueOrDie();
        tmp_output_shapes.push_back(*shape);
      }

      auto output_shapes = xla::ShapeUtil::MakeTupleShape(tmp_output_shapes);

      std::stringstream ss;
      ss << N;
      for (int i = 0; i < N; i++) {
          auto& tmp_name = tensor_names_to_print[i];
          int tmp_size = get_buf_size(ctx, i);
          ss << " " << tmp_name;
          ss << " " << tmp_size;
      }
      ss << std::endl;
      xla::XlaOp results = xla::CustomCall(ctx->builder(),
        /*call_target_name=*/"PrintTensorsCustomOp",
        valid_values, output_shapes, ss.str());

      for (int i = 0; i < N; i++) {
        xla::XlaOp tmp_tensor = xla::GetTupleElement(results, i);
        ctx->SetOutput(i, tmp_tensor);
      }
    }

  private:
    std::vector<std::string> tensor_names_to_print;
};

REGISTER_OP("BytepsPrintTensors")
    .Input("values: N * T")
    .Output("sum: N * T")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("N: int >= 1")
    .Attr("tensor_names: list(string)")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::Status::OK();
    });
REGISTER_XLA_OP(Name("BytepsPrintTensors").Device(::tensorflow::DEVICE_GPU_XLA_JIT),
  BytePSPrintTensorsXlaOp);
XLA_REGISTER_CUSTOM_CALL_TARGET(PrintTensorsCustomOp, "CUDA");

//////////////////////////////////////////////////////////////////////////
class BytepsPushPullXlaOpV2 : public ::tensorflow::XlaOpKernel {
  public:
    explicit BytepsPushPullXlaOpV2(::tensorflow::OpKernelConstruction* context) : ::tensorflow::XlaOpKernel(context) {
      context->GetAttr("input_name", &input_tensor_name);
    }
    ~BytepsPushPullXlaOpV2() override = default;

    void Compile(::tensorflow::XlaOpKernelContext* context) override {
      int my_rank = common::byteps_rank();
      OP_REQUIRES_OK(context, ConvertStatus(common::CheckInitialized()));

      xla::XlaOp input_tensor = context->Input(0);
      xla::XlaOp dummy_tensor = context->Input(1);
#if TF_VERSION < 2003000000
      const ::tensorflow::TensorShape input_tensor_shape = context->InputShape(0);
      auto input_tensor_xla_shape_or =
          TensorShapeToXLAShape(context->input_xla_type(0), input_tensor_shape);
      xla::Shape output_tensor_shape = input_tensor_xla_shape_or;
#else
      auto input_tensor_xla_shape_or = context->InputXlaShape(0);
      xla::Shape output_tensor_shape = input_tensor_xla_shape_or.ValueOrDie();
#endif

      auto node_name = name();
      std::string tmp_name;
      if (input_tensor_name == "default_tensor_name") {
        tmp_name = node_name;
      } else {
        tmp_name = input_tensor_name;
      }

      std::stringstream ss;
      ss << tmp_name;
      ss << " " << context->input_type(0);
      ss << " " << xla::ShapeUtil::ByteSizeOfPrimitiveType(output_tensor_shape.element_type());
      ss << " " << output_tensor_shape.rank();
      for (int i = 0; i < output_tensor_shape.rank(); i++) {
        ss << " " << output_tensor_shape.dimensions(i) ;
      }
      ss << std::endl;
      auto output_shape_tuple = xla::ShapeUtil::MakeTupleShape({
        // output_tensor_shape, // output tensor
        xla::ShapeUtil::MakeShape(xla::S32, {2}) // handle
      });
      // auto output_shape = xla::ShapeUtil::MakeShape(xla::S32, {2});
      auto results = xla::CustomCall(context->builder(),
          /*call_target_name=*/"StartTaskWrapperV2",
          {input_tensor}, output_shape_tuple, ss.str());

      // auto out_0 = xla::GetTupleElement(results, 0);
      // context->SetOutput(0, out_0);
      context->op_kernel_context()->set_output(
        0, context->op_kernel_context()->input(0));
      auto out_1 = xla::GetTupleElement(results, 0);
      context->SetOutput(1, out_1);
    }

  private:
     std::string input_tensor_name;
};

void StartTaskXlaV2(::tensorflow::OpKernelContext* context,
               std::string node_name, std::shared_ptr<common::Tensor> byteps_input,
               std::shared_ptr<common::Tensor> byteps_output,
               std::shared_ptr<common::ReadyEvent> ready_event) {
  int my_rank =  common::byteps_rank();

  auto& byteps_context = common::GetContextFromName(node_name);
  int device;
  CUDA_CALL(cudaGetDevice(&device));
  auto size = byteps_input->size();
  auto dtype = byteps_input->dtype();
  void* cpubuff = nullptr;
  common::InitTensor(byteps_context, size, dtype, cpubuff);

  auto queue_list = common::GetPushQueueList(device);
  auto queue_list_pull = common::GetPullQueueList(device);
  queue_list->insert(queue_list->end(), queue_list_pull->begin(),
                     queue_list_pull->end());

  // TODO: assign priority based on topological sort
  std::string name_key(node_name);
  std::replace(name_key.begin(), name_key.end(), '/', '_');

  auto enqueue_result =
      EnqueueTensor(byteps_context, byteps_input, byteps_output, ready_event,
                    device, -byteps_context.declared_key, 0,
                    [name_key, my_rank](const common::Status& status) {
                      std::unique_lock<std::mutex> my_lk(_name_to_done_args_mtx);
                      auto it = _name_to_done_args.find(name_key);
                      ASSERTF(it != _name_to_done_args.end(), "YOU SHOULD NOT SEE ME");
                      auto args = _name_to_done_args[name_key];
                      my_lk.unlock();
                      {
                        std::lock_guard<std::mutex> lk(args->mtx);
                        args->is_done = true;
                      }

                      args->cv.notify_one();
                    },
                    queue_list);
  if (ConvertStatus(enqueue_result) != ::tensorflow::Status::OK()) {
    assert(0);
  }
}

void StartTaskWrapperV2(CUstream stream, void** buffers,
                      const char* opaque, size_t opaque_len) {
  int my_rank = common::byteps_rank();
  std::stringstream ss(opaque);
  std::string tmp_name;
  ::tensorflow::OpKernelContext* context = nullptr;

  ss >> tmp_name;
  ::tensorflow::DataType dt_type;
  int tmp_dt_type;
  ss >> std::dec >> tmp_dt_type;
  dt_type = static_cast<::tensorflow::DataType>(tmp_dt_type);
  size_t elem_size;
  ss >> elem_size;
  int ndim = 0;
  ss >> std::dec >> ndim;
  size_t buffer_size = 0;
  size_t num_elem = 1;
  for (int i = 0; i < ndim; i++) {
    size_t dim;
    ss >> std::dec >> dim;
    num_elem *= dim;
  }

  buffer_size = elem_size * num_elem;

  auto bps_input = std::make_shared<XlaTensor>(buffers[0], num_elem, dt_type, buffer_size);
  auto my_stream = byteps::common::BytePSGlobal::GetCopyDevice2DeviceStream();
  auto gpu_allocator = ::tensorflow::GPUProcessState::singleton()->GetGPUAllocator(
    ::tensorflow::GPUOptions(), ::tensorflow::TfGpuId(0), buffer_size);


  ////////
  std::string name_key(tmp_name);
  std::replace(name_key.begin(), name_key.end(), '/', '_');

  std::unique_lock<std::mutex> my_lk(_name_to_done_args_mtx);
  auto it = _name_to_done_args.find(name_key);
  // ASSERTF(it == _name_to_done_args.end(), std::string("duplicate tensor_name ") +
  //   std::string(name_key));
  std::shared_ptr<::tensorflow::Tensor> myTensor;
  void *myTensor_flat = nullptr;
  if (it != _name_to_done_args.end()) {
    // myTensor = it->second->output_tensor;
    BPS_LOG(DEBUG, my_rank) << " x2682 name_key exists: " << name_key <<
      " data_ptr: " << it->second->bps_in_buf << std::flush;
  } else {
    // myTensor = std::make_shared<::tensorflow::Tensor>(
    //   gpu_allocator, dt_type, ::tensorflow::TensorShape({num_elem}));
    // void *myTensor_flat = const_cast<void *>((const void *)(myTensor.get()->tensor_data().data()));

    cudaMalloc(&myTensor_flat, buffer_size);

    std::shared_ptr<Xla_done_cb_args> new_args(new Xla_done_cb_args);
    new_args->is_done = false;
    new_args->bps_out_buf = myTensor_flat;
    new_args->bps_in_buf = myTensor_flat;
    new_args->bps_buf_size = buffer_size;
    // new_args->output_tensor = myTensor;

    _name_to_done_args[name_key] = new_args;
    BPS_LOG(DEBUG, my_rank) << " x2682 inserting name_key " << name_key <<
      " data_ptr: " << new_args->bps_in_buf << std::flush;
  }
  my_lk.unlock();
  BPS_LOG(DEBUG, my_rank) << " x2682 name_key pos 1" << name_key << std::flush;
  _name_to_done_args_cv.notify_all();
  // void *myTensor_flat = const_cast<void *>((const void *)(myTensor.get()->tensor_data().data()));
  BPS_LOG(DEBUG, my_rank) << " x2682 name_key pos 2" << name_key << std::flush;
  cudaStreamSynchronize(*my_stream);
  cudaMemcpyAsync(myTensor_flat, buffers[0], buffer_size, cudaMemcpyDeviceToDevice, *my_stream);
  cudaStreamSynchronize(*my_stream);
  BPS_LOG(DEBUG, my_rank) << " x2682 name_key pos 3" << name_key << std::flush;
  auto bps_output = std::make_shared<XlaTensor>(myTensor_flat, num_elem, dt_type, buffer_size);
  ////////
  auto ready_event =
    std::shared_ptr<common::ReadyEvent>(RecordReadyEvent(*my_stream));

  ////////
  // return;

  std::thread t(StartTaskXlaV2, context, tmp_name, bps_output, bps_output, ready_event);
  t.detach();
  BPS_LOG(DEBUG, my_rank) << " x2682 exit " << __func__ << std::flush;
}

REGISTER_OP("BytepsPushPullXlaV2")
  .Attr("T: {int32, int64, float16, float32, float64}")
  .Attr("input_name: string = 'default_tensor_name'")
  .Input("tensor: T")
  .Input("input_dummy: int32") // force xla compilation
  .Output("sum: T")
  .Output("out_handle: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return ::tensorflow::Status::OK();
  });

REGISTER_XLA_OP(Name("BytepsPushPullXlaV2").Device(::tensorflow::DEVICE_GPU_XLA_JIT),
    BytepsPushPullXlaOpV2);

XLA_REGISTER_CUSTOM_CALL_TARGET(StartTaskWrapperV2, "CUDA");

////////////////////////////////////////////////////////////////////////////////

class BytepsSyncTensorHandleOutXlaOpV2 : public ::tensorflow::XlaOpKernel {
  public:
    explicit BytepsSyncTensorHandleOutXlaOpV2(::tensorflow::OpKernelConstruction* context) : ::tensorflow::XlaOpKernel(context) {
      context->GetAttr("tensor_name", &input_tensor_name);
    }
    ~BytepsSyncTensorHandleOutXlaOpV2() override = default;

    void Compile(::tensorflow::XlaOpKernelContext* context) override {
      OP_REQUIRES_OK(context, ConvertStatus(common::CheckInitialized()));
      xla::XlaOp input_tensor = context->Input(0);
      xla::XlaOp input_handle = context->Input(1);
      xla::XlaOp dummy_tensor = context->Input(2);

#if TF_VERSION < 2003000000
      const ::tensorflow::TensorShape input_tensor_shape = context->InputShape(0);
      auto input_tensor_xla_shape_or =
          TensorShapeToXLAShape(context->input_xla_type(0), input_tensor_shape);

      const ::tensorflow::TensorShape input_handle_shape = context->InputShape(1);
      auto input_handle_xla_shape_or =
          TensorShapeToXLAShape(context->input_xla_type(1), input_handle_shape);
#else
      auto input_tensor_xla_shape_or = context->InputXlaShape(0);
#endif
      auto output_tensor_shape = input_tensor_xla_shape_or;

      auto node_name = name();
      std::string tmp_name;
      if (input_tensor_name == "default_tensor_name") {
        tmp_name = node_name;
      } else {
        tmp_name = input_tensor_name;
      }

      std::stringstream ss;
      ss << tmp_name;
      ss << " " << context->input_type(1);
      ss << " " <<
        xla::ShapeUtil::ByteSizeOfPrimitiveType(input_tensor_xla_shape_or.element_type());
      ss << " " << input_tensor_xla_shape_or.rank();
      for (int i = 0; i < input_tensor_xla_shape_or.rank(); i++) {
        ss << " " << input_tensor_xla_shape_or.dimensions(i) ;
      }
      ss << std::endl;

      auto output_shape_tuple = xla::ShapeUtil::MakeTupleShape({
        output_tensor_shape, // output tensor
      });

      auto results = xla::CustomCall(context->builder(),
          /*call_target_name=*/"SyncTensorHandleOutCustomOpV2",
          {input_handle}, output_shape_tuple, ss.str());
          // {input_tensor, input_handle}, output_shape_tuple, ss.str());

      auto out_0 = xla::GetTupleElement(results, 0);
      context->SetOutput(0, out_0);
    }

  private:
     std::string input_tensor_name;
};

void SyncTensorHandleOutCustomOpV2(CUstream stream, void** buffers,
                      const char* opaque, size_t opaque_len) {
  // return;
  int my_rank = common::byteps_rank();
  std::string tmp_name;
  std::stringstream ss(opaque);
// return;
  ss >> tmp_name;
  int tmp_dt_type;
  ss >> std::dec >> tmp_dt_type;
  size_t elem_size;
  ss >> elem_size;
  int ndim = 0;
  ss >> std::dec >> ndim;
  size_t buffer_size = 0;
  size_t num_elem = 1;
  for (int i = 0; i < ndim; i++) {
    size_t dim;
    ss >> std::dec >> dim;
    num_elem *= dim;
  }
  buffer_size = elem_size * num_elem;
  std::unique_lock<std::mutex> my_big_lk(_name_to_done_args_mtx);
  _name_to_done_args_cv.wait(my_big_lk,
    [&tmp_name]{
      std::this_thread::yield();
      return _name_to_done_args.find(tmp_name) != _name_to_done_args.end();
    });

  auto it = _name_to_done_args.find(tmp_name);
  ASSERTF(it != _name_to_done_args.end(), "post 2");
  auto args = _name_to_done_args[tmp_name];
  my_big_lk.unlock();

    // auto expecting = args->bps_in_buf;
    // auto got_ptr = buffers[0];
    // if (got_ptr != expecting) {
    //   BPS_LOG(DEBUG, my_rank) << " x2682_error expecting ptr: " << expecting
    //     << " but got ptr: " << got_ptr << " name_key: " << tmp_name
    //     << std::flush;
    // } else {
    //   BPS_LOG(DEBUG, my_rank) << " x2682_correct expecting ptr: " << expecting
    //     << " got ptr: " << got_ptr << " name_key: " << tmp_name
    //     << std::flush;
    // }

  BPS_LOG(DEBUG, my_rank) << " x2682 " << __func__ <<
      " got name_key: " << tmp_name << std::flush;
  // return;
  {
    std::unique_lock<std::mutex> lk(args->mtx);
    args->cv.wait(lk, [args]{
      std::this_thread::yield();
      return args->is_done;});
    // args->num_waiting--;
    // if (args->num_waiting == 0) {
    //   args->is_done = false;
    // }
    lk.unlock();

    auto my_stream = byteps::common::BytePSGlobal::GetCopyDevice2DeviceStream();

    // void *outputtensor_flat = const_cast<void *>((const void *)(
    //   args->output_tensor.get()->tensor_data().data()));

    void *outputtensor_flat = args->bps_in_buf;

    // cudaMemcpyAsync(buffers[1], outputtensor_flat, buffer_size, cudaMemcpyDeviceToDevice, *my_stream);
    // cudaStreamSynchronize(*my_stream);
    cudaMemcpyAsync(buffers[1], outputtensor_flat, buffer_size, cudaMemcpyDeviceToDevice, stream);
    int myDev = -1;
    cudaGetDevice(&myDev);
    BPS_LOG(DEBUG, my_rank) << " x2682 " << __func__ << " myDev: " << myDev <<
      std::flush;
    BPS_LOG(DEBUG, my_rank) << " x2682 " << __func__ <<
      " name_key: " << tmp_name << " data_ptr: " << outputtensor_flat << std::flush;

    cudaStreamSynchronize(stream);
    printMatOnGPU(tmp_name, outputtensor_flat, 1);
    // args->output_tensor.reset();
    cudaFree(outputtensor_flat);
  }
  std::unique_lock<std::mutex> my_lk(_name_to_done_args_mtx);
  // _name_to_done_args[tmp_name]->is_done = false;
  _name_to_done_args.erase(tmp_name);
  my_lk.unlock();
}

REGISTER_OP("BytepsSyncTensorHandleOutV2")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("tensor_name: string = 'default_tensor_name'")
    .Input("in_tensor: T")
    .Input("in_handle: int32")
    .Input("input_dummy: int32") // force xla compilation
    .Output("out_tensor: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::Status::OK();
    });
REGISTER_XLA_OP(Name("BytepsSyncTensorHandleOutV2").Device(::tensorflow::DEVICE_GPU_XLA_JIT),
  BytepsSyncTensorHandleOutXlaOpV2);
XLA_REGISTER_CUSTOM_CALL_TARGET(SyncTensorHandleOutCustomOpV2, "CUDA");
/////////////////////////////////////////////////////

}  // namespace tensorflow
}  // namespace byteps
