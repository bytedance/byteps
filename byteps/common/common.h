// Copyright 2019 ByteDance, Inc. All Rights Reserved.
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

#ifndef BYTEPS_COMMON_H
#define BYTEPS_COMMON_H

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace byteps {
namespace common {

// Device ID used for CPU.
#define CPU_DEVICE_ID (-1)

enum DataType {
  BYTEPS_UINT8 = 0,
  BYTEPS_INT8 = 1,
  BYTEPS_UINT16 = 2,
  BYTEPS_INT16 = 3,
  BYTEPS_INT32 = 4,
  BYTEPS_INT64 = 5,
  BYTEPS_FLOAT16 = 6,
  BYTEPS_FLOAT32 = 7,
  BYTEPS_FLOAT64 = 8,
  BYTEPS_BOOL = 9,
  BYTEPS_BYTE = 10,
};

// List of supported frameworks.
enum Framework { TENSORFLOW, PYTORCH, MXNET };

enum StatusType { OK, UNKNOWN_ERROR, PRECONDITION_ERROR, ABORTED, INVALID_ARGUMENT, IN_PROGRESS };

enum DeviceType { CPU, GPU };

class Status {
public:
  Status();
  static Status OK();
  static Status UnknownError(std::string message);
  static Status PreconditionError(std::string message);
  static Status Aborted(std::string message);
  static Status InvalidArgument(std::string message);
  static Status InProgress();
  bool ok() const;
  bool in_progress() const;
  StatusType type() const;
  const std::string& reason() const;

private:
  StatusType type_ = StatusType::OK;
  std::string reason_ = "";
  Status(StatusType type, std::string reason);
};

class TensorShape {
public:
  void AddDim(int64_t dim);
  void AppendShape(TensorShape& other);

  const std::string DebugString() const;
  int dims() const;
  int64_t dim_size(int idx) const;
  int64_t num_elements() const;

  inline bool operator==(const TensorShape& rhs) const {
    return shape_ == rhs.shape_;
  }

  inline bool operator!=(const TensorShape& rhs) const {
    return shape_ != rhs.shape_;
  }

private:
  std::vector<int64_t> shape_;
};

class ReadyEvent {
public:
  virtual bool Ready() const = 0;
  virtual ~ReadyEvent() = default;
};

class OpContext;

class PersistentBuffer {
public:
  virtual const void* AccessData(std::shared_ptr<OpContext> context) const = 0;
  virtual ~PersistentBuffer() = default;
};

class Tensor {
public:
  virtual const DataType dtype() const = 0;
  virtual const TensorShape shape() const = 0;
  virtual const void* data() const = 0;
  virtual int64_t size() const = 0;
  virtual ~Tensor() = default;
};

class OpContext {
public:
  // These allocators are fully synchronous, unlike TensorFlow counterparts.
  virtual Status
  AllocatePersistent(int64_t size,
                     std::shared_ptr<PersistentBuffer>* tensor) = 0;
  virtual Status AllocateOutput(TensorShape shape,
                                std::shared_ptr<Tensor>* tensor) = 0;
  virtual Framework framework() const = 0;
  virtual ~OpContext() = default;
};

// A callback to call after the MPI communication completes. Since the
// allreduce and allgather ops are asynchronous, this callback is what resumes
// computation after the reduction is completed.
using StatusCallback = std::function<void(const Status&)>;

// Table storing Tensors to be reduced, keyed by unique name.
// This table contains everything necessary to do the reduction.
struct TensorTableEntry {
  // Name of the tensor.
  std::string tensor_name;
  // Operation context.
  std::shared_ptr<OpContext> context;
  // Input tensor.
  std::shared_ptr<Tensor> tensor;
  // Pre-allocated output tensor.
  std::shared_ptr<Tensor> output;
  // Root rank for broadcast operation.
  int root_rank = 0;
  // Event indicating that data is ready.
  std::shared_ptr<ReadyEvent> ready_event;
  // GPU to do reduction on, or CPU_DEVICE_ID in case of CPU.
  int device = CPU_DEVICE_ID;
  // A callback to call with the status.
  StatusCallback callback;
};
using TensorTable = std::unordered_map<std::string, TensorTableEntry>;

} // namespace common
} // namespace byteps

#endif // BYTEPS_COMMON_H
