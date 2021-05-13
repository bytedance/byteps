// Copyright 2019 Bytedance Inc. or its affiliates. All Rights Reserved.
// Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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

#if BYTEPS_BUILDING_CUDA == 1
#include <cuda_runtime.h>
#include <nccl.h>
#endif

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Add for profiling communication events
#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <queue>
#include <thread>
#include <Python.h>
#include <malloc.h>
#include <signal.h>
#include "../server/common.h"

namespace byteps {
namespace common {
namespace compressor {
struct BPSTensor;
typedef BPSTensor tensor_t;
class Compressor;
class ErrorFeedback;
}  // namespace compressor

// Device ID used for CPU.
#define CPU_DEVICE_ID (-1)
#define UNDECIDED_DEVICE_ID (-2)

// Keep the order consistent with DMLC/mshadow
// https://github.com/dmlc/mshadow/blob/master/mshadow/base.h
enum DataType {
  BYTEPS_FLOAT32 = 0,
  BYTEPS_FLOAT64 = 1,
  BYTEPS_FLOAT16 = 2,
  BYTEPS_UINT8 = 3,
  BYTEPS_INT32 = 4,
  BYTEPS_INT8 = 5,
  BYTEPS_INT64 = 6,
  // below are not in mshadow, should avoid using these
  // BYTEPS_UINT16 = 7,
  // BYTEPS_INT16 = 8,
  BYTEPS_BOOL = 9,
  // BYTEPS_BYTE = 10,
};

// List of supported frameworks.
enum Framework { TENSORFLOW, PYTORCH, MXNET };

enum StatusType {
  OK,
  UNKNOWN_ERROR,
  PRECONDITION_ERROR,
  ABORTED,
  INVALID_ARGUMENT,
  IN_PROGRESS
};

enum DeviceType { CPU, GPU };

enum QueueType {
  COORDINATE_REDUCE,
  REDUCE,
  COPYD2H,
  PCIE_REDUCE,
  COORDINATE_PUSH,
  COMPRESS,
  PUSH,
  PULL,
  DECOMPRESS,
  COPYH2D,
  COORDINATE_BROADCAST,
  BROADCAST,
  // for peer-to-peer send
  SEND,
  // for host-to-device copy used in peer-to-peer
  P2P_COPYH2D,
  // for device-to-host copy used in peer-to-peer
  P2P_COPYD2H,
  // for host-to-host/device-to-device copy used in peer-to-peer
  P2P_COPYD2H_SEND,
  // for alltoall recv when the recv split is unknown
  // it waits for the entire group of data before starting to copy
  P2P_GROUP_COPYH2D,
  // for pure CPU allreduce
  CPU_COPY,
  COORDINATE_CPU_REDUCE,
  CPU_REDUCE,
  CPU_REDUCE_FINISH,
  // for pure CPU allreduce
  CPU_BCAST,
  CPU_BCAST_FINISH,
  QUEUE_NUM_AND_NOT_A_REAL_QUEUE_TYPE_AND_MUST_BE_THE_LAST
};

enum OperationType {
  UNKNOWN_OP,
  // push pull (a.k.a all-reduce)
  PUSH_PULL_OP,
  // peer-to-peer operations (send/recv, alltoall)
  P2P_OP,
};

const int QueueNum =
    (int)QUEUE_NUM_AND_NOT_A_REAL_QUEUE_TYPE_AND_MUST_BE_THE_LAST;

const std::vector<std::string> LogStrings = {"COORDINATE_REDUCE",
                                             "REDUCE",
                                             "COPYD2H",
                                             "PCIE_REDUCE",
                                             "COORDINATE_PUSH",
                                             "COMPRESS",
                                             "PUSH",
                                             "PULL",
                                             "DECOMPRESS",
                                             "COPYH2D",
                                             "COORDINATE_BROADCAST",
                                             "BROADCAST",
                                             "SEND",
                                             "P2P_COPYH2D",
                                             "P2P_COPYD2H",
                                             "P2P_COPYD2H_SEND",
                                             "P2P_GROUP_COPYH2D",
                                             "CPU_COPY",
                                             "COORDINATE_CPU_REDUCE",
                                             "CPU_REDUCE",
                                             "CPU_REDUCE_FINISH",
                                             "CPU_BCAST",
                                             "CPU_BCAST_FINISH"};

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

  std::vector<int64_t> shape_;
};

class ReadyEvent {
 public:
  virtual bool Ready() const = 0;
  virtual ~ReadyEvent() = default;
};

// add for profiling
typedef struct CommTime {
  long long start_t;
  long long dur = 0;
  bool end = false;
  int key = -1;
  int type = -1;
} BPSCommTime;

typedef struct BytePSContext {
  bool initialized;
  std::mutex init_mutex;
  // tensor name
  std::string tensor_name;
  // using ps::Key = uint64_t
  uint64_t declared_key;
  // the actual keys being used
  std::vector<uint64_t> key_list;
  // a copy on CPU, backed by shm
  // Note that this buff is optional for p2p operations
  void* cpubuff;
  // GPU ptr if the tensor is on CPU
  // Only used by push_pull operations
  void* gpu_ptr;
  // CPU buffer for cross-PCIe-switch merging
  std::vector<void*> pcie_cpubuff;
  std::vector<void*> numa_cpubuff;
  size_t buff_len;
  // Used for profiling communication events
  std::queue<BPSCommTime*> comm_time;
  bool profile_flag = false;
  int step_cnt = 0;
  int local_rank = 0;
  std::unordered_map<uint64_t, std::unordered_map<int, std::queue<BPSCommTime*>>> part_comm_time;
  // Compressor list
  std::vector<std::shared_ptr<compressor::Compressor>> compressor_list;
  // kwargs
  std::unordered_map<std::string, std::string> kwargs;
  // the rank of the sender for p2p operations.
  int sender = -1;
  // the rank of the receiver for p2p operations.
  int receiver = -1;
  // Used for p2p multiple shm
  std::vector<void*> cpubuff_list;
  // The type of the operation. this field is checked during tensor initialization
  OperationType op_type;
} BPSContext;

class Tensor {
 public:
  virtual const DataType dtype() const = 0;
  virtual const TensorShape shape() const = 0;
  virtual const void* data() const = 0;
  virtual int64_t size() const = 0;
  virtual void resize(const common::TensorShape&) = 0;
  virtual ~Tensor() = default;
};

class TensorView : public Tensor {
 public:
  // A view on [begin, end) on the original tensor.
  // tensor: the original tensor
  // begin: index of the starting element of the flatten tensor
  // end: index of the starting element of the flatten tensor
  TensorView(std::shared_ptr<Tensor> tensor, int begin, int end);

  // the data type of the view
  const DataType dtype() const;
  
  // the shape of the view
  const TensorShape shape() const;

  // the pointer to the view data
  const void* data() const;

  // the size of the view data in bytes
  int64_t size() const;

  // resize the tensor.
  void resize(const common::TensorShape&);

  ~TensorView() = default;

  std::shared_ptr<Tensor> tensor_;
  int begin_;
  int end_;
  int size_;
  void* data_;
  TensorShape shape_;
};

// A callback to call after the PS communication completes.
using StatusCallback = std::function<void(const Status&)>;

// Table storing Tensors to be reduced, keyed by unique name.
// This table contains everything necessary to do the reduction.
struct TensorTableEntry {
  // Name of the tensor.
  std::string tensor_name;
  // Key of the tensor, using ps::Key = uint64_t
  uint64_t key;
  // Operation context.
  BPSContext* context;
  // Input tensor.
  std::shared_ptr<Tensor> tensor;
  // Pre-allocated output tensor.
  std::shared_ptr<Tensor> output;
  // Pre-allocated auxiliary output tensor.
  std::shared_ptr<Tensor> aux_output;
  // Priroity
  int priority = 0;
  // The version of tensor
  int version = 0;
  // Root rank for broadcast operation.
  int root_rank = 0;
  // Event indicating that data is ready.
  std::shared_ptr<ReadyEvent> ready_event;
  // GPU to do reduction on, or CPU_DEVICE_ID in case of CPU.
  int device = CPU_DEVICE_ID;
  // A callback to call with the status.
  StatusCallback callback;
  // CPU buffer address
  void* cpubuff;
  // GPU ptr if the tensor is on CPU
  void* gpu_ptr;
  // CPU buffer for cross-PCIe-switch merging
  std::vector<void*> pcie_cpubuff;
  std::vector<void*> numa_cpubuff;
  // The (deep copy of) queue list of this task
  std::vector<QueueType> queue_list;
  // The offset of this partition
  unsigned int offset = 0;
  // The length of this partition
  unsigned int len = 0;
  // Atomic counter
  std::shared_ptr<std::atomic_int> counter_ptr;
  // How many partitions
  unsigned int total_partnum = 0;
  // Compressor
  std::shared_ptr<compressor::Compressor> compressor;
  // Compressed
  std::shared_ptr<compressor::tensor_t> compressed;
  // queue index. Used when there are multiple queues for the same task.
  int queue_idx = 0;
  // list of offsets, used for alltoall only
  std::vector<int> offset_list;
  // list of involved keys, used for alltoall only
  std::vector<uint64_t> key_list;
};
using TensorTable = std::unordered_map<std::string, TensorTableEntry>;

#if BYTEPS_BUILDING_CUDA == 1
ncclDataType_t getNcclDataType(DataType dtype);
#endif

int getDataTypeLength(int dtype);

inline size_t Align(size_t size, int dtype) {
  const size_t min_size =
      (getDataTypeLength(dtype) * getDataTypeLength(dtype)) * 8;
  return size + (min_size - size % min_size) % min_size;
}
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMMON_H
