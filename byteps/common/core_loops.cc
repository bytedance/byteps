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

#include "core_loops.h"
#if BYTEPS_BUILDING_CUDA == 1
#include <cuda_runtime.h>
#endif
#include <chrono>
#include <memory>

#include "common.h"
#include "compressor/compressor.h"
#include "core_loops.h"
#include "global.h"
#include "logging.h"
#include "../server/server.h"

namespace byteps {
namespace common {


// returns true if the last partition is done
template <typename T>
bool DoFinishOrProceed(T& task) {
  auto &queue_list = task->queue_list;
  BPS_CHECK_GE(queue_list.size(), 1);
  auto this_op = queue_list[0];
  auto q = BytePSGlobal::GetScheduledQueue(this_op, task->queue_idx);
  q->reportFinish(task->len);

  if (BytePSGlobal::IsTensorSampled(task->key)) {
    // We only support sampling
    BPS_CHECK(task->tensor->dtype() == common::BYTEPS_FLOAT32);
    size_t i = task->offset / 4;
    size_t j = (task->offset + task->len) / 4 - 1;
    if (task->device == CPU_DEVICE_ID) {
      BPS_LOG(DEBUG) << "Sampled key=" << task->key
                     << " rank=" << BytePSGlobal::GetLocalRank()
                     << " input[0]=" << *((float *)(task->tensor->data()) + i)
                     << "\tinput[-1]=" << *((float *)(task->tensor->data()) + j)
                     << "\toutput[0]=" << *((float *)(task->output->data()) + i)
                     << "\toutput[-1]="
                     << *((float *)(task->output->data()) + j)
                     << "\t after stage: " << LogStrings[this_op];
    } else {
#if BYTEPS_BUILDING_CUDA == 1
      float i0, i1, o0, o1;
      cudaMemcpy(&i0, (float *)(task->tensor->data()) + i, 4,
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(&i1, (float *)(task->tensor->data()) + j, 4,
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(&o0, (float *)(task->output->data()) + i, 4,
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(&o1, (float *)(task->output->data()) + j, 4,
                 cudaMemcpyDeviceToHost);
      BPS_LOG(DEBUG) << "Sampled key=" << task->key
                     << " rank=" << BytePSGlobal::GetLocalRank()
                     << " input[0]=" << i0 << "\tinput[-1]=" << i1
                     << "\toutput[0]=" << o0 << "\toutput[-1]=" << o1
                     << "\t after stage: " << LogStrings[this_op];
#endif
    }
  }
  auto ctx = task->context;
  BPS_CHECK(ctx != nullptr);
  if (ctx->profile_flag) {
    BPS_CHECK(task->context->part_comm_time[task->key][this_op].back()->dur ==
              0)
        << " tensor: " << task->tensor_name << " task->key:" << task->key
        << " type:" << this_op << " 'dur' has already been assigned:"
        << task->context->part_comm_time[task->key][this_op].back()->dur;
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(duration);
    auto _ts =
        task->context->part_comm_time[task->key][this_op].back()->start_t;
    BPS_CHECK(task->context->part_comm_time.find(task->key) !=
              task->context->part_comm_time.end())
        << " tensor: " << task->tensor_name << " task->key:" << task->key
        << " type:" << this_op;
    BPS_CHECK(task->context->part_comm_time[task->key].find(this_op) !=
              task->context->part_comm_time[task->key].end())
        << " tensor: " << task->tensor_name << " task->key:" << task->key
        << " type:" << this_op;

    task->context->part_comm_time[task->key][this_op].back()->dur =
        (long long)(us.count()) - _ts;
  }

  // finish current QueueType of this task, erase current QueueType.
  queue_list.erase(queue_list.begin());
  if (queue_list.size() > 0) {
    BPS_CHECK(task->tensor_name != "");
    BPS_LOG(TRACE) << "Rank=" << BytePSGlobal::GetRank() << " finishes "
                   << LogStrings[this_op] << ", tensor: " << task->tensor_name
                   << ", key=" << task->key << "; Passing to the next queue.";
    BytePSGlobal::GetScheduledQueue(queue_list[0], task->queue_idx)->addTask(task);
  } else {
    // this is the last QueueType of this current sub-task.
    BPS_CHECK(task->counter_ptr) << task->tensor_name << " counter_ptr is null";
    int v = task->counter_ptr.get()->fetch_add(1);
    if (v == (int)(task->total_partnum - 1)) {
      // if meet this condition, that means all sub-tasks of this task have been
      // done
      BPS_CHECK(task->tensor_name != "");
      BPS_LOG(TRACE) << "Rank=" << BytePSGlobal::GetRank()
                     << " finish processing tensor: " << task->tensor_name;

      if (PushPullSpeed::ShouldRecord()) {
        // PushPullSpeed::RecordSpeed(task);
      }

      task->callback(Status::OK());
      //* Add for profiling communication events
      if (ctx->profile_flag) {
        BPS_CHECK(task->context->comm_time.back()->dur == 0)
            << " tensor: " << task->tensor_name
            << " 'dur' has already been assigned:"
            << task->context->comm_time.back()->dur;
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        auto us =
            std::chrono::duration_cast<std::chrono::microseconds>(duration);
        auto _ts = task->context->comm_time.back()->start_t;
        task->context->comm_time.back()->dur = (long long)(us.count()) - _ts;
      }
      // Set the profile_flag first
      // *step_cnt* denotes the number this gradient has been synchronized.
      ctx->step_cnt += 1;
      BytePSGlobal::SetProfileFlag(task->context);
      return true;
    }
  }
  return false;
}

// the lite version which does not rely on shared_ptr for memory management
void FinishOrProceedLite(TensorTableEntry* task) {
  bool done = DoFinishOrProceed(task);
  if (done) delete task;
}

void FinishOrProceed(std::shared_ptr<TensorTableEntry> task) {
  DoFinishOrProceed(task);
}

#if BYTEPS_BUILDING_CUDA == 1
bool RunCoordinateLoopOnce(QueueType this_op) {
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  auto task = q->getTask();
  if (task) {
    int rank = BytePSGlobal::GetLocalRank();
    auto key = task->key;

    // first send to next queue and then broadcast signal
    // to guarantee the entry is available when getTask(key) at Reduce/Broadcast
    // thread
    FinishOrProceed(task);

    BytePSCommSignal sig = PUSH_READY;
    std::shared_ptr<BytePSComm> comm;

    switch (this_op) {
      case COORDINATE_REDUCE: {
        sig = REDUCE_READY;
        comm = BytePSGlobal::GetNccl()->GetSignalComm();
        break;
      }
      case COORDINATE_BROADCAST: {
        sig = BCAST_READY;
        comm = BytePSGlobal::GetNccl()->GetSignalComm();
        break;
      }
      case COORDINATE_PUSH: {
        sig = PUSH_READY;
        comm = BytePSGlobal::GetBasicComm();
        break;
      }
      default:
        BPS_CHECK(0) << "unsupported op: " << this_op;
    }

    BPS_CHECK_NE(rank, comm->getRoot())
        << "only non-root device should enter COORDINATE loop";

    struct BytePSCommMsg msg = {rank, sig, key};
    comm->sendSignalToRoot(&msg, sizeof(BytePSCommMsg));

    BPS_CHECK(task->tensor_name != "");
    BPS_LOG(TRACE) << task->tensor_name << " send coordinate info: "
                   << "Signal=" << sig << ", rank=" << rank << ", key=" << key;

  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

inline void PostNcclCalls(
    std::shared_ptr<byteps::common::TensorTableEntry> task, QueueType this_op) {
  BPS_CHECK(this_op == REDUCE || this_op == BROADCAST)
      << "Only REDUCE and BROADCAST use NCCL.";
  auto tensor = (this_op == REDUCE) ? task->tensor : task->output;
  BPS_CHECK(tensor);
  BPS_CHECK_EQ(0, tensor->size() % tensor->shape().num_elements());

  auto key = task->key;
  auto len = task->len;
  auto offset = task->offset;
  auto unit_len = tensor->size() / tensor->shape().num_elements();
  auto p = (char *)(tensor->data()) + offset;
  if (task->device == CPU_DEVICE_ID) {
    p = (char *)(task->gpu_ptr) + offset;
  }

  auto nccl_dtype = getNcclDataType(tensor->dtype());

  auto nccl = BytePSGlobal::GetNccl();
  auto nccl_stream = nccl->GetStream(key, this_op);
  auto nccl_comm = nccl->GetComm(key, this_op);
  auto nccl_root = nccl->GetRoot(key, this_op);
  auto nccl_size = nccl->GetSize();
  auto nccl_rank = nccl->GetRank(key, this_op);

  auto num_elem_per_gpu = len / nccl_size / unit_len;
  auto left_elem = (len / unit_len) - (num_elem_per_gpu * nccl_size);
  if (BytePSGlobal::IsUsingReduce()) {
    nccl_root = BytePSGlobal::GetReduceRootByKey(key);
    num_elem_per_gpu = 0;
    left_elem = len / unit_len;
    BPS_LOG(TRACE) << "Reduce key=" << key << " to root=" << nccl_root
                   << " rank=" << BytePSGlobal::GetLocalRank();
  }

  BPS_CHECK(task->tensor_name != "");
  BPS_LOG(TRACE) << task->tensor_name << " calling NCCL " << LogStrings[this_op]
                 << " (rank=" << nccl_rank << ") key=" << key
                 << ", elements=" << len / unit_len
                 << ", device=" << task->device;

  if (this_op == REDUCE) {
    // We reduce to task->output except that it is a CPU tensor
    auto out_p = (char *)(task->output->data()) + offset;
    if (task->device == CPU_DEVICE_ID && task->tensor == task->output) {
      out_p = p;
    }

    if (num_elem_per_gpu) {
      NCCLCHECK(ncclReduceScatter(
          (const void *)p,
          (void *)(out_p + nccl_rank * num_elem_per_gpu * unit_len),
          (size_t)num_elem_per_gpu, (ncclDataType_t)nccl_dtype,
          (ncclRedOp_t)ncclSum, (ncclComm_t)nccl_comm,
          (cudaStream_t)nccl_stream));
    }
    if (left_elem) {
      NCCLCHECK(ncclReduce((const void *)(p + len - left_elem * unit_len),
                           (void *)(out_p + len - left_elem * unit_len),
                           (size_t)left_elem, (ncclDataType_t)nccl_dtype,
                           (ncclRedOp_t)ncclSum, (int)nccl_root,
                           (ncclComm_t)nccl_comm, (cudaStream_t)nccl_stream));
    }
  } else {
    if (num_elem_per_gpu) {
      NCCLCHECK(ncclAllGather(
          (const void *)(p + nccl_rank * num_elem_per_gpu * unit_len),
          (void *)p, (size_t)num_elem_per_gpu, (ncclDataType_t)nccl_dtype,
          (ncclComm_t)nccl_comm, (cudaStream_t)nccl_stream));
    }
    if (left_elem) {
      NCCLCHECK(ncclBroadcast((const void *)(p + len - left_elem * unit_len),
                              (void *)(p + len - left_elem * unit_len),
                              (size_t)left_elem, (ncclDataType_t)nccl_dtype,
                              (int)nccl_root, (ncclComm_t)nccl_comm,
                              (cudaStream_t)nccl_stream));
    }
  }
}

bool RunRootNcclLoopOnce() {
  auto signal_comm = BytePSGlobal::GetNccl()->GetSignalComm();
  int root = signal_comm->getRoot();
  int rank = BytePSGlobal::GetLocalRank();
  BPS_CHECK_EQ(rank, root);

  int nccl_size = BytePSGlobal::GetNccl()->GetSize();
  QueueType nccl_ops[] = {REDUCE, BROADCAST};

  auto nccl_entry = std::make_shared<NcclGroupEntry>();
  auto &tasks = nccl_entry->tasks;
  auto &queues = nccl_entry->queues;

  NCCLCHECK(ncclGroupStart());
  for (auto this_op : nccl_ops) {
    auto q = BytePSGlobal::GetScheduledQueue(this_op);
    for (int i = 0; i < BytePSGlobal::GetNccl()->GetGroupSize(); i++) {
      auto task = q->getTask();
      if (!task) {
        break;
      }
      tasks.push_back(task);
      queues.push_back(q);

      if (nccl_size > 1) {
        // notify non-root devices
        struct BytePSCommMsg msg = {
            rank, (this_op == REDUCE) ? DO_REDUCE : DO_BROADCAST, task->key};
        signal_comm->broadcastSignal(&msg, sizeof(BytePSCommMsg));
        PostNcclCalls(task, this_op);
      }
    }
  }
  if (tasks.size()) {
    struct BytePSCommMsg msg = {rank, DO_GROUP, 0};
    signal_comm->broadcastSignal(&msg, sizeof(BytePSCommMsg));
    NCCLCHECK(ncclGroupEnd());
    nccl_entry->RecordEvents();
    BPS_LOG(TRACE) << "NCCL Group size=" << tasks.size() << " rank=" << rank;
    BytePSGlobal::GetNccl()->EnqueueGroup(nccl_entry);
  } else {
    NCCLCHECK(ncclGroupEnd());
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  return true;
}

bool RunNonRootNcclLoopOnce() {
  auto signal_comm = BytePSGlobal::GetNccl()->GetSignalComm();
  int root = signal_comm->getRoot();
  int rank = BytePSGlobal::GetLocalRank();
  BPS_CHECK_NE(rank, root);

  auto nccl_entry = std::make_shared<NcclGroupEntry>();
  auto &tasks = nccl_entry->tasks;
  auto &queues = nccl_entry->queues;
  struct BytePSCommMsg msg = {};

  NCCLCHECK(ncclGroupStart());
  while (1) {
    signal_comm->recvSignalFromRoot(&msg, sizeof(BytePSCommMsg));
    if (BytePSGlobal::ShouldShutdown()) return true;
    if (msg.signal == DO_GROUP) {
      break;
    }
    QueueType this_op = REDUCE;
    if (msg.signal == DO_BROADCAST) {
      this_op = BROADCAST;
    } else {
      BPS_CHECK_EQ(msg.signal, DO_REDUCE) << msg.signal << ", " << DO_REDUCE;
    }

    auto key = msg.key;

    auto q = BytePSGlobal::GetScheduledQueue(this_op);
    auto task = q->getTask(key);
    BPS_CHECK(task);

    tasks.push_back(task);
    queues.push_back(q);

    PostNcclCalls(task, this_op);
  }
  NCCLCHECK(ncclGroupEnd());

  nccl_entry->RecordEvents();
  BytePSGlobal::GetNccl()->EnqueueGroup(nccl_entry);
  return true;
}

bool RunSyncNcclOnce() {
  auto nccl_entry = BytePSGlobal::GetNccl()->DequeueGroup();
  if (nccl_entry) {
    nccl_entry->SynchronizeEvents();
    for (size_t i = 0; i < nccl_entry->tasks.size(); i++) {
      FinishOrProceed(nccl_entry->tasks[i]);
    }
    nccl_entry->DestroyEvents();
    BPS_LOG(TRACE) << "Finished NCCL Group size=" << nccl_entry->tasks.size()
                   << " rank=" << BytePSGlobal::GetLocalRank();
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

bool RunCopyDevice2HostLoopOnce() {
  QueueType this_op = COPYD2H;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  auto task = q->getTask();

  if (task) {
    auto copy_d2h_Stream = BytePSGlobal::GetCopyDevice2HostStream();
    // If we ran NCCL reduce, we should copy from task->output
    auto tensor =
        (BytePSGlobal::GetNccl()->GetSize() > 1) ? task->output : task->tensor;
    BPS_CHECK(tensor);
    auto key = task->key;

    auto nccl = BytePSGlobal::GetNccl();
    auto nccl_root = nccl->GetRoot(key, REDUCE);
    auto nccl_size = nccl->GetSize();
    auto nccl_rank = nccl->GetRank(key, REDUCE);

    auto len = task->len;
    auto offset = task->offset;
    auto p = (char *)(tensor->data()) + offset;
    if (task->device == CPU_DEVICE_ID) {
      p = (char *)(task->gpu_ptr) + offset;
    }
    auto unit_len = tensor->size() / tensor->shape().num_elements();
    char *cpubuff;
    if (BytePSGlobal::IsCrossPcieSwitch()) {
      BPS_CHECK(task->pcie_cpubuff.size());
      cpubuff =
          (char *)(task->pcie_cpubuff[BytePSGlobal::GetPcieSwitchIndex()]) +
          offset;
    } else {
      cpubuff = (char *)(task->cpubuff) + offset;
    }

    BPS_CHECK(cpubuff) << task->tensor_name
                       << ": CPU buffer not initialized, size=" << len;

    int copy_offset = 0;
    int copy_len = len;
    auto num_elem_per_gpu = len / nccl_size / unit_len;
    auto left_elem = (len / unit_len) - (num_elem_per_gpu * nccl_size);

    copy_offset = nccl_rank * num_elem_per_gpu * unit_len;
    copy_len = num_elem_per_gpu * unit_len;
    if (left_elem && (nccl_root == nccl_rank)) {
      copy_len += left_elem * unit_len;
    }
    if (BytePSGlobal::IsUsingReduce()) {
      copy_offset = 0;
      copy_len = (BytePSGlobal::GetReduceRootByKey(key) == nccl_rank) ? len : 0;
    }

    if (copy_len) {
      CUDA_CALL(cudaMemcpyAsync(
          (void *)(cpubuff + copy_offset), (const void *)(p + copy_offset),
          (size_t)copy_len, (cudaMemcpyKind)cudaMemcpyDeviceToHost,
          (cudaStream_t)*copy_d2h_Stream));
      CUDA_CALL(cudaStreamSynchronize(*copy_d2h_Stream));
    }

    FinishOrProceed(task);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

bool RunPcieReduceLoopOnce() {
  BPS_CHECK(BytePSGlobal::IsCrossPcieSwitch());
  QueueType this_op = PCIE_REDUCE;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  auto task = q->getTask();
  if (task) {
    auto reducer = BytePSGlobal::GetCpuReducer();
    if (!reducer->isRoot()) {
      // send signal to root
      int rank = BytePSGlobal::GetLocalRank();
      auto key = task->key;
      BytePSCommSignal sig = PCIE_REDUCE_READY;
      struct BytePSCommMsg msg = {rank, sig, key};
      reducer->getComm()->sendSignalToRoot(&msg, sizeof(BytePSCommMsg));
    } else {
      auto tensor = task->tensor;

      auto key = task->key;
      auto len = task->len;
      auto offset = task->offset;
      auto unit_len = tensor->size() / tensor->shape().num_elements();

      auto nccl = BytePSGlobal::GetNccl();
      auto nccl_root = nccl->GetRoot(key, REDUCE);
      auto nccl_size = nccl->GetSize();
      auto nccl_rank = nccl->GetRank(key, REDUCE);

      auto num_elem_per_gpu = len / nccl_size / unit_len;
      auto left_elem = (len / unit_len) - (num_elem_per_gpu * nccl_size);

      auto copy_len = num_elem_per_gpu * unit_len;
      if (left_elem && (nccl_root == nccl_rank)) {
        copy_len += left_elem * unit_len;
      }

      if (copy_len) {
        auto total_offset = offset + nccl_rank * num_elem_per_gpu * unit_len;

        // Below we assume there are only two PCIe switch
        // and we run reducer in the context of the second switch
        reducer->sum((void *)((char *)(task->cpubuff) + total_offset),
                     (void *)((char *)(task->pcie_cpubuff[0]) + total_offset),
                     copy_len, tensor->dtype());
      }
    }

    FinishOrProceed(task);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

bool RunCompressLoopOnce() {
  QueueType this_op = COMPRESS;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  auto task = q->getTask();
  if (task) {
    BPS_CHECK(BytePSGlobal::IsRootDevice())
        << "only root device should enter COMPRESS loop";
    BPS_CHECK(task->compressor != nullptr);
    BPS_CHECK(task->compressed == nullptr);

    // spawn
    BytePSGlobal::GetThreadPool()->enqueue([task]() {
      char *data = const_cast<char *>(static_cast<const char *>(task->cpubuff) +
                                      task->offset);
      int len = task->len;
      int dtype = task->tensor->dtype();
      compressor::tensor_t grad(data, len, dtype);
      auto compressed = task->compressor->Compress(grad);
      BPS_CHECK_LE(compressed.size, len)
          << "Compressor Implementation Error "
          << ", key=" << task->key << ", src_len=" << len
          << ", compressed_len=" << compressed.size;

      task->compressed = std::make_shared<decltype(compressed)>(compressed);

      // restore rt
      auto &queue_list = task->queue_list;
      BytePSGlobal::GetScheduledQueue(queue_list[1])
          ->reset(task->key, BytePSGlobal::GetLocalSize() - 1);

      FinishOrProceed(task);
    });

  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  return true;
}

bool RunDecompressLoopOnce() {
  QueueType this_op = DECOMPRESS;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  auto task = q->getTask();
  if (task) {
    BPS_CHECK(BytePSGlobal::IsRootDevice())
        << "only root device should enter DECOMPRESS loop";
    BPS_CHECK(task->compressor != nullptr);

    // spawn
    BytePSGlobal::GetThreadPool()->enqueue([task]() {
      char *data = const_cast<char *>(static_cast<const char *>(task->cpubuff) +
                                      task->offset);
      auto &pskv = BytePSGlobal::EncodeDefaultKey(task->key, 0);
      auto len = pskv.lens[0];
      int dtype = task->tensor->dtype();
      compressor::tensor_t compressed(data, len, dtype);
      auto decompressed = task->compressor->Decompress(compressed);
      BPS_LOG(DEBUG) << "PULL with gradient compression. key=" << task->key;

      FinishOrProceed(task);
    });

  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }

  return true;
}

void CopyHost2Device(std::shared_ptr<byteps::common::TensorTableEntry> task) {
  auto copy_h2d_stream = BytePSGlobal::GetCopyHost2DeviceStream();
  auto tensor = task->output;
  BPS_CHECK(tensor);
  auto key = task->key;
  auto nccl = BytePSGlobal::GetNccl();
  auto nccl_root = nccl->GetRoot(key, BROADCAST);
  auto nccl_size = nccl->GetSize();
  auto nccl_rank = nccl->GetRank(key, BROADCAST);
  auto len = task->len;
  auto offset = task->offset;
  auto cpubuff = (char *)(task->cpubuff) + offset;
  BPS_CHECK(cpubuff) << task->tensor_name
                     << ": CPU buffer not initialized, size=" << len;

  auto gpu_addr = (char *)(tensor->data()) + offset;
  if (task->device == CPU_DEVICE_ID) {
    gpu_addr = (char *)(task->gpu_ptr) + offset;
  }

  auto unit_len = tensor->size() / tensor->shape().num_elements();
  auto num_elem_per_gpu = len / nccl_size / unit_len;
  auto left_elem = (len / unit_len) - (num_elem_per_gpu * nccl_size);

  auto copy_offset = nccl_rank * num_elem_per_gpu * unit_len;
  auto copy_len = num_elem_per_gpu * unit_len;
  if (left_elem && (nccl_root == nccl_rank)) {
    copy_len += left_elem * unit_len;
  }

  if (BytePSGlobal::IsUsingReduce()) {
    copy_offset = 0;
    copy_len = (BytePSGlobal::GetReduceRootByKey(key) == nccl_rank) ? len : 0;
  }

  if (copy_len) {
    CUDA_CALL(cudaMemcpyAsync(
        (void *)(gpu_addr + copy_offset), (const void *)(cpubuff + copy_offset),
        (size_t)copy_len, (cudaMemcpyKind)cudaMemcpyHostToDevice,
        (cudaStream_t)*copy_h2d_stream));
    CUDA_CALL(cudaStreamSynchronize(*copy_h2d_stream));
  }

  return;
}

bool RunRootCopyHost2DeviceLoopOnce() {
  QueueType this_op = COPYH2D;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  auto task = q->getTask();

  if (task) {
    auto key = task->key;
    int local_rank = BytePSGlobal::GetLocalRank();
    int local_size = BytePSGlobal::GetLocalSize();

    if (local_size > 1) {
      // notify non-root devices
      struct BytePSCommMsg msg = {local_rank, DO_COPYH2D, key};
      BytePSGlobal::GetBasicComm()->broadcastSignal(&msg,
                                                    sizeof(BytePSCommMsg));
    }
    CopyHost2Device(task);
    FinishOrProceed(task);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

bool RunNonRootCopyListenLoopOnce() {
  auto signal_comm = BytePSGlobal::GetBasicComm();
  int root = signal_comm->getRoot();
  int rank = BytePSGlobal::GetLocalRank();
  BPS_CHECK_NE(root, rank);

  struct BytePSCommMsg msg = {};

  signal_comm->recvSignalFromRoot(&msg, sizeof(BytePSCommMsg));
  if (BytePSGlobal::ShouldShutdown()) return true;
  BPS_CHECK_EQ(msg.signal, DO_COPYH2D) << msg.signal;

  BytePSGlobal::GetCopyTable()->AddReadyCount(msg.key);

  BPS_LOG(TRACE) << "NonRootCopyListenLoop recved from root"
                 << ", signal=" << msg.signal << ", key=" << msg.key
                 << ", myrank=" << rank;
  return true;
}

bool RunNonRootCopyHost2DeviceLoopOnce() {
  QueueType this_op = COPYH2D;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  auto task = q->getTask();

  if (task) {
    CopyHost2Device(task);
    FinishOrProceed(task);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

void CoordinateReduceLoop() {
  while (RunCoordinateLoopOnce(COORDINATE_REDUCE) &&
         !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void CoordinateBroadcastLoop() {
  while (RunCoordinateLoopOnce(COORDINATE_BROADCAST) &&
         !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void CoordinatePushLoop() {
  while (RunCoordinateLoopOnce(COORDINATE_PUSH) &&
         !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void PcieReduceLoop() {
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank() % BytePSGlobal::GetNumDevice()));
  while (RunPcieReduceLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void RootNcclLoop() {
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank() % BytePSGlobal::GetNumDevice()));
  while (RunRootNcclLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void NonRootNcclLoop() {
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank() % BytePSGlobal::GetNumDevice()));
  while (RunNonRootNcclLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void SyncNcclLoop() {
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank() % BytePSGlobal::GetNumDevice()));
  while (RunSyncNcclOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void CopyDevice2HostLoop() {
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank() % BytePSGlobal::GetNumDevice()));
  while (RunCopyDevice2HostLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void CompressLoop() {
  while (RunCompressLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void DecompressLoop() {
  while (RunDecompressLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void RootCopyHost2DeviceLoop() {
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank() % BytePSGlobal::GetNumDevice()));
  while (RunRootCopyHost2DeviceLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void NonRootCopyListenLoop() {
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank() % BytePSGlobal::GetNumDevice()));
  while (RunNonRootCopyListenLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void NonRootCopyHost2DeviceLoop() {
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank() % BytePSGlobal::GetNumDevice()));
  while (RunNonRootCopyHost2DeviceLoopOnce() &&
         !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}
#endif

bool RunPushLoopOnce() {
  QueueType this_op = PUSH;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  auto task = q->getTask();
  if (task) {
    BPS_CHECK(BytePSGlobal::IsRootDevice())
        << "only root device should enter PUSH loop";

    if (BytePSGlobal::IsDistributed()) {
      auto offset = task->offset;
      auto len = task->len;

      char *data;
      BPS_CHECK(task->cpubuff);
      data =
          const_cast<char *>(static_cast<const char *>(task->cpubuff) + offset);

      // get metadata
      const int dtype = task->tensor->dtype();

      // use compressed data/len
      if (task->compressed) {
        BPS_LOG(DEBUG) << "PUSH with gradient compression. key=" << task->key;
        data = task->compressed->data;
        len = task->compressed->size;
        task->compressed = nullptr;
      }

      // false means not to delete data when SArray is deleted
      ps::SArray<char> vals(data, len, false);


      int output_device = task->device == CPU_DEVICE_ID ? CPU : GPU;
      int cmd = server::GetCommandType(server::RequestType::kLeaderPushPull, dtype, output_device);
      if (task->reduce_op == REDUCE_OP_AVERAGE) {
        cmd = server::GetCommandType(server::RequestType::kLeaderPushPullAvg, dtype, output_device);
      }
      auto &pskv = BytePSGlobal::EncodeDefaultKey(task->key, len);
      BytePSGlobal::GetPS()->ZPush(pskv.keys, vals, pskv.lens, cmd,
                                   [task, q]() { FinishOrProceed(task); });
      BPS_LOG(TRACE) << " push finished for key=" << task->key;
    } else {
      // This is a dummy barrier for IsCrossPcieSwitch()
      BPS_CHECK(BytePSGlobal::IsCrossPcieSwitch());
      FinishOrProceed(task);
    }
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

bool RunPullLoopOnce() {
  QueueType this_op = PULL;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  auto task = q->getTask();
  if (task) {
    BPS_CHECK(BytePSGlobal::IsRootDevice())
        << "only root device should enter PULL loop";
    // TODO: allow merging
    auto offset = task->offset;
    auto len = task->len;

    char *data;
    BPS_CHECK(task->cpubuff);
    data =
        const_cast<char *>(static_cast<const char *>(task->cpubuff) + offset);

    // get metadata
    const int dtype = task->output->dtype();

    // false means not to delete data when SArray is deleted
    auto vals = new ps::SArray<char>(data, len, false);

    int cmd = server::GetCommandType(server::RequestType::kLeaderPushPull, dtype, CPU);
    auto &pskv = BytePSGlobal::EncodeDefaultKey(task->key, len);
    // issue pull
    BytePSGlobal::GetPS()->ZPull(pskv.keys, vals, &pskv.lens, cmd,
                                 [vals, task, q]() {
                                   delete vals;
                                   FinishOrProceed(task);
                                 });
    BPS_LOG(TRACE) << " pull finished for key=" << task->key;
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

void PushLoop() {
  BPS_LOG(TRACE) << "Started thread: " << __PRETTY_FUNCTION__;
  while (RunPushLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
}

void PullLoop() {
  while (RunPullLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void P2PCopyGroup(std::vector<TensorTableEntry*>& tasks) {
  int my_rank = BytePSGlobal::GetRank();
#if BYTEPS_BUILDING_CUDA == 1
  cudaStream_t* p2p_d2d_stream = BytePSGlobal::GetP2PCopyD2DStream();
  cudaStream_t* p2p_d2h_stream = BytePSGlobal::GetP2PCopyD2HStream();
  cudaStream_t* p2p_h2d_stream = BytePSGlobal::GetP2PCopyH2DStream();
  bool has_d2d_copy = false;
  bool has_d2h_copy = false;
  bool has_h2d_copy = false;
#endif
  for (auto& task : tasks) {
    int send_dev = task->tensor 
                 ? task->tensor->device() 
                 : task->group_tensors[0]->device();
    int recv_dev = task->output 
                 ? task->output->device()
                 : task->group_outputs[0]->device();
    auto key = task->key;
    auto len = task->len;
    auto offset = task->offset;
    int sender = task->context->sender;
    bool is_recv_cpu = (CPU_DEVICE_ID == recv_dev);
    bool is_send_cpu = (CPU_DEVICE_ID == send_dev);
    if (sender == my_rank) {
      // copy to myself
      BPS_LOG(TRACE) << "self H2D key=" << key << " offset=" << offset
                    << " len=" << len << " cpu=" << is_recv_cpu << " device=" << recv_dev;
      auto src_addr = task->tensor
                    ? (char*)task->tensor->data() + offset
                    : (char*)task->group_tensors[my_rank]->data();
      auto dst_addr = task->output
                    ? (char*)task->output->data() + task->offset_list[my_rank]
                    : (char*)task->group_outputs[my_rank]->data();
      if (is_recv_cpu) {
        if (is_send_cpu) {
          BytePSGlobal::GetCpuReducer()->copy(dst_addr, src_addr, len);
        } else {
#if BYTEPS_BUILDING_CUDA == 1
          BytePSGlobal::GetGpuReducer()
              ->copy_async(dst_addr, src_addr, len, cudaMemcpyDeviceToHost, p2p_d2h_stream);
          has_d2h_copy = true;
#else 
          BPS_LOG(FATAL) << "Please build BytePS with BYTEPS_WITH_GPU=1";
#endif
        }
      } else {
#if BYTEPS_BUILDING_CUDA == 1
        // it only happens with CPU-GPU alltoall.
        if (is_send_cpu) {
          BytePSGlobal::GetGpuReducer()
              ->copy_async(dst_addr, src_addr, len, cudaMemcpyHostToDevice, p2p_h2d_stream);
          has_h2d_copy = true;
        } else {
          BytePSGlobal::GetGpuReducer()
              ->copy_async(dst_addr, src_addr, len, cudaMemcpyDeviceToDevice, p2p_d2d_stream);
          has_d2d_copy = true;
        }
#else 
        BPS_LOG(FATAL) << "Please build BytePS with BYTEPS_WITH_GPU=1";
#endif
      }
    } else { // not myself
      auto dst_addr = task->output
                    ? (char*)task->output->data() + task->offset_list[sender]
                    : (char*)task->group_outputs[sender]->data();
      auto recv_arr = server::BytePSServer::GetRecvPartition(key);
      int recv_len = recv_arr.len;
      void* recv_addr = recv_arr.val.data();
      BPS_CHECK(recv_len == len) << recv_len << ", " << len;
      // update the output (data)
      BPS_LOG(TRACE) << "P2P_COPY key=" << key << (long long) recv_addr 
          << " len=" << len << " recv_cpu=" << is_recv_cpu
          << " send_cpu=" << is_send_cpu;
      CHECK(recv_addr) << key;
      if (is_recv_cpu) {
        BytePSGlobal::GetCpuReducer()->copy(dst_addr, recv_addr, recv_len);
      } else {
#if BYTEPS_BUILDING_CUDA == 1
        // ps-lite buffer is already on GPU
        BytePSGlobal::GetGpuReducer()
            ->copy_async(dst_addr, recv_addr, recv_len, cudaMemcpyDeviceToDevice, p2p_d2d_stream);
        has_d2d_copy = true;
#else 
        BPS_LOG(FATAL) << "Please build BytePS with BYTEPS_WITH_GPU=1";
#endif
      }
    }
  }
#if BYTEPS_BUILDING_CUDA == 1
  if (has_d2d_copy) CUDA_CALL(cudaStreamSynchronize(*p2p_d2d_stream));
  if (has_d2h_copy) CUDA_CALL(cudaStreamSynchronize(*p2p_d2h_stream));
  if (has_h2d_copy) CUDA_CALL(cudaStreamSynchronize(*p2p_h2d_stream));
#endif
  return;
}

bool RunP2PCopyHost2DeviceLoopOnce() {
  QueueType this_op = P2P_COPYH2D;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  std::vector<TensorTableEntry*> tasks;
  for (int i = 0; i < BytePSGlobal::GetP2PCopyGroupSize(); ++i) {
    auto task = q->getTaskLite();
    if (!task) {
      break;
    }
    tasks.push_back(task);
  }
  if (tasks.size()) {
    if (!BytePSGlobal::IsSkipH2D()) {
      if (BytePSGlobal::IsProfileAlltoall()) {
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(duration);
        P2PCopyGroup(tasks);
        auto then = std::chrono::system_clock::now();
        auto then_duration = then.time_since_epoch();
        auto then_us = std::chrono::duration_cast<std::chrono::microseconds>(then_duration);
        BPS_LOG(INFO) << "Group Copy time = " << then_us.count() - us.count() << " us (" << tasks.size()<< " tensors)";
      } else {
        P2PCopyGroup(tasks);
      }
    }
    for (auto& task : tasks) {
      if (BytePSGlobal::IsDirectResponse() == 0 
          && task->context->sender != BytePSGlobal::GetRank()) {
        server::BytePSServer::SendPushResponse(task->key);
      }
      FinishOrProceedLite(task);
    }
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

bool RunP2PGroupCopyHost2DeviceLoopOnce() {
  QueueType this_op = P2P_GROUP_COPYH2D;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  auto task = q->getTaskLite();
  if (task) {
    if (!BytePSGlobal::IsSkipH2D()) {
      int my_rank =  BytePSGlobal::GetRank();
      int num_ranks = BytePSGlobal::GetSize();
      const int dtype = task->tensor->dtype();
      bool is_cpu = task->device == CPU_DEVICE_ID;
      BPS_CHECK(task->output->data() == nullptr);
      BPS_CHECK(is_cpu);
      int total_recv_len = 0;
      std::vector<int> offsets;
      offsets.push_back(0);
      // now we allocate the output tensor based on the aggregated received size
      auto in_shape = task->tensor->shape();
      auto ndims = in_shape.dims();
      // handle the case with [0] input
      int remaining_dims = 1;
      for (int i = 1; i < ndims; ++i) {
        BPS_CHECK(in_shape.dim_size(i)) << in_shape.dim_size(i);
        remaining_dims *= in_shape.dim_size(i);
      }
      int unit_size = common::getDataTypeLength(dtype);
      BPS_CHECK(task->aux_output != nullptr);
      BPS_CHECK(unit_size) << unit_size;
      BPS_CHECK(remaining_dims) << remaining_dims;
      void* aux_data = const_cast<void*>(task->aux_output->data());
      int32_t* aux_data_int = (int32_t*) aux_data;

      std::vector<uint64_t> keys;
      // prepare all keys
      uint32_t req_key = (uint32_t) P2P_OP << 10;
      for (uint64_t i = 0; i < num_ranks; ++i) {
        uint64_t key = (i << 32) + (task->key << 16) + req_key;
        if (i != my_rank) {
          keys.push_back(key);
        }
      }
      // recv_arrs does not include the one for self send-recv
      std::vector<server::RecvArray> recv_arrs = server::BytePSServer::GetRecvPartitions(keys);
      // get the length of all ranks
      for (uint64_t i = 0; i < num_ranks; ++i) {
        int64_t recv_len;
        if (i == my_rank) {
          // self send-recv does not go through ps-lite
          recv_len = task->offset_list[my_rank + 1] - task->offset_list[my_rank];
          total_recv_len += recv_len;
        } else {
          int idx = i < my_rank ? i : i - 1;
          recv_len = recv_arrs[idx].len;
          total_recv_len += recv_len;
        }
        offsets.push_back(total_recv_len);
        BPS_LOG(TRACE) << "offsets[" << i+1 << "] = " << total_recv_len;
        // fill in aux_output with recv_split at dim0
        *aux_data_int = recv_len / unit_size / remaining_dims;
        aux_data_int += 1;
      }
      int recv_num_elements = total_recv_len / unit_size;
      int recv_dim0 = recv_num_elements / remaining_dims;
      CHECK(recv_num_elements % remaining_dims == 0)
        << recv_num_elements << "," << remaining_dims;
      common::TensorShape output_shape;
      output_shape.AddDim(recv_dim0);
      for (int i = 1; i < in_shape.shape_.size(); ++i) {
        output_shape.AddDim(in_shape.shape_[i]);
      }
      task->output->resize(output_shape);
      // finally, perform copy.
      char* dst = (char*)(const_cast<void*>(task->output->data()));
      for (int i = 0; i < num_ranks; ++i) {
        // calculate output offset
        if (i == my_rank) {
          // copy to myself
          int recv_len = offsets[i + 1] - offsets[i];
          const char* src = ((const char*) task->tensor->data()) + task->offset_list[i];
          BytePSGlobal::GetCpuReducer()->copy(dst + offsets[i], src, recv_len);
        } else {
          int idx = i < my_rank ? i : i - 1;
          int recv_len = recv_arrs[idx].len;
          void* recv_addr = recv_arrs[idx].val.data();
          BytePSGlobal::GetCpuReducer()->copy(dst + offsets[i], recv_addr, recv_len);
          if (BytePSGlobal::IsDirectResponse() == 0) {
            server::BytePSServer::SendPushResponse(keys[idx]);
          }
        }
      }
    }
    FinishOrProceedLite(task);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

// TODO: remove sync?
void CopyD2H(char* dst, const char* src, int len, bool from_cpu) {
  if (!len) {
    return;
  }
  CHECK(len > 0) << len;
  if (from_cpu) {
    BytePSGlobal::GetCpuReducer()->copy(dst, src, len);
  } else {
#if BYTEPS_BUILDING_CUDA == 1
    auto copy_d2h_Stream = BytePSGlobal::GetCopyDevice2HostStream();
    CUDA_CALL(cudaMemcpyAsync(
      (void *)(dst),
      (const void *)(src),
      (size_t)len, (cudaMemcpyKind)cudaMemcpyDeviceToHost,
      (cudaStream_t)*copy_d2h_Stream));
    CUDA_CALL(cudaStreamSynchronize(*copy_d2h_Stream));
#else
    BPS_LOG(FATAL) << "Please build BytePS with BYTEPS_WITH_GPU=1";
#endif
  }
}

bool RunP2PCopyDevice2HostLoopOnce(int index) {
  QueueType this_op = P2P_COPYD2H;
  auto q = BytePSGlobal::GetScheduledQueue(this_op, index);
  auto task = q->getTask();

  if (task) {
    if (BytePSGlobal::IsSkipD2H()) {
      FinishOrProceed(task);
      return true;
    }
    // note: task does not have `input` field.
    auto tensor = task->tensor;
    BPS_CHECK(tensor);
    auto key = task->key;
    auto len = task->len;
    auto offset = task->offset;
    auto p = (char *)(tensor->data()) + offset;
    bool is_cpu = task->device == CPU_DEVICE_ID;
    char *cpubuff = (char *)(task->cpubuff);
    BPS_CHECK(cpubuff) << task->tensor_name << ": CPU buffer not initialized, size=" << len << ", key=" << key;
    CopyD2H(cpubuff, p, len, is_cpu);
    FinishOrProceed(task);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

bool RunP2PCopyDevice2HostSendLoopOnce(int index) {
  QueueType this_op = P2P_COPYD2H_SEND;
  auto q = BytePSGlobal::GetScheduledQueue(this_op, index);
  auto task = q->getTaskLite();
  if (task) {
    // shuffle the receiver rank
    int my_rank =  BytePSGlobal::GetRank();
    int num_ranks = task->pcie_cpubuff.size();
    bool is_group = (task->group_tensors.size() > 0);
    const int dtype = is_group
                    ? task->group_tensors[0]->dtype()
                    : task->tensor->dtype();
    bool output_size_unknown 
                    = is_group
                    ? false
                    : (task->output == nullptr);
    long long total_zpush_time = 0;
    // task->device denotes the output device,
    // while task->tensor->device() denotes the input devices
    int output_device = task->device == CPU_DEVICE_ID ? CPU : GPU;
    auto req_type = server::RequestType::kDefaultSend;
    // the output tensor is not yet allocated. the receiver
    // must receive the entire group of tensors before copying
    // them to the output
    if (output_size_unknown) {  
      req_type = server::RequestType::kGroupSend;
    }
    int cmd = server::GetCommandType(req_type, dtype, output_device);
    // used for group send with split=0
    int empty_cmd = server::GetCommandType(server::RequestType::kEmptyGroupSend,
                                           dtype, output_device);
    for (int rank_offset = 0; rank_offset < num_ranks; ++rank_offset) {
      int i = (my_rank + rank_offset + 1) % num_ranks;
      auto len = task->offset_list[i + 1] - task->offset_list[i];
      if (i == BytePSGlobal::GetRank()) continue;
      int receiver = i;
      if (len == 0) {
        // split = 0, nothing to copy
        if (output_size_unknown) {
          // when output size is unknown and split=0
          // we still perform send with 1 byte data
          char* cpubuff = (char*) task->pcie_cpubuff[i];
          ps::SArray<char> vals(cpubuff, 1, false);
          auto pskv = BytePSGlobal::EncodeP2PKey(task->key_list[i], 1, receiver);
          // for empty send without known output_size, we never wait for response
          BytePSGlobal::GetPS(index)->ZPush(pskv.keys, vals, pskv.lens, empty_cmd);
        }
        continue;
      } else {
        // split != 0. Do a copy followed by send
        const char* data = is_group
                    ? (const char*) task->group_tensors[i]->data()
                    : (const char*) task->tensor->data();
        auto offset = is_group 
                    ? 0
                    : task->offset_list[i];
        auto pskv = BytePSGlobal::EncodeP2PKey(task->key_list[i], len, receiver);
        char* tensor;
        int input_device = is_group
                    ? task->group_tensors[i]->device()
                    : task->tensor->device();
        if (input_device == CPU_DEVICE_ID 
            && BytePSGlobal::IsDirectResponse() == 2) {
          char* cpubuff = (char*) task->pcie_cpubuff[i];
          CHECK(cpubuff != nullptr);
          CopyD2H(cpubuff, data + offset, len, true);
          tensor = cpubuff;
        } else {
          tensor = (char*) data + offset;
        }
        // perform send. false means not to delete data when SArray is deleted
        ps::SArray<char> vals(tensor, len, false);
        if (BytePSGlobal::IsDirectResponse() == 2) {
          BytePSGlobal::GetPS(index)
            ->ZPush(pskv.keys, vals, pskv.lens, cmd);
        } else {
          if (BytePSGlobal::IsProfileAlltoall()) {
            auto now = std::chrono::system_clock::now();
            auto duration = now.time_since_epoch();
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(duration);
            BytePSGlobal::GetPS(index)
              ->ZPush(pskv.keys, vals, pskv.lens, cmd, [task]() {
                int v = task->counter_a2a.get()->fetch_sub(1);
                if (v == 1) {
                  FinishOrProceedLite(task);
                }
              });
            auto then = std::chrono::system_clock::now();
            auto then_duration = then.time_since_epoch();
            auto then_us = std::chrono::duration_cast<std::chrono::microseconds>(then_duration);
            total_zpush_time += (then_us.count() - us.count());
          } else {
            BytePSGlobal::GetPS(index)
              ->ZPush(pskv.keys, vals, pskv.lens, cmd, [task]() {
                int v = task->counter_a2a.get()->fetch_sub(1);
                if (v == 1) {
                  FinishOrProceedLite(task);
                }
              });
          }
        }
      }
    }
    if (BytePSGlobal::IsProfileAlltoall()) {
      BPS_LOG(INFO) << "ZPush time = " << total_zpush_time << " us (" << task->tensor_name << ")";
    }
    if (BytePSGlobal::IsDirectResponse() == 2 || task->counter_a2a == nullptr) {
      FinishOrProceedLite(task);
    }
    return true;
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

bool RunP2PPullLoopOnce() {
  QueueType this_op = P2P_PULL;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  auto task = q->getTaskLite();
  if (task) {
    int my_rank = BytePSGlobal::GetRank();
    int num_ranks = task->pcie_cpubuff.size();
    int output_device = (task->device == CPU_DEVICE_ID) ? CPU : GPU;
    bool is_group = (task->group_tensors.size() > 0);
    for (int r = 0; r < num_ranks; ++r) {
      int i = (my_rank + r + 1) % num_ranks;
      if (i == my_rank) continue;
      auto len = task->offset_list[i+1] - task->offset_list[i];
      auto pskv = BytePSGlobal::EncodeP2PKey(task->key_list[i], len, i);
      const char* dest = is_group
                  ? (const char*) task->group_outputs[i]->data()
                  : (const char*) task->output->data();
      auto offset = is_group 
                  ? 0
                  : task->offset_list[i];
      const int dtype = is_group 
                  ? task->group_outputs[i]->dtype()
                  : task->output->dtype();
      int cmd = server::GetCommandType(server::RequestType::kDefaultPull, 
                                       dtype, output_device);
      char* output = (char*) dest + offset;
      auto vals = new ps::SArray<char>(output, len, false);
      BytePSGlobal::GetPS()
        ->ZPull(pskv.keys, vals, &pskv.lens, cmd, [i, vals, task, dtype, output_device]() {
          if (!BytePSGlobal::IsP2PAckDisabled()) {
            // notify the server that the response is received
            // perform zpush with 1 byte data
            char* cpubuff = (char*) task->pcie_cpubuff[i];
            ps::SArray<char> ack_vals(cpubuff, 1, false);
            auto pskv = BytePSGlobal::EncodeP2PKey(task->key_list[i], 1, i);
            int ack_cmd = server::GetCommandType(server::RequestType::kAckSignal, dtype, output_device);
            // no need to wait for this zpush 
            BytePSGlobal::GetPS()->ZPush(pskv.keys, ack_vals, pskv.lens, ack_cmd);
          }
          // the rest of the callbacks are for zpull          
          delete vals;
          int v = task->counter_a2a.get()->fetch_sub(1);
          if (v == 1) {
            FinishOrProceedLite(task);
          }
        });
    }
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

bool RunP2PPullResponseOnce() {
  QueueType this_op = P2P_PULL_RESPONSE;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  auto task = q->getTaskLite();
  if (task) {
    BPS_CHECK(task->tensor);
    int sender = task->context->sender;
    int my_rank = BytePSGlobal::GetRank();
    if (sender == my_rank) {
      // copy to myself
      bool is_group = (task->group_tensors.size() > 0);
      auto src_addr = (char*) task->tensor->data() 
                    + (is_group ? task->offset : task->offset_list[my_rank]);
      auto dst_addr = (char*) task->output->data() + task->offset;
      // FIXME: add h2d / d2h
      BytePSGlobal::GetGpuReducer()->copy_d2d(dst_addr, src_addr, task->len);
      if (!BytePSGlobal::IsP2PAckDisabled()) {
        BytePSGlobal::GetP2PAckTable()->AddReadyCount(task->key);
      }
    } else {
      char* tensor = (char*) task->tensor->data() + task->offset;
      if (BytePSGlobal::IsProfileAlltoall()) {
        auto ts = std::chrono::high_resolution_clock::now();
        server::BytePSServer::SendPullResponse(task->key, tensor, task->len);
        auto te = std::chrono::high_resolution_clock::now();
        BPS_LOG(INFO)<< "RESP time = " << std::chrono::duration<double, std::milli>(te-ts).count() << " ms";
      } else {
        server::BytePSServer::SendPullResponse(task->key, tensor, task->len);
      }
    } 
    FinishOrProceedLite(task);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

bool RunP2PAckLoopOnce() {
  QueueType this_op = P2P_WAIT_ACK;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  auto task = q->getTaskLite();
  if (task) {
    FinishOrProceedLite(task);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

void P2PPullLoop() {
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank() % BytePSGlobal::GetNumDevice()));
  while (RunP2PPullLoopOnce() &&
         !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
};

void P2PPullResponseLoop() {
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank() % BytePSGlobal::GetNumDevice()));
  while (RunP2PPullResponseOnce() &&
         !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void P2PAckLoop() {
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank() % BytePSGlobal::GetNumDevice()));
  while (RunP2PAckLoopOnce() &&
         !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void P2PCopyHost2DeviceLoop() {
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank() % BytePSGlobal::GetNumDevice()));
  while (RunP2PCopyHost2DeviceLoopOnce() &&
         !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void P2PGroupCopyHost2DeviceLoop() {
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank() % BytePSGlobal::GetNumDevice()));
  while (RunP2PGroupCopyHost2DeviceLoopOnce() &&
         !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void P2PCopyDevice2HostLoop(int index) {
  CHECK(index >= 0);
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank() % BytePSGlobal::GetNumDevice()));
  while (RunP2PCopyDevice2HostLoopOnce(index) && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void P2PCopyDevice2HostSendLoop(int index) {
  CHECK(index >= 0);
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetLocalRank() % BytePSGlobal::GetNumDevice()));
  while (RunP2PCopyDevice2HostSendLoopOnce(index) && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

bool RunCpuCopyLoopOnce() {
  QueueType this_op = CPU_COPY;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  auto task = q->getTask();
  if (!task) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    return true;
  }

  auto reducer = BytePSGlobal::GetCpuReducer();

  // task->tensor->data() is the ptr given to us by the DL framework, which is a
  // cpu pointer in this case
  auto len = task->len;
  auto offset = task->offset;
  reducer->copy((void *)((char *)(task->cpubuff) + offset),
                  (char *)(task->tensor->data()) + offset, len);

  FinishOrProceed(task);
  std::shared_ptr<BytePSComm> comm = BytePSGlobal::GetBasicComm();
  int rank = BytePSGlobal::GetLocalRank();
  if (rank != comm->getRoot()) {
    BytePSCommSignal sig = CPU_REDUCE_READY;
    // only non-root device should enter COORDINATE loop
    struct BytePSCommMsg msg = {rank, sig, task->key};
    comm->sendSignalToRoot(&msg, sizeof(BytePSCommMsg));
    BPS_CHECK(task->tensor_name != "");
    BPS_LOG(TRACE) << task->tensor_name << " sent coordinate info: "
                   << "Signal=" << sig << "(" << SigLogStrings[sig] << ")"
                   << ", rank=" << rank << ", key=" << task->key;
  }
  return true;
}

void CpuCopyLoop() {
  BPS_LOG(TRACE) << "Started thread: " << __PRETTY_FUNCTION__;
  while (RunCpuCopyLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
}

bool RunCpuReduceLoopOnce() {
  QueueType this_op = CPU_REDUCE;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  auto task = q->getTask();
  if (!task) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    return true;
  }

  auto reducer = BytePSGlobal::GetCpuReducer();

  // task->gpu_ptr is the ptr given to us by the DL framework, which is a cpu
  // pointer in this case
  auto tensor = task->tensor;
  auto key = task->key;
  auto len = task->len;
  auto offset = task->offset;
  auto unit_len = tensor->size() / tensor->shape().num_elements();
  int my_lrank = BytePSGlobal::GetLocalRank();
  int local_size = BytePSGlobal::GetLocalSize();
  auto comm = BytePSGlobal::GetBasicComm();
  int local_root = comm->getRoot();

  auto num_elem_per_lrank = len / local_size / unit_len;
  auto left_elem = (len / unit_len) - (num_elem_per_lrank * local_size);

  auto copy_len = num_elem_per_lrank * unit_len;
  if (left_elem && (local_root == my_lrank)) {
    copy_len += left_elem * unit_len;
  }

  if (!BytePSGlobal::IsRootDevice()) {
    if (copy_len) {
      auto total_offset = offset + my_lrank * num_elem_per_lrank * unit_len;
      for (int i = 0; i < local_size; ++i) {
        if (i == local_root) {
          continue;
        }
        BPS_LOG(TRACE) << "dst " << (void *)((char *)(task->numa_cpubuff[local_root]) + total_offset)
          << " src " << (void *)((char *)(task->numa_cpubuff[i]) + total_offset);
        reducer->sum((void *)((char *)(task->numa_cpubuff[local_root]) + total_offset),
          (void *)((char *)(task->numa_cpubuff[i]) + total_offset),
          copy_len, tensor->dtype());
      }
    }

    // send signal to root
    BytePSCommSignal sig = PUSH_READY;
    struct BytePSCommMsg msg = {my_lrank, sig, key};
    comm->sendSignalToRoot(&msg, sizeof(BytePSCommMsg));
    BPS_LOG(TRACE) << task->tensor_name << " sent coordinate info: "
                   << "Signal=" << sig << "(" << SigLogStrings[sig] << ")"
                   << ", myrank=" << my_lrank << ", key=" << key;
  } else {
    BytePSCommSignal sig = DO_CPU_REDUCE;
    struct BytePSCommMsg msg = {my_lrank, sig, key};
    comm->broadcastSignal(&msg, sizeof(BytePSCommMsg));

    if (copy_len) {
      auto total_offset = offset + my_lrank * num_elem_per_lrank * unit_len;
      for (int i = 0; i < local_size; ++i) {
        if (i == local_root) {
          continue;
        }
        BPS_LOG(TRACE) << "dst " << (void *)((char *)(task->cpubuff) + offset)
          << " src " << (void *)((char *)(task->numa_cpubuff[i]) + total_offset);
        reducer->sum((void *)((char *)(task->cpubuff) + total_offset),
          (void *)((char *)(task->numa_cpubuff[i]) + total_offset),
          copy_len, tensor->dtype());

      }
    }
  }

  FinishOrProceed(task);
  return true;
}

void CpuReduceLoop() {
  BPS_LOG(TRACE) << "Started thread: " << __PRETTY_FUNCTION__;
  while (RunCpuReduceLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
}

bool RunCpuBcastLoopOnce() {
  QueueType this_op = CPU_BCAST;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  auto task = q->getTask();
  if (!task) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    return true;
  }

  auto reducer = BytePSGlobal::GetCpuReducer();
  auto len = task->len;
  auto offset = task->offset;
  if (BytePSGlobal::IsRootDevice()) {
    // send signal to non root workers
    int rank = BytePSGlobal::GetLocalRank();
    auto key = task->key;
    BytePSCommSignal sig = DO_CPU_BCAST;
    struct BytePSCommMsg msg = {rank, sig, key};
    auto comm = BytePSGlobal::GetBasicComm();
    comm->broadcastSignal(&msg, sizeof(BytePSCommMsg));
    reducer->copy((void *)((char *)(task->output->data()) + offset),
                  (char *)((task->cpubuff) + offset), len);
  } else {
    auto tensor = task->tensor;

    auto key = task->key;
    int my_lrank = BytePSGlobal::GetLocalRank();
    auto basic_comm = BytePSGlobal::GetBasicComm();
    int local_root = basic_comm->getRoot();

    BPS_LOG(DEBUG) << "Sampled key=" << task->key << "local_root " << local_root;
    BPS_LOG(TRACE) << "dst " << (void *) ((char *)(task->cpubuff) + offset)
      << " src " << (void *) ((task->numa_cpubuff[local_root]) + offset) ;
    reducer->copy((void *)((char *)(task->output->data()) + offset),
                  (char *)(task->numa_cpubuff[local_root]) + offset, len);
    BytePSCommSignal sig = CPU_BCAST_DONE;
    struct BytePSCommMsg msg = {my_lrank, sig, key};
    auto comm = BytePSGlobal::GetBasicComm();
    comm->sendSignalToRoot(&msg, sizeof(BytePSCommMsg));
  }

  FinishOrProceed(task);
  return true;
}

void CpuBcastLoop() {
  while (RunCpuBcastLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
}

bool RunCpuBcastFinishLoopOnce() {
  QueueType this_op = CPU_BCAST_FINISH;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  auto task = q->getTask();
  if (!task) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    return true;
  }

  FinishOrProceed(task);
  return true;
}

void CpuBcastFinishLoop() {
  while (RunCpuBcastFinishLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
}

#if BYTEPS_BUILDING_CUDA == 0
void CoordinateReduceLoop() {
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
  BytePSGlobal::ReportThreadFinish();
}

void CoordinateBroadcastLoop() {
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
  BytePSGlobal::ReportThreadFinish();
}

void CoordinatePushLoop() {
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
  BytePSGlobal::ReportThreadFinish();
}

void PcieReduceLoop() {
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
  BytePSGlobal::ReportThreadFinish();
}

void RootNcclLoop() {
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
  BytePSGlobal::ReportThreadFinish();
}

void NonRootNcclLoop() {
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
  BytePSGlobal::ReportThreadFinish();
}

void SyncNcclLoop() {
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
  BytePSGlobal::ReportThreadFinish();
}

void CopyDevice2HostLoop() {
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
  BytePSGlobal::ReportThreadFinish();
}

void CompressLoop() {
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
  BytePSGlobal::ReportThreadFinish();
}

void DecompressLoop() {
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
  BytePSGlobal::ReportThreadFinish();
}

void RootCopyHost2DeviceLoop() {
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
  BytePSGlobal::ReportThreadFinish();
}

void NonRootCopyListenLoop() {
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
  BytePSGlobal::ReportThreadFinish();
}

void NonRootCopyHost2DeviceLoop() {
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
  BytePSGlobal::ReportThreadFinish();
}
#endif

}  // namespace common
}  // namespace byteps
