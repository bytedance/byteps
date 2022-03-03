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

#if BYTEPS_BUILDING_CUDA == 1
#include <cuda_runtime.h>
#endif
#include <chrono>
#include <memory>
#include <sys/syscall.h>

#include "common.h"
#include "compressor/compressor.h"
#include "core_loops.h"
#include "global.h"
#include "error.h"
#include "logging.h"
#include "../server/server.h"

namespace byteps {
namespace common {

#define gettid() syscall(SYS_gettid)

// returns true if the last partition is done
template <typename T>
bool DoFinishOrProceed(T& task) {
  auto &queue_list = task->queue_list;
  BPS_CHECK_GE(queue_list.size(), 1);
  auto this_op = queue_list[0];
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->reportFinish(task->len);

  AllgatherPullRespTableUpdate(this_op, task->key, task->context->key_list);

  if (BytePSGlobal::IsTensorSampled(task->key)) {
    // We only support sampling
    BPS_CHECK(task->tensor->dtype() == common::BYTEPS_FLOAT32);
    size_t i = task->offset / 4;
    size_t j = (task->offset + task->len) / 4 - 1;
    if (task->device == CPU_DEVICE_ID) {
      BPS_LOG(DEBUG) << "Sampled key=" << task->key
                     << " rank=" << BytePSGlobal::GetLocalRank()
                     << " input[0]=" << *((float *)(task->cpubuff) + i)
                     << "\tinput[-1]=" << *((float *)(task->cpubuff) + j)
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
    BytePSGlobal::GetScheduledQueue(queue_list[0])->addTask(task);
  } else {
    // this is the last QueueType of this current sub-task.
    BPS_CHECK(task->counter_ptr) << task->tensor_name << " counter_ptr is null";
    int v = task->counter_ptr.get()->fetch_add(1);
    if (v == (int)(task->total_partnum - 1)) {
      // if meet this condition, that means all sub-tasks of this task have been
      // done
      BPS_CHECK(task->tensor_name != "");
      BPS_LOG(TRACE) << "Rank=" << BytePSGlobal::GetRank()
                     << " finishes processing tensor: " << task->tensor_name;
      Telemetry::RecordEnd(task->context->base_tensor_name);

      task->callback(Status::OK());
      // error handling: remove the callback from the pending list
      BytePSError::RemoveCallback(task->context->key_list[0]);

      // Add for profiling communication events
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
    } else {
      BPS_LOG(TRACE) << "Rank=" << BytePSGlobal::GetRank() << " finishes sub_task: "
                     << task->tensor_name << " key=" << task->key << " count=" << v;
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
bool RunCoordinateLoopOnce() {
  QueueType ops[] = {COORDINATE_REDUCE, COORDINATE_BROADCAST, 
                     COORDINATE_PUSH, COORDINATE_ALLGATHER,
                     COORDINATE_ALLGATHER_BCAST};
  for (auto this_op : ops) {
    auto q = BytePSGlobal::GetScheduledQueue(this_op);
    q->wait();

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
        case COORDINATE_ALLGATHER: {
          sig = ALLGATHER_REDAY;
          comm = BytePSGlobal::GetBasicComm();
          break;
        }
        case COORDINATE_ALLGATHER_BCAST: {
          sig = ALLGATHER_BCAST_READY;
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
                     << "Signal=" << sig << "(" << SigLogStrings[sig] << ")"
                     << ", rank=" << rank << ", key=" << task->key;

    } else {
      std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }
  }
  
  return true;
}

inline void PostNcclCalls(
    std::shared_ptr<byteps::common::TensorTableEntry> task, QueueType this_op) {
  BPS_CHECK(this_op == REDUCE || this_op == BROADCAST || 
            this_op == ALLGATHER || this_op == ALLGATHER_BCAST)
      << "Only REDUCE, BROADCAST, ALLGATHER, and ALLGATHER_BCAST use NCCL.";
  auto tensor = (this_op == REDUCE || this_op == ALLGATHER) ? task->tensor : task->output;
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

  BPS_CHECK(task->tensor_name != "");
  BPS_LOG(TRACE) << task->tensor_name << " calling NCCL " << LogStrings[this_op]
                 << " (rank=" << nccl_rank << ") key=" << key
                 << ", elements=" << len / unit_len
                 << ", device=" << task->device;

  auto num_elem_per_gpu = len / nccl_size / unit_len;
  auto left_elem = (len / unit_len) - (num_elem_per_gpu * nccl_size);
  if (BytePSGlobal::IsUsingReduce()) {
    nccl_root = BytePSGlobal::GetReduceRootByKey(key);
    num_elem_per_gpu = 0;
    left_elem = len / unit_len;
    BPS_LOG(TRACE) << "Reduce key=" << key << " to root=" << nccl_root
                   << " rank=" << BytePSGlobal::GetLocalRank();
  }

  switch (this_op) {
    case REDUCE: {
      // We reduce to task->output except that it is a CPU tensor
      auto out_p = (char *)(task->output->data()) + offset;
      if (task->device == CPU_DEVICE_ID && task->tensor == task->output) {
        out_p = p;
      }

      if (BytePSGlobal::IsGDR() && BytePSGlobal::GetPhyNodeNum() == 1) {
        NCCLCHECK(ncclAllReduce(
            (const void *)p,
            (void *)out_p,
            (size_t)(len/unit_len), (ncclDataType_t)nccl_dtype,
            (ncclRedOp_t)ncclSum, (ncclComm_t)nccl_comm,
            (cudaStream_t)nccl_stream));
      } else {
        if (num_elem_per_gpu) {
          NCCLCHECK(ncclReduceScatter(
              (const void *)p,
              (void *)(out_p + nccl_rank * num_elem_per_gpu * unit_len),
              (size_t)num_elem_per_gpu, (ncclDataType_t)nccl_dtype,
              (ncclRedOp_t)ncclSum, (ncclComm_t)nccl_comm,
              (cudaStream_t)nccl_stream));
        }
        if (left_elem) {
          NCCLCHECK(ncclReduce(
              (const void *)(p + len - left_elem * unit_len),
              (void *)(out_p + len - left_elem * unit_len),
              (size_t)left_elem, (ncclDataType_t)nccl_dtype,
              (ncclRedOp_t)ncclSum, (int)nccl_root,
              (ncclComm_t)nccl_comm, (cudaStream_t)nccl_stream));
        }
      }
      break;
    }
    case BROADCAST: {
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
      break;
    }
    case ALLGATHER: {
      auto t = reinterpret_cast<P2PTensorTableEntry*>(task.get());
      auto out_p = (char *)(t->output->data());
      if (t->shape_list.empty()) {
        int phy_id = BytePSGlobal::GetPhyNodeID();
        int local_size = BytePSGlobal::GetLocalSize();
        out_p += phy_id * local_size * len;

        NCCLCHECK(ncclAllGather(
            (const void *)p, (void *)out_p, 
            (size_t)len / unit_len, (ncclDataType_t)nccl_dtype,
            (ncclComm_t)nccl_comm, (cudaStream_t)nccl_stream));
      } else {
        // for allgatherv
        // similar as NCCLAllgather: https://github.com/horovod/horovod/blob/master/horovod/common/ops/nccl_operations.cc#L700
        BPS_CHECK((int)t->offset_list.size() == BytePSGlobal::GetSize() + 1);
        int phy_id = BytePSGlobal::GetPhyNodeID();
        int local_size = BytePSGlobal::GetLocalSize();
        for (int i = 0; i < local_size; ++i) {
          int index = phy_id * local_size + i;
          size_t num_elem = t->offset_list[index + 1] - t->offset_list[index];
          void* new_out_p = out_p + t->offset_list[index] * unit_len;
          NCCLCHECK(ncclBroadcast((const void *)(p),
                                  (void *)(new_out_p),
                                  (size_t)num_elem, (ncclDataType_t)nccl_dtype,
                                  (int)i, (ncclComm_t)nccl_comm,
                                  (cudaStream_t)nccl_stream));
        }
      }
      break;
    }
    case ALLGATHER_BCAST: {
      if (BytePSGlobal::IsGDRAllgather()) {
        nccl_root = nccl->GetSignalComm()->getRoot();
      }
      
      NCCLCHECK(ncclBroadcast(
          (const void *)p, (void *)p,
          (size_t)tensor->shape().num_elements(), (ncclDataType_t)nccl_dtype,
          (int)nccl_root, (ncclComm_t)nccl_comm,
          (cudaStream_t)nccl_stream));
      break;
    }
    default:
      BPS_CHECK(false) << "nccl op not supported";
  }
}

bool RunRootNcclLoopOnce() {
  auto signal_comm = BytePSGlobal::GetNccl()->GetSignalComm();
  int root = signal_comm->getRoot();
  int rank = BytePSGlobal::GetLocalRank();
  BPS_CHECK_EQ(rank, root);

  int nccl_size = BytePSGlobal::GetNccl()->GetSize();
  QueueType nccl_ops[] = {REDUCE, BROADCAST, ALLGATHER, 
                          ALLGATHER_BCAST};
  auto q = BytePSGlobal::GetScheduledQueue(nccl_ops[0]);
  q->wait();

  auto nccl_entry = std::make_shared<NcclGroupEntry>();
  auto &tasks = nccl_entry->tasks;
  auto &queues = nccl_entry->queues;

  bool started = false;
  for (auto this_op : nccl_ops) {
    auto q = BytePSGlobal::GetScheduledQueue(this_op);

    for (int i = 0; i < BytePSGlobal::GetNccl()->GetGroupSize(); i++) {
      auto task = q->getTask();
      if (!task) {
        break;
      }
      tasks.push_back(task);
      queues.push_back(q);
      // Only start nccl group at the first actual task.
      // ncclGroupStart & ncclGroupEnd without any task will lead to
      // nccl unhandled error in some cases
      if (!started) {
        NCCLCHECK(ncclGroupStart());
        started = true;
      }

      if (nccl_size > 1) {
        // notify non-root devices
        struct BytePSCommMsg msg = {rank, DO_REDUCE, task->key};
        switch (this_op) {
          case REDUCE:
            msg.signal = DO_REDUCE;
            break;
          case BROADCAST:
            msg.signal = DO_BROADCAST;
            break;
          case ALLGATHER:
            msg.signal = DO_ALLGATHER;
            break;
          case ALLGATHER_BCAST:
            msg.signal = DO_ALLGATHER_BCAST;
            break;
          default:
            BPS_CHECK(false) << "nccl op not supported";
        }
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
    switch (msg.signal) {
      case DO_REDUCE:
        this_op = REDUCE;
        break;
      case DO_BROADCAST:
        this_op = BROADCAST;
        break;
      case DO_ALLGATHER:
        this_op = ALLGATHER;
        break;
      case DO_ALLGATHER_BCAST:
        this_op = ALLGATHER_BCAST;
        break;
      default:
        BPS_CHECK(false) << msg.signal << ", unknown signal for non-root nccl loop";
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
  BytePSGlobal::GetNccl()->wait();
  auto nccl_entry = BytePSGlobal::GetNccl()->DequeueGroup();
  if (nccl_entry) {
    nccl_entry->BusyWaitEvents();
    for (size_t i = 0; i < nccl_entry->tasks.size(); i++) {
      FinishOrProceed(nccl_entry->tasks[i]);
    }
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
  q->wait();
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
  q->wait();
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
      BPS_CHECK_LE(compressed.size, (size_t) len)
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
  q->wait();
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
      task->compressor->Decompress(compressed);
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
  q->wait();
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
  q->wait();
  auto task = q->getTask();

  if (task) {
    CopyHost2Device(task);
    FinishOrProceed(task);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

bool RunAllgatherCopyDevice2HostLoopOnce() {
  QueueType this_op = ALLGATHER_COPYD2H;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->wait();
  auto t = q->getTask();
  auto task = reinterpret_cast<P2PTensorTableEntry*>(t.get());

  if (task) {
    BPS_CHECK(task->tensor) << task->tensor_name
                    << ": input tensor is empty, size=" << task->len;
    BPS_CHECK(task->output) << task->tensor_name
                    << ": output tensor is empty, size=" << task->len;
    BPS_CHECK(task->cpubuff) << task->tensor_name
                       << ": CPU buffer not initialized, size=" << task->len;
    
    auto key = task->key;
    auto nccl = BytePSGlobal::GetNccl();
    auto nccl_root = nccl->GetRoot(key, this_op);
    auto nccl_rank = nccl->GetRank(key, this_op);
    
    auto cal_len_offset = [&](int& copy_len, int& copy_offset) {
      copy_len = 0;
      copy_offset = 0;
      auto len = task->len;
      auto unit_len = task->tensor->size() / task->tensor->shape().num_elements();
      const auto& shape_list = task->shape_list;
      const auto& offset_list = task->offset_list;
      int phy_id = BytePSGlobal::GetPhyNodeID();
      int local_size = BytePSGlobal::GetLocalSize();
      if (!shape_list.empty()) {
        BPS_CHECK((int)offset_list.size() == BytePSGlobal::GetSize() + 1);
        int rank_offset = phy_id * local_size;
        copy_len = (offset_list[rank_offset + local_size] - offset_list[rank_offset]) * unit_len;
        copy_offset = offset_list[rank_offset] * unit_len;
      } else {
        copy_len = BytePSGlobal::GetLocalSize() * len;
        copy_offset = phy_id * local_size * len;
      }
    };

    int copy_len = 0;   
    int copy_offset = 0; 
    if (BytePSGlobal::IsUsingReduce()) {
      if (BytePSGlobal::GetReduceRootByKey(key) == nccl_rank) {
        cal_len_offset(copy_len, copy_offset);
      }
    } else {
      // TODO: make every gpu to participate in copy
      if (nccl_rank == nccl_root) {
        cal_len_offset(copy_len, copy_offset);
      }
    }

    if (copy_len) {
      auto copy_d2h_Stream = BytePSGlobal::GetAllgatherCopyDevice2HostStream();
      CUDA_CALL(cudaMemcpyAsync(
          (void *)((char*)task->cpubuff + copy_offset), (const void *)((char*)task->output->data() + copy_offset),
          (size_t)copy_len, (cudaMemcpyKind)cudaMemcpyDeviceToHost,
          (cudaStream_t)*copy_d2h_Stream));
      CUDA_CALL(cudaStreamSynchronize(*copy_d2h_Stream));

      // TODO: change to IsRootDevice()
      if (BytePSGlobal::GetNccl()->IsSignalRoot()) {
        for (auto key : task->context->key_list) {
          if (key != task->key) {
            BytePSGlobal::GetAllgatherPullRespTable()->AddReadyCount(key);
          }
        }
      }
      else {
        int local_rank = BytePSGlobal::GetLocalRank();
        std::shared_ptr<BytePSComm> comm = BytePSGlobal::GetBasicComm();
        for (auto key : task->context->key_list) {
          if (key != task->key) {
            struct BytePSCommMsg msg = {local_rank, ALLGATHER_COPY_D2H_READY, key};
            comm->sendSignalToRoot(&msg, sizeof(BytePSCommMsg));
          }
        }
      }
    }

    FinishOrProceed(t);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

void AllgatherCopyHost2Device(std::shared_ptr<byteps::common::TensorTableEntry> task) {
  BPS_CHECK(task->output) << task->tensor_name
                          << ": output tensor is empty, size=" << task->len;
  BPS_CHECK(task->cpubuff) << task->tensor_name
                          << ": CPU buffer not initialized, size=" << task->len;

  QueueType this_op = ALLGATHER_COPYH2D;
  auto key = task->key;
  auto nccl = BytePSGlobal::GetNccl();
  auto nccl_root = nccl->GetRoot(key, this_op);
  auto nccl_rank = nccl->GetRank(key, this_op);

  int copy_len = 0;
  if (BytePSGlobal::IsUsingReduce()) {
    if (BytePSGlobal::GetReduceRootByKey(key) == nccl_rank) {
      copy_len = task->output->size();
    }
  } else {
    // TODO: make every gpu to participate in copy
    if (nccl_rank == nccl_root) {
      copy_len = task->output->size();
    }
  }

  if (copy_len) {
    auto copy_h2d_stream = BytePSGlobal::GetAllgatherCopyHost2DeviceStream();
    CUDA_CALL(cudaMemcpyAsync(
        (void *)task->output->data(), (const void *)task->cpubuff,
        (size_t)copy_len, (cudaMemcpyKind)cudaMemcpyHostToDevice,
        (cudaStream_t)*copy_h2d_stream));
    CUDA_CALL(cudaStreamSynchronize(*copy_h2d_stream));
  }
}

bool RunAllgatherRootCopyHost2DeviceLoopOnce() {
  QueueType this_op = ALLGATHER_COPYH2D;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->wait();
  auto task = q->getTask();

  if (task) {
    // notify non-root devices
    int local_rank = BytePSGlobal::GetLocalRank();
    struct BytePSCommMsg msg = {local_rank, DO_ALLGATHER_COPYH2D, task->key};
    BytePSGlobal::GetBasicComm()->broadcastSignal(&msg,
                                                   sizeof(BytePSCommMsg));

    AllgatherCopyHost2Device(task);
    FinishOrProceed(task);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

bool RunAllgatherNonRootCopyHost2DeviceLoopOnce() {
  QueueType this_op = ALLGATHER_COPYH2D;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->wait();
  auto task = q->getTask();

  if (task) {
    AllgatherCopyHost2Device(task);
    FinishOrProceed(task);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

void CoordinateLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();

  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  QueueType ops[] = {COORDINATE_REDUCE, COORDINATE_BROADCAST, 
                     COORDINATE_PUSH, COORDINATE_ALLGATHER,
                     COORDINATE_ALLGATHER_BCAST};
  for (auto this_op : ops) {
    auto q = BytePSGlobal::GetScheduledQueue(this_op);
    q->subscribe(cond_var);
  }

  while (RunCoordinateLoopOnce() &&
         !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void PcieReduceLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();

  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  QueueType this_op = PCIE_REDUCE;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->subscribe(cond_var);

  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetVisibleDevice()));
  while (RunPcieReduceLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void RootNcclLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();

  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  QueueType nccl_ops[] = {REDUCE, BROADCAST, ALLGATHER, 
                          ALLGATHER_BCAST};
  for (auto this_op : nccl_ops) {
    auto q = BytePSGlobal::GetScheduledQueue(this_op);
    q->subscribe(cond_var);
  }

  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetVisibleDevice()));
  while (RunRootNcclLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void NonRootNcclLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetVisibleDevice()));
  while (RunNonRootNcclLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void SyncNcclLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetVisibleDevice()));
  while (RunSyncNcclOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void CopyDevice2HostLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();

  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  QueueType this_op = COPYD2H;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->subscribe(cond_var);

  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetVisibleDevice()));
  while (RunCopyDevice2HostLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void CompressLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();

  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  QueueType this_op = COMPRESS;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->subscribe(cond_var);

  while (RunCompressLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void DecompressLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();

  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  QueueType this_op = DECOMPRESS;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->subscribe(cond_var);

  while (RunDecompressLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void RootCopyHost2DeviceLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();

  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  QueueType this_op = COPYH2D;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->subscribe(cond_var);

  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetVisibleDevice()));
  while (RunRootCopyHost2DeviceLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void NonRootCopyListenLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();
  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetVisibleDevice()));
  while (RunNonRootCopyListenLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void NonRootCopyHost2DeviceLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();

  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  QueueType this_op = COPYH2D;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->subscribe(cond_var);

  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetVisibleDevice()));
  while (RunNonRootCopyHost2DeviceLoopOnce() &&
         !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void AllgatherCopyDevice2HostLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();

  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  QueueType this_op = ALLGATHER_COPYD2H;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->subscribe(cond_var);

  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetVisibleDevice()));
  while (RunAllgatherCopyDevice2HostLoopOnce() && 
         !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void AllgatherRootCopyHost2DeviceLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();

  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  QueueType this_op = ALLGATHER_COPYH2D;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->subscribe(cond_var);

  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetVisibleDevice()));
  while (RunAllgatherRootCopyHost2DeviceLoopOnce() && 
         !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void AllgatherNonRootCopyHost2DeviceLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();

  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  QueueType this_op = ALLGATHER_COPYH2D;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->subscribe(cond_var);

  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetVisibleDevice()));
  while (RunAllgatherNonRootCopyHost2DeviceLoopOnce() && 
         !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

#endif

bool RunPushLoopOnce() {
  QueueType this_op = PUSH;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->wait();
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

bool RunGDRv1PushPullLoopOnce() {
  QueueType this_op = GDR_V1_PUSH_PULL;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->wait();
  auto task = q->getTask();
  if (task) {
    BPS_CHECK_NE(task->device, CPU_DEVICE_ID); // must be GPU tensor
    auto local_rank = BytePSGlobal::GetLocalRank();
    auto local_size = BytePSGlobal::GetLocalSize();
    auto tensor = (local_size > 1) ? task->output : task->tensor;
    auto offset = task->offset; 
    auto len = task->len;    
    int dtype = task->tensor->dtype();
    auto unit_len = tensor->size() / tensor->shape().num_elements();

    auto num_elem_per_gpu = len / local_size / unit_len;
    int lrs_offset = local_rank * num_elem_per_gpu * unit_len; // lrs: local reduce scatter
    int push_len = num_elem_per_gpu * unit_len;
    auto left_elem = (len / unit_len) - (num_elem_per_gpu * local_size);

    if (left_elem && (local_rank == local_size - 1)) {
      // We assume the last GPU is the root.
      // If the assumption breaks, we should fix 
      // zpush and the associated reduce stage
      BPS_CHECK(BytePSGlobal::IsRootDevice()); 
      push_len += left_elem * unit_len;
    }

    char* data = (char*)(tensor->data()) + offset + lrs_offset;
    
    // note: grs stands for "global reduce scatter"
    auto grs_rank = BytePSGlobal::GetPhyNodeID();
    auto grs_size = BytePSGlobal::GetPhyNodeNum();
    for (int i = 0; i < grs_size; ++i) {
      auto num_elem_per_node = push_len / grs_size / unit_len;
      auto grs_left_elem = (push_len / unit_len) - (num_elem_per_node * grs_size);
      int grs_offset = i * num_elem_per_node * unit_len; 
      int grs_len = num_elem_per_node * unit_len;
      if (grs_left_elem && (i == grs_size - 1)) {
        grs_len += grs_left_elem * unit_len;
      }

      char* grs_push_data = data + grs_offset;
      char* grs_pull_data = (char*)task->output->data() + task->offset + lrs_offset + grs_offset;
      if (i == grs_rank) {
        server::BytePSServer::LocalPushPull(task->key, grs_push_data, grs_pull_data, grs_len, dtype);
      } else {
        ps::SArray<char> vals(grs_push_data, grs_len, false);
        int cmd = server::GetCommandType(server::RequestType::kGDRPushPull, dtype, GPU);
        int receiver = i * local_size + local_rank;
        auto pskv = BytePSGlobal::EncodeP2PKey(task->key, grs_len, receiver);
        BytePSGlobal::GetPS()->ZPush(pskv.keys, vals, pskv.lens, cmd,
            [task, grs_pull_data, grs_len, cmd, receiver]() { 
                auto pull_vals = new ps::SArray<char>(grs_pull_data, grs_len, false);
                auto pskv = BytePSGlobal::EncodeP2PKey(task->key, grs_len, receiver);
                BytePSGlobal::GetPS()->ZPull(pskv.keys, 
                    pull_vals, &pskv.lens, cmd, [task, pull_vals]() { 
                        delete pull_vals; 
                        int v = task->push_pull_counter_ptr->fetch_sub(1);
                        if (v == 1) {
                          FinishOrProceed(task); 
                        }
                    });
            });
      }
    }
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

bool RunGDRv2PushPullLoopOnce() {
  QueueType this_op = GDR_V2_PUSH_PULL;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->wait();
  auto task = q->getTask();
  if (!task) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    return true;
  }
  BPS_CHECK_NE(task->device, CPU_DEVICE_ID); // must be GPU tensor
  auto local_rank = BytePSGlobal::GetLocalRank();
  auto local_size = BytePSGlobal::GetLocalSize();
  auto tensor = (local_size > 1) ? task->output : task->tensor;
  auto offset = task->offset; 
  auto len = task->len;
  int dtype = task->tensor->dtype();
  auto unit_len = tensor->size() / tensor->shape().num_elements();
  auto num_elem_per_gpu = len / local_size / unit_len;
  
  // If BYTEPS_REDUCE_ROOTS is specified, we skip phase-1
  bool is_phase1_small_tensor = (len <= BytePSGlobal::GetGDRPhase1Threshold() && !BytePSGlobal::IsUsingReduce());
  if (is_phase1_small_tensor) {
    int global_reduce_root = BytePSGlobal::Hash_DJB2(task->key) % BytePSGlobal::GetSize();
    auto pskv = BytePSGlobal::EncodeP2PKey(task->key, len, global_reduce_root);
    char* push_data = (char*)(task->tensor->data()) + offset;
    char* pull_data = (char*)(task->output->data()) + offset;
    ps::SArray<char> push_vals(push_data + offset, len, false);
    int cmd = server::GetCommandType(server::RequestType::kGDRv2PushPullSmall, dtype, GPU);
    if (!BytePSGlobal::IsGDRKeyInited(task->key, global_reduce_root)) {
      BPS_LOG(DEBUG) << "rank " << BytePSGlobal::GetRank() 
                    << ": GDR init push, key " << task->key << ", len " << len
                    << ", receiver " << global_reduce_root 
                    << " (small tensor < threshold1)";
      BytePSGlobal::GetPS()->Wait(BytePSGlobal::GetPS()->ZPush(pskv.keys, push_vals, pskv.lens, cmd));
    }
    BytePSGlobal::GetPS()->ZPush(pskv.keys, push_vals, pskv.lens, cmd,
        [task, pull_data, len, cmd, global_reduce_root]() { 
            auto pull_vals = new ps::SArray<char>(pull_data, len, false);
            auto pskv = BytePSGlobal::EncodeP2PKey(task->key, len, global_reduce_root);
            BytePSGlobal::GetPS()->ZPull(pskv.keys, pull_vals, &pskv.lens, cmd, 
                [task, pull_vals, global_reduce_root]() { 
                  delete pull_vals; 
                  FinishOrProceed(task); 
                });
        });
    return true;
  }

  // lrs: local reduce scatter
  int lrs_offset = local_rank * num_elem_per_gpu * unit_len; 
  size_t comm_len = num_elem_per_gpu * unit_len;
  auto left_elem = (len / unit_len) - (num_elem_per_gpu * local_size);
  if (left_elem && (local_rank == local_size - 1)) {
    // We assume the last GPU is the root.
    // If the assumption breaks, we should fix 
    // zpush and the associated reduce stage
    BPS_CHECK(BytePSGlobal::IsRootDevice()); 
    comm_len += left_elem * unit_len;
  }

  if (BytePSGlobal::IsUsingReduce()) {
    lrs_offset = 0;
    comm_len = len;
    if (local_rank != BytePSGlobal::GetReduceRootByKey(task->key)) {
      // non reduce-root ranks skip this stage directly
      for (int k = 0; k < BytePSGlobal::GetPhyNodeNum(); ++k) {
        BytePSGlobal::GetGDRPushPullTable()->AddReadyCount(task->key);
      }
      FinishOrProceed(task);
      return true;
    }
  }

  char* data = (char*)(tensor->data()) + offset + lrs_offset;
  
  // note: grs stands for "global reduce scatter"
  auto grs_rank = BytePSGlobal::GetPhyNodeID();
  auto grs_size = BytePSGlobal::GetPhyNodeNum();
  auto num_elem_per_node = comm_len / grs_size / unit_len;
  auto grs_left_elem = (comm_len / unit_len) - (num_elem_per_node * grs_size);
  bool is_phase2_small_tensor = ((num_elem_per_node * unit_len) < BytePSGlobal::GetGDRPhase2Threshold());
  for (int j = 0; j < grs_size; ++j) {
    int i = (grs_rank + 1 + j) % grs_size; // form a ring to balance the traffic
    int grs_offset = i * num_elem_per_node * unit_len;    
    int grs_len = num_elem_per_node * unit_len;
    if (grs_left_elem && (i == grs_size - 1)) {
      grs_len += grs_left_elem * unit_len;
    }
    if (is_phase2_small_tensor) {
      grs_len = comm_len;
      grs_offset = 0;
      i = BytePSGlobal::GetGlobalReduceRoot(task->key);
      for (int k = 0; k < BytePSGlobal::GetPhyNodeNum() - 1; ++k) {
        BytePSGlobal::GetGDRPushPullTable()->AddReadyCount(task->key);
      }
    }
    BPS_LOG(TRACE) << "GDR_ALLREDUCE send info: rank " << BytePSGlobal::GetRank() 
        << ", grs_len=" << grs_len << ", i=" << i << ", grs_rank=" << grs_rank
        << ", key=" << task->key;
    char* grs_push_data = data + grs_offset;
    if (i == grs_rank) {
      char* input = (char*) task->tensor->data() + offset + lrs_offset + grs_offset;
      char* output = grs_push_data;
      if (grs_len) {
        server::BytePSServer::EnqueueLocalGpuSumTask(
            task->key, input, output, grs_len, dtype, /*do_copy*/(local_size==1));
      } else {
        BPS_CHECK(is_phase2_small_tensor);
      }
      if (is_phase2_small_tensor) break;
      continue;
    }
    char* grs_pull_data = (char*)task->output->data() + task->offset + lrs_offset + grs_offset;
    ps::SArray<char> vals(grs_push_data, grs_len, false);
    int cmd = server::GetCommandType(server::RequestType::kGDRv2PushPull, dtype, GPU);
    int receiver = i * local_size + local_rank;
    auto pskv = BytePSGlobal::EncodeP2PKey(task->key, grs_len, receiver);
    if (grs_len) {
      if (!BytePSGlobal::IsGDRKeyInited(task->key, receiver)) {
        BPS_LOG(DEBUG) << "rank " << BytePSGlobal::GetRank() 
                      << ": GDR init push, key " << task->key << ", len " << grs_len
                      << ", receiver " << receiver 
                      << (is_phase2_small_tensor ? " (small tensor < threshold2)" : ", partition ")
                      << (is_phase2_small_tensor ? "" : std::to_string(j));
        BytePSGlobal::GetPS()->Wait(
            BytePSGlobal::GetPS()->ZPush(pskv.keys, vals, pskv.lens, cmd));
      }
      BytePSGlobal::GetPS()->ZPush(pskv.keys, vals, pskv.lens, cmd,
          [task, grs_pull_data, grs_len, cmd, receiver]() { 
              auto pull_vals = new ps::SArray<char>(grs_pull_data, grs_len, false);
              auto pskv = BytePSGlobal::EncodeP2PKey(task->key, grs_len, receiver);
              BytePSGlobal::GetPS()->ZPull(pskv.keys, 
                  pull_vals, &pskv.lens, cmd, [task, pull_vals, receiver]() { 
                      // step 1: callbacks of zpull
                      delete pull_vals; 
                      BytePSGlobal::GetGDRPushPullTable()->AddReadyCount(task->key);
                  });
          });
    } else {
      BPS_CHECK(is_phase2_small_tensor);
      BytePSGlobal::GetGDRPushPullTable()->AddReadyCount(task->key);
    }
    if (is_phase2_small_tensor) break; // small tensor only push_pull once
  }
  FinishOrProceed(task); 
  return true;
}

bool RunPullLoopOnce() {
  QueueType this_op = PULL;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->wait();
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

bool RunGDRWaitLoopOnce() {
  QueueType wait_ops[] = { GDR_WAIT_PUSH_PULL };
  for (auto this_op : wait_ops) {
    auto q = BytePSGlobal::GetScheduledQueue(this_op);
    q->wait();
    auto task = q->getTask();
    if (task) {
      FinishOrProceed(task);
    } else {
      std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    }
  }
  return true;
}

void PushLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();

  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  QueueType this_op = PUSH;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->subscribe(cond_var);

  while (RunPushLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
}

void GDRv1PushPullLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();

  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  QueueType this_op = GDR_V1_PUSH_PULL;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->subscribe(cond_var);

  while (RunGDRv1PushPullLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
}

void GDRWaitLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();

  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  QueueType wait_ops[] = { GDR_WAIT_PUSH_PULL };
  for (auto this_op : wait_ops) {
    auto q = BytePSGlobal::GetScheduledQueue(this_op);
    q->subscribe(cond_var);
  }

  while (RunGDRWaitLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
}

void GDRv2PushPullLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();

  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  QueueType this_op = GDR_V2_PUSH_PULL;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->subscribe(cond_var);

  while (RunGDRv2PushPullLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
}

void PullLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();

  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  QueueType this_op = PULL;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->subscribe(cond_var);

  while (RunPullLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
}

// copy from comm buffer to output tensor (send/recv)
void P2PCopyGroup(std::vector<std::shared_ptr<TensorTableEntry>>& tasks) {
  bool has_gpu = false;
  for (auto& task : tasks) {
    auto key = task->key;
    auto len = task->len;
    bool is_gpu = CPU_DEVICE_ID != task->device;
    has_gpu = has_gpu || is_gpu;
    auto dst_addr = (char*)(task->output->data()) + task->offset;
    auto recv_arr = server::BytePSServer::GetRecvPartition(key);
    int recv_len = recv_arr.len;
    char* src_addr = (char*) recv_arr.val.data();
    // TODO(haibin.lin): support local send-recv
    BPS_CHECK((size_t) recv_len == len) << recv_len << ", " << len;
    BPS_LOG(TRACE) << "P2P_COPY key=" << key << " addr="
        << (long long) src_addr << " len=" << len;
    if (is_gpu) {
      BytePSGlobal::GetGpuReducer()->copy_h2d(dst_addr, src_addr, len, true);
    } else {
      BytePSGlobal::GetCpuReducer()->copy(dst_addr, src_addr, len);
    }
  }
  if (has_gpu) BytePSGlobal::GetGpuReducer()->sync_h2d();
  return;
}

// copy from comm buffer to output tensor (alltoall)
void P2PCopyGroup(std::vector<P2PTensorTableEntry*>& tasks) {
  int my_rank = BytePSGlobal::GetRank();
  bool has_d2d_copy = false;
  bool has_d2h_copy = false;
  bool has_h2d_copy = false;
  for (auto& task : tasks) {
    auto key = task->key;
    auto len = task->len;
    auto offset = task->offset;
    int sender = server::GetAlltoallSender(key);
    bool is_recv_cpu = CPU_DEVICE_ID == task->output_device;
    bool is_send_cpu = CPU_DEVICE_ID == task->device;
    has_d2d_copy = has_d2d_copy || (!is_recv_cpu && !is_send_cpu);
    has_d2h_copy = has_d2h_copy || (is_recv_cpu && !is_send_cpu);
    has_h2d_copy = has_h2d_copy || (!is_recv_cpu && is_send_cpu);
    auto dst_addr = task->output
              ? (char*)task->output->data() + task->offset_list[sender]
              : (char*)task->group_outputs[sender]->data();
    if (sender == my_rank) {
      // copy to myself
      BPS_LOG(TRACE) << "self H2D key=" << key << " offset=" << offset
                    << " len=" << len << " recv_cpu=" << is_recv_cpu;
      auto src_addr = task->tensor
                    ? (char*)task->tensor->data() + offset
                    : (char*)task->group_tensors[my_rank]->data();
      if (is_recv_cpu && is_send_cpu) {
        BytePSGlobal::GetCpuReducer()->copy(dst_addr, src_addr, len);
      } else {
        BytePSGlobal::GetGpuReducer()->copy(dst_addr, !is_recv_cpu, src_addr, !is_send_cpu, len, true);
      }
    } else { // not myself
      auto recv_arr = server::BytePSServer::GetRecvPartition(key);
      int recv_len = recv_arr.len;
      void* src_addr = recv_arr.val.data();
      BPS_CHECK((size_t) recv_len == len) << recv_len << ", " << len;
      // update the output (data)
      BPS_LOG(TRACE) << "ALLTOALL_COPY key=" << key << " addr="
          << (long long) src_addr << " len=" << len 
          << " recv_cpu=" << is_recv_cpu << " send_cpu=" << is_send_cpu;
      CHECK(src_addr) << key;
      if (is_recv_cpu) {
        BytePSGlobal::GetCpuReducer()->copy(dst_addr, src_addr, recv_len);
      } else {
        // ps-lite buffer is already on GPU
        BytePSGlobal::GetGpuReducer()->copy_d2d(dst_addr, src_addr, len, true);
      }
    }
  }
  if (has_d2d_copy) BytePSGlobal::GetGpuReducer()->sync_d2d();
  if (has_d2h_copy) BytePSGlobal::GetGpuReducer()->sync_d2h();
  if (has_h2d_copy) BytePSGlobal::GetGpuReducer()->sync_h2d();
  return;
}

bool RunRecvLoopOnce() {
  QueueType this_op = RECV;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->wait();
  std::vector<P2PTensorTableEntry*> alltoall_tasks;
  std::vector<std::shared_ptr<TensorTableEntry>> p2p_tasks;
  for (int i = 0; i < BytePSGlobal::GetP2PCopyGroupSize(); ++i) {
    BPS_LOG(TRACE) << "recv loop about to gettasklite";
    auto task = q->getTaskLite();
    if (!task) break;
    alltoall_tasks.push_back(reinterpret_cast<P2PTensorTableEntry*>(task));
  }
  for (int i = 0; i < BytePSGlobal::GetP2PCopyGroupSize(); ++i) {
    BPS_LOG(TRACE) << "recv loop about to gettask";
    auto task = q->getTask();
    if (!task) break;
    p2p_tasks.push_back(task);
  }
  if (alltoall_tasks.size() || p2p_tasks.size()) {
    if (!BytePSGlobal::IsSkipH2D()) {
      P2PCopyGroup(alltoall_tasks);
      P2PCopyGroup(p2p_tasks);
    }
    for (auto& task : alltoall_tasks) {
      if (BytePSGlobal::IsDirectResponse() == 0) {
        if (server::GetAlltoallSender(task->key) != BytePSGlobal::GetRank()) {
          server::BytePSServer::SendPushResponse(task->key);
        }
      }
      FinishOrProceedLite(task);
    }
    for (auto& task : p2p_tasks) {
      server::BytePSServer::SendPushResponse(task->key);
      FinishOrProceed(task);
    }
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

bool RunP2PGroupCopyHost2DeviceLoopOnce() {
  QueueType this_op = P2P_GROUP_COPYH2D;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->wait();
  auto t = q->getTaskLite();
  if (t) {
    auto task = reinterpret_cast<P2PTensorTableEntry*>(t);
    if (!BytePSGlobal::IsSkipH2D()) {
      int my_rank =  BytePSGlobal::GetRank();
      int num_ranks = BytePSGlobal::GetSize();
      const int dtype = task->tensor->dtype();
      BPS_CHECK(task->output->data() == nullptr);
      BPS_CHECK(task->device == CPU_DEVICE_ID);
      BPS_CHECK(task->output_device == CPU_DEVICE_ID);
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
      for (uint64_t i = 0; i < (uint64_t) num_ranks; ++i) {
        if (i != (uint64_t) my_rank) {
          uint64_t key = server::ComposeAlltoallKey(task->key, i);
          keys.push_back(key);
        }
      }
      // recv_arrs does not include the one for self send-recv
      std::vector<server::RecvArray> recv_arrs = server::BytePSServer::GetRecvPartitions(keys);
      // get the length of all ranks
      for (uint64_t i = 0; i < (uint64_t) num_ranks; ++i) {
        int64_t recv_len;
        if (i == (uint64_t) my_rank) {
          // self send-recv does not go through ps-lite
          recv_len = task->offset_list[my_rank + 1] - task->offset_list[my_rank];
          total_recv_len += recv_len;
        } else {
          int idx = i < (uint64_t) my_rank ? i : i - 1;
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
      for (size_t i = 1; i < in_shape.shape_.size(); ++i) {
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

// Note that we use shared_ptr on purpose to help manage the life cycle of the
// object. For alltoall we need to make sure the object is alive inside
// RunAlltoallSend, with shared_ptr the liveness is guaranteed.
// In earlier versions, manually recycling the memory of task object
// inside the ZPush callback may lead to segfault, when the number of
// valid ZPush is very small and the object is released while the for-loop
// in RunAlltoallSend is not completed yet.
void RunAlltoallSend(std::shared_ptr<TensorTableEntry>& t) {
  P2PTensorTableEntry* task = reinterpret_cast<P2PTensorTableEntry*>(t.get());
  // shuffle the receiver rank
  auto resp_mode = BytePSGlobal::IsDirectResponse();
  int my_rank =  BytePSGlobal::GetRank();
  int num_ranks = task->pcie_cpubuff.size();
  bool is_group = (task->group_tensors.size() > 0);
  const int dtype = task->tensor_dtype();
  bool output_size_unknown = task->output_size_unknown;
  int output_device = task->output_device == CPU_DEVICE_ID ? CPU : GPU;
  // the output tensor is not yet allocated. the receiver
  // must receive the entire group of tensors before copying
  // them to the output
  auto req_type = server::RequestType::kDefaultSend;
  if (output_size_unknown) {
    req_type = server::RequestType::kGroupSend;
  }
  std::function<void()> cb = nullptr;
  if (resp_mode != 2) {
    cb = [task, t]() {
      int v = task->request_counter.get()->fetch_sub(1);
      if (v == 1) FinishOrProceed(t);
    };
  }
  int cmd = server::GetCommandType(req_type, dtype, output_device);
  // used for group send with split=0
  int empty_cmd = server::GetCommandType(server::RequestType::kEmptyGroupSend,
                                         dtype, output_device);
  for (int r_offset = 0; r_offset < num_ranks; ++r_offset) {
    // start send with the next rank
    // TODO(haibin.lin): in the future, maybe we can start with the
    // slowest rank detected dynamically
    int i = (my_rank + r_offset + 1) % num_ranks;
    if (i == BytePSGlobal::GetRank()) continue;
    BPS_CHECK(task->offset_list.size() > (size_t) i + 1) << i << " " << task->offset_list.size();
    auto len = task->offset_list[i + 1] - task->offset_list[i];
    int receiver = i;
    if (len == 0) {
      // split = 0, nothing to copy
      if (output_size_unknown) {
        // when output size is unknown and split=0
        // we still perform send with 1 byte data using
        // the tensor name string, as it should be available during
        // byteps's life time
        ps::SArray<char> vals((char*)task->tensor_name.c_str(), 1, false);
        auto pskv = BytePSGlobal::EncodeP2PKey(task->key_list[i], 1, receiver);
        BytePSGlobal::GetPS()->ZPush(pskv.keys, vals, pskv.lens, empty_cmd, cb);
      }
      continue;
    } else {
      // split != 0
      const char* data = task->tensor_data(i);
      auto offset = is_group ? 0 : task->offset_list[i];
      auto pskv = BytePSGlobal::EncodeP2PKey(task->key_list[i], len, receiver);
      char* tensor = (char*) data + offset;
      int input_device = task->device;
      // do a copy (optional) followed by send
      if (input_device == CPU_DEVICE_ID && resp_mode == 2 &&
          !BytePSGlobal::ShouldSkipInputCopy()) {
        char* cpubuff = (char*) task->pcie_cpubuff[i];
        CHECK(cpubuff != nullptr);
        CopyD2H(cpubuff, data + offset, len, true);
        tensor = cpubuff;
      }
      // perform send. false means not to delete data when SArray is deleted
      ps::SArray<char> vals(tensor, len, false);
      BytePSGlobal::GetPS()->ZPush(pskv.keys, vals, pskv.lens, cmd, cb);
    }
  }
  if (resp_mode == 2) {
    FinishOrProceed(t);
  }
}

void RunP2PSend(std::shared_ptr<TensorTableEntry> task) {
  const int dtype = task->tensor->dtype();
  auto req_type = server::RequestType::kDefaultSend;
  int cmd = server::GetCommandType(req_type, dtype, CPU);
  char* data = (char*) task->tensor->data();
  auto offset = task->offset;
  int len = task->len;
  auto pskv = BytePSGlobal::EncodeP2PKey(task->key, len, task->context->receiver);
  char* cpubuff = (char*) task->cpubuff;
  CHECK(cpubuff != nullptr);
  bool from_cpu = task->device == CPU_DEVICE_ID;
  CopyD2H(cpubuff, data + offset, len, from_cpu);
  // perform send. false means not to delete data when SArray is deleted
  ps::SArray<char> vals(cpubuff, len, false);
  BytePSGlobal::GetPS()->ZPush(pskv.keys, vals, pskv.lens, cmd, [task]() {
    FinishOrProceed(task);
  });
}

bool RunSendLoopOnce() {
  // this loop is shared with alltoall and send/recv op
  QueueType this_op = SEND;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->wait();
  // get the send/recv or alltoall task
  BPS_LOG(TRACE) << "send loop about to gettask";
  auto task = q->getTask();
  if (task) {
    if (task->context->op_type == ALLTOALL_OP) {
      RunAlltoallSend(task);
    } else {
      BPS_CHECK(task->context->op_type == P2P_OP);
      RunP2PSend(task);
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
  q->wait();
  auto t = q->getTask();
  if (t) {
    auto task = reinterpret_cast<P2PTensorTableEntry*>(t.get());
    int my_rank = BytePSGlobal::GetRank();
    int num_ranks = task->pcie_cpubuff.size();
    int output_device_type = (task->output_device == CPU_DEVICE_ID) ? CPU : GPU;
    bool is_group = (task->group_tensors.size() > 0);
    for (int r = 0; r < num_ranks; ++r) {
      int i = (my_rank + r + 1) % num_ranks;
      if (i == my_rank) continue;
      auto len = task->offset_list[i+1] - task->offset_list[i];
      if (len == 0){
        continue;
      } else {
        auto pskv = BytePSGlobal::EncodeP2PKey(task->key_list[i], len, i);
        const char* dest = task->output_data(i);
        auto offset = is_group ? 0 : task->offset_list[i];
        const int dtype = task->output_dtype();
        int cmd = server::GetCommandType(server::RequestType::kDefaultPull, 
                                         dtype, output_device_type);
        int ack_cmd = server::GetCommandType(server::RequestType::kAckSignal,
                                             dtype, output_device_type);
        char* output = (char*) dest + offset;
        auto vals = new ps::SArray<char>(output, len, false);
        BytePSGlobal::GetPS()
          ->ZPull(pskv.keys, vals, &pskv.lens, cmd, [i, vals, task, t, ack_cmd]() {
            if (!BytePSGlobal::IsP2PAckDisabled()) {
              // notify the server that the response is received
              // We perform zpush with 1 byte data using tensor_name,
              // which should persist until byteps is shutdown
              char* cpubuff = const_cast<char*>(task->context->tensor_name.c_str());
              ps::SArray<char> ack_vals(cpubuff, 1, false);
              auto pskv = BytePSGlobal::EncodeP2PKey(task->key_list[i], 1, i);
              // no need to wait for this zpush
              BytePSGlobal::GetPS()->ZPush(pskv.keys, ack_vals, pskv.lens, ack_cmd);
            }
            // the rest of the callbacks are for zpull
            delete vals;
            int v = task->request_counter.get()->fetch_sub(1);
            if (v == 1) {
              FinishOrProceed(t);
            }
          });
      }
    }
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

bool RunP2PPullResponseOnce() {
  QueueType this_op = P2P_PULL_RESPONSE;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->wait();
  auto t = q->getTaskLite();
  if (t) {
    auto task = reinterpret_cast<P2PTensorTableEntry*>(t);
    int sender = server::GetAlltoallSender(task->key);
    int my_rank = BytePSGlobal::GetRank();
    bool is_recv_cpu = (CPU_DEVICE_ID == task->output_device);
    bool is_send_cpu = (CPU_DEVICE_ID == task->device);
    if (sender == my_rank) {
      // copy to myself
      bool is_group = (task->group_tensors.size() > 0);
      auto src_addr = (char*) task->tensor_data(my_rank);
      auto dst_addr = (char*) task->output_data(my_rank);
      if (!is_group) {
        src_addr += task->offset_list[my_rank];
        dst_addr += task->offset;
      }
      if (is_recv_cpu && is_send_cpu) {
        BytePSGlobal::GetCpuReducer()->copy(dst_addr, src_addr, task->len);
      } else {
        BytePSGlobal::GetGpuReducer()->copy(dst_addr, !is_recv_cpu, src_addr,
                                            !is_send_cpu, task->len, false);
      }
      if (!BytePSGlobal::IsP2PAckDisabled()) {
        BytePSGlobal::GetP2PAckTable()->AddReadyCount(task->key);
      }
    } else {
      char* tensor = (char*) task->tensor_data(sender) + task->offset;
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

bool RunP2PAckLoopOnce(QueueType this_op) {
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->wait();
  auto task = q->getTaskLite();
  if (task) {
    FinishOrProceedLite(task);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

void P2PPullLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();

  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  QueueType this_op = P2P_PULL;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->subscribe(cond_var);

  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetVisibleDevice()));
  while (RunP2PPullLoopOnce() &&
         !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
};

void P2PPullResponseLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();

  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  QueueType this_op = P2P_PULL_RESPONSE;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->subscribe(cond_var);

  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetVisibleDevice()));
  while (RunP2PPullResponseOnce() &&
         !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void P2PAckLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();

  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  auto q = BytePSGlobal::GetScheduledQueue(P2P_WAIT_ACK);
  q->subscribe(cond_var);

  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetVisibleDevice()));
  while (RunP2PAckLoopOnce(P2P_WAIT_ACK) &&
         !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void RecvLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();

  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  QueueType this_op = RECV;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->subscribe(cond_var);

  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetVisibleDevice()));
  while (RunRecvLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void P2PGroupCopyHost2DeviceLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();

  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  QueueType this_op = P2P_GROUP_COPYH2D;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->subscribe(cond_var);

  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetVisibleDevice()));
  while (RunP2PGroupCopyHost2DeviceLoopOnce() &&
         !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void SendLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();

  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  QueueType this_op = SEND;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->subscribe(cond_var);

  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetVisibleDevice()));
  while (RunSendLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

bool RunCpuCopyLoopOnce() {
  QueueType this_op = CPU_COPY;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->wait();
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
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();

  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  QueueType this_op = CPU_COPY;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->subscribe(cond_var);

  while (RunCpuCopyLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
}

bool RunCpuReduceLoopOnce() {
  QueueType this_op = CPU_REDUCE;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->wait();
  auto task = q->getTask();
  if (!task) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    return true;
  }
  auto reducer = BytePSGlobal::GetCpuReducer();

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
    BytePSCommSignal sig =  BytePSGlobal::IsDistributed() ?  PUSH_READY : CPU_BCAST_READY;
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
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();

  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  QueueType this_op = CPU_REDUCE;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->subscribe(cond_var);

  while (RunCpuReduceLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
}

bool RunCpuBcastLoopOnce() {
  QueueType this_op = CPU_BCAST;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->wait();
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
                  ((char *)(task->cpubuff) + offset), len);
  } else {
    auto tensor = task->tensor;

    auto key = task->key;
    int my_lrank = BytePSGlobal::GetLocalRank();
    auto basic_comm = BytePSGlobal::GetBasicComm();
    int local_root = basic_comm->getRoot();
    if (BytePSGlobal::IsTensorSampled(task->key)) {
      BPS_LOG(DEBUG) << "Sampled key=" << task->key << " local_root=" << local_root;
    }
    BPS_LOG(TRACE) << "dst " << (void *) ((char *)(task->cpubuff) + offset)
      << " src " << (void *) ((char*)(task->numa_cpubuff[local_root]) + offset) ;
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
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();

  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  QueueType this_op = CPU_BCAST;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->subscribe(cond_var);

  while (RunCpuBcastLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
}

bool RunCpuBcastFinishLoopOnce() {
  QueueType this_op = CPU_BCAST_FINISH;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->wait();
  auto task = q->getTask();
  if (!task) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
    return true;
  }

  FinishOrProceed(task);
  return true;
}

void CpuBcastFinishLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();

  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  QueueType this_op = CPU_BCAST_FINISH;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->subscribe(cond_var);

  while (RunCpuBcastFinishLoopOnce() && !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
}

inline void AllgatherPullRespTableUpdate(QueueType op, uint64_t key, const std::vector<uint64_t>& key_list) {
  if (op == ALLGATHER && BytePSGlobal::IsGDRAllgather() &&
      BytePSGlobal::IsDistributed() && BytePSGlobal::IsRootDevice()) {
    for (auto k : key_list) {
      if (k != key) {
        BytePSGlobal::GetAllgatherPullRespTable()->AddReadyCount(k);
      }
    }
  }
}

bool RunAllgatherPullLoopOnce() {
  QueueType this_op = ALLGATHER_PULL;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->wait();
  auto t = q->getTask();
  if (t) {
    auto task = reinterpret_cast<P2PTensorTableEntry*>(t.get());
    const int dtype = task->tensor->dtype();
    const int device_type = (task->device == CPU_DEVICE_ID) ? CPU : GPU;

    int local_size = BytePSGlobal::GetLocalSize();
    int phy_id = BytePSGlobal::GetPhyNodeID();
    int num_phy_node = BytePSGlobal::GetPhyNodeNum();
    
    auto len = task->len;
    auto unit_len = task->tensor->size() / task->tensor->shape().num_elements();

    const auto& shape_list = task->shape_list;
    const auto& offset_list = task->offset_list;
    if (!shape_list.empty())
      BPS_CHECK((int)offset_list.size() == BytePSGlobal::GetSize() + 1);

    auto& worker_local_root_list = task->worker_local_root_list;
    BPS_CHECK((int)worker_local_root_list.size() == num_phy_node);

    for (int r = 0; r < num_phy_node; ++r) {
      int i = (phy_id + r + 1) % num_phy_node;
      if (i == phy_id) continue;

      int rank_offset = i * local_size;
      int receiver = rank_offset + worker_local_root_list[i];

      if (len == 0)
        continue;
      else {
        // TODO: len is correct here ?
        auto pskv = BytePSGlobal::EncodeP2PKey(task->key, len, receiver);
        int cmd = server::GetCommandType(server::RequestType::kAllgatherPull, 
                                         dtype, device_type);
        int ack_cmd = server::GetCommandType(server::RequestType::kAllgatherPullAck,
                                             dtype, device_type);

        char* output = nullptr;
        if (BytePSGlobal::IsGDRAllgather()) {
          if (shape_list.empty()) {
            output = (char*)task->output->data() + rank_offset * len;
          } else {
            output = (char*)task->output->data() + offset_list[rank_offset] * unit_len;
          }
        } else {
          BPS_CHECK(task->cpubuff) << task->tensor_name
              << ": CPU buffer not initialized, size=" << task->len;
          if (shape_list.empty()) {
            output = const_cast<char *>(static_cast<const char *>(task->cpubuff) + rank_offset * len);
          } else {
            output = const_cast<char *>(static_cast<const char *>(task->cpubuff) + offset_list[rank_offset] * unit_len);
          }
        }
        BPS_CHECK(output) << task->tensor_name
            << ": output buffer not initialized, size=" << task->len;

        int pull_len = shape_list.empty() ? 
            local_size * len : (offset_list[rank_offset + local_size] - offset_list[rank_offset]) * unit_len;
        auto vals = new ps::SArray<char>(output, pull_len, false);
        BytePSGlobal::GetPS()
          ->ZPull(pskv.keys, vals, &pskv.lens, cmd, [receiver, vals, task, t, ack_cmd]() {
            if (!BytePSGlobal::IsP2PAckDisabled()) {
              // notify the server that the response is received
              // We perform zpush with 1 byte data using tensor_name,
              // which should persist until byteps is shutdown
              char* cpubuff = const_cast<char*>(task->context->tensor_name.c_str());
              ps::SArray<char> ack_vals(cpubuff, 1, false);
              auto pskv = BytePSGlobal::EncodeP2PKey(task->key, 1, receiver);
              // no need to wait for this zpush
              BytePSGlobal::GetPS()->ZPush(pskv.keys, ack_vals, pskv.lens, ack_cmd);
            }
            // the rest of the callbacks are for zpull
            delete vals;
            int v = task->request_counter.get()->fetch_sub(1);
            if (v == 1) {
              FinishOrProceed(t);
            }
          });
      }
    }
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

bool RunAllgatherPullRespLoopOnce() {
  QueueType this_op = ALLGATHER_PULL_RESP;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->wait();
  auto t = q->getTaskLite();
  if (t) {
    auto task = reinterpret_cast<P2PTensorTableEntry*>(t);
    BPS_CHECK(task->tensor);
    BPS_CHECK(task->output);

    int phy_id = BytePSGlobal::GetPhyNodeID();
    int local_size = BytePSGlobal::GetLocalSize();
    int rank_offset = phy_id * local_size;

    int len = task->len;
    int unit_len = task->tensor->size() / task->tensor->shape().num_elements();

    const auto& shape_list = task->shape_list;
    const auto& offset_list = task->offset_list;
    if (!shape_list.empty())
      BPS_CHECK((int)offset_list.size() == BytePSGlobal::GetSize() + 1);

    char* data = nullptr;
    if (BytePSGlobal::IsGDRAllgather()) {
      if (task->shape_list.empty()) {
        data = (char*) task->output->data() + rank_offset * len;
      } else {
        data = (char*) task->output->data() + offset_list[rank_offset] * unit_len;
      }
    } else {
      BPS_CHECK(task->cpubuff) << task->tensor_name
          << ": CPU buffer not initialized, size=" << len;
      if (shape_list.empty()) {
        data = const_cast<char *>(static_cast<const char *>(task->cpubuff) + rank_offset * len);
      } else {
        data = const_cast<char *>(static_cast<const char *>(task->cpubuff) + offset_list[rank_offset] * unit_len);
      }
    }

    int resp_len = shape_list.empty() ? 
        local_size * len : (offset_list[rank_offset + local_size] - offset_list[rank_offset]) * unit_len;
    server::BytePSServer::SendAllgatherPullResponse(task->key, data, resp_len);
    FinishOrProceedLite(task);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

bool RunAllgatherPullWorkerLocalRootLoopOnce() {
  QueueType this_op = ALLGATHER_PULL_WORKER_LOCAL_ROOT;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->wait();
  auto t = q->getTask();
  if (t) {
    auto task = reinterpret_cast<P2PTensorTableEntry*>(t.get());
    const int dtype = task->tensor->dtype();
    const int device_type = (task->device == CPU_DEVICE_ID) ? CPU : GPU;

    int local_size = BytePSGlobal::GetLocalSize();
    int phy_id = BytePSGlobal::GetPhyNodeID();
    int num_phy_node = BytePSGlobal::GetPhyNodeNum();

    task->worker_local_root_list.resize(num_phy_node, BytePSGlobal::GetWorkerLocalRoot());

    int len = sizeof(int);
    for (int r = 0; r < num_phy_node; ++r) {
      int i = (phy_id + r + 1) % num_phy_node;
      if (i == phy_id) continue;

      int receiver = i * local_size;
      auto pskv = BytePSGlobal::EncodeP2PKey(task->key, len, receiver);
      int cmd = server::GetCommandType(server::RequestType::kAllgatherPullWorkerLocalRoot, 
                                       dtype, device_type);
      int ack_cmd = server::GetCommandType(server::RequestType::kAllgatherPullWorkerLocalRootAck,
                                           dtype, device_type);

      int* output = new int;
      auto vals = new ps::SArray<char>((char*)output, len, false);
      BytePSGlobal::GetPS()
        ->ZPull(pskv.keys, vals, &pskv.lens, cmd, [receiver, vals, task, t, ack_cmd, i, output]() {
          if (!BytePSGlobal::IsP2PAckDisabled()) {
            char* cpubuff = const_cast<char*>(task->context->tensor_name.c_str());
            ps::SArray<char> ack_vals(cpubuff, 1, false);
            auto pskv = BytePSGlobal::EncodeP2PKey(task->key, 1, receiver);
            BytePSGlobal::GetPS()->ZPush(pskv.keys, ack_vals, pskv.lens, ack_cmd);
          }

          task->worker_local_root_list[i] = *output;

          delete output;
          delete vals;
          
          int v = task->allgather_pull_local_root_counter.get()->fetch_sub(1);
          if (v == 1) {
            FinishOrProceed(t);
          }
        });
    }
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

bool RunAllgatherPullWorkerLocalRootRespLoopOnce() {
  QueueType this_op = ALLGATHER_PULL_WORKER_LOCAL_ROOT_RESP;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->wait();
  auto t = q->getTaskLite();
  if (t) {
    auto task = reinterpret_cast<P2PTensorTableEntry*>(t);
    int len = 4;
    server::BytePSServer::SendAllgatherPullWorkerLocalRootResp(task->key, (char*)&task->context->worker_local_root, len);
    FinishOrProceedLite(task);
  } else {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
  }
  return true;
}

void AllgatherPullLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();


  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  QueueType this_op = ALLGATHER_PULL;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->subscribe(cond_var);

  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetVisibleDevice()));
  while (RunAllgatherPullLoopOnce() &&
         !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
};

void AllgatherPullRespLoop() {

  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  QueueType this_op = ALLGATHER_PULL_RESP;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->subscribe(cond_var);

  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetVisibleDevice()));
  while (RunAllgatherPullRespLoopOnce() &&
         !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void AllgatherPullAckLoop() {
  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  QueueType ack_ops[] = { ALLGATHER_PULL_ACK,
                          ALLGATHER_PULL_WORKER_LOCAL_ROOT_ACK, };
  for (auto this_op : ack_ops) {
    auto q = BytePSGlobal::GetScheduledQueue(this_op);
    q->subscribe(cond_var);
  }

  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetVisibleDevice()));
  // TODO: call RunP2PAckLoopOnce twice will cause low performance ?
  while (RunP2PAckLoopOnce(ALLGATHER_PULL_ACK) &&
         RunP2PAckLoopOnce(ALLGATHER_PULL_WORKER_LOCAL_ROOT_ACK) &&
         !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void AllgatherPullWorkerLocalRootLoop() {

  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  QueueType this_op = ALLGATHER_PULL_WORKER_LOCAL_ROOT;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->subscribe(cond_var);

  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetVisibleDevice()));
  while (RunAllgatherPullWorkerLocalRootLoopOnce() &&
         !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
};

void AllgatherPullWorkerLocalRootRespLoop() {

  auto cond_var = new CondVar(__PRETTY_FUNCTION__);
  BytePSGlobal::GetCondVarStore()->insert(cond_var);
  QueueType this_op = ALLGATHER_PULL_WORKER_LOCAL_ROOT_RESP;
  auto q = BytePSGlobal::GetScheduledQueue(this_op);
  q->subscribe(cond_var);

  CUDA_CALL(cudaSetDevice(BytePSGlobal::GetVisibleDevice()));
  while (RunAllgatherPullWorkerLocalRootRespLoopOnce() &&
         !BytePSGlobal::ShouldShutdown()) {
  }
  BytePSGlobal::ReportThreadFinish();
}

void MonitorLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();
  std::chrono::seconds interval(BytePSGlobal::GetMonitorInterval());
  std::unordered_map<uint64_t, TaskMetaMap> prev_tasks;
  while (!BytePSGlobal::ShouldShutdown() && !BytePSGlobal::WaitForShutdown(interval)) {
    std::unordered_map<uint64_t, TaskMetaMap> curr_tasks;
    // interate over all queues
    for (int i = 0; i < QueueNum; i++) {
      auto type = static_cast<QueueType>(i);
      auto queue = BytePSGlobal::GetScheduledQueue(type);
      if (!queue) continue;
      // queue exists
      queue->getPendingTasks(&curr_tasks);
    }
#if BYTEPS_BUILDING_CUDA == 1
    BytePSGlobal::GetNccl()->GetPendingTasks(&curr_tasks);
#endif
    for (auto& it : curr_tasks) {
      auto op_count = it.first;
      if (prev_tasks.find(op_count) == prev_tasks.end()) {
        continue;
      }
      // found tasks with matching op_count
      bool found = false;
      auto& prev_meta = prev_tasks[op_count];
      for (auto& name_meta : it.second) {
        auto& name = name_meta.first;
        if (prev_meta.find(name) != prev_meta.end()) {
          std::string next_queue = "none";
          auto& meta = name_meta.second;
          if (meta.next_queue_ != -1) {
            next_queue = LogStrings.at(meta.next_queue_);
          }
          // found a pending task
          BPS_LOG(INFO) << "rank=" << BytePSGlobal::GetRank()
                        << " pending task context=" << meta.ctx_->tensor_name
                        << " name=" << name << " count="
                        << op_count << " queue=" << next_queue;
          found = true;
        }
      }
      if (BytePSGlobal::ShouldAbortOnTimeout() && found) {
        LOG(FATAL) << "Aborting BytePS ...";
      }
    }
    prev_tasks = curr_tasks;
  }
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
  BytePSGlobal::ReportThreadFinish();
}

#if BYTEPS_BUILDING_CUDA == 0
void CoordinateLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
  BytePSGlobal::ReportThreadFinish();
}

void PcieReduceLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
  BytePSGlobal::ReportThreadFinish();
}

void RootNcclLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
  BytePSGlobal::ReportThreadFinish();
}

void NonRootNcclLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
  BytePSGlobal::ReportThreadFinish();
}

void SyncNcclLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
  BytePSGlobal::ReportThreadFinish();
}

void CopyDevice2HostLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
  BytePSGlobal::ReportThreadFinish();
}

void CompressLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
  BytePSGlobal::ReportThreadFinish();
}

void DecompressLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
  BytePSGlobal::ReportThreadFinish();
}

void RootCopyHost2DeviceLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
  BytePSGlobal::ReportThreadFinish();
}

void NonRootCopyListenLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
  BytePSGlobal::ReportThreadFinish();
}

void NonRootCopyHost2DeviceLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
  BytePSGlobal::ReportThreadFinish();
}

void AllgatherCopyDevice2HostLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
  BytePSGlobal::ReportThreadFinish();
}

void AllgatherRootCopyHost2DeviceLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
  BytePSGlobal::ReportThreadFinish();
}

void AllgatherNonRootCopyHost2DeviceLoop() {
  BPS_LOG(DEBUG) << "Started thread: " << __PRETTY_FUNCTION__ << " thread_id: "
                 << gettid();
  BPS_LOG(TRACE) << "Exiting thread: " << __PRETTY_FUNCTION__;
  BytePSGlobal::ReportThreadFinish();
}
#endif

}  // namespace common
}  // namespace byteps
