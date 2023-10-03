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

#ifndef BYTEPS_COMMUNICATOR_H
#define BYTEPS_COMMUNICATOR_H

#include <errno.h>
#if BYTEPS_BUILDING_CUDA == 1
#include <nccl.h>
#endif
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>
#include <memory>
#include "logging.h"

#define BYTEPS_DEFAULT_BASE_SOCKET_PATH_RECV "/tmp/socket_recv_"
#define BYTEPS_DEFAULT_BASE_SOCKET_PATH_SEND "/tmp/socket_send_"
#define BYTEPS_COMM_MAX_LINE 8000

namespace byteps {
namespace common {
enum BytePSRole { LOCAL_ROOT, LOCAL_WORKER };

#define BPS_FOREACH_SIGNAL_NAME(ACTION)                                       \
  ACTION(REDUCE_READY)                                                        \
  ACTION(PCIE_REDUCE_READY)                                                   \
  ACTION(BCAST_READY)                                                         \
  ACTION(PUSH_READY)                                                          \
  ACTION(DO_REDUCE)                                                           \
  ACTION(DO_BROADCAST)                                                        \
  ACTION(DO_GROUP)                                                            \
  ACTION(DO_COPYH2D)                                                          \
  ACTION(CPU_REDUCE_READY)                                                    \
  ACTION(DO_CPU_REDUCE)                                                       \
  ACTION(CPU_BCAST_READY)                                                     \
  ACTION(DO_CPU_BCAST)                                                        \
  ACTION(CPU_BCAST_DONE)                                                      \
  ACTION(DO_ALLGATHER)                                                        \
  ACTION(ALLGATHER_REDAY)                                                     \
  ACTION(DO_ALLGATHER_BCAST)                                                  \
  ACTION(ALLGATHER_BCAST_READY)                                               \
  ACTION(ALLGATHER_COPY_D2H_READY)                                            \
  ACTION(DO_ALLGATHER_COPYH2D)                                                \
  ACTION(BARRIER)

#define BPS_DEFINE_SIGNAL(name) name,
#define BPS_DEFINE_SIGNAL_LOGSTR(name) #name,

enum BytePSCommSignal {
  BPS_FOREACH_SIGNAL_NAME(BPS_DEFINE_SIGNAL)
  SIGNAL_NUM_AND_NOT_A_REAL_SIGNAL_AND_MUST_BE_THE_LAST
};

const std::vector<std::string>
SigLogStrings = {
  BPS_FOREACH_SIGNAL_NAME(BPS_DEFINE_SIGNAL_LOGSTR)
};

struct BytePSCommMsg {
  int src;
  BytePSCommSignal signal;
  uint64_t key;
};

class BytePSComm {
 public:
  BytePSComm() { _comm = nullptr; }

  virtual void init(int* rank, int* size, int* local_rank, int* local_size,
                    int* worker_id, BytePSRole* my_role, int* num_phy_node,
                    int* phy_node_id) = 0;
  virtual int sendSignal(int destination, void* data, int len) = 0;
  virtual int sendSignalToRoot(void* data, int len) = 0;
  virtual int recvSignal(int* source, void* data, int max_len) = 0;
  virtual int recvSignalFromRoot(void* data, int max_len) = 0;
  virtual int broadcastSignal(void* data, int len) = 0;
  virtual void startListen() = 0;

  virtual int getRank() { return _rank; }
  virtual int getSize() { return _size; }
  virtual int getLocalRank() { return _local_rank; }
  virtual int getLocalSize() { return _local_size; }
  virtual int getWorkerID() { return _worker_id; }

  virtual std::vector<int> getMembers() { return _members; }
  virtual int getRoot() { return _root; }

 protected:
  int _rank;
  int _size;
  int _local_rank;
  int _local_size;
  int _worker_id;
  int _phy_node_id;
  int _num_phy_node;

  std::vector<int> _members;
  int _root = -1;

  void* _comm;
};

class BytePSCommSocket : public BytePSComm {
 public:
  BytePSCommSocket() {}
  BytePSCommSocket(std::shared_ptr<BytePSComm> comm,
                   const std::string& path_suffix,
                   const std::vector<int>& members);

  ~BytePSCommSocket() {
    if (_listen_thread) {
      _listen_thread->join();
    }
    close(_send_fd);
    close(_recv_fd);

    auto fd_path = _send_path + std::to_string(_local_rank);
    if (!std::remove(fd_path.c_str())) {
      BPS_LOG(DEBUG) << "Clear socket " << fd_path;
    }
    fd_path = _recv_path + std::to_string(_local_rank);
    if (!std::remove(fd_path.c_str())) {
      BPS_LOG(DEBUG) << "Clear socket " << fd_path;
    }

    BPS_LOG(DEBUG) << "Clear BytePSCommSocket"
                   << " (rank=" << _local_rank << ")";
  }

  void init(int* rank, int* size, int* local_rank, int* local_size,
            int* worker_id, BytePSRole* my_role, int* num_phy_node, int* phy_node_id);
  int sendSignal(int destination, void* data, int len);
  int sendSignalToRoot(void* data, int len);
  int recvSignal(int* source, void* data, int max_len);
  int recvSignalFromRoot(void* data, int max_len);
  int broadcastSignal(void* data, int len);

  int getSendFd() { return _send_fd; }
  int getRecvFd() { return _recv_fd; }

  void startListen();

  std::string getSendPath() { return _send_path; }
  std::string getRecvPath() { return _recv_path; }

 protected:
  void startListenThread();
  int initSocket(int rank, const std::string& path);

  std::thread* _listen_thread = nullptr;

  std::string _send_path;
  std::string _recv_path;
  int _recv_fd;
  int _send_fd;

  std::mutex _socket_mu;
};

}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMMUNICATOR_H
