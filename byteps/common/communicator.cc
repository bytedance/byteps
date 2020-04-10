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

#include "communicator.h"
#include <cerrno>
#include <cstring>
#include "global.h"
#include "logging.h"

namespace byteps {
namespace common {

// Copy constructor that provides the option to reconfigure members.
// The ranks in members always use local_rank, regardless that the members
// may be a subset of all local ranks.
BytePSCommSocket::BytePSCommSocket(std::shared_ptr<BytePSComm> comm,
                                   const std::string& path_suffix,
                                   const std::vector<int>& members) {
  std::shared_ptr<BytePSCommSocket> sock_comm =
      std::static_pointer_cast<BytePSCommSocket>(comm);
  // TODO: use private members directly
  _rank = sock_comm->getRank();
  _size = sock_comm->getSize();
  _local_rank = sock_comm->getLocalRank();
  _local_size = sock_comm->getLocalSize();
  _worker_id = sock_comm->getWorkerID();
  _send_path = sock_comm->getSendPath() + path_suffix;
  _recv_path = sock_comm->getRecvPath() + path_suffix;
  _send_fd = initSocket(_local_rank, _send_path);
  _recv_fd = initSocket(_local_rank, _recv_path);

  _members = (members.size() > 0) ? members : sock_comm->getMembers();
  _root = _members.back();

  auto my_role = (_local_rank == _root) ? LOCAL_ROOT : LOCAL_WORKER;
  bool is_root = (my_role == LOCAL_ROOT) ? true : false;
  // init socket comm
  if (is_root) {  // root
    _listen_thread =
        new std::thread(&BytePSCommSocket::startListenThread, this);
  }

  BPS_LOG(DEBUG) << "This is " << path_suffix << (is_root ? " ROOT" : " WORKER")
                 << " device, rank=" << _local_rank
                 << ", all sockets create successfully";
}

void BytePSCommSocket::init(int* rank, int* size, int* local_rank,
                            int* local_size, int* worker_id,
                            BytePSRole* my_role) {
  BPS_LOG(DEBUG) << "Using Communicator=Socket";

  // We should init rank, size, etc. using getenv
  // do env check
  BPS_CHECK(getenv("BYTEPS_LOCAL_RANK"))
      << "error: env BYTEPS_LOCAL_RANK not set";
  BPS_CHECK(getenv("BYTEPS_LOCAL_SIZE"))
      << "error: env BYTEPS_LOCAL_SIZE not set";
  BPS_CHECK(getenv("DMLC_WORKER_ID")) << "error: env DMLC_WORKER_ID not set";
  BPS_CHECK(getenv("DMLC_NUM_WORKER")) << "error: env DMLC_NUM_WORKER not set";

  *local_rank = atoi(getenv("BYTEPS_LOCAL_RANK"));
  *local_size = atoi(getenv("BYTEPS_LOCAL_SIZE"));
  *worker_id = atoi(getenv("DMLC_WORKER_ID"));
  auto num_worker = atoi(getenv("DMLC_NUM_WORKER"));

  // we assume _local_size (i.e., # GPU) is consistent on all workers
  *rank = (*local_rank) + (*worker_id) * (*local_size);
  // force setting global rank
  *rank = getenv("BYTEPS_GLOBAL_RANK") ? atoi(getenv("BYTEPS_GLOBAL_RANK")) : *rank;
  *size = num_worker * (*local_size);

  _rank = *rank;
  _size = *size;
  _local_rank = *local_rank;
  _local_size = *local_size;
  _worker_id = *worker_id;

  for (int i = 0; i < _local_size; i++) {
    _members.push_back(i);
  }
  _root = _members.back();

  *my_role = (_local_rank == _root) ? LOCAL_ROOT : LOCAL_WORKER;
  bool is_root = (*my_role == LOCAL_ROOT) ? true : false;

  if (getenv("BYTEPS_SOCKET_PATH")) {
    _send_path = std::string(getenv("BYTEPS_SOCKET_PATH")) +
                 std::string("/socket_send_");
    _recv_path = std::string(getenv("BYTEPS_SOCKET_PATH")) +
                 std::string("/socket_recv_");
  } else {
    _send_path = std::string(DEFAULT_BASE_SOCKET_PATH_SEND);
    _recv_path = std::string(DEFAULT_BASE_SOCKET_PATH_RECV);
  }

  _send_fd = initSocket(_local_rank, _send_path);
  _recv_fd = initSocket(_local_rank, _recv_path);

  // init socket comm
  if (is_root) {  // root
    _listen_thread =
        new std::thread(&BytePSCommSocket::startListenThread, this);

    // Just in case launching root earlier than non-root
    // TODO: use retry instead of sleep
    // if (_local_size > 1)
    // std::this_thread::sleep_for(std::chrono::microseconds(1000000));
  }

  BPS_LOG(DEBUG) << "This is " << (is_root ? "ROOT" : "WORKER")
                 << " device, rank=" << _local_rank
                 << ", all sockets create successfully";
}

int BytePSCommSocket::initSocket(int rank, const std::string& path) {
  int fd;
  // init the socket
  fd = socket(AF_UNIX, SOCK_DGRAM, 0);
  BPS_CHECK_GE(fd, 0) << "recv socket create failed";

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));

  // TODO: use absolute unique socket path name (consider multi-tenancy)
  std::string fd_path(path);
  fd_path +=
      std::to_string(rank);  // should use the rank id to guarantee no conflict

  // filling addr information
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, fd_path.c_str(), sizeof(addr.sun_path) - 1);

  // before bind, clear the path first
  unlink(fd_path.c_str());

  // set recv timeout value for socket
  struct timeval tv;
  tv.tv_sec = 3; // in seconds
  tv.tv_usec = 0;
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);

  // bind the socket to addr
  int ret = bind(fd, (struct sockaddr*)&addr, sizeof(addr));
  BPS_CHECK_GE(ret, 0) << fd_path << " bind failed: " << strerror(errno);

  BPS_LOG(DEBUG) << "Init socket at " << fd_path;

  return fd;
}

void BytePSCommSocket::startListenThread() {  // only root starts this in
                                              // background thread
  BPS_LOG(DEBUG) << "Listening on socket " << _local_rank;
  char buffer[MAX_LINE];
  while (true) {
    int rc;
    while (true) {
      rc = recv(_recv_fd, buffer, sizeof(buffer), MSG_WAITALL);
      if (rc < 0 && errno == EINTR) continue;
      if (rc < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) { // timeout
        if (BytePSGlobal::ShouldShutdown()) break; // on exit
        else continue; // normal timeout
      }
      BPS_CHECK_GE(rc, 0) << std::strerror(errno) << ", rank=" << _local_rank;
      break;
    }
    if (BytePSGlobal::ShouldShutdown()) break;

    auto message = *(BytePSCommMsg*)buffer;

    switch (message.signal) {
      case REDUCE_READY:
        BytePSGlobal::GetReduceTable()->AddReadyCount(message.key);
        break;
      case PCIE_REDUCE_READY:
        BytePSGlobal::GetPcieReduceTable()->AddReadyCount(message.key);
        break;
      case BCAST_READY:
        BytePSGlobal::GetBroadcastTable()->AddReadyCount(message.key);
        break;
      case PUSH_READY:
        BytePSGlobal::GetPushTable()->AddReadyCount(message.key);
        break;
      default:
        BPS_CHECK(0) << "unsupported signal: " << message.signal;
    }

    BPS_LOG(TRACE) << "root socket recved: src=" << message.src
                   << ", signal=" << message.signal << ", key=" << message.key
                   << ", myrank=" << _local_rank;
  }
  BPS_LOG(DEBUG) << "listen thread joined"
                 << " (rank=" << _local_rank << ")";
}

int BytePSCommSocket::sendSignal(int destination, void* data, int len) {
  std::lock_guard<std::mutex> lock(_socket_mu);
  struct sockaddr_un destaddr;
  memset(&destaddr, 0, sizeof(destaddr));
  destaddr.sun_family = AF_UNIX;

  std::string fd_path(_recv_path);
  fd_path += std::to_string(destination);
  strncpy(destaddr.sun_path, fd_path.c_str(), sizeof(destaddr.sun_path) - 1);

  int ret = -1;
  while (ret < 0) {
    ret = sendto(_send_fd, data, len, 0, (struct sockaddr*)&destaddr,
                 sizeof(struct sockaddr_un));
    if (ret < 0) {
      BPS_LOG(DEBUG) << "Socket send error " << std::strerror(errno)
                     << ", rank=" << _local_rank;
      std::this_thread::sleep_for(std::chrono::microseconds(1000000));
    }
  }

  return ret;
}

int BytePSCommSocket::sendSignalToRoot(void* data, int len) {
  return sendSignal(_root, data, len);
}

int BytePSCommSocket::recvSignal(int* source, void* data, int max_len) {
  int rc;
  while (true) {
    rc = recv(_recv_fd, data, MAX_LINE, MSG_WAITALL);
    if (rc < 0 && errno == EINTR) continue;
    if (rc < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) { // timeout
        if (BytePSGlobal::ShouldShutdown()) break; // on exit
        else continue; // normal timeout
    }
    BPS_CHECK_GE(rc, 0) << std::strerror(errno) << ", rank=" << _local_rank;
    BPS_CHECK_LE(rc, max_len)
        << "recv_len=" << rc << ", but given max_len=" << max_len;
    break;
  }
  if (BytePSGlobal::ShouldShutdown()) return rc;
  auto message = *(BytePSCommMsg*)data;
  *source = message.src;

  BPS_LOG(TRACE) << "non-root socket recved: src=" << message.src
                 << ", signal=" << message.signal << ", key=" << message.key
                 << ", myrank=" << _local_rank;

  return rc;
}

int BytePSCommSocket::recvSignalFromRoot(void* data, int max_len) {
  int src;
  int rc = recvSignal(&src, data, max_len);
  if (BytePSGlobal::ShouldShutdown()) return rc;
  BPS_CHECK_EQ(src, _root) << "Non-root received signal from another non-root";
  return rc;
}

int BytePSCommSocket::broadcastSignal(void* data, int len) {
  for (int i : _members) {
    if (i == _local_rank) continue;
    sendSignal(i, (void*)data, len);
  }
  return 0;
}

}  // namespace common
}  // namespace byteps
