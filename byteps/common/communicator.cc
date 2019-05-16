// Copyright 2019 ByteDance Inc. or its affiliates. All Rights Reserved.
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

#include "logging.h"
#include "communicator.h"
#include "global.h"
#include <cerrno>
#include <cstring>
namespace byteps {
namespace common {

int BytePSComm::broadcast(int root, void* data, int len) {
    // use _p2pGPUCopy here
    return 0;
}

int BytePSComm::reduce(int root, void* data, int len) {
    // call nccl here
    return 0;
}

int BytePSComm::_p2pGPUCopy(void* from, void* to, int len) {
    return 0;
}

void BytePSCommSocket::init(int* rank, int* size, int* local_rank, int* local_size, int* worker_id, BytePSRole* my_role) {

    BPS_LOG(DEBUG) << "Using Communicator=Socket";

    // We should init rank, size, etc. using getenv
    // do env check
    BPS_CHECK(getenv("BYTEPS_LOCAL_RANK")) << "error: env BYTEPS_LOCAL_RANK not set";
    BPS_CHECK(getenv("BYTEPS_LOCAL_SIZE")) << "error: env BYTEPS_LOCAL_SIZE not set";
    BPS_CHECK(getenv("DMLC_WORKER_ID")) << "error: env DMLC_WORKER_ID not set";
    BPS_CHECK(getenv("DMLC_NUM_WORKER")) << "error: env DMLC_NUM_WORKER not set";

    *local_rank = atoi(getenv("BYTEPS_LOCAL_RANK"));
    *local_size = atoi(getenv("BYTEPS_LOCAL_SIZE"));
    *worker_id = atoi(getenv("DMLC_WORKER_ID"));
    auto num_worker = atoi(getenv("DMLC_NUM_WORKER"));

    // we assume _local_size (i.e., # GPU) is consistent on all workers
    *rank = (*local_rank) + (*worker_id) * (*local_size);
    *size = num_worker * (*local_size);

    _rank = *rank;
    _size = *size;
    _local_rank = *local_rank;
    _local_size = *local_size;
    _worker_id = *worker_id;

    *my_role = (_local_rank == (_local_size - 1)) ? LOCAL_ROOT : LOCAL_WORKER;
    bool is_root = (*my_role==LOCAL_ROOT) ? true : false;

    // init socket comm
    if (is_root) { // root
        _send_fd = initSocket(_local_rank, BASE_SOCKET_PATH_SEND);
        _recv_fd = initSocket(_local_rank, BASE_SOCKET_PATH_RECV);

        BPS_LOG(DEBUG) << "This is ROOT device, sockets create successfully";
        _listen_thread = new std::thread(&BytePSCommSocket::startListenThread, this);
    } else { // non-root
        _send_fd   = initSocket(_local_rank, BASE_SOCKET_PATH_SEND);
        _recv_fd   = initSocket(_local_rank, BASE_SOCKET_PATH_RECV);
        _reduce_fd = initSocket(_local_rank, BASE_SOCKET_PATH_REDUCE);
        _bdcast_fd = initSocket(_local_rank, BASE_SOCKET_PATH_BDCAST);

        BPS_LOG(DEBUG) << "This is WORKER device, rank=" << _local_rank
                       << ", sockets create successfully";
    }
}

int BytePSCommSocket::initSocket(int rank, const char* path) {
    int fd;
    // init the socket
    fd = socket(AF_UNIX, SOCK_DGRAM, 0);
    BPS_CHECK_GE(fd, 0) << "recv socket create failed";

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));

    // TODO: use absolute unique socket path name (consider multi-tenancy)
    std::string fd_path;
    fd_path.append(path);
    fd_path += std::to_string(rank); // should use the rank id to guarantee no conflict

    // filling addr information
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, fd_path.c_str(), sizeof(addr.sun_path)-1);

    // before bind, clear the path first
    unlink(fd_path.c_str());

    // bind the socket to addr
    int ret = bind(fd, (struct sockaddr *)&addr, sizeof(addr));
    BPS_CHECK_GE(ret, 0) << fd_path << " bind failed: " << strerror(errno);

    return fd;
}


void BytePSCommSocket::startListenThread() { // only root starts this in background thread
    BPS_LOG(DEBUG) << "Listening on socket " << _local_rank;
    char buffer[MAX_LINE];
    while (true) {
        int rc;
        rc = recv(_recv_fd, buffer, sizeof(buffer), MSG_WAITALL);
        BPS_CHECK_GE(rc, 0) << std::strerror(errno) << ", rank=" << _local_rank;

        auto message = *(BytePSCommMsg*) buffer;
        BPS_CHECK_EQ(message.signal, REDUCE_READY) << message.signal;

        BytePSGlobal::AddReadyCount(message.key);

        BPS_LOG(TRACE) << "socket recved: src=" << message.src
                       << ", signal=" << message.signal
                       << ", key=" << message.key;
    }
}

int BytePSCommSocket::sendSignal(int destination, void* data, int len, BytePSCommFlag flag) {
    struct sockaddr_un destaddr;
    memset(&destaddr, 0, sizeof(destaddr));
    destaddr.sun_family = AF_UNIX;

    std::string fd_path;
    switch (flag) {
        case ROOT_SEND_TO_RECV:
        case NON_ROOT_SEND:
            fd_path.append(BASE_SOCKET_PATH_RECV);
            break;
        case ROOT_SEND_TO_REDUCE:
            fd_path.append(BASE_SOCKET_PATH_REDUCE);
            break;
        case ROOT_SEND_TO_BDCAST:
            fd_path.append(BASE_SOCKET_PATH_BDCAST);
            break;
        default:
            BPS_CHECK(0) << "inappropriate flag: " << flag;
    }
    fd_path += std::to_string(destination);
    strncpy(destaddr.sun_path, fd_path.c_str(), sizeof(destaddr.sun_path)-1);

    auto ret = sendto(_send_fd, data, len, 0,
        (struct sockaddr *)&destaddr, sizeof(struct sockaddr_un));

    BPS_CHECK_GE(ret, 0) << std::strerror(errno)
                         << ", rank=" << _local_rank;
    return 0;
}

int BytePSCommSocket::recvSignal(int* source, void* data, int max_len, BytePSCommFlag flag) {
    int rc;
    switch (flag) {
        case NON_ROOT_RECV:
        case ROOT_RECV:
            rc = recv(_recv_fd, data, MAX_LINE, MSG_WAITALL);
            break;
        case NON_ROOT_RECV_REDUCE:
            rc = recv(_reduce_fd, data, MAX_LINE, MSG_WAITALL);
            break;
        case NON_ROOT_RECV_BDCAST:
            rc = recv(_bdcast_fd, data, MAX_LINE, MSG_WAITALL);
            break;
        default:
            BPS_CHECK(0) << "inappropriate flag: " << flag;
    }
    BPS_CHECK_GE(rc, 0) << std::strerror(errno) << ", rank=" << _local_rank;
    BPS_CHECK_LE(rc, max_len) << "recv_len=" << rc << ", but given max_len=" << max_len;

    auto message = *(BytePSCommMsg*) data;
    *source = message.src;
    return rc;
}

int BytePSCommSocket::broadcastSignal(int root, void* data, int len, BytePSCommFlag flag) {
    for (int i = 0; i < _local_size; ++i) {
        if (i == _local_rank) continue;
        sendSignal(i, (void *)data, len, flag);
    }
    return 0;
}

#ifdef BYTEPS_USE_MPI

void BytePSCommMPI::init(int* rank, int* size, int* local_rank, int* local_size, int* worker_id, BytePSRole* my_role) {
    BPS_LOG(DEBUG) << "Using Communicator=MPI";

    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);

    _comm = (void*)malloc(sizeof(MPI_Comm));
    MPI_Comm* mpi_comm = (MPI_Comm*)_comm;

    MPI_Comm_dup(MPI_COMM_WORLD, mpi_comm);

    // Get MPI size to determine how many tensors to wait for before reducing.
    MPI_Comm_rank(*mpi_comm, local_rank);
    MPI_Comm_size(*mpi_comm, local_size);

    BPS_CHECK(getenv("DMLC_WORKER_ID")) << "error: env DMLC_WORKER_ID not set";
    BPS_CHECK(getenv("DMLC_NUM_WORKER")) << "error: env DMLC_NUM_WORKER not set";
    *worker_id = atoi(getenv("DMLC_WORKER_ID"));
    auto num_worker = atoi(getenv("DMLC_NUM_WORKER"));

    // we assume _local_size (i.e., # GPU) is consistent on all workers
    *rank = (*local_rank) + worker_id * (*local_size);
    *size = num_worker * (*local_size);

    _rank = *rank;
    _size = *size;
    _local_rank = *local_rank;
    _local_size = *local_size;

    return;
}

int BytePSCommMPI::sendSignal(int destination, void* data, int len, BytePSCommFlag flag) {
    return 0;
}

int BytePSCommMPI::recvSignal(int* source, void* data, int max_len, BytePSCommFlag flag) {
    return 0;
}

int BytePSCommMPI::broadcastSignal(int root, void* data, int len, BytePSCommFlag flag) {
    return 0;
}

#endif // BYTEPS_USE_MPI

}
}
