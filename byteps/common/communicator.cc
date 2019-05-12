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

void BytePSCommSocket::init(int* rank, int* size, int* local_rank, int* local_size, BytePSRole* my_role) {

    BPS_LOG(DEBUG) << "Using Communicator=Socket";

    // We should init rank, size, etc. using getenv
    // do env check
    BPS_CHECK(getenv("BYTEPS_LOCAL_RANK")) << "error: env BYTEPS_LOCAL_RANK not set";
    BPS_CHECK(getenv("BYTEPS_LOCAL_SIZE")) << "error: env BYTEPS_LOCAL_SIZE not set";
    BPS_CHECK(getenv("DMLC_WORKER_ID")) << "error: env DMLC_WORKER_ID not set";
    BPS_CHECK(getenv("DMLC_NUM_WORKER")) << "error: env DMLC_NUM_WORKER not set";

    *local_rank = atoi(getenv("BYTEPS_LOCAL_RANK"));
    *local_size = atoi(getenv("BYTEPS_LOCAL_SIZE"));
    *my_role = (_local_rank == (_local_size - 1)) ? LOCAL_ROOT : LOCAL_WORKER; // revisit here if root changes
    auto worker_id = atoi(getenv("DMLC_WORKER_ID"));
    auto num_worker = atoi(getenv("DMLC_NUM_WORKER"));

    // we assume _local_size (i.e., # GPU) is consistent on all workers
    *rank = (*local_rank) + worker_id * (*local_size);
    *size = num_worker * (*local_size);

    _rank = *rank;
    _size = *size;
    _local_rank = *local_rank;
    _local_size = *local_size;

    bool is_root = (*my_role==LOCAL_ROOT) ? true : false;

    // init the socket
    _recv_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    BPS_CHECK_GE(_recv_fd, 0) << "recv socket create failed";

    _send_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    BPS_CHECK_GE(_send_fd, 0) << "send socket create failed";

    int ret;

    // server
    struct sockaddr_un server_addr;
    memset(&server_addr, 0, sizeof(server_addr));

    // socket path name
    std::string server_fd_path;
    server_fd_path.append(BASE_SOCKET_PATH_RECV);
    server_fd_path += std::to_string(*local_rank); // should use the rank id to guarantee no conflict

    // filling addr information
    server_addr.sun_family = AF_UNIX;
    strncpy(server_addr.sun_path, server_fd_path.c_str(), sizeof(server_addr.sun_path)-1);

    // bind the socket to server_addr
    ret = bind(_recv_fd, (struct sockaddr *)&server_addr, sizeof(server_addr));
    BPS_CHECK_GE(ret, 0) << server_fd_path << " bind failed: " << strerror(errno);

    // not sure whether need this or not:
    // unlink(server_fd_path.c_str());

    BPS_LOG(DEBUG) << "This is " << (is_root ? "ROOT" : "WORKER")
                   << " device, socket create successfully";

    _listen_thread = new std::thread(&BytePSCommSocket::startListenThread, this);
}

void BytePSCommSocket::startListenThread() {
    BPS_LOG(DEBUG) << "Listening on socket " << _local_rank;
    char buffer[MAX_LINE];
    while (true) {
        int rc;
        while ((rc=read(_recv_fd, buffer, sizeof(buffer))) > 0) {
            BPS_LOG(TRACE) << "socket recved len=" << rc;
            // do the processing here
        }
        BPS_CHECK_GE(rc, 0);
    }
}

int BytePSCommSocket::sendSignal(int destination, void* data, int len) {
    struct sockaddr_un destaddr;
    memset(&destaddr, 0, sizeof(destaddr));
    destaddr.sun_family = AF_UNIX;

    std::string fd_path;
    fd_path.append(BASE_SOCKET_PATH_RECV);
    fd_path += std::to_string(destination);
    strncpy(destaddr.sun_path, fd_path.c_str(), sizeof(destaddr.sun_path)-1);

    auto ret = sendto(_send_fd, data, len, 0,
        (struct sockaddr *)&destaddr, sizeof(struct sockaddr_un));

    BPS_CHECK_GE(ret, 0);
    return 0;
}

int BytePSCommSocket::recvSignal(int* source, void* data, int len) {
    return 0;
}

int BytePSCommSocket::broadcastSignal(int root, void* data, int len) {
    return 0;
}

#ifdef BYTEPS_USE_MPI

void BytePSCommMPI::init(int* rank, int* size, int* local_rank, int* local_size, BytePSRole* my_role) {
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
    auto worker_id = atoi(getenv("DMLC_WORKER_ID"));
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

int BytePSCommMPI::sendSignal(int destination, void* data, int len) {
    return 0;
}

int BytePSCommMPI::recvSignal(int* source, void* data, int len) {
    return 0;
}

int BytePSCommMPI::broadcastSignal(int root, void* data, int len) {
    return 0;
}

#endif // BYTEPS_USE_MPI

}
}