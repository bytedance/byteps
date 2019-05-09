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
    // We should init rank, size, etc. using getenv
    BPS_LOG(DEBUG) << "Using Communicator=Socket";

    // do env check
    BPS_CHECK(getenv("BYTEPS_LOCAL_RANK")) << "error: env BYTEPS_LOCAL_RANK not set";
    BPS_CHECK(getenv("BYTEPS_LOCAL_SIZE")) << "error: env BYTEPS_LOCAL_SIZE not set";
    BPS_CHECK(getenv("DMLC_WORKER_ID")) << "error: env DMLC_WORKER_ID not set";
    BPS_CHECK(getenv("DMLC_NUM_WORKER")) << "error: env DMLC_NUM_WORKER not set";

    *local_rank = atoi(getenv("BYTEPS_LOCAL_RANK"));
    *local_size = atoi(getenv("BYTEPS_LOCAL_SIZE"));
    *my_role = (_local_rank == (_local_size - 1)) ? LOCAL_ROOT : LOCAL_WORKER;
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
    int socket_num =  is_root ? (_local_size-1) : 1;

    for (int i = 0; i < socket_num; ++i) {
        struct sockaddr_in sock_addr;
      	memset(&sock_addr, 0, sizeof(sock_addr));

        // filling server information
        sock_addr.sin_family = AF_INET; // IPv4
        sock_addr.sin_addr.s_addr = inet_addr("127.0.0.1"); // local

        // TODO: check if the port is available before allocating
        sock_addr.sin_port = htons(BASE_PORT + 2*(_local_rank+i) + (is_root?1:0)); // odd number for root, even number for non-root

        // create the socket
        int fd;
        BPS_CHECK_GE((fd = socket(AF_INET, SOCK_DGRAM, 0)), 0)
              << "socket " << i << " create failed";
        sockfd_list_.push_back(fd);
        // bind the socket to sock_addr
        BPS_CHECK_GE(bind(sockfd_list_.at(i), (struct sockaddr *)&sock_addr, sizeof(sock_addr)), 0)
              << "socket " << i << " bind failed";
    }
    BPS_LOG(DEBUG) << "This is " << (is_root ? "ROOT" : "WORKER") << " device, all "
                   << socket_num << " sockets create successfully";
}

int BytePSCommSocket::sendSignal(int destination, void* data, int len) {
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