#ifndef BYTEPS_SPARSE_LOOP_H
#define BYTEPS_SPARSE_LOOP_H

#include "queue_exec.h"

namespace byteps {
namespace sparse {

// void runDenseMemcpyH2DLoop(MemcpyH2DQueueExecLoop * queue_loop_ptr);

// void runDenseCPUReduceLoop(QueueExecLoop * queue_loop_ptr);

// void runDenseMemcpyD2HLoop(QueueExecLoop * queue_loop_ptr);

void runDenseReduceLoop(QueueExecLoop *& queue_loop_ptr);

void runDenseReducePipeline(MemcpyD2HQueueExecLoop *& _denseD2HLoop,
                            CPUReduceQueueExecLoop *& _denseReduceLoop,
                            MemcpyH2DQueueExecLoop *& _denseH2DLoop,
                            ::byteps::sparse::CpuReducer * cpuReducer,
                            std::mutex * _mtx_DenseLatestBuffers);

} // namespace sparse
} // namespace byteps 

#endif  // BYTEPS_SPARSE_LOOP_H