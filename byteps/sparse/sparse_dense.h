#ifndef BYTEPS_SPARSE_DENSE_H
#define BYTEPS_SPARSE_DENSE_H

// Communication APIs for the dense layers in a sparse model.
#include "loop.h"
#include "cpu_reducer.h"

namespace byteps {
namespace sparse {

// Buffers for dense layers when calling DenseReduceAsync
static std::vector<void*> _denseDeltaBeforeReduceBuffers;   // In GPU
static std::vector<void*> _denseDeltaAfterReduceBuffers;    // In GPU
static void* _cpuDenseDeltaBuffers;
static void* _cpuDenseLatestBuffers;

static size_t _denseDeltaBufferLength;  // Unit in bytes.

static QueueExecLoop* _denseReduceLoop;
static ::byteps::common::CpuReducer* _denseReducer;

// The mutex for each GPU to access async reduce readiness.
static std::vector<std::mutex *> _signal_mtx_per_gpu;
static std::vector<std::condition_variable *> _signal_cv_per_gpu;
static std::vector<bool> _is_ready_per_gpu;

// This is not thread safe! The caller must make sure it's not called concurrently and from rank 0 to num_gpu-1.
extern "C" void bytepsSparseInitDensePerGPU(int device_id /* starts with 0 */,
                                            void* denseDeltaBeforeReduceBuffer,
                                            void* denseDeltaAfterReduceBuffer,
                                            int sizeDenseDelta);
extern "C" void bytepsSparseInitDense(std::vector<void*>& denseDeltaBeforeReduceBuffers,
                                      std::vector<void*>& denseDeltaAfterReduceBuffers,
                                      int sizeDenseDelta);
extern "C" void bytepsDenseReduceExecAsync(int local_rank, cudaStream_t stream);
extern "C"  void bytepsDenseSynchronize(int local_rank, cudaStream_t stream);

} // namespace sparse
} // namespace byteps 

#endif  // BYTEPS_SPARSE_DENSE_H
