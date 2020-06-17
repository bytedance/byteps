#ifndef BYTEPS_SPARSE_QUEUE_EXEC_H
#define BYTEPS_SPARSE_QUEUE_EXEC_H

#include <functional>
#include <condition_variable>  // std::condition_variable
#include <functional>
#include <memory>
#include <mutex>  // std::mutex, std::unique_lock
#include <queue>  // std::queue
#include <thread>
#include <cuda_runtime.h>

#include "util.h"
#include "cpu_reducer.h"

namespace byteps {
namespace sparse {

class QueueExecLoop {
 public:
  static QueueExecLoop* init_loop();

  // This should be invoked by any participant that want to post a job.
  void add_worker(std::function<void()> job);

  // This should be only called at destructor or during test.
  void stop_executors();

 protected:
  QueueExecLoop() : running_(true) {}
  
  ~QueueExecLoop() {
    stop_executors();
  }

  void start_executors();

 private:

  // This is to enqueue and dequeue the forward jobs accordingly.
  std::mutex job_queue_mtx_;
  std::condition_variable job_queue_cv_;

  std::queue<std::function<void()>> job_queue_;  // Protected by job_queue_mtx_.
  volatile bool running_;

  // Since each TF session would not be able to consume everything, we would
  // like to make it more aggressive to consume all CPU/GPU via multi-session.
  std::vector<std::thread> background_job_polls_;
};

struct DenseTask{
  int workerID;
  int local_rank;
  size_t buffer_size; // In bytes.
  cudaStream_t streamH2D;
  cudaStream_t streamD2H;

  void * baseSrcPtr;
  void * cpuDenseDeltaPtr;
  void * cpuDenseLatestPtr;
  void * baseResultPtr;

  std::function<void(int local_rank)> allFinishCallback;
};


class PredefinedQueueExecLoop : public QueueExecLoop{
 public:
  void add_predefined_worker(DenseTask task);

  void set_downstream(PredefinedQueueExecLoop * downstream);

 protected:
  PredefinedQueueExecLoop() : QueueExecLoop() {}

 private:
  virtual void predefined_work(DenseTask task) = 0;
  PredefinedQueueExecLoop * downstream_;
};


class MemcpyH2DQueueExecLoop : public PredefinedQueueExecLoop{
 public:
  static MemcpyH2DQueueExecLoop* init_loop();

 private:
  MemcpyH2DQueueExecLoop() : PredefinedQueueExecLoop() {}

  void predefined_work(DenseTask task) override;
};

class CPUReduceQueueExecLoop : public PredefinedQueueExecLoop {
 public:
  static CPUReduceQueueExecLoop* init_loop(::byteps::common::CpuReducer* denseReducer);

 private:
  CPUReduceQueueExecLoop(::byteps::common::CpuReducer * denseReducer)
    : PredefinedQueueExecLoop(), _loopdenseReducer(denseReducer) {}

  void predefined_work(DenseTask task) override;

  ::byteps::common::CpuReducer* _loopdenseReducer;
};

class MemcpyD2HQueueExecLoop : public PredefinedQueueExecLoop{
 public:
  static MemcpyD2HQueueExecLoop* init_loop();

 private:
  MemcpyD2HQueueExecLoop() : PredefinedQueueExecLoop() {}

  void predefined_work(DenseTask task) override;
};


} // namespace sparse
} // namespace byteps 

#endif  // BYTEPS_SPARSE_QUEUE_EXEC_H