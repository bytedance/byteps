#ifndef BYTEPS_SPARSE_QUEUE_EXEC_H
#define BYTEPS_SPARSE_QUEUE_EXEC_H

#include <functional>
#include <condition_variable>  // std::condition_variable
#include <functional>
#include <memory>
#include <mutex>  // std::mutex, std::unique_lock
#include <queue>  // std::queue
#include <thread>

namespace byteps {
namespace sparse {

class QueueExecLoop {
 public:
  static QueueExecLoop* init_loop();

  // This should be invoked by any participant that want to post a job.
  void add_worker(std::function<void()> job);

  // This should be only called at destructor or during test.
  void stop_executors();

 private:
  QueueExecLoop() : running_(true) {
  }
  ~QueueExecLoop() {
    stop_executors();
  }

  void start_executors();

  // This is to enqueue and dequeue the forward jobs accordingly.
  std::mutex job_queue_mtx_;
  std::condition_variable job_queue_cv_;

  std::queue<std::function<void()>> job_queue_;  // Protected by job_queue_mtx_.
  volatile bool running_;

  // Since each TF session would not be able to consume everything, we would
  // like to make it more aggressive to consume all CPU/GPU via multi-session.
  std::vector<std::thread> background_job_polls_;
};

} // namespace sparse
} // namespace byteps 

#endif  // BYTEPS_SPARSE_QUEUE_EXEC_H