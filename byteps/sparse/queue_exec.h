#ifndef BYTEPS_SPARSE_QUEUE_EXEC_H
#define BYTEPS_SPARSE_QUEUE_EXEC_H

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

void QueueExecLoop::start_executors() {
  // The initialization will be ignored if the poller is already up.
  if (background_job_polls_.size() >= 1)
    return;

  running_ = true;

  auto background_job = [this] () {
    while (running_) {
      std::function<void()> job = nullptr;
      // The job is not executed in the critical section, otherwise it is not
      // different from mutex lock the whole thing.
      {
        std::unique_lock<std::mutex> lck(job_queue_mtx_);
        while (job_queue_.empty())
          job_queue_cv_.wait(lck);

        job = job_queue_.front();
        job_queue_.pop();
      }
      if (job != nullptr) {
        job();
      }
    }
  };
  background_job_polls_.emplace_back(background_job);
}

void QueueExecLoop::stop_executors() {
  running_ = false;
  auto exit_job = []() { 
      // BPS_LOG(INFO) << "Exiting the executor."; 
  };
  add_worker(exit_job);

  if (!background_job_polls_.empty()) {
    for (auto& job : background_job_polls_) {
      job.join();
    }
    background_job_polls_.clear();
  }
}

void QueueExecLoop::add_worker(std::function<void()> job) {
  std::unique_lock<std::mutex> lck(job_queue_mtx_);
  job_queue_.push(std::move(job));
  job_queue_cv_.notify_all();
}

QueueExecLoop* QueueExecLoop::init_loop() {
  auto queue_exec_ptr = new QueueExecLoop();
  return queue_exec_ptr;
}


} // namespace sparse
} // namespace byteps 

#endif  // BYTEPS_SPARSE_QUEUE_EXEC_H