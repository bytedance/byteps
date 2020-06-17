#include "queue_exec.h"

namespace byteps {
namespace sparse {

//********** QueueExecLoop **********//

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
  queue_exec_ptr->start_executors();
  return queue_exec_ptr;
}


//********** PredefinedQueueExecLoop **********//


void PredefinedQueueExecLoop::add_predefined_worker(DenseTask task){
  std::function<void()> job = [this, task] () {
    predefined_work(task);
    if (downstream_ != nullptr)
      downstream_->add_predefined_worker(task);
    else
      task.allFinishCallback(task.local_rank);
  };
  add_worker(job);
}

void PredefinedQueueExecLoop::set_downstream(PredefinedQueueExecLoop * downstream) {
  downstream_ = downstream;
}


//********** MemcpyH2DQueueExecLoop **********//

void MemcpyH2DQueueExecLoop::predefined_work(DenseTask task) {
  // Copy dense layer's param delta D2H.
  CUDA_CALL(cudaMemcpyAsync(task.cpuDenseDeltaPtr, task.baseSrcPtr,
                            task.buffer_size, cudaMemcpyDeviceToHost, task.streamD2H));
  CUDA_CALL(cudaStreamSynchronize(task.streamD2H));
}

MemcpyH2DQueueExecLoop * MemcpyH2DQueueExecLoop::init_loop(){
  auto loop = new MemcpyH2DQueueExecLoop();
  loop->start_executors();
  return loop;
}

//********** CPUReduceQueueExecLoop **********//

void CPUReduceQueueExecLoop::predefined_work(DenseTask task) {
  // CPU Work to reduce.
  _loopdenseReducer->sum(task.cpuDenseLatestPtr, task.cpuDenseDeltaPtr, task.buffer_size /* in bytes*/, DataType::BYTEPS_FLOAT32);
}

CPUReduceQueueExecLoop * CPUReduceQueueExecLoop::init_loop(::byteps::common::CpuReducer* denseReducer){
  auto loop = new CPUReduceQueueExecLoop(denseReducer);
  loop->start_executors();
  return loop;
}

//********** MemcpyD2HQueueExecLoop **********//

void MemcpyD2HQueueExecLoop::predefined_work(DenseTask task) {
  // Copy dense layer's latest param H2D.
  CUDA_CALL(cudaMemcpyAsync(task.baseResultPtr, task.cpuDenseLatestPtr, 
                            task.buffer_size, cudaMemcpyHostToDevice, task.streamD2H));
  CUDA_CALL(cudaStreamSynchronize(task.streamD2H));
}

MemcpyD2HQueueExecLoop * MemcpyD2HQueueExecLoop::init_loop(){
  auto loop = new MemcpyD2HQueueExecLoop();
  loop->start_executors();
  return loop;
}

} // namespace sparse
} // namespace byteps 
