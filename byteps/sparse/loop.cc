#include "loop.h"
#include "queue_exec.h"

namespace byteps {
namespace sparse {

void runDenseReduceLoop(QueueExecLoop * queue_loop_ptr) {
  queue_loop_ptr = QueueExecLoop::init_loop();
}

void runDenseReducePipeline(MemcpyD2HQueueExecLoop * denseD2HLoop,
                            CPUReduceQueueExecLoop * denseReduceLoop,
                            MemcpyH2DQueueExecLoop * denseH2DLoop,
                            ::byteps::common::CpuReducer * cpuReducer) {
  denseD2HLoop = MemcpyD2HQueueExecLoop::init_loop();
  denseReduceLoop = CPUReduceQueueExecLoop::init_loop(cpuReducer);
  denseH2DLoop = MemcpyH2DQueueExecLoop::init_loop();

  // Link exec_loops into a pipeline by setting downstream. Must explicitly set nullptr
  // as downstrea for the end stage.
  denseD2HLoop->set_downstream(denseReduceLoop);
  denseReduceLoop->set_downstream(denseH2DLoop);
  denseH2DLoop->set_downstream(nullptr);
}


} // namespace sparse
} // namespace byteps 
