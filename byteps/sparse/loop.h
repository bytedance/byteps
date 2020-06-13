#ifndef BYTEPS_SPARSE_LOOP_H
#define BYTEPS_SPARSE_LOOP_H

#include "queue_exec.h"

namespace byteps {
namespace sparse {

void runDenseReduceLoop(QueueExecLoop * queue_loop_ptr);

} // namespace sparse
} // namespace byteps 

#endif  // BYTEPS_SPARSE_LOOP_H