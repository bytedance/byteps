// Copyright 2019 ByteDance Inc. All Rights Reserved.
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

#ifndef BYTEPS_OPERATIONS_H
#define BYTEPS_OPERATIONS_H

#include <functional>

#include "global.h"

namespace byteps {
namespace common {


// Check that byteps is initialized.
Status CheckInitialized();

extern "C" {

// C interface to initialize byteps.
void byteps_init(int rank, int local_rank, int size, int local_size);

// C interface to shut down byteps.
void byteps_shutdown();

// C interface to get index of current byteps process.
// Returns -1 if byteps is not initialized.
int byteps_rank();

// C interface to get index of current byteps process in the node it is on.
// Returns -1 if byteps is not initialized.
int byteps_local_rank();

// C interface to return number of byteps processes.
// Returns -1 if byteps is not initialized.
int byteps_size();

// C interface to return number of byteps processes in the node it is on.
// Returns -1 if byteps is not initialized.
int byteps_local_size();

// C interface to return flag indicating whether MPI multi-threading is
// supported. Returns -1 if byteps is not initialized.
}

Status EnqueueTensorPush(std::shared_ptr<OpContext> context,
                        std::shared_ptr<Tensor> input,
                        std::shared_ptr<ReadyEvent> ready_event,
                        const std::string &name, ps::Key key,
                        const int device, const int priority, const int version,
                        StatusCallback callback);

Status EnqueueTensorPull(std::shared_ptr<OpContext> context,
                        std::shared_ptr<Tensor> output,
                        std::shared_ptr<ReadyEvent> ready_event,
                        const std::string &name, ps::Key key,
                        const int device, const int priority, const int version,
                        StatusCallback callback);

Status InitTensor(std::shared_ptr<OpContext> context,
                    std::shared_ptr<Tensor> tensor,
                    std::shared_ptr<ReadyEvent> ready_event,
                    const std::string &name, const int device,
                    StatusCallback callback);

} // namespace common
} // namespace byteps

#endif // BYTEPS_OPERATIONS_H