// Copyright 2019 Bytedance Inc. or its affiliates. All Rights Reserved.
// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

#ifndef BYTEPS_MXNET_READY_EVENT_H
#define BYTEPS_MXNET_READY_EVENT_H

#include <mxnet/ndarray.h>

#if HAVE_CUDA
#include <mutex>
#include <queue>
#include <unordered_map>
#include "cuda_runtime.h"

#include "../common/common.h"

namespace byteps {
namespace mxnet {

using namespace byteps::common;
typedef ::mxnet::NDArray NDArray;

template <class T>
class MXReadyEvent : public ReadyEvent {
 public:
  MXReadyEvent(NDArray* tensor);
  ~MXReadyEvent();
  virtual bool Ready() const override;

 private:
  NDArray* tensor_;
};

}  // namespace mxnet
}  // namespace byteps
#endif

#endif  // BYTEPS_MXNET_READY_EVENT_H
