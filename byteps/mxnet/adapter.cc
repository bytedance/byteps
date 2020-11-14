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

#if HAVE_CUDA
#include "cuda.h"
#endif

#include "adapter.h"
#include "cuda_util.h"
#include "tensor_util.h"

namespace byteps {
namespace mxnet {


template <class T>
MXTensor<T>::MXTensor(T* tensor) : tensor_(tensor) {}

template <class T>
const DataType MXTensor<T>::dtype() const {
  return TensorUtil::GetDType(tensor_);
}

template <class T>
const TensorShape MXTensor<T>::shape() const {
  auto shape = TensorUtil::GetShape(tensor_);
  if (shape.dims() == 0) {
    // Tensor with empty shape is a Tensor with no values in MXNet, unlike a
    // constant in TensorFlow. So, we inject a dummy zero dimension to make sure
    // that the number-of-elements calculation is correct.
    shape.AddDim(0);
  }
  return shape;
}

template <class T>
const void* MXTensor<T>::data() const {
  return TensorUtil::GetData(tensor_);
}

template <class T>
int64_t MXTensor<T>::size() const {
  return TensorUtil::GetSize(tensor_);
}

template class MXTensor<NDArray>;

}  // namespace mxnet
}  // namespace byteps
