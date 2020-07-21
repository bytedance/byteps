// Copyright 2019 Amazon Inc. or its affiliates. All Rights Reserved.
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

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "../compressor_registry.h"
#include "vanilla_error_feedback.h"

namespace byteps {
namespace common {
namespace compressor {
namespace {
CompressorRegistry::Register reg(
    "vanilla_ef",
    [](const kwargs_t& kwargs, size_t size,
       DataType dtype) -> std::unique_ptr<Compressor> {
      // register cptr
      auto kwargs_clone = kwargs;
      kwargs_clone.erase("ef_type");
      auto cptr = CompressorRegistry::Create(kwargs_clone, size, dtype);
      BPS_CHECK_NE(cptr, nullptr);
      return std::unique_ptr<VanillaErrorFeedbackCompressor>(
          new VanillaErrorFeedbackCompressor(size, dtype, std::move(cptr)));
    });
}

VanillaErrorFeedbackCompressor::VanillaErrorFeedbackCompressor(
    size_t size, DataType dtype, std::unique_ptr<Compressor> cptr)
    : ErrorFeedback(size, dtype, std::move(cptr)) {
  _fd = open("lr.s", O_RDONLY);
  BPS_CHECK(_fd > 0) << "open lr.s failed, errno=" << strerror(errno);
  void* ptr = mmap(0, 8, PROT_READ, MAP_SHARED, _fd, 0);
  BPS_CHECK_NE(ptr, MAP_FAILED) << "mmap failed, errno=" << strerror(errno);
  _mm = ptr;
  _pre_lr = _cur_lr = *reinterpret_cast<double*>(_mm);
}

VanillaErrorFeedbackCompressor::~VanillaErrorFeedbackCompressor() {
  munmap(_mm, 8);
  close(_fd);
}

void VanillaErrorFeedbackCompressor::UpdateGradient(tensor_t grad) {
  _cur_lr = *reinterpret_cast<double*>(_mm);
  this->_cpu_reducer->sum(grad.data, _error.get(), grad.size,
                          static_cast<DataType>(grad.dtype),
                          (_pre_lr / _cur_lr));
  _pre_lr = _cur_lr;
}

}  // namespace compressor
}  // namespace common
}  // namespace byteps