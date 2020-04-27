// Copyright 2020 ByteDance, Inc. All Rights Reserved.
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

#ifndef BYTEPS_SPARSE_OPS_H
#define BYTEPS_SPARSE_OPS_H

#include "../common/operations.h"

namespace byteps {
namespace sparse {

using namespace byteps::common;

extern "C" int byteps_torch_poll(int handle);
extern "C" void byteps_torch_wait_and_clear(int handle);
extern "C" void byteps_torch_declare_tensor(char* name);

}  // namespace sparse
}  // namespace byteps

#endif  // BYTEPS_SPARSE_OPS_H
