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

#ifndef BYTEPS_COMPRESSOR_UTILS_H
#define BYTEPS_COMPRESSOR_UTILS_H

#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>

#include "common.h"

namespace byteps {
namespace common {
namespace compressor {

/*!
 * \brief serialize key-vals hyper-params for network transmission
 *
 * \param kwargs hyper-params
 * \return std::string serialized data
 */
inline std::string Serialize(const kwargs_t& kwargs) {
  std::ostringstream os;
  os << kwargs.size();
  for (auto const& kwarg : kwargs) {
    os << " " << kwarg.first << " " << kwarg.second;
  }
  return os.str();
}

/*!
 * \brief deserialize serialized data into key-vals hyper-params
 *
 * \param content serialized data
 * \return kwargs_t hyper-params
 */
inline kwargs_t Deserialize(const std::string& content) {
  kwargs_t kwargs;
  std::istringstream is(content);
  size_t size = 0;
  is >> size;
  for (size_t i = 0; i < size; ++i) {
    kwargs_t::key_type key;
    kwargs_t::mapped_type val;
    is >> key >> val;
    kwargs[key] = val;
  }

  return kwargs;
}

/*!
 * \brief random number generator based on xorshift128plus
 *
 * refer to https://en.wikipedia.org/wiki/Xorshift#xorshift+
 */
class XorShift128PlusBitShifterRNG {
 public:
  XorShift128PlusBitShifterRNG() {
    std::random_device rd;
    _state = {rd(), rd()};
  }

  // uniform int among [low, high)
  uint64_t Randint(uint64_t low, uint64_t high) {
    return xorshift128p() % (high - low) + low;
  };

  // uniform [0, 1]
  double Rand() { return double(xorshift128p()) / MAX; }

  // Bernoulli Distributation
  bool Bernoulli(double p) { return xorshift128p() < uint64_t(p * MAX); }

  void set_seed(uint64_t seed) { _state = {seed, seed}; }

 private:
  struct xorshift128p_state {
    uint64_t a, b;
  };

  uint64_t xorshift128p() {
    uint64_t t = _state.a;
    uint64_t const s = _state.b;
    _state.a = s;
    t ^= t << 23;        // a
    t ^= t >> 17;        // b
    t ^= s ^ (s >> 26);  // c
    _state.b = t;
    return t + s;
  };

  xorshift128p_state _state;

  static constexpr uint64_t MAX = std::numeric_limits<uint64_t>::max();
};

}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESSOR_UTILS_H