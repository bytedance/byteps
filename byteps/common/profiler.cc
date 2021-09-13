// Copyright 2019 Bytedance Inc. or its affiliates. All Rights Reserved.
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

#include "profiler.h"
#include "logging.h"

namespace byteps {
namespace common {

std::mutex Telemetry::_mtx;
bool Telemetry::_should_record;
bool Telemetry::_enable_nvtx = false;
int Telemetry::_record_interval;
int Telemetry::_record_capacity;
std::unordered_map<std::string, uint64_t> Telemetry::_occurrences;
std::unordered_map<std::string, uint64_t> Telemetry::_completions;
std::unordered_map<std::string, Metric> Telemetry::_metrics;
std::unordered_map<std::string, MetricSummary> Telemetry::_summaries;
std::vector<std::unique_ptr<std::string>> Telemetry::_names;
#if BYTEPS_BUILDING_CUDA == 1
std::unordered_map<std::string, nvtxRangeId_t> Telemetry::_nvtx_ranges;
#endif

void MetricSummary::update(float current) {
  if (data_.size() > (size_t) capacity_) {
    auto front = data_.front();
    sum_ -= front;
    sum_square_ -= front * front;
  }
  data_.emplace(current);
  sum_ += current;
  sum_square_ += current * current;
}

float MetricSummary::stdev() const {
  auto count = data_.size();
  if (count == 0) return 0;
  // var(x) = E(x^2) - E(x)^2
  // stdev(x) = sqrt(var(x)) / (n - 1)
  float e_x = sum_ / count;
  float e_x2 = sum_square_ / count;
  return std::sqrt(e_x2 - e_x * e_x) / (count - 1);
}

float MetricSummary::mean() const {
  auto count = data_.size();
  if (count == 0) return 0;
  return sum_ / count;
}

void MetricSummary::reset() {
  sum_ = 0;
  sum_square_ = 0;
  data_ = std::queue<float>();
}

bool Telemetry::ShouldRecord() {
  return _should_record;
}

void Telemetry::InitEnv() {
  _should_record =
    getenv("BYTEPS_TELEMETRY_ON") ? atoi(getenv("BYTEPS_TELEMETRY_ON")) : false;
  _record_interval =
    getenv("BYTEPS_TELEMETRY_INTERVAL") ? atoi(getenv("BYTEPS_TELEMETRY_INTERVAL")) : 1;
  _record_capacity =
    getenv("BYTEPS_TELEMETRY_CAPACITY") ? atoi(getenv("BYTEPS_TELEMETRY_CAPACITY")) : 128;
  _enable_nvtx =
    getenv("BYTEPS_USE_NVTX") ? atoi(getenv("BYTEPS_USE_NVTX")) : false;
}

void Telemetry::GetData(const char** names, float* mean, float* stdev,
                        int* count, int* actual_size, int max_size) {
  std::lock_guard<std::mutex> lock(_mtx);
  size_t i = 0;
  for (; i < _names.size() && i < (size_t) max_size; ++i) {
    // note: we update names[i] with the pointer to the name c_string
    // such a pointer is valid during the lifecycle of BytePS
    names[i] = _names[i]->c_str();
    auto summary = _summaries.find(*_names[i]);
    if (summary == _summaries.end()) {
      mean[i] = 0;
      stdev[i] = 0;
      count[i] = 0;
    } else {
      mean[i] = summary->second.mean();
      stdev[i] = summary->second.stdev();
      count[i] = summary->second.size();
      summary->second.reset();
    }
  }
  *actual_size = i;
}

uint64_t Telemetry::RecordStart(const std::string& name) {
  std::lock_guard<std::mutex> lock(_mtx);
#if BYTEPS_BUILDING_CUDA == 1
  if (_enable_nvtx) {
    _nvtx_ranges[name] = nvtxRangeStartA(name.c_str());
  }
#endif
  uint64_t occurrence = ++_occurrences[name];
  if (!_should_record) return occurrence;
  if (occurrence % _record_interval == 0) {
    if (_metrics.find(name) == _metrics.end()) {
      BPS_CHECK(occurrence == 1) << occurrence;
      // initialize the name on first occurrence
      _names.emplace_back(new std::string(name));
      _summaries.emplace(name, _record_capacity);
    }
    // update the start timestamp
    _metrics[name].start_ts_ = std::chrono::system_clock::now();
  }
  return occurrence;
}

void Telemetry::RecordEnd(const std::string& name) {
  std::lock_guard<std::mutex> lock(_mtx);
#if BYTEPS_BUILDING_CUDA == 1
  if (_enable_nvtx) {
    nvtxRangeEnd(_nvtx_ranges[name]);
  }
#endif
  ++_completions[name];
  if (!_should_record) return;
  uint64_t occurrence = _occurrences[name];
  if (occurrence % _record_interval == 0) {
    // update the end timestamp
    _metrics[name].end_ts_ = std::chrono::system_clock::now();
    auto latency = _metrics[name].latency();
    auto summary = _summaries.find(name);
    BPS_CHECK(summary != _summaries.end());
    summary->second.update(latency);
  }
}

int Telemetry::size() {
  std::lock_guard<std::mutex> lock(_mtx);
  return _names.size();
}


}  // namespace common
}  // namespace byteps
