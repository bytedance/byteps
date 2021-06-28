// Copyright 2021 Bytedance Inc. or its affiliates. All Rights Reserved.
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

#ifndef BYTEPS_PROFILER_H
#define BYTEPS_PROFILER_H

#include "common.h"
#include "logging.h"
#include <cmath>

namespace byteps {
namespace common {

using TimePoint = std::chrono::time_point<std::chrono::system_clock>;

struct MetricSummary {
  int capacity_;
  float sum_;
  float sum_square_;
  std::queue<float> data_;

  MetricSummary(int capacity) : capacity_(capacity), sum_(0), sum_square_(0) {
    BPS_CHECK(capacity_ > 0) << capacity_;
  }
  // update the summary with an entry
  void update(float current);
  // calculate the stdev
  float stdev() const;
  // calculate the mean
  float mean() const;
  // reset the summary and clear historical entries
  void reset();
  // the number of occurrences
  size_t size() { return data_.size(); }
};

struct Metric {
  TimePoint start_ts_; // start timestamp
  TimePoint end_ts_;   // end timestamp

  float latency() const {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_ts_ - start_ts_);
    return duration.count();
  }
};

class Telemetry {
 public:
  static bool ShouldRecord();

  // the number of traced operations so far
  static int size();
  // initialize the configuration based on env vars
  static void InitEnv();
  // start recording an operation
  static void RecordStart(const std::string& name);
  // end recording an operation
  static void RecordEnd(const std::string& name);
  // return the telemetry data
  static void GetData(const char** names, float* mean, float* stdev,
                      int* count, int* actual_size, int max_size);

 private:
  static std::mutex _mtx;
  static bool _should_record;

  // an operation is recorded every `_record_interval` occurrences
  static int _record_interval;
  // the summary of an operation is calculated for the most
  // recent `_record_capacity` occurrences
  static int _record_capacity;
  // occurrence count
  static std::unordered_map<std::string, uint64_t> _occurrences;
  // current metric
  static std::unordered_map<std::string, Metric> _metrics;
  // metric summaries
  static std::unordered_map<std::string, MetricSummary> _summaries;
  // collection of all names
  static std::vector<std::unique_ptr<std::string>> _names;
};

}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_PROFILER_H
