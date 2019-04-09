// Copyright 2019 ByteDance Inc. or its affiliates. All Rights Reserved.
// Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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

#ifndef BYTEPS_LOGGING_H
#define BYTEPS_LOGGING_H

#include <sstream>
#include <string>

namespace byteps {
namespace common {

enum class LogLevel {
  TRACE, DEBUG, INFO, WARNING, ERROR, FATAL
};

#define LOG_LEVELS "TDIWEF"

class LogMessage : public std::basic_ostringstream<char> {
 public:
  LogMessage(const char* fname, int line, LogLevel severity);
  ~LogMessage();

 protected:
  void GenerateLogMessage(bool log_time);

 private:
  const char* fname_;
  int line_;
  LogLevel severity_;
};

// LogMessageFatal ensures the process will exit in failure after
// logging this message.
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line);
  ~LogMessageFatal();
};

#define _BPS_LOG_TRACE \
  LogMessage(__FILE__, __LINE__, LogLevel::TRACE)
#define _BPS_LOG_DEBUG \
  LogMessage(__FILE__, __LINE__, LogLevel::DEBUG)
#define _BPS_LOG_INFO \
  LogMessage(__FILE__, __LINE__, LogLevel::INFO)
#define _BPS_LOG_WARNING \
  LogMessage(__FILE__, __LINE__, LogLevel::WARNING)
#define _BPS_LOG_ERROR \
  LogMessage(__FILE__, __LINE__, LogLevel::ERROR)
#define _BPS_LOG_FATAL \
  LogMessageFatal(__FILE__, __LINE__)

#define _LOG(severity) _BPS_LOG_##severity

#define _LOG_RANK(severity, rank) _BPS_LOG_##severity << "[" << rank << "]: "

#define GET_LOG(_1, _2, NAME, ...) NAME
#define BPS_LOG(...) GET_LOG(__VA_ARGS__, _LOG_RANK, _LOG)(__VA_ARGS__)

LogLevel MinLogLevelFromEnv();
bool LogTimeFromEnv();

}
}

#endif // BYTEPS_LOGGING_H
