// Copyright 2019 Bytedance Inc. or its affiliates. All Rights Reserved.
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

enum class LogLevel { TRACE, DEBUG, INFO, WARNING, ERROR, FATAL };

#define LOG_LEVELS "TDIWEF"

// Always-on checking
#define BPS_CHECK(x) \
  if (!(x))          \
  byteps::common::LogMessageFatal(__FILE__, __LINE__) << "Check failed: " #x << ' '

#define BPS_CHECK_LT(x, y) BPS_CHECK((x) < (y))
#define BPS_CHECK_GT(x, y) BPS_CHECK((x) > (y))
#define BPS_CHECK_LE(x, y) BPS_CHECK((x) <= (y))
#define BPS_CHECK_GE(x, y) BPS_CHECK((x) >= (y))
#define BPS_CHECK_EQ(x, y) BPS_CHECK((x) == (y))
#define BPS_CHECK_NE(x, y) BPS_CHECK((x) != (y))
#define BPS_CHECK_NOTNULL(x)                                 \
  ((x) == NULL ? common::LogMessageFatal(__FILE__, __LINE__) \
                     << "Check  notnull: " #x << ' ',        \
   (x) : (x))  // NOLINT(*)

/*!
 * \brief Protected CUDA call.
 * \param func Expression to call.
 *
 * It checks for CUDA errors after invocation of the expression.
 */
#if BYTEPS_BUILDING_CUDA == 0
  #define CUDA_CALL(func) {;}
#else
  #define CUDA_CALL(func)                                          \
    {                                                              \
      cudaError_t e = (func);                                      \
      BPS_CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading) \
          << "CUDA: " << cudaGetErrorString(e);                    \
    }
#endif

/*
 * \brief Protected NCCL call.
 */
#if BYTEPS_BUILDING_CUDA == 0
  #define NCCLCHECK(cmd) {}
#else
  #define NCCLCHECK(cmd)                                                                   \
    {                                                                                      \
      ncclResult_t r = (cmd);                                                              \
      BPS_CHECK(r == ncclSuccess) << "NCCL error(" << r << "): " << ncclGetErrorString(r); \
    }
#endif



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
  bool unbuffered_ = false;
};

// LogMessageFatal ensures the process will exit in failure after
// logging this message.
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line);
  ~LogMessageFatal();
};

#define _BPS_LOG_TRACE byteps::common::LogMessage(__FILE__, __LINE__, byteps::common::LogLevel::TRACE)
#define _BPS_LOG_DEBUG byteps::common::LogMessage(__FILE__, __LINE__, byteps::common::LogLevel::DEBUG)
#define _BPS_LOG_INFO byteps::common::LogMessage(__FILE__, __LINE__, LogLevel::INFO)
#define _BPS_LOG_WARNING byteps::common::LogMessage(__FILE__, __LINE__, LogLevel::WARNING)
#define _BPS_LOG_ERROR byteps::common::LogMessage(__FILE__, __LINE__, LogLevel::ERROR)
#define _BPS_LOG_FATAL byteps::common::LogMessageFatal(__FILE__, __LINE__)

#define _LOG(severity) _BPS_LOG_##severity

#define _LOG_RANK(severity, rank) _BPS_LOG_##severity << "[" << rank << "]: "

#define GET_LOG(_1, _2, NAME, ...) NAME
#define BPS_LOG(...) GET_LOG(__VA_ARGS__, _LOG_RANK, _LOG)(__VA_ARGS__)
#define CUDA_BUILD_ERROR(...) BPS_LOG(FATAL) << "Please build BytePS with BYTEPS_WITH_GPU=1"

LogLevel MinLogLevelFromEnv();
bool LogTimeFromEnv();
bool LogUnbufferedFromEnv();

}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_LOGGING_H
