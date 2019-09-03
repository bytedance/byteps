#ifndef BYTESCHEDULER_TORCH_C_LIB_H
#define BYTESCHEDULER_TORCH_C_LIB_H

#include <unordered_map>
#include "ready_event.h"
#include <mutex>

namespace bytescheduler {
namespace torch {

extern "C" int handle;
extern "C" std::unordered_map<int, std::shared_ptr<ReadyEvent>> ready_events;
extern "C" std::mutex mutex;

extern "C" int bytescheduler_create_event(int device);
extern "C" int bytescheduler_poll_event(int handle);

} // namespace torch
} // namespace bytescheduler

#endif // BYTESCHEDULER_TORCH_C_LIB_H

