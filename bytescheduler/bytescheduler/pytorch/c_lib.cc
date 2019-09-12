#include "c_lib.h"
#include "common.h"
#include "assert.h"


namespace bytescheduler {
namespace torch {

int handle = 0;
std::unordered_map<int, std::shared_ptr<ReadyEvent>> ready_events;
std::mutex mutex;

extern "C" int bytescheduler_create_event(int device) {
    auto ready_event = RecordReadyEvent(device);
    handle++;
    std::lock_guard<std::mutex> guard(mutex);
    ready_events[handle] = ready_event;
    return handle;
}

extern "C" int bytescheduler_poll_event(int handle) {
    std::lock_guard<std::mutex> guard(mutex);
    auto ready_event = ready_events.find(handle);
    assert(ready_event != ready_events.end());
    int ready = ready_event->second->Ready() ? 1 : 0;
    if (ready) {
        ready_events.erase(ready_event);
    }
    return ready;
}


} // namespace torch
} // namespace bytescheduler



