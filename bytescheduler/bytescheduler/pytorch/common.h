#ifndef BYTESCHEDULER_TORCH_COMMON_H
#define BYTESCHEDULER_TORCH_COMMON_H

namespace bytescheduler {
namespace torch {

// Device ID used for CPU.
#define CPU_DEVICE_ID (-1)

enum DeviceType { CPU, GPU };

class ReadyEvent {
public:
  virtual bool Ready() const = 0;
  virtual ~ReadyEvent() = default;
};

}
}

#endif // BYTESCHEDULER_TORCH_COMMON_H