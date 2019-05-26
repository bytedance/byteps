/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef PS_RESENDER_H_
#define PS_RESENDER_H_
#include <chrono>
#include <vector>
#include <unordered_set>
#include <unordered_map>
namespace ps {

/**
 * \brief resend a messsage if no ack is received within a given time
 */
class Resender {
 public:
  /**
   * \param timeout timeout in millisecond
   */
  Resender(int timeout, int max_num_retry, Van* van) {
    timeout_ = timeout;
    max_num_retry_ = max_num_retry;
    van_ = van;
    monitor_ = new std::thread(&Resender::Monitoring, this);
  }
  ~Resender() {
    exit_ = true;
    monitor_->join();
    delete monitor_;
  }

  /**
   * \brief add an outgoining message
   *
   */
  void AddOutgoing(const Message& msg) {
    if (msg.meta.control.cmd == Control::ACK) return;
    CHECK_NE(msg.meta.timestamp, Meta::kEmpty) << msg.DebugString();
    auto key = GetKey(msg);
    std::lock_guard<std::mutex> lk(mu_);
    // already buffered, which often due to call Send by the monitor thread
    if (send_buff_.find(key) != send_buff_.end()) return;

    auto& ent = send_buff_[key];
    ent.msg = msg;
    ent.send = Now();
    ent.num_retry = 0;
  }

  /**
   * \brief add an incomming message
   * \brief return true if msg has been added before or a ACK message
   */
  bool AddIncomming(const Message& msg) {
    // a message can be received by multiple times
    if (msg.meta.control.cmd == Control::TERMINATE) {
      return false;
    } else if (msg.meta.control.cmd == Control::ACK) {
      mu_.lock();
      auto key = msg.meta.control.msg_sig;
      auto it = send_buff_.find(key);
      if (it != send_buff_.end()) send_buff_.erase(it);
      mu_.unlock();
      return true;
    } else {
      mu_.lock();
      auto key = GetKey(msg);
      auto it = acked_.find(key);
      bool duplicated = it != acked_.end();
      if (!duplicated) acked_.insert(key);
      mu_.unlock();
      // send back ack message (even if it is duplicated)
      Message ack;
      ack.meta.recver = msg.meta.sender;
      ack.meta.sender = msg.meta.recver;
      ack.meta.control.cmd = Control::ACK;
      ack.meta.control.msg_sig = key;
      van_->Send(ack);
      // warning
      if (duplicated) LOG(WARNING) << "Duplicated message: " << msg.DebugString();
      return duplicated;
    }
  }

 private:
  using Time = std::chrono::milliseconds;
  // the buffer entry
  struct Entry {
    Message msg;
    Time send;
    int num_retry = 0;
  };
  std::unordered_map<uint64_t, Entry> send_buff_;

  uint64_t GetKey(const Message& msg) {
    CHECK_NE(msg.meta.timestamp, Meta::kEmpty) << msg.DebugString();
    uint16_t id = msg.meta.app_id;
    uint8_t sender = msg.meta.sender == Node::kEmpty ?
                     van_->my_node().id : msg.meta.sender;
    uint8_t recver = msg.meta.recver;
    return (static_cast<uint64_t>(id) << 48) |
        (static_cast<uint64_t>(sender) << 40) |
        (static_cast<uint64_t>(recver) << 32) |
        (msg.meta.timestamp << 1) | msg.meta.request;
  }
  Time Now() {
    return std::chrono::duration_cast<Time>(
        std::chrono::high_resolution_clock::now().time_since_epoch());
  }

  void Monitoring() {
    while (!exit_) {
      std::this_thread::sleep_for(Time(timeout_));
      std::vector<Message> resend;
      Time now = Now();
      mu_.lock();
      for (auto& it : send_buff_) {
        if (it.second.send + Time(timeout_) * (1+it.second.num_retry) < now) {
          resend.push_back(it.second.msg);
          ++it.second.num_retry;
          LOG(WARNING) << van_->my_node().ShortDebugString()
                       << ": Timeout to get the ACK message. Resend (retry="
                       << it.second.num_retry << ") " << it.second.msg.DebugString();
          CHECK_LT(it.second.num_retry, max_num_retry_);
        }
      }
      mu_.unlock();

      for (auto& msg : resend) van_->Send(msg);
    }
  }
  std::thread* monitor_;
  std::unordered_set<uint64_t> acked_;
  std::atomic<bool> exit_{false};
  std::mutex mu_;
  int timeout_;
  int max_num_retry_;
  Van* van_;
};
}  // namespace ps
#endif  // PS_RESENDER_H_
