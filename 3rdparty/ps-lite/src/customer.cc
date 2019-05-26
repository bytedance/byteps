/**
 *  Copyright (c) 2015 by Contributors
 */
#include "ps/internal/customer.h"
#include "ps/internal/postoffice.h"
#include "ps/internal/threadsafe_queue.h"
#include "ThreadPool.h"
#include <map>
#include <atomic>
#include <set>
#include <list>
#include <fstream>
namespace ps {
const int Node::kEmpty = std::numeric_limits<int>::max();
const int Meta::kEmpty = std::numeric_limits<int>::max();
int num_worker, num_server;
int init_sender_flag_=-1;
std::mutex mu_;
std::vector<std::list<Message> > buffered_push;
std::vector<std::list<Message> > buffered_pull;
std::unordered_map<uint64_t, std::atomic<bool> > is_push_finished;
std::unordered_map<uint64_t, std::set<int> > pull_collected_;
std::unordered_map<uint64_t, std::mutex > key_mu_;
std::vector<std::list<Message> > worker_buffer_;

Customer::Customer(int app_id, int customer_id, const Customer::RecvHandle& recv_handle)
    : app_id_(app_id), customer_id_(customer_id), recv_handle_(recv_handle) {
  Postoffice::Get()->AddCustomer(this);
  recv_thread_ = std::unique_ptr<std::thread>(new std::thread(&Customer::Receiving, this));
  // get the number of worker
  const char *val;
  val = Environment::Get()->find("DMLC_NUM_WORKER");
  num_worker = atoi(val);
  CHECK_GE(num_worker, 1);
  val = Environment::Get()->find("DMLC_NUM_SERVER");
  num_server = atoi(val);
  CHECK_GE(num_server, 1);
}

Customer::~Customer() {
  Postoffice::Get()->RemoveCustomer(this);
  Message msg;
  msg.meta.control.cmd = Control::TERMINATE;
  recv_queue_.Push(msg);
  recv_thread_->join();
}

int Customer::NewRequest(int recver) {
  std::lock_guard<std::mutex> lk(tracker_mu_);
  int num = Postoffice::Get()->GetNodeIDs(recver).size();
  tracker_.push_back(std::make_pair(num, 0));
  return tracker_.size() - 1;
}

void Customer::WaitRequest(int timestamp) {
  std::unique_lock<std::mutex> lk(tracker_mu_);
  tracker_cond_.wait(lk, [this, timestamp]{
      return tracker_[timestamp].first == tracker_[timestamp].second;
    });
}

int Customer::NumResponse(int timestamp) {
  std::lock_guard<std::mutex> lk(tracker_mu_);
  return tracker_[timestamp].second;
}

void Customer::AddResponse(int timestamp, int num) {
  std::lock_guard<std::mutex> lk(tracker_mu_);
  tracker_[timestamp].second += num;
}

bool Customer::IsValidPushpull(const Message &msg) {
  if (!msg.meta.control.empty()) return false;
  if (msg.meta.simple_app) return false;
  return true;
}

uint64_t Customer::GetKeyFromMsg(const Message &msg) { // should check if this is valid push/pull
  CHECK(IsValidPushpull(msg)) << "Perform key derivation on an invalid message";
  CHECK_GT(msg.data.size(), 0) << "Invalid data message: msg.data.size() is 0";
  uint64_t key = 0;
  uint64_t coef = 1;
  for (unsigned int i = 0; i < msg.data[0].size(); ++i) {
    key += coef * (uint8_t) msg.data[0].data()[i];
    coef *= 256; // 256=2^8 (uint8_t)
  }
  return key;
}

void Customer::ProcessPullRequest(int worker_id) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    LOG(INFO) << "Server inits Pull Thread-" << worker_id;
  }
  std::list<Message> pull_consumer;
  bool should_stop = false;
  while (!should_stop) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      CHECK_LE((unsigned int) worker_id, buffered_pull.size()) << worker_id << ", " << buffered_pull.size();
      pull_consumer.splice(pull_consumer.end(), buffered_pull[worker_id]);
      buffered_pull[worker_id].clear();
    }
    auto it = pull_consumer.begin();
    while (it != pull_consumer.end()) {
      Message &msg = *it;
      if (!msg.meta.control.empty() && msg.meta.control.cmd == Control::TERMINATE) {
        if (pull_consumer.size() == 1) {
          should_stop = true;
          break;
        }
        continue; // should first finish the requests that are still in the buffer
      }
      CHECK(!msg.meta.push);
      uint64_t key = GetKeyFromMsg(msg);
      std::lock_guard<std::mutex> lock(key_mu_[key]);
      if (is_push_finished[key].load() && (pull_collected_[key].find(worker_id) == pull_collected_[key].end())) {
        pull_collected_[key].insert(worker_id);
        if (pull_collected_[key].size() == (unsigned int) num_worker) {
          is_push_finished[key] = false;
          pull_collected_[key].clear();
        }
        recv_handle_(msg);
        it = pull_consumer.erase(it);
        break;
      } else {
        ++it;
      }
    }
  }
}

void Customer::ProcessPushRequest(int thread_id) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    LOG(INFO) << "Server inits Push Thread-" << thread_id;
  }
  std::unordered_map<uint64_t, int> push_finished_cnt;
  std::list<Message> push_consumer;
  bool should_stop = false;
  while (!should_stop) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      CHECK_LE((unsigned int) thread_id, buffered_push.size()) << thread_id << ", " << buffered_push.size();
      push_consumer.splice(push_consumer.end(), buffered_push[thread_id]);
      buffered_push[thread_id].clear();
    }
    auto it = push_consumer.begin();
    while (it != push_consumer.end()) {
      Message &msg = *it;

      if (!msg.meta.control.empty() && msg.meta.control.cmd == Control::TERMINATE) {
        if (push_consumer.size() == 1) {
          should_stop = true;
          break;
        }
        continue; // should first finish the requests that are still in the buffer
      }

      CHECK(msg.meta.push);
      uint64_t key = GetKeyFromMsg(msg);
      recv_handle_(msg);

      it = push_consumer.erase(it);

      // we probably don't need this, but just in case
      if (push_finished_cnt.find(key) == push_finished_cnt.end()) push_finished_cnt[key] = 0;

      // we assume the init has already been handled by main thread
      ++push_finished_cnt[key];

      if (push_finished_cnt[key] == num_worker) {
        std::lock_guard<std::mutex> lock(key_mu_[key]);
        is_push_finished[key] = true;
        push_finished_cnt[key] = 0;
      }

    }
  }
}

void Customer::ProcessResponse(int thread_id) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    LOG(INFO) << "Inits Thread-" << thread_id;
  }
  std::list<Message> consumer;
  while (true) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      consumer.splice(consumer.end(), worker_buffer_[thread_id]);
      worker_buffer_[thread_id].clear();
    }
    auto it = consumer.begin();
    while (it != consumer.end()) {
      Message &msg = *it;
      if (!msg.meta.control.empty() && msg.meta.control.cmd == Control::TERMINATE) { break; }
      recv_handle_(msg);
      CHECK(!msg.meta.request); // must be response
      std::lock_guard<std::mutex> lk(tracker_mu_);
      tracker_[msg.meta.timestamp].second++;
      tracker_cond_.notify_all();
      it = consumer.erase(it);
    }
  }
}

void Customer::Receiving() {
  const char *val;
  val = CHECK_NOTNULL(Environment::Get()->find("DMLC_ROLE"));
  std::string role(val);
  bool is_server = role == "server";
  val = Environment::Get()->find("ENABLE_SERVER_MULTIPULL");
  bool is_server_multi_pull_enabled = val ? atoi(val) : true; // default enabled
  val = Environment::Get()->find("ENABLE_WORKER_MULTIPULL");
  bool is_worker_multi_pull_enabled = val ? atoi(val) : false; // default disabled, has bug yet
  val = Environment::Get()->find("WORKER_THREAD_NUM");
  int worker_thread_num = val ? atoi(val) : 4;
  val = Environment::Get()->find("SERVER_PUSH_NTHREADS");
  int server_push_nthread = val ? atoi(val) : 1;
  val = Environment::Get()->find("ENABLE_PROFILING");

  if (is_server && is_server_multi_pull_enabled){ // server multi-thread
    LOG(INFO) << "Use seperate thread to process pull requests from each worker.";

    std::vector<std::thread *> push_thread;
    for (int i = 0; i < server_push_nthread; ++i) {
      std::list<Message> buf;
      buffered_push.push_back(buf);
    }
    CHECK_EQ(buffered_push.size(), (unsigned int) server_push_nthread);
    for (int i = 0; i < server_push_nthread; ++i) {
      std::lock_guard<std::mutex> lock(mu_);
      auto t = new std::thread(&Customer::ProcessPushRequest, this, i);
      push_thread.push_back(t);
    }
    LOG(INFO) << "Server uses " << server_push_nthread << " threads to process push requests.";

    std::vector<std::thread *> pull_thread;
    for (int i = 0; i < num_worker; ++i) {
      std::list<Message> buf;
      buffered_pull.push_back(buf);
    }
    CHECK_EQ(buffered_pull.size(), (unsigned int) num_worker);
    for (int i = 0; i < num_worker; ++i) {
      std::lock_guard<std::mutex> lock(mu_);
      auto t = new std::thread(&Customer::ProcessPullRequest, this, i);
      pull_thread.push_back(t);
    }

    LOG(INFO) << "All threads inited, ready to process message ";

    while (true) {
      Message recv;
      recv_queue_.WaitAndPop(&recv);
      if (!recv.meta.control.empty() && recv.meta.control.cmd == Control::TERMINATE) {
        Message terminate_msg;
        terminate_msg.meta.control.cmd = Control::TERMINATE;
        std::lock_guard<std::mutex> lock(mu_);
        for (auto buf : buffered_push) {
          buf.push_back(terminate_msg);
        }
        for (auto buf : buffered_pull) {
          buf.push_back(terminate_msg);
        }
        break;
      }
      if (!IsValidPushpull(recv)) {
        recv_handle_(recv);
        continue;
      }
      uint64_t key = GetKeyFromMsg(recv);

      if (is_push_finished.find(key) == is_push_finished.end()) { // the first push from worker0 (init)
        CHECK(recv.meta.push) << "key=" << key << ": pull arrives earlier than push!";

        recv_handle_(recv);
        is_push_finished[key] = true;

        // do check for init
        if (init_sender_flag_ == -1) {
          init_sender_flag_ = recv.meta.sender;
        } else {
          CHECK_EQ(init_sender_flag_, recv.meta.sender)
              << "Not correct init sequence, expected from sender " << init_sender_flag_
              << ", received from sender " << recv.meta.sender;
        }
        continue;
      } // init ends

      if (recv.meta.push) { // push: same key goes to same thread
        std::lock_guard<std::mutex> lock(mu_);
        buffered_push[(key/num_server)%server_push_nthread].push_back(recv);
      } else { // pull
        std::lock_guard<std::mutex> lock(mu_);
        int worker_id = (recv.meta.sender - 9) / 2; // worker id: 9, 11, 13 ...
        buffered_pull[worker_id].push_back(recv);
      }

    } // while

    // wait until the threads finish
    for (auto t : push_thread) t->join();
    for (auto t : pull_thread) t->join();

  } // server multi-thread

  else if (!is_server && is_worker_multi_pull_enabled) { // worker multithread
    CHECK_GE(worker_thread_num, 1);
    LOG(INFO) << "Use multiple threads to process pull responses.";
    std::vector<std::thread *> worker_thread;
    for (int i = 0; i < worker_thread_num; ++i) {
      std::list<Message> buf;
      worker_buffer_.push_back(buf);
      auto t = new std::thread(&Customer::ProcessResponse, this, i);
      worker_thread.push_back(t);
    }
    CHECK_EQ((unsigned int) worker_thread_num, worker_buffer_.size());
    while (true) {
      Message recv;
      recv_queue_.WaitAndPop(&recv);
      if (!recv.meta.control.empty() && recv.meta.control.cmd == Control::TERMINATE) {
        Message terminate_msg;
        terminate_msg.meta.control.cmd = Control::TERMINATE;
        std::lock_guard<std::mutex> lock(mu_);
        for (auto buf : worker_buffer_) {
          buf.push_back(terminate_msg);
        }
        break;
      }
      if (!IsValidPushpull(recv)) {
        recv_handle_(recv);
        if (!recv.meta.request) {
          std::lock_guard<std::mutex> lk(tracker_mu_);
          tracker_[recv.meta.timestamp].second++;
          tracker_cond_.notify_all();
        }
        continue;
      }
      int thread_id = GetKeyFromMsg(recv) % worker_thread_num; // a simple load balanced strategy
      std::lock_guard<std::mutex> lock(mu_);
      worker_buffer_[thread_id].push_back(recv);
    }
    for (auto t : worker_thread) {
      t->join();
    }
  }  // worker multithread

  else { // original
    LOG(INFO) << "Do not use thread pool for receiving.";

    while (true) {
      Message recv;
      recv_queue_.WaitAndPop(&recv);
      if (!recv.meta.control.empty() && recv.meta.control.cmd == Control::TERMINATE) {
        break;
      }
      recv_handle_(recv);
      if (!recv.meta.request) {
        std::lock_guard<std::mutex> lk(tracker_mu_);
        tracker_[recv.meta.timestamp].second++;
        tracker_cond_.notify_all();
      }
    }

  } // original
}

}  // namespace ps
