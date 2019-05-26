/**
 *  Copyright (c) 2019 by Chang Lan, Yimin Jiang, Jingrong Chen
 */
#ifndef PS_RDMA_VAN_H_
#define PS_RDMA_VAN_H_

#ifdef DMLC_USE_RDMA

#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <rdma/rdma_cma.h>

#include <algorithm>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "ps/internal/threadsafe_queue.h"
#include "ps/internal/van.h"

namespace ps {

static const int kStartDepth = 128;
static const int kWriteDepth = kStartDepth;

static const int kRxDepth = kStartDepth * 2;
static const int kReplyDepth = kRxDepth;

static const int kSGEntry = 4;
static const int kTimeoutms = 1000;
static const int kRdmaListenBacklog = 128;
static const int kMaxConcurrentWorkRequest =
    kRxDepth + kStartDepth + kReplyDepth + kWriteDepth;
static const int kMaxHostnameLength = 16;
static const int kMaxDataFields = 4;
static const size_t kAlignment = 8;

template <typename T>
static inline T align_floor(T v, T align) {
  return v - (v % align);
}

template <typename T>
static inline T align_ceil(T v, T align) {
  return align_floor(v + align - 1, align);
}

class SimpleMempool {
 public:
  explicit SimpleMempool(struct ibv_pd *pd, size_t size = 0x10000000) {
    std::lock_guard<std::mutex> lk(mu_);
    pd_ = pd;
    struct ibv_mr *mr;
    char *p = reinterpret_cast<char *>(aligned_alloc(kAlignment, size));
    total_allocated_size += size;
    CHECK(p);
    CHECK(mr = ibv_reg_mr(pd, p, size,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));
    mr_list.emplace(p+size, mr); // this mr is associated with memory address range [p, p+size]
    free_list.emplace(size, p);
  }

  ~SimpleMempool() {
    std::lock_guard<std::mutex> lk(mu_);
    for(auto it = mr_list.begin(); it != mr_list.end(); it++){
      CHECK_EQ(ibv_dereg_mr(it->second), 0);
      free(it->second->addr);
    }
  }

  char *Alloc(size_t size) {
    if (size == 0) {
      return nullptr;
    }

    std::lock_guard<std::mutex> lk(mu_);

    size_t proper_size = align_ceil(size, kAlignment);

    auto it = free_list.lower_bound(proper_size);

    if(it == free_list.end()) { // if there is no space left, need to allocate and register new memory
      size_t new_mem_size = total_allocated_size;
      while(proper_size > new_mem_size) {
        new_mem_size *= 2;
      }
      char *p = reinterpret_cast<char *>(aligned_alloc(kAlignment, new_mem_size));
      CHECK(p);
      struct ibv_mr *mr;
      CHECK(mr = ibv_reg_mr(pd_, p, new_mem_size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));
      mr_list.emplace(p+new_mem_size, mr);
      free_list.emplace(new_mem_size, p);
      it = free_list.lower_bound(proper_size);
      PS_VLOG(1) << "Not enough memory in the pool, requested size " << proper_size << ", new allocated size " << new_mem_size;
      total_allocated_size += new_mem_size;
    }

    CHECK_NE(free_list.end(), it) << "Not enough memory";
    CHECK_GE(it->first, proper_size);

    char *addr = it->second;
    size_t space_left = it->first - proper_size;

    free_list.erase(it);
    CHECK_EQ(used_list.find(addr), used_list.end())
        << "Address is already allocated";

    used_list.emplace(addr, proper_size);

    if (space_left) {
      free_list.emplace(space_left, addr + proper_size);
    }

    return addr;
  }

  void Free(char *addr) {
    if (!addr) {
      return;
    }

    std::lock_guard<std::mutex> lk(mu_);

    auto it = used_list.find(addr);
    CHECK_NE(used_list.end(), it)
        << "Cannot find info about address: " << (uintptr_t)addr;

    size_t size = it->second;
    used_list.erase(it);
    free_list.emplace(size, addr);
  }

  uint32_t LocalKey(char *addr) {
    struct ibv_mr *mr = Addr2MR(addr);
    return mr->lkey;
  }
  uint32_t RemoteKey(char *addr) {
    struct ibv_mr *mr = Addr2MR(addr);
    return mr->rkey;
  }

 private:
  std::mutex mu_;
  std::multimap<size_t, char *> free_list;
  std::unordered_map<char *, size_t> used_list;
  std::map<char *, struct ibv_mr*> mr_list; // first: `end` of this mr address (e.g., for mr with [addr, addr+size], point to `addr+size`)
  struct ibv_pd *pd_;
  size_t total_allocated_size = 0;

  inline struct ibv_mr* Addr2MR(char *addr) { // convert the memory address to its associated RDMA memory region
    std::lock_guard<std::mutex> lk(mu_);
    auto it = mr_list.lower_bound(addr);
    CHECK_NE(it, mr_list.end()) << "cannot find the associated memory region";
    return it->second;
  }

};

class Block {
 public:
  explicit Block(SimpleMempool *pool, char *addr, int count)
      : pool(pool), addr(addr), counter(count) {}

  ~Block() {
    CHECK_EQ(counter, 0);
    pool->Free(addr);
  }

  void Release() {
    int v = counter.fetch_sub(1);
    // LOG(INFO) << "Decrementing addr " << (uintptr_t)addr << ", counter: " <<
    // counter;
    if (v == 1) {
      delete this;
    }
  }

 private:
  SimpleMempool *pool;
  char *addr;
  std::atomic<int> counter;
};

enum MessageTypes : uint32_t {
  kRendezvousStart,
  kRendezvousReply,
};

struct RendezvousStart {
  uint64_t meta_len;
  uint64_t data_num;
  uint64_t data_len[kMaxDataFields];
  uint64_t origin_addr;
};

struct RendezvousReply {
  uint64_t addr;
  uint64_t origin_addr;
  uint32_t rkey;
  uint32_t idx;
};

enum WRContextType {
  kRendezvousStartContext,
  kRendezvousReplyContext,
  kWriteContext,
  kReceiveContext
};

struct WRContext {
  WRContextType type;
  struct ibv_mr *buffer;
  void *private_data;
};

struct BufferContext {
  char *buffer;
  size_t meta_len;
  size_t data_num;
  size_t data_len[kMaxDataFields];
};

typedef std::unique_ptr<struct ibv_mr, std::function<void(struct ibv_mr *)>>
    MRPtr;

struct MessageBuffer {
  size_t inline_len;
  char *inline_buf;
  WRContext *reserved_context;
  std::vector<SArray<char>> data;
  std::vector<std::pair<MRPtr, size_t>> mrs;
};

struct RequestContext {
  uint32_t node;
  uint16_t port;
  char hostname[kMaxHostnameLength];
};

static_assert(std::is_pod<RendezvousStart>::value,
              "RendezvousStart must be a POD type.");
static_assert(std::is_pod<RendezvousReply>::value,
              "RendezvousReply must be a POD type.");
static_assert(std::is_pod<RequestContext>::value,
              "RequestContext must be a POD type.");

static const size_t kMempoolChunkSize =
    std::max(sizeof(RendezvousStart), sizeof(RendezvousReply));

template <typename T>
class AddressPool {
 public:
  AddressPool() {
    std::lock_guard<std::mutex> lk(mu_);
    for (int i = 0; i < kMaxEntries; i++) {
      indices_.push(i);
      table_[i] = nullptr;
    }
  }

  T *GetAddressAndRelease(uint32_t index) {
    std::lock_guard<std::mutex> lk(mu_);
    T *ptr = table_[index];
    CHECK(ptr);
    indices_.push(index);
    table_[index] = nullptr;
    return ptr;
  }

  uint32_t StoreAddress(T *ptr) {
    std::lock_guard<std::mutex> lk(mu_);
    CHECK(ptr);
    uint32_t idx = indices_.front();
    indices_.pop();
    CHECK_EQ(table_[idx], nullptr);
    table_[idx] = ptr;
    return idx;
  }

 private:
  static const int kMaxEntries = 512;

  std::mutex mu_;
  std::queue<uint32_t> indices_;
  T *table_[kMaxEntries];
};

struct Endpoint {
  enum ConnectionStatus { IDLE, CONNECTING, CONNECTED, REJECTED };

  ConnectionStatus status;
  int node_id;
  std::condition_variable cv;
  std::mutex connect_mu;
  struct rdma_cm_id *cm_id;

  WRContext rx_ctx[kRxDepth];

  WRContext start_ctx[kStartDepth];
  WRContext reply_ctx[kReplyDepth];
  WRContext write_ctx[kWriteDepth];

  ThreadsafeQueue<WRContext *> free_start_ctx;
  ThreadsafeQueue<WRContext *> free_reply_ctx;
  ThreadsafeQueue<WRContext *> free_write_ctx;

  Endpoint() : status(IDLE), node_id(Node::kEmpty), cm_id(nullptr), rx_ctx() {}

  ~Endpoint() {
    for (int i = 0; i < kRxDepth; ++i) {
      if (!(rx_ctx[i].buffer)) {
        continue;
      }
      free(rx_ctx[i].buffer->addr);
      CHECK_EQ(ibv_dereg_mr(rx_ctx[i].buffer), 0);
    }

    for (int i = 0; i < kStartDepth; ++i) {
      if (start_ctx[i].buffer) {
        free(start_ctx[i].buffer->addr);
        CHECK_EQ(ibv_dereg_mr(start_ctx[i].buffer), 0);
      }
    }

    for (int i = 0; i < kReplyDepth; ++i) {
      if (reply_ctx[i].buffer) {
        free(reply_ctx[i].buffer->addr);
        CHECK_EQ(ibv_dereg_mr(reply_ctx[i].buffer), 0);
      }
    }

    for (int i = 0; i < kWriteDepth; ++i) {
      if (write_ctx[i].buffer) {
        free(write_ctx[i].buffer->addr);
        CHECK_EQ(ibv_dereg_mr(write_ctx[i].buffer), 0);
      }
    }

    rdma_destroy_qp(cm_id);
    CHECK_EQ(rdma_destroy_id(cm_id), 0) << strerror(errno);
  }

  void Disconnect() {
    std::unique_lock<std::mutex> lk(connect_mu);
    CHECK_EQ(rdma_disconnect(cm_id), 0) << strerror(errno);
    cv.wait(lk, [this] { return status == IDLE; });
  }

  void SetNodeID(int id) { node_id = id; }

  void InitSendContextHelper(struct ibv_pd *pd, WRContext *ctx,
                             ThreadsafeQueue<WRContext *> *queue, size_t num,
                             WRContextType type) {
    for (size_t i = 0; i < num; ++i) {
      void *buf = aligned_alloc(kAlignment, kMempoolChunkSize);
      CHECK(buf);
      struct ibv_mr *mr = ibv_reg_mr(pd, buf, kMempoolChunkSize, 0);
      CHECK(mr);

      ctx[i].type = type;
      ctx[i].buffer = mr;
      ctx[i].private_data = this;
      queue->Push(&ctx[i]);
    }
  }

  void Init(struct ibv_cq *cq, struct ibv_pd *pd) {
    struct ibv_qp_init_attr attr;
    memset(&attr, 0, sizeof(ibv_qp_init_attr));
    attr.send_cq = cq;
    attr.recv_cq = cq;
    attr.cap.max_send_wr = kStartDepth + kReplyDepth + kWriteDepth;
    attr.cap.max_recv_wr = kRxDepth;
    attr.cap.max_send_sge = kSGEntry;
    attr.cap.max_recv_sge = kSGEntry;
    attr.qp_type = IBV_QPT_RC;
    attr.sq_sig_all = 0;

    CHECK_EQ(rdma_create_qp(cm_id, pd, &attr), 0)
        << "Create RDMA queue pair failed";

    InitSendContextHelper(pd, start_ctx, &free_start_ctx, kStartDepth,
                          kRendezvousStartContext);
    InitSendContextHelper(pd, reply_ctx, &free_reply_ctx, kReplyDepth,
                          kRendezvousReplyContext);
    InitSendContextHelper(pd, write_ctx, &free_write_ctx, kWriteDepth,
                          kWriteContext);

    for (size_t i = 0; i < kRxDepth; ++i) {
      void *buf = aligned_alloc(kAlignment, kMempoolChunkSize);
      CHECK(buf);
      struct ibv_mr *mr =
          ibv_reg_mr(pd, buf, kMempoolChunkSize, IBV_ACCESS_LOCAL_WRITE);
      CHECK(mr);

      rx_ctx[i].type = kReceiveContext;
      rx_ctx[i].buffer = mr;
      rx_ctx[i].private_data = this;

      PostRecv(&rx_ctx[i]);
    }
  }

  void PostRecv(WRContext *ctx) {
    struct ibv_recv_wr wr, *bad_wr = nullptr;
    memset(&wr, 0, sizeof(wr));

    struct ibv_sge sge;
    sge.addr = reinterpret_cast<uint64_t>(ctx->buffer->addr);
    sge.length = kMempoolChunkSize;
    sge.lkey = ctx->buffer->lkey;

    wr.wr_id = reinterpret_cast<uint64_t>(ctx);
    wr.next = nullptr;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    CHECK_EQ(ibv_post_recv(cm_id->qp, &wr, &bad_wr), 0)
        << "ibv_post_recv failed.";
  }
};

class RDMAVan : public Van {
 public:
  RDMAVan() {}
  ~RDMAVan() {}

 protected:
  void Start(int customer_id) override {
    start_mu_.lock();
    should_stop_ = false;

    auto val = Environment::Get()->find("DMLC_ROLE");
    std::string role(val);
    is_server = role=="server";
    if (is_server) LOG(INFO) << "This is server";
    else LOG(INFO) << "This is " << ((role=="worker") ? "worker" : "scheduler");

    val = Environment::Get()->find("ENABLE_RDMA_LOG");
    enable_rdma_log_ = val? atoi(val) : false;
    if (enable_rdma_log_) LOG(INFO) << "Enable RDMA logging";
    else LOG(INFO) << "RDMA logging is disabled, you can enable it with ENABLE_RDMA_LOG=1";

    if (event_channel_ == nullptr) {
      event_channel_ = rdma_create_event_channel();
      CHECK(event_channel_) << "Create RDMA event channel failed";

      cm_event_polling_thread_.reset(
          new std::thread(&RDMAVan::PollEvents, this));
    }

    start_mu_.unlock();
    Van::Start(customer_id);
  }

  void Stop() override {
    PS_VLOG(1) << my_node_.ShortDebugString() << " is stopping";
    Van::Stop();

    should_stop_ = true;
    CHECK(should_stop_);

    PS_VLOG(1) << "Stopping cq_polling_thread_.";
    cq_polling_thread_->join();
    cq_polling_thread_.reset();

    PS_VLOG(1) << "Stopping cm_event_polling_thread_.";
    cm_event_polling_thread_->join();
    cm_event_polling_thread_.reset();

    PS_VLOG(1) << "Clearing mempool.";
    mempool_.reset();

    auto map_iter = memory_mr_map.begin();
    while (map_iter != memory_mr_map.end()) {
      ibv_dereg_mr(map_iter->second);
      map_iter++;
    }

    PS_VLOG(1) << "Clearing endpoints.";
    incoming_.clear();
    endpoints_.clear();

    PS_VLOG(1) << "Destroying cq and pd.";
    CHECK(!ibv_destroy_cq(cq_)) << "Failed to destroy CQ";
    CHECK(!ibv_destroy_comp_channel(comp_event_channel_))
        << "Failed to destroy channel";

    // TODO: ibv_dealloc_pd sometimes complains resource busy, need to fix this
    // CHECK(!ibv_dealloc_pd(pd_)) << "Failed to deallocate PD: " <<
    // strerror(errno);

    PS_VLOG(1) << "Destroying listener.";
    rdma_destroy_id(listener_);
    rdma_destroy_event_channel(event_channel_);
  }

  int Bind(const Node &node, int max_retry) override {
    CHECK(rdma_create_id(event_channel_, &listener_, nullptr, RDMA_PS_TCP) == 0)
        << "Create RDMA connection identifier failed";

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    int port = node.port;
    unsigned seed = static_cast<unsigned>(time(NULL) + port);
    for (int i = 0; i < max_retry + 1; ++i) {
      addr.sin_port = htons(port);
      if (rdma_bind_addr(listener_,
                         reinterpret_cast<struct sockaddr *>(&addr)) == 0) {
        break;
      }
      if (i == max_retry) {
        port = -1;
      } else {
        port = 10000 + rand_r(&seed) % 40000;
      }
    }
    CHECK(rdma_listen(listener_, kRdmaListenBacklog) == 0)
        << "Listen RDMA connection failed: " << strerror(errno);
    return port;
  }

  void Connect(const Node &node) override {
    PS_VLOG(1) << "Connecting to " << my_node_.ShortDebugString();
    CHECK_NE(node.id, node.kEmpty);
    CHECK_NE(node.port, node.kEmpty);
    CHECK(node.hostname.size());

    // worker doesn't need to connect to the other workers. same for server
    if ((node.role == my_node_.role) && (node.id != my_node_.id)) {
      return;
    }

    std::string node_host_ip = node.hostname + ":" + std::to_string(node.port);
    if (node.id != Node::kEmpty) {
      auto it = endpoints_.find(node.id);

      // if there is an endpoint with pending connection
      if (it != endpoints_.end()) {
        endpoints_.erase(it);
      }

      Endpoint *endpoint;
      endpoints_[node.id] = std::make_unique<Endpoint>();
      endpoint = endpoints_[node.id].get();

      endpoint->SetNodeID(node.id);

      struct addrinfo *remote_addr;
      CHECK_EQ(
          getaddrinfo(node.hostname.c_str(), std::to_string(node.port).c_str(),
                      nullptr, &remote_addr),
          0);

      while (endpoint->status != Endpoint::CONNECTED) {
        std::unique_lock<std::mutex> lk(endpoint->connect_mu);
        endpoint->status = Endpoint::CONNECTING;

        if (endpoint->cm_id != nullptr) {
          rdma_destroy_qp(endpoint->cm_id);
          CHECK_EQ(rdma_destroy_id(endpoint->cm_id), 0) << strerror(errno);
          endpoint->cm_id = nullptr;
        }

        CHECK_EQ(rdma_create_id(event_channel_, &endpoint->cm_id, nullptr,
                                RDMA_PS_TCP),
                 0)
            << "Create RDMA connection identifier failed";
        endpoint->cm_id->context = endpoint;

        CHECK_EQ(rdma_resolve_addr(endpoint->cm_id, nullptr,
                                   remote_addr->ai_addr, kTimeoutms),
                 0)
            << "Resolve RDMA address failed with errno: " << errno;

        endpoint->cv.wait(lk, [endpoint] {
          return endpoint->status != Endpoint::CONNECTING;
        });

        if (endpoint->status == Endpoint::CONNECTED) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }

      freeaddrinfo(remote_addr);
    }
  }

  bool IsValidPushpull(const Message &msg) {
    if (!msg.meta.control.empty()) return false;
    if (msg.meta.simple_app) return false;
    return true;
  }

  uint64_t DecodeKey(SArray<char> keys) { // just a translation, the decoded key might not be readable when we have multiple servers
    ps::Key key = 0;
    uint64_t coef = 1;
    for (unsigned int i = 0; i < keys.size(); ++i) {
      key += coef * (uint8_t) keys.data()[i];
      coef *= 256; // 256=2^8 (uint8_t)
    }
    return key;
  }

  int SendMsg(Message &msg) override {
    int remote_id = msg.meta.recver;
    CHECK_NE(remote_id, Meta::kEmpty);

    for (auto& sa : msg.data) {
      if (sa.size()) {
        std::lock_guard<std::mutex> lock(map_mu_);
        auto search_map_iterator = memory_mr_map.find(sa.data());
        if (search_map_iterator == memory_mr_map.end()) {
          struct ibv_mr *temp_mr;
          CHECK(sa.data()) << "address empty";
          CHECK (temp_mr = ibv_reg_mr(pd_, sa.data(), sa.size(),
                 IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE))
                  << "Failed to register the memory region: "
                  << strerror(errno)
                  << ", sa.size()=" << sa.size();
          memory_mr_map[sa.data()] = temp_mr;
        }
      }
    }

    if (IsValidPushpull(msg)) {
      if (!is_server) { // worker
        std::lock_guard<std::mutex> lock(map_mu_);
        uint64_t key = DecodeKey(msg.data[0]);
        msg.meta.key = key;
        //LOG(INFO) << "key=" << key << ", " << std::string(msg.meta.push?"push":"pull");

        if (msg.meta.push && msg.meta.request) { // push request
          CHECK_EQ(msg.data.size(), 3) << msg.data.size();
          CHECK_NE(memory_mr_map.find(msg.data[1].data()), memory_mr_map.end());

          auto& vals = msg.data[1];
          msg.meta.addr = reinterpret_cast<uint64_t>(vals.data()); // vals address
          msg.meta.val_len = vals.size();
          msg.meta.option = memory_mr_map[vals.data()]->rkey;

          if (enable_rdma_log_) {
            LOG(INFO) << "send push key=" << key
                    << ", val_len=" << msg.meta.val_len
                    << ", val_addr=" << msg.meta.addr
                    << ", rkey=" << msg.meta.option;
          }
        }
      }
      if (!msg.meta.push && !msg.meta.request) { // server, pull response
        CHECK(is_server);
        CHECK_EQ(msg.data.size(), 3) << msg.data.size();

        std::lock_guard<std::mutex> lock(map_mu_);
        uint64_t key = msg.meta.key;
        auto recver = msg.meta.recver;

        CHECK_NE(key_meta_map_.find(key), key_meta_map_.end())
            << "key=" << key << " not inited in key_meta_map";
        CHECK_NE(key_meta_map_[key].find(recver), key_meta_map_[key].end())
            << "key=" << key << ", recver=" << recver << " not inited in key_meta_map[key]";

        msg.meta.val_len = std::get<0>(key_meta_map_[key][recver]);
        msg.meta.addr = std::get<1>(key_meta_map_[key][recver]);
        msg.meta.option = std::get<2>(key_meta_map_[key][recver]);

        if (enable_rdma_log_) {
          LOG(INFO) << "send pull response key=" << key
                  << ", val_len=" << msg.meta.val_len
                  << ", val_addr=" << msg.meta.addr
                  << ", rkey=" << msg.meta.option;
        }
      }
    }

    PBMeta meta;
    PackMetaPB(msg.meta, &meta);

    CHECK_NE(endpoints_.find(remote_id), endpoints_.end());
    Endpoint *endpoint = endpoints_[remote_id].get();
    MessageBuffer *msg_buf = new MessageBuffer();

    size_t meta_len = meta.ByteSize();
    size_t data_len = msg.meta.data_size;
    size_t total_len = meta_len + data_len;

    CHECK(meta_len);

    if (msg.meta.simple_app || !msg.meta.control.empty()){ // simple_app or control message
      msg_buf->inline_len = total_len;
      msg_buf->inline_buf = mempool_->Alloc(total_len);
      meta.SerializeToArray(msg_buf->inline_buf, meta_len);
      char *cur = msg_buf->inline_buf + meta_len;
      for (auto &sa : msg.data) {
        size_t seg_len = sa.size();
        memcpy(cur, sa.data(), seg_len);
        cur += seg_len;
      }
    }
    else { // data message
      msg_buf->inline_len = meta_len;
      msg_buf->inline_buf = mempool_->Alloc(meta_len);
      msg_buf->data = msg.data;
      meta.SerializeToArray(msg_buf->inline_buf, meta_len);
      if (!is_server) { // worker remains the same
        for (auto &sa : msg_buf->data) {
          if (sa.size()) {
            auto search_map_iterator = memory_mr_map.find(sa.data());
            CHECK_NE(search_map_iterator, memory_mr_map.end()) << "not registered memory region";
            MRPtr ptr(search_map_iterator->second, [](struct ibv_mr *mr) {});
            CHECK(ptr.get()) << strerror(errno);
            msg_buf->mrs.push_back(std::make_pair(std::move(ptr), sa.size()));
          }
        }
      }
    }

    if (is_server && IsValidPushpull(msg) &&
          !msg.meta.push && !msg.meta.request) { // server send pull response (vals) with RDMA-write
      std::lock_guard<std::mutex> lock(map_mu_);
      auto key = msg.meta.key;
      auto recver = msg.meta.recver;

      CHECK_EQ(msg_buf->data.size(), 3) << "Actual msg_buf size is " << msg_buf->data.size();
      CHECK_NE(key_meta_map_.find(key), key_meta_map_.end())
            << "key=" << key << " not initiated";
      CHECK_NE(key_meta_map_[key].find(recver), key_meta_map_[key].end())
            << "key=" << key
            << ", recver=" << recver
            << " not initiated";

      auto len = std::get<0>(key_meta_map_[key][recver]);
      auto raddr = std::get<1>(key_meta_map_[key][recver]);
      auto rkey = std::get<2>(key_meta_map_[key][recver]);

      CHECK_EQ(msg_buf->data[1].size(), (unsigned int) len)
                << msg_buf->data[1].size() << ", " << len;

      auto temp_mr = memory_mr_map.find(msg_buf->data[1].data());
      CHECK_NE(temp_mr, memory_mr_map.end());

      struct ibv_sge sge;
      sge.addr = reinterpret_cast<uint64_t>(msg_buf->data[1].data());
      sge.length = msg_buf->data[1].size();
      sge.lkey = temp_mr->second->lkey;

      struct ibv_send_wr wr, *bad_wr = nullptr;
      memset(&wr, 0, sizeof(wr));

      wr.wr_id = reinterpret_cast<uint64_t>(raddr);
      wr.opcode = IBV_WR_RDMA_WRITE;
      wr.next = nullptr;
      //wr.send_flags = IBV_SEND_SIGNALED;
      wr.sg_list = &sge;
      wr.num_sge = 1;

      wr.wr.rdma.remote_addr = raddr;
      wr.wr.rdma.rkey = rkey;

      CHECK_EQ(ibv_post_send(endpoint->cm_id->qp, &wr, &bad_wr), 0)
        << "ibv_post_send failed.";
    }

    WRContext *context = nullptr, *reserved = nullptr;
    endpoint->free_write_ctx.WaitAndPop(&reserved);
    endpoint->free_start_ctx.WaitAndPop(&context);

    msg_buf->reserved_context = reserved;

    RendezvousStart *req =
        reinterpret_cast<RendezvousStart *>(context->buffer->addr);
    req->meta_len = meta_len;

    for (size_t i = 0; i < msg.data.size(); ++i) {
      req->data_len[i] = msg.data[i].size();
    }
    req->data_num = msg.data.size();
    req->origin_addr = reinterpret_cast<uint64_t>(msg_buf);

    struct ibv_sge sge;
    sge.addr = reinterpret_cast<uint64_t>(req);
    sge.length = sizeof(RendezvousStart);
    sge.lkey = context->buffer->lkey;

    struct ibv_send_wr wr, *bad_wr = nullptr;
    memset(&wr, 0, sizeof(wr));

    wr.wr_id = reinterpret_cast<uint64_t>(context);
    wr.opcode = IBV_WR_SEND_WITH_IMM;
    wr.next = nullptr;

    wr.imm_data = kRendezvousStart;

    wr.send_flags = IBV_SEND_SIGNALED;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    CHECK_EQ(ibv_post_send(endpoint->cm_id->qp, &wr, &bad_wr), 0)
        << strerror(errno);

    return total_len;
  }

  int RecvMsg(Message *msg) override {
    msg->data.clear();
    std::tuple<Endpoint *, BufferContext *> notification;
    recv_buffers_.WaitAndPop(&notification);

    Endpoint *endpoint = std::get<Endpoint *>(notification);
    BufferContext *buffer_ctx = std::get<BufferContext *>(notification);

    int total_len = 0;

    msg->meta.recver = my_node_.id;
    msg->meta.sender = endpoint->node_id;

    char *cur = buffer_ctx->buffer;

    UnpackMeta(cur, buffer_ctx->meta_len, &msg->meta);
    total_len += buffer_ctx->meta_len;
    uint64_t data_num = buffer_ctx->data_num;
    cur += buffer_ctx->meta_len;

    if (IsValidPushpull(*msg) && !msg->meta.push && !msg->meta.request) { // worker
      std::lock_guard<std::mutex> lock(map_mu_);
      auto key = msg->meta.key;
      CHECK(!is_server);
      if (key_len_map_.find(key) == key_len_map_.end()) {
        key_addr_map_[key] = (ps::Key) key;
        key_len_map_[key] = (int) msg->meta.val_len;
      }
      CHECK_NE(key_len_map_.find(key), key_len_map_.end()) << key;
      CHECK_NE(key_addr_map_.find(key), key_addr_map_.end()) << key;

      auto addr = msg->meta.addr;

      CHECK_NE(key_len_map_[key], 0) << msg->DebugString();

      SArray<char> keys;
      SArray<char> vals;
      SArray<char> lens;

      keys.reset(reinterpret_cast<char*>(&key_addr_map_[key]), sizeof(ps::Key), [](void *){});
      vals.reset(reinterpret_cast<char*>(addr), key_len_map_[key], [](void *){});
      lens.reset(reinterpret_cast<char*>(&key_len_map_[key]), sizeof(int), [](void *){});

      msg->data.push_back(keys);
      msg->data.push_back(vals);
      msg->data.push_back(lens);
      total_len += keys.size() + vals.size() + lens.size();
    } else if (data_num > 0) {
      Block *mem_block =
          new Block(mempool_.get(), buffer_ctx->buffer, data_num);

      for (size_t i = 0; i < data_num; i++) {
        uint32_t len = buffer_ctx->data_len[i];
        SArray<char> data;
        data.reset(cur, len, [mem_block](void *) {
          mem_block->Release();
        });  // Defer the deletion of block_ref
        msg->data.push_back(data);
        cur += len;
        total_len += len;
      }
    } else {
      mempool_->Free(buffer_ctx->buffer);
    }

    if (msg->meta.push && msg->meta.request) { // server
      CHECK(is_server);
      auto key = msg->meta.key;
      auto len = msg->meta.val_len;
      auto addr = msg->meta.addr;
      auto rkey = msg->meta.option;
      auto sender = msg->meta.sender;

      std::lock_guard<std::mutex> lock(map_mu_);
      if (key_meta_map_.find(key) == key_meta_map_.end()
            || key_meta_map_[key].find(sender) == key_meta_map_[key].end()) {
        if (enable_rdma_log_) {
          LOG(INFO) << "(init) recv key=" << key
                  << ", len=" << len
                  << ", sender=" << msg->meta.sender
                  << ", val_addr=" << addr
                  << ", rkey=" << rkey;
        }
        key_meta_map_[key][sender] = std::make_tuple(len, addr, rkey);
      } else {
        CHECK_EQ(len, std::get<0>(key_meta_map_[key][sender]));
        CHECK_EQ(addr, std::get<1>(key_meta_map_[key][sender]));
        CHECK_EQ(rkey, std::get<2>(key_meta_map_[key][sender]));

        if (enable_rdma_log_) {
          LOG(INFO) << "recv push key=" << key
                  << ", len=" << len
                  << ", val_addr=" << addr
                  << ", rkey=" << rkey;
        }
      }
    }

    delete buffer_ctx;
    return total_len;
  }

 private:
  void InitContext(struct ibv_context *context) {
    context_ = context;
    CHECK(context_) << "ibv_context* empty";

    pd_ = ibv_alloc_pd(context_);
    CHECK(pd_) << "Failed to allocate protection domain";

    mempool_.reset(new SimpleMempool(pd_));

    comp_event_channel_ = ibv_create_comp_channel(context_);

    // TODO(clan): Replace the rough estimate here
    cq_ = ibv_create_cq(context_, kMaxConcurrentWorkRequest * 2, NULL,
                        comp_event_channel_, 0);

    CHECK(cq_) << "Failed to create completion queue";
    CHECK(!ibv_req_notify_cq(cq_, 0)) << "Failed to request CQ notification";
  }

  void ReleaseWorkRequestContext(WRContext *context, Endpoint *endpoint) {
    switch (context->type) {
      case kRendezvousStartContext:
        endpoint->free_start_ctx.Push(context);
        break;
      case kRendezvousReplyContext:
        endpoint->free_reply_ctx.Push(context);
        break;
      case kWriteContext:
        endpoint->free_write_ctx.Push(context);
        break;
      case kReceiveContext:
        endpoint->PostRecv(context);
        break;
      default:
        CHECK(0);
    }
  }

  void PollCQ() {
    // Pre-allocated work completions array used for polling
    struct ibv_wc wc[kMaxConcurrentWorkRequest];
    while (!should_stop_.load()) {
      int ne = ibv_poll_cq(cq_, kMaxConcurrentWorkRequest, wc);
      CHECK_GE(ne, 0);
      for (int i = 0; i < ne; ++i) {
        CHECK(wc[i].status == IBV_WC_SUCCESS)
            << "Failed status \n"
            << ibv_wc_status_str(wc[i].status) << " " << wc[i].status << " "
            << static_cast<uint64_t>(wc[i].wr_id) << " " << wc[i].vendor_err;

        WRContext *context = reinterpret_cast<WRContext *>(wc[i].wr_id);
        Endpoint *endpoint =
            reinterpret_cast<Endpoint *>(context->private_data);

        CHECK(endpoint);

        switch (wc[i].opcode) {
          case IBV_WC_SEND:
            // LOG(INFO) << "opcode: IBV_WC_SEND";
            ReleaseWorkRequestContext(context, endpoint);
            break;
          case IBV_WC_RDMA_WRITE: {
            // LOG(INFO) << "opcode: IBV_WC_RDMA_WRITE";
            // Note: This is not a struct ibv_mr*
            MessageBuffer *msg_buf =
                *reinterpret_cast<MessageBuffer **>(context->buffer->addr);
            mempool_->Free(msg_buf->inline_buf);
            delete msg_buf;
            ReleaseWorkRequestContext(context, endpoint);
          } break;
          case IBV_WC_RECV_RDMA_WITH_IMM: {
            // LOG(INFO) << "opcode: IBV_WC_RECV_RDMA_WITH_IMM";
            uint32_t addr_idx = wc[i].imm_data;
            BufferContext *buf_ctx = addr_pool_.GetAddressAndRelease(addr_idx);
            recv_buffers_.Push(std::make_tuple(endpoint, buf_ctx));
            ReleaseWorkRequestContext(context, endpoint);
          } break;
          case IBV_WC_RECV: {
            CHECK(wc[i].wc_flags & IBV_WC_WITH_IMM);
            uint32_t imm = wc[i].imm_data;
            struct ibv_mr *mr = context->buffer;

            if (imm == kRendezvousStart) {
              // LOG(INFO) << "opcode: IBV_WC_RECV kRendezvousStart";
              RendezvousStart *req =
                  reinterpret_cast<RendezvousStart *>(mr->addr);
              BufferContext *buf_ctx = new BufferContext();

              uint64_t len = req->meta_len;
              buf_ctx->meta_len = req->meta_len;
              buf_ctx->data_num = req->data_num;
              for (size_t i = 0; i < req->data_num; ++i) {
                buf_ctx->data_len[i] = req->data_len[i];
                len += req->data_len[i];
              }

              char *buffer = mempool_->Alloc(is_server ? len : req->meta_len);
              CHECK(buffer) << "Alloc for " << len
                            << " bytes, data_num: " << req->data_num;

              buf_ctx->buffer = buffer;

              uint64_t origin_addr = req->origin_addr;

              WRContext *reply_ctx = nullptr;
              endpoint->free_reply_ctx.WaitAndPop(&reply_ctx);
              RendezvousReply *resp =
                  reinterpret_cast<RendezvousReply *>(reply_ctx->buffer->addr);

              resp->addr = reinterpret_cast<uint64_t>(buffer);
              resp->rkey = mempool_->RemoteKey(buffer);
              resp->origin_addr = origin_addr;
              resp->idx = addr_pool_.StoreAddress(buf_ctx);

              struct ibv_sge sge;
              sge.addr = reinterpret_cast<uint64_t>(resp);
              sge.length = sizeof(RendezvousReply);
              sge.lkey = reply_ctx->buffer->lkey;

              struct ibv_send_wr wr, *bad_wr = nullptr;
              memset(&wr, 0, sizeof(wr));

              wr.wr_id = reinterpret_cast<uint64_t>(reply_ctx);
              wr.opcode = IBV_WR_SEND_WITH_IMM;
              wr.next = nullptr;

              wr.imm_data = kRendezvousReply;

              wr.send_flags = IBV_SEND_SIGNALED;
              wr.sg_list = &sge;
              wr.num_sge = 1;

              CHECK_EQ(ibv_post_send(endpoint->cm_id->qp, &wr, &bad_wr), 0)
                  << "ibv_post_send failed.";

            } else if (imm == kRendezvousReply) {
              // LOG(INFO) << "opcode: IBV_WC_RECV kRendezvousReply";
              RendezvousReply *resp =
                  reinterpret_cast<RendezvousReply *>(mr->addr);
              uint64_t remote_addr = resp->addr;
              uint64_t origin_addr = resp->origin_addr;
              uint32_t rkey = resp->rkey;
              uint32_t idx = resp->idx;

              MessageBuffer *msg_buf =
                  reinterpret_cast<MessageBuffer *>(origin_addr);

              struct ibv_sge sge[1 + msg_buf->mrs.size()];

              sge[0].addr = reinterpret_cast<uint64_t>(msg_buf->inline_buf);
              sge[0].length = msg_buf->inline_len;
              sge[0].lkey = mempool_->LocalKey(msg_buf->inline_buf);

              size_t num_sge = 1;
              for (auto &pair : msg_buf->mrs) {
                size_t length = pair.second;
                CHECK(length);
                sge[num_sge].addr =
                    reinterpret_cast<uint64_t>(pair.first->addr);
                sge[num_sge].length = length;
                sge[num_sge].lkey = pair.first->lkey;
                ++num_sge;
              }
              if (is_server) CHECK_EQ(num_sge, 1) << num_sge;

              WRContext *write_ctx = msg_buf->reserved_context;

              MessageBuffer **tmp =
                  reinterpret_cast<MessageBuffer **>(write_ctx->buffer->addr);
              *tmp = msg_buf;  // write the addr of msg_buf into the mr buffer

              struct ibv_send_wr wr, *bad_wr = nullptr;
              memset(&wr, 0, sizeof(wr));

              wr.wr_id = reinterpret_cast<uint64_t>(write_ctx);
              wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
              wr.next = nullptr;

              wr.imm_data = idx;

              wr.send_flags = IBV_SEND_SIGNALED;
              wr.sg_list = sge;
              wr.num_sge = num_sge;

              wr.wr.rdma.remote_addr = remote_addr;
              wr.wr.rdma.rkey = rkey;

              CHECK_EQ(ibv_post_send(endpoint->cm_id->qp, &wr, &bad_wr), 0)
                  << "ibv_post_send failed.";

            } else {
              CHECK(0);
            }
            ReleaseWorkRequestContext(context, endpoint);
          } break;
          default:
            CHECK(0) << "Unexpected opcode: " << wc[i].opcode;
        }
      }
    }
  }

  void PollEvents() {
    int flags = fcntl(event_channel_->fd, F_GETFL);
    int rc = fcntl(event_channel_->fd, F_SETFL, flags | O_NONBLOCK);
    CHECK_GE(rc, 0);
    int error_flags = POLLERR | POLLHUP | POLLNVAL;

    while (!should_stop_.load()) {
      struct pollfd pfd = {
          .fd = event_channel_->fd, .events = POLLIN, .revents = 0};
      int ret = poll(&pfd, 1, 10);

      CHECK_GE(ret, 0) << strerror(errno);
      CHECK_EQ(pfd.revents & error_flags, 0);

      if (!(pfd.revents & POLLIN)) {
        continue;
      }

      struct rdma_cm_event *event;
      CHECK_EQ(rdma_get_cm_event(event_channel_, &event), 0);
      // TODO(clan): Reorder the list according to the event frequency
      switch (event->event) {
        case RDMA_CM_EVENT_CONNECT_REQUEST:
          OnConnectRequest(event);
          break;
        case RDMA_CM_EVENT_ADDR_RESOLVED:
          OnAddrResolved(event);
          break;
        case RDMA_CM_EVENT_ROUTE_RESOLVED:
          OnRouteResolved(event);
          break;
        case RDMA_CM_EVENT_ESTABLISHED:
          OnConnected(event);
          break;
        case RDMA_CM_EVENT_DISCONNECTED:
          OnDisconnected(event);
          break;
        case RDMA_CM_EVENT_REJECTED:
          OnRejected(event);
          break;
        default:
          CHECK(0) << "OnEvent: unknown event " << event->event << " ("
                   << rdma_event_str(event->event) << ")";
      }
      rdma_ack_cm_event(event);
    }
  }

  void OnRejected(struct rdma_cm_event *event) {
    struct rdma_cm_id *id = event->id;
    Endpoint *endpoint = reinterpret_cast<Endpoint *>(id->context);

    auto it = endpoints_.find(endpoint->node_id);
    CHECK(it != endpoints_.end()) << "Connection not ready.";
    CHECK_EQ(endpoint->status, Endpoint::CONNECTING);
    CHECK_EQ(endpoint->cm_id, id);

    PS_VLOG(1) << "Connection rejected, retrying...";
    {
      std::lock_guard<std::mutex> lk(endpoint->connect_mu);
      endpoint->status = Endpoint::REJECTED;
    }
    endpoint->cv.notify_all();
  }

  void OnConnectRequest(struct rdma_cm_event *event) {
    struct rdma_cm_id *id = event->id;
    CHECK_NOTNULL(id);

    CHECK_LE(sizeof(RequestContext), event->param.conn.private_data_len)
        << "RequestContext size mismatch. Actual: "
        << (size_t)event->param.conn.private_data_len
        << ", Expected: " << sizeof(RequestContext);
    CHECK_NOTNULL(event->param.conn.private_data);

    const RequestContext *remote_ctx = reinterpret_cast<const RequestContext *>(
        event->param.conn.private_data);

    const auto r = incoming_.emplace(std::make_unique<Endpoint>());
    Endpoint *endpoint = r.first->get();
    endpoint->SetNodeID(remote_ctx->node);
    endpoint->cm_id = id;
    id->context = endpoint;

    if (context_ == nullptr) {
      InitContext(id->verbs);
    }

    endpoint->Init(cq_, pd_);

    RequestContext ctx;
    ctx.node = static_cast<uint32_t>(my_node_.id);
    ctx.port = static_cast<uint16_t>(my_node_.port);
    snprintf(ctx.hostname, kMaxHostnameLength, "%s", my_node_.hostname.c_str());

    struct rdma_conn_param cm_params;
    memset(&cm_params, 0, sizeof(cm_params));
    cm_params.retry_count = 7;
    cm_params.rnr_retry_count = 7;
    cm_params.private_data = &ctx;
    cm_params.private_data_len = sizeof(RequestContext);

    CHECK_EQ(rdma_accept(id, &cm_params), 0)
        << "Accept RDMA connection failed: " << strerror(errno);
  }

  // Resolve a route after address is resolved
  void OnAddrResolved(struct rdma_cm_event *event) {
    struct rdma_cm_id *id = event->id;
    CHECK_EQ(rdma_resolve_route(id, kTimeoutms), 0)
        << "Resolve RDMA route failed";
  }

  // Make a connection after route is resolved
  void OnRouteResolved(struct rdma_cm_event *event) {
    struct rdma_cm_id *id = event->id;
    Endpoint *endpoint = reinterpret_cast<Endpoint *>(id->context);

    if (context_ == nullptr) {
      InitContext(id->verbs);
    }

    endpoint->Init(cq_, pd_);

    RequestContext ctx;
    ctx.node = static_cast<uint32_t>(my_node_.id);
    ctx.port = static_cast<uint16_t>(my_node_.port);
    snprintf(ctx.hostname, kMaxHostnameLength, "%s", my_node_.hostname.c_str());

    struct rdma_conn_param cm_params;
    memset(&cm_params, 0, sizeof(cm_params));
    cm_params.retry_count = 7;
    cm_params.rnr_retry_count = 7;
    cm_params.private_data = &ctx;
    cm_params.private_data_len = sizeof(RequestContext);

    CHECK_EQ(rdma_connect(id, &cm_params), 0)
        << "RDMA connect failed" << strerror(errno);
  }

  void OnConnected(struct rdma_cm_event *event) {
    struct rdma_cm_id *id = event->id;
    CHECK(id) << "rdma_cm_id not found.";
    Endpoint *endpoint = reinterpret_cast<Endpoint *>(id->context);
    CHECK(endpoint) << "Endpoint not found.";

    if (cq_polling_thread_ == nullptr) {
      cq_polling_thread_.reset(new std::thread(&RDMAVan::PollCQ, this));
    }

    CHECK_EQ(endpoint->cm_id, id);
    {
      std::lock_guard<std::mutex> lk(endpoint->connect_mu);
      endpoint->status = Endpoint::CONNECTED;
    }
    endpoint->cv.notify_all();
  }

  void OnDisconnected(struct rdma_cm_event *event) {
    LOG(INFO) << "OnDisconnected from Node " << my_node_.id;
    struct rdma_cm_id *id = event->id;
    Endpoint *endpoint = reinterpret_cast<Endpoint *>(id->context);
    {
      std::lock_guard<std::mutex> lk(endpoint->connect_mu);
      endpoint->status = Endpoint::IDLE;
    }
    endpoint->cv.notify_all();
  }

  AddressPool<BufferContext> addr_pool_;
  std::unique_ptr<SimpleMempool> mempool_;

  struct rdma_cm_id *listener_ = nullptr;
  std::atomic<bool> should_stop_;

  std::unordered_map<int, std::unique_ptr<Endpoint>> endpoints_;
  std::unordered_set<std::unique_ptr<Endpoint>> incoming_;

  struct rdma_event_channel *event_channel_ = nullptr;
  struct ibv_context *context_ = nullptr;

  std::unordered_map<char *, struct ibv_mr *> memory_mr_map;

  // ibverbs protection domain
  struct ibv_pd *pd_ = nullptr;
  // Completion event channel, to wait for work completions
  struct ibv_comp_channel *comp_event_channel_ = nullptr;
  // Completion queue, to poll on work completions
  struct ibv_cq *cq_ = nullptr;
  // cq thread
  std::unique_ptr<std::thread> cq_polling_thread_;
  // event thread
  std::unique_ptr<std::thread> cm_event_polling_thread_;
  // Recv buffer queue
  ThreadsafeQueue<std::tuple<Endpoint *, BufferContext *>> recv_buffers_;

  // JYM: the following are for push/pull buffer reuse

  // whether my role is server or not
  bool is_server;
  // RDMA logging info
  bool enable_rdma_log_;

  // macros for key_meta_map
  using MetaInfo = std::tuple<int, uint64_t, int>; // len, addr, rkey
  using SenderMeta = std::unordered_map<int, MetaInfo>; // sender as the key
  // (key, sender) --> MetaInfo
  std::unordered_map<ps::Key, SenderMeta> key_meta_map_;
  // a static address for the key
  std::unordered_map<ps::Key, ps::Key> key_addr_map_;
  // a static address for the length
  std::unordered_map<ps::Key, int> key_len_map_;

  std::mutex map_mu_;

};  // namespace ps
};  // namespace ps

#endif  // DMLC_USE_RDMA
#endif  // PS_RDMA_VAN_H_
