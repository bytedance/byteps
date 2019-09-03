#define EIGEN_USE_THREADS

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/edgeset.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

#include <map>
#include <queue>
#include <set>
#include <unordered_map>
#include <vector>

using namespace tensorflow;
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

using VariableName = string;
using Priority = size_t;
using GradientName = string;

// TODO(byronyi): auto-try other credit sizes for better performance
const size_t WINDOW_SIZE = 8 * 1024 * 1024;

static std::unordered_map<string, uint64> device_incarnations;

static std::unordered_map<GradientName, Priority> priorities;

static ShapeHandle ShapeOrHandleShape(InferenceContext *c, int input) {
  auto *handle_data = c->input_handle_shapes_and_types(input);
  if (handle_data != nullptr && !handle_data->empty() &&
      (*handle_data)[0].dtype != DT_INVALID) {
    return (*handle_data)[0].shape;
  }
  return c->input(input);
}

static Status ApplyGradientDescentShapeFn(InferenceContext *c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                 // var
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused)); // alpha
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("SendGradient")
    .Input("tensor: T")
    .Attr("T: type")
    .Attr("gradient_name: string")
    .Attr("tensor_name: string")
    .Attr("send_device: string")
    .Attr("send_device_incarnation: int")
    .Attr("recv_device: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Sends the named tensor from send_device to recv_device.
)doc");

REGISTER_OP("RecvApplyGradientDescent")
    .Input("var: Ref(T)")
    .Input("alpha: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("variable_name: string")
    .Attr("tensor_name: string")
    .Attr("send_device: string")
    .Attr("send_device_incarnation: int")
    .Attr("recv_device: string")
    .SetIsStateful()
    .SetShapeFn(ApplyGradientDescentShapeFn)
    .Doc(R"doc(
Receives the named tensor from send_device on recv_device and apply
GradientDescent algorithm to variable with the received tensor as delta.
)doc");

REGISTER_OP("SendParameter")
    .Input("tensor: T")
    .Attr("T: type")
    .Attr("variable_name: string")
    .Attr("tensor_name: string")
    .Attr("send_device: string")
    .Attr("send_device_incarnation: int")
    .Attr("recv_device: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Sends the named tensor from send_device to recv_device.
)doc");

REGISTER_OP("RecvParameter")
    .Output("tensor: tensor_type")
    .Attr("tensor_type: type")
    .Attr("tensor_name: string")
    .Attr("send_device: string")
    .Attr("send_device_incarnation: int")
    .Attr("recv_device: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Receives the named tensor from send_device to recv_device.
)doc");

Status ReverseKey(const Rendezvous::ParsedKey &key,
                  Rendezvous::ParsedKey *reversed) {
  int64 device_incarnation = device_incarnations[string(key.dst_device)];
  string reversed_key_str = Rendezvous::CreateKey(
      string(key.dst_device), device_incarnation, string(key.src_device),
      string(key.edge_name), FrameAndIter(0, 0));

  return Rendezvous::ParseKey(reversed_key_str, reversed);
}

using DoneCallback = std::function<void(const Status &)>;

struct BaseOobUpdate {
  virtual ~BaseOobUpdate() {}
  virtual void Execute(DoneCallback done) = 0;
  virtual string Name() const = 0;
};

template <typename T> struct OobUpdate : public BaseOobUpdate {
  explicit OobUpdate(string variable_name, Rendezvous *rendezvous,
                     Rendezvous::ParsedKey parsed_key, Rendezvous::Args args,
                     const Eigen::ThreadPoolDevice *device, Tensor var,
                     const Tensor &alpha)
      : variable_name_(variable_name), rendezvous_(rendezvous),
        parsed_key_(parsed_key), args_(args), device_(device),
        var_(std::move(var)), alpha_(alpha) {}

  ~OobUpdate() override {}

  void Execute(DoneCallback done) override {
    rendezvous_->RecvAsync(
        parsed_key_, args_,
        [this, done](const Status &s, const Rendezvous::Args &send_args,
                     const Rendezvous::Args &recv_args, const Tensor &delta,
                     bool is_dead) {
          if (!s.ok()) {
            return;
          }

          Rendezvous::ParsedKey ack_key;
          Status status = ReverseKey(parsed_key_, &ack_key);
          if (!status.ok()) {
            LOG(WARNING) << status;
          }
          rendezvous_->Send(ack_key, send_args, Tensor{}, false);

          if (s.ok() && !is_dead) {
            VLOG(2) << "Start gradient update to " << variable_name_;
            typename TTypes<T>::Flat var = var_.flat<T>();
            typename TTypes<T>::ConstFlat grad = delta.flat<T>();
            typename TTypes<T>::ConstScalar lr = alpha_.scalar<T>();
            var.device(*device_) -= grad * lr();
            VLOG(2) << "Finish gradient update to " << variable_name_;
          }
          done(s);
        });
  }

  string Name() const override { return variable_name_; }

  string variable_name_;

  Rendezvous *rendezvous_;
  Rendezvous::ParsedKey parsed_key_;
  Rendezvous::Args args_;

  const Eigen::ThreadPoolDevice *device_;
  Tensor var_;
  const Tensor alpha_;
};

struct GradientPush {

  explicit GradientPush(string gradient_name, Rendezvous *rendezvous,
                        Rendezvous::ParsedKey parsed_key, Rendezvous::Args args,
                        const Tensor &gradient, bool is_dead)
      : gradient_name_(gradient_name), rendezvous_(rendezvous),
        parsed_key_(parsed_key), args_(args), gradient_(gradient),
        is_dead_(is_dead) {}

  void Execute(DoneCallback done) {
    rendezvous_->Send(parsed_key_, args_, gradient_, is_dead_);

    Rendezvous::ParsedKey ack_key;
    Status status = ReverseKey(parsed_key_, &ack_key);
    if (!status.ok()) {
      LOG(WARNING) << status;
    }

    int64 start = Env::Default()->NowMicros();
    rendezvous_->RecvAsync(
        ack_key, args_,
        [this, done, start](const Status &s, const Rendezvous::Args &send_args,
                            const Rendezvous::Args &recv_args, const Tensor &t,
                            bool is_dead) {
          if (!s.ok()) {
            LOG(WARNING) << s;
          } else {
            int64 duration = Env::Default()->NowMicros() - start;
            VLOG(2) << "Ack RTT for " << gradient_name_ << " takes " << duration
                    << " us";
          }
          done(s);
        });
  }

  size_t NumBytes() const { return gradient_.TotalBytes(); }

  size_t Priority() const { return priorities[gradient_name_]; }

  string gradient_name_;

  Rendezvous *rendezvous_;
  Rendezvous::ParsedKey parsed_key_;
  Rendezvous::Args args_;
  const Tensor gradient_;
  bool is_dead_;
};

class OobUpdateManager {
public:
  explicit OobUpdateManager() : bytes_in_flight_(0) {}

  void Schedule(string gradient_name, Rendezvous *rendezvous,
                Rendezvous::ParsedKey parsed_key, Rendezvous::Args args,
                const Tensor &gradient, bool is_dead) {
    GradientPush *push = new GradientPush(gradient_name, rendezvous, parsed_key,
                                          args, gradient, is_dead);
    Schedule(push);
  }

  void Schedule(GradientPush *push) {
    if (bytes_in_flight_ < WINDOW_SIZE) {
      VLOG(2) << "Scheduling gradient " << push->gradient_name_;
      bytes_in_flight_ += push->NumBytes();
      push->Execute([this, push](const Status &s) {
        VLOG(2) << "Finished pushing gradient " << push->gradient_name_;
        bytes_in_flight_ -= push->NumBytes();
        GradientPush *next = nullptr;
        {
          mutex_lock l(mu_);
          if (queue_.size() > 0) {
            next = queue_.top();
            queue_.pop();
          }
        }
        if (next) {
          Schedule(next);
        }
        delete push;
      });
    } else {
      mutex_lock l(mu_);
      queue_.push(push);
    }
  }

  template <typename T>
  void RecvUpdate(string variable_name, Rendezvous *rendezvous,
                  Rendezvous::ParsedKey parsed_key, Rendezvous::Args args,
                  const Eigen::ThreadPoolDevice *device, Tensor var,
                  const Tensor &alpha) {
    string src_device = string(parsed_key.src_device);
    VLOG(2) << "Fetching updates to " << variable_name;
    BaseOobUpdate *update = new OobUpdate<T>(
        variable_name, rendezvous, parsed_key, args, device, var, alpha);
    update->Execute([this, update, src_device](const Status &s) {
      Ready(src_device, update->Name(), s);
      delete update;
    });
  }

  void Ready(string device, string variable_name, Status s) {
    DoneCallback done;
    string key = strings::StrCat(device, variable_name);
    {
      mutex_lock l(mu_);
      auto iter = callbacks_.find(key);
      if (iter != std::end(callbacks_)) {
        done = std::move(iter->second);
        callbacks_.erase(iter);
      } else {
        decltype(completed_status_)::iterator _;
        bool success;
        std::tie(_, success) = completed_status_.insert(std::make_pair(key, s));
      }
    }
    if (done) {
      done(s);
    }
  }

  void Poll(string device, string variable_name, DoneCallback done) {
    Status s = Status::OK();
    bool valid = false;
    string key = strings::StrCat(device, variable_name);
    {
      mutex_lock l(mu_);
      if (seen_keys_.find(key) == std::end(seen_keys_)) {
        seen_keys_.insert(key);
        valid = true;
      }
    }
    if (!valid) {
      mutex_lock l(mu_);
      auto iter = completed_status_.find(key);
      if (iter != std::end(completed_status_)) {
        s = iter->second;
        valid = true;
        completed_status_.erase(iter);
      } else {
        decltype(callbacks_)::iterator _;
        bool success;
        std::tie(_, success) =
            callbacks_.insert(std::make_pair(key, std::move(done)));
      }
    }
    if (valid) {
      done(s);
    }
  }

  static OobUpdateManager *Get() {
    static OobUpdateManager *manager = new OobUpdateManager;
    return manager;
  }

private:
  struct Comparator {
    bool operator()(GradientPush *a, GradientPush *b) const {
      return a->Priority() > b->Priority();
    }
  };

  std::atomic<size_t> bytes_in_flight_;

  mutex mu_;
  // Worker side
  std::priority_queue<GradientPush *, std::vector<GradientPush *>, Comparator>
      queue_ GUARDED_BY(mu_);
  // PS side
  std::unordered_map<string, DoneCallback> callbacks_ GUARDED_BY(mu_);
  std::unordered_map<string, Status> completed_status_ GUARDED_BY(mu_);
  std::set<string> seen_keys_ GUARDED_BY(mu_);
};

class SendGradientOp : public OpKernel {
public:
  explicit SendGradientOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("gradient_name", &gradient_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_name", &tensor_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("send_device", &send_device_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("recv_device", &recv_device_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("send_device_incarnation",
                                     reinterpret_cast<int64 *>(
                                         &send_device_incarnation_)));
  }

  void Compute(OpKernelContext *ctx) override {
    OP_REQUIRES(
        ctx, ctx->rendezvous() != nullptr,
        errors::Internal("Op kernel context needs to provide a rendezvous."));

    Rendezvous::Args args;
    args.device_context = ctx->op_device_context();
    args.alloc_attrs = ctx->input_alloc_attr(0);

    string key =
        Rendezvous::CreateKey(send_device_, send_device_incarnation_,
                              recv_device_, tensor_name_, ctx->frame_iter());
    Rendezvous::ParsedKey parsed_key;
    Rendezvous::ParseKey(key, &parsed_key);

    OobUpdateManager::Get()->Schedule(gradient_name_, ctx->rendezvous(),
                                      parsed_key, args, ctx->input(0),
                                      ctx->is_input_dead());
  }

private:
  string gradient_name_;
  string tensor_name_;
  string send_device_;
  string recv_device_;
  uint64 send_device_incarnation_;

  TF_DISALLOW_COPY_AND_ASSIGN(SendGradientOp);
};

template <typename T> class RecvApplyGradientDescentOp : public AsyncOpKernel {
public:
  explicit RecvApplyGradientDescentOp(OpKernelConstruction *ctx)
      : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("variable_name", &variable_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_name", &tensor_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("send_device", &send_device_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("recv_device", &recv_device_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("send_device_incarnation",
                                     reinterpret_cast<int64 *>(
                                         &send_device_incarnation_)));
  }

  void ComputeAsync(OpKernelContext *ctx, DoneCallback done) override {

    OP_REQUIRES_ASYNC(
        ctx, ctx->rendezvous() != nullptr,
        errors::Internal("Op kernel context needs to provide a rendezvous."),
        done);

    Rendezvous::Args args;
    AllocatorAttributes alloc_attrs;
    alloc_attrs.set_nic_compatible(true);
    alloc_attrs.set_on_host(true);
    args.alloc_attrs = alloc_attrs;
    args.device_context = ctx->op_device_context();

    string key =
        Rendezvous::CreateKey(send_device_, send_device_incarnation_,
                              recv_device_, tensor_name_, ctx->frame_iter());
    Rendezvous::ParsedKey parsed_key;
    Rendezvous::ParseKey(key, &parsed_key);

    OobUpdateManager::Get()->RecvUpdate<T>(
        variable_name_, ctx->rendezvous(), parsed_key, args,
        &ctx->eigen_cpu_device(), ctx->mutable_input(0, false), ctx->input(1));

    ctx->forward_ref_input_to_ref_output(0, 0);
    ctx->SetStatus(Status::OK());
    done();
  }

private:
  string variable_name_;
  string tensor_name_;
  string send_device_;
  string recv_device_;
  uint64 send_device_incarnation_;

  TF_DISALLOW_COPY_AND_ASSIGN(RecvApplyGradientDescentOp);
};

class SendParameterOp : public AsyncOpKernel {
public:
  explicit SendParameterOp(OpKernelConstruction *ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("variable_name", &variable_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_name", &tensor_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("send_device", &send_device_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("recv_device", &recv_device_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("send_device_incarnation",
                                     reinterpret_cast<int64 *>(
                                         &send_device_incarnation_)));
    device_incarnations.insert(
        std::make_pair(send_device_, send_device_incarnation_));
  }

  void ComputeAsync(OpKernelContext *ctx, DoneCallback done) override {
    OP_REQUIRES(
        ctx, ctx->rendezvous() != nullptr,
        errors::Internal("Op kernel context needs to provide a rendezvous."));

    Rendezvous::Args args;
    args.device_context = ctx->op_device_context();
    args.alloc_attrs = ctx->input_alloc_attr(0);

    string key =
        Rendezvous::CreateKey(send_device_, send_device_incarnation_,
                              recv_device_, tensor_name_, ctx->frame_iter());
    Rendezvous::ParsedKey parsed_key;
    Rendezvous::ParseKey(key, &parsed_key);

    OobUpdateManager::Get()->Poll(
        recv_device_, variable_name_,
        [this, ctx, parsed_key, args, done](const Status &s) {
          if (!s.ok()) {
            LOG(WARNING) << s;
            ctx->SetStatus(s);
            done();
            return;
          }
          ctx->rendezvous()->Send(parsed_key, args, ctx->input(0),
                                  ctx->is_input_dead());
          done();
        });
  }

private:
  string variable_name_;
  string tensor_name_;
  string send_device_;
  string recv_device_;
  uint64 send_device_incarnation_;

  TF_DISALLOW_COPY_AND_ASSIGN(SendParameterOp);
};

class RecvParameterOp : public AsyncOpKernel {
public:
  explicit RecvParameterOp(OpKernelConstruction *ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_name", &tensor_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("send_device", &send_device_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("recv_device", &recv_device_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("send_device_incarnation",
                                     reinterpret_cast<int64 *>(
                                         &send_device_incarnation_)));
    device_incarnations.insert(
        std::make_pair(send_device_, send_device_incarnation_));
  }

  void ComputeAsync(OpKernelContext *ctx, DoneCallback done) override {
    OP_REQUIRES(
        ctx, ctx->rendezvous() != nullptr,
        errors::Internal("Op kernel context needs to provide a rendezvous."));

    Rendezvous::Args args;
    args.device_context = ctx->op_device_context();
    args.alloc_attrs = ctx->output_alloc_attr(0);

    string key =
        Rendezvous::CreateKey(send_device_, send_device_incarnation_,
                              recv_device_, tensor_name_, ctx->frame_iter());
    Rendezvous::ParsedKey parsed_key;
    Rendezvous::ParseKey(key, &parsed_key);

    ctx->rendezvous()->RecvAsync(parsed_key, args,
                                 [ctx, done](const Status &s,
                                             const Rendezvous::Args &send_args,
                                             const Rendezvous::Args &recv_args,
                                             const Tensor &t, bool is_dead) {
                                   ctx->SetStatus(s);
                                   if (s.ok() && !is_dead) {
                                     ctx->set_output(0, t);
                                   }
                                   done();
                                 });
  }

private:
  string tensor_name_;
  string send_device_;
  string recv_device_;
  uint64 send_device_incarnation_;

  TF_DISALLOW_COPY_AND_ASSIGN(RecvParameterOp);
};

REGISTER_KERNEL_BUILDER(Name("SendGradient").Device(DEVICE_CPU),
                        SendGradientOp);
REGISTER_KERNEL_BUILDER(Name("SendGradient").Device(DEVICE_GPU),
                        SendGradientOp);
REGISTER_KERNEL_BUILDER(Name("RecvApplyGradientDescent")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        RecvApplyGradientDescentOp<float>);
REGISTER_KERNEL_BUILDER(Name("SendParameter").Device(DEVICE_CPU),
                        SendParameterOp);
REGISTER_KERNEL_BUILDER(Name("RecvParameter").Device(DEVICE_CPU),
                        RecvParameterOp);
REGISTER_KERNEL_BUILDER(Name("RecvParameter").Device(DEVICE_GPU),
                        RecvParameterOp);

class GraphRewritePass : public GraphOptimizationPass {
public:
  struct VariableInfo {
    Node *grad_op;
    Node *apply_op;
    VariableName variable_name;
  };

  Status Run(const GraphOptimizationPassOptions &options) override {

    VLOG(1) << "Successfully loaded GraphRewritePass";
    Graph *graph = options.graph->get();
    if (graph == nullptr) {
      return errors::Internal("Graph is not available");
    }

    std::unordered_map<VariableName, VariableInfo> variables;

    for (Node *node : graph->op_nodes()) {
      if (node->type_string() == "ApplyGradientDescent") {
        Node *var, *grad;
        Status s = node->input_node(0, &var);
        if (!s.ok() || !IsVariable(var)) {
          return errors::Internal("Cannot find variable for apply");
        }
        s = node->input_node(2, &grad);
        if (!s.ok()) {
          return errors::Internal("Cannot find gradient for apply");
        }

        VariableInfo info = {};
        info.variable_name = var->name();
        info.apply_op = node;
        info.grad_op = grad;
        variables.insert(std::make_pair(var->name(), info));
      }
    }

    std::vector<Node *> order;
    GetReversePostOrder(*graph, &order);

    for (Node *node : order) {
      if (node->IsVariable()) {
        auto iter = variables.find(node->name());
        if (iter != std::end(variables)) {
          LOG(INFO) << "Instrumenting variable " << node->name()
                    << " with gradient " << iter->second.grad_op->name();
          GradientName grad = iter->second.grad_op->name();
          Priority prio = priorities.size();
          priorities.insert(std::make_pair(grad, prio));
        }
      }
    }
    return Status::OK();
  }
};

struct WorkerRewriteTask {
  Node *send_op;
  Node *grad_op;
  int grad_index;
  std::vector<NodeBuilder::NodeOut> send_out_nodes;
};

class WorkerRewritePass : public GraphOptimizationPass {
public:
  Status Run(const GraphOptimizationPassOptions &options) override {

    VLOG(1) << "Successfully loaded WorkerRewritePass";
    std::unordered_map<string, std::unique_ptr<Graph>> *partition_graphs =
        options.partition_graphs;
    if (partition_graphs == nullptr) {
      return errors::Internal("Partitioned graphs is not available");
    }

    for (auto &kv : *partition_graphs) {
      if (str_util::StrContains(kv.first, "worker")) {
        Graph *graph = kv.second.get();

        std::vector<WorkerRewriteTask> tasks;

        for (Node *node : graph->op_nodes()) {
          if (node->IsSend()) {
            Node *send = node;
            Node *grad;
            TF_RETURN_IF_ERROR(send->input_node(0, &grad));
            auto iter = priorities.find(grad->name());
            if (iter != std::end(priorities)) {
              WorkerRewriteTask task;
              task.send_op = send;
              task.grad_op = grad;
              tasks.push_back(task);
            }
          }
        }

        for (Edge *edge : graph->edges()) {
          for (auto &task : tasks) {
            if (edge->src() == task.send_op) {
              task.send_out_nodes.emplace_back(edge->dst(), edge->dst_input());
            } else if (edge->src() == task.grad_op &&
                       edge->dst() == task.send_op) {
              task.grad_index = edge->src_output();
            }
          }
        }

        for (WorkerRewriteTask &task : tasks) {
          DataType dtype;
          TF_RETURN_IF_ERROR(GetNodeAttr(task.send_op->attrs(), "T", &dtype));
          string tensor_name;
          TF_RETURN_IF_ERROR(
              GetNodeAttr(task.send_op->attrs(), "tensor_name", &tensor_name));
          string send_device;
          TF_RETURN_IF_ERROR(
              GetNodeAttr(task.send_op->attrs(), "send_device", &send_device));
          string recv_device;
          TF_RETURN_IF_ERROR(
              GetNodeAttr(task.send_op->attrs(), "recv_device", &recv_device));
          int64 send_device_incarnation;
          TF_RETURN_IF_ERROR(GetNodeAttr(task.send_op->attrs(),
                                         "send_device_incarnation",
                                         &send_device_incarnation));

          NodeBuilder builder(task.send_op->name(), "SendGradient");
          builder.Input(task.grad_op, task.grad_index);
          builder.Attr("T", dtype);
          builder.Attr("gradient_name", task.grad_op->name());
          builder.Attr("tensor_name", tensor_name);
          builder.Attr("send_device", send_device);
          builder.Attr("recv_device", recv_device);
          builder.Attr("send_device_incarnation", send_device_incarnation);

          Node *node;
          TF_RETURN_IF_ERROR(builder.Finalize(graph, &node));

          graph->RemoveNode(task.send_op);

          for (const auto &out_node : task.send_out_nodes) {
            if (out_node.index == Graph::kControlSlot) {
              graph->AddControlEdge(node, out_node.node);
            } else {
              graph->AddEdge(node, 0, out_node.node, out_node.index);
            }
          }
          TF_RETURN_IF_ERROR(graph->IsValidNode(node));

          LOG(INFO) << "ByteScheduler taking over gradient "
                    << task.grad_op->name();
        }

        std::unordered_map<Node *, std::vector<NodeBuilder::NodeOut>> recv_ops;

        for (Node *node : graph->nodes()) {
          if (node->IsRecv()) {
            recv_ops.insert(
                std::make_pair(node, std::vector<NodeBuilder::NodeOut>()));
          }
        }

        for (Edge *edge : graph->edges()) {
          if (edge->src()->IsRecv()) {
            Node *recv = edge->src();
            auto iter = recv_ops.find(recv);
            if (iter != std::end(recv_ops)) {
              iter->second.emplace_back(edge->dst(), edge->dst_input());
            }
          }
        }

        for (auto &p : recv_ops) {
          Node *recv_op = p.first;
          std::vector<NodeBuilder::NodeOut> &out_nodes = p.second;

          DataType dtype;
          TF_RETURN_IF_ERROR(
              GetNodeAttr(recv_op->attrs(), "tensor_type", &dtype));
          string tensor_name;
          TF_RETURN_IF_ERROR(
              GetNodeAttr(recv_op->attrs(), "tensor_name", &tensor_name));
          string send_device;
          TF_RETURN_IF_ERROR(
              GetNodeAttr(recv_op->attrs(), "send_device", &send_device));
          string recv_device;
          TF_RETURN_IF_ERROR(
              GetNodeAttr(recv_op->attrs(), "recv_device", &recv_device));
          int64 send_device_incarnation;
          TF_RETURN_IF_ERROR(GetNodeAttr(recv_op->attrs(),
                                         "send_device_incarnation",
                                         &send_device_incarnation));

          NodeBuilder builder(recv_op->name(), "RecvParameter");
          builder.Attr("tensor_type", dtype);
          builder.Attr("tensor_name", tensor_name);
          builder.Attr("send_device", send_device);
          builder.Attr("recv_device", recv_device);
          builder.Attr("send_device_incarnation", send_device_incarnation);

          Node *node;
          TF_RETURN_IF_ERROR(builder.Finalize(graph, &node));

          graph->RemoveNode(recv_op);

          for (const auto &out_node : out_nodes) {
            if (out_node.index == Graph::kControlSlot) {
              graph->AddControlEdge(node, out_node.node);
            } else {
              graph->AddEdge(node, 0, out_node.node, out_node.index);
            }
          }
          TF_RETURN_IF_ERROR(graph->IsValidNode(node));
        }
      }
    }

    return Status::OK();
  }
};

struct PSRewriteTask {
  VariableName variable_name;
  Node *update_op;
  Node *recv_op;
  Node *var_op;
  Node *send_op;
  std::vector<NodeBuilder::NodeOut> update_out_nodes;
  std::vector<NodeBuilder::NodeOut> send_out_nodes;
};

class PSRewritePass : public GraphOptimizationPass {
public:
  Status Run(const GraphOptimizationPassOptions &options) override {

    VLOG(1) << "Successfully loaded PSRewritePass";
    std::unordered_map<string, std::unique_ptr<Graph>> *partition_graphs =
        options.partition_graphs;
    if (partition_graphs == nullptr) {
      return errors::Internal("Partitioned graphs is not available");
    }

    std::unordered_map<string, PSRewriteTask> task_map;

    for (auto &kv : *partition_graphs) {
      if (str_util::StrContains(kv.first, "ps")) {
        Graph *graph = kv.second.get();
        for (Node *node : graph->op_nodes()) {
          if (node->type_string() == "ApplyGradientDescent") {
            Node *var, *grad;
            Status s = node->input_node(0, &var);
            if (!s.ok() || !IsVariable(var)) {
              return errors::Internal("Cannot find variable for apply");
            }
            s = node->input_node(2, &grad);
            if (!s.ok() || !IsRecv(grad)) {
              return errors::Internal("Cannot find grad for apply");
            }

            PSRewriteTask task = {};
            task.variable_name = var->name();
            task.update_op = node;
            task.recv_op = grad;
            task_map.insert(std::make_pair(var->name(), task));
          }
        }

        for (Edge *edge : graph->edges()) {
          for (auto &kv : task_map) {
            if (edge->src() == kv.second.update_op) {
              kv.second.update_out_nodes.emplace_back(edge->dst(),
                                                      edge->dst_input());
            } else if (edge->src() == kv.second.send_op) {
              kv.second.send_out_nodes.emplace_back(edge->dst(),
                                                    edge->dst_input());
            }
          }
        }

        for (Node *node : graph->op_nodes()) {
          if (IsSend(node)) {
            Node *var;
            TF_RETURN_IF_ERROR(node->input_node(0, &var));
            auto iter = task_map.find(var->name());
            if (iter != std::end(task_map)) {
              iter->second.var_op = var;
              iter->second.send_op = node;
            }
          }
        }

        for (auto &kv : task_map) {
          PSRewriteTask &task = kv.second;
          {
            DataType dtype;
            TF_RETURN_IF_ERROR(
                GetNodeAttr(task.update_op->attrs(), "T", &dtype));
            string tensor_name;
            TF_RETURN_IF_ERROR(GetNodeAttr(task.recv_op->attrs(), "tensor_name",
                                           &tensor_name));
            string send_device;
            TF_RETURN_IF_ERROR(GetNodeAttr(task.recv_op->attrs(), "send_device",
                                           &send_device));
            string recv_device;
            TF_RETURN_IF_ERROR(GetNodeAttr(task.recv_op->attrs(), "recv_device",
                                           &recv_device));
            int64 send_device_incarnation;
            TF_RETURN_IF_ERROR(GetNodeAttr(task.recv_op->attrs(),
                                           "send_device_incarnation",
                                           &send_device_incarnation));

            NodeBuilder builder(task.update_op->name(),
                                "RecvApplyGradientDescent");
            builder.Attr("T", dtype);
            builder.Attr("variable_name", task.variable_name);
            builder.Attr("tensor_name", tensor_name);
            builder.Attr("send_device", send_device);
            builder.Attr("recv_device", recv_device);
            builder.Attr("send_device_incarnation", send_device_incarnation);

            Node *var, *lr;
            TF_RETURN_IF_ERROR(task.update_op->input_node(0, &var));
            builder.Input(var, 0);
            TF_RETURN_IF_ERROR(task.update_op->input_node(1, &lr));
            builder.Input(lr, 0);

            Node *fused_op;
            TF_RETURN_IF_ERROR(builder.Finalize(graph, &fused_op));

            graph->RemoveNode(task.recv_op);
            graph->RemoveNode(task.update_op);

            for (const auto &out_node : task.update_out_nodes) {
              if (out_node.index == Graph::kControlSlot) {
                graph->AddControlEdge(fused_op, out_node.node);
              } else {
                graph->AddEdge(fused_op, 0, out_node.node, out_node.index);
              }
            }
            TF_RETURN_IF_ERROR(graph->IsValidNode(fused_op));
          }
          {
            DataType dtype;
            TF_RETURN_IF_ERROR(GetNodeAttr(task.send_op->attrs(), "T", &dtype));
            string tensor_name;
            TF_RETURN_IF_ERROR(GetNodeAttr(task.send_op->attrs(), "tensor_name",
                                           &tensor_name));
            string send_device;
            TF_RETURN_IF_ERROR(GetNodeAttr(task.send_op->attrs(), "send_device",
                                           &send_device));
            string recv_device;
            TF_RETURN_IF_ERROR(GetNodeAttr(task.send_op->attrs(), "recv_device",
                                           &recv_device));
            int64 send_device_incarnation;
            TF_RETURN_IF_ERROR(GetNodeAttr(task.send_op->attrs(),
                                           "send_device_incarnation",
                                           &send_device_incarnation));

            Node *node;
            NodeBuilder builder(task.send_op->name(), "SendParameter");
            builder.Input(task.var_op);
            builder.Attr("T", dtype);
            builder.Attr("variable_name", task.variable_name);
            builder.Attr("tensor_name", tensor_name);
            builder.Attr("send_device", send_device);
            builder.Attr("recv_device", recv_device);
            builder.Attr("send_device_incarnation", send_device_incarnation);

            TF_RETURN_IF_ERROR(builder.Finalize(graph, &node));

            graph->RemoveNode(task.send_op);

            for (const auto &out_node : task.send_out_nodes) {
              if (out_node.index == Graph::kControlSlot) {
                graph->AddControlEdge(node, out_node.node);
              } else {
                graph->AddEdge(node, 0, out_node.node, out_node.index);
              }
            }
            TF_RETURN_IF_ERROR(graph->IsValidNode(node));
          }
          LOG(INFO) << "ByteScheduler taking over " << task.variable_name;
        }
      }
    }

    return Status::OK();
  }
};

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PLACEMENT, 0,
                      GraphRewritePass);
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PARTITIONING, 0,
                      WorkerRewritePass);
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PARTITIONING, 0,
                      PSRewritePass);
