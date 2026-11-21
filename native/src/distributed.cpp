#include "norch/distributed.hpp"

#include <ATen/ATen.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

#ifdef NORCH_HAS_NCCL
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#endif

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "norch/addon_state.hpp"
#include "norch/convert.hpp"
#include "norch/tensor_wrap.hpp"

namespace norch {

namespace {

[[nodiscard]] c10d::ReduceOp ParseReduceOp(Napi::Env env, const Napi::Value& val) {
  if (val.IsUndefined() || val.IsNull()) {
    return c10d::ReduceOp::SUM;
  }
  if (val.IsNumber()) {
    return static_cast<c10d::ReduceOp::RedOpType>(val.As<Napi::Number>().Uint32Value());
  }
  const std::string s = val.As<Napi::String>().Utf8Value();
  if (s == "sum" || s == "SUM") return c10d::ReduceOp::SUM;
  if (s == "avg" || s == "AVG") return c10d::ReduceOp::AVG;
  if (s == "product" || s == "PRODUCT" || s == "prod") return c10d::ReduceOp::PRODUCT;
  if (s == "min" || s == "MIN") return c10d::ReduceOp::MIN;
  if (s == "max" || s == "MAX") return c10d::ReduceOp::MAX;
  throw Napi::RangeError::New(env, "Unknown ReduceOp: " + s);
}

[[nodiscard]] std::vector<std::uint8_t> SerializeTensorBytes(const at::Tensor& t) {
  at::Tensor cpu_contig = t.detach().to(c10::kCPU).contiguous();
  const std::size_t nbytes = cpu_contig.nbytes();
  std::vector<std::uint8_t> bytes(nbytes);
  if (nbytes > 0) {
    std::memcpy(bytes.data(), cpu_contig.data_ptr(), nbytes);
  }
  return bytes;
}

[[nodiscard]] at::Tensor DeserializeTensorLike(
    const std::vector<std::uint8_t>& bytes,
    const at::Tensor& ref) {
  at::Tensor out = at::empty(ref.sizes(), at::TensorOptions().dtype(ref.scalar_type()).device(c10::kCPU));
  TORCH_CHECK(
      bytes.size() == out.nbytes(),
      "Distributed tensor byte size mismatch: got ",
      bytes.size(),
      ", expected ",
      out.nbytes());
  if (!bytes.empty()) {
    std::memcpy(out.data_ptr(), bytes.data(), bytes.size());
  }
  return out;
}

class CompletedWork final : public c10d::Work {
 public:
  explicit CompletedWork(int rank, c10d::OpType op_type) : c10d::Work(rank, op_type) {}
  bool isCompleted() override { return true; }
  bool isSuccess() const override { return true; }
  bool wait(std::chrono::milliseconds /*timeout*/) override { return true; }
};

/**
 * Multi-process c10d::Backend over c10d::Store (TCPStore or FileStore).
 * Enables real multi-process distributed training on CPU (and fallback when Gloo headers
 * are absent from prebuilt Python wheel libtorch) while sharing the exact c10d::Backend
 * interface used by c10d::ProcessGroupNCCL on multi-GPU.
 */
class NorchStoreBackend final : public c10d::Backend {
 public:
  NorchStoreBackend(
      c10::intrusive_ptr<c10d::Store> store,
      int rank,
      int size,
      std::string backend_name = "gloo")
      : c10d::Backend(rank, size),
        store_(std::move(store)),
        backend_name_(std::move(backend_name)) {}

  const std::string getBackendName() const override { return backend_name_; }

  c10::intrusive_ptr<c10d::Work> barrier(
      const c10d::BarrierOptions& /*opts*/ = c10d::BarrierOptions()) override {
    const auto seq = next_seq();
    const std::string base = "bar/" + std::to_string(seq);
    store_->set(base + "/" + std::to_string(rank_), std::vector<std::uint8_t>{1});
    std::vector<std::string> keys;
    keys.reserve(static_cast<std::size_t>(size_));
    for (int r = 0; r < size_; ++r) {
      keys.push_back(base + "/" + std::to_string(r));
    }
    store_->wait(keys);
    return c10::make_intrusive<CompletedWork>(rank_, c10d::OpType::BARRIER);
  }

  c10::intrusive_ptr<c10d::Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const c10d::BroadcastOptions& opts = c10d::BroadcastOptions()) override {
    TORCH_CHECK(tensors.size() == 1, "broadcast expects single-tensor list");
    const c10::NoGradGuard no_grad;
    const auto seq = next_seq();
    const std::string key = "bcast/" + std::to_string(seq);
    const int root = static_cast<int>(opts.rootRank);
    auto& t = tensors[0];
    if (rank_ == root) {
      store_->set(key, SerializeTensorBytes(t));
    } else {
      store_->wait({key});
      const auto bytes = store_->get(key);
      at::Tensor received = DeserializeTensorLike(bytes, t);
      t.copy_(received);
    }
    barrier();
    return c10::make_intrusive<CompletedWork>(rank_, c10d::OpType::BROADCAST);
  }

  c10::intrusive_ptr<c10d::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceOptions& opts = c10d::AllreduceOptions()) override {
    TORCH_CHECK(tensors.size() == 1, "allreduce expects single-tensor list");
    const c10::NoGradGuard no_grad;
    const auto seq = next_seq();
    const std::string base = "ar/" + std::to_string(seq);
    auto& t = tensors[0];
    store_->set(base + "/" + std::to_string(rank_), SerializeTensorBytes(t));

    std::vector<std::string> keys;
    keys.reserve(static_cast<std::size_t>(size_));
    for (int r = 0; r < size_; ++r) {
      keys.push_back(base + "/" + std::to_string(r));
    }
    store_->wait(keys);

    at::Tensor acc = DeserializeTensorLike(store_->get(keys[0]), t);
    for (int r = 1; r < size_; ++r) {
      at::Tensor other = DeserializeTensorLike(store_->get(keys[static_cast<std::size_t>(r)]), t);
      switch (opts.reduceOp.op_) {
        case c10d::ReduceOp::SUM:
        case c10d::ReduceOp::AVG:
          acc.add_(other);
          break;
        case c10d::ReduceOp::PRODUCT:
          acc.mul_(other);
          break;
        case c10d::ReduceOp::MIN:
          acc = at::minimum(acc, other);
          break;
        case c10d::ReduceOp::MAX:
          acc = at::maximum(acc, other);
          break;
        default:
          TORCH_CHECK(false, "Unsupported ReduceOp in NorchStoreBackend");
      }
    }
    if (opts.reduceOp.op_ == c10d::ReduceOp::AVG) {
      if (acc.is_floating_point() || acc.is_complex()) {
        acc.div_(static_cast<double>(size_));
      } else {
        acc.floor_divide_(size_);
      }
    }
    t.copy_(acc);
    barrier();
    return c10::make_intrusive<CompletedWork>(rank_, c10d::OpType::ALLREDUCE);
  }

  c10::intrusive_ptr<c10d::Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const c10d::AllgatherOptions& /*opts*/ = c10d::AllgatherOptions()) override {
    TORCH_CHECK(inputTensors.size() == 1 && outputTensors.size() == 1, "allgather expects 1 input list");
    const c10::NoGradGuard no_grad;
    auto& out_list = outputTensors[0];
    TORCH_CHECK(
        std::cmp_equal(out_list.size(), size_),
        "allgather output list length must equal worldSize");
    const auto seq = next_seq();
    const std::string base = "ag/" + std::to_string(seq);
    store_->set(base + "/" + std::to_string(rank_), SerializeTensorBytes(inputTensors[0]));

    std::vector<std::string> keys;
    keys.reserve(static_cast<std::size_t>(size_));
    for (int r = 0; r < size_; ++r) {
      keys.push_back(base + "/" + std::to_string(r));
    }
    store_->wait(keys);

    for (int r = 0; r < size_; ++r) {
      const auto bytes = store_->get(keys[static_cast<std::size_t>(r)]);
      out_list[static_cast<std::size_t>(r)].copy_(
          DeserializeTensorLike(bytes, out_list[static_cast<std::size_t>(r)]));
    }
    barrier();
    return c10::make_intrusive<CompletedWork>(rank_, c10d::OpType::ALLGATHER);
  }

  c10::intrusive_ptr<c10d::Work> _allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const c10d::AllgatherOptions& /*opts*/ = c10d::AllgatherOptions()) override {
    const c10::NoGradGuard no_grad;
    const auto seq = next_seq();
    const std::string base = "agb/" + std::to_string(seq);
    store_->set(base + "/" + std::to_string(rank_), SerializeTensorBytes(inputBuffer));

    std::vector<std::string> keys;
    keys.reserve(static_cast<std::size_t>(size_));
    for (int r = 0; r < size_; ++r) {
      keys.push_back(base + "/" + std::to_string(r));
    }
    store_->wait(keys);

    std::vector<at::Tensor> parts;
    parts.reserve(static_cast<std::size_t>(size_));
    for (int r = 0; r < size_; ++r) {
      parts.push_back(DeserializeTensorLike(store_->get(keys[static_cast<std::size_t>(r)]), inputBuffer));
    }
    at::Tensor stacked = at::cat(parts, 0).reshape(outputBuffer.sizes());
    outputBuffer.copy_(stacked);
    barrier();
    return c10::make_intrusive<CompletedWork>(rank_, c10d::OpType::_ALLGATHER_BASE);
  }

  c10::intrusive_ptr<c10d::Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions()) override {
    TORCH_CHECK(outputTensors.size() == 1 && inputTensors.size() == 1, "reduce_scatter expects 1 list");
    const c10::NoGradGuard no_grad;
    const auto& inputs = inputTensors[0];
    TORCH_CHECK(std::cmp_equal(inputs.size(), size_), "reduce_scatter input list length must match worldSize");
    const auto seq = next_seq();
    const std::string base = "rs/" + std::to_string(seq);
    for (int dst = 0; dst < size_; ++dst) {
      const std::string k = base + "/" + std::to_string(dst) + "/" + std::to_string(rank_);
      store_->set(k, SerializeTensorBytes(inputs[static_cast<std::size_t>(dst)]));
    }

    std::vector<std::string> my_keys;
    my_keys.reserve(static_cast<std::size_t>(size_));
    for (int src = 0; src < size_; ++src) {
      my_keys.push_back(base + "/" + std::to_string(rank_) + "/" + std::to_string(src));
    }
    store_->wait(my_keys);

    at::Tensor acc = DeserializeTensorLike(store_->get(my_keys[0]), outputTensors[0]);
    for (int src = 1; src < size_; ++src) {
      at::Tensor other = DeserializeTensorLike(
          store_->get(my_keys[static_cast<std::size_t>(src)]), outputTensors[0]);
      switch (opts.reduceOp.op_) {
        case c10d::ReduceOp::SUM:
        case c10d::ReduceOp::AVG:
          acc.add_(other);
          break;
        case c10d::ReduceOp::PRODUCT:
          acc.mul_(other);
          break;
        case c10d::ReduceOp::MIN:
          acc = at::minimum(acc, other);
          break;
        case c10d::ReduceOp::MAX:
          acc = at::maximum(acc, other);
          break;
        default:
          TORCH_CHECK(false, "Unsupported ReduceOp in reduce_scatter");
      }
    }
    if (opts.reduceOp.op_ == c10d::ReduceOp::AVG) {
      if (acc.is_floating_point() || acc.is_complex()) {
        acc.div_(static_cast<double>(size_));
      } else {
        acc.floor_divide_(size_);
      }
    }
    outputTensors[0].copy_(acc);
    barrier();
    return c10::make_intrusive<CompletedWork>(rank_, c10d::OpType::REDUCE_SCATTER);
  }

  c10::intrusive_ptr<c10d::Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override {
    TORCH_CHECK(tensors.size() == 1, "send expects 1 tensor");
    const auto p2p_seq = p2p_send_seq_[dstRank]++;
    const std::string key = "p2p/" + std::to_string(rank_) + "/" + std::to_string(dstRank) + "/" +
                            std::to_string(tag) + "/" + std::to_string(p2p_seq);
    store_->set(key, SerializeTensorBytes(tensors[0]));
    return c10::make_intrusive<CompletedWork>(rank_, c10d::OpType::SEND);
  }

  c10::intrusive_ptr<c10d::Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override {
    TORCH_CHECK(tensors.size() == 1, "recv expects 1 tensor");
    const c10::NoGradGuard no_grad;
    const auto p2p_seq = p2p_recv_seq_[srcRank]++;
    const std::string key = "p2p/" + std::to_string(srcRank) + "/" + std::to_string(rank_) + "/" +
                            std::to_string(tag) + "/" + std::to_string(p2p_seq);
    store_->wait({key});
    const auto bytes = store_->get(key);
    tensors[0].copy_(DeserializeTensorLike(bytes, tensors[0]));
    return c10::make_intrusive<CompletedWork>(rank_, c10d::OpType::RECV);
  }

 private:
  [[nodiscard]] std::uint64_t next_seq() noexcept { return seq_.fetch_add(1); }

  c10::intrusive_ptr<c10d::Store> store_;
  std::string backend_name_;
  std::atomic<std::uint64_t> seq_{0};
  std::unordered_map<int, std::uint64_t> p2p_send_seq_;
  std::unordered_map<int, std::uint64_t> p2p_recv_seq_;
};

[[nodiscard]] c10::intrusive_ptr<c10d::Store> CreateStoreFromOptions(
    Napi::Env env,
    const Napi::Object& opts,
    int rank,
    int world_size) {
  const std::string store_type =
      opts.Has("storeType") && opts.Get("storeType").IsString()
          ? opts.Get("storeType").As<Napi::String>().Utf8Value()
          : "tcp";
  const std::int64_t timeout_ms =
      opts.Has("timeoutMs") && opts.Get("timeoutMs").IsNumber()
          ? opts.Get("timeoutMs").As<Napi::Number>().Int64Value()
          : 60000;

  if (store_type == "file") {
    if (!opts.Has("filePath") || !opts.Get("filePath").IsString()) {
      throw Napi::TypeError::New(env, "FileStore requires filePath");
    }
    const std::string path = opts.Get("filePath").As<Napi::String>().Utf8Value();
    auto store = c10::make_intrusive<c10d::FileStore>(path, world_size);
    store->setTimeout(std::chrono::milliseconds(timeout_ms));
    return store;
  }

  const std::string host =
      opts.Has("masterAddr") && opts.Get("masterAddr").IsString()
          ? opts.Get("masterAddr").As<Napi::String>().Utf8Value()
          : "127.0.0.1";
  const auto port =
      opts.Has("masterPort") && opts.Get("masterPort").IsNumber()
          ? static_cast<std::uint16_t>(opts.Get("masterPort").As<Napi::Number>().Uint32Value())
          : static_cast<std::uint16_t>(29500);

  c10d::TCPStoreOptions tcp_opts;
  tcp_opts.port = port;
  tcp_opts.isServer = (rank == 0);
  tcp_opts.numWorkers = static_cast<std::size_t>(world_size);
  tcp_opts.waitWorkers = true;
  tcp_opts.timeout = std::chrono::milliseconds(timeout_ms);
  tcp_opts.useLibUV = true;
  return c10::make_intrusive<c10d::TCPStore>(host, tcp_opts);
}

class ProcessGroupWrap : public Napi::ObjectWrap<ProcessGroupWrap> {
 public:
  static void Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(
        env,
        "NativeProcessGroup",
        {
            InstanceMethod("rank", &ProcessGroupWrap::Rank),
            InstanceMethod("worldSize", &ProcessGroupWrap::WorldSize),
            InstanceMethod("backendName", &ProcessGroupWrap::BackendName),
            InstanceMethod("allReduce", &ProcessGroupWrap::AllReduce),
            InstanceMethod("broadcast", &ProcessGroupWrap::Broadcast),
            InstanceMethod("allGather", &ProcessGroupWrap::AllGather),
            InstanceMethod("allGatherIntoTensor", &ProcessGroupWrap::AllGatherIntoTensor),
            InstanceMethod("reduceScatter", &ProcessGroupWrap::ReduceScatter),
            InstanceMethod("barrier", &ProcessGroupWrap::Barrier),
            InstanceMethod("send", &ProcessGroupWrap::Send),
            InstanceMethod("recv", &ProcessGroupWrap::Recv),
            InstanceMethod("storeSet", &ProcessGroupWrap::StoreSet),
            InstanceMethod("storeGet", &ProcessGroupWrap::StoreGet),
            InstanceMethod("destroy", &ProcessGroupWrap::Destroy),
            InstanceMethod("isDestroyed", &ProcessGroupWrap::IsDestroyed),
        });
    AddonState::Get(env).process_group_ctor = Napi::Persistent(func);
    exports.Set("NativeProcessGroup", func);
  }

  explicit ProcessGroupWrap(const Napi::CallbackInfo& info)
      : Napi::ObjectWrap<ProcessGroupWrap>(info) {
    const Napi::Env env = info.Env();
    try {
      if (info.Length() < 1 || !info[0].IsObject()) {
        throw Napi::TypeError::New(env, "NativeProcessGroup(options) expects an options object");
      }
      const auto opts = info[0].As<Napi::Object>();
      const std::string backend =
          opts.Has("backend") && opts.Get("backend").IsString()
              ? opts.Get("backend").As<Napi::String>().Utf8Value()
              : "gloo";
      const int rank = opts.Get("rank").As<Napi::Number>().Int32Value();
      const int world_size = opts.Get("worldSize").As<Napi::Number>().Int32Value();
      const std::string prefix =
          opts.Has("groupName") && opts.Get("groupName").IsString()
              ? opts.Get("groupName").As<Napi::String>().Utf8Value()
              : "default";

      auto base_store = CreateStoreFromOptions(env, opts, rank, world_size);
      store_ = c10::make_intrusive<c10d::PrefixStore>(prefix, std::move(base_store));

      if (backend == "nccl") {
#ifdef NORCH_HAS_NCCL
        auto nccl_opts = c10d::ProcessGroupNCCL::Options::create();
        backend_ = c10::make_intrusive<c10d::ProcessGroupNCCL>(
            store_, rank, world_size, std::move(nccl_opts));
#else
        throw Napi::Error::New(env, "NCCL backend unavailable in this norch build");
#endif
      } else if (backend == "gloo" || backend == "tcp") {
        backend_ = c10::make_intrusive<NorchStoreBackend>(store_, rank, world_size, backend);
      } else {
        throw Napi::RangeError::New(env, "Unsupported distributed backend: " + backend);
      }
    } catch (const Napi::Error& err) {
      err.ThrowAsJavaScriptException();
    } catch (const c10::Error& err) {
      Napi::Error::New(env, err.what_without_backtrace()).ThrowAsJavaScriptException();
    } catch (const std::exception& ex) {
      Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
  }

  ~ProcessGroupWrap() override = default;
  ProcessGroupWrap(const ProcessGroupWrap&) = delete;
  ProcessGroupWrap& operator=(const ProcessGroupWrap&) = delete;
  ProcessGroupWrap(ProcessGroupWrap&&) = delete;
  ProcessGroupWrap& operator=(ProcessGroupWrap&&) = delete;

 private:
  void EnsureAlive(Napi::Env env) const {
    if (!backend_) {
      throw Napi::Error::New(env, "ProcessGroup has been destroyed");
    }
  }

  Napi::Value Rank(const Napi::CallbackInfo& info) {
    return GuardNapi(info.Env(), [&]() -> Napi::Value {
      EnsureAlive(info.Env());
      return Napi::Number::New(info.Env(), static_cast<double>(backend_->getRank()));
    });
  }

  Napi::Value WorldSize(const Napi::CallbackInfo& info) {
    return GuardNapi(info.Env(), [&]() -> Napi::Value {
      EnsureAlive(info.Env());
      return Napi::Number::New(info.Env(), static_cast<double>(backend_->getSize()));
    });
  }

  Napi::Value BackendName(const Napi::CallbackInfo& info) {
    return GuardNapi(info.Env(), [&]() -> Napi::Value {
      EnsureAlive(info.Env());
      return Napi::String::New(info.Env(), backend_->getBackendName());
    });
  }

  Napi::Value AllReduce(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      EnsureAlive(env);
      std::vector<at::Tensor> tensors = {TensorWrap::UnwrapMutableTensor(env, info[0])};
      c10d::AllreduceOptions opts;
      if (info.Length() >= 2) {
        opts.reduceOp = ParseReduceOp(env, info[1]);
      }
      auto work = backend_->allreduce(tensors, opts);
      work->wait();
      return env.Undefined();
    });
  }

  Napi::Value Broadcast(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      EnsureAlive(env);
      std::vector<at::Tensor> tensors = {TensorWrap::UnwrapMutableTensor(env, info[0])};
      c10d::BroadcastOptions opts;
      opts.rootRank = info.Length() >= 2 ? info[1].As<Napi::Number>().Int64Value() : 0;
      auto work = backend_->broadcast(tensors, opts);
      work->wait();
      return env.Undefined();
    });
  }

  Napi::Value AllGather(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      EnsureAlive(env);
      std::vector<std::vector<at::Tensor>> outputs = {ParseTensorList(env, info[0])};
      std::vector<at::Tensor> inputs = {TensorWrap::UnwrapMutableTensor(env, info[1])};
      auto work = backend_->allgather(outputs, inputs);
      work->wait();
      return env.Undefined();
    });
  }

  Napi::Value AllGatherIntoTensor(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      EnsureAlive(env);
      auto& out = TensorWrap::UnwrapMutableTensor(env, info[0]);
      auto& in = TensorWrap::UnwrapMutableTensor(env, info[1]);
      auto work = backend_->_allgather_base(out, in);
      work->wait();
      return env.Undefined();
    });
  }

  Napi::Value ReduceScatter(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      EnsureAlive(env);
      std::vector<at::Tensor> outputs = {TensorWrap::UnwrapMutableTensor(env, info[0])};
      std::vector<std::vector<at::Tensor>> inputs = {ParseTensorList(env, info[1])};
      c10d::ReduceScatterOptions opts;
      if (info.Length() >= 3) {
        opts.reduceOp = ParseReduceOp(env, info[2]);
      }
      auto work = backend_->reduce_scatter(outputs, inputs, opts);
      work->wait();
      return env.Undefined();
    });
  }

  Napi::Value Barrier(const Napi::CallbackInfo& info) {
    return GuardNapi(info.Env(), [&]() -> Napi::Value {
      EnsureAlive(info.Env());
      auto work = backend_->barrier();
      work->wait();
      return info.Env().Undefined();
    });
  }

  Napi::Value Send(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      EnsureAlive(env);
      std::vector<at::Tensor> tensors = {TensorWrap::UnwrapMutableTensor(env, info[0])};
      const int dst = info[1].As<Napi::Number>().Int32Value();
      const int tag = info.Length() >= 3 ? info[2].As<Napi::Number>().Int32Value() : 0;
      auto work = backend_->send(tensors, dst, tag);
      work->wait();
      return env.Undefined();
    });
  }

  Napi::Value Recv(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      EnsureAlive(env);
      std::vector<at::Tensor> tensors = {TensorWrap::UnwrapMutableTensor(env, info[0])};
      const int src = info[1].As<Napi::Number>().Int32Value();
      const int tag = info.Length() >= 3 ? info[2].As<Napi::Number>().Int32Value() : 0;
      auto work = backend_->recv(tensors, src, tag);
      work->wait();
      return env.Undefined();
    });
  }

  Napi::Value StoreSet(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      EnsureAlive(env);
      const std::string key = info[0].As<Napi::String>().Utf8Value();
      const std::string val = info[1].As<Napi::String>().Utf8Value();
      store_->set(key, std::vector<std::uint8_t>(val.begin(), val.end()));
      return env.Undefined();
    });
  }

  Napi::Value StoreGet(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      EnsureAlive(env);
      const std::string key = info[0].As<Napi::String>().Utf8Value();
      const auto bytes = store_->get(key);
      return Napi::String::New(env, std::string(bytes.begin(), bytes.end()));
    });
  }

  Napi::Value Destroy(const Napi::CallbackInfo& info) {
    backend_.reset();
    store_.reset();
    return info.Env().Undefined();
  }

  Napi::Value IsDestroyed(const Napi::CallbackInfo& info) {
    return Napi::Boolean::New(info.Env(), !backend_);
  }

  c10::intrusive_ptr<c10d::Store> store_;
  c10::intrusive_ptr<c10d::Backend> backend_;
};

Napi::Value DistIsNcclAvailable(const Napi::CallbackInfo& info) {
#ifdef NORCH_HAS_NCCL
  return Napi::Boolean::New(info.Env(), at::cuda::is_available());
#else
  return Napi::Boolean::New(info.Env(), false);
#endif
}

Napi::Value DistIsGlooAvailable(const Napi::CallbackInfo& info) {
  return Napi::Boolean::New(info.Env(), true);
}

}  // namespace

void RegisterDistributed(Napi::Env env, Napi::Object exports) {
  ProcessGroupWrap::Init(env, exports);
  exports.Set("distIsNcclAvailable", Napi::Function::New(env, DistIsNcclAvailable));
  exports.Set("distIsGlooAvailable", Napi::Function::New(env, DistIsGlooAvailable));
}

}  // namespace norch
