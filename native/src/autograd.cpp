#include "norch/autograd.hpp"

#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <c10/core/AutogradState.h>
#include <c10/core/Device.h>
#include <c10/core/GradMode.h>
#include <c10/core/InferenceMode.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/autograd/variable_info.h>

#include <atomic>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "norch/addon_state.hpp"
#include "norch/convert.hpp"
#include "norch/tensor_wrap.hpp"

namespace norch {

namespace {

[[nodiscard]] std::vector<std::unique_ptr<c10::InferenceMode>>& InferenceModeStack() {
  static thread_local std::vector<std::unique_ptr<c10::InferenceMode>> stack;
  return stack;
}

struct SingleThreadAutogradGuard {
  bool prev_mt = true;
  bool active = false;

  explicit SingleThreadAutogradGuard(const AddonState& state) {
    if (state.custom_autograd_nodes_alive.load() > 0) {
      auto& tls = c10::AutogradState::get_tls_state();
      prev_mt = tls.get_multithreading_enabled();
      tls.set_multithreading_enabled(false);
      active = true;
    }
  }
  ~SingleThreadAutogradGuard() {
    if (active) {
      c10::AutogradState::get_tls_state().set_multithreading_enabled(prev_mt);
    }
  }
  SingleThreadAutogradGuard(const SingleThreadAutogradGuard&) = delete;
  SingleThreadAutogradGuard& operator=(const SingleThreadAutogradGuard&) = delete;
  SingleThreadAutogradGuard(SingleThreadAutogradGuard&&) = delete;
  SingleThreadAutogradGuard& operator=(SingleThreadAutogradGuard&&) = delete;
};

class JsAutogradNode final : public torch::autograd::Node {
 public:
  JsAutogradNode(
      std::string op_name,
      Napi::Env env,
      AddonState* state,
      Napi::Function backward_fn,
      Napi::Object ctx_obj,
      std::size_t num_tensor_inputs)
      : op_name_(std::move(op_name)),
        env_(env),
        state_(state),
        backward_fn_(Napi::Persistent(backward_fn)),
        ctx_obj_(Napi::Persistent(ctx_obj)),
        num_tensor_inputs_(num_tensor_inputs) {
    state_->custom_autograd_nodes_alive.fetch_add(1);
  }

  ~JsAutogradNode() noexcept override {
    try {
      if (state_ != nullptr) {
        state_->custom_autograd_nodes_alive.fetch_sub(1);
        if (std::this_thread::get_id() == state_->js_thread_id()) {
          const Napi::Env env(env_);
          if (env.GetInstanceData<AddonState>() != nullptr) {
            backward_fn_.Reset();
            ctx_obj_.Reset();
          }
        }
      }
    } catch (...) {
      std::fputs("norch: suppressed exception in ~JsAutogradNode\n", stderr);
    }
  }

  JsAutogradNode(const JsAutogradNode&) = delete;
  JsAutogradNode& operator=(const JsAutogradNode&) = delete;
  JsAutogradNode(JsAutogradNode&&) = delete;
  JsAutogradNode& operator=(JsAutogradNode&&) = delete;

  std::string name() const override { return op_name_; }

  void SetOutputInfo(std::vector<torch::autograd::VariableInfo> info) {
    output_info_ = std::move(info);
  }

  void SetMaterializeGrads(bool materialize) noexcept { materialize_grads_ = materialize; }

  void SaveVariables(const torch::autograd::variable_list& vars, const std::unordered_set<at::TensorImpl*>& outputs) {
    saved_vars_.clear();
    saved_vars_.reserve(vars.size());
    for (const auto& v : vars) {
      const bool is_output = v.defined() && outputs.contains(v.unsafeGetTensorImpl());
      saved_vars_.emplace_back(v, is_output);
    }
  }

  void release_variables() override {
    saved_vars_.clear();
    freed_ = true;
  }

  torch::autograd::variable_list apply(torch::autograd::variable_list&& grad_outputs) override {
    auto incoming = std::move(grad_outputs);
    TORCH_CHECK(
        state_ != nullptr && std::this_thread::get_id() == state_->js_thread_id(),
        "Custom JS AutogradFunction '",
        op_name_,
        "' backward invoked off the JS main thread");
    TORCH_CHECK(
        !freed_,
        "Trying to backward through custom JS AutogradFunction '",
        op_name_,
        "' a second time without retainGraph=true");

    const Napi::Env env(env_);
    const Napi::HandleScope scope(env);

    // Unpack saved variables (runs libtorch version-counter check)
    auto saved_arr = Napi::Array::New(env, saved_vars_.size());
    auto self_ptr = c10::intrusive_ptr<torch::autograd::Node>::unsafe_reclaim_from_nonowning(this);
    // Keep intrusive refcount balanced: unsafe_reclaim does not increment, so copy + release!
    auto self_guard = self_ptr;
    self_ptr.release();

    for (std::size_t i = 0; i < saved_vars_.size(); ++i) {
      auto unpacked = saved_vars_[i].unpack(self_guard);
      if (unpacked.defined()) {
        saved_arr.Set(static_cast<std::uint32_t>(i), TensorWrap::NewInstance(env, std::move(unpacked)));
      } else {
        saved_arr.Set(static_cast<std::uint32_t>(i), env.Null());
      }
    }

    auto ctx = ctx_obj_.Value();
    ctx.Set("_savedNativeTensors", saved_arr);

    // Populate ctx.needsInputGrad matching PyTorch Function backward semantics
    const auto num_edges = next_edges().size();
    auto needs_grad_arr = Napi::Array::New(env, num_edges);
    for (std::size_t i = 0; i < num_edges; ++i) {
      needs_grad_arr.Set(
          static_cast<std::uint32_t>(i), Napi::Boolean::New(env, task_should_compute_output(i)));
    }
    ctx.Set("needsInputGrad", needs_grad_arr);

    at::OptionalDeviceGuard device_guard;
    auto js_grads = Napi::Array::New(env, incoming.size());
    for (std::size_t i = 0; i < incoming.size(); ++i) {
      if (incoming[i].defined()) {
        js_grads.Set(static_cast<std::uint32_t>(i), TensorWrap::NewInstance(env, std::move(incoming[i])));
      } else if (materialize_grads_ && i < output_info_.size()) {
        js_grads.Set(
            static_cast<std::uint32_t>(i),
            TensorWrap::NewInstance(env, output_info_[i].zeros(device_guard)));
      } else {
        js_grads.Set(static_cast<std::uint32_t>(i), env.Null());
      }
    }

    const Napi::Value ret = backward_fn_.Call({ctx, js_grads});
    if (env.IsExceptionPending()) {
      const Napi::Error err = env.GetAndClearPendingException();
      TORCH_CHECK(false, "JS exception in ", op_name_, ".backward: ", err.Message());
    }

    torch::autograd::variable_list result;
    result.reserve(num_tensor_inputs_);
    if (ret.IsArray()) {
      const auto arr = ret.As<Napi::Array>();
      TORCH_CHECK(
          arr.Length() == num_tensor_inputs_,
          op_name_,
          ".backward returned ",
          arr.Length(),
          " gradients, expected ",
          num_tensor_inputs_);
      for (std::uint32_t i = 0; i < arr.Length(); ++i) {
        const auto item = arr.Get(i);
        if (item.IsNull() || item.IsUndefined()) {
          result.emplace_back();
        } else {
          result.push_back(TensorWrap::UnwrapTensor(env, item));
        }
      }
    } else if (num_tensor_inputs_ == 1) {
      if (ret.IsNull() || ret.IsUndefined()) {
        result.emplace_back();
      } else {
        result.push_back(TensorWrap::UnwrapTensor(env, ret));
      }
    } else {
      TORCH_CHECK(
          false,
          op_name_,
          ".backward must return an Array of ",
          num_tensor_inputs_,
          " gradients");
    }
    return result;
  }

 private:
  std::string op_name_;
  napi_env env_;
  AddonState* state_;
  Napi::FunctionReference backward_fn_;
  Napi::ObjectReference ctx_obj_;
  std::size_t num_tensor_inputs_;
  std::vector<torch::autograd::SavedVariable> saved_vars_;
  std::vector<torch::autograd::VariableInfo> output_info_;
  bool materialize_grads_ = true;
  bool freed_ = false;
};

Napi::Value IsGradEnabled(const Napi::CallbackInfo& info) {
  return Napi::Boolean::New(info.Env(), c10::GradMode::is_enabled());
}

Napi::Value SetGradEnabled(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const bool prev = c10::GradMode::is_enabled();
    const bool next = info.Length() >= 1 && info[0].ToBoolean().Value();
    c10::GradMode::set_enabled(next);
    return Napi::Boolean::New(env, prev);
  });
}

Napi::Value IsInferenceModeEnabled(const Napi::CallbackInfo& info) {
  return Napi::Boolean::New(info.Env(), c10::InferenceMode::is_enabled());
}

Napi::Value PushInferenceMode(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const bool enabled = info.Length() < 1 || info[0].ToBoolean().Value();
    auto& stack = InferenceModeStack();
    stack.push_back(std::make_unique<c10::InferenceMode>(enabled));
    return Napi::Number::New(env, static_cast<double>(stack.size()));
  });
}

Napi::Value PopInferenceMode(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    auto& stack = InferenceModeStack();
    if (!stack.empty()) {
      stack.pop_back();
    }
    return env.Undefined();
  });
}

Napi::Value IsAutocastEnabled(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto dev = info.Length() >= 1 ? ParseDevice(env, info[0]).type() : c10::kCUDA;
    return Napi::Boolean::New(env, at::autocast::is_autocast_enabled(dev));
  });
}

Napi::Value SetAutocastEnabled(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto dev = ParseDevice(env, info[0]).type();
    const bool enabled = info[1].ToBoolean().Value();
    at::autocast::set_autocast_enabled(dev, enabled);
    return env.Undefined();
  });
}

Napi::Value GetAutocastDtype(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto dev = info.Length() >= 1 ? ParseDevice(env, info[0]).type() : c10::kCUDA;
    return Napi::String::New(env, ScalarTypeToString(at::autocast::get_autocast_dtype(dev)));
  });
}

Napi::Value SetAutocastDtype(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto dev = ParseDevice(env, info[0]).type();
    const auto dtype = ParseScalarType(env, info[1]);
    at::autocast::set_autocast_dtype(dev, dtype);
    return env.Undefined();
  });
}

Napi::Value AutocastClearCache(const Napi::CallbackInfo& info) {
  at::autocast::clear_cache();
  return info.Env().Undefined();
}

Napi::Value AutocastIncrementNesting(const Napi::CallbackInfo& info) {
  return Napi::Number::New(info.Env(), static_cast<double>(at::autocast::increment_nesting()));
}

Napi::Value AutocastDecrementNesting(const Napi::CallbackInfo& info) {
  return Napi::Number::New(info.Env(), static_cast<double>(at::autocast::decrement_nesting()));
}

Napi::Value AutogradBackward(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    if (info.Length() < 1) {
      throw Napi::TypeError::New(env, "autogradBackward(tensors, gradTensors?, options?) expects tensors");
    }
    const auto tensors = ParseTensorList(env, info[0]);
    torch::autograd::variable_list grad_tensors;
    if (info.Length() >= 2 && info[1].IsArray()) {
      const auto arr = info[1].As<Napi::Array>();
      grad_tensors.reserve(arr.Length());
      for (std::uint32_t i = 0; i < arr.Length(); ++i) {
        const auto item = arr.Get(i);
        if (item.IsNull() || item.IsUndefined()) {
          grad_tensors.emplace_back();
        } else {
          grad_tensors.push_back(TensorWrap::UnwrapTensor(env, item));
        }
      }
    }
    std::optional<bool> retain_graph = std::nullopt;
    bool create_graph = false;
    torch::autograd::variable_list inputs;
    if (info.Length() >= 3 && info[2].IsObject()) {
      const auto opts = info[2].As<Napi::Object>();
      if (opts.Has("retainGraph") && opts.Get("retainGraph").IsBoolean()) {
        retain_graph = opts.Get("retainGraph").As<Napi::Boolean>().Value();
      }
      if (opts.Has("createGraph") && opts.Get("createGraph").IsBoolean()) {
        create_graph = opts.Get("createGraph").As<Napi::Boolean>().Value();
      }
      if (opts.Has("inputs") && opts.Get("inputs").IsArray()) {
        inputs = ParseTensorList(env, opts.Get("inputs"));
      }
    }

    const SingleThreadAutogradGuard mt_guard(AddonState::Get(env));
    torch::autograd::backward(tensors, grad_tensors, retain_graph, create_graph, inputs);
    return env.Undefined();
  });
}

Napi::Value AutogradGrad(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    if (info.Length() < 2) {
      throw Napi::TypeError::New(env, "autogradGrad(outputs, inputs, gradOutputs?, options?) expects >= 2 args");
    }
    const auto outputs = ParseTensorList(env, info[0]);
    const auto inputs = ParseTensorList(env, info[1]);
    torch::autograd::variable_list grad_outputs;
    if (info.Length() >= 3 && info[2].IsArray()) {
      const auto arr = info[2].As<Napi::Array>();
      grad_outputs.reserve(arr.Length());
      for (std::uint32_t i = 0; i < arr.Length(); ++i) {
        const auto item = arr.Get(i);
        if (item.IsNull() || item.IsUndefined()) {
          grad_outputs.emplace_back();
        } else {
          grad_outputs.push_back(TensorWrap::UnwrapTensor(env, item));
        }
      }
    }
    std::optional<bool> retain_graph = std::nullopt;
    bool create_graph = false;
    bool allow_unused = false;
    if (info.Length() >= 4 && info[3].IsObject()) {
      const auto opts = info[3].As<Napi::Object>();
      if (opts.Has("retainGraph") && opts.Get("retainGraph").IsBoolean()) {
        retain_graph = opts.Get("retainGraph").As<Napi::Boolean>().Value();
      }
      if (opts.Has("createGraph") && opts.Get("createGraph").IsBoolean()) {
        create_graph = opts.Get("createGraph").As<Napi::Boolean>().Value();
      }
      if (opts.Has("allowUnused") && opts.Get("allowUnused").IsBoolean()) {
        allow_unused = opts.Get("allowUnused").As<Napi::Boolean>().Value();
      }
    }

    const SingleThreadAutogradGuard mt_guard(AddonState::Get(env));
    auto grads =
        torch::autograd::grad(outputs, inputs, grad_outputs, retain_graph, create_graph, allow_unused);
    auto js_arr = Napi::Array::New(env, grads.size());
    for (std::size_t i = 0; i < grads.size(); ++i) {
      if (grads[i].defined()) {
        js_arr.Set(static_cast<std::uint32_t>(i), TensorWrap::NewInstance(env, std::move(grads[i])));
      } else {
        js_arr.Set(static_cast<std::uint32_t>(i), env.Null());
      }
    }
    return js_arr;
  });
}

Napi::Value ApplyCustomAutograd(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    if (info.Length() < 5 || !info[0].IsString() || !info[1].IsArray() || !info[2].IsObject() ||
        !info[3].IsFunction() || !info[4].IsFunction()) {
      throw Napi::TypeError::New(
          env,
          "applyCustomAutograd(name, tensorInputs, ctxObj, forwardFn, backwardFn) invalid arguments");
    }

    const std::string op_name = info[0].As<Napi::String>().Utf8Value();
    const auto input_vars = ParseTensorList(env, info[1]);
    const auto ctx_obj = info[2].As<Napi::Object>();
    const auto forward_fn = info[3].As<Napi::Function>();
    const auto backward_fn = info[4].As<Napi::Function>();

    auto& state = AddonState::Get(env);
    const bool is_executable =
        c10::GradMode::is_enabled() && torch::autograd::any_variable_requires_grad(input_vars);

    c10::intrusive_ptr<JsAutogradNode> node;
    if (is_executable) {
      node = c10::make_intrusive<JsAutogradNode>(
          op_name + "Backward", env, &state, backward_fn, ctx_obj, input_vars.size());
      node->set_next_edges(torch::autograd::collect_next_edges(input_vars));
      node->clear_input_metadata();
    }

    // Run JS forward inside NoGradGuard, matching PyTorch Function.forward semantics
    Napi::Value fwd_ret;
    {
      const c10::NoGradGuard no_grad;
      fwd_ret = forward_fn.Call({ctx_obj});
      if (env.IsExceptionPending()) {
        throw env.GetAndClearPendingException();
      }
    }

    if (!fwd_ret.IsArray()) {
      throw Napi::TypeError::New(env, "Internal forward wrapper must return Array of NativeTensors");
    }
    const auto js_outputs = fwd_ret.As<Napi::Array>();
    std::vector<std::optional<torch::autograd::Variable>> raw_outputs;
    raw_outputs.reserve(js_outputs.Length());
    std::unordered_set<at::TensorImpl*> output_ptrs;
    for (std::uint32_t i = 0; i < js_outputs.Length(); ++i) {
      const auto item = js_outputs.Get(i);
      if (item.IsNull() || item.IsUndefined()) {
        raw_outputs.emplace_back(std::nullopt);
      } else {
        auto t = TensorWrap::UnwrapTensor(env, item);
        output_ptrs.insert(t.unsafeGetTensorImpl());
        raw_outputs.emplace_back(std::move(t));
      }
    }

    std::unordered_set<at::TensorImpl*> non_diff;
    if (ctx_obj.Has("_nonDifferentiable") && ctx_obj.Get("_nonDifferentiable").IsArray()) {
      const auto nd = ctx_obj.Get("_nonDifferentiable").As<Napi::Array>();
      for (std::uint32_t i = 0; i < nd.Length(); ++i) {
        const auto& t = TensorWrap::UnwrapTensor(env, nd.Get(i));
        non_diff.insert(t.unsafeGetTensorImpl());
      }
    }

    std::unordered_set<at::TensorImpl*> dirty_inputs;
    if (ctx_obj.Has("_dirtyInputs") && ctx_obj.Get("_dirtyInputs").IsArray()) {
      const auto di = ctx_obj.Get("_dirtyInputs").As<Napi::Array>();
      for (std::uint32_t i = 0; i < di.Length(); ++i) {
        auto& t = TensorWrap::UnwrapMutableTensor(env, di.Get(i));
        dirty_inputs.insert(t.unsafeGetTensorImpl());
      }
    }

    const torch::autograd::_jvp_fn_t jvp_fn =
        [](const torch::autograd::variable_list& /*inputs*/,
           const torch::autograd::variable_list& /*gI*/) -> torch::autograd::variable_list {
      TORCH_CHECK(false, "jvp is not implemented for custom JS AutogradFunction");
    };
    const torch::autograd::_view_as_self_fn_t view_as_self_fn = [](const at::Tensor& x) {
      return x.view_as(x);
    };

    c10::intrusive_ptr<torch::autograd::Node> attached_node;
    const std::unordered_set<at::TensorImpl*> empty_set;
    auto wrapped_outputs = torch::autograd::_wrap_outputs(
        input_vars,
        non_diff,
        dirty_inputs,
        raw_outputs,
        is_executable ? c10::intrusive_ptr<torch::autograd::Node>(node) : nullptr,
        jvp_fn,
        empty_set,
        view_as_self_fn,
        /*pure_view=*/false,
        attached_node);

    if (is_executable) {
      if (ctx_obj.Has("_materializeGrads") && ctx_obj.Get("_materializeGrads").IsBoolean()) {
        node->SetMaterializeGrads(ctx_obj.Get("_materializeGrads").As<Napi::Boolean>().Value());
      }
      std::vector<torch::autograd::VariableInfo> out_info;
      out_info.reserve(wrapped_outputs.size());
      for (const auto& out : wrapped_outputs) {
        if (out.has_value()) {
          out_info.emplace_back(*out);
        } else {
          out_info.emplace_back();
        }
      }
      node->SetOutputInfo(std::move(out_info));

      if (ctx_obj.Has("_toSave") && ctx_obj.Get("_toSave").IsArray()) {
        const auto to_save = ParseTensorList(env, ctx_obj.Get("_toSave"));
        node->SaveVariables(to_save, output_ptrs);
      }
    }

    auto result = Napi::Array::New(env, wrapped_outputs.size());
    for (std::size_t i = 0; i < wrapped_outputs.size(); ++i) {
      if (wrapped_outputs[i].has_value() && wrapped_outputs[i]->defined()) {
        result.Set(
            static_cast<std::uint32_t>(i),
            TensorWrap::NewInstance(env, std::move(*wrapped_outputs[i])));
      } else {
        result.Set(static_cast<std::uint32_t>(i), env.Null());
      }
    }
    return result;
  });
}

Napi::Value EnterDualLevel(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto level = torch::autograd::forward_ad::enter_dual_level();
    return Napi::Number::New(env, static_cast<double>(level));
  });
}

Napi::Value ExitDualLevel(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    if (info.Length() < 1 || !info[0].IsNumber()) {
      throw Napi::TypeError::New(env, "exitDualLevel(level) expects a numeric level");
    }
    const auto level = static_cast<std::uint64_t>(info[0].As<Napi::Number>().Int64Value());
    torch::autograd::forward_ad::exit_dual_level(level);
    return env.Undefined();
  });
}

Napi::Value MakeDual(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& primal = TensorWrap::UnwrapTensor(env, info[0]);
    const auto& tangent = TensorWrap::UnwrapTensor(env, info[1]);
    const auto level = info[2].As<Napi::Number>().Int64Value();
    return TensorWrap::NewInstance(env, at::_make_dual(primal, tangent, level));
  });
}

Napi::Value UnpackDual(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& dual = TensorWrap::UnwrapTensor(env, info[0]);
    const auto level = info[1].As<Napi::Number>().Int64Value();
    auto [primal, tangent] = at::_unpack_dual(dual, level);
    auto obj = Napi::Object::New(env);
    obj.Set("primal", TensorWrap::NewInstance(env, std::move(primal)));
    obj.Set(
        "tangent",
        tangent.defined() ? TensorWrap::NewInstance(env, tangent) : env.Null());
    return obj;
  });
}

}  // namespace

void RegisterAutograd(Napi::Env env, Napi::Object exports) {
  exports.Set("isGradEnabled", Napi::Function::New(env, IsGradEnabled));
  exports.Set("setGradEnabled", Napi::Function::New(env, SetGradEnabled));
  exports.Set("isInferenceModeEnabled", Napi::Function::New(env, IsInferenceModeEnabled));
  exports.Set("pushInferenceMode", Napi::Function::New(env, PushInferenceMode));
  exports.Set("popInferenceMode", Napi::Function::New(env, PopInferenceMode));
  exports.Set("isAutocastEnabled", Napi::Function::New(env, IsAutocastEnabled));
  exports.Set("setAutocastEnabled", Napi::Function::New(env, SetAutocastEnabled));
  exports.Set("getAutocastDtype", Napi::Function::New(env, GetAutocastDtype));
  exports.Set("setAutocastDtype", Napi::Function::New(env, SetAutocastDtype));
  exports.Set("autocastClearCache", Napi::Function::New(env, AutocastClearCache));
  exports.Set("autocastIncrementNesting", Napi::Function::New(env, AutocastIncrementNesting));
  exports.Set("autocastDecrementNesting", Napi::Function::New(env, AutocastDecrementNesting));
  exports.Set("autogradBackward", Napi::Function::New(env, AutogradBackward));
  exports.Set("autogradGrad", Napi::Function::New(env, AutogradGrad));
  exports.Set("applyCustomAutograd", Napi::Function::New(env, ApplyCustomAutograd));
  exports.Set("enterDualLevel", Napi::Function::New(env, EnterDualLevel));
  exports.Set("exitDualLevel", Napi::Function::New(env, ExitDualLevel));
  exports.Set("makeDual", Napi::Function::New(env, MakeDual));
  exports.Set("unpackDual", Napi::Function::New(env, UnpackDual));
}

}  // namespace norch
