#include "norch/export_runner.hpp"

#include <ATen/ATen.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/script.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "norch/addon_state.hpp"
#include "norch/convert.hpp"
#include "norch/tensor_wrap.hpp"

namespace norch {

namespace {

template <typename LoaderT>
[[nodiscard]] std::unique_ptr<LoaderT> MakeAotiLoader(
    const std::string& pkg_path,
    const std::string& model_name,
    c10::DeviceIndex device_index) {
  if constexpr (std::is_constructible_v<
                    LoaderT,
                    const std::string&,
                    const std::string&,
                    bool,
                    std::size_t,
                    c10::DeviceIndex>) {
    return std::make_unique<LoaderT>(
        pkg_path, model_name, /*run_single_threaded=*/false, std::size_t{1}, device_index);
  } else if constexpr (std::is_constructible_v<
                           LoaderT,
                           const std::string&,
                           const std::string&,
                           bool,
                           std::size_t>) {
    (void)device_index;
    return std::make_unique<LoaderT>(
        pkg_path, model_name, /*run_single_threaded=*/false, std::size_t{1});
  } else {
    (void)device_index;
    return std::make_unique<LoaderT>(pkg_path, model_name, /*run_single_threaded=*/false);
  }
}

class AotiModelWrap : public Napi::ObjectWrap<AotiModelWrap> {
 public:
  static void Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(
        env,
        "NativeAotiModel",
        {
            InstanceMethod("run", &AotiModelWrap::Run),
            InstanceMethod("getMetadata", &AotiModelWrap::GetMetadata),
            InstanceMethod("getCallSpec", &AotiModelWrap::GetCallSpec),
            InstanceMethod("getConstantFqns", &AotiModelWrap::GetConstantFqns),
            InstanceMethod("dispose", &AotiModelWrap::Dispose),
            InstanceMethod("isDisposed", &AotiModelWrap::IsDisposed),
        });
    AddonState::Get(env).aoti_model_ctor = Napi::Persistent(func);
    exports.Set("NativeAotiModel", func);
  }

  explicit AotiModelWrap(const Napi::CallbackInfo& info) : Napi::ObjectWrap<AotiModelWrap>(info) {
    const Napi::Env env = info.Env();
    try {
      if (info.Length() < 1 || !info[0].IsString()) {
        throw Napi::TypeError::New(env, "NativeAotiModel(packagePath, modelName?, deviceIndex?) expects path");
      }
      const std::string pkg_path = info[0].As<Napi::String>().Utf8Value();
      const std::string model_name =
          (info.Length() >= 2 && info[1].IsString()) ? info[1].As<Napi::String>().Utf8Value() : "model";
      const auto device_index =
          (info.Length() >= 3 && info[2].IsNumber())
              ? static_cast<c10::DeviceIndex>(info[2].As<Napi::Number>().Int32Value())
              : static_cast<c10::DeviceIndex>(-1);
      loader_ = MakeAotiLoader<torch::inductor::AOTIModelPackageLoader>(
          pkg_path, model_name, device_index);
    } catch (const Napi::Error& err) {
      err.ThrowAsJavaScriptException();
    } catch (const c10::Error& err) {
      Napi::Error::New(env, err.what_without_backtrace()).ThrowAsJavaScriptException();
    } catch (const std::exception& ex) {
      Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
  }

  ~AotiModelWrap() override = default;
  AotiModelWrap(const AotiModelWrap&) = delete;
  AotiModelWrap& operator=(const AotiModelWrap&) = delete;
  AotiModelWrap(AotiModelWrap&&) = delete;
  AotiModelWrap& operator=(AotiModelWrap&&) = delete;

 private:
  void EnsureAlive(Napi::Env env) const {
    if (!loader_) {
      throw Napi::Error::New(env, "AOTI model has been disposed");
    }
  }

  Napi::Value Run(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      EnsureAlive(env);
      if (info.Length() < 1 || !info[0].IsArray()) {
        throw Napi::TypeError::New(env, "run(inputs) expects an Array of Tensors");
      }
      const auto inputs = ParseTensorList(env, info[0]);
      auto outputs = loader_->run(inputs);
      auto arr = Napi::Array::New(env, outputs.size());
      for (std::size_t i = 0; i < outputs.size(); ++i) {
        arr.Set(static_cast<std::uint32_t>(i), TensorWrap::NewInstance(env, std::move(outputs[i])));
      }
      return arr;
    });
  }

  Napi::Value GetMetadata(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      EnsureAlive(env);
      const auto meta = loader_->get_metadata();
      auto obj = Napi::Object::New(env);
      for (const auto& [k, v] : meta) {
        obj.Set(k, Napi::String::New(env, v));
      }
      return obj;
    });
  }

  Napi::Value GetCallSpec(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      EnsureAlive(env);
      const auto spec = loader_->get_call_spec();
      auto arr = Napi::Array::New(env, spec.size());
      for (std::size_t i = 0; i < spec.size(); ++i) {
        arr.Set(static_cast<std::uint32_t>(i), Napi::String::New(env, spec[i]));
      }
      return arr;
    });
  }

  Napi::Value GetConstantFqns(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      EnsureAlive(env);
      const auto fqns = loader_->get_constant_fqns();
      auto arr = Napi::Array::New(env, fqns.size());
      for (std::size_t i = 0; i < fqns.size(); ++i) {
        arr.Set(static_cast<std::uint32_t>(i), Napi::String::New(env, fqns[i]));
      }
      return arr;
    });
  }

  Napi::Value Dispose(const Napi::CallbackInfo& info) {
    loader_.reset();
    return info.Env().Undefined();
  }

  Napi::Value IsDisposed(const Napi::CallbackInfo& info) {
    return Napi::Boolean::New(info.Env(), !loader_);
  }

  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader_;
};

class JitModuleWrap : public Napi::ObjectWrap<JitModuleWrap> {
 public:
  static void Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(
        env,
        "NativeJitModule",
        {
            InstanceMethod("forward", &JitModuleWrap::Forward),
            InstanceMethod("runMethod", &JitModuleWrap::RunMethod),
            InstanceMethod("to", &JitModuleWrap::To),
            InstanceMethod("eval", &JitModuleWrap::Eval),
            InstanceMethod("train", &JitModuleWrap::Train),
            InstanceMethod("dispose", &JitModuleWrap::Dispose),
            InstanceMethod("isDisposed", &JitModuleWrap::IsDisposed),
        });
    AddonState::Get(env).jit_module_ctor = Napi::Persistent(func);
    exports.Set("NativeJitModule", func);
  }

  explicit JitModuleWrap(const Napi::CallbackInfo& info) : Napi::ObjectWrap<JitModuleWrap>(info) {
    const Napi::Env env = info.Env();
    try {
      if (info.Length() < 1 || !info[0].IsString()) {
        throw Napi::TypeError::New(env, "NativeJitModule(path, device?) expects a file path");
      }
      const std::string path = info[0].As<Napi::String>().Utf8Value();
      std::optional<c10::Device> dev = std::nullopt;
      if (info.Length() >= 2 && !info[1].IsUndefined() && !info[1].IsNull()) {
        dev = ParseDevice(env, info[1]);
      }
      module_ = std::make_unique<torch::jit::script::Module>(torch::jit::load(path, dev));
    } catch (const Napi::Error& err) {
      err.ThrowAsJavaScriptException();
    } catch (const c10::Error& err) {
      Napi::Error::New(env, err.what_without_backtrace()).ThrowAsJavaScriptException();
    } catch (const std::exception& ex) {
      Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
  }

  ~JitModuleWrap() override = default;
  JitModuleWrap(const JitModuleWrap&) = delete;
  JitModuleWrap& operator=(const JitModuleWrap&) = delete;
  JitModuleWrap(JitModuleWrap&&) = delete;
  JitModuleWrap& operator=(JitModuleWrap&&) = delete;

 private:
  void EnsureAlive(Napi::Env env) const {
    if (!module_) {
      throw Napi::Error::New(env, "TorchScript module has been disposed");
    }
  }

  [[nodiscard]] static std::vector<torch::jit::IValue> JsArgsToIValues(
      Napi::Env env,
      const Napi::Array& arr) {
    std::vector<torch::jit::IValue> out;
    out.reserve(arr.Length());
    for (std::uint32_t i = 0; i < arr.Length(); ++i) {
      const auto item = arr.Get(i);
      if (TensorWrap::IsTensorWrap(env, item)) {
        out.emplace_back(TensorWrap::UnwrapTensor(env, item));
      } else if (item.IsBoolean()) {
        out.emplace_back(item.As<Napi::Boolean>().Value());
      } else if (item.IsNumber()) {
        out.emplace_back(ParseScalar(env, item));
      } else if (item.IsString()) {
        out.emplace_back(item.As<Napi::String>().Utf8Value());
      } else if (item.IsNull() || item.IsUndefined()) {
        out.emplace_back();
      } else {
        throw Napi::TypeError::New(env, "Unsupported TorchScript input argument type");
      }
    }
    return out;
  }

  Napi::Value Forward(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      EnsureAlive(env);
      if (info.Length() < 1 || !info[0].IsArray()) {
        throw Napi::TypeError::New(env, "forward(inputs) expects an Array");
      }
      const auto inputs = JsArgsToIValues(env, info[0].As<Napi::Array>());
      return IValueToJs(env, module_->forward(inputs));
    });
  }

  Napi::Value RunMethod(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      EnsureAlive(env);
      if (info.Length() < 2 || !info[0].IsString() || !info[1].IsArray()) {
        throw Napi::TypeError::New(env, "runMethod(name, inputs) expects (string, Array)");
      }
      const std::string name = info[0].As<Napi::String>().Utf8Value();
      const auto inputs = JsArgsToIValues(env, info[1].As<Napi::Array>());
      return IValueToJs(env, module_->get_method(name)(inputs));
    });
  }

  Napi::Value To(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      EnsureAlive(env);
      if (info.Length() < 1) {
        throw Napi::TypeError::New(env, "to(deviceOrDtype) expects an argument");
      }
      const auto val = info[0];
      if (val.IsString()) {
        const std::string s = val.As<Napi::String>().Utf8Value();
        if (s == "cpu" || s == "cuda" || s.starts_with("cuda:") || s == "mps" || s == "meta") {
          module_->to(c10::Device(s));
        } else {
          module_->to(ParseScalarType(env, val));
        }
      } else {
        module_->to(ParseDevice(env, val));
      }
      return info.This();
    });
  }

  Napi::Value Eval(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      EnsureAlive(env);
      module_->eval();
      return info.This();
    });
  }

  Napi::Value Train(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      EnsureAlive(env);
      const bool mode = info.Length() < 1 || info[0].ToBoolean().Value();
      module_->train(mode);
      return info.This();
    });
  }

  Napi::Value Dispose(const Napi::CallbackInfo& info) {
    module_.reset();
    return info.Env().Undefined();
  }

  Napi::Value IsDisposed(const Napi::CallbackInfo& info) {
    return Napi::Boolean::New(info.Env(), !module_);
  }

  std::unique_ptr<torch::jit::script::Module> module_;
};

}  // namespace

void RegisterExportRunners(Napi::Env env, Napi::Object exports) {
  AotiModelWrap::Init(env, exports);
  JitModuleWrap::Init(env, exports);
}

}  // namespace norch
