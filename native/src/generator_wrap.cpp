#include "norch/generator_wrap.hpp"

#include <ATen/CPUGeneratorImpl.h>
#ifdef NORCH_HAS_CUDA
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#include <cstdint>
#include <utility>

#include "norch/addon_state.hpp"
#include "norch/convert.hpp"
#include "norch/tensor_wrap.hpp"

namespace norch {

Napi::Function GeneratorWrap::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func = DefineClass(
      env,
      "NativeGenerator",
      {
          InstanceMethod("manualSeed", &GeneratorWrap::ManualSeed),
          InstanceMethod("seed", &GeneratorWrap::Seed),
          InstanceMethod("initialSeed", &GeneratorWrap::InitialSeed),
          InstanceMethod("device", &GeneratorWrap::Device),
          InstanceMethod("getState", &GeneratorWrap::GetState),
          InstanceMethod("setState", &GeneratorWrap::SetState),
      });
  AddonState::Get(env).generator_ctor = Napi::Persistent(func);
  exports.Set("NativeGenerator", func);
  return func;
}

Napi::Object GeneratorWrap::NewInstance(Napi::Env env, at::Generator gen) {
  auto& state = AddonState::Get(env);
  auto obj = state.generator_ctor.New({});
  auto* wrap = Napi::ObjectWrap<GeneratorWrap>::Unwrap(obj);
  wrap->SetGenerator(std::move(gen));
  return obj;
}

GeneratorWrap* GeneratorWrap::TryUnwrap(Napi::Env env, const Napi::Value& val) {
  if (!val.IsObject()) {
    return nullptr;
  }
  const auto obj = val.As<Napi::Object>();
  auto& state = AddonState::Get(env);
  if (obj.InstanceOf(state.generator_ctor.Value())) {
    return Napi::ObjectWrap<GeneratorWrap>::Unwrap(obj);
  }
  if (obj.Has("_native")) {
    const auto inner = obj.Get("_native");
    if (inner.IsObject() && inner.As<Napi::Object>().InstanceOf(state.generator_ctor.Value())) {
      return Napi::ObjectWrap<GeneratorWrap>::Unwrap(inner.As<Napi::Object>());
    }
  }
  return nullptr;
}

bool GeneratorWrap::IsGeneratorWrap(Napi::Env env, const Napi::Value& val) {
  return TryUnwrap(env, val) != nullptr;
}

at::Generator GeneratorWrap::UnwrapGenerator(Napi::Env env, const Napi::Value& val) {
  auto* wrap = TryUnwrap(env, val);
  if (wrap == nullptr) {
    throw Napi::TypeError::New(env, "Expected a Generator");
  }
  return wrap->generator();
}

GeneratorWrap::GeneratorWrap(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<GeneratorWrap>(info) {
  const Napi::Env env = info.Env();
  try {
    c10::Device dev(c10::kCPU);
    if (info.Length() >= 1 && !info[0].IsUndefined() && !info[0].IsNull()) {
      dev = ParseDevice(env, info[0]);
    }
    if (dev.is_cpu()) {
      gen_ = at::detail::createCPUGenerator();
    } else if (dev.is_cuda()) {
#ifdef NORCH_HAS_CUDA
      gen_ = at::cuda::detail::createCUDAGenerator(dev.index());
#else
      throw Napi::Error::New(env, "CUDA Generator unavailable: built against CPU-only libtorch");
#endif
    } else {
      throw Napi::RangeError::New(env, "Unsupported Generator device: " + dev.str());
    }
  } catch (const Napi::Error& err) {
    err.ThrowAsJavaScriptException();
  } catch (const c10::Error& err) {
    Napi::Error::New(env, err.what_without_backtrace()).ThrowAsJavaScriptException();
  } catch (const std::exception& ex) {
    Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
  }
}

Napi::Value GeneratorWrap::ManualSeed(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    if (info.Length() < 1) {
      throw Napi::TypeError::New(env, "manualSeed(seed) expects a number or BigInt");
    }
    std::uint64_t seed = 0;
    if (info[0].IsBigInt()) {
      bool lossless = false;
      seed = info[0].As<Napi::BigInt>().Uint64Value(&lossless);
    } else if (info[0].IsNumber()) {
      seed = static_cast<std::uint64_t>(info[0].As<Napi::Number>().Int64Value());
    } else {
      throw Napi::TypeError::New(env, "manualSeed(seed) expects a number or BigInt");
    }
    gen_.set_current_seed(seed);
    return info.This();
  });
}

Napi::Value GeneratorWrap::Seed(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return Napi::BigInt::New(env, gen_.seed());
  });
}

Napi::Value GeneratorWrap::InitialSeed(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return Napi::BigInt::New(env, gen_.current_seed());
  });
}

Napi::Value GeneratorWrap::Device(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return Napi::String::New(env, DeviceToString(gen_.device()));
  });
}

Napi::Value GeneratorWrap::GetState(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return TensorWrap::NewInstance(env, gen_.get_state());
  });
}

Napi::Value GeneratorWrap::SetState(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    if (info.Length() < 1) {
      throw Napi::TypeError::New(env, "setState(tensor) expects a ByteTensor state");
    }
    gen_.set_state(TensorWrap::UnwrapTensor(env, info[0]));
    return env.Undefined();
  });
}

}  // namespace norch
