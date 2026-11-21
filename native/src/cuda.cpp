#include "norch/cuda.hpp"

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <torch/version.h>

#ifdef NORCH_HAS_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include <cstdint>
#include <string>

#include "norch/addon_state.hpp"
#include "norch/convert.hpp"

namespace norch {

namespace {

Napi::Value CudaIsAvailable(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return Napi::Boolean::New(env, at::cuda::is_available());
  });
}

Napi::Value CudaDeviceCount(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return Napi::Number::New(env, static_cast<double>(at::cuda::device_count()));
  });
}

Napi::Value CudaCurrentDevice(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
#ifdef NORCH_HAS_CUDA
    if (!at::cuda::is_available()) {
      throw Napi::Error::New(env, "CUDA is not available");
    }
    return Napi::Number::New(env, static_cast<double>(c10::cuda::current_device()));
#else
    throw Napi::Error::New(env, "norch was built against CPU-only libtorch");
#endif
  });
}

Napi::Value CudaSetDevice(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
#ifdef NORCH_HAS_CUDA
    if (info.Length() < 1 || !info[0].IsNumber()) {
      throw Napi::TypeError::New(env, "cudaSetDevice(index) expects a device index");
    }
    const auto idx = static_cast<c10::DeviceIndex>(info[0].As<Napi::Number>().Int32Value());
    c10::cuda::set_device(idx);
    return env.Undefined();
#else
    throw Napi::Error::New(env, "norch was built against CPU-only libtorch");
#endif
  });
}

Napi::Value CudaSynchronize(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
#ifdef NORCH_HAS_CUDA
    if (!at::cuda::is_available()) {
      return env.Undefined();
    }
    if (info.Length() >= 1 && info[0].IsNumber()) {
      const auto idx = static_cast<c10::DeviceIndex>(info[0].As<Napi::Number>().Int32Value());
      c10::cuda::device_synchronize();
      c10::cuda::CUDAGuard guard(idx);
      c10::cuda::device_synchronize();
    } else {
      c10::cuda::device_synchronize();
    }
    return env.Undefined();
#else
    return env.Undefined();
#endif
  });
}

Napi::Value CudaEmptyCache(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
#ifdef NORCH_HAS_CUDA
    if (at::cuda::is_available()) {
      c10::cuda::CUDACachingAllocator::emptyCache();
    }
#endif
    return env.Undefined();
  });
}

Napi::Value CudaMemoryStats(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
#ifdef NORCH_HAS_CUDA
    if (!at::cuda::is_available()) {
      throw Napi::Error::New(env, "CUDA is not available");
    }
    const auto idx = (info.Length() >= 1 && info[0].IsNumber())
                         ? static_cast<c10::DeviceIndex>(info[0].As<Napi::Number>().Int32Value())
                         : c10::cuda::current_device();
    const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(idx);
    auto obj = Napi::Object::New(env);
    const auto idx_all = static_cast<std::size_t>(c10::CachingDeviceAllocator::StatType::AGGREGATE);
    obj.Set("allocated", Napi::Number::New(env, static_cast<double>(stats.allocated_bytes[idx_all].current)));
    obj.Set("maxAllocated", Napi::Number::New(env, static_cast<double>(stats.allocated_bytes[idx_all].peak)));
    obj.Set("reserved", Napi::Number::New(env, static_cast<double>(stats.reserved_bytes[idx_all].current)));
    obj.Set("maxReserved", Napi::Number::New(env, static_cast<double>(stats.reserved_bytes[idx_all].peak)));
    return obj;
#else
    throw Napi::Error::New(env, "norch was built against CPU-only libtorch");
#endif
  });
}

Napi::Value CudaDeviceName(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
#ifdef NORCH_HAS_CUDA
    if (!at::cuda::is_available()) {
      throw Napi::Error::New(env, "CUDA is not available");
    }
    const auto idx = (info.Length() >= 1 && info[0].IsNumber())
                         ? static_cast<c10::DeviceIndex>(info[0].As<Napi::Number>().Int32Value())
                         : c10::cuda::current_device();
    const auto* prop = at::cuda::getDeviceProperties(idx);
    return Napi::String::New(env, std::string(&prop->name[0]));
#else
    throw Napi::Error::New(env, "norch was built against CPU-only libtorch");
#endif
  });
}

Napi::Value ManualSeed(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    if (info.Length() < 1) {
      throw Napi::TypeError::New(env, "manualSeed(seed) expects an integer seed");
    }
    std::uint64_t seed = 0;
    if (info[0].IsBigInt()) {
      bool lossless = false;
      seed = info[0].As<Napi::BigInt>().Uint64Value(&lossless);
    } else if (info[0].IsNumber()) {
      seed = static_cast<std::uint64_t>(info[0].As<Napi::Number>().Int64Value());
    } else {
      throw Napi::TypeError::New(env, "manualSeed(seed) expects an integer or BigInt");
    }
    at::manual_seed(seed);
    return env.Undefined();
  });
}

Napi::Value GetNumThreads(const Napi::CallbackInfo& info) {
  return Napi::Number::New(info.Env(), static_cast<double>(at::get_num_threads()));
}

Napi::Value SetNumThreads(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    if (info.Length() < 1 || !info[0].IsNumber()) {
      throw Napi::TypeError::New(env, "setNumThreads(n) expects an integer");
    }
    at::set_num_threads(info[0].As<Napi::Number>().Int32Value());
    return env.Undefined();
  });
}

Napi::Value MemoryStats(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  const auto& state = AddonState::Get(env);
  auto obj = Napi::Object::New(env);
  obj.Set("liveTensors", Napi::Number::New(env, static_cast<double>(state.live_tensors)));
  obj.Set(
      "trackedStorages",
      Napi::Number::New(env, static_cast<double>(state.tracked_storages.size())));
  obj.Set(
      "reportedExternalBytes",
      Napi::Number::New(env, static_cast<double>(state.total_reported_bytes)));
  return obj;
}

Napi::Value LibtorchVersion(const Napi::CallbackInfo& info) {
  return Napi::String::New(info.Env(), TORCH_VERSION);
}

}  // namespace

void RegisterCudaAndRuntime(Napi::Env env, Napi::Object exports) {
  exports.Set("cudaIsAvailable", Napi::Function::New(env, CudaIsAvailable));
  exports.Set("cudaDeviceCount", Napi::Function::New(env, CudaDeviceCount));
  exports.Set("cudaCurrentDevice", Napi::Function::New(env, CudaCurrentDevice));
  exports.Set("cudaSetDevice", Napi::Function::New(env, CudaSetDevice));
  exports.Set("cudaSynchronize", Napi::Function::New(env, CudaSynchronize));
  exports.Set("cudaEmptyCache", Napi::Function::New(env, CudaEmptyCache));
  exports.Set("cudaMemoryStats", Napi::Function::New(env, CudaMemoryStats));
  exports.Set("cudaDeviceName", Napi::Function::New(env, CudaDeviceName));
  exports.Set("manualSeed", Napi::Function::New(env, ManualSeed));
  exports.Set("getNumThreads", Napi::Function::New(env, GetNumThreads));
  exports.Set("setNumThreads", Napi::Function::New(env, SetNumThreads));
  exports.Set("memoryStats", Napi::Function::New(env, MemoryStats));
  exports.Set("libtorchVersion", Napi::Function::New(env, LibtorchVersion));
}

}  // namespace norch
