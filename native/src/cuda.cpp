#include "norch/cuda.hpp"

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <torch/version.h>

#ifdef NORCH_HAS_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "norch/addon_state.hpp"
#include "norch/convert.hpp"

namespace norch {

namespace {

[[nodiscard]] std::uint64_t ParseSeedValue(Napi::Env env, const Napi::Value& val) {
  if (val.IsBigInt()) {
    bool lossless = false;
    return val.As<Napi::BigInt>().Uint64Value(&lossless);
  }
  if (val.IsNumber()) {
    return static_cast<std::uint64_t>(val.As<Napi::Number>().Int64Value());
  }
  throw Napi::TypeError::New(env, "Expected an integer or BigInt seed");
}

#ifdef NORCH_HAS_CUDA

class CudaEventWrap;

class CudaStreamWrap : public Napi::ObjectWrap<CudaStreamWrap> {
 public:
  static void Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(
        env,
        "NativeCudaStream",
        {
            InstanceMethod("synchronize", &CudaStreamWrap::Synchronize),
            InstanceMethod("query", &CudaStreamWrap::Query),
            InstanceMethod("deviceIndex", &CudaStreamWrap::DeviceIndex),
            InstanceMethod("streamId", &CudaStreamWrap::StreamId),
            InstanceMethod("priority", &CudaStreamWrap::Priority),
            InstanceMethod("isCapturing", &CudaStreamWrap::IsCapturing),
            InstanceMethod("waitEvent", &CudaStreamWrap::WaitEvent),
            InstanceMethod("waitStream", &CudaStreamWrap::WaitStream),
        });
    AddonState::Get(env).cuda_stream_ctor = Napi::Persistent(func);
    exports.Set("NativeCudaStream", func);
  }

  static Napi::Object FromStream(Napi::Env env, c10::cuda::CUDAStream s) {
    auto& state = AddonState::Get(env);
    auto obj = state.cuda_stream_ctor.New({Napi::Boolean::New(env, true)});
    auto* wrap = Napi::ObjectWrap<CudaStreamWrap>::Unwrap(obj);
    wrap->stream_ = std::make_unique<c10::cuda::CUDAStream>(s);
    return obj;
  }

  [[nodiscard]] static const c10::cuda::CUDAStream& UnwrapStream(Napi::Env env, const Napi::Value& val) {
    if (!val.IsObject()) {
      throw Napi::TypeError::New(env, "Expected a CudaStream");
    }
    auto obj = val.As<Napi::Object>();
    auto& state = AddonState::Get(env);
    if (!obj.InstanceOf(state.cuda_stream_ctor.Value()) && obj.Has("_native")) {
      obj = obj.Get("_native").As<Napi::Object>();
    }
    if (!obj.InstanceOf(state.cuda_stream_ctor.Value())) {
      throw Napi::TypeError::New(env, "Expected a CudaStream");
    }
    auto* wrap = Napi::ObjectWrap<CudaStreamWrap>::Unwrap(obj);
    if (!wrap->stream_) {
      throw Napi::Error::New(env, "CudaStream is not initialized");
    }
    return *wrap->stream_;
  }

  explicit CudaStreamWrap(const Napi::CallbackInfo& info) : Napi::ObjectWrap<CudaStreamWrap>(info) {
    const Napi::Env env = info.Env();
    // Internal sentinel: FromStream passes (true) to bypass pool allocation
    if (info.Length() == 1 && info[0].IsBoolean() && info[0].As<Napi::Boolean>().Value()) {
      return;
    }
    try {
      if (!at::cuda::is_available()) {
        throw Napi::Error::New(env, "CUDA is not available");
      }
      const c10::DeviceIndex dev =
          (info.Length() >= 1 && info[0].IsNumber())
              ? static_cast<c10::DeviceIndex>(info[0].As<Napi::Number>().Int32Value())
              : static_cast<c10::DeviceIndex>(-1);
      const int priority = (info.Length() >= 2 && info[1].IsNumber())
                               ? info[1].As<Napi::Number>().Int32Value()
                               : 0;
      stream_ = std::make_unique<c10::cuda::CUDAStream>(
          c10::cuda::getStreamFromPool(priority, dev));
    } catch (const Napi::Error& err) {
      err.ThrowAsJavaScriptException();
    } catch (const c10::Error& err) {
      Napi::Error::New(env, err.what_without_backtrace()).ThrowAsJavaScriptException();
    } catch (const std::exception& ex) {
      Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
  }

  ~CudaStreamWrap() override = default;
  CudaStreamWrap(const CudaStreamWrap&) = delete;
  CudaStreamWrap& operator=(const CudaStreamWrap&) = delete;
  CudaStreamWrap(CudaStreamWrap&&) = delete;
  CudaStreamWrap& operator=(CudaStreamWrap&&) = delete;

 private:
  [[nodiscard]] const c10::cuda::CUDAStream& Checked(Napi::Env env) const {
    if (!stream_) {
      throw Napi::Error::New(env, "CudaStream is uninitialized");
    }
    return *stream_;
  }

  Napi::Value Synchronize(const Napi::CallbackInfo& info) {
    return GuardNapi(info.Env(), [&]() -> Napi::Value {
      Checked(info.Env()).synchronize();
      return info.Env().Undefined();
    });
  }

  Napi::Value Query(const Napi::CallbackInfo& info) {
    return GuardNapi(info.Env(), [&]() -> Napi::Value {
      return Napi::Boolean::New(info.Env(), Checked(info.Env()).query());
    });
  }

  Napi::Value DeviceIndex(const Napi::CallbackInfo& info) {
    return GuardNapi(info.Env(), [&]() -> Napi::Value {
      return Napi::Number::New(info.Env(), static_cast<double>(Checked(info.Env()).device_index()));
    });
  }

  Napi::Value StreamId(const Napi::CallbackInfo& info) {
    return GuardNapi(info.Env(), [&]() -> Napi::Value {
      return Napi::BigInt::New(info.Env(), Checked(info.Env()).id());
    });
  }

  Napi::Value Priority(const Napi::CallbackInfo& info) {
    return GuardNapi(info.Env(), [&]() -> Napi::Value {
      return Napi::Number::New(info.Env(), static_cast<double>(Checked(info.Env()).priority()));
    });
  }

  Napi::Value IsCapturing(const Napi::CallbackInfo& info) {
    return GuardNapi(info.Env(), [&]() -> Napi::Value {
      return Napi::Boolean::New(info.Env(), Checked(info.Env()).is_capturing());
    });
  }

  Napi::Value WaitEvent(const Napi::CallbackInfo& info);

  Napi::Value WaitStream(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      const auto& other = UnwrapStream(env, info[0]);
      c10::cuda::CUDAEvent ev(cudaEventDisableTiming);
      ev.record(other);
      ev.block(Checked(env));
      return env.Undefined();
    });
  }

  std::unique_ptr<c10::cuda::CUDAStream> stream_;
};

class CudaEventWrap : public Napi::ObjectWrap<CudaEventWrap> {
 public:
  static void Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(
        env,
        "NativeCudaEvent",
        {
            InstanceMethod("record", &CudaEventWrap::Record),
            InstanceMethod("wait", &CudaEventWrap::Wait),
            InstanceMethod("synchronize", &CudaEventWrap::Synchronize),
            InstanceMethod("query", &CudaEventWrap::Query),
            InstanceMethod("elapsedTime", &CudaEventWrap::ElapsedTime),
        });
    AddonState::Get(env).cuda_event_ctor = Napi::Persistent(func);
    exports.Set("NativeCudaEvent", func);
  }

  [[nodiscard]] static c10::cuda::CUDAEvent& UnwrapEvent(Napi::Env env, const Napi::Value& val) {
    if (!val.IsObject()) {
      throw Napi::TypeError::New(env, "Expected a CudaEvent");
    }
    auto obj = val.As<Napi::Object>();
    auto& state = AddonState::Get(env);
    if (!obj.InstanceOf(state.cuda_event_ctor.Value()) && obj.Has("_native")) {
      obj = obj.Get("_native").As<Napi::Object>();
    }
    if (!obj.InstanceOf(state.cuda_event_ctor.Value())) {
      throw Napi::TypeError::New(env, "Expected a CudaEvent");
    }
    return Napi::ObjectWrap<CudaEventWrap>::Unwrap(obj)->event_;
  }

  explicit CudaEventWrap(const Napi::CallbackInfo& info)
      : Napi::ObjectWrap<CudaEventWrap>(info), event_(ParseFlags(info)) {}

  ~CudaEventWrap() override = default;
  CudaEventWrap(const CudaEventWrap&) = delete;
  CudaEventWrap& operator=(const CudaEventWrap&) = delete;
  CudaEventWrap(CudaEventWrap&&) = delete;
  CudaEventWrap& operator=(CudaEventWrap&&) = delete;

 private:
  [[nodiscard]] static unsigned int ParseFlags(const Napi::CallbackInfo& info) {
    const bool enable_timing = info.Length() >= 1 && info[0].ToBoolean().Value();
    const bool blocking = info.Length() >= 2 && info[1].ToBoolean().Value();
    const bool interprocess = info.Length() >= 3 && info[2].ToBoolean().Value();
    unsigned int flags = enable_timing ? cudaEventDefault : cudaEventDisableTiming;
    if (blocking) flags |= cudaEventBlockingSync;
    if (interprocess) flags |= cudaEventInterprocess;
    return flags;
  }

  Napi::Value Record(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      if (info.Length() >= 1 && !info[0].IsUndefined() && !info[0].IsNull()) {
        event_.record(CudaStreamWrap::UnwrapStream(env, info[0]));
      } else {
        event_.record();
      }
      return env.Undefined();
    });
  }

  Napi::Value Wait(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      if (info.Length() >= 1 && !info[0].IsUndefined() && !info[0].IsNull()) {
        event_.block(CudaStreamWrap::UnwrapStream(env, info[0]));
      } else {
        event_.block(c10::cuda::getCurrentCUDAStream());
      }
      return env.Undefined();
    });
  }

  Napi::Value Synchronize(const Napi::CallbackInfo& info) {
    return GuardNapi(info.Env(), [&]() -> Napi::Value {
      event_.synchronize();
      return info.Env().Undefined();
    });
  }

  Napi::Value Query(const Napi::CallbackInfo& info) {
    return GuardNapi(info.Env(), [&]() -> Napi::Value {
      return Napi::Boolean::New(info.Env(), event_.query());
    });
  }

  Napi::Value ElapsedTime(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      const auto& end_ev = UnwrapEvent(env, info[0]);
      return Napi::Number::New(env, static_cast<double>(event_.elapsed_time(end_ev)));
    });
  }

  c10::cuda::CUDAEvent event_;
};

Napi::Value CudaStreamWrap::WaitEvent(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    auto& ev = CudaEventWrap::UnwrapEvent(env, info[0]);
    ev.block(Checked(env));
    return env.Undefined();
  });
}

class CudaGraphWrap : public Napi::ObjectWrap<CudaGraphWrap> {
 public:
  static void Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(
        env,
        "NativeCudaGraph",
        {
            InstanceMethod("captureBegin", &CudaGraphWrap::CaptureBegin),
            InstanceMethod("captureEnd", &CudaGraphWrap::CaptureEnd),
            InstanceMethod("replay", &CudaGraphWrap::Replay),
            InstanceMethod("reset", &CudaGraphWrap::Reset),
            InstanceMethod("pool", &CudaGraphWrap::Pool),
        });
    AddonState::Get(env).cuda_graph_ctor = Napi::Persistent(func);
    exports.Set("NativeCudaGraph", func);
  }

  explicit CudaGraphWrap(const Napi::CallbackInfo& info)
      : Napi::ObjectWrap<CudaGraphWrap>(info),
        graph_(std::make_unique<at::cuda::CUDAGraph>(
            info.Length() >= 1 && info[0].ToBoolean().Value())) {}

  ~CudaGraphWrap() override = default;
  CudaGraphWrap(const CudaGraphWrap&) = delete;
  CudaGraphWrap& operator=(const CudaGraphWrap&) = delete;
  CudaGraphWrap(CudaGraphWrap&&) = delete;
  CudaGraphWrap& operator=(CudaGraphWrap&&) = delete;

 private:
  Napi::Value CaptureBegin(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      at::MempoolId_t pool = {0, 0};
      if (info.Length() >= 1 && info[0].IsArray()) {
        const auto arr = info[0].As<Napi::Array>();
        if (arr.Length() == 2) {
          bool l0 = false;
          bool l1 = false;
          pool.first = arr.Get(static_cast<std::uint32_t>(0)).IsBigInt()
                           ? arr.Get(static_cast<std::uint32_t>(0)).As<Napi::BigInt>().Uint64Value(&l0)
                           : static_cast<std::uint64_t>(
                                 arr.Get(static_cast<std::uint32_t>(0)).As<Napi::Number>().Int64Value());
          pool.second = arr.Get(static_cast<std::uint32_t>(1)).IsBigInt()
                            ? arr.Get(static_cast<std::uint32_t>(1)).As<Napi::BigInt>().Uint64Value(&l1)
                            : static_cast<std::uint64_t>(
                                  arr.Get(static_cast<std::uint32_t>(1)).As<Napi::Number>().Int64Value());
        }
      }
      graph_->capture_begin(pool);
      return env.Undefined();
    });
  }

  Napi::Value CaptureEnd(const Napi::CallbackInfo& info) {
    return GuardNapi(info.Env(), [&]() -> Napi::Value {
      graph_->capture_end();
      return info.Env().Undefined();
    });
  }

  Napi::Value Replay(const Napi::CallbackInfo& info) {
    return GuardNapi(info.Env(), [&]() -> Napi::Value {
      graph_->replay();
      return info.Env().Undefined();
    });
  }

  Napi::Value Reset(const Napi::CallbackInfo& info) {
    return GuardNapi(info.Env(), [&]() -> Napi::Value {
      graph_->reset();
      return info.Env().Undefined();
    });
  }

  Napi::Value Pool(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      const auto p = graph_->pool();
      auto arr = Napi::Array::New(env, 2);
      arr.Set(static_cast<std::uint32_t>(0), Napi::BigInt::New(env, static_cast<std::uint64_t>(p.first)));
      arr.Set(static_cast<std::uint32_t>(1), Napi::BigInt::New(env, static_cast<std::uint64_t>(p.second)));
      return arr;
    });
  }

  std::unique_ptr<at::cuda::CUDAGraph> graph_;
};

#endif  // NORCH_HAS_CUDA

Napi::Value CudaIsAvailable(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return Napi::Boolean::New(env, at::hasCUDA());
  });
}

Napi::Value CudaDeviceCount(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return Napi::Number::New(env, static_cast<double>(at::getNumGPUs()));
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
      const c10::cuda::CUDAGuard guard(idx);
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
      at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
      c10::cuda::CUDACachingAllocator::emptyCache();
    }
#endif
    return env.Undefined();
  });
}

Napi::Value CudaSetMemoryFraction(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
#ifdef NORCH_HAS_CUDA
    if (!at::cuda::is_available()) {
      throw Napi::Error::New(env, "CUDA is not available");
    }
    at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
    const double fraction = info[0].As<Napi::Number>().DoubleValue();
    const auto idx = (info.Length() >= 2 && info[1].IsNumber())
                         ? static_cast<c10::DeviceIndex>(info[1].As<Napi::Number>().Int32Value())
                         : c10::cuda::current_device();
    c10::cuda::CUDACachingAllocator::setMemoryFraction(fraction, idx);
    return env.Undefined();
#else
    throw Napi::Error::New(env, "norch was built against CPU-only libtorch");
#endif
  });
}

Napi::Value CudaResetPeakMemoryStats(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
#ifdef NORCH_HAS_CUDA
    if (!at::cuda::is_available()) {
      return env.Undefined();
    }
    at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
    const auto idx = (info.Length() >= 1 && info[0].IsNumber())
                         ? static_cast<c10::DeviceIndex>(info[0].As<Napi::Number>().Int32Value())
                         : c10::cuda::current_device();
    c10::cuda::CUDACachingAllocator::resetPeakStats(idx);
    return env.Undefined();
#else
    return env.Undefined();
#endif
  });
}

Napi::Value CudaMemoryStats(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
#ifdef NORCH_HAS_CUDA
    if (!at::cuda::is_available()) {
      throw Napi::Error::New(env, "CUDA is not available");
    }
    at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
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

Napi::Value CudaDeviceProperties(const Napi::CallbackInfo& info) {
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
    auto obj = Napi::Object::New(env);
    obj.Set("name", Napi::String::New(env, std::string(&prop->name[0])));
    obj.Set("major", Napi::Number::New(env, static_cast<double>(prop->major)));
    obj.Set("minor", Napi::Number::New(env, static_cast<double>(prop->minor)));
    obj.Set("totalMemory", Napi::Number::New(env, static_cast<double>(prop->totalGlobalMem)));
    obj.Set("multiProcessorCount", Napi::Number::New(env, static_cast<double>(prop->multiProcessorCount)));
    return obj;
#else
    throw Napi::Error::New(env, "norch was built against CPU-only libtorch");
#endif
  });
}

Napi::Value CudaManualSeed(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
#ifdef NORCH_HAS_CUDA
    const auto seed = ParseSeedValue(env, info[0]);
    if (at::cuda::is_available()) {
      auto gen = at::cuda::detail::getDefaultCUDAGenerator();
      gen.set_current_seed(seed);
    }
    return env.Undefined();
#else
    return env.Undefined();
#endif
  });
}

Napi::Value CudaManualSeedAll(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
#ifdef NORCH_HAS_CUDA
    const auto seed = ParseSeedValue(env, info[0]);
    if (at::cuda::is_available()) {
      const auto count = at::cuda::device_count();
      for (c10::DeviceIndex i = 0; i < count; ++i) {
        auto gen = at::cuda::detail::getDefaultCUDAGenerator(i);
        gen.set_current_seed(seed);
      }
    }
    return env.Undefined();
#else
    return env.Undefined();
#endif
  });
}

Napi::Value CudaCurrentStream(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
#ifdef NORCH_HAS_CUDA
    const c10::DeviceIndex dev =
        (info.Length() >= 1 && info[0].IsNumber())
            ? static_cast<c10::DeviceIndex>(info[0].As<Napi::Number>().Int32Value())
            : static_cast<c10::DeviceIndex>(-1);
    return CudaStreamWrap::FromStream(env, c10::cuda::getCurrentCUDAStream(dev));
#else
    throw Napi::Error::New(env, "norch was built against CPU-only libtorch");
#endif
  });
}

Napi::Value CudaDefaultStream(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
#ifdef NORCH_HAS_CUDA
    const c10::DeviceIndex dev =
        (info.Length() >= 1 && info[0].IsNumber())
            ? static_cast<c10::DeviceIndex>(info[0].As<Napi::Number>().Int32Value())
            : static_cast<c10::DeviceIndex>(-1);
    return CudaStreamWrap::FromStream(env, c10::cuda::getDefaultCUDAStream(dev));
#else
    throw Napi::Error::New(env, "norch was built against CPU-only libtorch");
#endif
  });
}

Napi::Value CudaSetStream(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
#ifdef NORCH_HAS_CUDA
    const auto& s = CudaStreamWrap::UnwrapStream(env, info[0]);
    c10::cuda::setCurrentCUDAStream(s);
    return env.Undefined();
#else
    throw Napi::Error::New(env, "norch was built against CPU-only libtorch");
#endif
  });
}

Napi::Value CudaGraphPoolHandle(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
#ifdef NORCH_HAS_CUDA
    const auto p = at::cuda::graph_pool_handle();
    auto arr = Napi::Array::New(env, 2);
    arr.Set(static_cast<std::uint32_t>(0), Napi::BigInt::New(env, static_cast<std::uint64_t>(p.first)));
    arr.Set(static_cast<std::uint32_t>(1), Napi::BigInt::New(env, static_cast<std::uint64_t>(p.second)));
    return arr;
#else
    throw Napi::Error::New(env, "norch was built against CPU-only libtorch");
#endif
  });
}

Napi::Value MpsIsAvailable(const Napi::CallbackInfo& info) {
  return Napi::Boolean::New(info.Env(), at::hasMPS());
}

Napi::Value ManualSeed(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    if (info.Length() < 1) {
      throw Napi::TypeError::New(env, "manualSeed(seed) expects an integer seed");
    }
    at::manual_seed(ParseSeedValue(env, info[0]));
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
#ifdef NORCH_HAS_CUDA
  CudaStreamWrap::Init(env, exports);
  CudaEventWrap::Init(env, exports);
  CudaGraphWrap::Init(env, exports);
#endif
  exports.Set("cudaIsAvailable", Napi::Function::New(env, CudaIsAvailable));
  exports.Set("cudaDeviceCount", Napi::Function::New(env, CudaDeviceCount));
  exports.Set("cudaCurrentDevice", Napi::Function::New(env, CudaCurrentDevice));
  exports.Set("cudaSetDevice", Napi::Function::New(env, CudaSetDevice));
  exports.Set("cudaSynchronize", Napi::Function::New(env, CudaSynchronize));
  exports.Set("cudaEmptyCache", Napi::Function::New(env, CudaEmptyCache));
  exports.Set("cudaSetMemoryFraction", Napi::Function::New(env, CudaSetMemoryFraction));
  exports.Set("cudaResetPeakMemoryStats", Napi::Function::New(env, CudaResetPeakMemoryStats));
  exports.Set("cudaMemoryStats", Napi::Function::New(env, CudaMemoryStats));
  exports.Set("cudaDeviceName", Napi::Function::New(env, CudaDeviceName));
  exports.Set("cudaDeviceProperties", Napi::Function::New(env, CudaDeviceProperties));
  exports.Set("cudaManualSeed", Napi::Function::New(env, CudaManualSeed));
  exports.Set("cudaManualSeedAll", Napi::Function::New(env, CudaManualSeedAll));
  exports.Set("cudaCurrentStream", Napi::Function::New(env, CudaCurrentStream));
  exports.Set("cudaDefaultStream", Napi::Function::New(env, CudaDefaultStream));
  exports.Set("cudaSetStream", Napi::Function::New(env, CudaSetStream));
  exports.Set("cudaGraphPoolHandle", Napi::Function::New(env, CudaGraphPoolHandle));
  exports.Set("mpsIsAvailable", Napi::Function::New(env, MpsIsAvailable));
  exports.Set("manualSeed", Napi::Function::New(env, ManualSeed));
  exports.Set("getNumThreads", Napi::Function::New(env, GetNumThreads));
  exports.Set("setNumThreads", Napi::Function::New(env, SetNumThreads));
  exports.Set("memoryStats", Napi::Function::New(env, MemoryStats));
  exports.Set("libtorchVersion", Napi::Function::New(env, LibtorchVersion));
}

}  // namespace norch
