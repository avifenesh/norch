#include "norch/profiler.hpp"

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <torch/csrc/autograd/profiler_kineto.h>

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "norch/addon_state.hpp"
#include "norch/convert.hpp"

namespace norch {

namespace {

namespace prof = torch::autograd::profiler;

class ProfilerResultWrap : public Napi::ObjectWrap<ProfilerResultWrap> {
 public:
  static void Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(
        env,
        "NativeProfilerResult",
        {
            InstanceMethod("events", &ProfilerResultWrap::Events),
            InstanceMethod("traceStartNs", &ProfilerResultWrap::TraceStartNs),
            InstanceMethod("saveChromeTrace", &ProfilerResultWrap::SaveChromeTrace),
            InstanceMethod("dispose", &ProfilerResultWrap::Dispose),
            InstanceMethod("isDisposed", &ProfilerResultWrap::IsDisposed),
        });
    AddonState::Get(env).profiler_result_ctor = Napi::Persistent(func);
    exports.Set("NativeProfilerResult", func);
  }

  static Napi::Object NewInstance(Napi::Env env, std::unique_ptr<prof::ProfilerResult> result) {
    auto obj = AddonState::Get(env).profiler_result_ctor.New({});
    auto* wrap = Napi::ObjectWrap<ProfilerResultWrap>::Unwrap(obj);
    wrap->result_ = std::move(result);
    return obj;
  }

  explicit ProfilerResultWrap(const Napi::CallbackInfo& info)
      : Napi::ObjectWrap<ProfilerResultWrap>(info) {}

  ~ProfilerResultWrap() override = default;
  ProfilerResultWrap(const ProfilerResultWrap&) = delete;
  ProfilerResultWrap& operator=(const ProfilerResultWrap&) = delete;
  ProfilerResultWrap(ProfilerResultWrap&&) = delete;
  ProfilerResultWrap& operator=(ProfilerResultWrap&&) = delete;

 private:
  void EnsureAlive(Napi::Env env) const {
    if (!result_) {
      throw Napi::Error::New(env, "ProfilerResult has been disposed");
    }
  }

  Napi::Value TraceStartNs(const Napi::CallbackInfo& info) {
    return GuardNapi(info.Env(), [&]() -> Napi::Value {
      EnsureAlive(info.Env());
      return Napi::Number::New(info.Env(), static_cast<double>(result_->trace_start_ns()));
    });
  }

  Napi::Value Events(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      EnsureAlive(env);
      const auto& events = result_->events();
      auto out = Napi::Array::New(env, events.size());
      for (std::size_t i = 0; i < events.size(); ++i) {
        const auto& ev = events[i];
        auto obj = Napi::Object::New(env);
        obj.Set("name", Napi::String::New(env, ev.name()));
        obj.Set("overloadName", Napi::String::New(env, ev.overload_name()));
        obj.Set("startNs", Napi::Number::New(env, static_cast<double>(ev.startNs())));
        obj.Set("endNs", Napi::Number::New(env, static_cast<double>(ev.endNs())));
        obj.Set("durationUs", Napi::Number::New(env, static_cast<double>(ev.durationNs()) / 1000.0));
        obj.Set("cudaTimeUs", Napi::Number::New(env, static_cast<double>(ev.cudaElapsedUs())));
        obj.Set("deviceType", Napi::String::New(env, c10::DeviceTypeName(ev.deviceType(), /*lower_case=*/true)));
        obj.Set("deviceIndex", Napi::Number::New(env, static_cast<double>(ev.deviceIndex())));
        obj.Set("nBytes", Napi::Number::New(env, static_cast<double>(ev.nBytes())));
        obj.Set("flops", Napi::Number::New(env, static_cast<double>(ev.flops())));
        obj.Set("threadId", Napi::Number::New(env, static_cast<double>(ev.startThreadId())));
        obj.Set("isAsync", Napi::Boolean::New(env, ev.isAsync()));

        if (ev.hasShapes()) {
          const auto shapes = ev.shapes();
          auto shapes_arr = Napi::Array::New(env, shapes.size());
          for (std::size_t s = 0; s < shapes.size(); ++s) {
            auto dims = Napi::Array::New(env, shapes[s].size());
            for (std::size_t d = 0; d < shapes[s].size(); ++d) {
              dims.Set(
                  static_cast<std::uint32_t>(d),
                  Napi::Number::New(env, static_cast<double>(shapes[s][d])));
            }
            shapes_arr.Set(static_cast<std::uint32_t>(s), dims);
          }
          obj.Set("shapes", shapes_arr);
        }
        out.Set(static_cast<std::uint32_t>(i), obj);
      }
      return out;
    });
  }

  Napi::Value SaveChromeTrace(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      EnsureAlive(env);
      if (info.Length() < 1 || !info[0].IsString()) {
        throw Napi::TypeError::New(env, "saveChromeTrace(path) expects a file path string");
      }
      result_->save(info[0].As<Napi::String>().Utf8Value());
      return env.Undefined();
    });
  }

  Napi::Value Dispose(const Napi::CallbackInfo& info) {
    result_.reset();
    return info.Env().Undefined();
  }

  Napi::Value IsDisposed(const Napi::CallbackInfo& info) {
    return Napi::Boolean::New(info.Env(), !result_);
  }

  std::unique_ptr<prof::ProfilerResult> result_;
};

Napi::Value ProfilerStart(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    bool record_shapes = false;
    bool profile_memory = false;
    bool with_flops = false;
    std::set<torch::profiler::impl::ActivityType> activities = {
        torch::profiler::impl::ActivityType::CPU,
    };

    if (info.Length() >= 1 && info[0].IsObject()) {
      const auto opts = info[0].As<Napi::Object>();
      if (opts.Has("recordShapes") && opts.Get("recordShapes").IsBoolean()) {
        record_shapes = opts.Get("recordShapes").As<Napi::Boolean>().Value();
      }
      if (opts.Has("profileMemory") && opts.Get("profileMemory").IsBoolean()) {
        profile_memory = opts.Get("profileMemory").As<Napi::Boolean>().Value();
      }
      if (opts.Has("withFlops") && opts.Get("withFlops").IsBoolean()) {
        with_flops = opts.Get("withFlops").As<Napi::Boolean>().Value();
      }
      if (opts.Has("activities") && opts.Get("activities").IsArray()) {
        activities.clear();
        const auto arr = opts.Get("activities").As<Napi::Array>();
        for (std::uint32_t i = 0; i < arr.Length(); ++i) {
          const std::string a = arr.Get(i).As<Napi::String>().Utf8Value();
          if (a == "cpu" || a == "CPU") {
            activities.insert(torch::profiler::impl::ActivityType::CPU);
          } else if ((a == "cuda" || a == "CUDA") && at::hasCUDA()) {
            activities.insert(torch::profiler::impl::ActivityType::CUDA);
          }
        }
        if (activities.empty()) {
          activities.insert(torch::profiler::impl::ActivityType::CPU);
        }
      }
    }

    const torch::profiler::impl::ProfilerConfig cfg(
        torch::profiler::impl::ProfilerState::KINETO,
        record_shapes,
        profile_memory,
        /*with_stack=*/false,
        with_flops);
    prof::prepareProfiler(cfg, activities);
    prof::enableProfiler(cfg, activities);
    return env.Undefined();
  });
}

Napi::Value ProfilerStop(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    auto res = prof::disableProfiler();
    return ProfilerResultWrap::NewInstance(env, std::move(res));
  });
}

Napi::Value ProfilerIsEnabled(const Napi::CallbackInfo& info) {
  return Napi::Boolean::New(info.Env(), prof::profilerEnabled());
}

}  // namespace

void RegisterProfiler(Napi::Env env, Napi::Object exports) {
  ProfilerResultWrap::Init(env, exports);
  exports.Set("profilerStart", Napi::Function::New(env, ProfilerStart));
  exports.Set("profilerStop", Napi::Function::New(env, ProfilerStop));
  exports.Set("profilerIsEnabled", Napi::Function::New(env, ProfilerIsEnabled));
}

}  // namespace norch
