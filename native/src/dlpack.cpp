#include "norch/dlpack.hpp"

#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>
#include <ATen/dlpack.h>

#include <cstdint>
#include <memory>

#include "norch/convert.hpp"
#include "norch/tensor_wrap.hpp"

namespace norch {

namespace {

struct DlpackCapsule {
  DLManagedTensor* managed = nullptr;
  bool consumed = false;

  DlpackCapsule() = default;
  ~DlpackCapsule() noexcept {
    if (!consumed && managed != nullptr && managed->deleter != nullptr) {
      managed->deleter(managed);
    }
  }

  DlpackCapsule(const DlpackCapsule&) = delete;
  DlpackCapsule& operator=(const DlpackCapsule&) = delete;
  DlpackCapsule(DlpackCapsule&&) = delete;
  DlpackCapsule& operator=(DlpackCapsule&&) = delete;
};

Napi::Value ToDlpackOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    if (info.Length() < 1) {
      throw Napi::TypeError::New(env, "toDlpack(tensor) expects a Tensor");
    }
    const auto& t = TensorWrap::UnwrapTensor(env, info[0]);
    DLManagedTensor* managed = at::toDLPack(t);
    auto cap = std::make_unique<DlpackCapsule>();
    cap->managed = managed;
    auto* raw_cap = cap.release();
    return Napi::External<DlpackCapsule>::New(
        env, raw_cap, [](Napi::Env /*env*/, DlpackCapsule* ptr) { delete ptr; });
  });
}

Napi::Value FromDlpackOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    if (info.Length() < 1 || !info[0].IsExternal()) {
      throw Napi::TypeError::New(env, "fromDlpack(capsule) expects a DLPack External capsule");
    }
    auto* cap = info[0].As<Napi::External<DlpackCapsule>>().Data();
    if (cap == nullptr) {
      throw Napi::Error::New(env, "Invalid or null DLPack capsule");
    }
    if (cap->consumed) {
      throw Napi::Error::New(env, "DLPack capsule has already been consumed");
    }
    if (cap->managed == nullptr) {
      throw Napi::Error::New(env, "Invalid or null DLPack capsule");
    }
    cap->consumed = true;
    DLManagedTensor* managed = cap->managed;
    cap->managed = nullptr;
    at::Tensor tensor = at::fromDLPack(managed);
    return TensorWrap::NewInstance(env, std::move(tensor));
  });
}

Napi::Value DlpackDeviceOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& t = TensorWrap::UnwrapTensor(env, info[0]);
    const DLDevice dev = at::torchDeviceToDLDevice(t.device());
    auto arr = Napi::Array::New(env, 2);
    arr.Set(0U, Napi::Number::New(env, static_cast<double>(dev.device_type)));
    arr.Set(1U, Napi::Number::New(env, static_cast<double>(dev.device_id)));
    return arr;
  });
}

Napi::Value DlpackPointerOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    if (info.Length() < 1 || !info[0].IsExternal()) {
      throw Napi::TypeError::New(env, "dlpackPointer(capsule) expects a DLPack External capsule");
    }
    const auto* cap = info[0].As<Napi::External<DlpackCapsule>>().Data();
    if (cap == nullptr || cap->consumed || cap->managed == nullptr) {
      return Napi::BigInt::New(env, static_cast<std::uint64_t>(0));
    }
    const auto addr = reinterpret_cast<std::uintptr_t>(cap->managed);
    return Napi::BigInt::New(env, static_cast<std::uint64_t>(addr));
  });
}

}  // namespace

void RegisterDlpack(Napi::Env env, Napi::Object exports) {
  exports.Set("toDlpack", Napi::Function::New(env, ToDlpackOp));
  exports.Set("fromDlpack", Napi::Function::New(env, FromDlpackOp));
  exports.Set("dlpackDevice", Napi::Function::New(env, DlpackDeviceOp));
  exports.Set("dlpackPointer", Napi::Function::New(env, DlpackPointerOp));
}

}  // namespace norch
