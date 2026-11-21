#pragma once

#include <ATen/TensorIndexing.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <c10/core/Device.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <napi.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace norch {

[[nodiscard]] c10::ScalarType ParseScalarType(Napi::Env env, const Napi::Value& val);
[[nodiscard]] std::string ScalarTypeToString(c10::ScalarType dtype);

[[nodiscard]] c10::Device ParseDevice(Napi::Env env, const Napi::Value& val);
[[nodiscard]] std::string DeviceToString(const c10::Device& device);

[[nodiscard]] c10::Layout ParseLayout(Napi::Env env, const Napi::Value& val);
[[nodiscard]] c10::MemoryFormat ParseMemoryFormat(Napi::Env env, const Napi::Value& val);

[[nodiscard]] c10::Scalar ParseScalar(Napi::Env env, const Napi::Value& val);
[[nodiscard]] std::vector<std::int64_t> ParseIntArray(Napi::Env env, const Napi::Value& val);
[[nodiscard]] std::vector<double> ParseDoubleArray(Napi::Env env, const Napi::Value& val);
[[nodiscard]] std::vector<at::Tensor> ParseTensorList(Napi::Env env, const Napi::Value& val);
[[nodiscard]] c10::TensorOptions ParseTensorOptions(
    Napi::Env env,
    const Napi::Value& val,
    bool* out_requires_grad = nullptr);

[[nodiscard]] std::vector<at::indexing::TensorIndex> ParseTensorIndices(
    Napi::Env env,
    const Napi::Value& val);

[[nodiscard]] c10::IValue JsToIValue(
    Napi::Env env,
    const Napi::Value& val,
    const c10::TypePtr& target_type);

[[nodiscard]] Napi::Value IValueToJs(Napi::Env env, const c10::IValue& ival);

[[nodiscard]] Napi::Value ScalarToJs(Napi::Env env, const c10::Scalar& scalar);

[[nodiscard]] Napi::Error TranslateException(Napi::Env env, const std::exception& ex);

template <typename Fn>
Napi::Value GuardNapi(Napi::Env env, Fn&& fn) {
  try {
    return std::forward<Fn>(fn)();
  } catch (const Napi::Error& err) {
    err.ThrowAsJavaScriptException();
    return env.Undefined();
  } catch (const c10::Error& err) {
    Napi::Error::New(env, err.what_without_backtrace()).ThrowAsJavaScriptException();
    return env.Undefined();
  } catch (const std::exception& ex) {
    TranslateException(env, ex).ThrowAsJavaScriptException();
    return env.Undefined();
  } catch (...) {
    Napi::Error::New(env, "Unknown native C++ exception in norch").ThrowAsJavaScriptException();
    return env.Undefined();
  }
}

}  // namespace norch
