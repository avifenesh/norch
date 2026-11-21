#include "norch/convert.hpp"

#include <ATen/core/List.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <c10/core/Device.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "norch/tensor_wrap.hpp"

namespace norch {

namespace {

[[nodiscard]] std::int64_t JsToInt64(Napi::Env env, const Napi::Value& val) {
  if (val.IsBigInt()) {
    bool lossless = false;
    const auto v = val.As<Napi::BigInt>().Int64Value(&lossless);
    if (!lossless) {
      throw Napi::RangeError::New(env, "BigInt out of int64 range");
    }
    return v;
  }
  if (val.IsNumber()) {
    return val.As<Napi::Number>().Int64Value();
  }
  if (val.IsBoolean()) {
    return val.As<Napi::Boolean>().Value() ? 1 : 0;
  }
  throw Napi::TypeError::New(env, "Expected an integer number or BigInt");
}

[[nodiscard]] bool IsExactSafeInteger(double d) noexcept {
  constexpr double kMaxSafe = 9007199254740991.0;  // 2^53 - 1
  return std::isfinite(d) && std::trunc(d) == d && d >= -kMaxSafe && d <= kMaxSafe;
}

[[nodiscard]] c10::IValue JsToTypedList(
    Napi::Env env,
    const Napi::Value& val,
    const c10::TypePtr& elem_type) {
  if (!val.IsArray() && !val.IsTypedArray()) {
    // Allow passing a single scalar/int where PyTorch expects IntList (e.g., kernel_size=3 -> [3])
    const auto kind = elem_type->kind();
    if (kind == c10::TypeKind::IntType || kind == c10::TypeKind::SymIntType) {
      c10::List<std::int64_t> list;
      list.push_back(JsToInt64(env, val));
      return list;
    }
    throw Napi::TypeError::New(env, "Expected an Array for list argument");
  }

  const auto kind = elem_type->kind();
  if (kind == c10::TypeKind::TensorType) {
    c10::List<at::Tensor> list;
    const auto arr = val.As<Napi::Array>();
    list.reserve(arr.Length());
    for (std::uint32_t i = 0; i < arr.Length(); ++i) {
      list.push_back(TensorWrap::UnwrapTensor(env, arr.Get(i)));
    }
    return list;
  }

  if (kind == c10::TypeKind::OptionalType) {
    const auto opt = elem_type->expect<c10::OptionalType>();
    if (opt->getElementType()->kind() == c10::TypeKind::TensorType) {
      c10::List<std::optional<at::Tensor>> list;
      const auto arr = val.As<Napi::Array>();
      list.reserve(arr.Length());
      for (std::uint32_t i = 0; i < arr.Length(); ++i) {
        const auto item = arr.Get(i);
        if (item.IsNull() || item.IsUndefined()) {
          list.push_back(std::nullopt);
        } else {
          list.push_back(TensorWrap::UnwrapTensor(env, item));
        }
      }
      return list;
    }
  }

  if (kind == c10::TypeKind::IntType || kind == c10::TypeKind::SymIntType) {
    const auto vec = ParseIntArray(env, val);
    c10::List<std::int64_t> list;
    list.reserve(vec.size());
    for (const auto x : vec) {
      list.push_back(x);
    }
    return list;
  }

  if (kind == c10::TypeKind::FloatType || kind == c10::TypeKind::SymFloatType) {
    const auto vec = ParseDoubleArray(env, val);
    c10::List<double> list;
    list.reserve(vec.size());
    for (const auto x : vec) {
      list.push_back(x);
    }
    return list;
  }

  if (kind == c10::TypeKind::BoolType || kind == c10::TypeKind::SymBoolType) {
    const auto arr = val.As<Napi::Array>();
    c10::List<bool> list;
    list.reserve(arr.Length());
    for (std::uint32_t i = 0; i < arr.Length(); ++i) {
      list.push_back(arr.Get(i).ToBoolean().Value());
    }
    return list;
  }

  if (kind == c10::TypeKind::StringType) {
    const auto arr = val.As<Napi::Array>();
    c10::List<std::string> list;
    list.reserve(arr.Length());
    for (std::uint32_t i = 0; i < arr.Length(); ++i) {
      list.push_back(arr.Get(i).As<Napi::String>().Utf8Value());
    }
    return list;
  }

  const auto arr = val.As<Napi::Array>();
  c10::impl::GenericList list(elem_type);
  list.reserve(arr.Length());
  for (std::uint32_t i = 0; i < arr.Length(); ++i) {
    list.push_back(JsToIValue(env, arr.Get(i), elem_type));
  }
  return list;
}

}  // namespace

c10::ScalarType ParseScalarType(Napi::Env env, const Napi::Value& val) {
  if (val.IsNumber()) {
    const auto code = val.As<Napi::Number>().Int32Value();
    return static_cast<c10::ScalarType>(code);
  }
  if (!val.IsString()) {
    throw Napi::TypeError::New(env, "Expected dtype string or integer enum");
  }
  const std::string s = val.As<Napi::String>().Utf8Value();
  if (s == "float32" || s == "float" || s == "f32") return c10::ScalarType::Float;
  if (s == "float64" || s == "double" || s == "f64") return c10::ScalarType::Double;
  if (s == "float16" || s == "half" || s == "f16") return c10::ScalarType::Half;
  if (s == "bfloat16" || s == "bf16") return c10::ScalarType::BFloat16;
  if (s == "int8" || s == "char" || s == "i8") return c10::ScalarType::Char;
  if (s == "uint8" || s == "byte" || s == "u8") return c10::ScalarType::Byte;
  if (s == "int16" || s == "short" || s == "i16") return c10::ScalarType::Short;
  if (s == "int32" || s == "int" || s == "i32") return c10::ScalarType::Int;
  if (s == "int64" || s == "long" || s == "i64") return c10::ScalarType::Long;
  if (s == "bool") return c10::ScalarType::Bool;
  if (s == "complex64" || s == "cfloat" || s == "c64") return c10::ScalarType::ComplexFloat;
  if (s == "complex128" || s == "cdouble" || s == "c128") return c10::ScalarType::ComplexDouble;
  if (s == "float8_e4m3fn") return c10::ScalarType::Float8_e4m3fn;
  if (s == "float8_e5m2") return c10::ScalarType::Float8_e5m2;
  throw Napi::RangeError::New(env, "Unknown dtype: " + s);
}

std::string ScalarTypeToString(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Float:
      return "float32";
    case c10::ScalarType::Double:
      return "float64";
    case c10::ScalarType::Half:
      return "float16";
    case c10::ScalarType::BFloat16:
      return "bfloat16";
    case c10::ScalarType::Char:
      return "int8";
    case c10::ScalarType::Byte:
      return "uint8";
    case c10::ScalarType::Short:
      return "int16";
    case c10::ScalarType::Int:
      return "int32";
    case c10::ScalarType::Long:
      return "int64";
    case c10::ScalarType::Bool:
      return "bool";
    case c10::ScalarType::ComplexFloat:
      return "complex64";
    case c10::ScalarType::ComplexDouble:
      return "complex128";
    case c10::ScalarType::Float8_e4m3fn:
      return "float8_e4m3fn";
    case c10::ScalarType::Float8_e5m2:
      return "float8_e5m2";
    default:
      return c10::toString(dtype);
  }
}

c10::Device ParseDevice(Napi::Env env, const Napi::Value& val) {
  if (val.IsString()) {
    return {val.As<Napi::String>().Utf8Value()};
  }
  if (val.IsObject()) {
    const auto obj = val.As<Napi::Object>();
    if (obj.Has("type")) {
      const std::string type_str = obj.Get("type").As<Napi::String>().Utf8Value();
      c10::Device dev(type_str);
      if (obj.Has("index") && obj.Get("index").IsNumber()) {
        dev.set_index(static_cast<c10::DeviceIndex>(obj.Get("index").As<Napi::Number>().Int32Value()));
      }
      return dev;
    }
  }
  throw Napi::TypeError::New(env, "Expected device string (e.g. 'cpu', 'cuda:0') or Device object");
}

std::string DeviceToString(const c10::Device& device) {
  return device.str();
}

c10::Layout ParseLayout(Napi::Env env, const Napi::Value& val) {
  if (val.IsNumber()) {
    return static_cast<c10::Layout>(val.As<Napi::Number>().Int32Value());
  }
  if (!val.IsString()) {
    throw Napi::TypeError::New(env, "Expected layout string or integer");
  }
  const std::string s = val.As<Napi::String>().Utf8Value();
  if (s == "strided") return c10::kStrided;
  if (s == "sparse_coo" || s == "sparse") return c10::kSparse;
  if (s == "sparse_csr") return c10::kSparseCsr;
  if (s == "sparse_csc") return c10::kSparseCsc;
  if (s == "sparse_bsr") return c10::kSparseBsr;
  if (s == "sparse_bsc") return c10::kSparseBsc;
  throw Napi::RangeError::New(env, "Unknown layout: " + s);
}

c10::MemoryFormat ParseMemoryFormat(Napi::Env env, const Napi::Value& val) {
  if (val.IsNumber()) {
    return static_cast<c10::MemoryFormat>(val.As<Napi::Number>().Int32Value());
  }
  if (!val.IsString()) {
    throw Napi::TypeError::New(env, "Expected memory format string or integer");
  }
  const std::string s = val.As<Napi::String>().Utf8Value();
  if (s == "contiguous" || s == "contiguous_format") return c10::MemoryFormat::Contiguous;
  if (s == "preserve" || s == "preserve_format") return c10::MemoryFormat::Preserve;
  if (s == "channels_last") return c10::MemoryFormat::ChannelsLast;
  if (s == "channels_last_3d") return c10::MemoryFormat::ChannelsLast3d;
  throw Napi::RangeError::New(env, "Unknown memory format: " + s);
}

c10::Scalar ParseScalar(Napi::Env env, const Napi::Value& val) {
  if (val.IsBoolean()) {
    return {val.As<Napi::Boolean>().Value()};
  }
  if (val.IsBigInt()) {
    bool lossless = false;
    const auto v = val.As<Napi::BigInt>().Int64Value(&lossless);
    if (!lossless) {
      throw Napi::RangeError::New(env, "BigInt scalar out of int64 range");
    }
    return {v};
  }
  if (val.IsNumber()) {
    const double d = val.As<Napi::Number>().DoubleValue();
    if (IsExactSafeInteger(d)) {
      return {static_cast<std::int64_t>(d)};
    }
    return {d};
  }
  throw Napi::TypeError::New(env, "Expected a scalar (number, boolean, or BigInt)");
}

std::vector<std::int64_t> ParseIntArray(Napi::Env env, const Napi::Value& val) {
  if (val.IsNumber() || val.IsBigInt()) {
    return {JsToInt64(env, val)};
  }
  if (val.IsTypedArray()) {
    const auto ta = val.As<Napi::TypedArray>();
    const auto len = ta.ElementLength();
    std::vector<std::int64_t> out;
    out.reserve(len);
    if (ta.TypedArrayType() == napi_int32_array) {
      const auto typed = val.As<Napi::Int32Array>();
      for (std::size_t i = 0; i < len; ++i) {
        out.push_back(static_cast<std::int64_t>(typed[i]));
      }
      return out;
    }
    if (ta.TypedArrayType() == napi_bigint64_array) {
      const auto typed = val.As<Napi::BigInt64Array>();
      for (std::size_t i = 0; i < len; ++i) {
        out.push_back(typed[i]);
      }
      return out;
    }
  }
  if (!val.IsArray()) {
    throw Napi::TypeError::New(env, "Expected an integer or array of integers");
  }
  const auto arr = val.As<Napi::Array>();
  std::vector<std::int64_t> out;
  out.reserve(arr.Length());
  for (std::uint32_t i = 0; i < arr.Length(); ++i) {
    out.push_back(JsToInt64(env, arr.Get(i)));
  }
  return out;
}

std::vector<double> ParseDoubleArray(Napi::Env env, const Napi::Value& val) {
  if (val.IsNumber()) {
    return {val.As<Napi::Number>().DoubleValue()};
  }
  if (!val.IsArray()) {
    throw Napi::TypeError::New(env, "Expected a number or array of numbers");
  }
  const auto arr = val.As<Napi::Array>();
  std::vector<double> out;
  out.reserve(arr.Length());
  for (std::uint32_t i = 0; i < arr.Length(); ++i) {
    out.push_back(arr.Get(i).As<Napi::Number>().DoubleValue());
  }
  return out;
}

std::vector<at::Tensor> ParseTensorList(Napi::Env env, const Napi::Value& val) {
  if (!val.IsArray()) {
    throw Napi::TypeError::New(env, "Expected an Array of Tensors");
  }
  const auto arr = val.As<Napi::Array>();
  std::vector<at::Tensor> out;
  out.reserve(arr.Length());
  for (std::uint32_t i = 0; i < arr.Length(); ++i) {
    out.push_back(TensorWrap::UnwrapTensor(env, arr.Get(i)));
  }
  return out;
}

c10::TensorOptions ParseTensorOptions(
    Napi::Env env,
    const Napi::Value& val,
    bool* out_requires_grad) {
  c10::TensorOptions opts = at::TensorOptions().dtype(c10::ScalarType::Float);
  if (out_requires_grad != nullptr) {
    *out_requires_grad = false;
  }
  if (val.IsUndefined() || val.IsNull()) {
    return opts;
  }
  if (!val.IsObject()) {
    throw Napi::TypeError::New(env, "Expected tensor options object");
  }
  const auto obj = val.As<Napi::Object>();
  if (obj.Has("dtype")) {
    const auto v = obj.Get("dtype");
    if (!v.IsUndefined() && !v.IsNull()) {
      opts = opts.dtype(ParseScalarType(env, v));
    }
  }
  if (obj.Has("device")) {
    const auto v = obj.Get("device");
    if (!v.IsUndefined() && !v.IsNull()) {
      opts = opts.device(ParseDevice(env, v));
    }
  }
  if (obj.Has("layout")) {
    const auto v = obj.Get("layout");
    if (!v.IsUndefined() && !v.IsNull()) {
      opts = opts.layout(ParseLayout(env, v));
    }
  }
  if (obj.Has("pinMemory")) {
    const auto v = obj.Get("pinMemory");
    if (v.IsBoolean()) {
      opts = opts.pinned_memory(v.As<Napi::Boolean>().Value());
    }
  }
  if (obj.Has("requiresGrad")) {
    const auto v = obj.Get("requiresGrad");
    if (v.IsBoolean()) {
      const bool req = v.As<Napi::Boolean>().Value();
      opts = opts.requires_grad(req);
      if (out_requires_grad != nullptr) {
        *out_requires_grad = req;
      }
    }
  }
  return opts;
}

std::vector<at::indexing::TensorIndex> ParseTensorIndices(Napi::Env env, const Napi::Value& val) {
  if (!val.IsArray()) {
    throw Napi::TypeError::New(env, "Expected an Array of indices");
  }
  const auto arr = val.As<Napi::Array>();
  std::vector<at::indexing::TensorIndex> out;
  out.reserve(arr.Length());

  for (std::uint32_t i = 0; i < arr.Length(); ++i) {
    const auto item = arr.Get(i);
    if (item.IsNull() || item.IsUndefined()) {
      out.emplace_back(at::indexing::None);
    } else if (item.IsBoolean()) {
      out.emplace_back(item.As<Napi::Boolean>().Value());
    } else if (item.IsNumber() || item.IsBigInt()) {
      out.emplace_back(JsToInt64(env, item));
    } else if (item.IsString()) {
      const std::string s = item.As<Napi::String>().Utf8Value();
      if (s == "...") {
        out.emplace_back(at::indexing::Ellipsis);
      } else if (s == ":") {
        out.emplace_back(at::indexing::Slice());
      } else {
        throw Napi::RangeError::New(env, "Unsupported string index: " + s);
      }
    } else if (TensorWrap::IsTensorWrap(env, item)) {
      out.emplace_back(TensorWrap::UnwrapTensor(env, item));
    } else if (item.IsObject()) {
      const auto obj = item.As<Napi::Object>();
      if (obj.Has("__slice") && obj.Get("__slice").ToBoolean().Value()) {
        std::optional<c10::SymInt> start = std::nullopt;
        std::optional<c10::SymInt> stop = std::nullopt;
        std::optional<c10::SymInt> step = std::nullopt;
        if (obj.Has("start")) {
          const auto v = obj.Get("start");
          if (!v.IsNull() && !v.IsUndefined()) start = c10::SymInt(JsToInt64(env, v));
        }
        if (obj.Has("stop")) {
          const auto v = obj.Get("stop");
          if (!v.IsNull() && !v.IsUndefined()) stop = c10::SymInt(JsToInt64(env, v));
        }
        if (obj.Has("step")) {
          const auto v = obj.Get("step");
          if (!v.IsNull() && !v.IsUndefined()) step = c10::SymInt(JsToInt64(env, v));
        }
        out.emplace_back(at::indexing::Slice(start, stop, step));
      } else {
        throw Napi::TypeError::New(env, "Unsupported object in tensor index list");
      }
    } else {
      throw Napi::TypeError::New(env, "Unsupported tensor index element");
    }
  }
  return out;
}

c10::IValue JsToIValue(
    Napi::Env env,
    const Napi::Value& val,
    const c10::TypePtr& target_type) {
  const auto kind = target_type->kind();
  if (kind == c10::TypeKind::OptionalType) {
    if (val.IsUndefined() || val.IsNull()) {
      return {};
    }
    return JsToIValue(env, val, target_type->expect<c10::OptionalType>()->getElementType());
  }

  switch (kind) {
    case c10::TypeKind::TensorType:
      return {TensorWrap::UnwrapTensor(env, val)};
    case c10::TypeKind::IntType:
    case c10::TypeKind::SymIntType:
      if (val.IsString()) {
        // Handle enum-encoded IntType arguments like ScalarType/Layout/MemoryFormat
        try {
          return {static_cast<std::int64_t>(ParseScalarType(env, val))};
        } catch (...) {
          try {
            return {static_cast<std::int64_t>(ParseMemoryFormat(env, val))};
          } catch (...) {
            return {static_cast<std::int64_t>(ParseLayout(env, val))};
          }
        }
      }
      return {JsToInt64(env, val)};
    case c10::TypeKind::FloatType:
    case c10::TypeKind::SymFloatType:
      return {val.As<Napi::Number>().DoubleValue()};
    case c10::TypeKind::BoolType:
    case c10::TypeKind::SymBoolType:
      return {val.ToBoolean().Value()};
    case c10::TypeKind::NumberType:
      return {ParseScalar(env, val)};
    case c10::TypeKind::StringType:
      return {val.As<Napi::String>().Utf8Value()};
    case c10::TypeKind::DeviceObjType:
      return {ParseDevice(env, val)};
    case c10::TypeKind::ScalarTypeType:
      return {ParseScalarType(env, val)};
    case c10::TypeKind::LayoutType:
      return {ParseLayout(env, val)};
    case c10::TypeKind::MemoryFormatType:
      return {ParseMemoryFormat(env, val)};
    case c10::TypeKind::ListType:
      return JsToTypedList(env, val, target_type->expect<c10::ListType>()->getElementType());
    case c10::TypeKind::NoneType:
      return {};
    case c10::TypeKind::GeneratorType:
      if (val.IsUndefined() || val.IsNull()) {
        return {};
      }
      throw Napi::TypeError::New(env, "Custom Generator instances are not supported in boxed ops yet");
    default:
      throw Napi::TypeError::New(
          env, std::string("Unsupported ATen schema argument type: ") + target_type->repr_str());
  }
}

Napi::Value ScalarToJs(Napi::Env env, const c10::Scalar& scalar) {
  if (scalar.isBoolean()) {
    return Napi::Boolean::New(env, scalar.toBool());
  }
  if (scalar.isIntegral(false)) {
    const auto v = scalar.toLong();
    constexpr std::int64_t kMaxSafe = 9007199254740991LL;
    if (v >= -kMaxSafe && v <= kMaxSafe) {
      return Napi::Number::New(env, static_cast<double>(v));
    }
    return Napi::BigInt::New(env, v);
  }
  if (scalar.isComplex()) {
    const auto c = scalar.toComplexDouble();
    auto obj = Napi::Object::New(env);
    obj.Set("real", Napi::Number::New(env, c.real()));
    obj.Set("imag", Napi::Number::New(env, c.imag()));
    return obj;
  }
  return Napi::Number::New(env, scalar.toDouble());
}

Napi::Value IValueToJs(Napi::Env env, const c10::IValue& ival) {
  if (ival.isNone()) {
    return env.Null();
  }
  if (ival.isTensor()) {
    const auto& t = ival.toTensor();
    if (!t.defined()) {
      return env.Null();
    }
    return TensorWrap::NewInstance(env, t);
  }
  if (ival.isBool()) {
    return Napi::Boolean::New(env, ival.toBool());
  }
  if (ival.isInt()) {
    return ScalarToJs(env, c10::Scalar(ival.toInt()));
  }
  if (ival.isDouble()) {
    return Napi::Number::New(env, ival.toDouble());
  }
  if (ival.isScalar()) {
    return ScalarToJs(env, ival.toScalar());
  }
  if (ival.isString()) {
    return Napi::String::New(env, ival.toStringRef());
  }
  if (ival.isDevice()) {
    return Napi::String::New(env, DeviceToString(ival.toDevice()));
  }
  if (ival.isTensorList()) {
    const auto list = ival.toTensorList();
    auto arr = Napi::Array::New(env, list.size());
    for (std::size_t i = 0; i < list.size(); ++i) {
      arr.Set(static_cast<std::uint32_t>(i), TensorWrap::NewInstance(env, list[i]));
    }
    return arr;
  }
  if (ival.isList()) {
    const auto list = ival.toList();
    auto arr = Napi::Array::New(env, list.size());
    for (std::size_t i = 0; i < list.size(); ++i) {
      arr.Set(static_cast<std::uint32_t>(i), IValueToJs(env, list.get(i)));
    }
    return arr;
  }
  if (ival.isTuple()) {
    const auto& elems = ival.toTupleRef().elements();
    auto arr = Napi::Array::New(env, elems.size());
    for (std::size_t i = 0; i < elems.size(); ++i) {
      arr.Set(static_cast<std::uint32_t>(i), IValueToJs(env, elems[i]));
    }
    return arr;
  }
  if (ival.isGenericDict()) {
    const auto dict = ival.toGenericDict();
    auto obj = Napi::Object::New(env);
    for (const auto& entry : dict) {
      const auto k = entry.key().isString() ? entry.key().toStringRef() : entry.key().tagKind();
      obj.Set(k, IValueToJs(env, entry.value()));
    }
    return obj;
  }
  throw Napi::Error::New(env, std::string("Unsupported IValue return type: ") + ival.tagKind());
}

Napi::Error TranslateException(Napi::Env env, const std::exception& ex) {
  return Napi::Error::New(env, ex.what());
}

}  // namespace norch
