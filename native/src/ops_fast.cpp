#include "norch/ops_fast.hpp"

#include <ATen/ATen.h>
#include <c10/core/GradMode.h>
#include <c10/core/ScalarType.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "norch/convert.hpp"
#include "norch/generator_wrap.hpp"
#include "norch/tensor_wrap.hpp"

namespace norch {

namespace {

[[nodiscard]] std::optional<at::Generator> ParseOptionalGenerator(
    Napi::Env env,
    const Napi::Value& opts_val) {
  if (!opts_val.IsObject() || opts_val.IsNull()) {
    return std::nullopt;
  }
  const auto obj = opts_val.As<Napi::Object>();
  if (obj.Has("generator")) {
    const auto g = obj.Get("generator");
    if (!g.IsUndefined() && !g.IsNull()) {
      return GeneratorWrap::UnwrapGenerator(env, g);
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::vector<std::int64_t>> ParseOptionalIntArray(
    Napi::Env env,
    const Napi::Value& val) {
  if (val.IsUndefined() || val.IsNull()) {
    return std::nullopt;
  }
  return ParseIntArray(env, val);
}

[[nodiscard]] std::optional<double> ParseOptionalDouble(const Napi::Value& val) {
  if (val.IsUndefined() || val.IsNull()) {
    return std::nullopt;
  }
  return val.As<Napi::Number>().DoubleValue();
}

[[nodiscard]] std::optional<at::Tensor> ParseOptionalTensor(Napi::Env env, const Napi::Value& val) {
  if (val.IsUndefined() || val.IsNull()) {
    return std::nullopt;
  }
  return TensorWrap::UnwrapTensor(env, val);
}

// Helper for flattening nested JS arrays of numbers/booleans into doubles + inferred shape
void CollectNestedArray(
    Napi::Env env,
    const Napi::Value& val,
    std::vector<std::int64_t>& shape,
    std::size_t depth,
    std::vector<double>& flat,
    bool& all_bool,
    bool& all_int) {
  if (val.IsArray()) {
    const auto arr = val.As<Napi::Array>();
    const auto len = static_cast<std::int64_t>(arr.Length());
    if (shape.size() == depth) {
      shape.push_back(len);
    } else if (shape[depth] != len) {
      throw Napi::RangeError::New(
          env, "Ragged nested arrays are not supported when creating a Tensor");
    }
    for (std::uint32_t i = 0; i < arr.Length(); ++i) {
      CollectNestedArray(env, arr.Get(i), shape, depth + 1, flat, all_bool, all_int);
    }
    return;
  }
  if (depth != shape.size()) {
    throw Napi::RangeError::New(env, "Inconsistent nesting depth when creating a Tensor");
  }
  if (val.IsBoolean()) {
    flat.push_back(val.As<Napi::Boolean>().Value() ? 1.0 : 0.0);
    all_int = false;
  } else if (val.IsBigInt()) {
    bool lossless = false;
    const auto v = val.As<Napi::BigInt>().Int64Value(&lossless);
    flat.push_back(static_cast<double>(v));
    all_bool = false;
  } else if (val.IsNumber()) {
    const double d = val.As<Napi::Number>().DoubleValue();
    flat.push_back(d);
    all_bool = false;
    if (!std::isfinite(d) || std::trunc(d) != d) {
      all_int = false;
    }
  } else {
    throw Napi::TypeError::New(env, "Tensor data elements must be numbers, booleans, or BigInts");
  }
}

// ---------- Factories ----------

Napi::Value TensorFromData(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    if (info.Length() < 1) {
      throw Napi::TypeError::New(env, "tensor(data, options?) expects at least 1 argument");
    }

    const bool has_opts = info.Length() >= 2 && info[1].IsObject() && !info[1].IsNull();
    bool has_explicit_dtype = false;
    bool requires_grad = false;
    c10::TensorOptions opts = at::TensorOptions().dtype(c10::ScalarType::Float);
    if (has_opts) {
      const auto obj = info[1].As<Napi::Object>();
      has_explicit_dtype =
          obj.Has("dtype") && !obj.Get("dtype").IsUndefined() && !obj.Get("dtype").IsNull();
      opts = ParseTensorOptions(env, info[1], &requires_grad);
    }

    if (TensorWrap::IsTensorWrap(env, info[0])) {
      at::Tensor t = TensorWrap::UnwrapTensor(env, info[0]).clone();
      if (has_opts) {
        const auto obj = info[1].As<Napi::Object>();
        c10::Device dev = t.device();
        c10::ScalarType dt = t.scalar_type();
        if (obj.Has("device") && !obj.Get("device").IsUndefined() && !obj.Get("device").IsNull()) {
          dev = ParseDevice(env, obj.Get("device"));
        }
        if (has_explicit_dtype) {
          dt = ParseScalarType(env, obj.Get("dtype"));
        }
        t = t.to(dev, dt);
      }
      if (requires_grad) t.set_requires_grad(true);
      return TensorWrap::NewInstance(env, std::move(t));
    }

    std::vector<std::int64_t> shape;
    std::vector<double> flat;
    bool all_bool = true;
    bool all_int = true;
    CollectNestedArray(env, info[0], shape, 0, flat, all_bool, all_int);

    c10::ScalarType target_dtype = opts.dtype().toScalarType();
    if (!has_explicit_dtype) {
      if (!flat.empty() && all_bool) {
        target_dtype = c10::ScalarType::Bool;
      } else {
        target_dtype = c10::ScalarType::Float;
      }
    }

    at::Tensor cpu_t;
    if (target_dtype == c10::ScalarType::Bool) {
      cpu_t = at::empty(shape, at::TensorOptions().dtype(c10::ScalarType::Bool));
      auto* dst = cpu_t.data_ptr<bool>();
      for (std::size_t i = 0; i < flat.size(); ++i) {
        dst[i] = (flat[i] != 0.0);
      }
    } else if (target_dtype == c10::ScalarType::Long) {
      cpu_t = at::empty(shape, at::TensorOptions().dtype(c10::ScalarType::Long));
      auto* dst = cpu_t.data_ptr<std::int64_t>();
      for (std::size_t i = 0; i < flat.size(); ++i) {
        dst[i] = static_cast<std::int64_t>(flat[i]);
      }
    } else if (target_dtype == c10::ScalarType::Int) {
      cpu_t = at::empty(shape, at::TensorOptions().dtype(c10::ScalarType::Int));
      auto* dst = cpu_t.data_ptr<std::int32_t>();
      for (std::size_t i = 0; i < flat.size(); ++i) {
        dst[i] = static_cast<std::int32_t>(flat[i]);
      }
    } else if (target_dtype == c10::ScalarType::Double) {
      cpu_t = at::empty(shape, at::TensorOptions().dtype(c10::ScalarType::Double));
      auto* dst = cpu_t.data_ptr<double>();
      for (std::size_t i = 0; i < flat.size(); ++i) {
        dst[i] = flat[i];
      }
    } else {
      cpu_t = at::empty(shape, at::TensorOptions().dtype(c10::ScalarType::Double));
      auto* dst = cpu_t.data_ptr<double>();
      for (std::size_t i = 0; i < flat.size(); ++i) {
        dst[i] = flat[i];
      }
      cpu_t = cpu_t.to(target_dtype);
    }

    at::Tensor result = cpu_t.to(opts.device());
    if (requires_grad) {
      result.set_requires_grad(true);
    }
    return TensorWrap::NewInstance(env, std::move(result));
  });
}

[[nodiscard]] Napi::Value WrapFactory(Napi::Env env, at::Tensor t, bool requires_grad) {
  if (requires_grad) {
    t.set_requires_grad(true);
  }
  return TensorWrap::NewInstance(env, std::move(t));
}

Napi::Value EmptyOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto shape = ParseIntArray(env, info[0]);
    bool requires_grad = false;
    const auto opts = info.Length() >= 2 ? ParseTensorOptions(env, info[1], &requires_grad)
                                         : at::TensorOptions().dtype(at::kFloat);
    return WrapFactory(env, at::empty(shape, opts), requires_grad);
  });
}

Napi::Value ZerosOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto shape = ParseIntArray(env, info[0]);
    bool requires_grad = false;
    const auto opts = info.Length() >= 2 ? ParseTensorOptions(env, info[1], &requires_grad)
                                         : at::TensorOptions().dtype(at::kFloat);
    return WrapFactory(env, at::zeros(shape, opts), requires_grad);
  });
}

Napi::Value OnesOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto shape = ParseIntArray(env, info[0]);
    bool requires_grad = false;
    const auto opts = info.Length() >= 2 ? ParseTensorOptions(env, info[1], &requires_grad)
                                         : at::TensorOptions().dtype(at::kFloat);
    return WrapFactory(env, at::ones(shape, opts), requires_grad);
  });
}

Napi::Value FullOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto shape = ParseIntArray(env, info[0]);
    const auto fill_val = ParseScalar(env, info[1]);
    bool requires_grad = false;
    const auto opts = info.Length() >= 3 ? ParseTensorOptions(env, info[2], &requires_grad)
                                         : at::TensorOptions().dtype(at::kFloat);
    return WrapFactory(env, at::full(shape, fill_val, opts), requires_grad);
  });
}

Napi::Value RandOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto shape = ParseIntArray(env, info[0]);
    bool requires_grad = false;
    const auto opts = info.Length() >= 2 ? ParseTensorOptions(env, info[1], &requires_grad)
                                         : at::TensorOptions().dtype(at::kFloat);
    const auto gen = info.Length() >= 2 ? ParseOptionalGenerator(env, info[1]) : std::nullopt;
    return WrapFactory(env, at::rand(shape, gen, opts), requires_grad);
  });
}

Napi::Value RandnOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto shape = ParseIntArray(env, info[0]);
    bool requires_grad = false;
    const auto opts = info.Length() >= 2 ? ParseTensorOptions(env, info[1], &requires_grad)
                                         : at::TensorOptions().dtype(at::kFloat);
    const auto gen = info.Length() >= 2 ? ParseOptionalGenerator(env, info[1]) : std::nullopt;
    return WrapFactory(env, at::randn(shape, gen, opts), requires_grad);
  });
}

Napi::Value RandintOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto low = info[0].As<Napi::Number>().Int64Value();
    const auto high = info[1].As<Napi::Number>().Int64Value();
    const auto shape = ParseIntArray(env, info[2]);
    bool requires_grad = false;
    auto opts = at::TensorOptions().dtype(c10::ScalarType::Long);
    std::optional<at::Generator> gen = std::nullopt;
    if (info.Length() >= 4 && info[3].IsObject() && !info[3].IsNull()) {
      const auto obj = info[3].As<Napi::Object>();
      opts = ParseTensorOptions(env, info[3], &requires_grad);
      gen = ParseOptionalGenerator(env, info[3]);
      if (!obj.Has("dtype") || obj.Get("dtype").IsUndefined()) {
        opts = opts.dtype(c10::ScalarType::Long);
      }
    }
    return WrapFactory(env, at::randint(low, high, shape, gen, opts), requires_grad);
  });
}

Napi::Value RandpermOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto n = info[0].As<Napi::Number>().Int64Value();
    bool requires_grad = false;
    auto opts = at::TensorOptions().dtype(c10::ScalarType::Long);
    std::optional<at::Generator> gen = std::nullopt;
    if (info.Length() >= 2 && info[1].IsObject() && !info[1].IsNull()) {
      const auto obj = info[1].As<Napi::Object>();
      opts = ParseTensorOptions(env, info[1], &requires_grad);
      gen = ParseOptionalGenerator(env, info[1]);
      if (!obj.Has("dtype") || obj.Get("dtype").IsUndefined()) {
        opts = opts.dtype(c10::ScalarType::Long);
      }
    }
    return WrapFactory(env, at::randperm(n, gen, opts), requires_grad);
  });
}

Napi::Value ArangeOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto start = ParseScalar(env, info[0]);
    const auto end = ParseScalar(env, info[1]);
    const auto step =
        (info.Length() >= 3 && !info[2].IsUndefined()) ? ParseScalar(env, info[2]) : c10::Scalar(1);
    const bool has_opts = info.Length() >= 4 && info[3].IsObject() && !info[3].IsNull();
    bool requires_grad = false;
    c10::TensorOptions opts = at::TensorOptions();
    if (has_opts) {
      const auto obj = info[3].As<Napi::Object>();
      opts = ParseTensorOptions(env, info[3], &requires_grad);
      if (!obj.Has("dtype") || obj.Get("dtype").IsUndefined()) {
        if (start.isIntegral(false) && end.isIntegral(false) && step.isIntegral(false)) {
          opts = opts.dtype(c10::ScalarType::Long);
        }
      }
    } else if (start.isIntegral(false) && end.isIntegral(false) && step.isIntegral(false)) {
      opts = opts.dtype(c10::ScalarType::Long);
    } else {
      opts = opts.dtype(c10::ScalarType::Float);
    }
    return WrapFactory(env, at::arange(start, end, step, opts), requires_grad);
  });
}

Napi::Value LinspaceOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto start = ParseScalar(env, info[0]);
    const auto end = ParseScalar(env, info[1]);
    const auto steps = info[2].As<Napi::Number>().Int64Value();
    bool requires_grad = false;
    const auto opts = info.Length() >= 4 ? ParseTensorOptions(env, info[3], &requires_grad)
                                         : at::TensorOptions().dtype(at::kFloat);
    return WrapFactory(env, at::linspace(start, end, steps, opts), requires_grad);
  });
}

Napi::Value EyeOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto n = info[0].As<Napi::Number>().Int64Value();
    const auto m =
        (info.Length() >= 2 && info[1].IsNumber()) ? info[1].As<Napi::Number>().Int64Value() : n;
    bool requires_grad = false;
    const auto opts = info.Length() >= 3 ? ParseTensorOptions(env, info[2], &requires_grad)
                                         : at::TensorOptions().dtype(at::kFloat);
    return WrapFactory(env, at::eye(n, m, opts), requires_grad);
  });
}

Napi::Value ZerosLikeOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& self = TensorWrap::UnwrapTensor(env, info[0]);
    return TensorWrap::NewInstance(env, at::zeros_like(self));
  });
}

Napi::Value OnesLikeOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& self = TensorWrap::UnwrapTensor(env, info[0]);
    return TensorWrap::NewInstance(env, at::ones_like(self));
  });
}

Napi::Value EmptyLikeOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& self = TensorWrap::UnwrapTensor(env, info[0]);
    return TensorWrap::NewInstance(env, at::empty_like(self));
  });
}

Napi::Value FullLikeOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& self = TensorWrap::UnwrapTensor(env, info[0]);
    const auto val = ParseScalar(env, info[1]);
    return TensorWrap::NewInstance(env, at::full_like(self, val));
  });
}

Napi::Value RandnLikeOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& self = TensorWrap::UnwrapTensor(env, info[0]);
    return TensorWrap::NewInstance(env, at::randn_like(self));
  });
}

// ---------- Binary / Unary Math ----------

Napi::Value AddOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& a = TensorWrap::UnwrapTensor(env, info[0]);
    const auto alpha =
        (info.Length() >= 3 && !info[2].IsUndefined()) ? ParseScalar(env, info[2]) : c10::Scalar(1);
    if (TensorWrap::IsTensorWrap(env, info[1])) {
      return TensorWrap::NewInstance(env, at::add(a, TensorWrap::UnwrapTensor(env, info[1]), alpha));
    }
    return TensorWrap::NewInstance(env, at::add(a, ParseScalar(env, info[1]), alpha));
  });
}

Napi::Value SubOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& a = TensorWrap::UnwrapTensor(env, info[0]);
    const auto alpha =
        (info.Length() >= 3 && !info[2].IsUndefined()) ? ParseScalar(env, info[2]) : c10::Scalar(1);
    if (TensorWrap::IsTensorWrap(env, info[1])) {
      return TensorWrap::NewInstance(env, at::sub(a, TensorWrap::UnwrapTensor(env, info[1]), alpha));
    }
    return TensorWrap::NewInstance(env, at::sub(a, ParseScalar(env, info[1]), alpha));
  });
}

Napi::Value MulOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& a = TensorWrap::UnwrapTensor(env, info[0]);
    if (TensorWrap::IsTensorWrap(env, info[1])) {
      return TensorWrap::NewInstance(env, at::mul(a, TensorWrap::UnwrapTensor(env, info[1])));
    }
    return TensorWrap::NewInstance(env, at::mul(a, ParseScalar(env, info[1])));
  });
}

Napi::Value DivOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& a = TensorWrap::UnwrapTensor(env, info[0]);
    if (TensorWrap::IsTensorWrap(env, info[1])) {
      return TensorWrap::NewInstance(env, at::div(a, TensorWrap::UnwrapTensor(env, info[1])));
    }
    return TensorWrap::NewInstance(env, at::div(a, ParseScalar(env, info[1])));
  });
}

Napi::Value PowOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& a = TensorWrap::UnwrapTensor(env, info[0]);
    if (TensorWrap::IsTensorWrap(env, info[1])) {
      return TensorWrap::NewInstance(env, at::pow(a, TensorWrap::UnwrapTensor(env, info[1])));
    }
    return TensorWrap::NewInstance(env, at::pow(a, ParseScalar(env, info[1])));
  });
}

Napi::Value MatmulOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return TensorWrap::NewInstance(
        env, at::matmul(TensorWrap::UnwrapTensor(env, info[0]), TensorWrap::UnwrapTensor(env, info[1])));
  });
}

Napi::Value BmmOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return TensorWrap::NewInstance(
        env, at::bmm(TensorWrap::UnwrapTensor(env, info[0]), TensorWrap::UnwrapTensor(env, info[1])));
  });
}

Napi::Value EinsumOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const std::string eq = info[0].As<Napi::String>().Utf8Value();
    const auto tensors = ParseTensorList(env, info[1]);
    return TensorWrap::NewInstance(env, at::einsum(eq, tensors));
  });
}

template <at::Tensor (*AtenFn)(const at::Tensor&)>
Napi::Value UnaryTensorOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return TensorWrap::NewInstance(env, AtenFn(TensorWrap::UnwrapTensor(env, info[0])));
  });
}

Napi::Value NegOp(const Napi::CallbackInfo& info) { return UnaryTensorOp<at::neg>(info); }
Napi::Value AbsOp(const Napi::CallbackInfo& info) { return UnaryTensorOp<at::abs>(info); }
Napi::Value ExpOp(const Napi::CallbackInfo& info) { return UnaryTensorOp<at::exp>(info); }
Napi::Value LogOp(const Napi::CallbackInfo& info) { return UnaryTensorOp<at::log>(info); }
Napi::Value SqrtOp(const Napi::CallbackInfo& info) { return UnaryTensorOp<at::sqrt>(info); }
Napi::Value RsqrtOp(const Napi::CallbackInfo& info) { return UnaryTensorOp<at::rsqrt>(info); }
Napi::Value SinOp(const Napi::CallbackInfo& info) { return UnaryTensorOp<at::sin>(info); }
Napi::Value CosOp(const Napi::CallbackInfo& info) { return UnaryTensorOp<at::cos>(info); }
Napi::Value TanhOp(const Napi::CallbackInfo& info) { return UnaryTensorOp<at::tanh>(info); }
Napi::Value SigmoidOp(const Napi::CallbackInfo& info) { return UnaryTensorOp<at::sigmoid>(info); }
Napi::Value ReluOp(const Napi::CallbackInfo& info) { return UnaryTensorOp<at::relu>(info); }
Napi::Value SiluOp(const Napi::CallbackInfo& info) { return UnaryTensorOp<at::silu>(info); }
Napi::Value IsnanOp(const Napi::CallbackInfo& info) { return UnaryTensorOp<at::isnan>(info); }
Napi::Value IsinfOp(const Napi::CallbackInfo& info) { return UnaryTensorOp<at::isinf>(info); }
Napi::Value IsfiniteOp(const Napi::CallbackInfo& info) { return UnaryTensorOp<at::isfinite>(info); }

Napi::Value GeluOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& x = TensorWrap::UnwrapTensor(env, info[0]);
    const std::string approx =
        (info.Length() >= 2 && info[1].IsString()) ? info[1].As<Napi::String>().Utf8Value() : "none";
    return TensorWrap::NewInstance(env, at::gelu(x, approx));
  });
}

Napi::Value LeakyReluOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& x = TensorWrap::UnwrapTensor(env, info[0]);
    const auto slope =
        (info.Length() >= 2 && !info[1].IsUndefined()) ? ParseScalar(env, info[1]) : c10::Scalar(0.01);
    return TensorWrap::NewInstance(env, at::leaky_relu(x, slope));
  });
}

Napi::Value ClampOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& x = TensorWrap::UnwrapTensor(env, info[0]);
    std::optional<c10::Scalar> lo = std::nullopt;
    std::optional<c10::Scalar> hi = std::nullopt;
    if (info.Length() >= 2 && !info[1].IsUndefined() && !info[1].IsNull()) {
      lo = ParseScalar(env, info[1]);
    }
    if (info.Length() >= 3 && !info[2].IsUndefined() && !info[2].IsNull()) {
      hi = ParseScalar(env, info[2]);
    }
    return TensorWrap::NewInstance(env, at::clamp(x, lo, hi));
  });
}

Napi::Value WhereOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& cond = TensorWrap::UnwrapTensor(env, info[0]);
    const auto& x = TensorWrap::UnwrapTensor(env, info[1]);
    const auto& y = TensorWrap::UnwrapTensor(env, info[2]);
    return TensorWrap::NewInstance(env, at::where(cond, x, y));
  });
}

// ---------- In-place ops ----------

Napi::Value AddInplaceOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    auto& a = TensorWrap::UnwrapMutableTensor(env, info[0]);
    const auto alpha =
        (info.Length() >= 3 && !info[2].IsUndefined()) ? ParseScalar(env, info[2]) : c10::Scalar(1);
    if (TensorWrap::IsTensorWrap(env, info[1])) {
      a.add_(TensorWrap::UnwrapTensor(env, info[1]), alpha);
    } else {
      a.add_(ParseScalar(env, info[1]), alpha);
    }
    return info[0];
  });
}

Napi::Value SubInplaceOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    auto& a = TensorWrap::UnwrapMutableTensor(env, info[0]);
    const auto alpha =
        (info.Length() >= 3 && !info[2].IsUndefined()) ? ParseScalar(env, info[2]) : c10::Scalar(1);
    if (TensorWrap::IsTensorWrap(env, info[1])) {
      a.sub_(TensorWrap::UnwrapTensor(env, info[1]), alpha);
    } else {
      a.sub_(ParseScalar(env, info[1]), alpha);
    }
    return info[0];
  });
}

Napi::Value MulInplaceOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    auto& a = TensorWrap::UnwrapMutableTensor(env, info[0]);
    if (TensorWrap::IsTensorWrap(env, info[1])) {
      a.mul_(TensorWrap::UnwrapTensor(env, info[1]));
    } else {
      a.mul_(ParseScalar(env, info[1]));
    }
    return info[0];
  });
}

Napi::Value DivInplaceOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    auto& a = TensorWrap::UnwrapMutableTensor(env, info[0]);
    if (TensorWrap::IsTensorWrap(env, info[1])) {
      a.div_(TensorWrap::UnwrapTensor(env, info[1]));
    } else {
      a.div_(ParseScalar(env, info[1]));
    }
    return info[0];
  });
}

Napi::Value ZeroInplaceOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    TensorWrap::UnwrapMutableTensor(env, info[0]).zero_();
    return info[0];
  });
}

Napi::Value FillInplaceOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    TensorWrap::UnwrapMutableTensor(env, info[0]).fill_(ParseScalar(env, info[1]));
    return info[0];
  });
}

Napi::Value UniformInplaceOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const double lo = info.Length() >= 2 ? info[1].As<Napi::Number>().DoubleValue() : 0.0;
    const double hi = info.Length() >= 3 ? info[2].As<Napi::Number>().DoubleValue() : 1.0;
    TensorWrap::UnwrapMutableTensor(env, info[0]).uniform_(lo, hi);
    return info[0];
  });
}

Napi::Value NormalInplaceOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const double mean = info.Length() >= 2 ? info[1].As<Napi::Number>().DoubleValue() : 0.0;
    const double std_dev = info.Length() >= 3 ? info[2].As<Napi::Number>().DoubleValue() : 1.0;
    TensorWrap::UnwrapMutableTensor(env, info[0]).normal_(mean, std_dev);
    return info[0];
  });
}

Napi::Value AddcmulInplaceOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    auto& self = TensorWrap::UnwrapMutableTensor(env, info[0]);
    const auto& t1 = TensorWrap::UnwrapTensor(env, info[1]);
    const auto& t2 = TensorWrap::UnwrapTensor(env, info[2]);
    const auto val =
        (info.Length() >= 4 && !info[3].IsUndefined()) ? ParseScalar(env, info[3]) : c10::Scalar(1);
    self.addcmul_(t1, t2, val);
    return info[0];
  });
}

Napi::Value AddcdivInplaceOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    auto& self = TensorWrap::UnwrapMutableTensor(env, info[0]);
    const auto& t1 = TensorWrap::UnwrapTensor(env, info[1]);
    const auto& t2 = TensorWrap::UnwrapTensor(env, info[2]);
    const auto val =
        (info.Length() >= 4 && !info[3].IsUndefined()) ? ParseScalar(env, info[3]) : c10::Scalar(1);
    self.addcdiv_(t1, t2, val);
    return info[0];
  });
}

Napi::Value ClampInplaceOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    auto& self = TensorWrap::UnwrapMutableTensor(env, info[0]);
    std::optional<c10::Scalar> lo = std::nullopt;
    std::optional<c10::Scalar> hi = std::nullopt;
    if (info.Length() >= 2 && !info[1].IsUndefined() && !info[1].IsNull()) lo = ParseScalar(env, info[1]);
    if (info.Length() >= 3 && !info[2].IsUndefined() && !info[2].IsNull()) hi = ParseScalar(env, info[2]);
    self.clamp_(lo, hi);
    return info[0];
  });
}

// ---------- Reductions ----------

Napi::Value SumOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& self = TensorWrap::UnwrapTensor(env, info[0]);
    const auto dim = info.Length() >= 2 ? ParseOptionalIntArray(env, info[1]) : std::nullopt;
    const bool keepdim = info.Length() >= 3 && info[2].ToBoolean().Value();
    std::optional<c10::ScalarType> dtype = std::nullopt;
    if (info.Length() >= 4 && !info[3].IsUndefined() && !info[3].IsNull()) {
      dtype = ParseScalarType(env, info[3]);
    }
    if (dim.has_value()) {
      return TensorWrap::NewInstance(env, at::sum(self, *dim, keepdim, dtype));
    }
    return TensorWrap::NewInstance(env, at::sum(self, dtype));
  });
}

Napi::Value MeanOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& self = TensorWrap::UnwrapTensor(env, info[0]);
    const auto dim = info.Length() >= 2 ? ParseOptionalIntArray(env, info[1]) : std::nullopt;
    const bool keepdim = info.Length() >= 3 && info[2].ToBoolean().Value();
    std::optional<c10::ScalarType> dtype = std::nullopt;
    if (info.Length() >= 4 && !info[3].IsUndefined() && !info[3].IsNull()) {
      dtype = ParseScalarType(env, info[3]);
    }
    if (dim.has_value()) {
      return TensorWrap::NewInstance(env, at::mean(self, *dim, keepdim, dtype));
    }
    return TensorWrap::NewInstance(env, at::mean(self, dtype));
  });
}

Napi::Value MaxOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& self = TensorWrap::UnwrapTensor(env, info[0]);
    if (info.Length() >= 2 && !info[1].IsUndefined() && !info[1].IsNull()) {
      const auto dim = info[1].As<Napi::Number>().Int64Value();
      const bool keepdim = info.Length() >= 3 && info[2].ToBoolean().Value();
      auto [values, indices] = at::max(self, dim, keepdim);
      auto res = Napi::Object::New(env);
      res.Set("values", TensorWrap::NewInstance(env, std::move(values)));
      res.Set("indices", TensorWrap::NewInstance(env, std::move(indices)));
      return res;
    }
    return TensorWrap::NewInstance(env, at::max(self));
  });
}

Napi::Value MinOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& self = TensorWrap::UnwrapTensor(env, info[0]);
    if (info.Length() >= 2 && !info[1].IsUndefined() && !info[1].IsNull()) {
      const auto dim = info[1].As<Napi::Number>().Int64Value();
      const bool keepdim = info.Length() >= 3 && info[2].ToBoolean().Value();
      auto [values, indices] = at::min(self, dim, keepdim);
      auto res = Napi::Object::New(env);
      res.Set("values", TensorWrap::NewInstance(env, std::move(values)));
      res.Set("indices", TensorWrap::NewInstance(env, std::move(indices)));
      return res;
    }
    return TensorWrap::NewInstance(env, at::min(self));
  });
}

Napi::Value ArgmaxOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& self = TensorWrap::UnwrapTensor(env, info[0]);
    std::optional<std::int64_t> dim = std::nullopt;
    if (info.Length() >= 2 && !info[1].IsUndefined() && !info[1].IsNull()) {
      dim = info[1].As<Napi::Number>().Int64Value();
    }
    const bool keepdim = info.Length() >= 3 && info[2].ToBoolean().Value();
    return TensorWrap::NewInstance(env, at::argmax(self, dim, keepdim));
  });
}

Napi::Value ArgminOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& self = TensorWrap::UnwrapTensor(env, info[0]);
    std::optional<std::int64_t> dim = std::nullopt;
    if (info.Length() >= 2 && !info[1].IsUndefined() && !info[1].IsNull()) {
      dim = info[1].As<Napi::Number>().Int64Value();
    }
    const bool keepdim = info.Length() >= 3 && info[2].ToBoolean().Value();
    return TensorWrap::NewInstance(env, at::argmin(self, dim, keepdim));
  });
}

Napi::Value SoftmaxOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& self = TensorWrap::UnwrapTensor(env, info[0]);
    const auto dim = info[1].As<Napi::Number>().Int64Value();
    return TensorWrap::NewInstance(env, at::softmax(self, dim));
  });
}

Napi::Value LogSoftmaxOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& self = TensorWrap::UnwrapTensor(env, info[0]);
    const auto dim = info[1].As<Napi::Number>().Int64Value();
    return TensorWrap::NewInstance(env, at::log_softmax(self, dim));
  });
}

Napi::Value NormOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& self = TensorWrap::UnwrapTensor(env, info[0]);
    const auto p =
        (info.Length() >= 2 && !info[1].IsUndefined() && !info[1].IsNull())
            ? ParseScalar(env, info[1])
            : c10::Scalar(2.0);
    const auto dim = info.Length() >= 3 ? ParseOptionalIntArray(env, info[2]) : std::nullopt;
    const bool keepdim = info.Length() >= 4 && info[3].ToBoolean().Value();
    if (dim.has_value()) {
      return TensorWrap::NewInstance(env, at::linalg_vector_norm(self, p, *dim, keepdim));
    }
    return TensorWrap::NewInstance(env, at::linalg_vector_norm(self, p));
  });
}

// ---------- Shape / View / Join ----------

Napi::Value ReshapeOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return TensorWrap::NewInstance(
        env, TensorWrap::UnwrapTensor(env, info[0]).reshape(ParseIntArray(env, info[1])));
  });
}

Napi::Value ViewOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return TensorWrap::NewInstance(
        env, TensorWrap::UnwrapTensor(env, info[0]).view(ParseIntArray(env, info[1])));
  });
}

Napi::Value TransposeOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return TensorWrap::NewInstance(
        env,
        TensorWrap::UnwrapTensor(env, info[0])
            .transpose(
                info[1].As<Napi::Number>().Int64Value(), info[2].As<Napi::Number>().Int64Value()));
  });
}

Napi::Value PermuteOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return TensorWrap::NewInstance(
        env, TensorWrap::UnwrapTensor(env, info[0]).permute(ParseIntArray(env, info[1])));
  });
}

Napi::Value SqueezeOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& t = TensorWrap::UnwrapTensor(env, info[0]);
    if (info.Length() >= 2 && !info[1].IsUndefined() && !info[1].IsNull()) {
      return TensorWrap::NewInstance(env, t.squeeze(info[1].As<Napi::Number>().Int64Value()));
    }
    return TensorWrap::NewInstance(env, t.squeeze());
  });
}

Napi::Value UnsqueezeOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return TensorWrap::NewInstance(
        env, TensorWrap::UnwrapTensor(env, info[0]).unsqueeze(info[1].As<Napi::Number>().Int64Value()));
  });
}

Napi::Value FlattenOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& t = TensorWrap::UnwrapTensor(env, info[0]);
    const auto start = info.Length() >= 2 ? info[1].As<Napi::Number>().Int64Value() : 0;
    const auto end = info.Length() >= 3 ? info[2].As<Napi::Number>().Int64Value() : -1;
    return TensorWrap::NewInstance(env, at::flatten(t, start, end));
  });
}

Napi::Value ExpandOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return TensorWrap::NewInstance(
        env, TensorWrap::UnwrapTensor(env, info[0]).expand(ParseIntArray(env, info[1])));
  });
}

Napi::Value CatOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto list = ParseTensorList(env, info[0]);
    const auto dim = info.Length() >= 2 ? info[1].As<Napi::Number>().Int64Value() : 0;
    return TensorWrap::NewInstance(env, at::cat(list, dim));
  });
}

Napi::Value StackOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto list = ParseTensorList(env, info[0]);
    const auto dim = info.Length() >= 2 ? info[1].As<Napi::Number>().Int64Value() : 0;
    return TensorWrap::NewInstance(env, at::stack(list, dim));
  });
}

Napi::Value SplitOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& self = TensorWrap::UnwrapTensor(env, info[0]);
    const auto dim = info.Length() >= 3 ? info[2].As<Napi::Number>().Int64Value() : 0;
    std::vector<at::Tensor> parts;
    if (info[1].IsArray()) {
      parts = at::split_with_sizes(self, ParseIntArray(env, info[1]), dim);
    } else {
      parts = at::split(self, info[1].As<Napi::Number>().Int64Value(), dim);
    }
    auto arr = Napi::Array::New(env, parts.size());
    for (std::size_t i = 0; i < parts.size(); ++i) {
      arr.Set(static_cast<std::uint32_t>(i), TensorWrap::NewInstance(env, std::move(parts[i])));
    }
    return arr;
  });
}

Napi::Value ChunkOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& self = TensorWrap::UnwrapTensor(env, info[0]);
    const auto chunks = info[1].As<Napi::Number>().Int64Value();
    const auto dim = info.Length() >= 3 ? info[2].As<Napi::Number>().Int64Value() : 0;
    auto parts = at::chunk(self, chunks, dim);
    auto arr = Napi::Array::New(env, parts.size());
    for (std::size_t i = 0; i < parts.size(); ++i) {
      arr.Set(static_cast<std::uint32_t>(i), TensorWrap::NewInstance(env, std::move(parts[i])));
    }
    return arr;
  });
}

Napi::Value NarrowOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& self = TensorWrap::UnwrapTensor(env, info[0]);
    const auto dim = info[1].As<Napi::Number>().Int64Value();
    const auto start = info[2].As<Napi::Number>().Int64Value();
    const auto length = info[3].As<Napi::Number>().Int64Value();
    return TensorWrap::NewInstance(env, self.narrow(dim, start, length));
  });
}

Napi::Value MaskedFillOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& self = TensorWrap::UnwrapTensor(env, info[0]);
    const auto& mask = TensorWrap::UnwrapTensor(env, info[1]);
    const auto val = ParseScalar(env, info[2]);
    return TensorWrap::NewInstance(env, self.masked_fill(mask, val));
  });
}

Napi::Value TrilOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& self = TensorWrap::UnwrapTensor(env, info[0]);
    const auto diag = info.Length() >= 2 ? info[1].As<Napi::Number>().Int64Value() : 0;
    return TensorWrap::NewInstance(env, at::tril(self, diag));
  });
}

Napi::Value TriuOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& self = TensorWrap::UnwrapTensor(env, info[0]);
    const auto diag = info.Length() >= 2 ? info[1].As<Napi::Number>().Int64Value() : 0;
    return TensorWrap::NewInstance(env, at::triu(self, diag));
  });
}

// ---------- NN / Loss Kernels ----------

Napi::Value LinearOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto& weight = TensorWrap::UnwrapTensor(env, info[1]);
    const auto bias = info.Length() >= 3 ? ParseOptionalTensor(env, info[2]) : std::nullopt;
    return TensorWrap::NewInstance(env, at::linear(input, weight, bias));
  });
}

Napi::Value Conv1dOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto& weight = TensorWrap::UnwrapTensor(env, info[1]);
    const auto bias = info.Length() >= 3 ? ParseOptionalTensor(env, info[2]) : std::nullopt;
    const auto stride =
        (info.Length() >= 4 && !info[3].IsUndefined()) ? ParseIntArray(env, info[3]) : std::vector<std::int64_t>{1};
    const auto padding =
        (info.Length() >= 5 && !info[4].IsUndefined()) ? ParseIntArray(env, info[4]) : std::vector<std::int64_t>{0};
    const auto dilation =
        (info.Length() >= 6 && !info[5].IsUndefined()) ? ParseIntArray(env, info[5]) : std::vector<std::int64_t>{1};
    const auto groups = info.Length() >= 7 ? info[6].As<Napi::Number>().Int64Value() : 1;
    return TensorWrap::NewInstance(
        env, at::conv1d(input, weight, bias, stride, padding, dilation, groups));
  });
}

Napi::Value Conv2dOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto& weight = TensorWrap::UnwrapTensor(env, info[1]);
    const auto bias = info.Length() >= 3 ? ParseOptionalTensor(env, info[2]) : std::nullopt;
    const auto stride = (info.Length() >= 4 && !info[3].IsUndefined())
                            ? ParseIntArray(env, info[3])
                            : std::vector<std::int64_t>{1, 1};
    const auto padding = (info.Length() >= 5 && !info[4].IsUndefined())
                             ? ParseIntArray(env, info[4])
                             : std::vector<std::int64_t>{0, 0};
    const auto dilation = (info.Length() >= 6 && !info[5].IsUndefined())
                              ? ParseIntArray(env, info[5])
                              : std::vector<std::int64_t>{1, 1};
    const auto groups = info.Length() >= 7 ? info[6].As<Napi::Number>().Int64Value() : 1;
    return TensorWrap::NewInstance(
        env, at::conv2d(input, weight, bias, stride, padding, dilation, groups));
  });
}

Napi::Value MaxPool2dOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto kernel = ParseIntArray(env, info[1]);
    const auto stride = (info.Length() >= 3 && !info[2].IsUndefined() && !info[2].IsNull())
                            ? ParseIntArray(env, info[2])
                            : std::vector<std::int64_t>{};
    const auto padding = (info.Length() >= 4 && !info[3].IsUndefined())
                             ? ParseIntArray(env, info[3])
                             : std::vector<std::int64_t>{0, 0};
    const auto dilation = (info.Length() >= 5 && !info[4].IsUndefined())
                              ? ParseIntArray(env, info[4])
                              : std::vector<std::int64_t>{1, 1};
    const bool ceil_mode = info.Length() >= 6 && info[5].ToBoolean().Value();
    return TensorWrap::NewInstance(
        env, at::max_pool2d(input, kernel, stride, padding, dilation, ceil_mode));
  });
}

Napi::Value AvgPool2dOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto kernel = ParseIntArray(env, info[1]);
    const auto stride = (info.Length() >= 3 && !info[2].IsUndefined() && !info[2].IsNull())
                            ? ParseIntArray(env, info[2])
                            : std::vector<std::int64_t>{};
    const auto padding = (info.Length() >= 4 && !info[3].IsUndefined())
                             ? ParseIntArray(env, info[3])
                             : std::vector<std::int64_t>{0, 0};
    const bool ceil_mode = info.Length() >= 5 && info[4].ToBoolean().Value();
    const bool count_include_pad = info.Length() < 6 || info[5].ToBoolean().Value();
    return TensorWrap::NewInstance(
        env, at::avg_pool2d(input, kernel, stride, padding, ceil_mode, count_include_pad));
  });
}

Napi::Value AdaptiveAvgPool2dOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto output_size = ParseIntArray(env, info[1]);
    return TensorWrap::NewInstance(env, at::adaptive_avg_pool2d(input, output_size));
  });
}

Napi::Value LayerNormOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto normalized_shape = ParseIntArray(env, info[1]);
    const auto weight = info.Length() >= 3 ? ParseOptionalTensor(env, info[2]) : std::nullopt;
    const auto bias = info.Length() >= 4 ? ParseOptionalTensor(env, info[3]) : std::nullopt;
    const double eps = info.Length() >= 5 ? info[4].As<Napi::Number>().DoubleValue() : 1e-5;
    return TensorWrap::NewInstance(
        env, at::layer_norm(input, normalized_shape, weight, bias, eps, /*cudnn_enable=*/true));
  });
}

Napi::Value RmsNormOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto normalized_shape = ParseIntArray(env, info[1]);
    const auto weight = info.Length() >= 3 ? ParseOptionalTensor(env, info[2]) : std::nullopt;
    const auto eps = info.Length() >= 4 ? ParseOptionalDouble(info[3]) : std::nullopt;
    return TensorWrap::NewInstance(env, at::rms_norm(input, normalized_shape, weight, eps));
  });
}

Napi::Value BatchNormOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto weight = ParseOptionalTensor(env, info[1]);
    const auto bias = ParseOptionalTensor(env, info[2]);
    const auto running_mean = ParseOptionalTensor(env, info[3]);
    const auto running_var = ParseOptionalTensor(env, info[4]);
    const bool training = info[5].ToBoolean().Value();
    const double momentum = info.Length() >= 7 ? info[6].As<Napi::Number>().DoubleValue() : 0.1;
    const double eps = info.Length() >= 8 ? info[7].As<Napi::Number>().DoubleValue() : 1e-5;
    return TensorWrap::NewInstance(
        env,
        at::batch_norm(
            input,
            weight,
            bias,
            running_mean,
            running_var,
            training,
            momentum,
            eps,
            /*cudnn_enabled=*/true));
  });
}

Napi::Value GroupNormOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto num_groups = info[1].As<Napi::Number>().Int64Value();
    const auto weight = info.Length() >= 3 ? ParseOptionalTensor(env, info[2]) : std::nullopt;
    const auto bias = info.Length() >= 4 ? ParseOptionalTensor(env, info[3]) : std::nullopt;
    const double eps = info.Length() >= 5 ? info[4].As<Napi::Number>().DoubleValue() : 1e-5;
    return TensorWrap::NewInstance(
        env, at::group_norm(input, num_groups, weight, bias, eps, /*cudnn_enabled=*/true));
  });
}

Napi::Value EmbeddingOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& weight = TensorWrap::UnwrapTensor(env, info[0]);
    const auto& indices = TensorWrap::UnwrapTensor(env, info[1]);
    const auto padding_idx =
        (info.Length() >= 3 && info[2].IsNumber()) ? info[2].As<Napi::Number>().Int64Value() : -1;
    const bool scale_grad_by_freq = info.Length() >= 4 && info[3].ToBoolean().Value();
    const bool sparse = info.Length() >= 5 && info[4].ToBoolean().Value();
    return TensorWrap::NewInstance(
        env, at::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse));
  });
}

Napi::Value DropoutOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const double p = info.Length() >= 2 ? info[1].As<Napi::Number>().DoubleValue() : 0.5;
    const bool train = info.Length() < 3 || info[2].ToBoolean().Value();
    return TensorWrap::NewInstance(env, at::dropout(input, p, train));
  });
}

Napi::Value ScaledDotProductAttentionOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& q = TensorWrap::UnwrapTensor(env, info[0]);
    const auto& k = TensorWrap::UnwrapTensor(env, info[1]);
    const auto& v = TensorWrap::UnwrapTensor(env, info[2]);
    const auto attn_mask = info.Length() >= 4 ? ParseOptionalTensor(env, info[3]) : std::nullopt;
    const double dropout_p = info.Length() >= 5 ? info[4].As<Napi::Number>().DoubleValue() : 0.0;
    const bool is_causal = info.Length() >= 6 && info[5].ToBoolean().Value();
    const auto scale = info.Length() >= 7 ? ParseOptionalDouble(info[6]) : std::nullopt;
    const bool enable_gqa = info.Length() >= 8 && info[7].ToBoolean().Value();
    return TensorWrap::NewInstance(
        env,
        at::scaled_dot_product_attention(
            q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa));
  });
}

[[nodiscard]] std::int64_t ParseReductionMode(Napi::Env env, const Napi::Value& val) {
  if (val.IsUndefined() || val.IsNull()) return static_cast<std::int64_t>(at::Reduction::Mean);
  const std::string s = val.As<Napi::String>().Utf8Value();
  if (s == "none") return static_cast<std::int64_t>(at::Reduction::None);
  if (s == "mean") return static_cast<std::int64_t>(at::Reduction::Mean);
  if (s == "sum") return static_cast<std::int64_t>(at::Reduction::Sum);
  throw Napi::RangeError::New(env, "Unknown reduction mode: " + s);
}

Napi::Value CrossEntropyOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto& target = TensorWrap::UnwrapTensor(env, info[1]);
    const auto weight = info.Length() >= 3 ? ParseOptionalTensor(env, info[2]) : std::nullopt;
    const auto reduction = info.Length() >= 4
                               ? ParseReductionMode(env, info[3])
                               : static_cast<std::int64_t>(at::Reduction::Mean);
    const auto ignore_index =
        (info.Length() >= 5 && info[4].IsNumber()) ? info[4].As<Napi::Number>().Int64Value() : -100;
    const double label_smoothing =
        (info.Length() >= 6 && info[5].IsNumber()) ? info[5].As<Napi::Number>().DoubleValue() : 0.0;
    return TensorWrap::NewInstance(
        env, at::cross_entropy_loss(input, target, weight, reduction, ignore_index, label_smoothing));
  });
}

Napi::Value NllLossOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto& target = TensorWrap::UnwrapTensor(env, info[1]);
    const auto weight = info.Length() >= 3 ? ParseOptionalTensor(env, info[2]) : std::nullopt;
    const auto reduction = info.Length() >= 4
                               ? ParseReductionMode(env, info[3])
                               : static_cast<std::int64_t>(at::Reduction::Mean);
    const auto ignore_index =
        (info.Length() >= 5 && info[4].IsNumber()) ? info[4].As<Napi::Number>().Int64Value() : -100;
    return TensorWrap::NewInstance(env, at::nll_loss_nd(input, target, weight, reduction, ignore_index));
  });
}

Napi::Value MseLossOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto& target = TensorWrap::UnwrapTensor(env, info[1]);
    const auto reduction = info.Length() >= 3
                               ? ParseReductionMode(env, info[2])
                               : static_cast<std::int64_t>(at::Reduction::Mean);
    return TensorWrap::NewInstance(env, at::mse_loss(input, target, reduction));
  });
}

Napi::Value L1LossOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto& target = TensorWrap::UnwrapTensor(env, info[1]);
    const auto reduction = info.Length() >= 3
                               ? ParseReductionMode(env, info[2])
                               : static_cast<std::int64_t>(at::Reduction::Mean);
    return TensorWrap::NewInstance(env, at::l1_loss(input, target, reduction));
  });
}

Napi::Value SmoothL1LossOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto& target = TensorWrap::UnwrapTensor(env, info[1]);
    const auto reduction = info.Length() >= 3
                               ? ParseReductionMode(env, info[2])
                               : static_cast<std::int64_t>(at::Reduction::Mean);
    const double beta = info.Length() >= 4 ? info[3].As<Napi::Number>().DoubleValue() : 1.0;
    return TensorWrap::NewInstance(env, at::smooth_l1_loss(input, target, reduction, beta));
  });
}

Napi::Value BceWithLogitsLossOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto& target = TensorWrap::UnwrapTensor(env, info[1]);
    const auto weight = info.Length() >= 3 ? ParseOptionalTensor(env, info[2]) : std::nullopt;
    const auto pos_weight = info.Length() >= 4 ? ParseOptionalTensor(env, info[3]) : std::nullopt;
    const auto reduction = info.Length() >= 5
                               ? ParseReductionMode(env, info[4])
                               : static_cast<std::int64_t>(at::Reduction::Mean);
    return TensorWrap::NewInstance(
        env, at::binary_cross_entropy_with_logits(input, target, weight, pos_weight, reduction));
  });
}

Napi::Value KlDivOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto& target = TensorWrap::UnwrapTensor(env, info[1]);
    const auto reduction = info.Length() >= 3
                               ? ParseReductionMode(env, info[2])
                               : static_cast<std::int64_t>(at::Reduction::Mean);
    const bool log_target = info.Length() >= 4 && info[3].ToBoolean().Value();
    return TensorWrap::NewInstance(env, at::kl_div(input, target, reduction, log_target));
  });
}

Napi::Value HuberLossOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto& target = TensorWrap::UnwrapTensor(env, info[1]);
    const auto reduction = info.Length() >= 3
                               ? ParseReductionMode(env, info[2])
                               : static_cast<std::int64_t>(at::Reduction::Mean);
    const double delta = info.Length() >= 4 ? info[3].As<Napi::Number>().DoubleValue() : 1.0;
    return TensorWrap::NewInstance(env, at::huber_loss(input, target, reduction, delta));
  });
}

Napi::Value CosineSimilarityOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& x1 = TensorWrap::UnwrapTensor(env, info[0]);
    const auto& x2 = TensorWrap::UnwrapTensor(env, info[1]);
    const auto dim = info.Length() >= 3 ? info[2].As<Napi::Number>().Int64Value() : 1;
    const double eps = info.Length() >= 4 ? info[3].As<Napi::Number>().DoubleValue() : 1e-8;
    return TensorWrap::NewInstance(env, at::cosine_similarity(x1, x2, dim, eps));
  });
}

Napi::Value Conv3dOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto& weight = TensorWrap::UnwrapTensor(env, info[1]);
    const auto bias = info.Length() >= 3 ? ParseOptionalTensor(env, info[2]) : std::nullopt;
    const auto stride = (info.Length() >= 4 && !info[3].IsUndefined())
                            ? ParseIntArray(env, info[3])
                            : std::vector<std::int64_t>{1, 1, 1};
    const auto padding = (info.Length() >= 5 && !info[4].IsUndefined())
                             ? ParseIntArray(env, info[4])
                             : std::vector<std::int64_t>{0, 0, 0};
    const auto dilation = (info.Length() >= 6 && !info[5].IsUndefined())
                              ? ParseIntArray(env, info[5])
                              : std::vector<std::int64_t>{1, 1, 1};
    const auto groups = info.Length() >= 7 ? info[6].As<Napi::Number>().Int64Value() : 1;
    return TensorWrap::NewInstance(
        env, at::conv3d(input, weight, bias, stride, padding, dilation, groups));
  });
}

Napi::Value ConvTranspose1dOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto& weight = TensorWrap::UnwrapTensor(env, info[1]);
    const auto bias = info.Length() >= 3 ? ParseOptionalTensor(env, info[2]) : std::nullopt;
    const auto stride = (info.Length() >= 4 && !info[3].IsUndefined())
                            ? ParseIntArray(env, info[3])
                            : std::vector<std::int64_t>{1};
    const auto padding = (info.Length() >= 5 && !info[4].IsUndefined())
                             ? ParseIntArray(env, info[4])
                             : std::vector<std::int64_t>{0};
    const auto output_padding = (info.Length() >= 6 && !info[5].IsUndefined())
                                    ? ParseIntArray(env, info[5])
                                    : std::vector<std::int64_t>{0};
    const auto groups = info.Length() >= 7 ? info[6].As<Napi::Number>().Int64Value() : 1;
    const auto dilation = (info.Length() >= 8 && !info[7].IsUndefined())
                              ? ParseIntArray(env, info[7])
                              : std::vector<std::int64_t>{1};
    return TensorWrap::NewInstance(
        env,
        at::conv_transpose1d(
            input, weight, bias, stride, padding, output_padding, groups, dilation));
  });
}

Napi::Value ConvTranspose2dOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto& weight = TensorWrap::UnwrapTensor(env, info[1]);
    const auto bias = info.Length() >= 3 ? ParseOptionalTensor(env, info[2]) : std::nullopt;
    const auto stride = (info.Length() >= 4 && !info[3].IsUndefined())
                            ? ParseIntArray(env, info[3])
                            : std::vector<std::int64_t>{1, 1};
    const auto padding = (info.Length() >= 5 && !info[4].IsUndefined())
                             ? ParseIntArray(env, info[4])
                             : std::vector<std::int64_t>{0, 0};
    const auto output_padding = (info.Length() >= 6 && !info[5].IsUndefined())
                                    ? ParseIntArray(env, info[5])
                                    : std::vector<std::int64_t>{0, 0};
    const auto groups = info.Length() >= 7 ? info[6].As<Napi::Number>().Int64Value() : 1;
    const auto dilation = (info.Length() >= 8 && !info[7].IsUndefined())
                              ? ParseIntArray(env, info[7])
                              : std::vector<std::int64_t>{1, 1};
    return TensorWrap::NewInstance(
        env,
        at::conv_transpose2d(
            input, weight, bias, stride, padding, output_padding, groups, dilation));
  });
}

Napi::Value Upsample2dOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto out_size = ParseIntArray(env, info[1]);
    const std::string mode =
        (info.Length() >= 3 && info[2].IsString()) ? info[2].As<Napi::String>().Utf8Value() : "nearest";
    const bool align_corners = info.Length() >= 4 && info[3].ToBoolean().Value();
    if (mode == "nearest") {
      return TensorWrap::NewInstance(env, at::upsample_nearest2d(input, out_size));
    }
    if (mode == "bilinear") {
      return TensorWrap::NewInstance(env, at::upsample_bilinear2d(input, out_size, align_corners));
    }
    if (mode == "bicubic") {
      return TensorWrap::NewInstance(env, at::upsample_bicubic2d(input, out_size, align_corners));
    }
    throw Napi::RangeError::New(env, "Unsupported upsample mode: " + mode);
  });
}

Napi::Value PadOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto pad = ParseIntArray(env, info[1]);
    const std::string mode =
        (info.Length() >= 3 && info[2].IsString()) ? info[2].As<Napi::String>().Utf8Value() : "constant";
    const auto value = info.Length() >= 4 ? ParseOptionalDouble(info[3]) : std::nullopt;
    return TensorWrap::NewInstance(env, at::pad(input, pad, mode, value));
  });
}

Napi::Value OneHotOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto num_classes = info.Length() >= 2 ? info[1].As<Napi::Number>().Int64Value() : -1;
    return TensorWrap::NewInstance(env, at::one_hot(input, num_classes));
  });
}

Napi::Value LstmOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto hx = ParseTensorList(env, info[1]);
    const auto params = ParseTensorList(env, info[2]);
    const bool has_biases = info[3].ToBoolean().Value();
    const auto num_layers = info[4].As<Napi::Number>().Int64Value();
    const double dropout = info[5].As<Napi::Number>().DoubleValue();
    const bool train = info[6].ToBoolean().Value();
    const bool bidirectional = info[7].ToBoolean().Value();
    const bool batch_first = info[8].ToBoolean().Value();
    auto [output, hn, cn] =
        at::lstm(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
    auto res = Napi::Object::New(env);
    res.Set("output", TensorWrap::NewInstance(env, std::move(output)));
    res.Set("hn", TensorWrap::NewInstance(env, std::move(hn)));
    res.Set("cn", TensorWrap::NewInstance(env, std::move(cn)));
    return res;
  });
}

Napi::Value GruOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto& hx = TensorWrap::UnwrapTensor(env, info[1]);
    const auto params = ParseTensorList(env, info[2]);
    const bool has_biases = info[3].ToBoolean().Value();
    const auto num_layers = info[4].As<Napi::Number>().Int64Value();
    const double dropout = info[5].As<Napi::Number>().DoubleValue();
    const bool train = info[6].ToBoolean().Value();
    const bool bidirectional = info[7].ToBoolean().Value();
    const bool batch_first = info[8].ToBoolean().Value();
    auto [output, hn] =
        at::gru(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
    auto res = Napi::Object::New(env);
    res.Set("output", TensorWrap::NewInstance(env, std::move(output)));
    res.Set("hn", TensorWrap::NewInstance(env, std::move(hn)));
    return res;
  });
}

Napi::Value RnnOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto& hx = TensorWrap::UnwrapTensor(env, info[1]);
    const auto params = ParseTensorList(env, info[2]);
    const bool has_biases = info[3].ToBoolean().Value();
    const auto num_layers = info[4].As<Napi::Number>().Int64Value();
    const double dropout = info[5].As<Napi::Number>().DoubleValue();
    const bool train = info[6].ToBoolean().Value();
    const bool bidirectional = info[7].ToBoolean().Value();
    const bool batch_first = info[8].ToBoolean().Value();
    const std::string nonlinearity =
        (info.Length() >= 10 && info[9].IsString()) ? info[9].As<Napi::String>().Utf8Value() : "tanh";
    auto [output, hn] =
        (nonlinearity == "relu")
            ? at::rnn_relu(
                  input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first)
            : at::rnn_tanh(
                  input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
    auto res = Napi::Object::New(env);
    res.Set("output", TensorWrap::NewInstance(env, std::move(output)));
    res.Set("hn", TensorWrap::NewInstance(env, std::move(hn)));
    return res;
  });
}

Napi::Value FusedAdamStepOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    auto params = ParseTensorList(env, info[0]);
    auto grads = ParseTensorList(env, info[1]);
    auto exp_avgs = ParseTensorList(env, info[2]);
    auto exp_avg_sqs = ParseTensorList(env, info[3]);
    auto max_exp_avg_sqs = ParseTensorList(env, info[4]);
    auto state_steps = ParseTensorList(env, info[5]);
    const double lr = info[6].As<Napi::Number>().DoubleValue();
    const double beta1 = info[7].As<Napi::Number>().DoubleValue();
    const double beta2 = info[8].As<Napi::Number>().DoubleValue();
    const double weight_decay = info[9].As<Napi::Number>().DoubleValue();
    const double eps = info[10].As<Napi::Number>().DoubleValue();
    const bool amsgrad = info[11].ToBoolean().Value();
    const bool decoupled = info[12].ToBoolean().Value();
    const auto grad_scale = info.Length() >= 14 ? ParseOptionalTensor(env, info[13]) : std::nullopt;
    const auto found_inf = info.Length() >= 15 ? ParseOptionalTensor(env, info[14]) : std::nullopt;
    if (decoupled) {
      at::_fused_adamw_(
          params,
          grads,
          exp_avgs,
          exp_avg_sqs,
          max_exp_avg_sqs,
          state_steps,
          lr,
          beta1,
          beta2,
          weight_decay,
          eps,
          amsgrad,
          /*maximize=*/false,
          grad_scale,
          found_inf);
    } else {
      at::_fused_adam_(
          params,
          grads,
          exp_avgs,
          exp_avg_sqs,
          max_exp_avg_sqs,
          state_steps,
          lr,
          beta1,
          beta2,
          weight_decay,
          eps,
          amsgrad,
          /*maximize=*/false,
          grad_scale,
          found_inf);
    }
    return env.Undefined();
  });
}

Napi::Value ComplexOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return TensorWrap::NewInstance(
        env, at::complex(TensorWrap::UnwrapTensor(env, info[0]), TensorWrap::UnwrapTensor(env, info[1])));
  });
}

Napi::Value PolarOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return TensorWrap::NewInstance(
        env, at::polar(TensorWrap::UnwrapTensor(env, info[0]), TensorWrap::UnwrapTensor(env, info[1])));
  });
}

Napi::Value AngleOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return TensorWrap::NewInstance(env, at::angle(TensorWrap::UnwrapTensor(env, info[0])));
  });
}

Napi::Value ViewAsRealOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return TensorWrap::NewInstance(env, at::view_as_real(TensorWrap::UnwrapTensor(env, info[0])));
  });
}

Napi::Value ViewAsComplexOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return TensorWrap::NewInstance(env, at::view_as_complex(TensorWrap::UnwrapTensor(env, info[0])));
  });
}

Napi::Value SparseCooTensorOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& indices = TensorWrap::UnwrapTensor(env, info[0]);
    const auto& values = TensorWrap::UnwrapTensor(env, info[1]);
    if (info.Length() >= 3 && !info[2].IsUndefined() && !info[2].IsNull()) {
      const auto size = ParseIntArray(env, info[2]);
      return TensorWrap::NewInstance(env, at::sparse_coo_tensor(indices, values, size));
    }
    return TensorWrap::NewInstance(env, at::sparse_coo_tensor(indices, values));
  });
}

Napi::Value SparseMmOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return TensorWrap::NewInstance(
        env, at::mm(TensorWrap::UnwrapTensor(env, info[0]), TensorWrap::UnwrapTensor(env, info[1])));
  });
}

Napi::Value SparseCsrTensorOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& crow = TensorWrap::UnwrapTensor(env, info[0]);
    const auto& col = TensorWrap::UnwrapTensor(env, info[1]);
    const auto& vals = TensorWrap::UnwrapTensor(env, info[2]);
    const auto opts = at::TensorOptions().dtype(vals.scalar_type()).device(vals.device());
    if (info.Length() >= 4 && !info[3].IsUndefined() && !info[3].IsNull()) {
      return TensorWrap::NewInstance(
          env, at::sparse_csr_tensor(crow, col, vals, ParseIntArray(env, info[3]), opts));
    }
    return TensorWrap::NewInstance(env, at::sparse_csr_tensor(crow, col, vals, opts));
  });
}

Napi::Value AmpUnscaleAndCheckOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    auto tensors = ParseTensorList(env, info[0]);
    auto& found_inf = TensorWrap::UnwrapMutableTensor(env, info[1]);
    const auto& inv_scale = TensorWrap::UnwrapTensor(env, info[2]);
    at::_amp_foreach_non_finite_check_and_unscale_(tensors, found_inf, inv_scale);
    return env.Undefined();
  });
}

Napi::Value AmpUpdateScaleOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    auto& current_scale = TensorWrap::UnwrapMutableTensor(env, info[0]);
    auto& growth_tracker = TensorWrap::UnwrapMutableTensor(env, info[1]);
    const auto& found_inf = TensorWrap::UnwrapTensor(env, info[2]);
    const double growth_factor = info[3].As<Napi::Number>().DoubleValue();
    const double backoff_factor = info[4].As<Napi::Number>().DoubleValue();
    const auto growth_interval = info[5].As<Napi::Number>().Int64Value();
    at::_amp_update_scale_(
        current_scale, growth_tracker, found_inf, growth_factor, backoff_factor, growth_interval);
    return info[0];
  });
}

[[nodiscard]] at::Tensor ZeropowerViaNewtonSchulz(
    const at::Tensor& grad,
    double ns_a,
    double ns_b,
    double ns_c,
    std::int64_t ns_steps,
    double eps) {
  TORCH_CHECK(grad.dim() == 2, "Muon Newton-Schulz expects a 2D matrix, got dim=", grad.dim());
  TORCH_CHECK(ns_steps >= 0 && ns_steps < 100, "nsSteps must be in [0, 99], got ", ns_steps);

  at::Tensor ortho = grad.to(at::kBFloat16, /*non_blocking=*/false, /*copy=*/true);
  const bool transposed = grad.size(0) > grad.size(1);
  if (transposed) {
    ortho = ortho.t();
  }
  ortho.div_(ortho.norm().clamp_min(eps));
  for (std::int64_t step = 0; step < ns_steps; ++step) {
    const auto gram = at::mm(ortho, ortho.t());
    const auto gram_update = at::addmm(gram, gram, gram, /*beta=*/ns_b, /*alpha=*/ns_c);
    ortho = at::addmm(ortho, gram_update, ortho, /*beta=*/ns_a, /*alpha=*/1.0);
  }
  if (transposed) {
    ortho = ortho.t();
  }
  return ortho;
}

Napi::Value ZeropowerViaNewtonSchulzOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const c10::NoGradGuard no_grad;
    const auto& grad = TensorWrap::UnwrapTensor(env, info[0]);
    const double ns_a = info.Length() >= 2 ? info[1].As<Napi::Number>().DoubleValue() : 3.4445;
    const double ns_b = info.Length() >= 3 ? info[2].As<Napi::Number>().DoubleValue() : -4.7750;
    const double ns_c = info.Length() >= 4 ? info[3].As<Napi::Number>().DoubleValue() : 2.0315;
    const auto ns_steps = info.Length() >= 5 ? info[4].As<Napi::Number>().Int64Value() : 5;
    const double eps = info.Length() >= 6 ? info[5].As<Napi::Number>().DoubleValue() : 1e-7;
    return TensorWrap::NewInstance(
        env, ZeropowerViaNewtonSchulz(grad, ns_a, ns_b, ns_c, ns_steps, eps));
  });
}

Napi::Value MuonStepOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const c10::NoGradGuard no_grad;
    auto params = ParseTensorList(env, info[0]);
    auto grads = ParseTensorList(env, info[1]);
    auto momentum_bufs = ParseTensorList(env, info[2]);
    const double lr = info[3].As<Napi::Number>().DoubleValue();
    const double weight_decay = info[4].As<Napi::Number>().DoubleValue();
    const double momentum = info[5].As<Napi::Number>().DoubleValue();
    const bool nesterov = info[6].ToBoolean().Value();
    const double ns_a = info[7].As<Napi::Number>().DoubleValue();
    const double ns_b = info[8].As<Napi::Number>().DoubleValue();
    const double ns_c = info[9].As<Napi::Number>().DoubleValue();
    const auto ns_steps = info[10].As<Napi::Number>().Int64Value();
    const double eps = info[11].As<Napi::Number>().DoubleValue();
    const std::string adjust_lr_fn =
        (info.Length() >= 13 && info[12].IsString()) ? info[12].As<Napi::String>().Utf8Value()
                                                     : "original";

    TORCH_CHECK(
        params.size() == grads.size() && params.size() == momentum_bufs.size(),
        "muonStep: params, grads, and momentumBufs must have identical lengths");

    for (std::size_t i = 0; i < params.size(); ++i) {
      auto& param = params[i];
      const auto& grad = grads[i];
      auto& buf = momentum_bufs[i];
      TORCH_CHECK(param.dim() == 2 && grad.dim() == 2, "Muon only supports 2D parameters");
      TORCH_CHECK(!param.is_complex(), "Muon does not support complex parameters");
      TORCH_CHECK(!grad.is_sparse(), "Muon does not support sparse gradients");

      buf.lerp_(grad, 1.0 - momentum);
      const at::Tensor update_in = nesterov ? at::lerp(grad, buf, momentum) : buf;
      const at::Tensor ortho =
          ZeropowerViaNewtonSchulz(update_in, ns_a, ns_b, ns_c, ns_steps, eps);

      const auto rows = static_cast<double>(param.size(0));
      const auto cols = static_cast<double>(param.size(1));
      double adjusted_ratio = 1.0;
      if (adjust_lr_fn.empty() || adjust_lr_fn == "original") {
        adjusted_ratio = std::sqrt(std::max(1.0, rows / cols));
      } else if (adjust_lr_fn == "match_rms_adamw") {
        adjusted_ratio = 0.2 * std::sqrt(std::max(rows, cols));
      } else if (adjust_lr_fn == "spectral_unclamped") {
        adjusted_ratio = std::sqrt(rows / cols);
      } else {
        throw Napi::RangeError::New(env, "Unsupported Muon adjustLrFn: " + adjust_lr_fn);
      }

      const double adjusted_lr = lr * adjusted_ratio;
      param.mul_(1.0 - lr * weight_decay);
      param.add_(ortho, /*alpha=*/-adjusted_lr);
    }
    return env.Undefined();
  });
}

Napi::Value ConvTranspose3dOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto& weight = TensorWrap::UnwrapTensor(env, info[1]);
    const auto bias = info.Length() >= 3 ? ParseOptionalTensor(env, info[2]) : std::nullopt;
    const auto stride = (info.Length() >= 4 && !info[3].IsUndefined())
                            ? ParseIntArray(env, info[3])
                            : std::vector<std::int64_t>{1, 1, 1};
    const auto padding = (info.Length() >= 5 && !info[4].IsUndefined())
                             ? ParseIntArray(env, info[4])
                             : std::vector<std::int64_t>{0, 0, 0};
    const auto output_padding = (info.Length() >= 6 && !info[5].IsUndefined())
                                    ? ParseIntArray(env, info[5])
                                    : std::vector<std::int64_t>{0, 0, 0};
    const auto groups = info.Length() >= 7 ? info[6].As<Napi::Number>().Int64Value() : 1;
    const auto dilation = (info.Length() >= 8 && !info[7].IsUndefined())
                              ? ParseIntArray(env, info[7])
                              : std::vector<std::int64_t>{1, 1, 1};
    return TensorWrap::NewInstance(
        env,
        at::conv_transpose3d(
            input, weight, bias, stride, padding, output_padding, groups, dilation));
  });
}

Napi::Value MaxPool1dOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto kernel = ParseIntArray(env, info[1]);
    const auto stride = (info.Length() >= 3 && !info[2].IsUndefined() && !info[2].IsNull())
                            ? ParseIntArray(env, info[2])
                            : std::vector<std::int64_t>{};
    const auto padding = (info.Length() >= 4 && !info[3].IsUndefined())
                             ? ParseIntArray(env, info[3])
                             : std::vector<std::int64_t>{0};
    const auto dilation = (info.Length() >= 5 && !info[4].IsUndefined())
                              ? ParseIntArray(env, info[4])
                              : std::vector<std::int64_t>{1};
    const bool ceil_mode = info.Length() >= 6 && info[5].ToBoolean().Value();
    return TensorWrap::NewInstance(
        env, at::max_pool1d(input, kernel, stride, padding, dilation, ceil_mode));
  });
}

Napi::Value MaxPool3dOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto kernel = ParseIntArray(env, info[1]);
    const auto stride = (info.Length() >= 3 && !info[2].IsUndefined() && !info[2].IsNull())
                            ? ParseIntArray(env, info[2])
                            : std::vector<std::int64_t>{};
    const auto padding = (info.Length() >= 4 && !info[3].IsUndefined())
                             ? ParseIntArray(env, info[3])
                             : std::vector<std::int64_t>{0, 0, 0};
    const auto dilation = (info.Length() >= 5 && !info[4].IsUndefined())
                              ? ParseIntArray(env, info[4])
                              : std::vector<std::int64_t>{1, 1, 1};
    const bool ceil_mode = info.Length() >= 6 && info[5].ToBoolean().Value();
    return TensorWrap::NewInstance(
        env, at::max_pool3d(input, kernel, stride, padding, dilation, ceil_mode));
  });
}

Napi::Value AvgPool1dOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto kernel = ParseIntArray(env, info[1]);
    const auto stride = (info.Length() >= 3 && !info[2].IsUndefined() && !info[2].IsNull())
                            ? ParseIntArray(env, info[2])
                            : std::vector<std::int64_t>{};
    const auto padding = (info.Length() >= 4 && !info[3].IsUndefined())
                             ? ParseIntArray(env, info[3])
                             : std::vector<std::int64_t>{0};
    const bool ceil_mode = info.Length() >= 5 && info[4].ToBoolean().Value();
    const bool count_include_pad = info.Length() < 6 || info[5].ToBoolean().Value();
    return TensorWrap::NewInstance(
        env, at::avg_pool1d(input, kernel, stride, padding, ceil_mode, count_include_pad));
  });
}

Napi::Value AvgPool3dOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto kernel = ParseIntArray(env, info[1]);
    const auto stride = (info.Length() >= 3 && !info[2].IsUndefined() && !info[2].IsNull())
                            ? ParseIntArray(env, info[2])
                            : std::vector<std::int64_t>{};
    const auto padding = (info.Length() >= 4 && !info[3].IsUndefined())
                             ? ParseIntArray(env, info[3])
                             : std::vector<std::int64_t>{0, 0, 0};
    const bool ceil_mode = info.Length() >= 5 && info[4].ToBoolean().Value();
    const bool count_include_pad = info.Length() < 6 || info[5].ToBoolean().Value();
    return TensorWrap::NewInstance(
        env, at::avg_pool3d(input, kernel, stride, padding, ceil_mode, count_include_pad));
  });
}

Napi::Value AdaptiveMaxPool2dOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto out_size = ParseIntArray(env, info[1]);
    auto [out, indices] = at::adaptive_max_pool2d(input, out_size);
    const bool return_indices = info.Length() >= 3 && info[2].ToBoolean().Value();
    if (return_indices) {
      auto res = Napi::Object::New(env);
      res.Set("output", TensorWrap::NewInstance(env, std::move(out)));
      res.Set("indices", TensorWrap::NewInstance(env, std::move(indices)));
      return res;
    }
    return TensorWrap::NewInstance(env, std::move(out));
  });
}

Napi::Value InstanceNormOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto weight = ParseOptionalTensor(env, info[1]);
    const auto bias = ParseOptionalTensor(env, info[2]);
    const auto running_mean = ParseOptionalTensor(env, info[3]);
    const auto running_var = ParseOptionalTensor(env, info[4]);
    const bool use_input_stats = info[5].ToBoolean().Value();
    const double momentum = info.Length() >= 7 ? info[6].As<Napi::Number>().DoubleValue() : 0.1;
    const double eps = info.Length() >= 8 ? info[7].As<Napi::Number>().DoubleValue() : 1e-5;
    return TensorWrap::NewInstance(
        env,
        at::instance_norm(
            input,
            weight,
            bias,
            running_mean,
            running_var,
            use_input_stats,
            momentum,
            eps,
            /*cudnn_enabled=*/true));
  });
}

Napi::Value BilinearOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& x1 = TensorWrap::UnwrapTensor(env, info[0]);
    const auto& x2 = TensorWrap::UnwrapTensor(env, info[1]);
    const auto& w = TensorWrap::UnwrapTensor(env, info[2]);
    const auto b = info.Length() >= 4 ? ParseOptionalTensor(env, info[3]) : std::nullopt;
    return TensorWrap::NewInstance(env, at::bilinear(x1, x2, w, b));
  });
}

Napi::Value PixelShuffleOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return TensorWrap::NewInstance(
        env,
        at::pixel_shuffle(
            TensorWrap::UnwrapTensor(env, info[0]), info[1].As<Napi::Number>().Int64Value()));
  });
}

Napi::Value PixelUnshuffleOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return TensorWrap::NewInstance(
        env,
        at::pixel_unshuffle(
            TensorWrap::UnwrapTensor(env, info[0]), info[1].As<Napi::Number>().Int64Value()));
  });
}

Napi::Value Im2ColOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto kernel = ParseIntArray(env, info[1]);
    const auto dilation = (info.Length() >= 3 && !info[2].IsUndefined())
                              ? ParseIntArray(env, info[2])
                              : std::vector<std::int64_t>{1, 1};
    const auto padding = (info.Length() >= 4 && !info[3].IsUndefined())
                             ? ParseIntArray(env, info[3])
                             : std::vector<std::int64_t>{0, 0};
    const auto stride = (info.Length() >= 5 && !info[4].IsUndefined())
                            ? ParseIntArray(env, info[4])
                            : std::vector<std::int64_t>{1, 1};
    return TensorWrap::NewInstance(env, at::im2col(input, kernel, dilation, padding, stride));
  });
}

Napi::Value Col2ImOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& input = TensorWrap::UnwrapTensor(env, info[0]);
    const auto output_size = ParseIntArray(env, info[1]);
    const auto kernel = ParseIntArray(env, info[2]);
    const auto dilation = (info.Length() >= 4 && !info[3].IsUndefined())
                              ? ParseIntArray(env, info[3])
                              : std::vector<std::int64_t>{1, 1};
    const auto padding = (info.Length() >= 5 && !info[4].IsUndefined())
                             ? ParseIntArray(env, info[4])
                             : std::vector<std::int64_t>{0, 0};
    const auto stride = (info.Length() >= 6 && !info[5].IsUndefined())
                            ? ParseIntArray(env, info[5])
                            : std::vector<std::int64_t>{1, 1};
    return TensorWrap::NewInstance(
        env, at::col2im(input, output_size, kernel, dilation, padding, stride));
  });
}

Napi::Value EmbeddingBagOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& weight = TensorWrap::UnwrapTensor(env, info[0]);
    const auto& indices = TensorWrap::UnwrapTensor(env, info[1]);
    const auto& offsets = TensorWrap::UnwrapTensor(env, info[2]);
    const bool scale_grad_by_freq = info.Length() >= 4 && info[3].ToBoolean().Value();
    const std::string mode_str =
        (info.Length() >= 5 && info[4].IsString()) ? info[4].As<Napi::String>().Utf8Value() : "mean";
    std::int64_t mode = 1;
    if (mode_str == "sum") {
      mode = 0;
    } else if (mode_str == "mean") {
      mode = 1;
    } else if (mode_str == "max") {
      mode = 2;
    } else {
      throw Napi::RangeError::New(env, "Unsupported EmbeddingBag mode: " + mode_str);
    }
    const bool sparse = info.Length() >= 6 && info[5].ToBoolean().Value();
    const auto per_sample_weights = info.Length() >= 7 ? ParseOptionalTensor(env, info[6]) : std::nullopt;
    const bool include_last_offset = info.Length() >= 8 && info[7].ToBoolean().Value();
    std::optional<std::int64_t> padding_idx = std::nullopt;
    if (info.Length() >= 9 && info[8].IsNumber()) {
      padding_idx = info[8].As<Napi::Number>().Int64Value();
    }
    auto [out, offset2bag, bag_size, max_indices] = at::embedding_bag(
        weight,
        indices,
        offsets,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        include_last_offset,
        padding_idx);
    return TensorWrap::NewInstance(env, std::move(out));
  });
}

}  // namespace

void RegisterFastOps(Napi::Env env, Napi::Object exports) {
  auto ops = Napi::Object::New(env);
  // Factories
  ops.Set("tensor", Napi::Function::New(env, TensorFromData));
  ops.Set("empty", Napi::Function::New(env, EmptyOp));
  ops.Set("zeros", Napi::Function::New(env, ZerosOp));
  ops.Set("ones", Napi::Function::New(env, OnesOp));
  ops.Set("full", Napi::Function::New(env, FullOp));
  ops.Set("rand", Napi::Function::New(env, RandOp));
  ops.Set("randn", Napi::Function::New(env, RandnOp));
  ops.Set("randint", Napi::Function::New(env, RandintOp));
  ops.Set("randperm", Napi::Function::New(env, RandpermOp));
  ops.Set("arange", Napi::Function::New(env, ArangeOp));
  ops.Set("linspace", Napi::Function::New(env, LinspaceOp));
  ops.Set("eye", Napi::Function::New(env, EyeOp));
  ops.Set("zerosLike", Napi::Function::New(env, ZerosLikeOp));
  ops.Set("onesLike", Napi::Function::New(env, OnesLikeOp));
  ops.Set("emptyLike", Napi::Function::New(env, EmptyLikeOp));
  ops.Set("fullLike", Napi::Function::New(env, FullLikeOp));
  ops.Set("randnLike", Napi::Function::New(env, RandnLikeOp));

  // Math
  ops.Set("add", Napi::Function::New(env, AddOp));
  ops.Set("sub", Napi::Function::New(env, SubOp));
  ops.Set("mul", Napi::Function::New(env, MulOp));
  ops.Set("div", Napi::Function::New(env, DivOp));
  ops.Set("pow", Napi::Function::New(env, PowOp));
  ops.Set("matmul", Napi::Function::New(env, MatmulOp));
  ops.Set("bmm", Napi::Function::New(env, BmmOp));
  ops.Set("einsum", Napi::Function::New(env, EinsumOp));
  ops.Set("neg", Napi::Function::New(env, NegOp));
  ops.Set("abs", Napi::Function::New(env, AbsOp));
  ops.Set("exp", Napi::Function::New(env, ExpOp));
  ops.Set("log", Napi::Function::New(env, LogOp));
  ops.Set("sqrt", Napi::Function::New(env, SqrtOp));
  ops.Set("rsqrt", Napi::Function::New(env, RsqrtOp));
  ops.Set("sin", Napi::Function::New(env, SinOp));
  ops.Set("cos", Napi::Function::New(env, CosOp));
  ops.Set("tanh", Napi::Function::New(env, TanhOp));
  ops.Set("sigmoid", Napi::Function::New(env, SigmoidOp));
  ops.Set("relu", Napi::Function::New(env, ReluOp));
  ops.Set("gelu", Napi::Function::New(env, GeluOp));
  ops.Set("silu", Napi::Function::New(env, SiluOp));
  ops.Set("leakyRelu", Napi::Function::New(env, LeakyReluOp));
  ops.Set("clamp", Napi::Function::New(env, ClampOp));
  ops.Set("where", Napi::Function::New(env, WhereOp));
  ops.Set("isnan", Napi::Function::New(env, IsnanOp));
  ops.Set("isinf", Napi::Function::New(env, IsinfOp));
  ops.Set("isfinite", Napi::Function::New(env, IsfiniteOp));

  // In-place
  ops.Set("add_", Napi::Function::New(env, AddInplaceOp));
  ops.Set("sub_", Napi::Function::New(env, SubInplaceOp));
  ops.Set("mul_", Napi::Function::New(env, MulInplaceOp));
  ops.Set("div_", Napi::Function::New(env, DivInplaceOp));
  ops.Set("zero_", Napi::Function::New(env, ZeroInplaceOp));
  ops.Set("fill_", Napi::Function::New(env, FillInplaceOp));
  ops.Set("uniform_", Napi::Function::New(env, UniformInplaceOp));
  ops.Set("normal_", Napi::Function::New(env, NormalInplaceOp));
  ops.Set("addcmul_", Napi::Function::New(env, AddcmulInplaceOp));
  ops.Set("addcdiv_", Napi::Function::New(env, AddcdivInplaceOp));
  ops.Set("clamp_", Napi::Function::New(env, ClampInplaceOp));

  // Reductions
  ops.Set("sum", Napi::Function::New(env, SumOp));
  ops.Set("mean", Napi::Function::New(env, MeanOp));
  ops.Set("max", Napi::Function::New(env, MaxOp));
  ops.Set("min", Napi::Function::New(env, MinOp));
  ops.Set("argmax", Napi::Function::New(env, ArgmaxOp));
  ops.Set("argmin", Napi::Function::New(env, ArgminOp));
  ops.Set("softmax", Napi::Function::New(env, SoftmaxOp));
  ops.Set("logSoftmax", Napi::Function::New(env, LogSoftmaxOp));
  ops.Set("norm", Napi::Function::New(env, NormOp));

  // Views & Shape
  ops.Set("reshape", Napi::Function::New(env, ReshapeOp));
  ops.Set("view", Napi::Function::New(env, ViewOp));
  ops.Set("transpose", Napi::Function::New(env, TransposeOp));
  ops.Set("permute", Napi::Function::New(env, PermuteOp));
  ops.Set("squeeze", Napi::Function::New(env, SqueezeOp));
  ops.Set("unsqueeze", Napi::Function::New(env, UnsqueezeOp));
  ops.Set("flatten", Napi::Function::New(env, FlattenOp));
  ops.Set("expand", Napi::Function::New(env, ExpandOp));
  ops.Set("cat", Napi::Function::New(env, CatOp));
  ops.Set("stack", Napi::Function::New(env, StackOp));
  ops.Set("split", Napi::Function::New(env, SplitOp));
  ops.Set("chunk", Napi::Function::New(env, ChunkOp));
  ops.Set("narrow", Napi::Function::New(env, NarrowOp));
  ops.Set("maskedFill", Napi::Function::New(env, MaskedFillOp));
  ops.Set("tril", Napi::Function::New(env, TrilOp));
  ops.Set("triu", Napi::Function::New(env, TriuOp));

  // NN / Loss
  ops.Set("linear", Napi::Function::New(env, LinearOp));
  ops.Set("conv1d", Napi::Function::New(env, Conv1dOp));
  ops.Set("conv2d", Napi::Function::New(env, Conv2dOp));
  ops.Set("maxPool2d", Napi::Function::New(env, MaxPool2dOp));
  ops.Set("avgPool2d", Napi::Function::New(env, AvgPool2dOp));
  ops.Set("adaptiveAvgPool2d", Napi::Function::New(env, AdaptiveAvgPool2dOp));
  ops.Set("layerNorm", Napi::Function::New(env, LayerNormOp));
  ops.Set("rmsNorm", Napi::Function::New(env, RmsNormOp));
  ops.Set("batchNorm", Napi::Function::New(env, BatchNormOp));
  ops.Set("groupNorm", Napi::Function::New(env, GroupNormOp));
  ops.Set("embedding", Napi::Function::New(env, EmbeddingOp));
  ops.Set("dropout", Napi::Function::New(env, DropoutOp));
  ops.Set("scaledDotProductAttention", Napi::Function::New(env, ScaledDotProductAttentionOp));
  ops.Set("crossEntropy", Napi::Function::New(env, CrossEntropyOp));
  ops.Set("nllLoss", Napi::Function::New(env, NllLossOp));
  ops.Set("mseLoss", Napi::Function::New(env, MseLossOp));
  ops.Set("l1Loss", Napi::Function::New(env, L1LossOp));
  ops.Set("smoothL1Loss", Napi::Function::New(env, SmoothL1LossOp));
  ops.Set("bceWithLogitsLoss", Napi::Function::New(env, BceWithLogitsLossOp));
  ops.Set("klDiv", Napi::Function::New(env, KlDivOp));
  ops.Set("huberLoss", Napi::Function::New(env, HuberLossOp));
  ops.Set("cosineSimilarity", Napi::Function::New(env, CosineSimilarityOp));
  ops.Set("conv3d", Napi::Function::New(env, Conv3dOp));
  ops.Set("convTranspose1d", Napi::Function::New(env, ConvTranspose1dOp));
  ops.Set("convTranspose2d", Napi::Function::New(env, ConvTranspose2dOp));
  ops.Set("upsample2d", Napi::Function::New(env, Upsample2dOp));
  ops.Set("pad", Napi::Function::New(env, PadOp));
  ops.Set("oneHot", Napi::Function::New(env, OneHotOp));
  ops.Set("lstm", Napi::Function::New(env, LstmOp));
  ops.Set("gru", Napi::Function::New(env, GruOp));
  ops.Set("rnn", Napi::Function::New(env, RnnOp));
  ops.Set("fusedAdamStep", Napi::Function::New(env, FusedAdamStepOp));
  ops.Set("complex", Napi::Function::New(env, ComplexOp));
  ops.Set("polar", Napi::Function::New(env, PolarOp));
  ops.Set("angle", Napi::Function::New(env, AngleOp));
  ops.Set("viewAsReal", Napi::Function::New(env, ViewAsRealOp));
  ops.Set("viewAsComplex", Napi::Function::New(env, ViewAsComplexOp));
  ops.Set("sparseCooTensor", Napi::Function::New(env, SparseCooTensorOp));
  ops.Set("sparseCsrTensor", Napi::Function::New(env, SparseCsrTensorOp));
  ops.Set("sparseMm", Napi::Function::New(env, SparseMmOp));
  ops.Set("ampUnscaleAndCheck", Napi::Function::New(env, AmpUnscaleAndCheckOp));
  ops.Set("ampUpdateScale", Napi::Function::New(env, AmpUpdateScaleOp));
  ops.Set("zeropowerViaNewtonSchulz", Napi::Function::New(env, ZeropowerViaNewtonSchulzOp));
  ops.Set("muonStep", Napi::Function::New(env, MuonStepOp));
  ops.Set("convTranspose3d", Napi::Function::New(env, ConvTranspose3dOp));
  ops.Set("maxPool1d", Napi::Function::New(env, MaxPool1dOp));
  ops.Set("maxPool3d", Napi::Function::New(env, MaxPool3dOp));
  ops.Set("avgPool1d", Napi::Function::New(env, AvgPool1dOp));
  ops.Set("avgPool3d", Napi::Function::New(env, AvgPool3dOp));
  ops.Set("adaptiveMaxPool2d", Napi::Function::New(env, AdaptiveMaxPool2dOp));
  ops.Set("instanceNorm", Napi::Function::New(env, InstanceNormOp));
  ops.Set("bilinear", Napi::Function::New(env, BilinearOp));
  ops.Set("pixelShuffle", Napi::Function::New(env, PixelShuffleOp));
  ops.Set("pixelUnshuffle", Napi::Function::New(env, PixelUnshuffleOp));
  ops.Set("im2col", Napi::Function::New(env, Im2ColOp));
  ops.Set("col2im", Napi::Function::New(env, Col2ImOp));
  ops.Set("embeddingBag", Napi::Function::New(env, EmbeddingBagOp));

  exports.Set("fast", ops);
}

}  // namespace norch
