#include "norch/tensor_wrap.hpp"

#include <ATen/ATen.h>
#include <c10/core/AutogradState.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/function.h>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "norch/addon_state.hpp"
#include "norch/convert.hpp"

namespace norch {

namespace {

[[nodiscard]] c10::ScalarType TypedArrayToScalarType(Napi::Env env, napi_typedarray_type type) {
  switch (type) {
    case napi_float32_array:
      return c10::ScalarType::Float;
    case napi_float64_array:
      return c10::ScalarType::Double;
    case napi_int8_array:
      return c10::ScalarType::Char;
    case napi_uint8_array:
    case napi_uint8_clamped_array:
      return c10::ScalarType::Byte;
    case napi_int16_array:
      return c10::ScalarType::Short;
    case napi_uint16_array:
      return c10::ScalarType::UInt16;
    case napi_int32_array:
      return c10::ScalarType::Int;
    case napi_bigint64_array:
      return c10::ScalarType::Long;
    default:
      throw Napi::TypeError::New(env, "Unsupported TypedArray element type for libtorch tensor");
  }
}

[[nodiscard]] Napi::Value WrapBufferAsTypedArray(
    Napi::Env env,
    Napi::ArrayBuffer ab,
    c10::ScalarType dtype,
    std::size_t numel) {
  auto to_val = [](napi_env e, napi_value v) -> Napi::Value { return {e, v}; };
  switch (dtype) {
    case c10::ScalarType::Float:
      return to_val(env, Napi::Float32Array::New(env, numel, ab, 0));
    case c10::ScalarType::Double:
      return to_val(env, Napi::Float64Array::New(env, numel, ab, 0));
    case c10::ScalarType::Char:
      return to_val(env, Napi::Int8Array::New(env, numel, ab, 0));
    case c10::ScalarType::Byte:
    case c10::ScalarType::Bool:
      return to_val(env, Napi::Uint8Array::New(env, numel, ab, 0));
    case c10::ScalarType::Short:
      return to_val(env, Napi::Int16Array::New(env, numel, ab, 0));
    case c10::ScalarType::Int:
      return to_val(env, Napi::Int32Array::New(env, numel, ab, 0));
    case c10::ScalarType::Long:
      return to_val(env, Napi::BigInt64Array::New(env, numel, ab, 0));
    case c10::ScalarType::Half:
    case c10::ScalarType::BFloat16:
      return to_val(env, Napi::Uint16Array::New(env, numel, ab, 0));
    case c10::ScalarType::ComplexFloat:
      return to_val(env, Napi::Float32Array::New(env, numel * 2, ab, 0));
    case c10::ScalarType::ComplexDouble:
      return to_val(env, Napi::Float64Array::New(env, numel * 2, ab, 0));
    default:
      return to_val(env, Napi::Uint8Array::New(env, ab.ByteLength(), ab, 0));
  }
}

struct ZeroCopyPin {
  Napi::Reference<Napi::Value> ref;
  std::thread::id owner_thread;
  AddonState* state = nullptr;
  std::shared_ptr<std::atomic<bool>> env_alive;
};

Napi::Value FromTypedArray(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    if (info.Length() < 1 || !info[0].IsTypedArray()) {
      throw Napi::TypeError::New(env, "fromTypedArray(typedArray, shape?, options?) expects a TypedArray");
    }
    const auto ta = info[0].As<Napi::TypedArray>();
    const auto inferred_dtype = TypedArrayToScalarType(env, ta.TypedArrayType());

    std::vector<std::int64_t> shape;
    if (info.Length() >= 2 && !info[1].IsUndefined() && !info[1].IsNull()) {
      shape = ParseIntArray(env, info[1]);
    } else {
      shape = {static_cast<std::int64_t>(ta.ElementLength())};
    }

    bool copy = true;
    c10::ScalarType target_dtype = inferred_dtype;
    c10::Device target_device(c10::kCPU);
    bool requires_grad = false;

    if (info.Length() >= 3 && info[2].IsObject()) {
      const auto opts_obj = info[2].As<Napi::Object>();
      if (opts_obj.Has("copy") && opts_obj.Get("copy").IsBoolean()) {
        copy = opts_obj.Get("copy").As<Napi::Boolean>().Value();
      }
      if (opts_obj.Has("dtype") && !opts_obj.Get("dtype").IsUndefined() && !opts_obj.Get("dtype").IsNull()) {
        target_dtype = ParseScalarType(env, opts_obj.Get("dtype"));
      }
      if (opts_obj.Has("device") && !opts_obj.Get("device").IsUndefined() && !opts_obj.Get("device").IsNull()) {
        target_device = ParseDevice(env, opts_obj.Get("device"));
      }
      if (opts_obj.Has("requiresGrad") && opts_obj.Get("requiresGrad").IsBoolean()) {
        requires_grad = opts_obj.Get("requiresGrad").As<Napi::Boolean>().Value();
      }
    }

    // Validate shape matches buffer element count (supporting raw Uint16Array for fp16/bf16 or Uint8Array for raw bytes)
    std::int64_t shape_numel = 1;
    for (const auto d : shape) {
      if (d < 0) {
        throw Napi::RangeError::New(env, "Negative dimension in shape");
      }
      shape_numel *= d;
    }

    auto ab = ta.ArrayBuffer();
    auto* raw_ptr = static_cast<std::uint8_t*>(ab.Data()) + ta.ByteOffset();
    const std::size_t byte_len = ta.ByteLength();

    const bool raw_reinterpret =
        (ta.TypedArrayType() == napi_uint16_array &&
         (target_dtype == c10::ScalarType::Half || target_dtype == c10::ScalarType::BFloat16)) ||
        (ta.TypedArrayType() == napi_float32_array &&
         target_dtype == c10::ScalarType::ComplexFloat) ||
        (ta.TypedArrayType() == napi_float64_array &&
         target_dtype == c10::ScalarType::ComplexDouble) ||
        (ta.TypedArrayType() == napi_uint8_array &&
         (target_dtype == c10::ScalarType::Bool || target_dtype == c10::ScalarType::Byte)) ||
        (ta.TypedArrayType() == napi_uint8_array &&
         static_cast<std::size_t>(shape_numel) * c10::elementSize(target_dtype) == byte_len);

    const c10::ScalarType blob_dtype = raw_reinterpret ? target_dtype : inferred_dtype;
    const std::size_t expected_bytes =
        static_cast<std::size_t>(shape_numel) * c10::elementSize(blob_dtype);
    if (expected_bytes != byte_len) {
      throw Napi::RangeError::New(
          env,
          "TypedArray byte length (" + std::to_string(byte_len) +
              ") does not match shape/dtype byte length (" + std::to_string(expected_bytes) + ")");
    }

    at::Tensor result;
    const auto blob_opts = at::TensorOptions().dtype(blob_dtype).device(c10::kCPU);
    if (!copy && target_dtype == blob_dtype && target_device.is_cpu()) {
      auto& state = AddonState::Get(env);
      auto pin = std::make_shared<ZeroCopyPin>();
      pin->ref = Napi::Persistent(info[0]);
      pin->owner_thread = state.js_thread_id();
      pin->state = &state;
      pin->env_alive = state.env_alive;

      result = at::from_blob(
          raw_ptr,
          shape,
          [pin = std::move(pin)](void* /*ptr*/) {
            if (!pin->env_alive || !pin->env_alive->load()) {
              pin->ref.SuppressDestruct();
              return;
            }
            if (std::this_thread::get_id() == pin->owner_thread) {
              pin->ref.Reset();
            } else if (pin->state != nullptr) {
              pin->state->EnqueuePinRelease(std::move(pin->ref));
            }
          },
          blob_opts);
    } else {
      const at::Tensor view = at::from_blob(raw_ptr, shape, blob_opts);
      result = view.to(at::TensorOptions().dtype(target_dtype).device(target_device),
                       /*non_blocking=*/false,
                       /*copy=*/true);
    }

    if (requires_grad) {
      result.set_requires_grad(true);
    }
    return TensorWrap::NewInstance(env, std::move(result));
  });
}

}  // namespace

Napi::Function TensorWrap::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func = DefineClass(
      env,
      "NativeTensor",
      {
          InstanceMethod("dispose", &TensorWrap::Dispose),
          InstanceMethod("isDisposed", &TensorWrap::IsDisposedJs),
          InstanceMethod("shape", &TensorWrap::Shape),
          InstanceMethod("strides", &TensorWrap::Strides),
          InstanceMethod("dim", &TensorWrap::Dim),
          InstanceMethod("numel", &TensorWrap::Numel),
          InstanceMethod("elementSize", &TensorWrap::ElementSize),
          InstanceMethod("nbytes", &TensorWrap::Nbytes),
          InstanceMethod("storageOffset", &TensorWrap::StorageOffset),
          InstanceMethod("dtype", &TensorWrap::Dtype),
          InstanceMethod("device", &TensorWrap::Device),
          InstanceMethod("layout", &TensorWrap::Layout),
          InstanceMethod("isContiguous", &TensorWrap::IsContiguous),
          InstanceMethod("isCuda", &TensorWrap::IsCuda),
          InstanceMethod("isPinned", &TensorWrap::IsPinned),
          InstanceMethod("pinMemory", &TensorWrap::PinMemory),
          InstanceMethod("isComplex", &TensorWrap::IsComplex),
          InstanceMethod("isSparse", &TensorWrap::IsSparse),
          InstanceMethod("real", &TensorWrap::Real),
          InstanceMethod("imag", &TensorWrap::Imag),
          InstanceMethod("conj", &TensorWrap::Conj),
          InstanceMethod("toDense", &TensorWrap::ToDense),
          InstanceMethod("toSparseCoo", &TensorWrap::ToSparseCoo),
          InstanceMethod("toSparseCsr", &TensorWrap::ToSparseCsr),
          InstanceMethod("toSparseCsc", &TensorWrap::ToSparseCsc),
          InstanceMethod("coalesce", &TensorWrap::Coalesce),
          InstanceMethod("sparseIndices", &TensorWrap::SparseIndices),
          InstanceMethod("sparseValues", &TensorWrap::SparseValues),
          InstanceMethod("crowIndices", &TensorWrap::CrowIndices),
          InstanceMethod("colIndices", &TensorWrap::ColIndices),
          InstanceMethod("ccolIndices", &TensorWrap::CcolIndices),
          InstanceMethod("rowIndices", &TensorWrap::RowIndices),
          InstanceMethod("sparseDim", &TensorWrap::SparseDim),
          InstanceMethod("denseDim", &TensorWrap::DenseDim),
          InstanceMethod("nnz", &TensorWrap::Nnz),
          InstanceMethod("requiresGrad", &TensorWrap::RequiresGrad),
          InstanceMethod("setRequiresGrad", &TensorWrap::SetRequiresGrad),
          InstanceMethod("isLeaf", &TensorWrap::IsLeaf),
          InstanceMethod("retainsGrad", &TensorWrap::RetainsGrad),
          InstanceMethod("retainGrad", &TensorWrap::RetainGrad),
          InstanceMethod("grad", &TensorWrap::Grad),
          InstanceMethod("setGrad", &TensorWrap::SetGrad),
          InstanceMethod("gradFnName", &TensorWrap::GradFnName),
          InstanceMethod("registerHook", &TensorWrap::RegisterHook),
          InstanceMethod("removeHook", &TensorWrap::RemoveHook),
          InstanceMethod("dataPtr", &TensorWrap::DataPtr),
          InstanceMethod("storagePtr", &TensorWrap::StoragePtr),
          InstanceMethod("item", &TensorWrap::Item),
          InstanceMethod("toTypedArray", &TensorWrap::ToTypedArray),
          InstanceMethod("toTypedArrayAsync", &TensorWrap::ToTypedArrayAsync),
          InstanceMethod("dataView", &TensorWrap::DataView),
          InstanceMethod("copyFrom", &TensorWrap::CopyFrom),
          InstanceMethod("detach", &TensorWrap::Detach),
          InstanceMethod("detachInplace", &TensorWrap::DetachInplace),
          InstanceMethod("clone", &TensorWrap::Clone),
          InstanceMethod("contiguous", &TensorWrap::Contiguous),
          InstanceMethod("to", &TensorWrap::To),
          InstanceMethod("backward", &TensorWrap::Backward),
          InstanceMethod("indexGet", &TensorWrap::IndexGet),
          InstanceMethod("indexPut", &TensorWrap::IndexPut),
          InstanceMethod("equal", &TensorWrap::Equal),
          InstanceMethod("formatString", &TensorWrap::FormatString),
      });

  AddonState::Get(env).tensor_ctor = Napi::Persistent(func);
  exports.Set("NativeTensor", func);
  return func;
}

Napi::Object TensorWrap::NewInstance(Napi::Env env, at::Tensor tensor) {
  auto& state = AddonState::Get(env);
  auto obj = state.tensor_ctor.New({});
  auto* wrap = Napi::ObjectWrap<TensorWrap>::Unwrap(obj);
  wrap->SetTensor(env, std::move(tensor));
  return obj;
}

bool TensorWrap::IsTensorWrap(Napi::Env env, const Napi::Value& val) {
  return TryUnwrap(env, val) != nullptr;
}

TensorWrap* TensorWrap::TryUnwrap(Napi::Env env, const Napi::Value& val) {
  if (!val.IsObject()) {
    return nullptr;
  }
  const auto obj = val.As<Napi::Object>();
  auto& state = AddonState::Get(env);
  if (obj.InstanceOf(state.tensor_ctor.Value())) {
    return Napi::ObjectWrap<TensorWrap>::Unwrap(obj);
  }
  if (obj.Has("_native")) {
    const auto inner = obj.Get("_native");
    if (inner.IsObject() && inner.As<Napi::Object>().InstanceOf(state.tensor_ctor.Value())) {
      return Napi::ObjectWrap<TensorWrap>::Unwrap(inner.As<Napi::Object>());
    }
  }
  return nullptr;
}

const at::Tensor& TensorWrap::UnwrapTensor(Napi::Env env, const Napi::Value& val) {
  auto* wrap = TryUnwrap(env, val);
  if (wrap == nullptr) {
    throw Napi::TypeError::New(env, "Expected a Tensor");
  }
  if (wrap->IsDisposed()) {
    throw Napi::Error::New(env, "Tensor has been disposed");
  }
  return wrap->tensor();
}

at::Tensor& TensorWrap::UnwrapMutableTensor(Napi::Env env, const Napi::Value& val) {
  auto* wrap = TryUnwrap(env, val);
  if (wrap == nullptr) {
    throw Napi::TypeError::New(env, "Expected a Tensor");
  }
  if (wrap->IsDisposed()) {
    throw Napi::Error::New(env, "Tensor has been disposed");
  }
  return wrap->mutable_tensor();
}

TensorWrap::TensorWrap(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<TensorWrap>(info), env_(info.Env()) {}

TensorWrap::~TensorWrap() {
  if (!disposed_ && env_ != nullptr) {
    const Napi::Env env(env_);
    if (env.GetInstanceData<AddonState>() != nullptr) {
      ReleaseTracking(env);
    }
  }
  tensor_.reset();
  disposed_ = true;
}

void TensorWrap::SetTensor(Napi::Env env, at::Tensor tensor) {
  if (!disposed_) {
    ReleaseTracking(env);
  }
  env_ = env;
  tensor_ = std::move(tensor);
  disposed_ = !tensor_.defined();
  if (!disposed_) {
    auto& state = AddonState::Get(env);
    state.live_tensors += 1;
    if (tensor_.has_storage()) {
      const auto& storage = tensor_.storage();
      tracked_storage_ = storage.unsafeGetStorageImpl();
      state.TrackStorage(env, tracked_storage_, static_cast<std::int64_t>(storage.nbytes()));
    }
  }
}

void TensorWrap::ReleaseTracking(Napi::Env env) {
  auto* state = env.GetInstanceData<AddonState>();
  if (state == nullptr) {
    tracked_storage_ = nullptr;
    return;
  }
  if (!disposed_ && state->live_tensors > 0) {
    state->live_tensors -= 1;
  }
  if (tracked_storage_ != nullptr) {
    state->UntrackStorage(env, tracked_storage_);
    tracked_storage_ = nullptr;
  }
}

void TensorWrap::DisposeInternal(Napi::Env env) {
  if (disposed_) {
    return;
  }
  ReleaseTracking(env);
  tensor_.reset();
  disposed_ = true;
}

const at::Tensor& TensorWrap::tensor() const {
  if (disposed_ || !tensor_.defined()) {
    throw std::runtime_error("Tensor has been disposed");
  }
  return tensor_;
}

at::Tensor& TensorWrap::mutable_tensor() {
  if (disposed_ || !tensor_.defined()) {
    throw std::runtime_error("Tensor has been disposed");
  }
  return tensor_;
}

Napi::Value TensorWrap::Dispose(const Napi::CallbackInfo& info) {
  DisposeInternal(info.Env());
  return info.Env().Undefined();
}

Napi::Value TensorWrap::IsDisposedJs(const Napi::CallbackInfo& info) {
  return Napi::Boolean::New(info.Env(), disposed_);
}

Napi::Value TensorWrap::Shape(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto sizes = tensor().sizes();
    auto arr = Napi::Array::New(env, sizes.size());
    for (std::size_t i = 0; i < sizes.size(); ++i) {
      arr.Set(static_cast<std::uint32_t>(i), Napi::Number::New(env, static_cast<double>(sizes[i])));
    }
    return arr;
  });
}

Napi::Value TensorWrap::Strides(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto strides = tensor().strides();
    auto arr = Napi::Array::New(env, strides.size());
    for (std::size_t i = 0; i < strides.size(); ++i) {
      arr.Set(
          static_cast<std::uint32_t>(i), Napi::Number::New(env, static_cast<double>(strides[i])));
    }
    return arr;
  });
}

Napi::Value TensorWrap::Dim(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return Napi::Number::New(env, static_cast<double>(tensor().dim()));
  });
}

Napi::Value TensorWrap::Numel(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return Napi::Number::New(env, static_cast<double>(tensor().numel()));
  });
}

Napi::Value TensorWrap::ElementSize(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return Napi::Number::New(env, static_cast<double>(tensor().element_size()));
  });
}

Napi::Value TensorWrap::Nbytes(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return Napi::Number::New(env, static_cast<double>(tensor().nbytes()));
  });
}

Napi::Value TensorWrap::StorageOffset(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return Napi::Number::New(env, static_cast<double>(tensor().storage_offset()));
  });
}

Napi::Value TensorWrap::Dtype(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return Napi::String::New(env, ScalarTypeToString(tensor().scalar_type()));
  });
}

Napi::Value TensorWrap::Device(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return Napi::String::New(env, DeviceToString(tensor().device()));
  });
}

Napi::Value TensorWrap::Layout(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    switch (tensor().layout()) {
      case c10::kStrided:
        return Napi::String::New(env, "strided");
      case c10::kSparse:
        return Napi::String::New(env, "sparse_coo");
      case c10::kSparseCsr:
        return Napi::String::New(env, "sparse_csr");
      case c10::kSparseCsc:
        return Napi::String::New(env, "sparse_csc");
      case c10::kSparseBsr:
        return Napi::String::New(env, "sparse_bsr");
      case c10::kSparseBsc:
        return Napi::String::New(env, "sparse_bsc");
      default:
        return Napi::String::New(env, "other");
    }
  });
}

Napi::Value TensorWrap::IsContiguous(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return Napi::Boolean::New(env, tensor().is_contiguous());
  });
}

Napi::Value TensorWrap::IsCuda(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return Napi::Boolean::New(env, tensor().is_cuda());
  });
}

Napi::Value TensorWrap::IsPinned(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return Napi::Boolean::New(env, tensor().is_pinned());
  });
}

Napi::Value TensorWrap::PinMemory(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return NewInstance(env, tensor().pin_memory());
  });
}

Napi::Value TensorWrap::IsComplex(const Napi::CallbackInfo& info) {
  return GuardNapi(info.Env(), [&]() -> Napi::Value {
    return Napi::Boolean::New(info.Env(), tensor().is_complex());
  });
}

Napi::Value TensorWrap::IsSparse(const Napi::CallbackInfo& info) {
  return GuardNapi(info.Env(), [&]() -> Napi::Value {
    return Napi::Boolean::New(info.Env(), tensor().layout() != c10::kStrided);
  });
}

Napi::Value TensorWrap::Real(const Napi::CallbackInfo& info) {
  return GuardNapi(info.Env(), [&]() -> Napi::Value {
    return NewInstance(info.Env(), at::real(tensor()));
  });
}

Napi::Value TensorWrap::Imag(const Napi::CallbackInfo& info) {
  return GuardNapi(info.Env(), [&]() -> Napi::Value {
    return NewInstance(info.Env(), at::imag(tensor()));
  });
}

Napi::Value TensorWrap::Conj(const Napi::CallbackInfo& info) {
  return GuardNapi(info.Env(), [&]() -> Napi::Value {
    return NewInstance(info.Env(), at::conj(tensor()).resolve_conj());
  });
}

Napi::Value TensorWrap::ToDense(const Napi::CallbackInfo& info) {
  return GuardNapi(info.Env(), [&]() -> Napi::Value {
    return NewInstance(info.Env(), tensor().to_dense());
  });
}

Napi::Value TensorWrap::ToSparseCoo(const Napi::CallbackInfo& info) {
  return GuardNapi(info.Env(), [&]() -> Napi::Value {
    return NewInstance(info.Env(), tensor().to_sparse());
  });
}

Napi::Value TensorWrap::ToSparseCsr(const Napi::CallbackInfo& info) {
  return GuardNapi(info.Env(), [&]() -> Napi::Value {
    return NewInstance(info.Env(), tensor().to_sparse_csr());
  });
}

Napi::Value TensorWrap::ToSparseCsc(const Napi::CallbackInfo& info) {
  return GuardNapi(info.Env(), [&]() -> Napi::Value {
    return NewInstance(info.Env(), tensor().to_sparse_csc());
  });
}

Napi::Value TensorWrap::Coalesce(const Napi::CallbackInfo& info) {
  return GuardNapi(info.Env(), [&]() -> Napi::Value {
    return NewInstance(info.Env(), tensor().coalesce());
  });
}

Napi::Value TensorWrap::SparseIndices(const Napi::CallbackInfo& info) {
  return GuardNapi(info.Env(), [&]() -> Napi::Value {
    return NewInstance(info.Env(), tensor()._indices());
  });
}

Napi::Value TensorWrap::SparseValues(const Napi::CallbackInfo& info) {
  return GuardNapi(info.Env(), [&]() -> Napi::Value {
    const auto& t = tensor();
    return NewInstance(info.Env(), t.is_sparse() ? t._values() : t.values());
  });
}

Napi::Value TensorWrap::CrowIndices(const Napi::CallbackInfo& info) {
  return GuardNapi(info.Env(), [&]() -> Napi::Value {
    return NewInstance(info.Env(), tensor().crow_indices());
  });
}

Napi::Value TensorWrap::ColIndices(const Napi::CallbackInfo& info) {
  return GuardNapi(info.Env(), [&]() -> Napi::Value {
    return NewInstance(info.Env(), tensor().col_indices());
  });
}

Napi::Value TensorWrap::CcolIndices(const Napi::CallbackInfo& info) {
  return GuardNapi(info.Env(), [&]() -> Napi::Value {
    return NewInstance(info.Env(), tensor().ccol_indices());
  });
}

Napi::Value TensorWrap::RowIndices(const Napi::CallbackInfo& info) {
  return GuardNapi(info.Env(), [&]() -> Napi::Value {
    return NewInstance(info.Env(), tensor().row_indices());
  });
}

Napi::Value TensorWrap::SparseDim(const Napi::CallbackInfo& info) {
  return GuardNapi(info.Env(), [&]() -> Napi::Value {
    return Napi::Number::New(info.Env(), static_cast<double>(tensor().sparse_dim()));
  });
}

Napi::Value TensorWrap::DenseDim(const Napi::CallbackInfo& info) {
  return GuardNapi(info.Env(), [&]() -> Napi::Value {
    return Napi::Number::New(info.Env(), static_cast<double>(tensor().dense_dim()));
  });
}

Napi::Value TensorWrap::Nnz(const Napi::CallbackInfo& info) {
  return GuardNapi(info.Env(), [&]() -> Napi::Value {
    return Napi::Number::New(info.Env(), static_cast<double>(tensor()._nnz()));
  });
}

Napi::Value TensorWrap::RequiresGrad(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return Napi::Boolean::New(env, tensor().requires_grad());
  });
}

Napi::Value TensorWrap::SetRequiresGrad(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const bool req = info.Length() < 1 || info[0].ToBoolean().Value();
    mutable_tensor().set_requires_grad(req);
    return info.This();
  });
}

Napi::Value TensorWrap::IsLeaf(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return Napi::Boolean::New(env, tensor().is_leaf());
  });
}

Napi::Value TensorWrap::RetainsGrad(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return Napi::Boolean::New(env, tensor().retains_grad());
  });
}

Napi::Value TensorWrap::RetainGrad(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    mutable_tensor().retain_grad();
    return env.Undefined();
  });
}

Napi::Value TensorWrap::Grad(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& g = tensor().grad();
    if (!g.defined()) {
      return env.Null();
    }
    return NewInstance(env, g);
  });
}

Napi::Value TensorWrap::SetGrad(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    if (info.Length() < 1 || info[0].IsNull() || info[0].IsUndefined()) {
      mutable_tensor().mutable_grad() = at::Tensor();
    } else {
      mutable_tensor().mutable_grad() = UnwrapTensor(env, info[0]);
    }
    return env.Undefined();
  });
}

Napi::Value TensorWrap::GradFnName(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& fn = tensor().grad_fn();
    if (!fn) {
      return env.Null();
    }
    return Napi::String::New(env, fn->name());
  });
}

namespace {

struct JsHookHolder {
  napi_env env;
  AddonState* state;
  Napi::FunctionReference fn;

  JsHookHolder(Napi::Env e, AddonState* s, Napi::Function f)
      : env(e), state(s), fn(Napi::Persistent(f)) {
    state->custom_autograd_nodes_alive.fetch_add(1);
  }
  ~JsHookHolder() noexcept {
    try {
      if (state != nullptr) {
        state->custom_autograd_nodes_alive.fetch_sub(1);
        if (std::this_thread::get_id() == state->js_thread_id()) {
          const Napi::Env e(env);
          if (e.GetInstanceData<AddonState>() != nullptr) {
            fn.Reset();
          }
        }
      }
    } catch (...) {
      std::fputs("norch: suppressed exception in ~JsHookHolder\n", stderr);
    }
  }
  JsHookHolder(const JsHookHolder&) = delete;
  JsHookHolder& operator=(const JsHookHolder&) = delete;
  JsHookHolder(JsHookHolder&&) = delete;
  JsHookHolder& operator=(JsHookHolder&&) = delete;
};

class ToTypedArrayWorker final : public Napi::AsyncWorker {
 public:
  ToTypedArrayWorker(Napi::Env env, at::Tensor src, Napi::Promise::Deferred deferred)
      : Napi::AsyncWorker(env),
        src_(std::move(src)),
        deferred_(deferred) {}

  void Execute() override {
    try {
      at::Tensor cpu_contig = src_.to(c10::kCPU).contiguous();
      dtype_ = cpu_contig.scalar_type();
      numel_ = static_cast<std::size_t>(cpu_contig.numel());
      const std::size_t nbytes = cpu_contig.nbytes();
      bytes_.resize(nbytes);
      if (nbytes > 0) {
        std::memcpy(bytes_.data(), cpu_contig.data_ptr(), nbytes);
      }
    } catch (const c10::Error& err) {
      SetError(err.what_without_backtrace());
    } catch (const std::exception& ex) {
      SetError(ex.what());
    }
  }

  void OnOK() override {
    const Napi::Env env = Env();
    auto ab = Napi::ArrayBuffer::New(env, bytes_.size());
    if (!bytes_.empty()) {
      std::memcpy(ab.Data(), bytes_.data(), bytes_.size());
    }
    deferred_.Resolve(WrapBufferAsTypedArray(env, ab, dtype_, numel_));
  }

  void OnError(const Napi::Error& err) override { deferred_.Reject(err.Value()); }

 private:
  at::Tensor src_;
  Napi::Promise::Deferred deferred_;
  std::vector<std::uint8_t> bytes_;
  c10::ScalarType dtype_ = c10::ScalarType::Float;
  std::size_t numel_ = 0;
};

}  // namespace

Napi::Value TensorWrap::RegisterHook(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    if (info.Length() < 1 || !info[0].IsFunction()) {
      throw Napi::TypeError::New(env, "registerHook(fn) expects a callback function");
    }
    auto& state = AddonState::Get(env);
    auto holder = std::make_shared<JsHookHolder>(env, &state, info[0].As<Napi::Function>());
    const unsigned hook_id = tensor().register_hook([holder](const at::Tensor& grad) -> at::Tensor {
      TORCH_CHECK(
          holder->state != nullptr && std::this_thread::get_id() == holder->state->js_thread_id(),
          "Tensor autograd hook invoked off the JS main thread");
      const Napi::Env cb_env(holder->env);
      const Napi::HandleScope scope(cb_env);
      const Napi::Value ret = holder->fn.Call({TensorWrap::NewInstance(cb_env, grad)});
      if (cb_env.IsExceptionPending()) {
        const Napi::Error err = cb_env.GetAndClearPendingException();
        TORCH_CHECK(false, "JS exception in tensor gradient hook: ", err.Message());
      }
      if (ret.IsUndefined() || ret.IsNull()) {
        return grad;
      }
      return TensorWrap::UnwrapTensor(cb_env, ret);
    });
    return Napi::Number::New(env, static_cast<double>(hook_id));
  });
}

Napi::Value TensorWrap::RemoveHook(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    if (info.Length() < 1 || !info[0].IsNumber()) {
      throw Napi::TypeError::New(env, "removeHook(handleId) expects a numeric handle id");
    }
    const auto hook_id = info[0].As<Napi::Number>().Uint32Value();
    tensor().remove_hook(hook_id);
    return env.Undefined();
  });
}

Napi::Value TensorWrap::DataPtr(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto addr = reinterpret_cast<std::uintptr_t>(tensor().data_ptr());
    return Napi::BigInt::New(env, static_cast<std::uint64_t>(addr));
  });
}

Napi::Value TensorWrap::StoragePtr(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& t = tensor();
    if (!t.has_storage()) {
      return Napi::BigInt::New(env, static_cast<std::uint64_t>(0));
    }
    const auto addr = reinterpret_cast<std::uintptr_t>(t.storage().unsafeGetStorageImpl());
    return Napi::BigInt::New(env, static_cast<std::uint64_t>(addr));
  });
}

Napi::Value TensorWrap::Item(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return ScalarToJs(env, tensor().item());
  });
}

Napi::Value TensorWrap::ToTypedArray(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    at::Tensor cpu_contig = tensor().detach().to(c10::kCPU).contiguous();
    const std::size_t nbytes = cpu_contig.nbytes();
    const auto numel = static_cast<std::size_t>(cpu_contig.numel());
    auto ab = Napi::ArrayBuffer::New(env, nbytes);
    if (nbytes > 0) {
      std::memcpy(ab.Data(), cpu_contig.data_ptr(), nbytes);
    }
    return WrapBufferAsTypedArray(env, ab, cpu_contig.scalar_type(), numel);
  });
}

Napi::Value TensorWrap::ToTypedArrayAsync(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    auto deferred = Napi::Promise::Deferred::New(env);
    auto* worker = new ToTypedArrayWorker(env, tensor().detach(), deferred);
    worker->Queue();
    return deferred.Promise();
  });
}

Napi::Value TensorWrap::DataView(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& t = tensor();
    if (!t.device().is_cpu()) {
      throw Napi::Error::New(
          env, "dataView() requires a CPU tensor (call .cpu() or .toTypedArray() instead)");
    }
    if (!t.is_contiguous()) {
      throw Napi::Error::New(
          env, "dataView() requires a contiguous tensor (call .contiguous() first)");
    }
    const std::size_t nbytes = t.nbytes();
    const auto numel = static_cast<std::size_t>(t.numel());
    if (nbytes == 0) {
      auto ab = Napi::ArrayBuffer::New(env, 0);
      return WrapBufferAsTypedArray(env, ab, t.scalar_type(), 0);
    }
    // Keep a refcounted copy of the Tensor alive until V8 collects the ArrayBuffer
    auto* holder = new at::Tensor(t.detach());
    auto ab = Napi::ArrayBuffer::New(
        env,
        holder->data_ptr(),
        nbytes,
        [](Napi::Env /*cb_env*/, void* /*data*/, at::Tensor* h) { delete h; },
        holder);
    return WrapBufferAsTypedArray(env, ab, t.scalar_type(), numel);
  });
}

Napi::Value TensorWrap::CopyFrom(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    if (info.Length() < 1) {
      throw Napi::TypeError::New(env, "copyFrom(src, nonBlocking?) expects a source tensor");
    }
    const auto& src = UnwrapTensor(env, info[0]);
    const bool non_blocking = info.Length() >= 2 && info[1].ToBoolean().Value();
    mutable_tensor().copy_(src, non_blocking);
    return info.This();
  });
}

Napi::Value TensorWrap::Detach(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return NewInstance(env, tensor().detach());
  });
}

Napi::Value TensorWrap::DetachInplace(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    mutable_tensor().detach_();
    return info.This();
  });
}

Napi::Value TensorWrap::Clone(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    return NewInstance(env, tensor().clone());
  });
}

Napi::Value TensorWrap::Contiguous(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    c10::MemoryFormat mf = c10::MemoryFormat::Contiguous;
    if (info.Length() >= 1 && !info[0].IsUndefined() && !info[0].IsNull()) {
      mf = ParseMemoryFormat(env, info[0]);
    }
    return NewInstance(env, tensor().contiguous(mf));
  });
}

Napi::Value TensorWrap::To(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    const auto& t = tensor();
    c10::Device device = t.device();
    c10::ScalarType dtype = t.scalar_type();
    bool non_blocking = false;
    bool copy = false;
    std::optional<c10::MemoryFormat> mem_format = std::nullopt;

    if (info.Length() >= 1 && info[0].IsObject()) {
      const auto obj = info[0].As<Napi::Object>();
      if (obj.Has("device") && !obj.Get("device").IsUndefined() && !obj.Get("device").IsNull()) {
        device = ParseDevice(env, obj.Get("device"));
      }
      if (obj.Has("dtype") && !obj.Get("dtype").IsUndefined() && !obj.Get("dtype").IsNull()) {
        dtype = ParseScalarType(env, obj.Get("dtype"));
      }
      if (obj.Has("nonBlocking") && obj.Get("nonBlocking").IsBoolean()) {
        non_blocking = obj.Get("nonBlocking").As<Napi::Boolean>().Value();
      }
      if (obj.Has("copy") && obj.Get("copy").IsBoolean()) {
        copy = obj.Get("copy").As<Napi::Boolean>().Value();
      }
      if (obj.Has("memoryFormat") && !obj.Get("memoryFormat").IsUndefined() &&
          !obj.Get("memoryFormat").IsNull()) {
        mem_format = ParseMemoryFormat(env, obj.Get("memoryFormat"));
      }
    }
    return NewInstance(env, t.to(device, dtype, non_blocking, copy, mem_format));
  });
}

Napi::Value TensorWrap::Backward(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    torch::autograd::variable_list grad_tensors;
    std::optional<bool> retain_graph = std::nullopt;
    bool create_graph = false;
    torch::autograd::variable_list inputs;

    if (info.Length() >= 1 && !info[0].IsUndefined() && !info[0].IsNull()) {
      grad_tensors.push_back(UnwrapTensor(env, info[0]));
    }
    if (info.Length() >= 2 && info[1].IsObject()) {
      const auto opts = info[1].As<Napi::Object>();
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

    auto& state = AddonState::Get(env);
    const bool force_single_thread = state.custom_autograd_nodes_alive.load() > 0;
    auto& tls = c10::AutogradState::get_tls_state();
    const bool prev_mt = tls.get_multithreading_enabled();
    if (force_single_thread) {
      tls.set_multithreading_enabled(false);
    }
    struct RestoreMt {
      bool prev;
      bool active;
      RestoreMt(bool p, bool a) noexcept : prev(p), active(a) {}
      ~RestoreMt() {
        if (active) {
          c10::AutogradState::get_tls_state().set_multithreading_enabled(prev);
        }
      }
      RestoreMt(const RestoreMt&) = delete;
      RestoreMt& operator=(const RestoreMt&) = delete;
      RestoreMt(RestoreMt&&) = delete;
      RestoreMt& operator=(RestoreMt&&) = delete;
    } const guard(prev_mt, force_single_thread);

    torch::autograd::backward({tensor()}, grad_tensors, retain_graph, create_graph, inputs);
    return env.Undefined();
  });
}

Napi::Value TensorWrap::IndexGet(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    if (info.Length() < 1) {
      throw Napi::TypeError::New(env, "indexGet(indices) expects an Array of indices");
    }
    auto indices = ParseTensorIndices(env, info[0]);

    // Check if any slice has a negative step (libtorch's at::indexing::Slice requires step > 0)
    bool has_neg_step = false;
    for (const auto& idx : indices) {
      if (idx.is_slice() && idx.slice().step().expect_int() < 0) {
        has_neg_step = true;
        break;
      }
    }
    if (!has_neg_step) {
      return NewInstance(env, tensor().index(indices));
    }

    // Rewrite negative-step slices into positive-step slices on flipped dimensions
    at::Tensor cur = tensor();
    const auto ndim = cur.dim();
    std::int64_t specified_dims = 0;
    for (const auto& idx : indices) {
      if (!idx.is_none() && !idx.is_ellipsis()) {
        specified_dims += 1;
      }
    }

    std::int64_t dim = 0;
    std::vector<std::int64_t> flip_dims;
    std::vector<at::indexing::TensorIndex> rewritten;
    rewritten.reserve(indices.size());

    for (const auto& idx : indices) {
      if (idx.is_none()) {
        rewritten.push_back(idx);
        continue;
      }
      if (idx.is_ellipsis()) {
        const auto skipped = ndim - specified_dims;
        dim += skipped;
        rewritten.push_back(idx);
        continue;
      }
      if (idx.is_slice() && idx.slice().step().expect_int() < 0) {
        const auto dim_size = cur.size(dim);
        const auto neg_step = idx.slice().step().expect_int();
        const auto pos_step = -neg_step;
        const auto raw_start = idx.slice().start().expect_int();
        const auto raw_stop = idx.slice().stop().expect_int();

        // Normalize Python negative-step slice [start, stop, step] on length dim_size
        std::int64_t start =
            (raw_start == at::indexing::INDEX_MAX)
                ? (dim_size - 1)
                : (raw_start < 0 ? raw_start + dim_size : raw_start);
        std::int64_t stop =
            (raw_stop == at::indexing::INDEX_MIN)
                ? -1
                : (raw_stop < 0 ? raw_stop + dim_size : raw_stop);
        if (start >= dim_size) start = dim_size - 1;
        if (start < -1) start = -1;
        if (stop >= dim_size) stop = dim_size - 1;
        if (stop < -1) stop = -1;

        // On flipped dimension, index i maps to (dim_size - 1 - i)
        flip_dims.push_back(dim);
        if (start <= stop) {
          rewritten.emplace_back(at::indexing::Slice(0, 0, 1));
        } else {
          const auto flipped_start = dim_size - 1 - start;
          const auto flipped_stop = dim_size - 1 - stop;
          rewritten.emplace_back(at::indexing::Slice(flipped_start, flipped_stop, pos_step));
        }
        dim += 1;
      } else {
        rewritten.push_back(idx);
        dim += 1;
      }
    }

    if (!flip_dims.empty()) {
      cur = at::flip(cur, flip_dims);
    }
    return NewInstance(env, cur.index(rewritten));
  });
}

Napi::Value TensorWrap::IndexPut(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    if (info.Length() < 2) {
      throw Napi::TypeError::New(env, "indexPut(indices, value, accumulate?) expects >= 2 args");
    }
    const bool accumulate = info.Length() >= 3 && info[2].ToBoolean().Value();
    if (accumulate) {
      if (!info[0].IsArray()) {
        throw Napi::TypeError::New(env, "indexPut(..., accumulate=true) expects an array of Tensor/null indices");
      }
      const auto arr = info[0].As<Napi::Array>();
      c10::List<std::optional<at::Tensor>> idx_list;
      idx_list.reserve(arr.Length());
      for (std::uint32_t i = 0; i < arr.Length(); ++i) {
        const auto elem = arr.Get(i);
        if (elem.IsNull() || elem.IsUndefined() ||
            (elem.IsString() && elem.As<Napi::String>().Utf8Value() == ":")) {
          idx_list.push_back(std::nullopt);
        } else if (IsTensorWrap(env, elem)) {
          idx_list.push_back(UnwrapTensor(env, elem));
        } else if (elem.IsNumber()) {
          idx_list.push_back(
              at::tensor(elem.As<Napi::Number>().Int64Value(),
                         at::TensorOptions().dtype(c10::kLong).device(tensor().device())));
        } else {
          throw Napi::TypeError::New(
              env, "indexPut(..., accumulate=true) supports Tensor, integer, or ':'/null indices");
        }
      }
      at::Tensor rhs = IsTensorWrap(env, info[1])
                           ? UnwrapTensor(env, info[1])
                           : at::scalar_tensor(
                                 ParseScalar(env, info[1]),
                                 at::TensorOptions().dtype(tensor().scalar_type()).device(tensor().device()));
      mutable_tensor().index_put_(idx_list, rhs, /*accumulate=*/true);
      return info.This();
    }
    const auto indices = ParseTensorIndices(env, info[0]);
    if (IsTensorWrap(env, info[1])) {
      mutable_tensor().index_put_(indices, UnwrapTensor(env, info[1]));
    } else {
      mutable_tensor().index_put_(indices, ParseScalar(env, info[1]));
    }
    return info.This();
  });
}

Napi::Value TensorWrap::Equal(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    if (info.Length() < 1) {
      throw Napi::TypeError::New(env, "equal(other) expects a Tensor");
    }
    return Napi::Boolean::New(env, tensor().equal(UnwrapTensor(env, info[0])));
  });
}

Napi::Value TensorWrap::FormatString(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    std::ostringstream oss;
    oss << tensor();
    return Napi::String::New(env, oss.str());
  });
}

void RegisterTensorHelpers(Napi::Env env, Napi::Object exports) {
  exports.Set("fromTypedArray", Napi::Function::New(env, FromTypedArray));
}

}  // namespace norch
