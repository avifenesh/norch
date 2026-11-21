#pragma once

#include <ATen/core/Tensor.h>
#include <napi.h>

#include <cstdint>

namespace c10 {
struct StorageImpl;
}  // namespace c10

namespace norch {

class TensorWrap : public Napi::ObjectWrap<TensorWrap> {
 public:
  static Napi::Function Init(Napi::Env env, Napi::Object exports);
  static Napi::Object NewInstance(Napi::Env env, at::Tensor tensor);
  [[nodiscard]] static bool IsTensorWrap(Napi::Env env, const Napi::Value& val);
  [[nodiscard]] static TensorWrap* TryUnwrap(Napi::Env env, const Napi::Value& val);
  [[nodiscard]] static const at::Tensor& UnwrapTensor(Napi::Env env, const Napi::Value& val);
  [[nodiscard]] static at::Tensor& UnwrapMutableTensor(Napi::Env env, const Napi::Value& val);

  explicit TensorWrap(const Napi::CallbackInfo& info);
  ~TensorWrap() override;

  TensorWrap(const TensorWrap&) = delete;
  TensorWrap& operator=(const TensorWrap&) = delete;
  TensorWrap(TensorWrap&&) = delete;
  TensorWrap& operator=(TensorWrap&&) = delete;

  void SetTensor(Napi::Env env, at::Tensor tensor);
  void DisposeInternal(Napi::Env env);

  [[nodiscard]] bool IsDisposed() const noexcept { return disposed_; }
  [[nodiscard]] const at::Tensor& tensor() const;
  [[nodiscard]] at::Tensor& mutable_tensor();

 private:
  // Instance methods exposed on NativeTensor
  Napi::Value Dispose(const Napi::CallbackInfo& info);
  Napi::Value IsDisposedJs(const Napi::CallbackInfo& info);
  Napi::Value Shape(const Napi::CallbackInfo& info);
  Napi::Value Strides(const Napi::CallbackInfo& info);
  Napi::Value Dim(const Napi::CallbackInfo& info);
  Napi::Value Numel(const Napi::CallbackInfo& info);
  Napi::Value ElementSize(const Napi::CallbackInfo& info);
  Napi::Value Nbytes(const Napi::CallbackInfo& info);
  Napi::Value StorageOffset(const Napi::CallbackInfo& info);
  Napi::Value Dtype(const Napi::CallbackInfo& info);
  Napi::Value Device(const Napi::CallbackInfo& info);
  Napi::Value Layout(const Napi::CallbackInfo& info);
  Napi::Value IsContiguous(const Napi::CallbackInfo& info);
  Napi::Value IsCuda(const Napi::CallbackInfo& info);
  Napi::Value IsPinned(const Napi::CallbackInfo& info);
  Napi::Value PinMemory(const Napi::CallbackInfo& info);
  Napi::Value IsComplex(const Napi::CallbackInfo& info);
  Napi::Value IsSparse(const Napi::CallbackInfo& info);
  Napi::Value Real(const Napi::CallbackInfo& info);
  Napi::Value Imag(const Napi::CallbackInfo& info);
  Napi::Value Conj(const Napi::CallbackInfo& info);
  Napi::Value ToDense(const Napi::CallbackInfo& info);
  Napi::Value ToSparseCoo(const Napi::CallbackInfo& info);
  Napi::Value ToSparseCsr(const Napi::CallbackInfo& info);
  Napi::Value ToSparseCsc(const Napi::CallbackInfo& info);
  Napi::Value Coalesce(const Napi::CallbackInfo& info);
  Napi::Value SparseIndices(const Napi::CallbackInfo& info);
  Napi::Value SparseValues(const Napi::CallbackInfo& info);
  Napi::Value CrowIndices(const Napi::CallbackInfo& info);
  Napi::Value ColIndices(const Napi::CallbackInfo& info);
  Napi::Value CcolIndices(const Napi::CallbackInfo& info);
  Napi::Value RowIndices(const Napi::CallbackInfo& info);
  Napi::Value SparseDim(const Napi::CallbackInfo& info);
  Napi::Value DenseDim(const Napi::CallbackInfo& info);
  Napi::Value Nnz(const Napi::CallbackInfo& info);
  Napi::Value RequiresGrad(const Napi::CallbackInfo& info);
  Napi::Value SetRequiresGrad(const Napi::CallbackInfo& info);
  Napi::Value IsLeaf(const Napi::CallbackInfo& info);
  Napi::Value RetainsGrad(const Napi::CallbackInfo& info);
  Napi::Value RetainGrad(const Napi::CallbackInfo& info);
  Napi::Value Grad(const Napi::CallbackInfo& info);
  Napi::Value SetGrad(const Napi::CallbackInfo& info);
  Napi::Value GradFnName(const Napi::CallbackInfo& info);
  Napi::Value RegisterHook(const Napi::CallbackInfo& info);
  Napi::Value RemoveHook(const Napi::CallbackInfo& info);
  Napi::Value DataPtr(const Napi::CallbackInfo& info);
  Napi::Value StoragePtr(const Napi::CallbackInfo& info);
  Napi::Value Item(const Napi::CallbackInfo& info);
  Napi::Value ToTypedArray(const Napi::CallbackInfo& info);
  Napi::Value ToTypedArrayAsync(const Napi::CallbackInfo& info);
  Napi::Value DataView(const Napi::CallbackInfo& info);
  Napi::Value CopyFrom(const Napi::CallbackInfo& info);
  Napi::Value Detach(const Napi::CallbackInfo& info);
  Napi::Value DetachInplace(const Napi::CallbackInfo& info);
  Napi::Value Clone(const Napi::CallbackInfo& info);
  Napi::Value Contiguous(const Napi::CallbackInfo& info);
  Napi::Value To(const Napi::CallbackInfo& info);
  Napi::Value Backward(const Napi::CallbackInfo& info);
  Napi::Value IndexGet(const Napi::CallbackInfo& info);
  Napi::Value IndexPut(const Napi::CallbackInfo& info);
  Napi::Value Equal(const Napi::CallbackInfo& info);
  Napi::Value FormatString(const Napi::CallbackInfo& info);

  void ReleaseTracking(Napi::Env env);

  at::Tensor tensor_;
  const c10::StorageImpl* tracked_storage_ = nullptr;
  napi_env env_ = nullptr;
  bool disposed_ = true;
};

void RegisterTensorHelpers(Napi::Env env, Napi::Object exports);

}  // namespace norch
