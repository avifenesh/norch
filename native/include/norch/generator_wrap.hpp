#pragma once

#include <ATen/core/Generator.h>
#include <napi.h>

namespace norch {

class GeneratorWrap : public Napi::ObjectWrap<GeneratorWrap> {
 public:
  static Napi::Function Init(Napi::Env env, Napi::Object exports);
  static Napi::Object NewInstance(Napi::Env env, at::Generator gen);
  [[nodiscard]] static bool IsGeneratorWrap(Napi::Env env, const Napi::Value& val);
  [[nodiscard]] static GeneratorWrap* TryUnwrap(Napi::Env env, const Napi::Value& val);
  [[nodiscard]] static at::Generator UnwrapGenerator(Napi::Env env, const Napi::Value& val);

  explicit GeneratorWrap(const Napi::CallbackInfo& info);
  ~GeneratorWrap() override = default;

  GeneratorWrap(const GeneratorWrap&) = delete;
  GeneratorWrap& operator=(const GeneratorWrap&) = delete;
  GeneratorWrap(GeneratorWrap&&) = delete;
  GeneratorWrap& operator=(GeneratorWrap&&) = delete;

  [[nodiscard]] const at::Generator& generator() const { return gen_; }
  void SetGenerator(at::Generator gen) { gen_ = std::move(gen); }

 private:
  Napi::Value ManualSeed(const Napi::CallbackInfo& info);
  Napi::Value Seed(const Napi::CallbackInfo& info);
  Napi::Value InitialSeed(const Napi::CallbackInfo& info);
  Napi::Value Device(const Napi::CallbackInfo& info);
  Napi::Value GetState(const Napi::CallbackInfo& info);
  Napi::Value SetState(const Napi::CallbackInfo& info);

  at::Generator gen_;
};

}  // namespace norch
