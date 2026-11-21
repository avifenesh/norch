#include <napi.h>

#include <memory>
#include <thread>

#include "norch/addon_state.hpp"
#include "norch/autograd.hpp"
#include "norch/cuda.hpp"
#include "norch/dispatcher.hpp"
#include "norch/export_runner.hpp"
#include "norch/generator_wrap.hpp"
#include "norch/ops_fast.hpp"
#include "norch/tensor_wrap.hpp"

namespace {

Napi::Object InitAddon(Napi::Env env, Napi::Object exports) {
  auto state = std::make_unique<norch::AddonState>(std::this_thread::get_id());
  env.SetInstanceData(state.release());

  norch::TensorWrap::Init(env, exports);
  norch::GeneratorWrap::Init(env, exports);
  norch::RegisterTensorHelpers(env, exports);
  norch::RegisterDispatcher(env, exports);
  norch::RegisterAutograd(env, exports);
  norch::RegisterFastOps(env, exports);
  norch::RegisterCudaAndRuntime(env, exports);
  norch::RegisterExportRunners(env, exports);
  return exports;
}

}  // namespace

NODE_API_MODULE(norch_native, InitAddon)
