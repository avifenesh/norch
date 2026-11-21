#pragma once

#include <napi.h>

namespace norch {

void RegisterCudaAndRuntime(Napi::Env env, Napi::Object exports);

}  // namespace norch
