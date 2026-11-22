#pragma once

#include <napi.h>

namespace norch {

void RegisterJitCompile(Napi::Env env, Napi::Object exports);

}  // namespace norch
