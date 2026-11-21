#pragma once

#include <napi.h>

namespace norch {

void RegisterProfiler(Napi::Env env, Napi::Object exports);

}  // namespace norch
