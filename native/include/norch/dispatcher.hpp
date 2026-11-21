#pragma once

#include <napi.h>

namespace norch {

void RegisterDispatcher(Napi::Env env, Napi::Object exports);

}  // namespace norch
