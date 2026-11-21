#pragma once

#include <napi.h>

namespace norch {

void RegisterAutograd(Napi::Env env, Napi::Object exports);

}  // namespace norch
