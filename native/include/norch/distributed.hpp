#pragma once

#include <napi.h>

namespace norch {

void RegisterDistributed(Napi::Env env, Napi::Object exports);

}  // namespace norch
