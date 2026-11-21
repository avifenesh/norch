#include "norch/addon_state.hpp"

#include <napi.h>

#include <cstdint>
#include <thread>

namespace norch {

AddonState::AddonState(std::thread::id js_thread) : js_thread_id_(js_thread) {}

AddonState& AddonState::Get(Napi::Env env) {
  auto* state = env.GetInstanceData<AddonState>();
  if (state == nullptr) {
    throw Napi::Error::New(env, "norch AddonState is not initialized");
  }
  return *state;
}

void AddonState::TrackStorage(Napi::Env env, const c10::StorageImpl* storage, std::int64_t nbytes) {
  if (storage == nullptr) {
    return;
  }
  auto& rec = tracked_storages[storage];
  if (rec.ref_count == 0 && nbytes > 0) {
    rec.reported_bytes = nbytes;
    total_reported_bytes += nbytes;
    Napi::MemoryManagement::AdjustExternalMemory(env, nbytes);
  }
  rec.ref_count += 1;
}

void AddonState::UntrackStorage(Napi::Env env, const c10::StorageImpl* storage) {
  if (storage == nullptr) {
    return;
  }
  auto it = tracked_storages.find(storage);
  if (it == tracked_storages.end()) {
    return;
  }
  if (it->second.ref_count > 1) {
    it->second.ref_count -= 1;
    return;
  }
  const std::int64_t bytes = it->second.reported_bytes;
  tracked_storages.erase(it);
  if (bytes > 0) {
    total_reported_bytes -= bytes;
    Napi::MemoryManagement::AdjustExternalMemory(env, -bytes);
  }
}

}  // namespace norch
