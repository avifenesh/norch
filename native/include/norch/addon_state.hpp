#pragma once

#include <napi.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <unordered_map>

namespace c10 {
struct StorageImpl;
}  // namespace c10

namespace norch {

struct StorageMemRecord {
  std::size_t ref_count = 0;
  std::int64_t reported_bytes = 0;
};

class AddonState {
 public:
  explicit AddonState(std::thread::id js_thread);

  [[nodiscard]] static AddonState& Get(Napi::Env env);

  [[nodiscard]] std::thread::id js_thread_id() const noexcept { return js_thread_id_; }

  Napi::FunctionReference tensor_ctor;
  std::unordered_map<const c10::StorageImpl*, StorageMemRecord> tracked_storages;
  std::int64_t total_reported_bytes = 0;
  std::uint64_t live_tensors = 0;
  std::atomic<std::int64_t> custom_autograd_nodes_alive{0};

  void TrackStorage(Napi::Env env, const c10::StorageImpl* storage, std::int64_t nbytes);
  void UntrackStorage(Napi::Env env, const c10::StorageImpl* storage);

 private:
  std::thread::id js_thread_id_;
};

}  // namespace norch
