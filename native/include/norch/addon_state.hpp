#pragma once

#include <napi.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

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
  ~AddonState() noexcept;

  AddonState(const AddonState&) = delete;
  AddonState& operator=(const AddonState&) = delete;
  AddonState(AddonState&&) = delete;
  AddonState& operator=(AddonState&&) = delete;

  [[nodiscard]] static AddonState& Get(Napi::Env env);

  [[nodiscard]] std::thread::id js_thread_id() const noexcept { return js_thread_id_; }

  Napi::FunctionReference tensor_ctor;
  Napi::FunctionReference generator_ctor;
  Napi::FunctionReference cuda_stream_ctor;
  Napi::FunctionReference cuda_event_ctor;
  Napi::FunctionReference cuda_graph_ctor;
  Napi::FunctionReference aoti_model_ctor;
  Napi::FunctionReference jit_module_ctor;
  Napi::FunctionReference traced_graph_ctor;
  Napi::FunctionReference process_group_ctor;
  Napi::FunctionReference dist_work_ctor;
  Napi::FunctionReference profiler_result_ctor;

  std::unordered_map<const c10::StorageImpl*, StorageMemRecord> tracked_storages;
  std::int64_t total_reported_bytes = 0;
  std::uint64_t live_tensors = 0;
  std::atomic<std::int64_t> custom_autograd_nodes_alive{0};

  // Safe deferred release queue for zero-copy TypedArray pins freed off-thread
  std::shared_ptr<std::atomic<bool>> env_alive = std::make_shared<std::atomic<bool>>(true);
  std::mutex deferred_pins_mutex;
  std::vector<Napi::Reference<Napi::Value>> deferred_pin_releases;

  void EnqueuePinRelease(Napi::Reference<Napi::Value> ref);
  void DrainDeferredPinReleases();

  void TrackStorage(Napi::Env env, const c10::StorageImpl* storage, std::int64_t nbytes);
  void UntrackStorage(Napi::Env env, const c10::StorageImpl* storage);

 private:
  std::thread::id js_thread_id_;
};

}  // namespace norch
