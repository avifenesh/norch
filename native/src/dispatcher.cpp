#include "norch/dispatcher.hpp"

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "norch/convert.hpp"
#include "norch/tensor_wrap.hpp"

namespace norch {

namespace {

[[nodiscard]] std::string NormalizeOpName(std::string name) {
  if (name.find("::") == std::string::npos) {
    return "aten::" + name;
  }
  return name;
}

[[nodiscard]] Napi::Object SerializeSchema(
    Napi::Env env,
    const c10::OperatorName& op_name,
    const c10::FunctionSchema& schema) {
  auto obj = Napi::Object::New(env);
  obj.Set("name", Napi::String::New(env, op_name.name));
  obj.Set("overload", Napi::String::New(env, op_name.overload_name));
  obj.Set("isMutable", Napi::Boolean::New(env, schema.is_mutable()));

  const auto& args = schema.arguments();
  auto js_args = Napi::Array::New(env, args.size());
  for (std::size_t i = 0; i < args.size(); ++i) {
    const auto& arg = args[i];
    auto a = Napi::Object::New(env);
    a.Set("name", Napi::String::New(env, arg.name()));
    a.Set("type", Napi::String::New(env, arg.type()->repr_str()));
    a.Set("realType", Napi::String::New(env, arg.real_type()->repr_str()));
    a.Set("kwargOnly", Napi::Boolean::New(env, arg.kwarg_only()));
    a.Set("isOut", Napi::Boolean::New(env, arg.is_out()));
    a.Set("hasDefault", Napi::Boolean::New(env, arg.default_value().has_value()));
    js_args.Set(static_cast<std::uint32_t>(i), a);
  }
  obj.Set("arguments", js_args);

  const auto& rets = schema.returns();
  auto js_rets = Napi::Array::New(env, rets.size());
  for (std::size_t i = 0; i < rets.size(); ++i) {
    const auto& ret = rets[i];
    auto r = Napi::Object::New(env);
    r.Set("name", Napi::String::New(env, ret.name()));
    r.Set("type", Napi::String::New(env, ret.type()->repr_str()));
    js_rets.Set(static_cast<std::uint32_t>(i), r);
  }
  obj.Set("returns", js_rets);
  return obj;
}

[[nodiscard]] bool TryPackStack(
    Napi::Env env,
    const c10::FunctionSchema& schema,
    const Napi::Array& pos_args,
    const std::optional<Napi::Object>& kw_args,
    torch::jit::Stack& stack,
    std::string& error_out) {
  const auto& schema_args = schema.arguments();
  const std::size_t num_pos = pos_args.Length();

  // Check for unknown keyword arguments early
  if (kw_args.has_value()) {
    const auto keys = kw_args->GetPropertyNames();
    for (std::uint32_t k = 0; k < keys.Length(); ++k) {
      const std::string key = keys.Get(k).As<Napi::String>().Utf8Value();
      const auto val = kw_args->Get(key);
      if (val.IsUndefined()) {
        continue;
      }
      bool found = false;
      for (const auto& sa : schema_args) {
        if (sa.name() == key) {
          found = true;
          break;
        }
      }
      if (!found) {
        error_out = "Unknown keyword argument '" + key + "' for " + schema.name();
        return false;
      }
    }
  }

  std::size_t pos_slots = 0;
  for (const auto& sa : schema_args) {
    if (!sa.kwarg_only()) {
      pos_slots += 1;
    }
  }
  if (num_pos > pos_slots) {
    error_out = "Too many positional arguments for " + schema.name() + " (expected at most " +
                std::to_string(pos_slots) + ", got " + std::to_string(num_pos) + ")";
    return false;
  }

  stack.clear();
  stack.reserve(schema_args.size());

  for (std::size_t i = 0; i < schema_args.size(); ++i) {
    const auto& sa = schema_args[i];
    const bool has_pos = (!sa.kwarg_only() && i < num_pos);
    bool has_kw = false;
    Napi::Value kw_val = env.Undefined();
    if (kw_args.has_value() && kw_args->Has(sa.name())) {
      kw_val = kw_args->Get(sa.name());
      has_kw = !kw_val.IsUndefined();
    }

    if (has_pos && has_kw) {
      error_out = "Argument '" + sa.name() + "' supplied both positionally and by keyword";
      return false;
    }

    if (has_pos) {
      try {
        const auto raw_val = pos_args.Get(static_cast<std::uint32_t>(i));
        if (raw_val.IsUndefined() && sa.default_value().has_value()) {
          stack.push_back(*sa.default_value());
        } else {
          stack.push_back(JsToIValue(env, raw_val, sa.real_type()));
        }
      } catch (const std::exception& ex) {
        error_out = "Arg '" + sa.name() + "': " + ex.what();
        return false;
      }
    } else if (has_kw) {
      try {
        stack.push_back(JsToIValue(env, kw_val, sa.real_type()));
      } catch (const std::exception& ex) {
        error_out = "Kwarg '" + sa.name() + "': " + ex.what();
        return false;
      }
    } else if (sa.default_value().has_value()) {
      stack.push_back(*sa.default_value());
    } else if (sa.type()->kind() == c10::TypeKind::OptionalType) {
      stack.emplace_back();
    } else {
      error_out = "Missing required argument '" + sa.name() + "' for " + schema.name();
      return false;
    }
  }
  return true;
}

[[nodiscard]] Napi::Value UnpackStackReturns(Napi::Env env, const torch::jit::Stack& stack) {
  if (stack.empty()) {
    return env.Undefined();
  }
  if (stack.size() == 1) {
    return IValueToJs(env, stack[0]);
  }
  auto arr = Napi::Array::New(env, stack.size());
  for (std::size_t i = 0; i < stack.size(); ++i) {
    arr.Set(static_cast<std::uint32_t>(i), IValueToJs(env, stack[i]));
  }
  return arr;
}

Napi::Value ListOps(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    std::string prefix;
    if (info.Length() >= 1 && info[0].IsString()) {
      prefix = info[0].As<Napi::String>().Utf8Value();
    }
    const auto all_ops = c10::Dispatcher::singleton().getAllOpNames();
    std::vector<c10::OperatorName> filtered;
    filtered.reserve(all_ops.size());
    for (const auto& op : all_ops) {
      if (prefix.empty() || op.name.starts_with(prefix)) {
        filtered.push_back(op);
      }
    }
    auto arr = Napi::Array::New(env, filtered.size());
    for (std::size_t i = 0; i < filtered.size(); ++i) {
      auto item = Napi::Object::New(env);
      item.Set("name", Napi::String::New(env, filtered[i].name));
      item.Set("overload", Napi::String::New(env, filtered[i].overload_name));
      arr.Set(static_cast<std::uint32_t>(i), item);
    }
    return arr;
  });
}

Napi::Value GetOpSchema(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    if (info.Length() < 1 || !info[0].IsString()) {
      throw Napi::TypeError::New(env, "getOpSchema(name, overload?) expects op name string");
    }
    const std::string name = NormalizeOpName(info[0].As<Napi::String>().Utf8Value());
    const std::string overload =
        (info.Length() >= 2 && info[1].IsString()) ? info[1].As<Napi::String>().Utf8Value() : "";
    const auto handle = c10::Dispatcher::singleton().findSchema({name, overload});
    if (!handle.has_value()) {
      return env.Null();
    }
    return SerializeSchema(env, handle->operator_name(), handle->schema());
  });
}

Napi::Value GetAllSchemas(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    std::string prefix = "aten::";
    if (info.Length() >= 1 && info[0].IsString()) {
      prefix = info[0].As<Napi::String>().Utf8Value();
    }
    auto& disp = c10::Dispatcher::singleton();
    const auto all_ops = disp.getAllOpNames();
    auto arr = Napi::Array::New(env);
    std::uint32_t idx = 0;
    for (const auto& op : all_ops) {
      if (!prefix.empty() && !op.name.starts_with(prefix)) {
        continue;
      }
      const auto handle = disp.findSchema(op);
      if (!handle.has_value()) {
        continue;
      }
      arr.Set(idx++, SerializeSchema(env, op, handle->schema()));
    }
    return arr;
  });
}

Napi::Value CallOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    if (info.Length() < 3 || !info[0].IsString() || !info[1].IsString() || !info[2].IsArray()) {
      throw Napi::TypeError::New(
          env, "callOp(name, overload, posArgs, kwArgs?) expects (string, string, Array, Object?)");
    }
    const std::string name = NormalizeOpName(info[0].As<Napi::String>().Utf8Value());
    const std::string overload = info[1].As<Napi::String>().Utf8Value();
    const auto pos_args = info[2].As<Napi::Array>();
    std::optional<Napi::Object> kw_args = std::nullopt;
    if (info.Length() >= 4 && info[3].IsObject() && !info[3].IsNull()) {
      kw_args = info[3].As<Napi::Object>();
    }

    const auto handle = c10::Dispatcher::singleton().findSchema({name, overload});
    if (!handle.has_value()) {
      throw Napi::Error::New(
          env, "Operator not found in c10::Dispatcher: " + name + (overload.empty() ? "" : "." + overload));
    }

    torch::jit::Stack stack;
    std::string err;
    if (!TryPackStack(env, handle->schema(), pos_args, kw_args, stack, err)) {
      throw Napi::TypeError::New(env, err);
    }
    handle->callBoxed(&stack);
    return UnpackStackReturns(env, stack);
  });
}

Napi::Value CallOpAuto(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    if (info.Length() < 2 || !info[0].IsString() || !info[1].IsArray()) {
      throw Napi::TypeError::New(env, "callOpAuto(name, posArgs, kwArgs?) expects (string, Array, Object?)");
    }
    const std::string name = NormalizeOpName(info[0].As<Napi::String>().Utf8Value());
    const auto pos_args = info[1].As<Napi::Array>();
    std::optional<Napi::Object> kw_args = std::nullopt;
    if (info.Length() >= 3 && info[2].IsObject() && !info[2].IsNull()) {
      kw_args = info[2].As<Napi::Object>();
    }

    auto& disp = c10::Dispatcher::singleton();
    // Cache overloads grouped by full op name
    static std::unordered_map<std::string, std::vector<c10::OperatorHandle>> overload_cache;
    auto it = overload_cache.find(name);
    if (it == overload_cache.end()) {
      std::vector<c10::OperatorHandle> handles;
      for (const auto& op : disp.getAllOpNames()) {
        if (op.name == name) {
          const auto h = disp.findSchema(op);
          if (h.has_value()) {
            handles.push_back(*h);
          }
        }
      }
      // Prefer default (empty) overload first, then non-out overloads, then out overloads
      std::ranges::stable_sort(
          handles,
          [](const c10::OperatorHandle& a, const c10::OperatorHandle& b) {
            const auto rank = [](const c10::OperatorHandle& h) -> int {
              const auto& ov = h.operator_name().overload_name;
              if (ov.empty()) return 0;
              if (ov == "Tensor") return 1;
              if (ov == "Scalar") return 2;
              if (ov.find("out") != std::string::npos) return 4;
              return 3;
            };
            return rank(a) < rank(b);
          });
      it = overload_cache.emplace(name, std::move(handles)).first;
    }

    const auto& candidates = it->second;
    if (candidates.empty()) {
      throw Napi::Error::New(env, "Operator not found in c10::Dispatcher: " + name);
    }

    std::string errors;
    for (const auto& handle : candidates) {
      torch::jit::Stack stack;
      std::string err;
      if (TryPackStack(env, handle.schema(), pos_args, kw_args, stack, err)) {
        handle.callBoxed(&stack);
        return UnpackStackReturns(env, stack);
      }
      if (!errors.empty()) errors += "; ";
      errors += "[" + handle.operator_name().overload_name + "] " + err;
    }
    throw Napi::TypeError::New(
        env, "No overload matched for " + name + ": " + errors);
  });
}

}  // namespace

void RegisterDispatcher(Napi::Env env, Napi::Object exports) {
  exports.Set("listOps", Napi::Function::New(env, ListOps));
  exports.Set("getOpSchema", Napi::Function::New(env, GetOpSchema));
  exports.Set("getAllSchemas", Napi::Function::New(env, GetAllSchemas));
  exports.Set("callOp", Napi::Function::New(env, CallOp));
  exports.Set("callOpAuto", Napi::Function::New(env, CallOpAuto));
}

}  // namespace norch
