#include "norch/jit_compile.hpp"

#include <ATen/ATen.h>
#include <ATen/core/class_type.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/qualified_name.h>
#include <c10/core/GradMode.h>
#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "norch/addon_state.hpp"
#include "norch/convert.hpp"
#include "norch/tensor_wrap.hpp"

namespace norch {

namespace {

struct TracedGraphPayload {
  std::shared_ptr<torch::jit::Graph> graph;
  std::unique_ptr<torch::jit::GraphExecutor> executor;
  std::size_t num_inputs = 0;
  std::size_t num_outputs = 0;
  bool single_output = true;
  std::string name;
};

class TracedGraphWrap : public Napi::ObjectWrap<TracedGraphWrap> {
 public:
  static void Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(
        env,
        "NativeTracedGraph",
        {
            InstanceMethod("run", &TracedGraphWrap::Run),
            InstanceMethod("graphString", &TracedGraphWrap::GraphString),
            InstanceMethod("optimizedGraphString", &TracedGraphWrap::OptimizedGraphString),
            InstanceMethod("numInputs", &TracedGraphWrap::NumInputs),
            InstanceMethod("numOutputs", &TracedGraphWrap::NumOutputs),
            InstanceMethod("save", &TracedGraphWrap::Save),
            InstanceMethod("dispose", &TracedGraphWrap::Dispose),
            InstanceMethod("isDisposed", &TracedGraphWrap::IsDisposed),
        });
    AddonState::Get(env).traced_graph_ctor = Napi::Persistent(func);
    exports.Set("NativeTracedGraph", func);
  }

  [[nodiscard]] static Napi::Object NewInstance(
      Napi::Env env,
      std::unique_ptr<TracedGraphPayload> payload) {
    auto ext = Napi::External<TracedGraphPayload>::New(env, payload.release());
    return AddonState::Get(env).traced_graph_ctor.New({ext});
  }

  explicit TracedGraphWrap(const Napi::CallbackInfo& info)
      : Napi::ObjectWrap<TracedGraphWrap>(info) {
    const Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsExternal()) {
      Napi::TypeError::New(env, "NativeTracedGraph cannot be constructed directly; use traceGraph()")
          .ThrowAsJavaScriptException();
      return;
    }
    auto* raw = info[0].As<Napi::External<TracedGraphPayload>>().Data();
    payload_.reset(raw);
  }

  ~TracedGraphWrap() override = default;
  TracedGraphWrap(const TracedGraphWrap&) = delete;
  TracedGraphWrap& operator=(const TracedGraphWrap&) = delete;
  TracedGraphWrap(TracedGraphWrap&&) = delete;
  TracedGraphWrap& operator=(TracedGraphWrap&&) = delete;

 private:
  void EnsureAlive(Napi::Env env) const {
    if (!payload_ || !payload_->graph || !payload_->executor) {
      throw Napi::Error::New(env, "TracedGraph has been disposed");
    }
  }

  Napi::Value Run(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      EnsureAlive(env);
      if (info.Length() < 1 || !info[0].IsArray()) {
        throw Napi::TypeError::New(env, "run(inputs) expects an Array of Tensors");
      }
      const auto arr = info[0].As<Napi::Array>();
      const std::size_t len = arr.Length();
      if (len != payload_->num_inputs) {
        throw Napi::RangeError::New(
            env,
            "TracedGraph expected " + std::to_string(payload_->num_inputs) + " inputs, got " +
                std::to_string(len));
      }

      torch::jit::Stack stack;
      stack.reserve(len);
      for (std::size_t i = 0; i < len; ++i) {
        stack.emplace_back(TensorWrap::UnwrapTensor(env, arr.Get(static_cast<std::uint32_t>(i))));
      }

      payload_->executor->run(stack);

      if (payload_->single_output && stack.size() == 1) {
        return IValueToJs(env, stack[0]);
      }
      auto out = Napi::Array::New(env, stack.size());
      for (std::size_t i = 0; i < stack.size(); ++i) {
        out.Set(static_cast<std::uint32_t>(i), IValueToJs(env, stack[i]));
      }
      return out;
    });
  }

  Napi::Value GraphString(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      EnsureAlive(env);
      return Napi::String::New(env, payload_->graph->toString());
    });
  }

  Napi::Value OptimizedGraphString(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      EnsureAlive(env);
      if (info.Length() >= 1 && info[0].IsArray()) {
        const auto arr = info[0].As<Napi::Array>();
        torch::jit::Stack stack;
        stack.reserve(arr.Length());
        for (std::uint32_t i = 0; i < arr.Length(); ++i) {
          stack.emplace_back(TensorWrap::UnwrapTensor(env, arr.Get(i)));
        }
        const auto& plan = payload_->executor->getPlanFor(stack);
        if (plan.graph) {
          return Napi::String::New(env, plan.graph->toString());
        }
      }
      const auto state = payload_->executor->getDebugState();
      if (!state.execution_plans.empty()) {
        const auto& first_plan = state.execution_plans.begin()->second;
        if (first_plan.graph) {
          return Napi::String::New(env, first_plan.graph->toString());
        }
      }
      return Napi::String::New(env, payload_->graph->toString());
    });
  }

  Napi::Value NumInputs(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      EnsureAlive(env);
      return Napi::Number::New(env, static_cast<double>(payload_->num_inputs));
    });
  }

  Napi::Value NumOutputs(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      EnsureAlive(env);
      return Napi::Number::New(env, static_cast<double>(payload_->num_outputs));
    });
  }

  Napi::Value Save(const Napi::CallbackInfo& info) {
    const Napi::Env env = info.Env();
    return GuardNapi(env, [&]() -> Napi::Value {
      EnsureAlive(env);
      if (info.Length() < 1 || !info[0].IsString()) {
        throw Napi::TypeError::New(env, "save(filePath) expects a file path string");
      }
      const std::string file_path = info[0].As<Napi::String>().Utf8Value();

      auto save_graph = payload_->graph->copy();
      auto cu = std::make_shared<torch::jit::CompilationUnit>();
      torch::jit::Module mod(
          c10::QualifiedName("__torch__", payload_->name), cu, /*shouldMangle=*/true);

      auto* self_val = save_graph->insertInput(0, "self");
      self_val->setType(mod.type());

      if (save_graph->outputs().empty()) {
        auto* none_val = save_graph->insertConstant(c10::IValue());
        save_graph->registerOutput(none_val);
      } else if (save_graph->outputs().size() > 1) {
        std::vector<torch::jit::Value*> outs(
            save_graph->outputs().begin(), save_graph->outputs().end());
        auto* tuple_node = save_graph->createTuple(outs);
        save_graph->insertNode(tuple_node);
        while (!save_graph->outputs().empty()) {
          save_graph->eraseOutput(save_graph->outputs().size() - 1);
        }
        save_graph->registerOutput(tuple_node->output());
      }

      std::vector<c10::Argument> schema_args;
      schema_args.reserve(save_graph->inputs().size());
      schema_args.emplace_back("self", mod.type());
      for (std::size_t i = 1; i < save_graph->inputs().size(); ++i) {
        schema_args.emplace_back(
            "x" + std::to_string(i - 1), save_graph->inputs()[i]->type());
      }
      std::vector<c10::Argument> schema_returns;
      schema_returns.emplace_back("", save_graph->outputs()[0]->type());
      c10::FunctionSchema schema(
          "forward", "", std::move(schema_args), std::move(schema_returns));

      const auto method_name = c10::QualifiedName(*mod.type()->name(), "forward");
      torch::jit::Function* fn = cu->create_function(method_name, save_graph);
      fn->setSchema(std::move(schema));
      mod.type()->addMethod(fn);
      mod.save(file_path);
      return env.Undefined();
    });
  }

  Napi::Value Dispose(const Napi::CallbackInfo& info) {
    payload_.reset();
    return info.Env().Undefined();
  }

  Napi::Value IsDisposed(const Napi::CallbackInfo& info) {
    return Napi::Boolean::New(info.Env(), !payload_);
  }

  std::unique_ptr<TracedGraphPayload> payload_;
};

Napi::Value TraceGraphOp(const Napi::CallbackInfo& info) {
  const Napi::Env env = info.Env();
  return GuardNapi(env, [&]() -> Napi::Value {
    if (info.Length() < 2 || !info[0].IsFunction() || !info[1].IsArray()) {
      throw Napi::TypeError::New(
          env, "traceGraph(fn, exampleInputs, opts?) expects (Function, Tensor[], Object?)");
    }

    const auto js_fn = info[0].As<Napi::Function>();
    const auto example_tensors = ParseTensorList(env, info[1]);

    bool strict = false;
    bool optimize = true;
    bool profiling_executor = true;
    bool freeze_constants = false;
    std::string name = "NorchTracedModule";
    if (info.Length() >= 3 && info[2].IsObject()) {
      const auto opts = info[2].As<Napi::Object>();
      if (opts.Has("strict") && !opts.Get("strict").IsUndefined()) {
        strict = opts.Get("strict").ToBoolean().Value();
      }
      if (opts.Has("optimize") && !opts.Get("optimize").IsUndefined()) {
        optimize = opts.Get("optimize").ToBoolean().Value();
        profiling_executor = optimize;
      }
      if (opts.Has("profilingExecutor") && !opts.Get("profilingExecutor").IsUndefined()) {
        profiling_executor = opts.Get("profilingExecutor").ToBoolean().Value();
      }
      if (opts.Has("freezeConstants") && !opts.Get("freezeConstants").IsUndefined()) {
        freeze_constants = opts.Get("freezeConstants").ToBoolean().Value();
      }
      if (opts.Has("name") && opts.Get("name").IsString()) {
        name = opts.Get("name").As<Napi::String>().Utf8Value();
      }
    }

    torch::jit::Stack trace_inputs;
    trace_inputs.reserve(example_tensors.size());
    std::vector<std::string> arg_names;
    arg_names.reserve(example_tensors.size());
    for (std::size_t i = 0; i < example_tensors.size(); ++i) {
      trace_inputs.emplace_back(example_tensors[i].detach());
      arg_names.push_back("x" + std::to_string(i));
    }

    bool single_output = true;
    auto traced_fn = [&](torch::jit::Stack traced_stack) -> torch::jit::Stack {
      try {
        auto js_inputs = Napi::Array::New(env, traced_stack.size());
        for (std::size_t i = 0; i < traced_stack.size(); ++i) {
          js_inputs.Set(
              static_cast<std::uint32_t>(i),
              TensorWrap::NewInstance(env, traced_stack[i].toTensor()));
        }

        const Napi::Value result = js_fn.Call({js_inputs});
        torch::jit::Stack out_stack;
        if (TensorWrap::IsTensorWrap(env, result)) {
          single_output = true;
          out_stack.emplace_back(TensorWrap::UnwrapTensor(env, result));
        } else if (result.IsArray()) {
          single_output = false;
          const auto out_arr = result.As<Napi::Array>();
          out_stack.reserve(out_arr.Length());
          for (std::uint32_t i = 0; i < out_arr.Length(); ++i) {
            out_stack.emplace_back(TensorWrap::UnwrapTensor(env, out_arr.Get(i)));
          }
        } else {
          throw Napi::TypeError::New(
              env, "traceGraph callback must return a NativeTensor or NativeTensor[]");
        }
        return out_stack;
      } catch (...) {
        torch::jit::tracer::abandon();
        throw;
      }
    };

    std::shared_ptr<torch::jit::Graph> graph;
    std::size_t num_outputs = 0;
    {
      const c10::NoGradGuard no_grad;
      auto [state, out_stack] = torch::jit::tracer::trace(
          std::move(trace_inputs),
          traced_fn,
          [](const torch::autograd::Variable&) -> std::string { return ""; },
          strict,
          /*force_outplace=*/false,
          /*self=*/nullptr,
          arg_names);
      graph = std::move(state->graph);
      num_outputs = out_stack.size();
    }

    if (optimize) {
      torch::jit::EliminateDeadCode(graph);
      torch::jit::PeepholeOptimize(graph);
      torch::jit::FuseAddMM(graph);
      if (freeze_constants) {
        torch::jit::ConstantPropagation(graph);
        torch::jit::EliminateDeadCode(graph);
      }
    }

    auto payload = std::make_unique<TracedGraphPayload>();
    payload->graph = graph;
    payload->executor = profiling_executor
                            ? std::make_unique<torch::jit::GraphExecutor>(graph, name)
                            : std::make_unique<torch::jit::GraphExecutor>(
                                  graph, name, torch::jit::ExecutorExecutionMode::SIMPLE);
    payload->num_inputs = example_tensors.size();
    payload->num_outputs = num_outputs;
    payload->single_output = single_output;
    payload->name = std::move(name);

    return TracedGraphWrap::NewInstance(env, std::move(payload));
  });
}

}  // namespace

void RegisterJitCompile(Napi::Env env, Napi::Object exports) {
  TracedGraphWrap::Init(env, exports);
  exports.Set("traceGraph", Napi::Function::New(env, TraceGraphOp));
}

}  // namespace norch
