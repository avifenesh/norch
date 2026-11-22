# norch

PyTorch equivalent for Node.js backed directly by `libtorch` (`c10`, `ATen`, `torch::autograd`, `c10d`, Kineto profiler, DLPack, and `torch::jit` tracer & `GraphExecutor`) through a C++23 Node-API addon — **zero Python required**.

## Architecture

- **Native addon (`native/`)**: C++23 Node-API (`node-addon-api`) built with CMake, Ninja, and `mold` (Linux, macOS arm64, Windows MSVC). Wraps `at::Tensor` (including complex64/128, sparse COO/CSR/CSC, negative-step slicing, `indexPut_` accumulate, and pinned memory), `at::Generator`, `c10::Dispatcher`, `torch::autograd` (`backward`, `grad`, `NoGradGuard`, `InferenceMode`, `autocast`, tensor backward hooks, custom JS `AutogradFunction` with `markDirty`/`needsInputGrad`/`SavedVariable`, and forward-mode AD `forwardAD`), `c10d` distributed process groups (`ProcessGroupNCCL` + `TCPStore`/`FileStore` multi-process backend with async `c10d::Work` handles, shared-store subgroups, and `scatter`/`gather`/`allToAll`), CUDA runtime (`Stream`, `Event`, `CUDAGraph`, allocator controls), zero-copy `DLPack` (`toDlpack`/`fromDlpack`), native Kineto profiler (`torch.profiler`), pure-Node `torch::jit::tracer::trace` + `torch::jit::GraphExecutor` + `.pt` serialization (`torch.jit.trace` / `torch.jit.load` / `torch.compile`), `AOTIModelPackageLoader`, plus unboxed C++ fast paths for core kernels including fused `Adam`/`AdamW`, zero-sync fused AMP (`_amp_foreach_non_finite_check_and_unscale_` + `_amp_update_scale_`), `LSTM`/`GRU`/`RNN`, `Conv1d/2d/3d`, `ConvTranspose1d/2d/3d`, 1D/2D/3D pooling, `InstanceNorm`, `EmbeddingBag`, `PixelShuffle`, `Unfold`/`Fold`, and `scaledDotProductAttention`.
- **Full ATen coverage (`src/generated/aten_ops.ts`)**: `scripts/codegen.mjs` introspects 3,120 live `c10::Dispatcher` schemas from `libtorch` and generates typed `torch.ops.aten.*` wrappers covering 1,016 public ATen operators (1,332 typed overloads) routed through `callBoxed` and `c10::IValue`.
- **Pure-Node JIT & `torch.compile()` (`src/jit.ts`)**:
  - `torch.jit.trace(fnOrModule, exampleInputs)` traces natively in C++ `libtorch`, runs IR passes (`EliminateDeadCode`, `PeepholeOptimize`, `FuseAddMM`, `ConstantPropagation`), executes via `GraphExecutor`, and saves standalone `.pt` archives via `.save(path)` loadable by `torch.jit.load(path, device?)` with zero Python.
  - `torch.compile(fnOrModule, { mode })` caches shape/dtype/device/autocast-guarded C++ graphs while keeping `nn.Module` parameters live for full autograd training and `optimizer.step()`, and supports `mode: 'reduce-overhead'` automatic CUDA Graph capture & replay on CUDA.
- **TypeScript training layer (`src/`)**: `Tensor`, `Generator`, `nn.Module` (with weight-tying preservation across `.to()`, forward pre/post hooks, and `registerFullBackwardHook`), bucketed async `nn.parallel.DistributedDataParallel`, `nn.utils.weightNorm`/`spectralNorm`, `Parameter`, full standard layers (`Linear`, `Bilinear`, `Conv1d/2d/3d`, `ConvTranspose1d/2d/3d`, `MaxPool1d/2d/3d`, `AvgPool1d/2d/3d`, `AdaptiveAvgPool2d`, `AdaptiveMaxPool2d`, `Upsample`, `PixelShuffle`/`PixelUnshuffle`, `Unfold`/`Fold`, `LSTM`/`GRU`/`RNN`, `BatchNorm1d/2d/3d`, `SyncBatchNorm`, `InstanceNorm1d/2d`, `LayerNorm`, `RMSNorm`, `GroupNorm`, `Embedding`, `EmbeddingBag`, `Dropout`, `MultiheadAttention`, `TransformerBlock`, `TransformerEncoder`/`TransformerDecoder`), `linalg`, `fft`, `special`, `distributions`, `distributed`, `func` (`grad`, `valueAndGrad`, `jacrev`, `jacfwd`, `hessian`, `vmap`, `functionalCall`), `profiler`, `ao.quantization` (`FakeQuantize` STE + dynamic INT8 `QuantizedLinear`), losses, optimizers (`SGD`, fused `Adam`/`AdamW`, `RMSprop`, `Adagrad`, `Adadelta`, `Adamax`, `NAdam`, `RAdam`, `LBFGS`) with `stateDict`/`loadStateDict`, `optim.AveragedModel` (SWA/EMA), LR schedulers, zero-sync fused `amp.GradScaler` (`stepNoSync`) with `stateDict`, activation checkpointing (`utils.checkpoint`), gradient clipping, multi-threaded `DataLoader` (`worker_threads` prefetching supporting all dtypes including `float16`/`bfloat16`, `DistributedSampler`, `RandomSampler`, `pinMemory`), `utils.dlpack`, binary `safetensors` I/O (`torch.save`/`torch.load`), and unified training checkpoints (`torch.saveCheckpoint`/`torch.loadCheckpoint`).
- **Deterministic memory management**: Native `Symbol.dispose` (`using t = torch.randn(...)`), `torch.tidy()` / `torch.scope()` arenas, V8 external memory accounting (`napi_adjust_external_memory` deduplicated per `c10::StorageImpl*`), non-blocking `toTypedArrayAsync()`, and `ObjectWrap` finalizers.

## Requirements

- Node.js >= 22 (Node 26 recommended for native `using` / `Symbol.dispose`)
- C++23 compiler (GCC 14+, Clang 19+, or MSVC 2022+), CMake >= 3.25, Ninja
- `libtorch` (CPU, CUDA, or macOS MPS). Run `npx norch-setup download --backend=cpu` (or `cu128` / `mps`), or set `LIBTORCH=/path/to/libtorch`.

## Build, Test, and Benchmark

```bash
npm install
npm run build             # compiles native/ -> runs ATen codegen -> compiles TypeScript
npm test                  # runs full CPU test suite (parity, training, DDP, workers, JIT/compile, profiler, DLPack, func)
npm run test:asan         # builds and runs all native tests under ASan + UBSan
npm run test:cuda         # runs CUDA tests (parity, Streams, Events, CUDAGraph, fused AdamW, zero-sync AMP, JIT/compile, NCCL)
npm run bench             # compares CPU latency against Python PyTorch on 4 threads
npm run bench:cuda        # compares CUDA latency against Python PyTorch on GPU
npm run lint:cpp          # runs clang-tidy across native/src/*.cpp
npm run stage:prebuild    # stages prebuilds/<platform>-<arch>/norch_native.node + manifest
```

## Verified RTX 5090 Benchmarks (`libtorch 2.15.0+cu132`)

| Benchmark | `norch` | Python PyTorch | Notes |
| --- | ---: | ---: | --- |
| `matmul [2048,2048] bf16` | `0.209 ms` | `0.209 ms` | **1.00x** parity |
| `TransformerBlock(d=512, B=16, T=128) bf16` fwd+bwd+fusedAdamW | `3.07–3.94 ms` | `2.27–3.16 ms` | `torch.compile` runs full forward in 1 C++ call (`3.84 ms`) |
| `torch.compile(..., { mode: 'reduce-overhead' })` (8 ops `[1024,1024] bf16`) | `0.056 ms` | — | C++ `GraphExecutor` fusion + CUDAGraph replay (faster than raw CUDAGraph `0.064 ms`) |

## Quick Example

```ts
import torch, { nn, optim } from 'norch';

class MLP extends nn.Module {
  readonly fc1 = new nn.Linear(4, 32);
  readonly fc2 = new nn.Linear(32, 1);

  forward(x: torch.Tensor): torch.Tensor {
    return this.fc2.forward(this.fc1.forward(x).gelu());
  }
}

using model = new MLP();
using compiled = torch.compile(model);
using optimizer = new optim.AdamW(compiled.parameters(), { lr: 1e-2, fused: true });
using x = torch.randn([64, 4]);
using y = torch.randn([64, 1]);

for (let step = 0; step < 50; step++) {
  const loss = torch.tidy(() => {
    optimizer.zeroGrad();
    const pred = compiled.forward(x) as torch.Tensor;
    const l = torch.F.mseLoss(pred, y);
    l.backward();
    optimizer.step();
    return l.item();
  });
  if (step % 10 === 0) console.log({ step, loss });
}

// Pure-Node TorchScript trace -> save (.pt) -> load (.pt), zero Python required
model.eval();
using traced = torch.jit.trace(model, x);
traced.save('mlp.pt');
using reloaded = torch.jit.load('mlp.pt');
```

## License

MIT — see [LICENSE](LICENSE).
