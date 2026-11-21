# norch

PyTorch equivalent for Node.js backed directly by `libtorch` (`c10`, `ATen`, `torch::autograd`, `c10d`, `AOTInductor`, and `torch::jit`) through a C++23 Node-API addon.

## Architecture

- **Native addon (`native/`)**: C++23 Node-API (`node-addon-api`) built with CMake, Ninja, and `mold`. Wraps `at::Tensor` (including complex64/128, sparse COO, negative-step slicing, and pinned memory), `at::Generator`, `c10::Dispatcher`, `torch::autograd` (`backward`, `grad`, `NoGradGuard`, `InferenceMode`, `autocast`, tensor backward hooks, custom JS `AutogradFunction` with `markDirty`/`needsInputGrad`/`SavedVariable`, and forward-mode AD `forwardAD`), `c10d` distributed process groups (`ProcessGroupNCCL` + `TCPStore`/`FileStore` multi-process backend), CUDA runtime (`Stream`, `Event`, `CUDAGraph`, allocator controls), `AOTIModelPackageLoader` (`.pt2`) and `torch::jit::load` (`.pt`), plus unboxed C++ fast paths for core kernels including fused `Adam`/`AdamW`, `LSTM`/`GRU`/`RNN`, `Conv1d/2d/3d`, transposed convolutions, and `scaledDotProductAttention`.
- **Full ATen coverage (`src/generated/aten_ops.ts`)**: `scripts/codegen.mjs` introspects 3,120 live `c10::Dispatcher` schemas from `libtorch` and generates typed `torch.ops.aten.*` wrappers covering 1,016 public ATen operators (1,332 typed overloads) routed through `callBoxed` and `c10::IValue`.
- **TypeScript training layer (`src/`)**: `Tensor`, `Generator`, `nn.Module` (with weight-tying preservation across `.to()` and forward pre/post hooks), `nn.parallel.DistributedDataParallel`, `Parameter`, standard layers (`Linear`, `Conv1d/2d/3d`, `ConvTranspose1d/2d`, `Upsample`, `LSTM`/`GRU`/`RNN`, `BatchNorm1d/2d`, `LayerNorm`, `RMSNorm`, `GroupNorm`, `Embedding`, `Dropout`, `MultiheadAttention`, `TransformerBlock`), `linalg`, `fft`, `special`, `distributions`, `distributed`, losses, optimizers (`SGD`, fused `Adam`/`AdamW`, `RMSprop`) with `stateDict`/`loadStateDict`, LR schedulers (`StepLR`, `CosineAnnealingLR`, `LinearLR`, `OneCycleLR`, `SequentialLR`, `ReduceLROnPlateau`), `amp.GradScaler`, activation checkpointing (`utils.checkpoint`), gradient clipping, multi-threaded `DataLoader` (`worker_threads` prefetching, `DistributedSampler`, `RandomSampler`, `pinMemory`), binary `safetensors` I/O (`torch.save`/`torch.load`), and unified training checkpoints (`torch.saveCheckpoint`/`torch.loadCheckpoint`).
- **Deterministic memory management**: Native `Symbol.dispose` (`using t = torch.randn(...)`), `torch.tidy()` / `torch.scope()` arenas, V8 external memory accounting (`napi_adjust_external_memory` deduplicated per `c10::StorageImpl*`), non-blocking `toTypedArrayAsync()`, and `ObjectWrap` finalizers.

## Requirements

- Node.js >= 22 (Node 26 recommended for native `using` / `Symbol.dispose`)
- C++23 compiler (GCC 14+ or Clang 19+), CMake >= 3.25, Ninja
- `libtorch` (CPU or CUDA). Run `npx norch-setup download --backend=cpu` (or `cu128`), set `LIBTORCH=/path/to/libtorch`, or let `scripts/find-libtorch.mjs` detect an existing Python PyTorch installation.

## Build, Test, and Benchmark

```bash
npm install
npm run build           # compiles native/ -> runs ATen codegen -> compiles TypeScript
npm test                # runs full CPU test suite (parity, training, DDP, workers, export)
npm run test:asan       # builds and runs native tests under ASan + UBSan
npm run test:cuda       # runs CUDA tests (Streams, Events, CUDAGraph, fused AdamW, NCCL)
npm run bench           # compares CPU latency against Python PyTorch on 4 threads
npm run lint:cpp        # runs clang-tidy across native/src/*.cpp
npm run prebuild        # stages prebuilds/<platform>-<arch>/norch_native.node + manifest
```

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
using optimizer = new optim.AdamW(model.parameters(), { lr: 1e-2 });
using x = torch.randn([64, 4]);
using y = torch.randn([64, 1]);

for (let step = 0; step < 50; step++) {
  const loss = torch.tidy(() => {
    optimizer.zeroGrad();
    const pred = model.forward(x);
    const l = torch.F.mseLoss(pred, y);
    l.backward();
    optimizer.step();
    return l.item();
  });
  if (step % 10 === 0) console.log({ step, loss });
}
```
