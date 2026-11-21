# norch

PyTorch equivalent for Node.js backed directly by `libtorch` (`c10`, `ATen`, and `torch::autograd`) through a C++23 Node-API addon.

## Architecture

- **Native addon (`native/`)**: C++23 Node-API (`node-addon-api`) built with CMake, Ninja, and `mold`. Wraps `at::Tensor`, `c10::Dispatcher`, `torch::autograd` (`backward`, `grad`, `NoGradGuard`, `InferenceMode`, `autocast`, custom JS `AutogradFunction`), CUDA runtime/allocator controls, and unboxed C++ fast paths for core kernels.
- **Full ATen coverage (`src/generated/aten_ops.ts`)**: `scripts/codegen.mjs` introspects the live `c10::Dispatcher` registry from `libtorch` and generates typed `torch.ops.aten.*` wrappers for over 1,000 public ATen operators routed through `callBoxed` and `c10::IValue`.
- **TypeScript training layer (`src/`)**: `Tensor`, `nn.Module`, `Parameter`, standard layers (`Linear`, `Conv1d`, `Conv2d`, `BatchNorm1d/2d`, `LayerNorm`, `RMSNorm`, `GroupNorm`, `Embedding`, `Dropout`, `MultiheadAttention`, `TransformerBlock` over fused `scaledDotProductAttention`), losses, optimizers (`SGD`, `Adam`, `AdamW`), LR schedulers (`StepLR`, `CosineAnnealingLR`, `LinearLR`), `amp.GradScaler`, activation checkpointing (`utils.checkpoint`), gradient clipping, `DataLoader`, and binary `safetensors` I/O (`torch.save` / `torch.load`).
- **Deterministic memory management**: Native `Symbol.dispose` (`using t = torch.randn(...)`), `torch.tidy()` / `torch.scope()` arenas, V8 external memory accounting (`napi_adjust_external_memory` deduplicated per `c10::StorageImpl*`), and `ObjectWrap` finalizers.

## Requirements

- Node.js >= 22 (Node 26 recommended for native `using` / `Symbol.dispose`)
- C++23 compiler (GCC 14+ or Clang 19+), CMake >= 3.25, Ninja
- `libtorch` (CPU or CUDA). Set `LIBTORCH=/path/to/libtorch` or let `scripts/find-libtorch.mjs` detect an existing Python PyTorch installation.

## Build and Test

```bash
npm install
npm run build        # compiles native/ -> runs ATen codegen -> compiles TypeScript
npm test             # runs CPU unit, dispatcher, autograd, PyTorch parity, and e2e training tests
npm run test:cuda    # runs CUDA tests (requires a CUDA build and free GPU)
npm run lint:cpp     # runs clang-tidy across native/src/*.cpp
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
