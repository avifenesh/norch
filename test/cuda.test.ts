import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { describe, it } from 'node:test';
import torch, { distributed as dist, F, Generator, nn, optim, profiler, utils, type Tensor } from '../src/index.js';

const runCuda = process.env.NORCH_TEST_CUDA === '1';

function findPythonTorch(): string | null {
  const candidates = [
    process.env.NORCH_PYTHON,
    path.join(os.homedir(), 'projects', 'torch', '.venv', 'bin', 'python'),
    'python3',
  ].filter((p): p is string => Boolean(p));
  const env = { ...process.env };
  delete env.LD_PRELOAD;
  delete env.ASAN_OPTIONS;
  delete env.UBSAN_OPTIONS;
  for (const py of candidates) {
    if (py.includes(path.sep) && !fs.existsSync(py)) continue;
    try {
      execFileSync(py, ['-c', 'import torch; assert torch.cuda.is_available()'], {
        stdio: 'ignore',
        timeout: 15000,
        env,
      });
      return py;
    } catch {
      // ignore
    }
  }
  return null;
}

function assertCloseArrays(actual: number[], expected: number[], atol: number, rtol: number, label: string): void {
  assert.equal(actual.length, expected.length, `${label}: length mismatch`);
  for (let i = 0; i < actual.length; i++) {
    const a = actual[i]!;
    const e = expected[i]!;
    const tol = atol + rtol * Math.abs(e);
    assert.ok(
      Math.abs(a - e) <= tol,
      `${label}[${i}]: got ${a}, expected ${e} (diff=${Math.abs(a - e)}, tol=${tol})`
    );
  }
}

function getFreePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.unref();
    srv.on('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const addr = srv.address();
      if (!addr || typeof addr === 'string') {
        srv.close(() => reject(new Error('Failed to bind ephemeral port')));
        return;
      }
      const port = addr.port;
      srv.close((err) => (err ? reject(err) : resolve(port)));
    });
  });
}

describe('CUDA backend (guarded by NORCH_TEST_CUDA=1)', { skip: !runCuda }, () => {
  it('runs GPU training with fused AdamW, bfloat16/float16 autocast, fused GradScaler, and weight tying', () => {
    assert.equal(torch.cuda.isAvailable(), true);
    assert.ok(torch.cuda.deviceCount() >= 1);
    const props = torch.cuda.deviceProperties(0);
    assert.ok(props.name.length > 0 && props.totalMemory > 0);
    torch.cuda.resetPeakMemoryStats(0);

    class TiedGpuNet extends nn.Module {
      readonly emb = new nn.Embedding(32, 16);
      readonly proj = new nn.Linear(16, 16);
      readonly head = new nn.Linear(16, 32, { bias: false });
      constructor() {
        super();
        const old = this.head.weight;
        (this.head as unknown as { weight: nn.Parameter }).weight = this.emb.weight;
        old.dispose();
      }
      forward(toks: torch.Tensor): torch.Tensor {
        return this.head.forward(this.proj.forward(this.emb.forward(toks)).gelu());
      }
    }

    using model = new TiedGpuNet();
    model.to('cuda');
    assert.equal(model.emb.weight, model.head.weight);
    assert.equal(model.emb.weight.isCuda, true);

    using opt = new optim.AdamW(model.parameters(), { lr: 1e-2, fused: true });
    const scaler = new torch.amp.GradScaler({ initScale: 256, backoffFactor: 0.5 });

    const lossVal = torch.tidy(() => {
      const toks = torch.randint(0, 32, [4, 8], { device: 'cuda' });
      const targets = torch.randint(0, 32, [4, 8], { device: 'cuda' });
      const logits = torch.autocast('cuda', torch.bfloat16, () => model.forward(toks));
      assert.equal(logits.isCuda, true);
      const loss = torch.F.crossEntropy(logits.float().reshape(-1, 32), targets.reshape(-1));
      scaler.scale(loss).backward();
      assert.equal(scaler.step(opt), true);
      scaler.update();
      torch.cuda.synchronize();
      return loss.item();
    });

    assert.ok(Number.isFinite(lossVal));

    // Verify fused CUDA _amp_foreach_non_finite_check_and_unscale_ skips step on Inf & backs off
    torch.tidy(() => {
      opt.zeroGrad();
      const toks = torch.randint(0, 32, [4, 8], { device: 'cuda' });
      const targets = torch.randint(0, 32, [4, 8], { device: 'cuda' });
      const logits = torch.autocast('cuda', torch.float16, () => model.forward(toks));
      const loss = torch.F.crossEntropy(logits.float().reshape(-1, 32), targets.reshape(-1));
      scaler.scale(loss).backward();
      model.proj.weight.grad!.fill_(Number.POSITIVE_INFINITY);
      assert.equal(scaler.step(opt), false);
      scaler.update();
    });
    assert.equal(scaler.getScale(), 128);

    torch.cuda.emptyCache();
    assert.ok(torch.cuda.memoryStats().maxAllocated > 0);
  });

  it('supports CUDA Streams, Events timing, CUDAGraph capture/replay, Generator(cuda), and pinned multi-worker DataLoader', async () => {
    const g1 = new Generator('cuda').manualSeed(123);
    const g2 = new Generator('cuda').manualSeed(123);
    torch.tidy(() => {
      const a = torch.randn([16, 16], { device: 'cuda', generator: g1 });
      const b = torch.randn([16, 16], { device: 'cuda', generator: g2 });
      assert.ok(a.equal(b));
    });

    // Streams + Events timing
    const s = new torch.cuda.Stream();
    const startEv = new torch.cuda.Event({ enableTiming: true });
    const endEv = new torch.cuda.Event({ enableTiming: true });
    torch.cuda.stream(s, () => {
      torch.tidy(() => {
        const m = torch.randn([256, 256], { device: 'cuda' });
        startEv.record(s);
        m.matmul(m);
        endEv.record(s);
      });
    });
    endEv.synchronize();
    assert.ok(startEv.elapsedTime(endEv) >= 0);

    // CUDA Graph capture & replay
    using staticIn = torch.ones([16, 16], { device: 'cuda' });
    using staticOut = torch.zeros([16, 16], { device: 'cuda' });
    // Warmup
    torch.tidy(() => {
      staticOut.copy_(staticIn.mul(2).add(1));
    });
    torch.cuda.synchronize();

    const graph = new torch.cuda.CUDAGraph();
    torch.cuda.graph(graph, () => {
      torch.tidy(() => {
        staticOut.copy_(staticIn.mul(2).add(1));
      });
    });

    staticIn.fill_(5);
    graph.replay();
    torch.cuda.synchronize();
    assert.ok(Math.abs(staticOut.mean().item() - 11) < 1e-5);
    graph.reset();

    // Non-blocking D2H toTypedArrayAsync() from CUDA
    using gpuArr = torch.arange(0, 16, 1, { dtype: 'float32', device: 'cuda' });
    const hostTa = (await gpuArr.toTypedArrayAsync()) as Float32Array;
    assert.equal(hostTa.length, 16);
    assert.equal(hostTa[15], 15);

    // Pinned memory DataLoader (both sync and 2-worker async) + non-blocking H2D transfer
    using cpuX = torch.randn([16, 8]);
    const ds = new utils.TensorDataset(cpuX);
    const syncLoader = new utils.DataLoader(ds, { batchSize: 4, pinMemory: true });
    for (const [bx] of syncLoader) {
      assert.equal(bx!.isPinned, true);
      using gpuX = bx!.to({ device: 'cuda', nonBlocking: true });
      assert.equal(gpuX.isCuda, true);
      bx!.dispose();
    }

    const asyncLoader = new utils.DataLoader(ds, { batchSize: 4, pinMemory: true, numWorkers: 2 });
    let rows = 0;
    for await (const [bx] of asyncLoader) {
      assert.equal(bx!.isPinned, true);
      using gpuX = bx!.to({ device: 'cuda', nonBlocking: true });
      assert.equal(gpuX.isCuda, true);
      rows += gpuX.size(0);
      bx!.dispose();
    }
    assert.equal(rows, 16);
  });

  it('supports GPU DLPack zero-copy, GPU Kineto profiler, live ProcessGroupNCCL, and GPU Transformer LM convergence', async () => {
    // 1. GPU DLPack zero-copy roundtrip (kDLCUDA = 2)
    torch.tidy(() => {
      const src = torch.tensor([1, 2, 3, 4], { dtype: 'float32', device: 'cuda' });
      assert.deepEqual(utils.dlpackDevice(src), [2, 0]);
      const cap = utils.toDlpack(src);
      const imported = utils.fromDlpack(cap);
      assert.equal(imported.isCuda, true);
      assert.equal(imported.dataPtr, src.dataPtr);
      imported.mul_(3);
      torch.cuda.synchronize();
      assert.deepEqual(Array.from(src.toTypedArray() as Float32Array), [3, 6, 9, 12]);
    });

    // 2. GPU Kineto profiler + Chrome trace
    const cacheDir = path.join(os.homedir(), '.cache', 'norch');
    fs.mkdirSync(cacheDir, { recursive: true });
    const tracePath = path.join(cacheDir, `cuda-trace-${process.pid}.json`);
    try {
      const { prof } = profiler.profile(
        { activities: ['cpu', 'cuda'], recordShapes: true },
        () => {
          torch.tidy(() => {
            const a = torch.randn([128, 128], { device: 'cuda' });
            const b = torch.randn([128, 128], { device: 'cuda' });
            const c = a.matmul(b).relu().sum();
            torch.cuda.synchronize();
            return c.item();
          });
        }
      );
      assert.ok(prof.events().length > 0);
      prof.exportChromeTrace(tracePath);
      assert.ok(fs.existsSync(tracePath) && fs.statSync(tracePath).size > 10);
      prof.dispose();
    } finally {
      fs.rmSync(tracePath, { force: true });
    }

    // 3. Live ProcessGroupNCCL (world_size=1) initialization + collectives
    assert.equal(dist.isNcclAvailable(), true);
    const port = await getFreePort();
    dist.initProcessGroup({
      backend: 'nccl',
      initMethod: `tcp://127.0.0.1:${port}`,
      rank: 0,
      worldSize: 1,
      timeoutMs: 30_000,
    });
    try {
      assert.equal(dist.getBackend(), 'nccl');
      using t = torch.tensor([1.0, 2.0, 3.0], { dtype: 'float32', device: 'cuda' });
      const work = dist.allReduce(t, dist.ReduceOp.SUM, undefined, true);
      assert.ok(work !== undefined);
      work!.wait();
      dist.broadcast(t, 0);
      dist.barrier();
      torch.cuda.synchronize();
      assert.deepEqual(Array.from(t.toTypedArray() as Float32Array), [1, 2, 3]);
    } finally {
      dist.destroyProcessGroup();
    }

    // 4. Mini causal Transformer training on GPU with zero tensor leaks
    class GpuMiniLM extends nn.Module {
      readonly tokEmb = new nn.Embedding(32, 64);
      readonly block = new nn.TransformerBlock(64, 4, { dimFeedforward: 128 });
      readonly norm = new nn.LayerNorm(64);
      readonly head = new nn.Linear(64, 32);
      forward(ids: torch.Tensor): torch.Tensor {
        return this.head.forward(this.norm.forward(this.block.forward(this.tokEmb.forward(ids), { isCausal: true })));
      }
    }

    torch.manualSeed(7);
    torch.cuda.manualSeed(7);
    using lm = new GpuMiniLM();
    lm.to('cuda');
    using opt = new optim.AdamW(lm.parameters(), { lr: 5e-3, fused: true });
    using seq = torch.randint(0, 32, [8, 17], { device: 'cuda' });
    using inp = seq.narrow(1, 0, 16).clone();
    using tgt = seq.narrow(1, 1, 16).clone();

    let firstLoss = 0;
    let lastLoss = 0;
    let liveAfterWarmup = 0;
    for (let step = 0; step < 30; step++) {
      const lossVal = torch.tidy(() => {
        opt.zeroGrad();
        const logits = torch.autocast('cuda', torch.bfloat16, () => lm.forward(inp));
        const loss = torch.F.crossEntropy(logits.float().reshape(-1, 32), tgt.reshape(-1));
        loss.backward();
        opt.step();
        return loss.item();
      });
      if (step === 0) {
        firstLoss = lossVal;
        liveAfterWarmup = torch.memoryStats().liveTensors;
      }
      lastLoss = lossVal;
    }
    torch.cuda.synchronize();
    assert.ok(lastLoss < firstLoss * 0.5, `GPU Transformer LM did not converge: ${firstLoss} -> ${lastLoss}`);
    assert.ok(torch.memoryStats().liveTensors <= liveAfterWarmup, 'Tensor leak detected in GPU training loop');
  });

  it('supports pure-Node GPU torch.jit.trace (.save/.load), GPU torch.compile (training + CUDAGraph reduce-overhead), zero-sync GPU GradScaler, and GPU checkpoint resume', () => {
    const cacheDir = path.join(os.homedir(), '.cache', 'norch');
    fs.mkdirSync(cacheDir, { recursive: true });
    const ptPath = path.join(cacheDir, `norch_gpu_jit_${process.pid}.pt`);
    const ckptPath = path.join(cacheDir, `norch_gpu_ckpt_${process.pid}.safetensors`);

    try {
      class GpuBlockNet extends nn.Module {
        readonly fc1 = new nn.Linear(32, 64);
        readonly norm = new nn.LayerNorm(64);
        readonly fc2 = new nn.Linear(64, 16);
        forward(x: Tensor): Tensor {
          return this.fc2.forward(this.norm.forward(this.fc1.forward(x)).gelu());
        }
      }

      // 1. Pure-Node GPU torch.jit.trace -> .save(.pt) -> torch.jit.load(path, 'cuda')
      {
        torch.manualSeed(11);
        torch.cuda.manualSeed(11);
        using model = new GpuBlockNet();
        model.to('cuda').eval();
        using ex = torch.randn([4, 32], { device: 'cuda' });
        using xTest = torch.randn([8, 32], { device: 'cuda' });

        using traced = torch.jit.trace(model, ex);
        traced.save(ptPath);
        using loaded = torch.jit.load(ptPath, 'cuda');
        loaded.eval();

        torch.noGrad(() => {
          torch.tidy(() => {
            const eager = model.forward(xTest);
            const tOut = traced.forward(xTest) as Tensor;
            const lOut = loaded.forward(xTest) as Tensor;
            assert.equal(tOut.isCuda, true);
            assert.equal(lOut.isCuda, true);
            assert.ok(tOut.allclose(eager, 1e-4, 1e-4), 'GPU TracedModule mismatch vs eager');
            assert.ok(lOut.allclose(eager, 1e-4, 1e-4), 'GPU saved+loaded .pt mismatch vs eager');
          });
        });
      }

      // 2. Pure-Node GPU torch.compile() BF16 autocast training parity + reduce-overhead CUDAGraph inference
      {
        torch.manualSeed(22);
        torch.cuda.manualSeed(22);
        using eagerNet = new GpuBlockNet();
        using compBase = new GpuBlockNet();
        const sd = eagerNet.stateDict();
        compBase.loadStateDict(sd);
        for (const t of Object.values(sd)) t.dispose();

        eagerNet.to('cuda');
        compBase.to('cuda');
        using compiled = torch.compile(compBase);
        using optE = new optim.AdamW(eagerNet.parameters(), { lr: 1e-2, fused: true });
        using optC = new optim.AdamW(compiled.parameters(), { lr: 1e-2, fused: true });

        using x = torch.randn([8, 32], { device: 'cuda' });
        using y = torch.randn([8, 16], { device: 'cuda' });

        for (let step = 0; step < 5; step++) {
          torch.tidy(() => {
            optE.zeroGrad();
            const oE = torch.autocast('cuda', torch.bfloat16, () => eagerNet.forward(x));
            F.mseLoss(oE.float(), y).backward();
            optE.step();

            optC.zeroGrad();
            const oC = torch.autocast('cuda', torch.bfloat16, () => compiled.forward(x) as Tensor);
            F.mseLoss(oC.float(), y).backward();
            optC.step();
          });
        }
        torch.cuda.synchronize();
        const pE = eagerNet.parameters();
        const pC = compBase.parameters();
        for (let i = 0; i < pE.length; i++) {
          assert.ok(pE[i]!.allclose(pC[i]!, 1e-3, 1e-3), `GPU compiled BF16 param ${i} diverged`);
        }

        // Also verify compiled TransformerBlock (Flash/cuDNN SDPA + LayerNorm + GELU) under BF16 autocast
        using tbEager = new nn.TransformerBlock(64, 4, { dimFeedforward: 128 });
        using tbCompBase = new nn.TransformerBlock(64, 4, { dimFeedforward: 128 });
        const tbSd = tbEager.stateDict();
        tbCompBase.loadStateDict(tbSd);
        for (const t of Object.values(tbSd)) t.dispose();
        tbEager.to('cuda');
        tbCompBase.to('cuda');
        using tbComp = torch.compile(tbCompBase);
        using optTbE = new optim.AdamW(tbEager.parameters(), { lr: 1e-2, fused: true });
        using optTbC = new optim.AdamW(tbComp.parameters(), { lr: 1e-2, fused: true });
        using tx = torch.randn([4, 16, 64], { device: 'cuda' });
        using ty = torch.randn([4, 16, 64], { device: 'cuda' });
        for (let step = 0; step < 5; step++) {
          torch.tidy(() => {
            optTbE.zeroGrad();
            const oE = torch.autocast('cuda', torch.bfloat16, () => tbEager.forward(tx, { isCausal: true }));
            F.mseLoss(oE.float(), ty).backward();
            optTbE.step();

            optTbC.zeroGrad();
            const oC = torch.autocast('cuda', torch.bfloat16, () => tbComp.forward(tx, { isCausal: true }) as Tensor);
            F.mseLoss(oC.float(), ty).backward();
            optTbC.step();
          });
        }
        torch.cuda.synchronize();
        const tpE = tbEager.parameters();
        const tpC = tbCompBase.parameters();
        for (let i = 0; i < tpE.length; i++) {
          assert.ok(tpE[i]!.allclose(tpC[i]!, 2e-3, 2e-3), `GPU compiled TransformerBlock param ${i} diverged`);
        }

        // mode: 'reduce-overhead' (CUDAGraph capture & replay after warmup)
        eagerNet.eval();
        using fastInf = torch.compile(eagerNet, { mode: 'reduce-overhead' });
        torch.noGrad(() => {
          for (let i = 0; i < 6; i++) {
            torch.tidy(() => {
              const xi = torch.randn([8, 32], { device: 'cuda' });
              const ref = eagerNet.forward(xi);
              const got = fastInf.forward(xi) as Tensor;
              assert.ok(got.allclose(ref, 1e-4, 1e-4), `GPU reduce-overhead iter ${i} mismatch`);
            });
          }
        });
      }

      // 3. Zero-sync GPU GradScaler({ device: 'cuda' }) -> fused AdamW handoff
      {
        using lin = new nn.Linear(32, 16);
        lin.to('cuda');
        using opt = new optim.AdamW(lin.parameters(), { lr: 1e-2, fused: true });
        using scaler = new torch.amp.GradScaler({
          device: 'cuda',
          initScale: 256,
          growthFactor: 2,
          backoffFactor: 0.5,
          growthInterval: 2,
        });
        using x = torch.randn([8, 32], { device: 'cuda' });
        using y = torch.randn([8, 16], { device: 'cuda' });

        for (let s = 0; s < 2; s++) {
          torch.tidy(() => {
            opt.zeroGrad();
            const out = torch.autocast('cuda', torch.float16, () => lin.forward(x));
            const loss = F.mseLoss(out.float(), y);
            scaler.scale(loss).backward();
            scaler.stepNoSync(opt);
            scaler.update();
          });
        }
        assert.equal(scaler.getScale(), 512);

        using snap = torch.noGrad(() => lin.weight.clone());
        torch.tidy(() => {
          opt.zeroGrad();
          const out = torch.autocast('cuda', torch.float16, () => lin.forward(x));
          const loss = F.mseLoss(out.float(), y);
          scaler.scale(loss).backward();
          lin.weight.grad!.fill_(Number.POSITIVE_INFINITY);
          scaler.stepNoSync(opt);
          scaler.update();
        });
        torch.cuda.synchronize();
        assert.ok(lin.weight.equal(snap), 'GPU stepNoSync mutated weights on Inf gradient');
        assert.equal(scaler.getScale(), 256);
      }

      // 4. GPU saveCheckpoint / loadCheckpoint(..., { device: 'cuda' }) roundtrip
      {
        torch.manualSeed(33);
        using mA = new GpuBlockNet();
        mA.to('cuda');
        using optA = new optim.AdamW(mA.parameters(), { lr: 1e-2, fused: true });
        using x = torch.randn([8, 32], { device: 'cuda' });
        using y = torch.randn([8, 16], { device: 'cuda' });

        for (let s = 0; s < 3; s++) {
          torch.tidy(() => {
            optA.zeroGrad();
            F.mseLoss(mA.forward(x), y).backward();
            optA.step();
          });
        }
        torch.saveCheckpoint(ckptPath, { model: mA, optimizer: optA, step: 3 });

        using mB = new GpuBlockNet();
        mB.to('cuda');
        using optB = new optim.AdamW(mB.parameters(), { lr: 1e-2, fused: true });
        const meta = torch.loadCheckpoint(ckptPath, { model: mB, optimizer: optB, device: 'cuda' });
        assert.equal(meta.step, 3);
        assert.equal(mB.fc1.weight.isCuda, true);

        for (let s = 0; s < 3; s++) {
          torch.tidy(() => {
            optA.zeroGrad();
            F.mseLoss(mA.forward(x), y).backward();
            optA.step();

            optB.zeroGrad();
            F.mseLoss(mB.forward(x), y).backward();
            optB.step();
          });
        }
        torch.cuda.synchronize();
        const pA = mA.parameters();
        const pB = mB.parameters();
        for (let i = 0; i < pA.length; i++) {
          assert.ok(pA[i]!.allclose(pB[i]!, 1e-5, 1e-5), `GPU checkpoint resumed param ${i} diverged`);
        }
      }
    } finally {
      fs.rmSync(ptPath, { force: true });
      fs.rmSync(ckptPath, { force: true });
    }
  });

  it('matches Python PyTorch numerically on CUDA (Flash/cuDNN SDPA, Conv2d, LayerNorm, RMSNorm, fused AdamW)', (t) => {
    const py = findPythonTorch();
    if (!py) {
      t.skip('No local Python PyTorch with CUDA found');
      return;
    }

    const pyOracle = `
import json, torch, torch.nn as nn, torch.nn.functional as F

def make_seq(shape, scale=0.1, offset=0.0):
    n = 1
    for s in shape: n *= s
    return (torch.arange(n, dtype=torch.float32, device="cuda").reshape(shape) * scale + offset)

# 1. Causal SDPA (float32 & bfloat16)
q = torch.sin(make_seq((2, 4, 16, 32), 0.13))
k = torch.cos(make_seq((2, 4, 16, 32), 0.17))
v = torch.sin(make_seq((2, 4, 16, 32), 0.19, 0.5))
sdpa_fp32 = F.scaled_dot_product_attention(q, k, v, is_causal=True)
sdpa_bf16 = F.scaled_dot_product_attention(q.bfloat16(), k.bfloat16(), v.bfloat16(), is_causal=True).float()

# 2. Conv2d forward + backward
cx = torch.sin(make_seq((2, 3, 8, 8), 0.11)).detach().requires_grad_(True)
cw = torch.cos(make_seq((4, 3, 3, 3), 0.07)).detach().requires_grad_(True)
cb = torch.sin(make_seq((4,), 0.3)).detach().requires_grad_(True)
cy = F.conv2d(cx, cw, cb, stride=1, padding=1)
(cy * cy).mean().backward()

# 3. LayerNorm & RMSNorm
nx = torch.sin(make_seq((3, 16), 0.23)).detach().requires_grad_(True)
nw = torch.ones(16, device="cuda").mul(1.1).detach().requires_grad_(True)
nb = torch.full((16,), -0.2, device="cuda").detach().requires_grad_(True)
ln_out = F.layer_norm(nx, (16,), nw, nb, 1e-5)
rms_out = F.rms_norm(nx, (16,), nw, 1e-5)
(ln_out.sum() + rms_out.sum()).backward()

# 4. Fused AdamW (3 steps)
p = torch.sin(make_seq((8, 8), 0.15)).detach().requires_grad_(True)
opt = torch.optim.AdamW([p], lr=0.05, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, fused=True)
target = torch.cos(make_seq((8, 8), 0.12))
for _ in range(3):
    opt.zero_grad()
    F.mse_loss(p * p, target).backward()
    opt.step()

# 5. Muon on CUDA (3 steps on 16x8 matrix)
mp = torch.sin(make_seq((16, 8), 0.13)).detach().requires_grad_(True)
opt_m = torch.optim.Muon([mp], lr=0.02, weight_decay=0.05, momentum=0.95, adjust_lr_fn="original")
mtgt = torch.cos(make_seq((16, 8), 0.11))
for _ in range(3):
    opt_m.zero_grad()
    F.mse_loss(mp, mtgt).backward()
    opt_m.step()

torch.cuda.synchronize()
print(json.dumps({
    "sdpa_fp32": sdpa_fp32.cpu().flatten().tolist()[:64],
    "sdpa_bf16": sdpa_bf16.cpu().flatten().tolist()[:64],
    "conv_out": cy.detach().cpu().flatten().tolist()[:64],
    "conv_gx": cx.grad.cpu().flatten().tolist()[:64],
    "conv_gw": cw.grad.cpu().flatten().tolist()[:64],
    "ln_out": ln_out.detach().cpu().flatten().tolist(),
    "rms_out": rms_out.detach().cpu().flatten().tolist(),
    "norm_gx": nx.grad.cpu().flatten().tolist(),
    "adamw_p": p.detach().cpu().flatten().tolist(),
    "muon_p": mp.detach().cpu().flatten().tolist(),
}))
`;

    const env = { ...process.env };
    delete env.LD_PRELOAD;
    delete env.ASAN_OPTIONS;
    delete env.UBSAN_OPTIONS;
    const ref = JSON.parse(
      execFileSync(py, ['-c', pyOracle], { encoding: 'utf8', timeout: 60000, env })
    ) as Record<string, number[]>;

    const makeSeq = (shape: number[], scale = 0.1, offset = 0.0): Tensor => {
      const n = shape.reduce((a, b) => a * b, 1);
      return torch.arange(0, n, 1, { dtype: 'float32', device: 'cuda' }).reshape(shape).mul(scale).add(offset);
    };

    torch.tidy(() => {
      // 1. Causal SDPA (float32 & bfloat16)
      const q = makeSeq([2, 4, 16, 32], 0.13).sin();
      const k = makeSeq([2, 4, 16, 32], 0.17).cos();
      const v = makeSeq([2, 4, 16, 32], 0.19, 0.5).sin();
      const sdpaFp32 = F.scaledDotProductAttention(q, k, v, { isCausal: true });
      const sdpaBf16 = F.scaledDotProductAttention(q.bfloat16(), k.bfloat16(), v.bfloat16(), {
        isCausal: true,
      }).float();
      torch.cuda.synchronize();
      assertCloseArrays(
        Array.from(sdpaFp32.toTypedArray() as Float32Array).slice(0, 64),
        ref['sdpa_fp32']!,
        1e-4,
        1e-4,
        'CUDA SDPA fp32'
      );
      assertCloseArrays(
        Array.from(sdpaBf16.toTypedArray() as Float32Array).slice(0, 64),
        ref['sdpa_bf16']!,
        2e-2,
        2e-2,
        'CUDA SDPA bf16'
      );

      // 2. Conv2d forward + backward
      const cx = makeSeq([2, 3, 8, 8], 0.11).sin().detach().requiresGrad_(true);
      const cw = makeSeq([4, 3, 3, 3], 0.07).cos().detach().requiresGrad_(true);
      const cb = makeSeq([4], 0.3).sin().detach().requiresGrad_(true);
      const cy = F.conv2d(cx, cw, { bias: cb, stride: 1, padding: 1 });
      cy.mul(cy).mean().backward();
      torch.cuda.synchronize();
      assertCloseArrays(
        Array.from(cy.detach().toTypedArray() as Float32Array).slice(0, 64),
        ref['conv_out']!,
        1e-4,
        1e-4,
        'CUDA Conv2d out'
      );
      assertCloseArrays(
        Array.from(cx.grad!.toTypedArray() as Float32Array).slice(0, 64),
        ref['conv_gx']!,
        1e-4,
        1e-4,
        'CUDA Conv2d gx'
      );
      assertCloseArrays(
        Array.from(cw.grad!.toTypedArray() as Float32Array).slice(0, 64),
        ref['conv_gw']!,
        1e-4,
        1e-4,
        'CUDA Conv2d gw'
      );

      // 3. LayerNorm & RMSNorm
      const nx = makeSeq([3, 16], 0.23).sin().detach().requiresGrad_(true);
      const nw = torch.ones([16], { device: 'cuda' }).mul(1.1).detach().requiresGrad_(true);
      const nb = torch.full([16], -0.2, { device: 'cuda' }).detach().requiresGrad_(true);
      const lnOut = F.layerNorm(nx, [16], nw, nb, 1e-5);
      const rmsOut = F.rmsNorm(nx, [16], nw, 1e-5);
      lnOut.sum().add(rmsOut.sum()).backward();
      torch.cuda.synchronize();
      assertCloseArrays(Array.from(lnOut.detach().toTypedArray() as Float32Array), ref['ln_out']!, 1e-4, 1e-4, 'CUDA LayerNorm');
      assertCloseArrays(Array.from(rmsOut.detach().toTypedArray() as Float32Array), ref['rms_out']!, 1e-4, 1e-4, 'CUDA RMSNorm');
      assertCloseArrays(Array.from(nx.grad!.toTypedArray() as Float32Array), ref['norm_gx']!, 1e-4, 1e-4, 'CUDA Norm gx');

      // 4. Fused AdamW (3 steps)
      using p = new nn.Parameter(makeSeq([8, 8], 0.15).sin().detach());
      using opt = new optim.AdamW([p], { lr: 0.05, betas: [0.9, 0.999], eps: 1e-8, weightDecay: 0.01, fused: true });
      const target = makeSeq([8, 8], 0.12).cos();
      for (let s = 0; s < 3; s++) {
        opt.zeroGrad();
        F.mseLoss(p.mul(p), target).backward();
        opt.step();
      }
      torch.cuda.synchronize();
      assertCloseArrays(Array.from(p.detach().toTypedArray() as Float32Array), ref['adamw_p']!, 1e-5, 1e-5, 'CUDA fused AdamW');

      // 5. Muon on CUDA (3 steps on 16x8 matrix)
      using mp = new nn.Parameter(makeSeq([16, 8], 0.13).sin().detach());
      using optM = new optim.Muon([mp], { lr: 0.02, weightDecay: 0.05, momentum: 0.95, adjustLrFn: 'original' });
      const mtgt = makeSeq([16, 8], 0.11).cos();
      for (let s = 0; s < 3; s++) {
        optM.zeroGrad();
        F.mseLoss(mp, mtgt).backward();
        optM.step();
      }
      torch.cuda.synchronize();
      assertCloseArrays(Array.from(mp.detach().toTypedArray() as Float32Array), ref['muon_p']!, 2e-3, 2e-3, 'CUDA Muon');
    });
  });
});
