import assert from 'node:assert/strict';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { describe, it } from 'node:test';
import torch, { distributed as dist, Generator, nn, optim, profiler, utils } from '../src/index.js';

const runCuda = process.env.NORCH_TEST_CUDA === '1';

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
});
