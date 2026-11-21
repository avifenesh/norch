import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import torch, { Generator, nn, optim, utils } from '../src/index.js';

const runCuda = process.env.NORCH_TEST_CUDA === '1';

describe('CUDA backend (guarded by NORCH_TEST_CUDA=1)', { skip: !runCuda }, () => {
  it('runs GPU training with fused AdamW, bfloat16 autocast, GradScaler, and weight tying', () => {
    assert.equal(torch.cuda.isAvailable(), true);
    assert.ok(torch.cuda.deviceCount() >= 1);
    const props = torch.cuda.deviceProperties(0);
    assert.ok(props.name.length > 0 && props.totalMemory > 0);

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
    const scaler = new torch.amp.GradScaler({ initScale: 256 });

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
    torch.cuda.emptyCache();
    assert.ok(torch.cuda.memoryStats().maxAllocated > 0);
  });

  it('supports CUDA Streams, Events timing, CUDAGraph capture/replay, Generator(cuda), and pinned DataLoader', () => {
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
        const m = torch.randn([128, 128], { device: 'cuda' });
        startEv.record(s);
        m.matmul(m);
        endEv.record(s);
      });
    });
    endEv.synchronize();
    assert.ok(startEv.elapsedTime(endEv) >= 0);

    // CUDA Graph capture & replay
    using staticIn = torch.ones([8, 8], { device: 'cuda' });
    using staticOut = torch.zeros([8, 8], { device: 'cuda' });
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

    // Pinned memory DataLoader + non-blocking H2D transfer
    using cpuX = torch.randn([8, 4]);
    const ds = new utils.TensorDataset(cpuX);
    const loader = new utils.DataLoader(ds, { batchSize: 4, pinMemory: true });
    for (const [bx] of loader) {
      assert.equal(bx!.isPinned, true);
      using gpuX = bx!.to({ device: 'cuda', nonBlocking: true });
      assert.equal(gpuX.isCuda, true);
      bx!.dispose();
    }
  });
});
