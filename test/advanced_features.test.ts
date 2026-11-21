import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, it } from 'node:test';
import torch, {
  AutogradFunction,
  F,
  Generator,
  nn,
  optim,
  type AutogradContext,
  type Tensor,
} from '../src/index.js';

torch.setNumThreads(4);

function findPythonTorch(): string | null {
  const candidates = [
    process.env.NORCH_PYTHON,
    path.join(os.homedir(), 'projects', 'torch', '.venv', 'bin', 'python'),
    'python3',
  ].filter((p): p is string => Boolean(p));
  for (const py of candidates) {
    if (py.includes(path.sep) && !fs.existsSync(py)) continue;
    try {
      execFileSync(py, ['-c', 'import torch'], { stdio: 'ignore', timeout: 15000 });
      return py;
    } catch {
      // ignore
    }
  }
  return null;
}

describe('Advanced training, GPU-prep primitives, hooks, forward AD, RNNs & export runners', () => {
  it('preserves weight tying across Module.to() with zero tensor leaks', () => {
    const baseline = torch.memoryStats();

    class TiedLM extends nn.Module {
      readonly tokEmb = new nn.Embedding(16, 8);
      readonly lmHead = new nn.Linear(8, 16, { bias: false });
      constructor() {
        super();
        // Tie weights cleanly (dispose unused original lmHead.weight)
        const old = this.lmHead.weight;
        (this.lmHead as unknown as { weight: nn.Parameter }).weight = this.tokEmb.weight;
        old.dispose();
      }
      forward(x: Tensor): Tensor {
        return this.lmHead.forward(this.tokEmb.forward(x));
      }
    }

    {
      using model = new TiedLM();
      assert.equal(model.tokEmb.weight, model.lmHead.weight);
      assert.equal(model.parameters().length, 1);

      model.to(torch.float64);
      assert.equal(model.tokEmb.weight, model.lmHead.weight);
      assert.equal(model.tokEmb.weight.dtype, 'float64');
      assert.equal(model.parameters().length, 1);

      model.to(torch.float32);
      assert.equal(model.tokEmb.weight, model.lmHead.weight);
      assert.equal(model.tokEmb.weight.dtype, 'float32');
    }

    assert.equal(torch.memoryStats().liveTensors, baseline.liveTensors);
  });

  it('matches fused AdamW against unfused AdamW and supports full checkpoint resume (model + optim + scheduler)', () => {
    const cacheDir = path.join(os.homedir(), '.cache', 'norch');
    fs.mkdirSync(cacheDir, { recursive: true });
    const ckptPath = path.join(cacheDir, 'resume_test.safetensors');

    try {
      torch.manualSeed(101);
      using x = torch.randn([16, 4]);
      using y = torch.randn([16, 2]);

      // 1. Compare fused vs unfused AdamW on identical starting weights
      using mFused = new nn.Sequential(new nn.Linear(4, 8), new nn.GELU(), new nn.Linear(8, 2));
      using mUnfused = new nn.Sequential(new nn.Linear(4, 8), new nn.GELU(), new nn.Linear(8, 2));
      const sd0 = mFused.stateDict();
      mUnfused.loadStateDict(sd0);
      for (const t of Object.values(sd0)) t.dispose();

      using optFused = new optim.AdamW(mFused.parameters(), { lr: 1e-2, weightDecay: 0.05, fused: true });
      using optUnfused = new optim.AdamW(mUnfused.parameters(), { lr: 1e-2, weightDecay: 0.05, fused: false });

      for (let i = 0; i < 4; i++) {
        torch.tidy(() => {
          optFused.zeroGrad();
          F.mseLoss(mFused.forward(x), y).backward();
          optFused.step();

          optUnfused.zeroGrad();
          F.mseLoss(mUnfused.forward(x), y).backward();
          optUnfused.step();
        });
      }
      const pFused = mFused.parameters();
      const pUnfused = mUnfused.parameters();
      for (let i = 0; i < pFused.length; i++) {
        assert.ok(pFused[i]!.allclose(pUnfused[i]!, 1e-5, 1e-5), `fused vs unfused param ${i} mismatch`);
      }

      // 2. Full checkpoint save at step 4 -> resume in a brand-new model/optimizer/scheduler for 4 more steps
      const schedA = new optim.CosineAnnealingLR(optFused, 10, 1e-4);
      for (let s = 0; s < 4; s++) schedA.step();
      torch.saveCheckpoint(ckptPath, {
        model: mFused,
        optimizer: optFused,
        scheduler: schedA,
        step: 4,
        epoch: 1,
        extra: { note: 'resume-check' },
      });

      using mResumed = new nn.Sequential(new nn.Linear(4, 8), new nn.GELU(), new nn.Linear(8, 2));
      using optResumed = new optim.AdamW(mResumed.parameters(), { lr: 1e-2, weightDecay: 0.05, fused: true });
      const schedB = new optim.CosineAnnealingLR(optResumed, 10, 1e-4);

      const meta = torch.loadCheckpoint(ckptPath, {
        model: mResumed,
        optimizer: optResumed,
        scheduler: schedB,
      });
      assert.equal(meta.step, 4);
      assert.equal(meta.epoch, 1);
      assert.equal(meta.extra?.['note'], 'resume-check');

      for (let step = 4; step < 8; step++) {
        torch.tidy(() => {
          optFused.zeroGrad();
          F.mseLoss(mFused.forward(x), y).backward();
          optFused.step();

          optResumed.zeroGrad();
          F.mseLoss(mResumed.forward(x), y).backward();
          optResumed.step();
        });
        schedA.step();
        schedB.step();
      }

      const pA = mFused.parameters();
      const pB = mResumed.parameters();
      for (let i = 0; i < pA.length; i++) {
        assert.ok(pA[i]!.allclose(pB[i]!, 1e-6, 1e-6), `resumed param ${i} diverged`);
      }
      assert.ok(Math.abs(schedA.getLastLr()[0]! - schedB.getLastLr()[0]!) < 1e-12);
    } finally {
      if (fs.existsSync(ckptPath)) fs.unlinkSync(ckptPath);
    }
  });

  it('supports RMSprop, OneCycleLR, SequentialLR, and ReduceLROnPlateau', () => {
    using p = new nn.Parameter(torch.tensor([3.0, -4.0]));
    using rms = new optim.RMSprop([p], { lr: 0.1, momentum: 0.5, centered: true });
    const startLoss = torch.tidy(() => p.pow(2).sum().item());
    for (let i = 0; i < 15; i++) {
      torch.tidy(() => {
        rms.zeroGrad();
        p.pow(2).sum().backward();
        rms.step();
      });
    }
    const endLoss = torch.tidy(() => p.pow(2).sum().item());
    assert.ok(endLoss < startLoss * 0.1, `RMSprop did not converge: ${startLoss} -> ${endLoss}`);

    // OneCycleLR warms up then anneals down
    using opt1 = new optim.SGD([p], { lr: 0.01 });
    const oc = new optim.OneCycleLR(opt1, { maxLr: 0.1, totalSteps: 10, pctStart: 0.3 });
    const lrs: number[] = [oc.getLastLr()[0]!];
    for (let i = 0; i < 9; i++) {
      oc.step();
      lrs.push(oc.getLastLr()[0]!);
    }
    assert.ok(lrs[3]! > lrs[0]! && lrs[9]! < lrs[3]!, `Unexpected OneCycleLR curve: ${lrs}`);

    // SequentialLR: Linear warmup -> Cosine decay
    using opt2 = new optim.SGD([p], { lr: 0.1 });
    const warmup = new optim.LinearLR(opt2, { startFactor: 0.2, endFactor: 1.0, totalIters: 4 });
    const cosine = new optim.CosineAnnealingLR(opt2, 6, 0.01);
    const seq = new optim.SequentialLR(opt2, [warmup, cosine], [4]);
    assert.ok(Math.abs(seq.getLastLr()[0]! - 0.02) < 1e-6);
    for (let i = 0; i < 4; i++) seq.step();
    assert.ok(Math.abs(seq.getLastLr()[0]! - 0.1) < 1e-6);
    for (let i = 0; i < 6; i++) seq.step();
    assert.ok(Math.abs(seq.getLastLr()[0]! - 0.01) < 1e-6);

    // ReduceLROnPlateau
    using opt3 = new optim.SGD([p], { lr: 0.1 });
    const plateau = new optim.ReduceLROnPlateau(opt3, { mode: 'min', factor: 0.5, patience: 1 });
    plateau.step(1.0);
    plateau.step(1.0);
    plateau.step(1.0);
    assert.ok(Math.abs(plateau.getLastLr()[0]! - 0.05) < 1e-9);
  });

  it('supports deterministic Generator with getState/setState replay', () => {
    torch.tidy(() => {
      const g1 = new Generator('cpu').manualSeed(777);
      const g2 = new Generator('cpu').manualSeed(777);
      const a = torch.randn([4, 4], { generator: g1 });
      const b = torch.randn([4, 4], { generator: g2 });
      assert.ok(a.equal(b));

      const savedState = g1.getState();
      const next1 = torch.rand([8], { generator: g1 });
      g1.setState(savedState);
      const next2 = torch.rand([8], { generator: g1 });
      assert.ok(next1.equal(next2));
    });
  });

  it('supports tensor backward hooks, Module forward hooks, custom markDirty/needsInputGrad, forwardAD JVP & toTypedArrayAsync', async () => {
    // Tensor hook scaling grad by 3x + removable handle
    torch.tidy(() => {
      const x = torch.tensor([2.0, 3.0], { requiresGrad: true });
      const handle = x.registerHook((g) => g.mul(3));
      x.pow(2).sum().backward();
      assert.deepEqual(x.grad!.toArray(), [12, 18]); // 2*x * 3
      handle.remove();

      x.grad = null;
      x.pow(2).sum().backward();
      assert.deepEqual(x.grad!.toArray(), [4, 6]);
    });

    // Module forward pre-hook & post-hook
    using lin = new nn.Linear(2, 2);
    torch.noGrad(() => {
      lin.weight.copy_(torch.eye(2));
      lin.bias!.zero_();
    });
    const seen: string[] = [];
    {
      using _pre = lin.registerForwardPreHook((_m, inputs) => {
        seen.push('pre');
        return [(inputs[0] as Tensor).mul(2)];
      });
      using _post = lin.registerForwardHook((_m, _in, out) => {
        seen.push('post');
        return (out as Tensor).add(10);
      });
      torch.tidy(() => {
        const y = lin.call(torch.tensor([[1.0, 2.0]])) as Tensor;
        assert.deepEqual(y.toArray(), [[12, 14]]);
      });
    }
    assert.deepEqual(seen, ['pre', 'post']);

    // Custom AutogradFunction: needsInputGrad + markDirty (in-place scaling)
    class InplaceScale extends AutogradFunction {
      static forward(ctx: AutogradContext, x: Tensor, y: Tensor, factor: number): Tensor {
        ctx['factor'] = factor;
        x.mul_(factor);
        ctx.markDirty(x);
        return x;
      }
      static backward(ctx: AutogradContext, gradOut: Tensor): (Tensor | null)[] {
        assert.deepEqual(ctx.needsInputGrad, [true, false]);
        return [gradOut.mul(ctx['factor'] as number), null, null];
      }
    }
    torch.tidy(() => {
      const base = torch.tensor([2.0, 4.0], { requiresGrad: true });
      const x = base.clone(); // non-leaf so in-place is legal
      const unused = torch.tensor([1.0], { requiresGrad: false });
      const out = InplaceScale.apply(x, unused, 5) as Tensor;
      out.sum().backward();
      assert.deepEqual(base.grad!.toArray(), [5, 5]);
    });

    // Forward-mode AD JVP: f(x) = x^3, f'(x) @ v = 3 * x^2 * v
    torch.tidy(() => {
      const x = torch.tensor([2.0, 3.0], { dtype: torch.float64 });
      const v = torch.tensor([1.0, 0.5], { dtype: torch.float64 });
      const { output, jvp } = torch.forwardAD.jvp((t) => t.pow(3), x, v);
      assert.deepEqual(output.toArray(), [8, 27]);
      assert.ok(jvp !== null);
      assert.deepEqual(jvp!.toArray(), [12, 13.5]);
    });

    // Async non-blocking toTypedArrayAsync()
    using big = torch.arange(16, { dtype: torch.float32 });
    const ta = (await big.toTypedArrayAsync()) as Float32Array;
    assert.equal(ta.length, 16);
    assert.equal(ta[15], 15);
  });

  it('runs LSTM, GRU, RNN, Conv3d, ConvTranspose2d, Upsample, HuberLoss, CosineSimilarity, linalg & fft with autograd', () => {
    torch.tidy(() => {
      // Bidirectional 2-layer LSTM
      using lstm = new nn.LSTM(6, 8, { numLayers: 2, batchFirst: true, bidirectional: true });
      const seq = torch.randn([3, 5, 6], { requiresGrad: true });
      const { output: lo, hn, cn } = lstm.forward(seq);
      assert.deepEqual(lo.shape, [3, 5, 16]);
      assert.deepEqual(hn.shape, [4, 3, 8]);
      assert.deepEqual(cn.shape, [4, 3, 8]);
      lo.sum().backward();
      assert.ok(seq.grad !== null);

      // GRU & RNN
      using gru = new nn.GRU(6, 4, { batchFirst: true });
      const { output: go } = gru.forward(seq.detach());
      assert.deepEqual(go.shape, [3, 5, 4]);

      using rnn = new nn.RNN(6, 4, { batchFirst: true, nonlinearity: 'relu' });
      const { output: ro } = rnn.forward(seq.detach());
      assert.deepEqual(ro.shape, [3, 5, 4]);

      // Conv3d, ConvTranspose2d, Upsample, pad, oneHot, HuberLoss, CosineSimilarity
      using c3 = new nn.Conv3d(2, 4, 3, { padding: 1 });
      assert.deepEqual(c3.forward(torch.randn([1, 2, 4, 4, 4])).shape, [1, 4, 4, 4, 4]);

      using ct2 = new nn.ConvTranspose2d(4, 2, 4, { stride: 2, padding: 1 });
      assert.deepEqual(ct2.forward(torch.randn([1, 4, 4, 4])).shape, [1, 2, 8, 8]);

      using up = new nn.Upsample({ scaleFactor: 2, mode: 'bilinear', alignCorners: false });
      assert.deepEqual(up.forward(torch.randn([1, 3, 4, 4])).shape, [1, 3, 8, 8]);

      assert.deepEqual(F.pad(torch.ones([2, 2]), [1, 1, 2, 2]).shape, [6, 4]);
      assert.deepEqual(F.oneHot(torch.tensor([0, 2, 1], { dtype: torch.int64 }), 3).toArray(), [
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
      ]);

      const huber = new nn.HuberLoss({ delta: 0.5 });
      assert.ok(huber.forward(torch.tensor([0.0, 2.0]), torch.tensor([0.0, 0.0])).item() > 0);
      const cos = new nn.CosineSimilarity(0);
      assert.ok(Math.abs(cos.forward(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])).item() - 1.0) < 1e-5);

      // init.truncNormal_ & init.orthogonal_
      const tn = torch.empty([256]);
      nn.init.truncNormal_(tn, 0, 1, -1.5, 1.5);
      assert.ok(tn.max().item() <= 1.5001 && tn.min().item() >= -1.5001);

      const q = torch.empty([8, 8], { dtype: torch.float64 });
      nn.init.orthogonal_(q);
      assert.ok(q.matmul(q.T).allclose(torch.eye(8, 8, { dtype: torch.float64 }), 1e-5, 1e-5));

      // linalg
      const spd = torch.tensor(
        [
          [4, 1],
          [1, 3],
        ],
        { dtype: torch.float64 }
      );
      const L = torch.linalg.cholesky(spd);
      assert.ok(L.matmul(L.T).allclose(spd, 1e-6, 1e-6));
      const { U, S, Vh } = torch.linalg.svd(spd);
      assert.deepEqual(U.shape, [2, 2]);
      assert.deepEqual(S.shape, [2]);
      assert.deepEqual(Vh.shape, [2, 2]);
      const bVec = torch.tensor([1.0, 2.0], { dtype: torch.float64 });
      const sol = torch.linalg.solve(spd, bVec);
      assert.ok(spd.matmul(sol).allclose(bVec, 1e-6, 1e-6));

      // fft roundtrip (rfft -> irfft)
      const sig = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], { dtype: torch.float64 });
      const freq = torch.fft.rfft(sig);
      const recon = torch.fft.irfft(freq, 8);
      assert.ok(recon.allclose(sig, 1e-6, 1e-6));
    });
  });

  it('loads and executes real Python-exported TorchScript (.pt) and AOTInductor (.pt2) models in Node.js', (t) => {
    const py = findPythonTorch();
    if (!py) {
      t.skip('No local Python PyTorch installation found');
      return;
    }

    const cacheDir = path.join(os.homedir(), '.cache', 'norch');
    fs.mkdirSync(cacheDir, { recursive: true });
    const tsPath = path.join(cacheDir, 'export_test_jit.pt');
    const pt2Path = path.join(cacheDir, 'export_test_aoti.pt2');

    const pyExport = `
import json, torch, torch.nn as nn

torch.manual_seed(0)
class ExportNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

m = ExportNet().eval()
x = torch.tensor([[0.5, -1.0, 2.0, 0.25], [-0.3, 0.7, -0.5, 1.5]], dtype=torch.float32)
with torch.no_grad():
    ref_out = m(x).flatten().tolist()

# 1. TorchScript export
traced = torch.jit.trace(m, x)
traced.save(${JSON.stringify(tsPath)})

# 2. AOTInductor (.pt2) export
aoti_ok = False
try:
    ep = torch.export.export(m, (x,))
    torch._inductor.aoti_compile_and_package(ep, package_path=${JSON.stringify(pt2Path)})
    aoti_ok = True
except Exception as e:
    aoti_ok = False

print(json.dumps({"ref_out": ref_out, "aoti_ok": aoti_ok}))
`;

    try {
      const info = JSON.parse(execFileSync(py, ['-c', pyExport], { encoding: 'utf8', timeout: 60000 })) as {
        ref_out: number[];
        aoti_ok: boolean;
      };

      torch.tidy(() => {
        const x = torch.tensor(
          [
            [0.5, -1.0, 2.0, 0.25],
            [-0.3, 0.7, -0.5, 1.5],
          ],
          { dtype: torch.float32 }
        );

        // Verify TorchScript execution
        using jit = torch.loadJitModule(tsPath, 'cpu');
        jit.eval();
        const jitOut = jit.forward(x) as Tensor;
        const jitVals = Array.from(jitOut.toTypedArray() as Float32Array);
        for (let i = 0; i < info.ref_out.length; i++) {
          assert.ok(Math.abs(jitVals[i]! - info.ref_out[i]!) < 1e-5, `TorchScript mismatch at ${i}`);
        }

        // Verify AOTInductor (.pt2) execution if Python compiled it cleanly
        if (info.aoti_ok && fs.existsSync(pt2Path)) {
          using aoti = torch.loadAotiModel(pt2Path);
          const [aotiOut] = aoti.run([x]);
          assert.ok(aotiOut !== undefined);
          const aotiVals = Array.from(aotiOut.toTypedArray() as Float32Array);
          for (let i = 0; i < info.ref_out.length; i++) {
            assert.ok(Math.abs(aotiVals[i]! - info.ref_out[i]!) < 1e-5, `AOTInductor mismatch at ${i}`);
          }
        }
      });
    } finally {
      if (fs.existsSync(tsPath)) fs.unlinkSync(tsPath);
      if (fs.existsSync(pt2Path)) fs.unlinkSync(pt2Path);
    }
  });
});
