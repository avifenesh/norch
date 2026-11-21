import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, it } from 'node:test';
import torch, {
  ao,
  distributions,
  func,
  nn,
  optim,
  profiler,
  QuantizedLinear,
  utils,
  type Tensor,
} from '../src/index.js';

function toNums(t: Tensor): number[] {
  return Array.from(t.toTypedArray() as ArrayLike<number | bigint>, Number);
}

function assertClose(a: number[], b: number[], tol = 1e-4, label = ''): void {
  assert.equal(a.length, b.length, `${label} length mismatch`);
  for (let i = 0; i < a.length; i++) {
    assert.ok(
      Math.abs(a[i]! - b[i]!) <= tol,
      `${label}[${i}] mismatch: ${a[i]} vs ${b[i]} (tol=${tol})`
    );
  }
}

describe('Fused AMP GradScaler + checkpoint stateDict', () => {
  it('unscales via fused ATen kernel, skips on inf, and roundtrips stateDict through saveCheckpoint', () => {
    const cacheDir = path.join(os.homedir(), '.cache', 'norch');
    fs.mkdirSync(cacheDir, { recursive: true });
    const ckptPath = path.join(cacheDir, `scaler-ckpt-${process.pid}.safetensors`);

    try {
      using lin = new nn.Linear(3, 2);
      using opt = new optim.AdamW(lin.parameters(), { lr: 1e-2 });
      const scaler = new torch.amp.GradScaler({
        initScale: 128,
        growthFactor: 2,
        backoffFactor: 0.5,
        growthInterval: 2,
      });

      // Step 1: finite grads -> unscaled and stepped
      torch.tidy(() => {
        opt.zeroGrad();
        const x = torch.ones([2, 3]);
        const loss = lin.forward(x).pow(2).sum();
        scaler.scale(loss).backward();
        const stepped = scaler.step(opt);
        assert.equal(stepped, true);
        scaler.update();
      });
      assert.equal(scaler.getScale(), 128);

      // Step 2: finite grads -> hits growthInterval=2, doubles scale to 256
      torch.tidy(() => {
        opt.zeroGrad();
        const loss = lin.forward(torch.ones([2, 3])).pow(2).sum();
        scaler.scale(loss).backward();
        assert.equal(scaler.step(opt), true);
        scaler.update();
      });
      assert.equal(scaler.getScale(), 256);

      // Step 3: inject Inf -> fused kernel flags foundInf, skips step, halves scale to 128
      torch.tidy(() => {
        opt.zeroGrad();
        const loss = lin.forward(torch.ones([2, 3])).sum();
        scaler.scale(loss).backward();
        lin.weight.grad!.fill_(Number.POSITIVE_INFINITY);
        const before = toNums(lin.weight);
        assert.equal(scaler.step(opt), false);
        assertClose(toNums(lin.weight), before, 0);
        scaler.update();
      });
      assert.equal(scaler.getScale(), 128);

      // Save & restore checkpoint including GradScaler
      torch.saveCheckpoint(ckptPath, { model: lin, optimizer: opt, scaler, step: 3 });
      using lin2 = new nn.Linear(3, 2);
      using opt2 = new optim.AdamW(lin2.parameters(), { lr: 1e-2 });
      const scaler2 = new torch.amp.GradScaler({ initScale: 65536 });
      const meta = torch.loadCheckpoint(ckptPath, { model: lin2, optimizer: opt2, scaler: scaler2 });
      assert.equal(meta.step, 3);
      assert.equal(scaler2.getScale(), 128);
      assert.deepEqual(scaler2.stateDict(), scaler.stateDict());
    } finally {
      fs.rmSync(ckptPath, { force: true });
    }
  });
});

describe('DLPack zero-copy interop & native libtorch profiler', () => {
  it('roundtrips tensors zero-copy through DLPack capsules with ownership safety', () => {
    torch.tidy(() => {
      const src = torch.tensor(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        { dtype: 'float32' }
      );
      const dev = utils.dlpackDevice(src);
      assert.deepEqual(dev, [1, 0]); // kDLCPU = 1, device_id = 0

      const capsule = utils.toDlpack(src);
      assert.ok(utils.dlpackPointer(capsule) !== 0n);

      const imported = utils.fromDlpack(capsule);
      assert.deepEqual(imported.shape, [2, 3]);
      assert.equal(imported.dtype, 'float32');
      assert.equal(imported.dataPtr, src.dataPtr); // true zero-copy shared storage!

      // Mutating imported tensor mutates src in place
      imported.mul_(10);
      assertClose(toNums(src), [10, 20, 30, 40, 50, 60], 1e-6);

      // Double-consume must throw cleanly
      assert.throws(() => utils.fromDlpack(capsule), /already been consumed/);

      // Unconsumed capsule must not leak or crash
      const unused = utils.toDlpack(src);
      assert.ok(utils.dlpackPointer(unused) !== 0n);
    });
  });

  it('captures native ATen profiler events, keyAverages table, and Chrome trace JSON', () => {
    const cacheDir = path.join(os.homedir(), '.cache', 'norch');
    fs.mkdirSync(cacheDir, { recursive: true });
    const tracePath = path.join(cacheDir, `trace-${process.pid}.json`);

    try {
      const { result, prof } = profiler.profile(
        { activities: ['cpu'], recordShapes: true, profileMemory: true },
        () =>
          torch.tidy(() => {
            const a = torch.randn([16, 16], { requiresGrad: true });
            const b = torch.randn([16, 16], { requiresGrad: true });
            const y = a.matmul(b).relu().sum();
            y.backward();
            return y.item();
          })
      );

      assert.ok(Number.isFinite(result));
      const events = prof.events();
      assert.ok(events.length > 0, 'Expected recorded ATen events');
      const avgs = prof.keyAverages();
      assert.ok(avgs.length > 0);
      const tableStr = prof.table({ rowLimit: 10 });
      assert.ok(tableStr.includes('CPU Total (us)'));

      prof.exportChromeTrace(tracePath);
      const traceRaw = fs.readFileSync(tracePath, 'utf8');
      const parsed = JSON.parse(traceRaw) as { traceEvents?: unknown[] };
      assert.ok(Array.isArray(parsed.traceEvents) && parsed.traceEvents.length > 0);
      prof.dispose();
    } finally {
      fs.rmSync(tracePath, { force: true });
    }
  });
});

describe('torch.func (grad, jacrev, jacfwd, hessian, vmap, functionalCall)', () => {
  it('computes exact jacrev/jacfwd agreement, analytical Hessian, vmap, and functionalCall', () => {
    torch.tidy(() => {
      const vecFn = (x: Tensor): Tensor => {
        const x0 = x.at(0);
        const x1 = x.at(1);
        return torch.stack([x0.pow(2).add(x1), x0.sin().mul(x1), x0.mul(x1.pow(2))], 0);
      };

      const x = torch.tensor([1.2, -0.7], { dtype: 'float64' });
      const jr = func.jacrev(vecFn)(x);
      const jf = func.jacfwd(vecFn)(x);
      assert.deepEqual(jr.shape, [3, 2]);
      assert.deepEqual(jf.shape, [3, 2]);
      assert.ok(jr.allclose(jf, 1e-5, 1e-6));

      // Hessian of f(x) = x0^3 + 2*x0*x1 + x1^2 -> [[6*x0, 2], [2, 2]]
      const scalarFn = (v: Tensor): Tensor => {
        const v0 = v.at(0);
        const v1 = v.at(1);
        return v0.pow(3).add(v0.mul(v1).mul(2)).add(v1.pow(2));
      };
      const pt = torch.tensor([2.0, 3.0], { dtype: 'float64' });
      const h = func.hessian(scalarFn)(pt);
      assert.deepEqual(h.shape, [2, 2]);
      assertClose(toNums(h), [12, 2, 2, 2], 1e-5, 'hessian');

      // valueAndGrad
      const { value, grad: g } = func.valueAndGrad(scalarFn)(pt);
      assert.ok(Math.abs(value.item() - (8 + 12 + 9)) < 1e-6);
      assertClose(toNums(g as Tensor), [12 + 6, 4 + 6], 1e-5, 'valueAndGrad');

      // vmap with autograd
      const mat = torch.randn([4, 3], { requiresGrad: true });
      const vec = torch.randn([4, 3], { requiresGrad: true });
      const batchedDot = func.vmap((a: Tensor, b: Tensor) => a.mul(b).sum());
      const dots = batchedDot(mat, vec);
      assert.deepEqual(dots.shape, [4]);
      const refDots = mat.mul(vec).sum(1);
      assert.ok(dots.allclose(refDots, 1e-5, 1e-6));
      dots.sum().backward();
      assert.ok(mat.grad!.allclose(vec, 1e-5, 1e-6));

      // functionalCall differentiates w.r.t. passed parameter dict without mutating module
      using lin = new nn.Linear(3, 2);
      const origW = toNums(lin.weight);
      const customW = torch.ones([2, 3], { requiresGrad: true });
      const customB = torch.zeros([2], { requiresGrad: true });
      const inp = torch.tensor([[1.0, 2.0, 3.0]]);
      const out = func.functionalCall<Tensor>(lin, { weight: customW, bias: customB }, inp);
      assertClose(toNums(out), [6, 6], 1e-5);
      out.sum().backward();
      assertClose(toNums(customW.grad!), [1, 2, 3, 1, 2, 3], 1e-5);
      assertClose(toNums(lin.weight), origW, 0);
    });
  });
});

describe('Module backward hooks, Sequential.call, indexPut_ accumulate, half-precision DataLoader & Sparse CSR/CSC', () => {
  it('fires full backward hooks through Sequential, accumulates duplicate indexPut_, and handles CSR/CSC', () => {
    torch.tidy(() => {
      // Module full backward hook through nn.Sequential
      using l1 = new nn.Linear(3, 3);
      using l2 = new nn.Linear(3, 2);
      using seq = new nn.Sequential(l1, l2);
      let hookCalled = false;
      using _bwd = l2.registerFullBackwardHook((_m, gradIn, gradOut) => {
        hookCalled = true;
        assert.ok(gradOut[0] !== null);
        // Scale gradInput by 2x to confirm modification propagates to l1!
        return [gradIn[0]!.mul(2)];
      });

      const x = torch.randn([2, 3], { requiresGrad: true });
      const y = seq.call(x) as Tensor;
      y.sum().backward();
      assert.equal(hookCalled, true);
      assert.ok(x.grad !== null);

      // indexPut_ with accumulate=true on duplicate indices
      const target = torch.zeros([4], { dtype: 'float32' });
      const idx = torch.tensor([0, 2, 0, 2], { dtype: 'int64' });
      const vals = torch.tensor([1.5, 2.0, 3.5, 4.0], { dtype: 'float32' });
      target.indexPut_([idx], vals, true);
      assertClose(toNums(target), [5.0, 0.0, 6.0, 0.0], 1e-6, 'indexPut accumulate');

      // Sparse CSR & CSC layouts
      const dense = torch.tensor(
        [
          [1, 0, 2],
          [0, 0, 3],
          [4, 5, 0],
        ],
        { dtype: 'float32' }
      );
      const csr = dense.toSparseCsr();
      assert.equal(csr.layout, 'sparse_csr');
      assert.equal(csr.isSparse, true);
      assert.deepEqual(toNums(csr.crowIndices()), [0, 2, 3, 5]);
      assert.deepEqual(toNums(csr.colIndices()), [0, 2, 2, 0, 1]);
      assert.ok(csr.toDense().equal(dense));

      const csrDirect = torch.sparseCsrTensor(
        torch.tensor([0, 2, 3, 5], { dtype: 'int64' }),
        torch.tensor([0, 2, 2, 0, 1], { dtype: 'int64' }),
        torch.tensor([1, 2, 3, 4, 5], { dtype: 'float32' }),
        [3, 3]
      );
      assert.ok(csrDirect.toDense().equal(dense));

      const rhs = torch.randn([3, 2]);
      assert.ok(torch.sparse.mm(csrDirect, rhs).allclose(dense.matmul(rhs), 1e-5, 1e-6));

      const csc = dense.toSparseCsc();
      assert.equal(csc.layout, 'sparse_csc');
      assert.ok(csc.ccolIndices().numel === 4);
      assert.ok(csc.rowIndices().numel === 5);
      assert.ok(csc.toDense().equal(dense));
    });
  });

  it('streams float16 and bfloat16 TensorDatasets through multi-worker DataLoader with zero leaks', async () => {
    using rawRange = torch.arange(0, 16, 1, { dtype: 'float32' });
    using reshaped = rawRange.reshape(8, 2);
    using xHalf = reshaped.half();
    using xBf16 = reshaped.bfloat16();
    const ds = new utils.TensorDataset(xHalf, xBf16);
    const loader = new utils.DataLoader(ds, { batchSize: 4, numWorkers: 2 });

    const runEpoch = async (): Promise<number> => {
      let totalRows = 0;
      for await (const [bh, bb] of loader) {
        assert.equal(bh!.dtype, 'float16');
        assert.equal(bb!.dtype, 'bfloat16');
        assert.deepEqual(bh!.shape, [4, 2]);
        torch.tidy(() => {
          assert.ok(bh!.float().allclose(bb!.float(), 1e-2, 1e-2));
        });
        totalRows += bh!.size(0);
        bh!.dispose();
        bb!.dispose();
      }
      return totalRows;
    };

    assert.equal(await runEpoch(), 8);
    const afterEpoch1 = torch.memoryStats().liveTensors;
    assert.equal(await runEpoch(), 8);
    assert.ok(torch.memoryStats().liveTensors <= afterEpoch1);
  });
});

describe('Extended nn layers, TransformerEncoder/Decoder & parametrizations', () => {
  it('runs ConvTranspose3d, 1D/3D pools, AdaptiveMaxPool2d, PixelShuffle, Unfold/Fold, InstanceNorm, BatchNorm3d, Bilinear, EmbeddingBag, and TransformerEncoder/Decoder', () => {
    torch.tidy(() => {
      using ct3 = new nn.ConvTranspose3d(2, 3, 2, { stride: 2 });
      assert.deepEqual(ct3.forward(torch.randn([1, 2, 2, 2, 2])).shape, [1, 3, 4, 4, 4]);

      using mp1 = new nn.MaxPool1d(2);
      assert.deepEqual(mp1.forward(torch.randn([2, 3, 8])).shape, [2, 3, 4]);
      using mp3 = new nn.MaxPool3d(2);
      assert.deepEqual(mp3.forward(torch.randn([1, 2, 4, 4, 4])).shape, [1, 2, 2, 2, 2]);
      using ap1 = new nn.AvgPool1d(2);
      assert.deepEqual(ap1.forward(torch.randn([2, 3, 8])).shape, [2, 3, 4]);
      using ap3 = new nn.AvgPool3d(2);
      assert.deepEqual(ap3.forward(torch.randn([1, 2, 4, 4, 4])).shape, [1, 2, 2, 2, 2]);
      using amp2 = new nn.AdaptiveMaxPool2d([2, 3]);
      assert.deepEqual(amp2.forward(torch.randn([1, 4, 6, 6])).shape, [1, 4, 2, 3]);

      // PixelShuffle / PixelUnshuffle exact inverse
      using ps = new nn.PixelShuffle(2);
      using pu = new nn.PixelUnshuffle(2);
      const img = torch.randn([1, 8, 4, 4]);
      const up = ps.forward(img);
      assert.deepEqual(up.shape, [1, 2, 8, 8]);
      assert.ok(pu.forward(up).equal(img));

      // Unfold / Fold
      using unf = new nn.Unfold(2, { stride: 2 });
      using fld = new nn.Fold([4, 4], 2, { stride: 2 });
      const patches = unf.forward(img);
      assert.deepEqual(patches.shape, [1, 32, 4]);
      assert.ok(fld.forward(patches).allclose(img, 1e-5, 1e-6));

      // Norms
      using in1 = new nn.InstanceNorm1d(4, { affine: true });
      assert.deepEqual(in1.forward(torch.randn([2, 4, 8])).shape, [2, 4, 8]);
      using in2 = new nn.InstanceNorm2d(4, { affine: true });
      assert.deepEqual(in2.forward(torch.randn([2, 4, 6, 6])).shape, [2, 4, 6, 6]);
      using bn3 = new nn.BatchNorm3d(3);
      assert.deepEqual(bn3.forward(torch.randn([2, 3, 4, 4, 4])).shape, [2, 3, 4, 4, 4]);

      // Bilinear & EmbeddingBag
      using bil = new nn.Bilinear(4, 5, 3);
      const bOut = bil.forward(torch.randn([2, 4]), torch.randn([2, 5]));
      assert.deepEqual(bOut.shape, [2, 3]);
      bOut.sum().backward();
      assert.ok(bil.weight.grad !== null);

      using eb = new nn.EmbeddingBag(10, 6, { mode: 'mean' });
      const ebOut = eb.forward(
        torch.tensor([0, 1, 2, 3, 4], { dtype: 'int64' }),
        torch.tensor([0, 2], { dtype: 'int64' })
      );
      assert.deepEqual(ebOut.shape, [2, 6]);
      ebOut.sum().backward();
      assert.ok(eb.weight.grad !== null);

      // TransformerEncoder & TransformerDecoder
      using enc = new nn.TransformerEncoder(() => new nn.TransformerEncoderLayer(16, 4), 2, new nn.LayerNorm(16));
      using dec = new nn.TransformerDecoder(() => new nn.TransformerDecoderLayer(16, 4), 2, new nn.LayerNorm(16));
      const src = torch.randn([2, 5, 16], { requiresGrad: true });
      const tgt = torch.randn([2, 4, 16], { requiresGrad: true });
      const mem = enc.forward(src);
      const decOut = dec.forward(tgt, mem, { tgtIsCausal: true });
      assert.deepEqual(decOut.shape, [2, 4, 16]);
      decOut.sum().backward();
      assert.ok(src.grad !== null && tgt.grad !== null);
    });
  });

  it('applies and removes weightNorm and spectralNorm parametrizations cleanly', () => {
    torch.tidy(() => {
      using linW = new nn.Linear(4, 3);
      const wn = nn.utils.weightNorm(linW, 'weight', 0);
      const x = torch.randn([2, 4]);
      const y = linW.call(x) as Tensor;
      y.sum().backward();
      wn.remove();
      assert.equal(linW.parameters().length, 2); // weight + bias restored

      using linS = new nn.Linear(4, 4, { bias: false });
      torch.noGrad(() => {
        linS.weight.mul_(5.0);
      });
      const sn = nn.utils.spectralNorm(linS, 'weight', { nPowerIterations: 10 });
      (linS.call(x) as Tensor).sum().backward();
      const svals = torch.linalg.svd(linS.weight.detach()).S;
      const topS = svals.at(0).item();
      assert.ok(Math.abs(topS - 1.0) < 0.05, `Expected spectral norm ~1.0, got ${topS}`);
      sn.remove();
      assert.equal(linS.parameters().length, 1);
    });
  });
});

describe('Extended optimizers, schedulers, EMA, distributions & quantization', () => {
  it('converges with Adagrad, Adadelta, Adamax, NAdam, RAdam, and LBFGS + runs new schedulers & EMA', () => {
    const testFirstOrder = (
      name: string,
      makeOpt: (p: nn.Parameter[]) => optim.Optimizer
    ): void => {
      using model = new nn.Linear(4, 2);
      using opt = makeOpt(model.parameters());
      using x = torch.randn([16, 4]);
      using y = torch.randn([16, 2]);
      let firstLoss = 0;
      let lastLoss = 0;
      for (let i = 0; i < 25; i++) {
        const l = torch.tidy(() => {
          opt.zeroGrad();
          const loss = torch.F.mseLoss(model.forward(x), y);
          loss.backward();
          opt.step();
          return loss.item();
        });
        if (i === 0) firstLoss = l;
        lastLoss = l;
      }
      assert.ok(lastLoss < firstLoss * 0.7, `${name} did not decrease loss (${firstLoss} -> ${lastLoss})`);
    };

    torch.manualSeed(42);
    testFirstOrder('Adagrad', (p) => new optim.Adagrad(p, { lr: 0.2 }));
    testFirstOrder('Adadelta', (p) => new optim.Adadelta(p, { lr: 2.0, eps: 1e-4 }));
    testFirstOrder('Adamax', (p) => new optim.Adamax(p, { lr: 0.1 }));
    testFirstOrder('NAdam', (p) => new optim.NAdam(p, { lr: 0.1 }));
    testFirstOrder('RAdam', (p) => new optim.RAdam(p, { lr: 0.1 }));

    // L-BFGS minimizing quadratic f(w) = ||w - target||^2
    using wInit = torch.tensor([5.0, -4.0, 3.0]);
    using w = new nn.Parameter(wInit, true);
    using target = torch.tensor([1.0, 2.0, -1.0]);
    using lbfgs = new optim.LBFGS([w], { lr: 1.0, maxIter: 15 });
    const finalLossT = lbfgs.step(() => {
      lbfgs.zeroGrad();
      const l = w.sub(target).pow(2).sum();
      l.backward();
      return l;
    });
    const finalLoss = finalLossT!.item();
    finalLossT!.dispose();
    assert.ok(finalLoss < 1e-4, `LBFGS should converge near 0, got ${finalLoss}`);

    // Schedulers
    using lin = new nn.Linear(2, 2);
    using opt = new optim.SGD(lin.parameters(), { lr: 1.0 });
    const exp = new optim.ExponentialLR(opt, 0.5);
    exp.step();
    assert.ok(Math.abs(opt.paramGroups[0]!.lr - 0.5) < 1e-6);

    opt.paramGroups[0]!.initialLr = 1.0;
    const ms = new optim.MultiStepLR(opt, [2, 4], 0.1);
    ms.step(2);
    assert.ok(Math.abs(opt.paramGroups[0]!.lr - 0.1) < 1e-6);

    const lam = new optim.LambdaLR(opt, (e) => 1 / (e + 1));
    lam.step(3);
    assert.ok(Math.abs(opt.paramGroups[0]!.lr - 0.25) < 1e-6);

    const warm = new optim.CosineAnnealingWarmRestarts(opt, 4, 2, 0.0);
    warm.step(4); // restarts back to base LR 1.0
    assert.ok(Math.abs(opt.paramGroups[0]!.lr - 1.0) < 1e-6);

    // EMA AveragedModel
    using liveModel = new nn.Linear(2, 2, { bias: false });
    using cloneModel = new nn.Linear(2, 2, { bias: false });
    using ema = new optim.AveragedModel(cloneModel, { avgFn: optim.getEmaAvgFn(0.9) });
    torch.noGrad(() => {
      liveModel.weight.fill_(2.0);
    });
    ema.updateParameters(liveModel); // init to 2.0
    torch.noGrad(() => {
      liveModel.weight.fill_(12.0);
    });
    ema.updateParameters(liveModel); // 0.9*2 + 0.1*12 = 3.0
    assertClose(toNums(ema.module.weight), [3, 3, 3, 3], 1e-5, 'EMA weights');
  });

  it('supports MultivariateNormal, Exponential, Poisson, Gamma, Beta, Dirichlet, Independent & quantization', () => {
    torch.tidy(() => {
      // Exponential + KL
      const e1 = new distributions.Exponential(2.0);
      const e2 = new distributions.Exponential(4.0);
      assert.ok(e1.sample([8]).shape[0] === 8);
      // KL(Exp(2) || Exp(4)) = 4/2 - ln(4/2) - 1 = 1 - ln(2)
      assert.ok(Math.abs(distributions.klDivergence(e1, e2).item() - (1 - Math.log(2))) < 1e-5);

      // Poisson, Gamma, Beta, Dirichlet
      const pois = new distributions.Poisson(3.0);
      assert.equal(pois.sample([10]).shape[0], 10);
      assert.ok(Number.isFinite(pois.logProb(torch.tensor([3.0])).item()));

      const gam = new distributions.Gamma(2.0, 1.0);
      assert.ok(gam.rsample([8]).min().item() > 0);
      assert.ok(Number.isFinite(gam.entropy().item()));

      const beta = new distributions.Beta(2.0, 5.0);
      const bS = beta.rsample([8]);
      assert.ok(bS.min().item() > 0 && bS.max().item() < 1);
      assert.ok(Number.isFinite(beta.logProb(torch.tensor([0.3])).item()));

      const dir = new distributions.Dirichlet(torch.tensor([2.0, 3.0, 5.0]));
      const dS = dir.rsample([6]);
      assert.deepEqual(dS.shape, [6, 3]);
      assertClose(toNums(dS.sum(-1)), [1, 1, 1, 1, 1, 1], 1e-5);

      // MultivariateNormal rsample gradient + KL
      const loc = torch.tensor([0.0, 1.0], { requiresGrad: true });
      const cov = torch.tensor([
        [2.0, 0.5],
        [0.5, 1.5],
      ]);
      const mvn = new distributions.MultivariateNormal(loc, { covarianceMatrix: cov });
      const s = mvn.rsample([16]);
      assert.deepEqual(s.shape, [16, 2]);
      s.sum().backward();
      assertClose(toNums(loc.grad!), [16, 16], 1e-4);
      assert.ok(Number.isFinite(mvn.logProb(torch.tensor([0.5, 0.5])).item()));
      assert.ok(Number.isFinite(mvn.entropy().item()));
      assert.ok(Math.abs(distributions.klDivergence(mvn, mvn).item()) < 1e-5);

      // Independent
      const baseN = new distributions.Normal(torch.zeros([3, 4]), torch.ones([3, 4]));
      const ind = new distributions.Independent(baseN, 1);
      assert.deepEqual(ind.batchShape, [3]);
      assert.deepEqual(ind.logProb(torch.zeros([3, 4])).shape, [3]);

      // QAT FakeQuantize STE + dynamic INT8 quantization
      using fq = new ao.quantization.FakeQuantize({ bits: 8, symmetric: true });
      const qIn = torch.linspace(-1.0, 1.0, 16).requiresGrad_(true);
      fq.calibrate(qIn);
      const qOut = fq.forward(qIn);
      qOut.sum().backward();
      assert.ok(qIn.grad!.sum().item() > 0, 'STE gradient must flow through FakeQuantize');

      class SmallNet extends nn.Module {
        readonly fc1 = new nn.Linear(8, 16);
        readonly fc2 = new nn.Linear(16, 4);
        forward(x: Tensor): Tensor {
          return this.fc2.forward(this.fc1.forward(x).relu());
        }
      }
      using net = new SmallNet();
      const probe = torch.randn([4, 8]);
      const fpOut = net.forward(probe).detach().clone();
      ao.quantization.quantizeDynamic(net);
      assert.ok(net.fc1 instanceof QuantizedLinear);
      assert.equal((net.fc1 as unknown as QuantizedLinear).weightInt8.dtype, 'int8');
      const qNetOut = net.forward(probe);
      assert.ok(qNetOut.allclose(fpOut, 5e-2, 5e-2));
    });
  });
});
