import assert from 'node:assert/strict';
import { execFile, execFileSync } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';
import torch, { distributions, nn, optim, special, utils } from '../src/index.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..', '..');

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

function runWorkerProcess(
  scriptPath: string,
  envOverrides: Record<string, string>
): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    execFile(
      process.execPath,
      [scriptPath],
      {
        cwd: rootDir,
        env: { ...process.env, OMP_NUM_THREADS: '2', ...envOverrides },
        timeout: 45_000,
      },
      (err, stdout, stderr) => {
        if (err) {
          reject(new Error(`Worker exited with error: ${err.message}\nSTDOUT:\n${stdout}\nSTDERR:\n${stderr}`));
          return;
        }
        resolve({ stdout, stderr });
      }
    );
  });
}

function toNums(t: torch.Tensor): number[] {
  return Array.from(t.toTypedArray() as ArrayLike<number | bigint>, Number);
}

function assertArraysClose(a: number[], b: number[], tol = 1e-5, label = ''): void {
  assert.equal(a.length, b.length, `${label} length mismatch (${a.length} vs ${b.length})`);
  for (let i = 0; i < a.length; i++) {
    assert.ok(
      Math.abs(a[i]! - b[i]!) <= tol,
      `${label}[${i}] mismatch: ${a[i]} vs ${b[i]} (tol=${tol})`
    );
  }
}

describe('Distributed training (c10d TCPStore, FileStore, collectives, DDP)', () => {
  it('runs 2-process collectives and matches single-process reference on 5-step DDP training', async () => {
    assert.equal(torch.distributed.isAvailable(), true);
    assert.equal(torch.distributed.isGlooAvailable(), true);

    const port = await getFreePort();
    const workerScript = path.join(__dirname, 'helpers', 'dist_worker.js');
    const initMethod = `tcp://127.0.0.1:${port}`;

    const [r0, r1] = await Promise.all([
      runWorkerProcess(workerScript, { RANK: '0', WORLD_SIZE: '2', NORCH_INIT_METHOD: initMethod }),
      runWorkerProcess(workerScript, { RANK: '1', WORLD_SIZE: '2', NORCH_INIT_METHOD: initMethod }),
    ]);

    const res0 = JSON.parse(r0.stdout.trim()) as {
      ok: boolean;
      rank: number;
      ep0: number[];
      ep1: number[];
      weights: Record<string, number[]>;
    };
    const res1 = JSON.parse(r1.stdout.trim()) as {
      ok: boolean;
      rank: number;
      ep0: number[];
      ep1: number[];
      weights: Record<string, number[]>;
    };

    assert.equal(res0.ok, true);
    assert.equal(res1.ok, true);

    // DistributedSampler partitions all 8 indices disjointly across the 2 ranks
    const combinedEp0 = [...res0.ep0, ...res1.ep0].sort((a, b) => a - b);
    assert.deepEqual(combinedEp0, [0, 1, 2, 3, 4, 5, 6, 7]);
    assert.notDeepEqual(res0.ep0, res0.ep1);

    // Rank 0 and rank 1 must converge to identical weights
    for (const key of ['fc1W', 'fc1B', 'fc2W', 'fc2B'] as const) {
      assertArraysClose(res0.weights[key]!, res1.weights[key]!, 1e-6, `rank0 vs rank1 ${key}`);
    }

    // Single-process full-batch reference using the exact same initial seed (777) and dataset seed (2026)
    class TinyMLP extends nn.Module {
      readonly fc1 = new nn.Linear(4, 8);
      readonly fc2 = new nn.Linear(8, 2);
      forward(x: torch.Tensor): torch.Tensor {
        return this.fc2.forward(this.fc1.forward(x).tanh());
      }
    }

    torch.manualSeed(777);
    using refModel = new TinyMLP();
    using refOpt = new optim.SGD(refModel.parameters(), { lr: 0.1 });
    torch.manualSeed(2026);
    using fullX = torch.randn([8, 4]);
    using fullY = torch.randn([8, 2]);

    for (let step = 0; step < 5; step++) {
      torch.tidy(() => {
        refOpt.zeroGrad();
        const pred = refModel.forward(fullX);
        const loss = torch.F.mseLoss(pred, fullY);
        loss.backward();
        refOpt.step();
      });
    }

    assertArraysClose(res0.weights.fc1W!, toNums(refModel.fc1.weight), 1e-5, 'ddp vs ref fc1W');
    assertArraysClose(res0.weights.fc1B!, toNums(refModel.fc1.bias!), 1e-5, 'ddp vs ref fc1B');
    assertArraysClose(res0.weights.fc2W!, toNums(refModel.fc2.weight), 1e-5, 'ddp vs ref fc2W');
    assertArraysClose(res0.weights.fc2B!, toNums(refModel.fc2.bias!), 1e-5, 'ddp vs ref fc2B');
  });

  it('supports FileStore rendezvous across 2 processes', async () => {
    const cacheDir = path.join(os.homedir(), '.cache', 'norch');
    fs.mkdirSync(cacheDir, { recursive: true });
    const storeFile = path.join(cacheDir, `filestore-test-${process.pid}-${Date.now()}`);
    const workerScript = path.join(__dirname, 'helpers', 'dist_worker.js');

    try {
      const [r0, r1] = await Promise.all([
        runWorkerProcess(workerScript, {
          RANK: '0',
          WORLD_SIZE: '2',
          NORCH_INIT_METHOD: `file://${storeFile}`,
          NORCH_DIST_MODE: 'filestore',
        }),
        runWorkerProcess(workerScript, {
          RANK: '1',
          WORLD_SIZE: '2',
          NORCH_INIT_METHOD: `file://${storeFile}`,
          NORCH_DIST_MODE: 'filestore',
        }),
      ]);
      assert.deepEqual(JSON.parse(r0.stdout.trim()).sum, [3, 30]);
      assert.deepEqual(JSON.parse(r1.stdout.trim()).sum, [3, 30]);
    } finally {
      fs.rmSync(storeFile, { force: true });
    }
  });
});

describe('Multi-threaded DataLoader (worker_threads)', () => {
  it('prefetches TensorDataset batches across 2 worker threads in exact order with zero leaks', async () => {
    using x = torch.arange(0, 40, 1, { dtype: 'float32' }).reshape(20, 2);
    using y = torch.arange(0, 20, 1, { dtype: 'int64' });
    const ds = new utils.TensorDataset(x, y);
    const loader = new utils.DataLoader(ds, {
      batchSize: 4,
      shuffle: false,
      numWorkers: 2,
      prefetchFactor: 2,
    });

    assert.throws(() => {
      for (const _ of loader) {
        // sync iteration disallowed when numWorkers > 0
      }
    }, /for await/);

    const baseLive = torch.memoryStats().liveTensors;
    const seenY: number[] = [];
    let batchCount = 0;

    for await (const [bx, by] of loader) {
      assert.deepEqual(bx!.shape, [4, 2]);
      assert.deepEqual(by!.shape, [4]);
      const yVals = toNums(by!);
      const xVals = toNums(bx!);
      for (let i = 0; i < yVals.length; i++) {
        assert.equal(xVals[i * 2], yVals[i]! * 2);
        assert.equal(xVals[i * 2 + 1], yVals[i]! * 2 + 1);
      }
      seenY.push(...yVals);
      bx!.dispose();
      by!.dispose();
      batchCount++;
    }

    assert.equal(batchCount, 5);
    assert.deepEqual(
      seenY,
      Array.from({ length: 20 }, (_, i) => i)
    );
    assert.equal(torch.memoryStats().liveTensors, baseLive);

    // Early break must dispose prefetched buffers and shut down workers cleanly
    for await (const [bx, by] of loader) {
      bx!.dispose();
      by!.dispose();
      break;
    }
    assert.equal(torch.memoryStats().liveTensors, baseLive);
  });

  it('supports custom workerScript for off-main-thread dataset preparation', async () => {
    const customWorker = path.join(__dirname, 'helpers', 'custom_loader_worker.js');
    const dummyDataset: utils.Dataset<Float32Array> = {
      length: 6,
      getItem: () => new Float32Array(0),
    };
    const loader = new utils.DataLoader<Float32Array, torch.Tensor>(dummyDataset, {
      batchSize: 3,
      numWorkers: 2,
      workerScript: customWorker,
      workerData: { scale: 10 },
      collateFn: (items) => {
        const flat = new Float32Array(items.length * 2);
        items.forEach((arr, idx) => flat.set(arr, idx * 2));
        return torch.fromTypedArray(flat, [items.length, 2], { copy: false });
      },
    });

    const batches: number[][] = [];
    for await (const batch of loader) {
      batches.push(toNums(batch));
      batch.dispose();
    }
    assert.deepEqual(batches, [
      [0, 1, 10, 11, 20, 21],
      [30, 31, 40, 41, 50, 51],
    ]);
  });
});

describe('Complex tensors, sparse COO tensors, and negative-step slicing', () => {
  it('supports complex64/complex128 ops, views, polar, conj, and interleaved TypedArrays', () => {
    torch.tidy(() => {
      const re = torch.tensor([3, 0], { dtype: 'float32' });
      const im = torch.tensor([4, 1], { dtype: 'float32' });
      const z = torch.complex(re, im);

      assert.equal(z.isComplex, true);
      assert.equal(z.dtype, 'complex64');
      assert.deepEqual(z.shape, [2]);
      assertArraysClose(toNums(z.real), [3, 0], 1e-6);
      assertArraysClose(toNums(z.imag), [4, 1], 1e-6);
      assertArraysClose(toNums(z.abs()), [5, 1], 1e-6);
      assertArraysClose(toNums(z.conj().imag), [-4, -1], 1e-6);
      assertArraysClose(toNums(z.angle()), [Math.atan2(4, 3), Math.PI / 2], 1e-5);

      const asReal = torch.viewAsReal(z);
      assert.deepEqual(asReal.shape, [2, 2]);
      const back = torch.viewAsComplex(asReal);
      assert.equal(back.isComplex, true);
      assert.deepEqual(back.toArray(), [
        { real: 3, imag: 4 },
        { real: 0, imag: 1 },
      ]);

      // Interleaved Float32Array -> complex64 zero-copy roundtrip
      const raw = new Float32Array([1, -2, 5, 6]);
      const z2 = torch.fromTypedArray(raw, [2], { dtype: 'complex64', copy: false });
      assert.equal(z2.dtype, 'complex64');
      assert.deepEqual(toNums(z2), [1, -2, 5, 6]);

      // Polar coordinates
      const mag = torch.tensor([2, 3], { dtype: 'float32' });
      const ang = torch.tensor([0, Math.PI / 2], { dtype: 'float32' });
      const p = torch.polar(mag, ang);
      assertArraysClose(toNums(p.real), [2, 0], 1e-5);
      assertArraysClose(toNums(p.imag), [0, 3], 1e-5);
    });
  });

  it('supports sparse COO tensors, coalesce, toDense, and sparse.mm', () => {
    torch.tidy(() => {
      const idx = torch.tensor(
        [
          [0, 1, 2, 0],
          [1, 0, 2, 1],
        ],
        { dtype: 'int64' }
      );
      const vals = torch.tensor([2.0, 3.0, 4.0, 5.0], { dtype: 'float32' });
      const sp = torch.sparseCooTensor(idx, vals, [3, 3]);
      assert.equal(sp.isSparse, true);
      assert.equal(sp.sparseDim, 2);
      assert.equal(sp.denseDim, 0);

      const coalesced = sp.coalesce();
      assert.equal(coalesced.nnz, 3);
      assert.deepEqual(coalesced.indices().shape, [2, 3]);
      assertArraysClose(toNums(coalesced.values()), [7, 3, 4], 1e-6);

      const dense = coalesced.toDense();
      assert.deepEqual(dense.toArray(), [
        [0, 7, 0],
        [3, 0, 0],
        [0, 0, 4],
      ]);

      const rhs = torch.randn([3, 2]);
      const spRes = torch.sparse.mm(coalesced, rhs);
      const denseRes = dense.matmul(rhs);
      assert.ok(spRes.allclose(denseRes, 1e-5, 1e-6));
    });
  });

  it('supports negative-step slicing across 1D/ND tensors with autograd', () => {
    torch.tidy(() => {
      const a = torch.arange(0, 8, 1, { dtype: 'float32' }).requiresGrad_(true);
      const rev = a.at(torch.slice(null, null, -1));
      assert.deepEqual(toNums(rev), [7, 6, 5, 4, 3, 2, 1, 0]);

      const strided = a.at(torch.slice(6, 1, -2));
      assert.deepEqual(toNums(strided), [6, 4, 2]);

      const weights = torch.tensor([1, 2, 3], { dtype: 'float32' });
      strided.mul(weights).sum().backward();
      assert.deepEqual(toNums(a.grad!), [0, 0, 3, 0, 2, 0, 1, 0]);

      // Multi-dim + Ellipsis + negative step
      const mat = torch.arange(0, 12, 1, { dtype: 'float32' }).reshape(3, 4);
      const flippedCols = mat.at(torch.Ellipsis, torch.slice(null, null, -1));
      assert.deepEqual(flippedCols.toArray(), [
        [3, 2, 1, 0],
        [7, 6, 5, 4],
        [11, 10, 9, 8],
      ]);

      const bothFlipped = mat.at(torch.slice(null, null, -1), torch.slice(3, 0, -2));
      assert.deepEqual(bothFlipped.toArray(), [
        [11, 9],
        [7, 5],
        [3, 1],
      ]);
    });
  });
});

describe('torch.special and torch.distributions', () => {
  it('computes special functions accurately against analytical values', () => {
    torch.tidy(() => {
      const z = torch.tensor([0.0], { dtype: 'float64' });
      assert.ok(Math.abs(special.erf(z).item() - 0.0) < 1e-12);
      assert.ok(Math.abs(special.erfc(z).item() - 1.0) < 1e-12);
      assert.ok(Math.abs(special.erfinv(z).item() - 0.0) < 1e-12);
      assert.ok(Math.abs(special.ndtr(z).item() - 0.5) < 1e-12);
      assert.ok(Math.abs(special.ndtri(torch.tensor([0.5], { dtype: 'float64' })).item() - 0.0) < 1e-12);
      assert.ok(Math.abs(special.expit(z).item() - 0.5) < 1e-12);
      assert.ok(Math.abs(special.logit(torch.tensor([0.5], { dtype: 'float64' })).item() - 0.0) < 1e-12);
      assert.ok(Math.abs(special.sinc(z).item() - 1.0) < 1e-12);

      // Gamma(5) = 4! = 24 -> gammaln(5) = ln(24)
      const five = torch.tensor([5.0], { dtype: 'float64' });
      assert.ok(Math.abs(special.gammaln(five).item() - Math.log(24)) < 1e-10);
      assert.ok(Number.isFinite(special.digamma(five).item()));

      const lse = special.logsumexp(torch.tensor([[0.0, 0.0]], { dtype: 'float64' }), -1);
      assert.ok(Math.abs(lse.item() - Math.log(2)) < 1e-12);

      // entr(0.5) = -0.5 * ln(0.5)
      const half = torch.tensor([0.5], { dtype: 'float64' });
      assert.ok(Math.abs(special.entr(half).item() - -0.5 * Math.log(0.5)) < 1e-12);

      // xlogy(0, 0) === 0 and xlogy(2, e) === 2
      const x = torch.tensor([0.0, 2.0], { dtype: 'float64' });
      const y = torch.tensor([0.0, Math.E], { dtype: 'float64' });
      assertArraysClose(toNums(special.xlogy(x, y)), [0.0, 2.0], 1e-10);
    });
  });

  it('supports Normal/Uniform/Bernoulli/Categorical rsample autograd, logProb, entropy, and KL', () => {
    torch.tidy(() => {
      // Normal rsample pathwise gradients
      const loc = torch.tensor([1.0, 2.0], { dtype: 'float32', requiresGrad: true });
      const scale = torch.tensor([0.5, 1.5], { dtype: 'float32', requiresGrad: true });
      const n = new distributions.Normal(loc, scale);
      const samples = n.rsample([16]);
      assert.deepEqual(samples.shape, [16, 2]);
      samples.mean().backward();
      // d(mean(loc + eps*scale))/dloc_i = 1/2
      assertArraysClose(toNums(loc.grad!), [0.5, 0.5], 1e-5);
      assert.ok(Number.isFinite(scale.grad!.sum().item()));

      // Normal logProb & KL(N(0,1) || N(1,2)) = ln(2) + (1 + 1)/(2*4) - 0.5 = ln(2) - 0.25
      const p = new distributions.Normal(0, 1);
      const q = new distributions.Normal(1, 2);
      const kl = distributions.klDivergence(p, q).item();
      assert.ok(Math.abs(kl - (Math.log(2) - 0.25)) < 1e-5);
      assert.ok(Math.abs(p.entropy().item() - 0.5 * (1 + Math.log(2 * Math.PI))) < 1e-5);

      // Uniform
      const u = new distributions.Uniform(0, 2);
      const uVals = torch.tensor([-0.5, 1.0, 2.5]);
      const uLp = toNums(u.logProb(uVals));
      assert.equal(uLp[0], Number.NEGATIVE_INFINITY);
      assert.ok(Math.abs(uLp[1]! - -Math.log(2)) < 1e-5);
      assert.equal(uLp[2], Number.NEGATIVE_INFINITY);
      assert.ok(Math.abs(u.entropy().item() - Math.log(2)) < 1e-5);

      // Bernoulli
      const b = new distributions.Bernoulli({ probs: torch.tensor([0.25, 0.75]) });
      const bSamp = b.sample([10]);
      assert.deepEqual(bSamp.shape, [10, 2]);
      const bLp = b.logProb(torch.tensor([1.0, 0.0]));
      assertArraysClose(toNums(bLp), [Math.log(0.25), Math.log(0.25)], 1e-4);

      // Categorical
      const cat = new distributions.Categorical({
        logits: torch.tensor([
          [0.0, 0.0, 0.0],
          [0.0, 10.0, 0.0],
        ]),
      });
      const cSamp = cat.sample([8]);
      assert.deepEqual(cSamp.shape, [8, 2]);
      const cEnt = toNums(cat.entropy());
      assert.ok(Math.abs(cEnt[0]! - Math.log(3)) < 1e-5);
      assert.ok(cEnt[1]! < 0.01);
      const cLp = toNums(cat.logProb(torch.tensor([0, 1])));
      assert.ok(Math.abs(cLp[0]! - -Math.log(3)) < 1e-5);
      assert.ok(cLp[1]! > -0.01);

      const cat2 = new distributions.Categorical({
        probs: torch.tensor([
          [0.5, 0.25, 0.25],
          [0.1, 0.8, 0.1],
        ]),
      });
      const klCat = distributions.klDivergence(cat, cat2);
      assert.ok(klCat.at(0).item() > 0 && klCat.at(1).item() > 0);
    });
  });
});

describe('Packaging and setup CLI', () => {
  it('resolves official libtorch download URLs in --dry-run mode and detects local libtorch', () => {
    const out = execFileSync(
      process.execPath,
      [
        path.join(rootDir, 'scripts', 'download-libtorch.mjs'),
        '--backend=cu128',
        '--version=2.7.0',
        '--platform=linux',
        '--arch=x64',
        '--dry-run',
        '--json',
      ],
      { encoding: 'utf8' }
    );
    const parsed = JSON.parse(out) as { url: string; backend: string; version: string };
    assert.equal(parsed.backend, 'cu128');
    assert.equal(parsed.version, '2.7.0');
    assert.equal(
      parsed.url,
      'https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcu128.zip'
    );

    const detected = execFileSync(
      process.execPath,
      [path.join(rootDir, 'bin', 'norch-setup.mjs'), 'detect'],
      { encoding: 'utf8' }
    ).trim();
    assert.ok(fs.existsSync(path.join(detected, 'include', 'ATen')));
  });
});
