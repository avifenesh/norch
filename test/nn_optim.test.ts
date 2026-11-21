import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, it } from 'node:test';
import torch, { nn, optim, utils } from '../src/index.js';

describe('nn.Module, optimizers, schedulers, DataLoader & safetensors', () => {
  it('auto-discovers parameters, buffers, train/eval modes, and stateDict', () => {
    class SmallNet extends nn.Module {
      readonly fc1 = new nn.Linear(8, 16);
      readonly bn = new nn.BatchNorm1d(16);
      readonly drop = new nn.Dropout(0.5);
      readonly fc2 = new nn.Linear(16, 4);

      forward(x: torch.Tensor): torch.Tensor {
        return this.fc2.forward(this.drop.forward(this.bn.forward(this.fc1.forward(x)).relu()));
      }
    }

    using net = new SmallNet();
    const names = Array.from(net.namedParameters(), ([n]) => n);
    assert.deepEqual(names, [
      'fc1.weight',
      'fc1.bias',
      'bn.weight',
      'bn.bias',
      'fc2.weight',
      'fc2.bias',
    ]);
    assert.equal(net.numParameters(), 8 * 16 + 16 + 16 + 16 + 16 * 4 + 4);

    net.eval();
    assert.equal(net.drop.training, false);
    assert.equal(net.bn.training, false);

    // Deterministic eval forward (dropout disabled)
    torch.tidy(() => {
      const x = torch.randn([4, 8]);
      const y1 = net.forward(x);
      const y2 = net.forward(x);
      assert.ok(y1.allclose(y2, 1e-6, 1e-6));
    });

    // stateDict roundtrip + safetensors binary serialization roundtrip
    const tmpDir = path.join(os.homedir(), '.cache', 'norch');
    fs.mkdirSync(tmpDir, { recursive: true });
    const stPath = path.join(tmpDir, 'test_model.safetensors');
    try {
      const sd = net.stateDict();
      utils.saveSafetensors(stPath, sd, { framework: 'norch', test: '1' });
      for (const t of Object.values(sd)) t.dispose();

      using net2 = new SmallNet();
      const loaded = utils.loadSafetensors(stPath);
      assert.equal(loaded.metadata['framework'], 'norch');
      const res = net2.loadStateDict(loaded.tensors, true);
      assert.equal(res.missingKeys.length, 0);
      assert.equal(res.unexpectedKeys.length, 0);
      for (const t of Object.values(loaded.tensors)) t.dispose();

      assert.ok(net.fc1.weight.equal(net2.fc1.weight));
      assert.ok(net.fc2.bias!.equal(net2.fc2.bias!));
    } finally {
      if (fs.existsSync(stPath)) fs.unlinkSync(stPath);
    }
  });

  it('runs LR schedulers and gradient clipping accurately', () => {
    using p = new nn.Parameter(torch.tensor([3.0, 4.0]));
    using opt = new optim.SGD([p], { lr: 0.1, momentum: 0.9, nesterov: true });

    const stepSched = new optim.StepLR(opt, 2, 0.5);
    assert.ok(Math.abs(opt.paramGroups[0]!.lr - 0.1) < 1e-9);
    stepSched.step();
    assert.ok(Math.abs(opt.paramGroups[0]!.lr - 0.1) < 1e-9);
    stepSched.step();
    assert.ok(Math.abs(opt.paramGroups[0]!.lr - 0.05) < 1e-9);

    const cosOpt = new optim.AdamW([p], { lr: 1.0 });
    const cosSched = new optim.CosineAnnealingLR(cosOpt, 4, 0.0);
    for (let i = 0; i < 4; i++) cosSched.step();
    assert.ok(Math.abs(cosOpt.paramGroups[0]!.lr) < 1e-9);
    cosOpt.dispose();

    // Gradient clipping
    torch.tidy(() => {
      const loss = p.pow(2).sum(); // grad = [6, 8], norm = 10
      loss.backward();
      const totalNorm = utils.clipGradNorm_([p], 5.0);
      assert.ok(Math.abs(totalNorm - 10.0) < 1e-4);
      const clippedNorm = p.grad!.norm().item();
      assert.ok(Math.abs(clippedNorm - 5.0) < 1e-4);
    });
  });

  it('batches and shuffles datasets synchronously and asynchronously via DataLoader', async () => {
    using x = torch.arange(20, { dtype: torch.float32 }).reshape(10, 2);
    using y = torch.arange(10, { dtype: torch.int64 });
    const ds = new utils.TensorDataset(x, y);
    const loader = new utils.DataLoader(ds, { batchSize: 4, shuffle: true, seed: 42 });
    assert.equal(loader.length, 3);

    const seenLabels: number[] = [];
    for (const [bx, by] of loader) {
      assert.equal(bx!.shape[1], 2);
      seenLabels.push(...(by!.toArray() as number[]));
      bx!.dispose();
      by!.dispose();
    }
    assert.equal(seenLabels.length, 10);
    assert.deepEqual([...seenLabels].sort((a, b) => a - b), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    let asyncBatches = 0;
    const dropLoader = new utils.DataLoader(ds, { batchSize: 4, dropLast: true });
    for await (const [bx, by] of dropLoader) {
      assert.deepEqual(bx!.shape, [4, 2]);
      assert.deepEqual(by!.shape, [4]);
      bx!.dispose();
      by!.dispose();
      asyncBatches += 1;
    }
    assert.equal(asyncBatches, 2);
  });

  it('roundtrips all dtypes including bfloat16/float16/int64/bool through safetensors and verifies topk/gather/var/std', () => {
    torch.tidy(() => {
      const tensors: Record<string, torch.Tensor> = {
        f32: torch.tensor([1.25, -2.5, 3.75], { dtype: torch.float32 }),
        f64: torch.tensor([Math.PI, Math.E], { dtype: torch.float64 }),
        f16: torch.tensor([0.5, -1.5, 2.0], { dtype: torch.float16 }),
        bf16: torch.tensor([1.0, -3.5, 4.25], { dtype: torch.bfloat16 }),
        i64: torch.tensor([10, -20, 30], { dtype: torch.int64 }),
        u8: torch.tensor([0, 128, 255], { dtype: torch.uint8 }),
        b: torch.tensor([true, false, true], { dtype: torch.bool }),
      };
      const encoded = utils.encodeSafetensors(tensors, { author: 'norch' });
      const { tensors: decoded, metadata } = utils.decodeSafetensors(encoded);
      assert.equal(metadata['author'], 'norch');
      for (const [k, orig] of Object.entries(tensors)) {
        const got = decoded[k];
        assert.ok(got !== undefined, `missing key ${k}`);
        assert.equal(got.dtype, orig.dtype);
        assert.deepEqual(got.shape, orig.shape);
        assert.ok(got.equal(orig), `bit-exact mismatch for ${k} (${orig.dtype})`);
      }

      const v = torch.tensor([10, 40, 20, 30], { dtype: torch.float32 });
      const { values, indices } = v.topk(2);
      assert.deepEqual(values.toArray(), [40, 30]);
      assert.deepEqual(indices.toArray(), [1, 3]);
      assert.deepEqual(v.gather(0, indices).toArray(), [40, 30]);
      assert.deepEqual(v.indexSelect(0, indices).toArray(), [40, 30]);
      assert.ok(Math.abs(v.std().item() - Math.sqrt(500 / 3)) < 1e-4);
    });
  });
});
