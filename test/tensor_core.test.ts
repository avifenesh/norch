import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import torch, { Ellipsis, slice } from '../src/index.js';

describe('Tensor core, indexing, zero-copy & memory management', () => {
  it('creates tensors from JS arrays, factories, and dtypes', () => {
    torch.tidy(() => {
      const a = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      assert.deepEqual(a.shape, [2, 3]);
      assert.equal(a.dtype, 'float32');
      assert.equal(a.device.type, 'cpu');
      assert.equal(a.numel, 6);
      assert.deepEqual(a.toArray(), [
        [1, 2, 3],
        [4, 5, 6],
      ]);

      const i64 = torch.arange(0, 10, 2);
      assert.equal(i64.dtype, 'int64');
      assert.deepEqual(i64.toArray(), [0, 2, 4, 6, 8]);

      const eye = torch.eye(3, 3, { dtype: torch.float64 });
      assert.equal(eye.dtype, 'float64');
      assert.equal(eye.sum().item(), 3);

      const lin = torch.linspace(0, 1, 5);
      assert.deepEqual(lin.shape, [5]);
      assert.ok(Math.abs(lin.at(2).item() - 0.5) < 1e-6);

      const b = torch.tensor([true, false, true]);
      assert.equal(b.dtype, 'bool');
      assert.deepEqual(b.toArray(), [true, false, true]);
    });
  });

  it('supports zero-copy fromTypedArray and dataView', () => {
    const src = new Float32Array([1.5, 2.5, 3.5, 4.5]);
    using t = torch.fromTypedArray(src, [2, 2], { copy: false });
    assert.deepEqual(t.shape, [2, 2]);
    assert.equal(t.at(0, 1).item(), 2.5);

    // Mutating JS TypedArray reflects immediately in libtorch tensor (proves zero-copy)
    src[1] = 99.0;
    assert.equal(t.at(0, 1).item(), 99.0);

    // Zero-copy view back to JS TypedArray reflects in-place libtorch mutations
    const view = t.dataView() as Float32Array;
    t.mul_(2);
    assert.equal(view[1], 198.0);
    assert.equal(src[1], 198.0);
  });

  it('supports PyTorch-style indexing, slicing, Ellipsis, null unsqueeze, and indexPut_', () => {
    torch.tidy(() => {
      const x = torch.arange(24, { dtype: torch.float32 }).reshape(2, 3, 4);

      const s1 = x.at(1, slice(0, 2), slice(1, 4, 2));
      assert.deepEqual(s1.shape, [2, 2]);
      assert.deepEqual(s1.toArray(), [
        [13, 15],
        [17, 19],
      ]);

      const s2 = x.at(Ellipsis, 0);
      assert.deepEqual(s2.shape, [2, 3]);
      assert.deepEqual(s2.toArray(), [
        [0, 4, 8],
        [12, 16, 20],
      ]);

      const s3 = x.at(':', null, 0);
      assert.deepEqual(s3.shape, [2, 1, 4]);

      const idx = torch.tensor([2, 0], { dtype: torch.int64 });
      const gathered = x.at(0, idx);
      assert.deepEqual(gathered.shape, [2, 4]);
      assert.deepEqual(gathered.toArray(), [
        [8, 9, 10, 11],
        [0, 1, 2, 3],
      ]);

      const y = torch.zeros([3, 3]);
      y.indexPut_([1, slice(0, 2)], 7);
      assert.deepEqual(y.toArray(), [
        [0, 0, 0],
        [7, 7, 0],
        [0, 0, 0],
      ]);
    });
  });

  it('deterministically releases native memory via Symbol.dispose and tidy()', () => {
    const before = torch.memoryStats();

    {
      using a = torch.randn([128, 128]);
      using aT = a.T;
      using b = a.matmul(aT);
      assert.ok(torch.memoryStats().liveTensors >= before.liveTensors + 3);
      assert.ok(torch.memoryStats().reportedExternalBytes > before.reportedExternalBytes);
    }

    const kept = torch.tidy(() => {
      const x = torch.randn([64, 64]);
      const y = x.matmul(x).relu().add(1);
      return y.sum();
    });
    assert.equal(kept.isDisposed, false);
    kept.dispose();
    assert.equal(kept.isDisposed, true);
    assert.throws(() => kept.item(), /disposed/i);

    // Every tensor created in tidy() was deterministically disposed
    const after = torch.memoryStats();
    assert.equal(after.liveTensors, before.liveTensors);
    assert.equal(after.reportedExternalBytes, before.reportedExternalBytes);
  });
});
