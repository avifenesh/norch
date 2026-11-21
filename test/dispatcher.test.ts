import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import torch from '../src/index.js';

describe('c10::Dispatcher generic bridge & codegen', () => {
  it('exposes thousands of registered ATen operators and schemas from libtorch', () => {
    const allAten = torch.ops.listOps('aten::');
    assert.ok(allAten.length > 2000, `Expected >2000 aten ops, got ${allAten.length}`);
    assert.ok(torch.ops.ATEN_OP_NAMES.length > 500);

    const schema = torch.ops.getOpSchema('aten::add', 'Tensor');
    assert.ok(schema !== null);
    assert.equal(schema.name, 'aten::add');
    assert.equal(schema.overload, 'Tensor');
    assert.equal(schema.arguments[0]?.name, 'self');
    assert.equal(schema.arguments[1]?.name, 'other');
  });

  it('calls un-hand-wrapped ATen operators via ops.aten.* with full autograd', () => {
    torch.tidy(() => {
      const x = torch.tensor([1, 2, 3, 4], { dtype: torch.float32, requiresGrad: true });

      // cumsum
      const cs = torch.ops.aten.cumsum(x, 0) as torch.Tensor;
      assert.deepEqual(cs.toArray(), [1, 3, 6, 10]);

      // flip
      const fl = torch.ops.aten.flip(x, [0]) as torch.Tensor;
      assert.deepEqual(fl.toArray(), [4, 3, 2, 1]);

      // roll
      const rl = torch.ops.aten.roll(x, [1], [0]) as torch.Tensor;
      assert.deepEqual(rl.toArray(), [4, 1, 2, 3]);

      // Backward through generic dispatched ops works natively in libtorch C++ autograd!
      const loss = cs.sum();
      loss.backward();
      // d(sum(cumsum(x)))/dx = [4, 3, 2, 1]
      assert.deepEqual(x.grad?.toArray(), [4, 3, 2, 1]);

      // Multi-return op: linalg_qr returns [Q, R]
      const m = torch.tensor(
        [
          [12, -51, 4],
          [6, 167, -68],
          [-4, 24, -41],
        ],
        { dtype: torch.float64 }
      );
      const [q, r] = torch.ops.aten.linalg_qr(m) as [torch.Tensor, torch.Tensor];
      assert.ok(q.matmul(r).allclose(m, 1e-6, 1e-6));

      // kwargs dispatch
      const sorted = torch.ops.aten.sort(torch.tensor([3, 1, 4, 2]), {
        dim: 0,
        descending: true,
      }) as [torch.Tensor, torch.Tensor];
      assert.deepEqual(sorted[0].toArray(), [4, 3, 2, 1]);
    });
  });

  it('throws clean JS errors on missing op or invalid arguments', () => {
    assert.throws(() => torch.ops.callOp('aten::definitely_not_a_real_op', '', []), /not found/i);
    assert.throws(
      () => torch.ops.callOp('aten::add', 'Tensor', [torch.ones([2])], { bogusKwarg: 1 }),
      /Unknown keyword argument/i
    );
  });
});
