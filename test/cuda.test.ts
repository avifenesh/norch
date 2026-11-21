import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import torch, { nn, optim } from '../src/index.js';

const runCuda = process.env.NORCH_TEST_CUDA === '1';

describe('CUDA backend (guarded by NORCH_TEST_CUDA=1)', { skip: !runCuda }, () => {
  it('runs GPU tensor ops, bfloat16 autocast, and a training step on CUDA', () => {
    assert.equal(torch.cuda.isAvailable(), true);
    assert.ok(torch.cuda.deviceCount() >= 1);
    assert.ok(torch.cuda.deviceName(0).length > 0);

    using model = new nn.Sequential(new nn.Linear(16, 32), new nn.GELU(), new nn.Linear(32, 4));
    model.to('cuda');
    using opt = new optim.AdamW(model.parameters(), { lr: 1e-2 });

    const lossVal = torch.tidy(() => {
      const x = torch.randn([8, 16], { device: 'cuda' });
      const y = torch.randn([8, 4], { device: 'cuda' });
      const out = torch.autocast('cuda', torch.bfloat16, () => model.forward(x));
      assert.equal(out.isCuda, true);
      const loss = torch.F.mseLoss(out.float(), y);
      loss.backward();
      opt.step();
      torch.cuda.synchronize();
      return loss.item();
    });

    assert.ok(Number.isFinite(lossVal));
    torch.cuda.emptyCache();
    const stats = torch.cuda.memoryStats();
    assert.ok(stats.maxAllocated > 0);
  });
});
