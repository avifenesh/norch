import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import torch, { AutogradFunction, type AutogradContext, type Tensor } from '../src/index.js';

describe('Autograd engine, guards, higher-order grads & custom JS AutogradFunction', () => {
  it('computes exact gradients, retainGrad, and higher-order derivatives (createGraph)', () => {
    torch.tidy(() => {
      // f(x) = x^3 + 2x^2 - 5x + 1 -> f'(x) = 3x^2 + 4x - 5 -> f''(x) = 6x + 4
      const x = torch.tensor(3.0, { dtype: torch.float64, requiresGrad: true });
      const y = x.pow(3).add(x.pow(2).mul(2)).sub(x.mul(5)).add(1);
      const [dyDx] = torch.grad(y, x, undefined, { createGraph: true });
      assert.ok(dyDx !== null);
      assert.ok(Math.abs(dyDx!.item() - (3 * 9 + 4 * 3 - 5)) < 1e-9);

      const [d2yDx2] = torch.grad(dyDx!, x);
      assert.ok(d2yDx2 !== null);
      assert.ok(Math.abs(d2yDx2!.item() - (6 * 3 + 4)) < 1e-9);

      // retainGrad on intermediate non-leaf tensor
      const a = torch.tensor([2.0, 3.0], { requiresGrad: true });
      const b = a.mul(4.0);
      b.retainGrad();
      const c = b.pow(2).sum();
      c.backward();
      assert.deepEqual(b.grad?.toArray(), [16, 24]);
      assert.deepEqual(a.grad?.toArray(), [64, 96]);
    });
  });

  it('respects noGrad, enableGrad, inferenceMode, and Symbol.dispose scopes', () => {
    torch.tidy(() => {
      const x = torch.tensor([1.0, 2.0], { requiresGrad: true });
      assert.equal(torch.isGradEnabled(), true);

      const yNoGrad = torch.noGrad(() => {
        assert.equal(torch.isGradEnabled(), false);
        return x.mul(3);
      });
      assert.equal(yNoGrad.requiresGrad, false);
      assert.equal(torch.isGradEnabled(), true);

      {
        using _ng = torch.noGradScope();
        assert.equal(torch.isGradEnabled(), false);
        const z = x.add(1);
        assert.equal(z.requiresGrad, false);
      }
      assert.equal(torch.isGradEnabled(), true);

      const yInf = torch.inferenceMode(() => {
        assert.equal(torch.isInferenceModeEnabled(), true);
        return x.add(2);
      });
      assert.equal(yInf.requiresGrad, false);
      assert.equal(torch.isInferenceModeEnabled(), false);
    });
  });

  it('supports CPU bfloat16 autocast', () => {
    torch.tidy(() => {
      const a = torch.randn([4, 8], { dtype: torch.float32 });
      const b = torch.randn([8, 4], { dtype: torch.float32 });
      const out = torch.autocast('cpu', torch.bfloat16, () => {
        assert.equal(torch.isAutocastEnabled('cpu'), true);
        return a.matmul(b);
      });
      assert.equal(out.dtype, 'bfloat16');
      assert.equal(torch.isAutocastEnabled('cpu'), false);
    });
  });

  it('supports custom JS AutogradFunction with saveForBackward and libtorch graph composition', () => {
    // Custom differentiable op: f(x, y, power) = (x^power) * y
    class PowerMul extends AutogradFunction {
      static forward(ctx: AutogradContext, x: Tensor, y: Tensor, power: number): Tensor {
        ctx.saveForBackward(x, y);
        ctx['power'] = power;
        return x.pow(power).mul(y);
      }

      static backward(ctx: AutogradContext, gradOut: Tensor): (Tensor | null)[] {
        const [x, y] = ctx.savedTensors;
        const p = ctx['power'] as number;
        const dx = gradOut.mul(y!).mul(p).mul(x!.pow(p - 1));
        const dy = gradOut.mul(x!.pow(p));
        return [dx, dy, null];
      }
    }

    torch.tidy(() => {
      const x = torch.tensor([2.0, 3.0], { dtype: torch.float64, requiresGrad: true });
      const y = torch.tensor([5.0, 7.0], { dtype: torch.float64, requiresGrad: true });

      const out = PowerMul.apply(x, y, 3) as Tensor;
      assert.equal(out.gradFnName, 'PowerMulBackward');
      // Compose with native ATen ops before backward
      const loss = out.mul(2).sum();
      loss.backward();

      // loss = 2 * sum(x^3 * y)
      // dloss/dx = 6 * x^2 * y = [6*4*5=120, 6*9*7=378]
      // dloss/dy = 2 * x^3     = [16, 54]
      const gx = x.grad!.toArray() as number[];
      const gy = y.grad!.toArray() as number[];
      assert.ok(Math.abs(gx[0]! - 120) < 1e-8);
      assert.ok(Math.abs(gx[1]! - 378) < 1e-8);
      assert.ok(Math.abs(gy[0]! - 16) < 1e-8);
      assert.ok(Math.abs(gy[1]! - 54) < 1e-8);
    });
  });

  it('supports gradient checkpointing (utils.checkpoint) matching direct backward', () => {
    torch.tidy(() => {
      const fn = (a: Tensor, w1: Tensor, w2: Tensor): Tensor =>
        a.matmul(w1).tanh().matmul(w2).sum();

      const x1 = torch.randn([4, 6], { dtype: torch.float64, requiresGrad: true });
      const w1a = torch.randn([6, 8], { dtype: torch.float64, requiresGrad: true });
      const w2a = torch.randn([8, 3], { dtype: torch.float64, requiresGrad: true });

      const x2 = x1.detach().clone().requiresGrad_(true);
      const w1b = w1a.detach().clone().requiresGrad_(true);
      const w2b = w2a.detach().clone().requiresGrad_(true);

      const lossDirect = fn(x1, w1a, w2a);
      lossDirect.backward();

      const lossCkpt = torch.utils.checkpoint(fn, x2, w1b, w2b);
      lossCkpt.backward();

      assert.ok(Math.abs(lossDirect.item() - lossCkpt.item()) < 1e-10);
      assert.ok(x1.grad!.allclose(x2.grad!, 1e-8, 1e-8));
      assert.ok(w1a.grad!.allclose(w1b.grad!, 1e-8, 1e-8));
      assert.ok(w2a.grad!.allclose(w2b.grad!, 1e-8, 1e-8));
    });
  });

  it('supports amp.GradScaler with scaling, unscaling, and inf overflow backoff', () => {
    using p = new torch.nn.Parameter(torch.tensor([2.0, -3.0]));
    using opt = new torch.optim.SGD([p], { lr: 0.1 });
    const scaler = new torch.amp.GradScaler({ initScale: 1024, growthInterval: 2 });

    // Normal scaled step matches unscaled gradient
    torch.tidy(() => {
      opt.zeroGrad();
      const loss = p.pow(2).sum(); // grad = [4, -6]
      scaler.scale(loss).backward();
      const stepped = scaler.step(opt);
      assert.equal(stepped, true);
      scaler.update();
    });
    // p should become [2 - 0.4, -3 + 0.6] = [1.6, -2.4]
    const afterNormal = p.toArray() as number[];
    assert.ok(Math.abs(afterNormal[0]! - 1.6) < 1e-5);
    assert.ok(Math.abs(afterNormal[1]! - -2.4) < 1e-5);

    // Overflowing inf grad skips optimizer step and halves scale
    torch.tidy(() => {
      opt.zeroGrad();
      p.grad = torch.tensor([Infinity, 1.0]);
      const stepped = scaler.step(opt);
      assert.equal(stepped, false);
      scaler.update();
    });
    assert.equal(scaler.getScale(), 512);
    const afterSkip = p.toArray() as number[];
    assert.ok(Math.abs(afterSkip[0]! - 1.6) < 1e-5);
  });
});
