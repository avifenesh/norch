import { noGrad } from '../autograd.js';
import { tidy } from '../memory.js';
import { stack } from '../ops.js';
import type { Tensor } from '../tensor.js';

export function clipGradNorm_(
  parameters: Iterable<Tensor>,
  maxNorm: number,
  normType = 2.0,
  errorIfNonfinite = false
): number {
  return noGrad(() =>
    tidy(() => {
      const grads: Tensor[] = [];
      for (const p of parameters) {
        const g = p.grad;
        if (g) grads.push(g);
      }
      if (grads.length === 0) return 0;

      let totalNorm: number;
      if (normType === Infinity) {
        const norms = grads.map((g) => g.detach().abs().max());
        totalNorm = stack(norms).max().item();
      } else {
        const norms = grads.map((g) => g.detach().norm(normType));
        totalNorm = stack(norms).norm(normType).item();
      }

      if (errorIfNonfinite && (!Number.isFinite(totalNorm) || Number.isNaN(totalNorm))) {
        throw new Error(`The total norm of order ${normType} for gradients is non-finite (${totalNorm})`);
      }

      const clipCoef = maxNorm / (totalNorm + 1e-6);
      if (clipCoef < 1.0) {
        for (const g of grads) {
          g.mul_(clipCoef);
        }
      }
      return totalNorm;
    })
  );
}

export function clipGradValue_(parameters: Iterable<Tensor>, clipValue: number): void {
  const limit = Math.abs(clipValue);
  noGrad(() => {
    for (const p of parameters) {
      const g = p.grad;
      if (g) {
        g.clamp_(-limit, limit);
      }
    }
  });
}
