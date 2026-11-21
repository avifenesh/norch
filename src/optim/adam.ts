import { noGrad } from '../autograd.js';
import { tidy } from '../memory.js';
import { zerosLike } from '../ops.js';
import type { Parameter } from '../nn/parameter.js';
import type { Tensor } from '../tensor.js';
import { Optimizer, type ParamGroupBase, type ParamGroupInput } from './optimizer.js';

export interface AdamOptions {
  lr?: number;
  betas?: [number, number];
  eps?: number;
  weightDecay?: number;
  amsgrad?: boolean;
}

export interface AdamParamGroup extends ParamGroupBase {
  betas: [number, number];
  eps: number;
  weightDecay: number;
  amsgrad: boolean;
}

function runAdamStep(
  paramGroups: AdamParamGroup[],
  getState: (p: Parameter) => Record<string, Tensor | number>,
  keepState: (t: Tensor) => Tensor,
  decoupledWeightDecay: boolean
): void {
  noGrad(() => {
    tidy(() => {
      for (const group of paramGroups) {
        const { lr, betas, eps, weightDecay, amsgrad } = group;
        const [beta1, beta2] = betas;

        for (const p of group.params) {
          const rawGrad = p.grad;
          if (!rawGrad) continue;

          const st = getState(p);
          let stepCount = (st['step'] as number | undefined) ?? 0;
          let expAvg = st['expAvg'] as Tensor | undefined;
          let expAvgSq = st['expAvgSq'] as Tensor | undefined;
          let maxExpAvgSq = st['maxExpAvgSq'] as Tensor | undefined;

          if (!expAvg || !expAvgSq) {
            expAvg = keepState(zerosLike(p));
            expAvgSq = keepState(zerosLike(p));
            st['expAvg'] = expAvg;
            st['expAvgSq'] = expAvgSq;
            if (amsgrad) {
              maxExpAvgSq = keepState(zerosLike(p));
              st['maxExpAvgSq'] = maxExpAvgSq;
            }
          }

          stepCount += 1;
          st['step'] = stepCount;

          let g = rawGrad;
          if (decoupledWeightDecay) {
            if (weightDecay !== 0) {
              p.mul_(1 - lr * weightDecay);
            }
          } else if (weightDecay !== 0) {
            g = g.add(p, weightDecay);
          }

          expAvg.mul_(beta1).add_(g, 1 - beta1);
          expAvgSq.mul_(beta2).addcmul_(g, g, 1 - beta2);

          const biasCorrection1 = 1 - Math.pow(beta1, stepCount);
          const biasCorrection2 = 1 - Math.pow(beta2, stepCount);
          const stepSize = lr / biasCorrection1;
          const biasCorrection2Sqrt = Math.sqrt(biasCorrection2);

          let denom: Tensor;
          if (amsgrad && maxExpAvgSq) {
            // elementwise max: maxExpAvgSq = max(maxExpAvgSq, expAvgSq)
            const nextMax = maxExpAvgSq.add(expAvgSq.sub(maxExpAvgSq).relu());
            maxExpAvgSq.copy_(nextMax);
            denom = maxExpAvgSq.sqrt().div(biasCorrection2Sqrt).add(eps);
          } else {
            denom = expAvgSq.sqrt().div(biasCorrection2Sqrt).add(eps);
          }
          p.addcdiv_(expAvg, denom, -stepSize);
        }
      }
    });
  });
}

export class Adam extends Optimizer<AdamParamGroup> {
  constructor(params: ParamGroupInput<AdamParamGroup>, opts: AdamOptions = {}) {
    const lr = opts.lr ?? 1e-3;
    const betas: [number, number] = opts.betas ?? [0.9, 0.999];
    const eps = opts.eps ?? 1e-8;
    const weightDecay = opts.weightDecay ?? 0;
    const amsgrad = opts.amsgrad ?? false;
    if (lr < 0) throw new RangeError(`Invalid learning rate: ${lr}`);
    super(params, { lr, betas, eps, weightDecay, amsgrad });
  }

  step(closure?: () => Tensor): Tensor | undefined {
    const loss = closure ? closure() : undefined;
    runAdamStep(
      this.paramGroups,
      (p) => this.getState(p),
      (t) => this.keepStateTensor(t),
      false
    );
    return loss;
  }
}

export class AdamW extends Optimizer<AdamParamGroup> {
  constructor(params: ParamGroupInput<AdamParamGroup>, opts: AdamOptions = {}) {
    const lr = opts.lr ?? 1e-3;
    const betas: [number, number] = opts.betas ?? [0.9, 0.999];
    const eps = opts.eps ?? 1e-8;
    const weightDecay = opts.weightDecay ?? 1e-2;
    const amsgrad = opts.amsgrad ?? false;
    if (lr < 0) throw new RangeError(`Invalid learning rate: ${lr}`);
    super(params, { lr, betas, eps, weightDecay, amsgrad });
  }

  step(closure?: () => Tensor): Tensor | undefined {
    const loss = closure ? closure() : undefined;
    runAdamStep(
      this.paramGroups,
      (p) => this.getState(p),
      (t) => this.keepStateTensor(t),
      true
    );
    return loss;
  }
}
