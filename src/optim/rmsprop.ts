import { noGrad } from '../autograd.js';
import { tidy } from '../memory.js';
import { zerosLike } from '../ops.js';
import type { Tensor } from '../tensor.js';
import { Optimizer, type ParamGroupBase, type ParamGroupInput } from './optimizer.js';

export interface RMSpropOptions {
  lr?: number;
  alpha?: number;
  eps?: number;
  weightDecay?: number;
  momentum?: number;
  centered?: boolean;
}

export interface RMSpropParamGroup extends ParamGroupBase {
  alpha: number;
  eps: number;
  weightDecay: number;
  momentum: number;
  centered: boolean;
}

export class RMSprop extends Optimizer<RMSpropParamGroup> {
  constructor(params: ParamGroupInput<RMSpropParamGroup>, opts: RMSpropOptions = {}) {
    const lr = opts.lr ?? 1e-2;
    const alpha = opts.alpha ?? 0.99;
    const eps = opts.eps ?? 1e-8;
    const weightDecay = opts.weightDecay ?? 0;
    const momentum = opts.momentum ?? 0;
    const centered = opts.centered ?? false;
    if (lr < 0) throw new RangeError(`Invalid learning rate: ${lr}`);
    super(params, { lr, alpha, eps, weightDecay, momentum, centered });
  }

  step(closure?: () => Tensor): Tensor | undefined {
    const loss = closure ? closure() : undefined;
    noGrad(() => {
      tidy(() => {
        for (const group of this.paramGroups) {
          const { lr, alpha, eps, weightDecay, momentum, centered } = group;
          for (const p of group.params) {
            const rawGrad = p.grad;
            if (!rawGrad) continue;
            let g = rawGrad;
            if (weightDecay !== 0) {
              g = g.add(p, weightDecay);
            }

            const st = this.getState(p);
            let squareAvg = st['squareAvg'] as Tensor | undefined;
            if (!squareAvg) {
              squareAvg = this.keepStateTensor(zerosLike(p));
              st['squareAvg'] = squareAvg;
            }
            squareAvg.mul_(alpha).addcmul_(g, g, 1 - alpha);

            let avg: Tensor;
            if (centered) {
              let gradAvg = st['gradAvg'] as Tensor | undefined;
              if (!gradAvg) {
                gradAvg = this.keepStateTensor(zerosLike(p));
                st['gradAvg'] = gradAvg;
              }
              gradAvg.mul_(alpha).add_(g, 1 - alpha);
              avg = squareAvg.addcmul_(gradAvg, gradAvg, -1).sqrt().add(eps);
            } else {
              avg = squareAvg.sqrt().add(eps);
            }

            if (momentum > 0) {
              let buf = st['momentumBuffer'] as Tensor | undefined;
              if (!buf) {
                buf = this.keepStateTensor(zerosLike(p));
                st['momentumBuffer'] = buf;
              }
              buf.mul_(momentum).addcdiv_(g, avg, 1);
              p.add_(buf, -lr);
            } else {
              p.addcdiv_(g, avg, -lr);
            }
          }
        }
      });
    });
    return loss;
  }
}
