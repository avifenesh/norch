import { noGrad } from '../autograd.js';
import { tidy } from '../memory.js';
import type { Parameter } from '../nn/parameter.js';
import type { Tensor } from '../tensor.js';
import { Optimizer, type ParamGroupBase, type ParamGroupInput } from './optimizer.js';

export interface SGDOptions {
  lr: number;
  momentum?: number;
  dampening?: number;
  weightDecay?: number;
  nesterov?: boolean;
}

export interface SGDParamGroup extends ParamGroupBase {
  momentum: number;
  dampening: number;
  weightDecay: number;
  nesterov: boolean;
}

export class SGD extends Optimizer<SGDParamGroup> {
  constructor(params: ParamGroupInput<SGDParamGroup>, opts: SGDOptions) {
    if (opts.lr < 0) throw new RangeError(`Invalid learning rate: ${opts.lr}`);
    const momentum = opts.momentum ?? 0;
    const dampening = opts.dampening ?? 0;
    const weightDecay = opts.weightDecay ?? 0;
    const nesterov = opts.nesterov ?? false;
    if (nesterov && (momentum <= 0 || dampening !== 0)) {
      throw new Error('Nesterov momentum requires a momentum > 0 and zero dampening');
    }
    super(params, { lr: opts.lr, momentum, dampening, weightDecay, nesterov });
  }

  step(closure?: () => Tensor): Tensor | undefined {
    let loss: Tensor | undefined;
    if (closure) {
      loss = closure();
    }
    noGrad(() => {
      tidy(() => {
        for (const group of this.paramGroups) {
          const { lr, momentum, dampening, weightDecay, nesterov } = group;
          for (const p of group.params) {
            const rawGrad = p.grad;
            if (!rawGrad) continue;
            let dP = rawGrad;
            if (weightDecay !== 0) {
              dP = dP.add(p, weightDecay);
            }
            if (momentum !== 0) {
              const st = this.getState(p);
              let buf = st['momentumBuffer'] as Tensor | undefined;
              if (!buf) {
                buf = this.keepStateTensor(dP.detach().clone());
                st['momentumBuffer'] = buf;
              } else {
                buf.mul_(momentum).add_(dP, 1 - dampening);
              }
              if (nesterov) {
                dP = dP.add(buf, momentum);
              } else {
                dP = buf;
              }
            }
            p.add_(dP, -lr);
          }
        }
      });
    });
    return loss;
  }
}
