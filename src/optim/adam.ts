import { noGrad } from '../autograd.js';
import { tidy } from '../memory.js';
import { native, type NativeTensor } from '../native.js';
import { tensor, zerosLike } from '../ops.js';
import type { Parameter } from '../nn/parameter.js';
import type { Tensor } from '../tensor.js';
import { Optimizer, type ParamGroupBase, type ParamGroupInput } from './optimizer.js';

export interface AdamOptions {
  lr?: number;
  betas?: [number, number];
  eps?: number;
  weightDecay?: number;
  amsgrad?: boolean;
  /**
   * Use libtorch's multi-tensor fused kernel (`aten::_fused_adamw_` / `aten::_fused_adam_`).
   * Defaults to true when all tensors in a group are eligible floating-point CPU/CUDA tensors.
   */
  fused?: boolean;
}

export interface AdamParamGroup extends ParamGroupBase {
  betas: [number, number];
  eps: number;
  weightDecay: number;
  amsgrad: boolean;
  fused: boolean;
}

function canUseFusedKernel(params: Parameter[]): boolean {
  if (params.length === 0) return false;
  const devType = params[0]!.device.type;
  if (devType !== 'cpu' && devType !== 'cuda') return false;
  const dtype = params[0]!.dtype;
  if (dtype !== 'float32' && dtype !== 'float64' && dtype !== 'float16' && dtype !== 'bfloat16') {
    return false;
  }
  for (const p of params) {
    if (p.device.type !== devType || p.dtype !== dtype) return false;
  }
  return true;
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
        const { lr, betas, eps, weightDecay, amsgrad, fused } = group;
        const [beta1, beta2] = betas;

        const activeParams: Parameter[] = [];
        const activeGrads: Tensor[] = [];
        for (const p of group.params) {
          const g = p.grad;
          if (!g) continue;
          activeParams.push(p);
          activeGrads.push(g);
        }
        if (activeParams.length === 0) continue;

        if (fused && canUseFusedKernel(activeParams)) {
          const nativeParams: NativeTensor[] = [];
          const nativeGrads: NativeTensor[] = [];
          const expAvgs: NativeTensor[] = [];
          const expAvgSqs: NativeTensor[] = [];
          const maxExpAvgSqs: NativeTensor[] = [];
          const stateSteps: NativeTensor[] = [];

          for (let i = 0; i < activeParams.length; i++) {
            const p = activeParams[i]!;
            const g = activeGrads[i]!;
            const st = getState(p);
            let expAvg = st['expAvg'] as Tensor | undefined;
            let expAvgSq = st['expAvgSq'] as Tensor | undefined;
            let maxExpAvgSq = st['maxExpAvgSq'] as Tensor | undefined;
            let stepTensor = st['stepTensor'] as Tensor | undefined;

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
            if (!stepTensor) {
              const initStep = (st['step'] as number | undefined) ?? 0;
              stepTensor = keepState(tensor(initStep, { dtype: 'float32', device: p.device }));
              st['stepTensor'] = stepTensor;
            }
            stepTensor.add_(1);
            st['step'] = ((st['step'] as number | undefined) ?? 0) + 1;

            nativeParams.push(p._native);
            nativeGrads.push(g._native);
            expAvgs.push(expAvg._native);
            expAvgSqs.push(expAvgSq._native);
            if (amsgrad && maxExpAvgSq) {
              maxExpAvgSqs.push(maxExpAvgSq._native);
            }
            stateSteps.push(stepTensor._native);
          }

          native.fast.fusedAdamStep(
            nativeParams,
            nativeGrads,
            expAvgs,
            expAvgSqs,
            maxExpAvgSqs,
            stateSteps,
            lr,
            beta1,
            beta2,
            weightDecay,
            eps,
            amsgrad,
            decoupledWeightDecay
          );
          continue;
        }

        // Unfused fallback
        for (let i = 0; i < activeParams.length; i++) {
          const p = activeParams[i]!;
          const rawGrad = activeGrads[i]!;
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
    const fused = opts.fused ?? true;
    if (lr < 0) throw new RangeError(`Invalid learning rate: ${lr}`);
    super(params, { lr, betas, eps, weightDecay, amsgrad, fused });
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
    const fused = opts.fused ?? true;
    if (lr < 0) throw new RangeError(`Invalid learning rate: ${lr}`);
    super(params, { lr, betas, eps, weightDecay, amsgrad, fused });
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
