import { noGrad } from '../autograd.js';
import { untrackTensor } from '../memory.js';
import type { Parameter } from '../nn/parameter.js';
import { fullLike, zerosLike } from '../ops.js';
import type { Tensor } from '../tensor.js';
import { Optimizer, type ParamGroupBase, type ParamGroupInput } from './optimizer.js';

// ---------- Adagrad ----------

export interface AdagradOptions {
  lr?: number;
  lrDecay?: number;
  weightDecay?: number;
  initialAccumulatorValue?: number;
  eps?: number;
}

export interface AdagradParamGroup extends ParamGroupBase {
  lrDecay: number;
  weightDecay: number;
  initialAccumulatorValue: number;
  eps: number;
}

export class Adagrad extends Optimizer<AdagradParamGroup> {
  constructor(params: ParamGroupInput<AdagradParamGroup>, opts: AdagradOptions = {}) {
    super(params, {
      lr: opts.lr ?? 1e-2,
      lrDecay: opts.lrDecay ?? 0,
      weightDecay: opts.weightDecay ?? 0,
      initialAccumulatorValue: opts.initialAccumulatorValue ?? 0,
      eps: opts.eps ?? 1e-10,
    });
  }

  step(closure?: () => Tensor): Tensor | undefined {
    const loss = closure ? closure() : undefined;
    noGrad(() => {
      for (const group of this.paramGroups) {
        for (const p of group.params) {
          const g = p.grad;
          if (!g) continue;
          const st = this.getState(p);
          const step = ((st['step'] as number | undefined) ?? 0) + 1;
          st['step'] = step;

          let sumSq = st['sum'] as Tensor | undefined;
          if (!sumSq) {
            sumSq = fullLike(p, group.initialAccumulatorValue);
            untrackTensor(sumSq);
            st['sum'] = sumSq;
          }

          let grad = g;
          let ownedGrad: Tensor | null = null;
          if (group.weightDecay !== 0) {
            ownedGrad = g.add(p, group.weightDecay);
            grad = ownedGrad;
          }

          const clr = group.lr / (1 + (step - 1) * group.lrDecay);
          sumSq.addcmul_(grad, grad, 1);
          const std = sumSq.sqrt().add(group.eps);
          p.addcdiv_(grad, std, -clr);
          std.dispose();
          if (ownedGrad) ownedGrad.dispose();
        }
      }
    });
    return loss;
  }
}

// ---------- Adadelta ----------

export interface AdadeltaOptions {
  lr?: number;
  rho?: number;
  eps?: number;
  weightDecay?: number;
}

export interface AdadeltaParamGroup extends ParamGroupBase {
  rho: number;
  eps: number;
  weightDecay: number;
}

export class Adadelta extends Optimizer<AdadeltaParamGroup> {
  constructor(params: ParamGroupInput<AdadeltaParamGroup>, opts: AdadeltaOptions = {}) {
    super(params, {
      lr: opts.lr ?? 1.0,
      rho: opts.rho ?? 0.9,
      eps: opts.eps ?? 1e-6,
      weightDecay: opts.weightDecay ?? 0,
    });
  }

  step(closure?: () => Tensor): Tensor | undefined {
    const loss = closure ? closure() : undefined;
    noGrad(() => {
      for (const group of this.paramGroups) {
        const { lr, rho, eps, weightDecay } = group;
        for (const p of group.params) {
          const g = p.grad;
          if (!g) continue;
          const st = this.getState(p);
          let squareAvg = st['squareAvg'] as Tensor | undefined;
          let accDelta = st['accDelta'] as Tensor | undefined;
          if (!squareAvg || !accDelta) {
            squareAvg = zerosLike(p);
            accDelta = zerosLike(p);
            untrackTensor(squareAvg);
            untrackTensor(accDelta);
            st['squareAvg'] = squareAvg;
            st['accDelta'] = accDelta;
          }

          let grad = g;
          let ownedGrad: Tensor | null = null;
          if (weightDecay !== 0) {
            ownedGrad = g.add(p, weightDecay);
            grad = ownedGrad;
          }

          squareAvg.mul_(rho).addcmul_(grad, grad, 1 - rho);
          const std = squareAvg.add(eps).sqrt();
          const delta = accDelta.add(eps).sqrt().div(std).mul(grad);
          p.add_(delta, -lr);
          accDelta.mul_(rho).addcmul_(delta, delta, 1 - rho);
          std.dispose();
          delta.dispose();
          if (ownedGrad) ownedGrad.dispose();
        }
      }
    });
    return loss;
  }
}

// ---------- Adamax ----------

export interface AdamaxOptions {
  lr?: number;
  betas?: [number, number];
  eps?: number;
  weightDecay?: number;
}

export interface AdamaxParamGroup extends ParamGroupBase {
  betas: [number, number];
  eps: number;
  weightDecay: number;
}

export class Adamax extends Optimizer<AdamaxParamGroup> {
  constructor(params: ParamGroupInput<AdamaxParamGroup>, opts: AdamaxOptions = {}) {
    super(params, {
      lr: opts.lr ?? 2e-3,
      betas: opts.betas ?? [0.9, 0.999],
      eps: opts.eps ?? 1e-8,
      weightDecay: opts.weightDecay ?? 0,
    });
  }

  step(closure?: () => Tensor): Tensor | undefined {
    const loss = closure ? closure() : undefined;
    noGrad(() => {
      for (const group of this.paramGroups) {
        const [beta1, beta2] = group.betas;
        for (const p of group.params) {
          const g = p.grad;
          if (!g) continue;
          const st = this.getState(p);
          const step = ((st['step'] as number | undefined) ?? 0) + 1;
          st['step'] = step;

          let expAvg = st['expAvg'] as Tensor | undefined;
          let expInf = st['expInf'] as Tensor | undefined;
          if (!expAvg || !expInf) {
            expAvg = zerosLike(p);
            expInf = zerosLike(p);
            untrackTensor(expAvg);
            untrackTensor(expInf);
            st['expAvg'] = expAvg;
            st['expInf'] = expInf;
          }

          let grad = g;
          let ownedGrad: Tensor | null = null;
          if (group.weightDecay !== 0) {
            ownedGrad = g.add(p, group.weightDecay);
            grad = ownedGrad;
          }

          expAvg.mul_(beta1).add_(grad, 1 - beta1);
          const scaledInf = expInf.mul(beta2);
          const absGrad = grad.abs().add(group.eps);
          const newInf = scaledInf.ge(absGrad).to(p.dtype);
          const nextInf = scaledInf.mul(newInf).add(absGrad.mul(newInf.neg().add(1)));
          expInf.copy_(nextInf);
          scaledInf.dispose();
          absGrad.dispose();
          newInf.dispose();
          nextInf.dispose();

          const biasCorr = 1 - Math.pow(beta1, step);
          const stepSize = group.lr / biasCorr;
          p.addcdiv_(expAvg, expInf, -stepSize);
          if (ownedGrad) ownedGrad.dispose();
        }
      }
    });
    return loss;
  }
}

// ---------- NAdam ----------

export interface NAdamOptions {
  lr?: number;
  betas?: [number, number];
  eps?: number;
  weightDecay?: number;
  momentumDecay?: number;
}

export interface NAdamParamGroup extends ParamGroupBase {
  betas: [number, number];
  eps: number;
  weightDecay: number;
  momentumDecay: number;
}

export class NAdam extends Optimizer<NAdamParamGroup> {
  constructor(params: ParamGroupInput<NAdamParamGroup>, opts: NAdamOptions = {}) {
    super(params, {
      lr: opts.lr ?? 2e-3,
      betas: opts.betas ?? [0.9, 0.999],
      eps: opts.eps ?? 1e-8,
      weightDecay: opts.weightDecay ?? 0,
      momentumDecay: opts.momentumDecay ?? 4e-3,
    });
  }

  step(closure?: () => Tensor): Tensor | undefined {
    const loss = closure ? closure() : undefined;
    noGrad(() => {
      for (const group of this.paramGroups) {
        const [beta1, beta2] = group.betas;
        const psi = group.momentumDecay;
        for (const p of group.params) {
          const g = p.grad;
          if (!g) continue;
          const st = this.getState(p);
          const step = ((st['step'] as number | undefined) ?? 0) + 1;
          st['step'] = step;

          let expAvg = st['expAvg'] as Tensor | undefined;
          let expAvgSq = st['expAvgSq'] as Tensor | undefined;
          if (!expAvg || !expAvgSq) {
            expAvg = zerosLike(p);
            expAvgSq = zerosLike(p);
            untrackTensor(expAvg);
            untrackTensor(expAvgSq);
            st['expAvg'] = expAvg;
            st['expAvgSq'] = expAvgSq;
            st['muProduct'] = 1.0;
          }

          let grad = g;
          let ownedGrad: Tensor | null = null;
          if (group.weightDecay !== 0) {
            ownedGrad = g.add(p, group.weightDecay);
            grad = ownedGrad;
          }

          const mu = beta1 * (1 - 0.5 * Math.pow(0.96, step * psi));
          const muNext = beta1 * (1 - 0.5 * Math.pow(0.96, (step + 1) * psi));
          const muProduct = ((st['muProduct'] as number) ?? 1.0) * mu;
          st['muProduct'] = muProduct;
          const muProductNext = muProduct * muNext;

          expAvg.mul_(beta1).add_(grad, 1 - beta1);
          expAvgSq.mul_(beta2).addcmul_(grad, grad, 1 - beta2);

          const biasCorr2 = 1 - Math.pow(beta2, step);
          const denom = expAvgSq.div(biasCorr2).sqrt().add(group.eps);
          const mHat = grad
            .mul((1 - mu) / (1 - muProduct))
            .add(expAvg.mul(muNext / (1 - muProductNext)));
          p.addcdiv_(mHat, denom, -group.lr);

          denom.dispose();
          mHat.dispose();
          if (ownedGrad) ownedGrad.dispose();
        }
      }
    });
    return loss;
  }
}

// ---------- RAdam ----------

export interface RAdamOptions {
  lr?: number;
  betas?: [number, number];
  eps?: number;
  weightDecay?: number;
}

export interface RAdamParamGroup extends ParamGroupBase {
  betas: [number, number];
  eps: number;
  weightDecay: number;
}

export class RAdam extends Optimizer<RAdamParamGroup> {
  constructor(params: ParamGroupInput<RAdamParamGroup>, opts: RAdamOptions = {}) {
    super(params, {
      lr: opts.lr ?? 1e-3,
      betas: opts.betas ?? [0.9, 0.999],
      eps: opts.eps ?? 1e-8,
      weightDecay: opts.weightDecay ?? 0,
    });
  }

  step(closure?: () => Tensor): Tensor | undefined {
    const loss = closure ? closure() : undefined;
    noGrad(() => {
      for (const group of this.paramGroups) {
        const [beta1, beta2] = group.betas;
        const rhoInf = 2 / (1 - beta2) - 1;
        for (const p of group.params) {
          const g = p.grad;
          if (!g) continue;
          const st = this.getState(p);
          const step = ((st['step'] as number | undefined) ?? 0) + 1;
          st['step'] = step;

          let expAvg = st['expAvg'] as Tensor | undefined;
          let expAvgSq = st['expAvgSq'] as Tensor | undefined;
          if (!expAvg || !expAvgSq) {
            expAvg = zerosLike(p);
            expAvgSq = zerosLike(p);
            untrackTensor(expAvg);
            untrackTensor(expAvgSq);
            st['expAvg'] = expAvg;
            st['expAvgSq'] = expAvgSq;
          }

          let grad = g;
          let ownedGrad: Tensor | null = null;
          if (group.weightDecay !== 0) {
            ownedGrad = g.add(p, group.weightDecay);
            grad = ownedGrad;
          }

          expAvg.mul_(beta1).add_(grad, 1 - beta1);
          expAvgSq.mul_(beta2).addcmul_(grad, grad, 1 - beta2);

          const biasCorr1 = 1 - Math.pow(beta1, step);
          const beta2T = Math.pow(beta2, step);
          const rhoT = rhoInf - (2 * step * beta2T) / (1 - beta2T);

          if (rhoT > 5) {
            const rect = Math.sqrt(
              ((rhoT - 4) * (rhoT - 2) * rhoInf) / ((rhoInf - 4) * (rhoInf - 2) * rhoT)
            );
            const denom = expAvgSq.div(1 - beta2T).sqrt().add(group.eps);
            p.addcdiv_(expAvg, denom, (-group.lr * rect) / biasCorr1);
            denom.dispose();
          } else {
            p.add_(expAvg, -group.lr / biasCorr1);
          }
          if (ownedGrad) ownedGrad.dispose();
        }
      }
    });
    return loss;
  }
}

// ---------- L-BFGS ----------

export interface LBFGSOptions {
  lr?: number;
  maxIter?: number;
  historySize?: number;
  toleranceGrad?: number;
  toleranceChange?: number;
  lineSearchFn?: 'backtracking' | null;
}

export interface LBFGSParamGroup extends ParamGroupBase {
  maxIter: number;
  historySize: number;
  toleranceGrad: number;
  toleranceChange: number;
  lineSearchFn: 'backtracking' | null;
}

/**
 * Limited-memory BFGS quasi-Newton optimizer using two-loop recursion
 * and Armijo backtracking line search. Requires a `closure` returning the scalar loss.
 */
export class LBFGS extends Optimizer<LBFGSParamGroup> {
  private readonly sHistory: Tensor[] = [];
  private readonly yHistory: Tensor[] = [];
  private readonly rhoHistory: number[] = [];

  constructor(params: ParamGroupInput<LBFGSParamGroup>, opts: LBFGSOptions = {}) {
    super(params, {
      lr: opts.lr ?? 1.0,
      maxIter: opts.maxIter ?? 20,
      historySize: opts.historySize ?? 100,
      toleranceGrad: opts.toleranceGrad ?? 1e-7,
      toleranceChange: opts.toleranceChange ?? 1e-9,
      lineSearchFn: opts.lineSearchFn ?? 'backtracking',
    });
  }

  override dispose(): void {
    for (const s of this.sHistory) {
      if (!s.isDisposed) s.dispose();
    }
    for (const y of this.yHistory) {
      if (!y.isDisposed) y.dispose();
    }
    this.sHistory.length = 0;
    this.yHistory.length = 0;
    this.rhoHistory.length = 0;
    super.dispose();
  }

  step(closure?: () => Tensor): Tensor | undefined {
    if (!closure) {
      throw new Error('LBFGS.step(closure) requires a closure that evaluates loss and calls backward()');
    }
    const group = this.paramGroups[0]!;
    const params = group.params;
    const flattenGrad = (): Tensor => {
      const views = params.map((p) => (p.grad ? p.grad.detach().reshape(-1) : zerosLike(p).reshape(-1)));
      const flat = concat1d(views);
      for (const v of views) v.dispose();
      return flat;
    };
    const addFlatStep = (direction: Tensor, alpha: number): void => {
      noGrad(() => {
        let offset = 0;
        for (const p of params) {
          const n = p.numel;
          const slice = direction.narrow(0, offset, n).reshape(p.shape);
          p.add_(slice, alpha);
          slice.dispose();
          offset += n;
        }
      });
    };

    let currentLossT = closure();
    let lossVal = currentLossT.item();

    let flatGrad = flattenGrad();
    let prevLoss = lossVal;

    try {
      for (let iter = 0; iter < group.maxIter; iter++) {
        const gradInf = flatGrad.abs().max().item();
        if (gradInf <= group.toleranceGrad) break;

        // Two-loop L-BFGS recursion for search direction d = -H * g
        const d = noGrad(() => {
          const q = flatGrad.clone();
          const m = this.sHistory.length;
          const alphas = new Array<number>(m);
          for (let i = m - 1; i >= 0; i--) {
            const a = this.rhoHistory[i]! * this.sHistory[i]!.mul(q).sum().item();
            alphas[i] = a;
            q.add_(this.yHistory[i]!, -a);
          }
          let gamma = 1.0;
          if (m > 0) {
            const lastS = this.sHistory[m - 1]!;
            const lastY = this.yHistory[m - 1]!;
            const ys = lastS.mul(lastY).sum().item();
            const yy = lastY.mul(lastY).sum().item();
            if (yy > 1e-20) gamma = ys / yy;
          }
          const r = q.mul(gamma);
          q.dispose();
          for (let i = 0; i < m; i++) {
            const beta = this.rhoHistory[i]! * this.yHistory[i]!.mul(r).sum().item();
            r.add_(this.sHistory[i]!, alphas[i]! - beta);
          }
          const negR = r.neg();
          r.dispose();
          return negR;
        });

        let stepSize = group.lr;
        if (this.sHistory.length === 0) {
          const gNorm = flatGrad.norm(2).item();
          if (gNorm > 0) stepSize = Math.min(group.lr, 1.0 / gNorm);
        }

        const dirDeriv = noGrad(() => flatGrad.mul(d).sum().item());
        if (dirDeriv >= 0) {
          noGrad(() => {
            d.copy_(flatGrad).neg();
          });
        }

        let acceptedLoss = lossVal;
        if (group.lineSearchFn === 'backtracking') {
          let applied = 0;
          for (let ls = 0; ls < 12; ls++) {
            addFlatStep(d, stepSize - applied);
            applied = stepSize;
            const trialT = closure();
            acceptedLoss = trialT.item();
            currentLossT.dispose();
            currentLossT = trialT;
            if (acceptedLoss <= lossVal + 1e-4 * stepSize * dirDeriv) {
              break;
            }
            stepSize *= 0.5;
          }
        } else {
          addFlatStep(d, stepSize);
          const trialT = closure();
          acceptedLoss = trialT.item();
          currentLossT.dispose();
          currentLossT = trialT;
        }

        const nextGrad = flattenGrad();
        const s = noGrad(() => {
          const out = d.mul(stepSize);
          untrackTensor(out);
          return out;
        });
        const y = noGrad(() => {
          const out = nextGrad.sub(flatGrad);
          untrackTensor(out);
          return out;
        });
        const ys = noGrad(() => s.mul(y).sum().item());
        d.dispose();

        if (ys > 1e-10) {
          if (this.sHistory.length >= group.historySize) {
            this.sHistory.shift()!.dispose();
            this.yHistory.shift()!.dispose();
            this.rhoHistory.shift();
          }
          this.sHistory.push(s);
          this.yHistory.push(y);
          this.rhoHistory.push(1.0 / ys);
        } else {
          s.dispose();
          y.dispose();
        }

        flatGrad.dispose();
        flatGrad = nextGrad;
        lossVal = acceptedLoss;

        if (Math.abs(lossVal - prevLoss) <= group.toleranceChange) {
          break;
        }
        prevLoss = lossVal;
      }
      return currentLossT;
    } finally {
      flatGrad.dispose();
    }
  }
}

import { cat as catOp } from '../ops.js';

function concat1d(parts: Tensor[]): Tensor {
  if (parts.length === 1) return parts[0]!.clone();
  return catOp(parts, 0);
}
