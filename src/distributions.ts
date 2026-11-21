import { noGrad } from './autograd.js';
import { functional as F } from './functional.js';
import { linalg } from './linalg.js';
import { native, type NativeTensor } from './native.js';
import { full, rand, randn, tensor as createTensor, where } from './ops.js';
import { digamma, gammaln } from './special.js';
import { Tensor, wrapTensor } from './tensor.js';

const LOG_2_PI = Math.log(2 * Math.PI);

function toTensorParam(val: Tensor | number, ref?: Tensor): Tensor {
  if (val instanceof Tensor) return val;
  if (ref) {
    return full([], val, { dtype: ref.dtype, device: ref.device });
  }
  return createTensor(val, { dtype: 'float32' });
}

function broadcastShape(a: number[], b: number[]): number[] {
  const len = Math.max(a.length, b.length);
  const out = new Array<number>(len);
  for (let i = 0; i < len; i++) {
    const da = a[a.length - 1 - i] ?? 1;
    const db = b[b.length - 1 - i] ?? 1;
    if (da !== db && da !== 1 && db !== 1) {
      throw new Error(`Shapes [${a.join(', ')}] and [${b.join(', ')}] are not broadcastable`);
    }
    out[len - 1 - i] = Math.max(da, db);
  }
  return out;
}

export abstract class Distribution {
  abstract readonly batchShape: number[];
  abstract get mean(): Tensor;
  abstract get variance(): Tensor;

  get stddev(): Tensor {
    return this.variance.sqrt();
  }

  abstract sample(sampleShape?: number[]): Tensor;

  rsample(_sampleShape?: number[]): Tensor {
    throw new Error(`${this.constructor.name} does not implement reparameterized rsample()`);
  }

  abstract logProb(value: Tensor): Tensor;
  abstract entropy(): Tensor;
}

export class Normal extends Distribution {
  readonly loc: Tensor;
  readonly scale: Tensor;
  readonly batchShape: number[];

  constructor(loc: Tensor | number, scale: Tensor | number) {
    super();
    const ref = loc instanceof Tensor ? loc : scale instanceof Tensor ? scale : undefined;
    this.loc = toTensorParam(loc, ref);
    this.scale = toTensorParam(scale, this.loc);
    this.batchShape = broadcastShape(this.loc.shape, this.scale.shape);
  }

  get mean(): Tensor {
    return this.loc.expand(this.batchShape);
  }

  get variance(): Tensor {
    return this.scale.pow(2).expand(this.batchShape);
  }

  override get stddev(): Tensor {
    return this.scale.expand(this.batchShape);
  }

  override rsample(sampleShape: number[] = []): Tensor {
    const shape = [...sampleShape, ...this.batchShape];
    const eps = randn(shape, { dtype: this.loc.dtype, device: this.loc.device });
    return this.loc.add(eps.mul(this.scale));
  }

  sample(sampleShape: number[] = []): Tensor {
    return noGrad(() => this.rsample(sampleShape));
  }

  logProb(value: Tensor): Tensor {
    const variance = this.scale.pow(2);
    const logScale = this.scale.log();
    return value
      .sub(this.loc)
      .pow(2)
      .neg()
      .div(variance.mul(2))
      .sub(logScale)
      .sub(0.5 * LOG_2_PI);
  }

  entropy(): Tensor {
    return this.scale.log().add(0.5 + 0.5 * LOG_2_PI);
  }
}

export class Uniform extends Distribution {
  readonly low: Tensor;
  readonly high: Tensor;
  readonly batchShape: number[];

  constructor(low: Tensor | number, high: Tensor | number) {
    super();
    const ref = low instanceof Tensor ? low : high instanceof Tensor ? high : undefined;
    this.low = toTensorParam(low, ref);
    this.high = toTensorParam(high, this.low);
    this.batchShape = broadcastShape(this.low.shape, this.high.shape);
  }

  get mean(): Tensor {
    return this.low.add(this.high).div(2).expand(this.batchShape);
  }

  get variance(): Tensor {
    return this.high.sub(this.low).pow(2).div(12).expand(this.batchShape);
  }

  override rsample(sampleShape: number[] = []): Tensor {
    const shape = [...sampleShape, ...this.batchShape];
    const u = rand(shape, { dtype: this.low.dtype, device: this.low.device });
    return this.low.add(u.mul(this.high.sub(this.low)));
  }

  sample(sampleShape: number[] = []): Tensor {
    return noGrad(() => this.rsample(sampleShape));
  }

  logProb(value: Tensor): Tensor {
    const lb = value.ge(this.low);
    const ub = value.lt(this.high);
    const inSupport = wrapTensor(native.callOpAuto('logical_and', [lb._native, ub._native]) as NativeTensor);
    const inside = this.high.sub(this.low).log().neg().expand(value.shape);
    const negInf = full(value.shape, Number.NEGATIVE_INFINITY, {
      dtype: inside.dtype,
      device: inside.device,
    });
    return where(inSupport, inside, negInf);
  }

  entropy(): Tensor {
    return this.high.sub(this.low).log();
  }
}

export interface BernoulliOptions {
  probs?: Tensor | number;
  logits?: Tensor | number;
}

export class Bernoulli extends Distribution {
  readonly probs: Tensor;
  readonly logits: Tensor;
  readonly batchShape: number[];

  constructor(opts: BernoulliOptions) {
    super();
    if ((opts.probs === undefined) === (opts.logits === undefined)) {
      throw new Error('Bernoulli requires exactly one of probs or logits');
    }
    if (opts.probs !== undefined) {
      this.probs = toTensorParam(opts.probs);
      const clamped = this.probs.clamp(1e-7, 1 - 1e-7);
      this.logits = clamped.div(clamped.neg().add(1)).log();
    } else {
      this.logits = toTensorParam(opts.logits!);
      this.probs = this.logits.sigmoid();
    }
    this.batchShape = this.probs.shape;
  }

  get mean(): Tensor {
    return this.probs;
  }

  get variance(): Tensor {
    return this.probs.mul(this.probs.neg().add(1));
  }

  sample(sampleShape: number[] = []): Tensor {
    return noGrad(() => {
      const shape = [...sampleShape, ...this.batchShape];
      const u = rand(shape, { dtype: this.probs.dtype, device: this.probs.device });
      return u.lt(this.probs).to(this.probs.dtype);
    });
  }

  logProb(value: Tensor): Tensor {
    return F.binaryCrossEntropyWithLogits(this.logits, value.to(this.logits.dtype), {
      reduction: 'none',
    }).neg();
  }

  entropy(): Tensor {
    return F.binaryCrossEntropyWithLogits(this.logits, this.probs, { reduction: 'none' });
  }
}

export interface CategoricalOptions {
  probs?: Tensor;
  logits?: Tensor;
}

export class Categorical extends Distribution {
  readonly probs: Tensor;
  readonly logits: Tensor;
  readonly numEvents: number;
  readonly batchShape: number[];

  constructor(opts: CategoricalOptions) {
    super();
    if ((opts.probs === undefined) === (opts.logits === undefined)) {
      throw new Error('Categorical requires exactly one of probs or logits');
    }
    if (opts.probs !== undefined) {
      const p = opts.probs;
      const denom = p.sum(-1, true);
      this.probs = p.div(denom);
      this.logits = this.probs.clamp(1e-37, null).log();
    } else {
      const raw = opts.logits!;
      this.logits = raw.logSoftmax(-1);
      this.probs = this.logits.exp();
    }
    const shape = this.probs.shape;
    if (shape.length < 1) {
      throw new Error('Categorical probs/logits must have at least 1 dimension');
    }
    this.numEvents = shape[shape.length - 1]!;
    this.batchShape = shape.slice(0, -1);
  }

  get mean(): Tensor {
    return full(this.batchShape, Number.NaN, { dtype: this.probs.dtype, device: this.probs.device });
  }

  get variance(): Tensor {
    return full(this.batchShape, Number.NaN, { dtype: this.probs.dtype, device: this.probs.device });
  }

  sample(sampleShape: number[] = []): Tensor {
    return noGrad(() => {
      const numSamples = sampleShape.reduce((a, b) => a * b, 1);
      const probs2d = this.probs.reshape(-1, this.numEvents);
      const samples2d = wrapTensor(
        native.callOpAuto('multinomial', [probs2d._native, numSamples, true]) as NativeTensor
      );
      return samples2d.transpose(0, 1).reshape([...sampleShape, ...this.batchShape]);
    });
  }

  logProb(value: Tensor): Tensor {
    const idx = value.long();
    const targetShape = broadcastShape(idx.shape, this.batchShape);
    const expandedLogits = this.logits.expand([...targetShape, this.numEvents]);
    const expandedIdx = idx.expand(targetShape).unsqueeze(-1);
    return expandedLogits.gather(-1, expandedIdx).squeeze(-1);
  }

  entropy(): Tensor {
    return this.probs.mul(this.logits).sum(-1).neg();
  }
}

export class Exponential extends Distribution {
  readonly rate: Tensor;
  readonly batchShape: number[];

  constructor(rate: Tensor | number) {
    super();
    this.rate = toTensorParam(rate);
    this.batchShape = this.rate.shape;
  }

  get mean(): Tensor {
    return this.rate.pow(-1);
  }

  get variance(): Tensor {
    return this.rate.pow(-2);
  }

  override rsample(sampleShape: number[] = []): Tensor {
    const shape = [...sampleShape, ...this.batchShape];
    const u = rand(shape, { dtype: this.rate.dtype, device: this.rate.device }).clamp(1e-7, 1 - 1e-7);
    const out = u.neg().add(1).log().neg().div(this.rate);
    u.dispose();
    return out;
  }

  sample(sampleShape: number[] = []): Tensor {
    return noGrad(() => this.rsample(sampleShape));
  }

  logProb(value: Tensor): Tensor {
    return this.rate.log().sub(this.rate.mul(value));
  }

  entropy(): Tensor {
    return this.rate.log().neg().add(1);
  }
}

export class Poisson extends Distribution {
  readonly rate: Tensor;
  readonly batchShape: number[];

  constructor(rate: Tensor | number) {
    super();
    this.rate = toTensorParam(rate);
    this.batchShape = this.rate.shape;
  }

  get mean(): Tensor {
    return this.rate;
  }

  get variance(): Tensor {
    return this.rate;
  }

  sample(sampleShape: number[] = []): Tensor {
    return noGrad(() => {
      const expanded = this.rate.expand([...sampleShape, ...this.batchShape]);
      return wrapTensor(native.callOpAuto('poisson', [expanded._native]) as NativeTensor);
    });
  }

  logProb(value: Tensor): Tensor {
    const v = value.to(this.rate.dtype);
    return v.mul(this.rate.clamp(1e-37, null).log()).sub(this.rate).sub(gammaln(v.add(1)));
  }

  entropy(): Tensor {
    throw new Error('Poisson does not have a closed-form entropy');
  }
}

export class Gamma extends Distribution {
  readonly concentration: Tensor;
  readonly rate: Tensor;
  readonly batchShape: number[];

  constructor(concentration: Tensor | number, rate: Tensor | number) {
    super();
    const ref =
      concentration instanceof Tensor ? concentration : rate instanceof Tensor ? rate : undefined;
    this.concentration = toTensorParam(concentration, ref);
    this.rate = toTensorParam(rate, this.concentration);
    this.batchShape = broadcastShape(this.concentration.shape, this.rate.shape);
  }

  get mean(): Tensor {
    return this.concentration.div(this.rate);
  }

  get variance(): Tensor {
    return this.concentration.div(this.rate.pow(2));
  }

  override rsample(sampleShape: number[] = []): Tensor {
    const shape = [...sampleShape, ...this.batchShape];
    const alpha = this.concentration.expand(shape);
    const standard = wrapTensor(native.callOpAuto('_standard_gamma', [alpha._native]) as NativeTensor);
    return standard.div(this.rate);
  }

  sample(sampleShape: number[] = []): Tensor {
    return noGrad(() => this.rsample(sampleShape));
  }

  logProb(value: Tensor): Tensor {
    return this.concentration
      .mul(this.rate.log())
      .add(this.concentration.sub(1).mul(value.log()))
      .sub(this.rate.mul(value))
      .sub(gammaln(this.concentration));
  }

  entropy(): Tensor {
    return this.concentration
      .sub(this.rate.log())
      .add(gammaln(this.concentration))
      .add(this.concentration.neg().add(1).mul(digamma(this.concentration)));
  }
}

export class Beta extends Distribution {
  readonly concentration1: Tensor;
  readonly concentration0: Tensor;
  readonly batchShape: number[];

  constructor(concentration1: Tensor | number, concentration0: Tensor | number) {
    super();
    const ref =
      concentration1 instanceof Tensor
        ? concentration1
        : concentration0 instanceof Tensor
          ? concentration0
          : undefined;
    this.concentration1 = toTensorParam(concentration1, ref);
    this.concentration0 = toTensorParam(concentration0, this.concentration1);
    this.batchShape = broadcastShape(this.concentration1.shape, this.concentration0.shape);
  }

  get mean(): Tensor {
    return this.concentration1.div(this.concentration1.add(this.concentration0));
  }

  get variance(): Tensor {
    const total = this.concentration1.add(this.concentration0);
    return this.concentration1.mul(this.concentration0).div(total.pow(2).mul(total.add(1)));
  }

  override rsample(sampleShape: number[] = []): Tensor {
    const shape = [...sampleShape, ...this.batchShape];
    const g1 = wrapTensor(
      native.callOpAuto('_standard_gamma', [this.concentration1.expand(shape)._native]) as NativeTensor
    );
    const g0 = wrapTensor(
      native.callOpAuto('_standard_gamma', [this.concentration0.expand(shape)._native]) as NativeTensor
    );
    return g1.div(g1.add(g0));
  }

  sample(sampleShape: number[] = []): Tensor {
    return noGrad(() => this.rsample(sampleShape));
  }

  logProb(value: Tensor): Tensor {
    const a = this.concentration1;
    const b = this.concentration0;
    const logBeta = gammaln(a).add(gammaln(b)).sub(gammaln(a.add(b)));
    return a
      .sub(1)
      .mul(value.log())
      .add(b.sub(1).mul(value.neg().add(1).log()))
      .sub(logBeta);
  }

  entropy(): Tensor {
    const a = this.concentration1;
    const b = this.concentration0;
    const ab = a.add(b);
    const logBeta = gammaln(a).add(gammaln(b)).sub(gammaln(ab));
    return logBeta
      .sub(a.sub(1).mul(digamma(a)))
      .sub(b.sub(1).mul(digamma(b)))
      .add(ab.sub(2).mul(digamma(ab)));
  }
}

export class Dirichlet extends Distribution {
  readonly concentration: Tensor;
  readonly batchShape: number[];

  constructor(concentration: Tensor) {
    super();
    if (concentration.ndim < 1) {
      throw new Error('Dirichlet concentration must have at least 1 dimension');
    }
    this.concentration = concentration;
    this.batchShape = concentration.shape.slice(0, -1);
  }

  get mean(): Tensor {
    return this.concentration.div(this.concentration.sum(-1, true));
  }

  get variance(): Tensor {
    const total = this.concentration.sum(-1, true);
    return this.concentration.mul(total.sub(this.concentration)).div(total.pow(2).mul(total.add(1)));
  }

  override rsample(sampleShape: number[] = []): Tensor {
    const shape = [...sampleShape, ...this.concentration.shape];
    const g = wrapTensor(
      native.callOpAuto('_standard_gamma', [this.concentration.expand(shape)._native]) as NativeTensor
    );
    return g.div(g.sum(-1, true));
  }

  sample(sampleShape: number[] = []): Tensor {
    return noGrad(() => this.rsample(sampleShape));
  }

  logProb(value: Tensor): Tensor {
    const a = this.concentration;
    const term = value.log().mul(a.sub(1)).sum(-1);
    const norm = gammaln(a.sum(-1)).sub(gammaln(a).sum(-1));
    return term.add(norm);
  }

  entropy(): Tensor {
    const a = this.concentration;
    const a0 = a.sum(-1);
    const k = a.size(-1);
    const logB = gammaln(a).sum(-1).sub(gammaln(a0));
    return logB.add(a0.sub(k).mul(digamma(a0))).sub(a.sub(1).mul(digamma(a)).sum(-1));
  }
}

export interface MultivariateNormalOptions {
  covarianceMatrix?: Tensor;
  scaleTril?: Tensor;
}

export class MultivariateNormal extends Distribution {
  readonly loc: Tensor;
  readonly scaleTril: Tensor;
  readonly batchShape: number[];
  readonly eventDim: number;

  constructor(loc: Tensor, opts: MultivariateNormalOptions) {
    super();
    if (loc.ndim < 1) {
      throw new Error('MultivariateNormal loc must have at least 1 dimension');
    }
    this.loc = loc;
    this.eventDim = loc.size(-1);
    if (opts.scaleTril) {
      this.scaleTril = opts.scaleTril;
    } else if (opts.covarianceMatrix) {
      this.scaleTril = linalg.cholesky(opts.covarianceMatrix);
    } else {
      throw new Error('MultivariateNormal requires either covarianceMatrix or scaleTril');
    }
    this.batchShape = broadcastShape(loc.shape.slice(0, -1), this.scaleTril.shape.slice(0, -2));
  }

  get mean(): Tensor {
    return this.loc.expand([...this.batchShape, this.eventDim]);
  }

  get covarianceMatrix(): Tensor {
    return this.scaleTril.matmul(this.scaleTril.transpose(-1, -2));
  }

  get variance(): Tensor {
    return this.scaleTril.pow(2).sum(-1);
  }

  override rsample(sampleShape: number[] = []): Tensor {
    const shape = [...sampleShape, ...this.batchShape, this.eventDim, 1];
    const eps = randn(shape, { dtype: this.loc.dtype, device: this.loc.device });
    const delta = this.scaleTril.matmul(eps).squeeze(-1);
    eps.dispose();
    return this.loc.add(delta);
  }

  sample(sampleShape: number[] = []): Tensor {
    return noGrad(() => this.rsample(sampleShape));
  }

  logProb(value: Tensor): Tensor {
    const diff = value.sub(this.loc).unsqueeze(-1);
    const solved = wrapTensor(
      native.callOp('aten::linalg_solve_triangular', '', [this.scaleTril._native, diff._native], {
        upper: false,
      }) as NativeTensor
    ).squeeze(-1);
    const maha = solved.pow(2).sum(-1);
    const diag = wrapTensor(
      native.callOpAuto('diagonal', [this.scaleTril._native, 0, -2, -1]) as NativeTensor
    );
    const halfLogDet = diag.log().sum(-1);
    return maha.add(this.eventDim * LOG_2_PI).mul(-0.5).sub(halfLogDet);
  }

  entropy(): Tensor {
    const diag = wrapTensor(
      native.callOpAuto('diagonal', [this.scaleTril._native, 0, -2, -1]) as NativeTensor
    );
    const halfLogDet = diag.log().sum(-1);
    return halfLogDet.add(0.5 * this.eventDim * (1 + LOG_2_PI));
  }
}

/**
 * Reinterprets `reinterpretedBatchNd` rightmost batch dimensions as event dimensions
 * (matching `torch.distributions.Independent`).
 */
export class Independent extends Distribution {
  readonly baseDist: Distribution;
  readonly reinterpretedBatchNd: number;
  readonly batchShape: number[];

  constructor(baseDist: Distribution, reinterpretedBatchNd: number) {
    super();
    if (reinterpretedBatchNd < 0 || reinterpretedBatchNd > baseDist.batchShape.length) {
      throw new RangeError(
        `reinterpretedBatchNd (${reinterpretedBatchNd}) out of range for batchShape [${baseDist.batchShape.join(', ')}]`
      );
    }
    this.baseDist = baseDist;
    this.reinterpretedBatchNd = reinterpretedBatchNd;
    this.batchShape = baseDist.batchShape.slice(0, baseDist.batchShape.length - reinterpretedBatchNd);
  }

  get mean(): Tensor {
    return this.baseDist.mean;
  }

  get variance(): Tensor {
    return this.baseDist.variance;
  }

  sample(sampleShape: number[] = []): Tensor {
    return this.baseDist.sample(sampleShape);
  }

  override rsample(sampleShape: number[] = []): Tensor {
    return this.baseDist.rsample(sampleShape);
  }

  private sumRightmost(t: Tensor): Tensor {
    if (this.reinterpretedBatchNd === 0) return t;
    const dims = Array.from({ length: this.reinterpretedBatchNd }, (_, i) => -(i + 1));
    return t.sum(dims, false);
  }

  logProb(value: Tensor): Tensor {
    return this.sumRightmost(this.baseDist.logProb(value));
  }

  entropy(): Tensor {
    return this.sumRightmost(this.baseDist.entropy());
  }
}

export function klDivergence(p: Distribution, q: Distribution): Tensor {
  if (p instanceof Independent && q instanceof Independent) {
    if (p.reinterpretedBatchNd !== q.reinterpretedBatchNd) {
      throw new Error('Independent KL requires identical reinterpretedBatchNd');
    }
    const baseKl = klDivergence(p.baseDist, q.baseDist);
    if (p.reinterpretedBatchNd === 0) return baseKl;
    const dims = Array.from({ length: p.reinterpretedBatchNd }, (_, i) => -(i + 1));
    return baseKl.sum(dims, false);
  }
  if (p instanceof Normal && q instanceof Normal) {
    const varRatio = p.scale.div(q.scale).pow(2);
    const t1 = p.loc.sub(q.loc).div(q.scale).pow(2);
    return varRatio.add(t1).sub(1).sub(varRatio.log()).mul(0.5);
  }
  if (p instanceof Exponential && q instanceof Exponential) {
    const rateRatio = q.rate.div(p.rate);
    return rateRatio.sub(rateRatio.log()).sub(1);
  }
  if (p instanceof MultivariateNormal && q instanceof MultivariateNormal) {
    const k = p.eventDim;
    const pDiag = wrapTensor(
      native.callOpAuto('diagonal', [p.scaleTril._native, 0, -2, -1]) as NativeTensor
    );
    const qDiag = wrapTensor(
      native.callOpAuto('diagonal', [q.scaleTril._native, 0, -2, -1]) as NativeTensor
    );
    const logDetTerm = qDiag.log().sum(-1).sub(pDiag.log().sum(-1));
    const m = wrapTensor(
      native.callOp('aten::linalg_solve_triangular', '', [q.scaleTril._native, p.scaleTril._native], {
        upper: false,
      }) as NativeTensor
    );
    const traceTerm = m.pow(2).sum([-2, -1]);
    const diff = q.loc.sub(p.loc).unsqueeze(-1);
    const solvedDiff = wrapTensor(
      native.callOp('aten::linalg_solve_triangular', '', [q.scaleTril._native, diff._native], {
        upper: false,
      }) as NativeTensor
    ).squeeze(-1);
    const quadTerm = solvedDiff.pow(2).sum(-1);
    return traceTerm.add(quadTerm).sub(k).mul(0.5).add(logDetTerm);
  }
  if (p instanceof Uniform && q instanceof Uniform) {
    const outside = wrapTensor(
      native.callOpAuto('logical_or', [p.low.lt(q.low)._native, p.high.gt(q.high)._native]) as NativeTensor
    );
    const val = q.high.sub(q.low).div(p.high.sub(p.low)).log();
    const inf = full(val.shape, Number.POSITIVE_INFINITY, { dtype: val.dtype, device: val.device });
    return where(outside, inf, val);
  }
  if (p instanceof Bernoulli && q instanceof Bernoulli) {
    const t1 = p.probs.mul(p.probs.clamp(1e-37, null).div(q.probs.clamp(1e-37, null)).log());
    const p1 = p.probs.neg().add(1);
    const q1 = q.probs.neg().add(1);
    const t2 = p1.mul(p1.clamp(1e-37, null).div(q1.clamp(1e-37, null)).log());
    return t1.add(t2);
  }
  if (p instanceof Categorical && q instanceof Categorical) {
    return p.probs.mul(p.logits.sub(q.logits)).sum(-1);
  }
  throw new Error(
    `klDivergence not implemented for (${p.constructor.name}, ${q.constructor.name})`
  );
}

export const distributions = {
  Distribution,
  Normal,
  Uniform,
  Bernoulli,
  Categorical,
  Exponential,
  Poisson,
  Gamma,
  Beta,
  Dirichlet,
  MultivariateNormal,
  Independent,
  klDivergence,
};
