import { noGrad } from './autograd.js';
import { functional as F } from './functional.js';
import { native, type NativeTensor } from './native.js';
import { full, rand, randn, tensor as createTensor, where } from './ops.js';
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

export function klDivergence(p: Distribution, q: Distribution): Tensor {
  if (p instanceof Normal && q instanceof Normal) {
    const varRatio = p.scale.div(q.scale).pow(2);
    const t1 = p.loc.sub(q.loc).div(q.scale).pow(2);
    return varRatio.add(t1).sub(1).sub(varRatio.log()).mul(0.5);
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
  klDivergence,
};
