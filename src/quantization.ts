import { AutogradContext, AutogradFunction, noGrad } from './autograd.js';
import * as F from './functional.js';
import { Linear } from './nn/linear.js';
import { Module } from './nn/module.js';
import { native, type NativeTensor } from './native.js';
import { full } from './ops.js';
import { Tensor, wrapTensor } from './tensor.js';

function roundTensor(t: Tensor): Tensor {
  return wrapTensor(native.callOpAuto('round', [t._native]) as NativeTensor);
}

class FakeQuantizeFn extends AutogradFunction {
  static forward(
    ctx: AutogradContext,
    x: Tensor,
    scale: Tensor,
    zeroPoint: Tensor,
    qmin: number,
    qmax: number
  ): Tensor {
    const q = roundTensor(x.div(scale).add(zeroPoint));
    const mask = q.ge(qmin).mul(q.le(qmax)).to(x.dtype);
    ctx.saveForBackward(mask);
    const clamped = q.clamp(qmin, qmax);
    q.dispose();
    const out = clamped.sub(zeroPoint).mul(scale);
    clamped.dispose();
    return out;
  }

  static backward(ctx: AutogradContext, gradOut: Tensor): (Tensor | null)[] {
    const [mask] = ctx.savedTensors;
    // Straight-through estimator (STE) masked to [qmin, qmax]
    return [gradOut.mul(mask!), null, null, null, null];
  }
}

export interface FakeQuantizeOptions {
  bits?: number;
  symmetric?: boolean;
  scale?: number;
  zeroPoint?: number;
}

/**
 * Quantization-Aware Training (QAT) fake-quantization module using a
 * Straight-Through Estimator (STE) custom AutogradFunction.
 */
export class FakeQuantize extends Module {
  readonly qmin: number;
  readonly qmax: number;
  readonly symmetric: boolean;
  readonly scale: Tensor;
  readonly zeroPoint: Tensor;
  enabled = true;

  constructor(opts?: FakeQuantizeOptions) {
    super();
    const bits = opts?.bits ?? 8;
    this.symmetric = opts?.symmetric ?? true;
    this.qmin = this.symmetric ? -(1 << (bits - 1)) : 0;
    this.qmax = this.symmetric ? (1 << (bits - 1)) - 1 : (1 << bits) - 1;
    this.scale = full([1], opts?.scale ?? 1 / 127, { dtype: 'float32' });
    this.zeroPoint = full([1], opts?.zeroPoint ?? 0, { dtype: 'float32' });
    this.registerBuffer('scale', this.scale);
    this.registerBuffer('zeroPoint', this.zeroPoint);
  }

  calibrate(sample: Tensor): void {
    noGrad(() => {
      if (this.symmetric) {
        const maxAbs = Math.max(sample.abs().max().item(), 1e-8);
        this.scale.fill_(maxAbs / this.qmax);
        this.zeroPoint.fill_(0);
      } else {
        const minVal = sample.min().item();
        const maxVal = sample.max().item();
        const range = Math.max(maxVal - minVal, 1e-8);
        const s = range / (this.qmax - this.qmin);
        const zp = Math.round(this.qmin - minVal / s);
        this.scale.fill_(s);
        this.zeroPoint.fill_(Math.min(this.qmax, Math.max(this.qmin, zp)));
      }
    });
  }

  forward(input: Tensor): Tensor {
    if (!this.enabled) return input;
    return FakeQuantizeFn.apply(input, this.scale, this.zeroPoint, this.qmin, this.qmax) as Tensor;
  }
}

/**
 * Dynamically quantized INT8 Linear layer storing per-output-channel symmetric
 * `int8` weights (~4x smaller memory & checkpoint footprint than float32).
 */
export class QuantizedLinear extends Module {
  readonly inFeatures: number;
  readonly outFeatures: number;
  readonly weightInt8: Tensor;
  readonly weightScale: Tensor;
  readonly bias: Tensor | null;

  constructor(
    inFeatures: number,
    outFeatures: number,
    weightInt8: Tensor,
    weightScale: Tensor,
    bias: Tensor | null
  ) {
    super();
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    this.weightInt8 = weightInt8;
    this.weightScale = weightScale;
    this.bias = bias;
    this.registerBuffer('weightInt8', weightInt8);
    this.registerBuffer('weightScale', weightScale);
    if (bias) {
      this.registerBuffer('bias', bias);
    }
  }

  static fromFloat(linear: Linear): QuantizedLinear {
    return noGrad(() => {
      const w = linear.weight.detach();
      const maxAbs = w.abs().max(1, true).values.clamp(1e-8, null);
      const scale = maxAbs.div(127.0).contiguous();
      const q = roundTensor(w.div(scale)).clamp(-127, 127).to('int8').contiguous();
      const biasCopy = linear.bias ? linear.bias.detach().clone() : null;
      w.dispose();
      maxAbs.dispose();
      return new QuantizedLinear(linear.inFeatures, linear.outFeatures, q, scale, biasCopy);
    });
  }

  forward(input: Tensor): Tensor {
    // Dynamic per-row activation quantization + dequantized weight matmul
    const deqW = this.weightInt8.to(input.dtype).mul(this.weightScale.to(input.dtype));
    const out = F.linear(input, deqW, this.bias ? this.bias.to(input.dtype) : null);
    deqW.dispose();
    return out;
  }
}

/**
 * Converts `nn.Linear` submodules inside `model` to `QuantizedLinear` (INT8 weights)
 * in-place, disposing the replaced float32 Linear parameters cleanly.
 */
export function quantizeDynamic<T extends Module>(model: T): T {
  const visit = (mod: Module): void => {
    const modMap = (mod as unknown as { _modules: Map<string, Module | null> })._modules;
    const rec = mod as unknown as Record<string, unknown>;
    const childKeys = new Set<string>();
    for (const [k, v] of modMap) {
      if (v) childKeys.add(k);
    }
    for (const [k, v] of Object.entries(mod)) {
      if (v instanceof Module) childKeys.add(k);
    }

    for (const key of childKeys) {
      const child = (modMap.get(key) ?? rec[key]) as Module | undefined;
      if (!child) continue;
      if (child instanceof Linear) {
        const q = QuantizedLinear.fromFloat(child);
        if (modMap.has(key)) modMap.set(key, q);
        if (Object.prototype.hasOwnProperty.call(mod, key)) rec[key] = q;
        child.dispose();
      } else {
        visit(child);
      }
    }
  };
  visit(model);
  return model;
}

export const quantization = {
  FakeQuantize,
  QuantizedLinear,
  quantizeDynamic,
};

export const ao = {
  quantization,
};
