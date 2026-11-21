import * as F from '../functional.js';
import { ones, zeros, type TensorOptions } from '../ops.js';
import type { Tensor } from '../tensor.js';
import { Module } from './module.js';
import { Parameter } from './parameter.js';

export interface LayerNormOptions extends TensorOptions {
  eps?: number;
  elementwiseAffine?: boolean;
  bias?: boolean;
}

export class LayerNorm extends Module {
  readonly normalizedShape: number[];
  readonly eps: number;
  readonly weight: Parameter | null;
  readonly bias: Parameter | null;

  constructor(normalizedShape: number | number[], opts?: LayerNormOptions) {
    super();
    this.normalizedShape = Array.isArray(normalizedShape) ? normalizedShape : [normalizedShape];
    this.eps = opts?.eps ?? 1e-5;
    const affine = opts?.elementwiseAffine ?? true;
    const useBias = opts?.bias ?? true;
    const factoryOpts: TensorOptions = { dtype: opts?.dtype, device: opts?.device };

    if (affine) {
      const w = ones(this.normalizedShape, factoryOpts);
      this.weight = new Parameter(w, true);
      w.dispose();
      if (useBias) {
        const b = zeros(this.normalizedShape, factoryOpts);
        this.bias = new Parameter(b, true);
        b.dispose();
      } else {
        this.bias = null;
      }
    } else {
      this.weight = null;
      this.bias = null;
    }
  }

  forward(input: Tensor): Tensor {
    return F.layerNorm(input, this.normalizedShape, this.weight, this.bias, this.eps);
  }
}

export interface RMSNormOptions extends TensorOptions {
  eps?: number;
  elementwiseAffine?: boolean;
}

export class RMSNorm extends Module {
  readonly normalizedShape: number[];
  readonly eps: number | null;
  readonly weight: Parameter | null;

  constructor(normalizedShape: number | number[], opts?: RMSNormOptions) {
    super();
    this.normalizedShape = Array.isArray(normalizedShape) ? normalizedShape : [normalizedShape];
    this.eps = opts?.eps ?? null;
    const affine = opts?.elementwiseAffine ?? true;
    const factoryOpts: TensorOptions = { dtype: opts?.dtype, device: opts?.device };

    if (affine) {
      const w = ones(this.normalizedShape, factoryOpts);
      this.weight = new Parameter(w, true);
      w.dispose();
    } else {
      this.weight = null;
    }
  }

  forward(input: Tensor): Tensor {
    return F.rmsNorm(input, this.normalizedShape, this.weight, this.eps);
  }
}

export interface BatchNormOptions extends TensorOptions {
  eps?: number;
  momentum?: number;
  affine?: boolean;
  trackRunningStats?: boolean;
}

abstract class BatchNormBase extends Module {
  readonly numFeatures: number;
  readonly eps: number;
  readonly momentum: number;
  readonly affine: boolean;
  readonly trackRunningStats: boolean;
  readonly weight: Parameter | null;
  readonly bias: Parameter | null;
  runningMean: Tensor | null = null;
  runningVar: Tensor | null = null;

  constructor(numFeatures: number, opts?: BatchNormOptions) {
    super();
    this.numFeatures = numFeatures;
    this.eps = opts?.eps ?? 1e-5;
    this.momentum = opts?.momentum ?? 0.1;
    this.affine = opts?.affine ?? true;
    this.trackRunningStats = opts?.trackRunningStats ?? true;
    const factoryOpts: TensorOptions = { dtype: opts?.dtype, device: opts?.device };

    if (this.affine) {
      const w = ones([numFeatures], factoryOpts);
      const b = zeros([numFeatures], factoryOpts);
      this.weight = new Parameter(w, true);
      this.bias = new Parameter(b, true);
      w.dispose();
      b.dispose();
    } else {
      this.weight = null;
      this.bias = null;
    }

    if (this.trackRunningStats) {
      this.runningMean = zeros([numFeatures], factoryOpts);
      this.runningVar = ones([numFeatures], factoryOpts);
      this.registerBuffer('runningMean', this.runningMean);
      this.registerBuffer('runningVar', this.runningVar);
    }
  }

  forward(input: Tensor): Tensor {
    return F.batchNorm(input, this.runningMean, this.runningVar, {
      weight: this.weight,
      bias: this.bias,
      training: this.training || !this.trackRunningStats,
      momentum: this.momentum,
      eps: this.eps,
    });
  }
}

export class BatchNorm1d extends BatchNormBase {}
export class BatchNorm2d extends BatchNormBase {}

export interface GroupNormOptions extends TensorOptions {
  eps?: number;
  affine?: boolean;
}

export class GroupNorm extends Module {
  readonly numGroups: number;
  readonly numChannels: number;
  readonly eps: number;
  readonly weight: Parameter | null;
  readonly bias: Parameter | null;

  constructor(numGroups: number, numChannels: number, opts?: GroupNormOptions) {
    super();
    this.numGroups = numGroups;
    this.numChannels = numChannels;
    this.eps = opts?.eps ?? 1e-5;
    const affine = opts?.affine ?? true;
    const factoryOpts: TensorOptions = { dtype: opts?.dtype, device: opts?.device };
    if (affine) {
      const w = ones([numChannels], factoryOpts);
      const b = zeros([numChannels], factoryOpts);
      this.weight = new Parameter(w, true);
      this.bias = new Parameter(b, true);
      w.dispose();
      b.dispose();
    } else {
      this.weight = null;
      this.bias = null;
    }
  }

  forward(input: Tensor): Tensor {
    return F.groupNorm(input, this.numGroups, this.weight, this.bias, this.eps);
  }
}
