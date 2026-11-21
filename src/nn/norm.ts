import { noGrad } from '../autograd.js';
import { distributed, ReduceOp, type ProcessGroup } from '../distributed.js';
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
export class BatchNorm3d extends BatchNormBase {}

/**
 * Synchronized BatchNorm across distributed ranks (`distributed.allReduce`).
 */
export class SyncBatchNorm extends BatchNormBase {
  readonly processGroup?: ProcessGroup;

  constructor(numFeatures: number, opts?: BatchNormOptions & { processGroup?: ProcessGroup }) {
    super(numFeatures, opts);
    this.processGroup = opts?.processGroup;
  }

  override forward(input: Tensor): Tensor {
    const syncDist =
      this.training &&
      (this.processGroup !== undefined || distributed.isInitialized()) &&
      distributed.getWorldSize(this.processGroup) > 1;

    if (!syncDist) {
      return super.forward(input);
    }

    const reduceDims = Array.from({ length: input.ndim }, (_, d) => d).filter((d) => d !== 1);
    const localCount = input.numel / this.numFeatures;
    const localSum = input.sum(reduceDims, false);
    const localSqSum = input.pow(2).sum(reduceDims, false);
    const countTensor = ones([1], { dtype: input.dtype, device: input.device }).mul(localCount);

    // Sync exact sufficient statistics across ranks (differentiable via mean/var formula)
    const syncedSum = localSum.clone();
    const syncedSqSum = localSqSum.clone();
    distributed.allReduce(syncedSum, ReduceOp.SUM, this.processGroup);
    distributed.allReduce(syncedSqSum, ReduceOp.SUM, this.processGroup);
    distributed.allReduce(countTensor, ReduceOp.SUM, this.processGroup);

    // Straight-through gradient connection to local batch statistics
    const sumDiff = syncedSum.sub(localSum).detach();
    const sqDiff = syncedSqSum.sub(localSqSum).detach();
    const globalSum = localSum.add(sumDiff);
    const globalSqSum = localSqSum.add(sqDiff);
    const totalCount = Math.max(1, countTensor.item());

    const mean = globalSum.div(totalCount);
    const varBiased = globalSqSum.div(totalCount).sub(mean.pow(2)).clamp(0, null);
    const varUnbiased = varBiased.mul(totalCount / Math.max(1, totalCount - 1));

    if (this.trackRunningStats && this.runningMean && this.runningVar) {
      noGrad(() => {
        this.runningMean!.mul_(1 - this.momentum).add_(mean, this.momentum);
        this.runningVar!.mul_(1 - this.momentum).add_(varUnbiased, this.momentum);
      });
    }

    const viewShape = input.shape.map((s, d) => (d === 1 ? s : 1));
    let out = input
      .sub(mean.reshape(viewShape))
      .mul(varBiased.add(this.eps).rsqrt().reshape(viewShape));
    if (this.weight) {
      out = out.mul(this.weight.reshape(viewShape));
    }
    if (this.bias) {
      out = out.add(this.bias.reshape(viewShape));
    }
    countTensor.dispose();
    syncedSum.dispose();
    syncedSqSum.dispose();
    return out;
  }
}

export interface InstanceNormLayerOptions extends TensorOptions {
  eps?: number;
  momentum?: number;
  affine?: boolean;
  trackRunningStats?: boolean;
}

abstract class InstanceNormBase extends Module {
  readonly numFeatures: number;
  readonly eps: number;
  readonly momentum: number;
  readonly affine: boolean;
  readonly trackRunningStats: boolean;
  readonly weight: Parameter | null;
  readonly bias: Parameter | null;
  runningMean: Tensor | null = null;
  runningVar: Tensor | null = null;

  constructor(numFeatures: number, opts?: InstanceNormLayerOptions) {
    super();
    this.numFeatures = numFeatures;
    this.eps = opts?.eps ?? 1e-5;
    this.momentum = opts?.momentum ?? 0.1;
    this.affine = opts?.affine ?? false;
    this.trackRunningStats = opts?.trackRunningStats ?? false;
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
    return F.instanceNorm(input, {
      runningMean: this.runningMean,
      runningVar: this.runningVar,
      weight: this.weight,
      bias: this.bias,
      useInputStats: this.training || !this.trackRunningStats,
      momentum: this.momentum,
      eps: this.eps,
    });
  }
}

export class InstanceNorm1d extends InstanceNormBase {}
export class InstanceNorm2d extends InstanceNormBase {}

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
