import { noGrad } from './autograd.js';
import type { Device } from './device.js';
import { tidy } from './memory.js';
import { native, type NativeTensor } from './native.js';
import { full } from './ops.js';
import type { Optimizer } from './optim/optimizer.js';
import type { Tensor } from './tensor.js';

export interface GradScalerOptions {
  initScale?: number;
  growthFactor?: number;
  backoffFactor?: number;
  growthInterval?: number;
  enabled?: boolean;
}

export interface GradScalerStateDict {
  scale: number;
  growthFactor: number;
  backoffFactor: number;
  growthInterval: number;
  growthTracker: number;
}

/**
 * Dynamic loss scaling for mixed-precision (FP16/BF16) training,
 * using libtorch's fused `_amp_foreach_non_finite_check_and_unscale_` kernel.
 */
export class GradScaler {
  private _scale: number;
  private growthFactor: number;
  private backoffFactor: number;
  private growthInterval: number;
  readonly enabled: boolean;
  private growthTracker = 0;
  private foundInf = false;
  private unscaled = false;

  constructor(opts: GradScalerOptions = {}) {
    this._scale = opts.initScale ?? 65536.0;
    this.growthFactor = opts.growthFactor ?? 2.0;
    this.backoffFactor = opts.backoffFactor ?? 0.5;
    this.growthInterval = opts.growthInterval ?? 2000;
    this.enabled = opts.enabled ?? true;
  }

  getScale(): number {
    return this.enabled ? this._scale : 1.0;
  }

  scale(loss: Tensor): Tensor {
    if (!this.enabled) return loss;
    return loss.mul(this._scale);
  }

  unscale_(optimizer: Optimizer): void {
    if (!this.enabled || this.unscaled) return;
    this.unscaled = true;
    this.foundInf = false;
    const invScaleVal = 1.0 / this._scale;

    noGrad(() => {
      tidy(() => {
        // Group gradients by device+dtype for fused multi-tensor unscale + inf check
        const buckets = new Map<string, { device: Device; grads: NativeTensor[] }>();
        for (const group of optimizer.paramGroups) {
          for (const p of group.params) {
            const g = p.grad;
            if (!g) continue;
            const dev = g.device;
            const key = `${dev.toString()}:${g.dtype}`;
            let bucket = buckets.get(key);
            if (!bucket) {
              bucket = { device: dev, grads: [] };
              buckets.set(key, bucket);
            }
            bucket.grads.push(g._native);
          }
        }

        // One fused kernel per device/dtype bucket, checking foundInf once per bucket
        for (const bucket of buckets.values()) {
          const foundInfTensor = full([1], 0.0, { dtype: 'float32', device: bucket.device });
          const invScaleTensor = full([1], invScaleVal, { dtype: 'float32', device: bucket.device });
          native.fast.ampUnscaleAndCheck(bucket.grads, foundInfTensor._native, invScaleTensor._native);
          if (foundInfTensor.item() !== 0) {
            this.foundInf = true;
          }
        }
      });
    });
  }

  step(optimizer: Optimizer): boolean {
    if (!this.enabled) {
      optimizer.step();
      return true;
    }
    if (!this.unscaled) {
      this.unscale_(optimizer);
    }
    if (this.foundInf) {
      return false;
    }
    optimizer.step();
    return true;
  }

  update(newScale?: number): void {
    if (!this.enabled) return;
    if (newScale !== undefined) {
      this._scale = newScale;
    } else if (this.foundInf) {
      this._scale *= this.backoffFactor;
      this.growthTracker = 0;
    } else {
      this.growthTracker += 1;
      if (this.growthTracker >= this.growthInterval) {
        this._scale *= this.growthFactor;
        this.growthTracker = 0;
      }
    }
    this.foundInf = false;
    this.unscaled = false;
  }

  stateDict(): GradScalerStateDict {
    return {
      scale: this._scale,
      growthFactor: this.growthFactor,
      backoffFactor: this.backoffFactor,
      growthInterval: this.growthInterval,
      growthTracker: this.growthTracker,
    };
  }

  loadStateDict(state: GradScalerStateDict): void {
    this._scale = state.scale;
    this.growthFactor = state.growthFactor;
    this.backoffFactor = state.backoffFactor;
    this.growthInterval = state.growthInterval;
    this.growthTracker = state.growthTracker;
    this.foundInf = false;
    this.unscaled = false;
  }
}
