import { noGrad } from './autograd.js';
import { Device, type DeviceLike } from './device.js';
import { keep, tidy } from './memory.js';
import { native, type NativeTensor } from './native.js';
import { full } from './ops.js';
import type { Optimizer } from './optim/optimizer.js';
import type { Tensor } from './tensor.js';

export interface GradScalerOptions {
  /**
   * Optional device (e.g. `'cuda'`) on which to keep scale, growth-tracker, and `foundInf`
   * tensors so `scale()` -> `stepNoSync()` -> `update()` run 100% on-device with zero CPU sync.
   */
  device?: DeviceLike;
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

interface FusedScalerCapableOptimizer {
  _stepWithFoundInf(foundInf: Tensor, gradScale?: Tensor): boolean;
}

function supportsZeroSyncFusedStep(opt: Optimizer): opt is Optimizer & FusedScalerCapableOptimizer {
  return typeof (opt as unknown as Partial<FusedScalerCapableOptimizer>)._stepWithFoundInf === 'function';
}

/**
 * Dynamic loss scaling for mixed-precision (FP16/BF16) training using libtorch's fused
 * `_amp_foreach_non_finite_check_and_unscale_` and `_amp_update_scale_` kernels.
 *
 * When constructed with `{ device: 'cuda' }` (or when calling `stepNoSync(optimizer)` with
 * fused `Adam`/`AdamW`), `scale()`, `unscale_()`, `stepNoSync()`, and `update()` execute
 * entirely on-device with **zero GPU-to-CPU synchronizations** per training step.
 */
export class GradScaler implements Disposable {
  private _scale: number;
  private growthFactor: number;
  private backoffFactor: number;
  private growthInterval: number;
  readonly enabled: boolean;
  private growthTracker = 0;
  private foundInf = false;
  private unscaled = false;

  // Optional persistent on-device state for zero-sync AMP steps
  private _device: Device | null = null;
  private _scaleTensor: Tensor | null = null;
  private _growthTrackerTensor: Tensor | null = null;
  private _foundInfTensor: Tensor | null = null;
  private _usedDeviceStep = false;

  constructor(opts: GradScalerOptions = {}) {
    this._scale = opts.initScale ?? 65536.0;
    this.growthFactor = opts.growthFactor ?? 2.0;
    this.backoffFactor = opts.backoffFactor ?? 0.5;
    this.growthInterval = opts.growthInterval ?? 2000;
    this.enabled = opts.enabled ?? true;
    if (opts.device !== undefined && this.enabled) {
      this._ensureDeviceTensors(new Device(opts.device));
    }
  }

  private _ensureDeviceTensors(dev: Device): void {
    if (this._scaleTensor && !this._scaleTensor.isDisposed) {
      return;
    }
    this._device = dev;
    noGrad(() => {
      this._scaleTensor = keep(full([1], this._scale, { dtype: 'float32', device: dev }));
      this._growthTrackerTensor = keep(
        full([1], this.growthTracker, { dtype: 'int32', device: dev })
      );
      this._foundInfTensor = keep(full([1], 0.0, { dtype: 'float32', device: dev }));
    });
  }

  getScale(): number {
    if (!this.enabled) return 1.0;
    if (this._scaleTensor && !this._scaleTensor.isDisposed) {
      this._scale = Number(this._scaleTensor.item());
    }
    return this._scale;
  }

  scale(loss: Tensor): Tensor {
    if (!this.enabled) return loss;
    if (this._scaleTensor && !this._scaleTensor.isDisposed) {
      const s = this._scaleTensor.device.equals(loss.device)
        ? this._scaleTensor
        : this._scaleTensor.to({ device: loss.device, nonBlocking: true });
      const out = loss.mul(s);
      if (s !== this._scaleTensor) s.dispose();
      return out;
    }
    return loss.mul(this._scale);
  }

  private _collectGradBuckets(
    optimizer: Optimizer
  ): Map<string, { device: Device; grads: NativeTensor[] }> {
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
    return buckets;
  }

  private _unscaleOnDevice(optimizer: Optimizer): boolean {
    let hadGrads = false;
    noGrad(() => {
      tidy(() => {
        const buckets = this._collectGradBuckets(optimizer);
        if (buckets.size === 0) return;
        hadGrads = true;
        const firstDev = buckets.values().next().value!.device;
        this._ensureDeviceTensors(firstDev);

        const scaleT = this._scaleTensor!;
        const masterFoundInf = this._foundInfTensor!;
        masterFoundInf.zero_();
        const invScaleMaster = full([1], 1.0, { dtype: 'float32', device: scaleT.device }).div(
          scaleT
        );

        for (const bucket of buckets.values()) {
          const sameDev = bucket.device.equals(scaleT.device);
          const invScale = sameDev
            ? invScaleMaster
            : invScaleMaster.to({ device: bucket.device, nonBlocking: true });
          const bucketFoundInf = sameDev
            ? masterFoundInf
            : full([1], 0.0, { dtype: 'float32', device: bucket.device });
          native.fast.ampUnscaleAndCheck(
            bucket.grads,
            bucketFoundInf._native,
            invScale._native
          );
          if (!sameDev) {
            masterFoundInf.add_(bucketFoundInf.to({ device: scaleT.device, nonBlocking: true }));
          }
        }
      });
    });
    return hadGrads;
  }

  unscale_(optimizer: Optimizer): void {
    if (!this.enabled || this.unscaled) return;
    this.unscaled = true;
    this.foundInf = false;

    if (this._scaleTensor && !this._scaleTensor.isDisposed) {
      this._usedDeviceStep = true;
      this._unscaleOnDevice(optimizer);
      return;
    }

    const invScaleVal = 1.0 / this._scale;
    noGrad(() => {
      tidy(() => {
        const buckets = this._collectGradBuckets(optimizer);
        for (const bucket of buckets.values()) {
          const foundInfTensor = full([1], 0.0, { dtype: 'float32', device: bucket.device });
          const invScaleTensor = full([1], invScaleVal, {
            dtype: 'float32',
            device: bucket.device,
          });
          native.fast.ampUnscaleAndCheck(
            bucket.grads,
            foundInfTensor._native,
            invScaleTensor._native
          );
          if (foundInfTensor.item() !== 0) {
            this.foundInf = true;
          }
        }
      });
    });
  }

  /**
   * Zero-sync optimizer step for fused `Adam`/`AdamW`: runs fused unscale+inf-check,
   * passes the on-device `foundInf` tensor directly into `at::_fused_adamw_`/`_fused_adam_`
   * (skipping the weight & moment update on-device if inf/nan was found), and leaves scale
   * update to `update()` via `at::_amp_update_scale_` — ZERO `.item()` CPU syncs!
   *
   * Falls back cleanly to standard `step(optimizer)` for non-fused optimizers.
   */
  stepNoSync(optimizer: Optimizer): void {
    if (!this.enabled) {
      optimizer.step();
      return;
    }
    if (!supportsZeroSyncFusedStep(optimizer)) {
      this.step(optimizer);
      return;
    }

    if (!this.unscaled) {
      this.unscaled = true;
      this._usedDeviceStep = true;
      const hadGrads = this._unscaleOnDevice(optimizer);
      if (!hadGrads) return;
    } else if (!this._foundInfTensor) {
      // unscale_() already ran on CPU path
      if (!this.foundInf) optimizer.step();
      return;
    }

    const ok = optimizer._stepWithFoundInf(this._foundInfTensor!);
    if (!ok) {
      // Optimizer had an unfused param group; fall back to CPU check
      this.foundInf = Number(this._foundInfTensor!.item()) !== 0;
      if (!this.foundInf) {
        optimizer.step();
      }
    }
  }

  step(optimizer: Optimizer): boolean {
    if (!this.enabled) {
      optimizer.step();
      return true;
    }
    if (!this.unscaled) {
      this.unscale_(optimizer);
    }
    if (this._usedDeviceStep && this._foundInfTensor && !this._foundInfTensor.isDisposed) {
      if (supportsZeroSyncFusedStep(optimizer)) {
        const ok = optimizer._stepWithFoundInf(this._foundInfTensor);
        const found = Number(this._foundInfTensor.item()) !== 0;
        this.foundInf = found;
        if (!ok && !found) {
          optimizer.step();
        }
        return !found;
      }
      this.foundInf = Number(this._foundInfTensor.item()) !== 0;
    }
    if (this.foundInf) {
      return false;
    }
    optimizer.step();
    return true;
  }

  update(newScale?: number): void {
    if (!this.enabled) return;

    if (
      this._scaleTensor &&
      !this._scaleTensor.isDisposed &&
      this._growthTrackerTensor &&
      this._foundInfTensor
    ) {
      noGrad(() => {
        if (newScale !== undefined) {
          this._scale = newScale;
          this._scaleTensor!.fill_(newScale);
        } else if (this._usedDeviceStep) {
          native.fast.ampUpdateScale(
            this._scaleTensor!._native,
            this._growthTrackerTensor!._native,
            this._foundInfTensor!._native,
            this.growthFactor,
            this.backoffFactor,
            this.growthInterval
          );
        }
      });
      this._usedDeviceStep = false;
      this.foundInf = false;
      this.unscaled = false;
      return;
    }

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
    const scale =
      this._scaleTensor && !this._scaleTensor.isDisposed
        ? Number(this._scaleTensor.item())
        : this._scale;
    const growthTracker =
      this._growthTrackerTensor && !this._growthTrackerTensor.isDisposed
        ? Number(this._growthTrackerTensor.item())
        : this.growthTracker;
    return {
      scale,
      growthFactor: this.growthFactor,
      backoffFactor: this.backoffFactor,
      growthInterval: this.growthInterval,
      growthTracker,
    };
  }

  loadStateDict(state: GradScalerStateDict): void {
    this._scale = state.scale;
    this.growthFactor = state.growthFactor;
    this.backoffFactor = state.backoffFactor;
    this.growthInterval = state.growthInterval;
    this.growthTracker = state.growthTracker;
    if (this._scaleTensor && !this._scaleTensor.isDisposed) {
      noGrad(() => {
        this._scaleTensor!.fill_(state.scale);
        this._growthTrackerTensor?.fill_(state.growthTracker);
      });
    }
    this.foundInf = false;
    this.unscaled = false;
    this._usedDeviceStep = false;
  }

  dispose(): void {
    if (this._scaleTensor && !this._scaleTensor.isDisposed) {
      this._scale = Number(this._scaleTensor.item());
      this._scaleTensor.dispose();
    }
    if (this._growthTrackerTensor && !this._growthTrackerTensor.isDisposed) {
      this.growthTracker = Number(this._growthTrackerTensor.item());
      this._growthTrackerTensor.dispose();
    }
    if (this._foundInfTensor && !this._foundInfTensor.isDisposed) {
      this._foundInfTensor.dispose();
    }
    this._scaleTensor = null;
    this._growthTrackerTensor = null;
    this._foundInfTensor = null;
    this._device = null;
  }

  [Symbol.dispose](): void {
    this.dispose();
  }
}
