import { noGrad } from './autograd.js';
import { tidy } from './memory.js';
import type { Optimizer } from './optim/optimizer.js';
import type { Tensor } from './tensor.js';

export interface GradScalerOptions {
  initScale?: number;
  growthFactor?: number;
  backoffFactor?: number;
  growthInterval?: number;
  enabled?: boolean;
}

/**
 * Dynamic loss scaling for mixed-precision (FP16/BF16) training,
 * matching torch.amp.GradScaler semantics.
 */
export class GradScaler {
  private _scale: number;
  private readonly growthFactor: number;
  private readonly backoffFactor: number;
  private readonly growthInterval: number;
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
    const invScale = 1.0 / this._scale;

    noGrad(() => {
      tidy(() => {
        for (const group of optimizer.paramGroups) {
          for (const p of group.params) {
            const g = p.grad;
            if (!g) continue;
            g.mul_(invScale);
            const allFinite = g.isfinite().all().item() !== 0;
            if (!allFinite) {
              this.foundInf = true;
            }
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
}
