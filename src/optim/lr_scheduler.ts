import type { Optimizer, ParamGroupBase } from './optimizer.js';

export abstract class LRScheduler<TGroup extends ParamGroupBase = ParamGroupBase> {
  protected readonly optimizer: Optimizer<TGroup>;
  protected readonly baseLrs: number[];
  protected lastEpoch = -1;
  protected _lastLr: number[] = [];

  constructor(optimizer: Optimizer<TGroup>) {
    this.optimizer = optimizer;
    this.baseLrs = optimizer.paramGroups.map((g) => g.lr);
  }

  abstract getLr(): number[];

  getLastLr(): number[] {
    return [...this._lastLr];
  }

  step(epoch?: number): void {
    this.lastEpoch = epoch === undefined ? this.lastEpoch + 1 : epoch;
    const lrs = this.getLr();
    this.optimizer.paramGroups.forEach((group, i) => {
      group.lr = lrs[i]!;
    });
    this._lastLr = lrs;
  }
}

export class StepLR<TGroup extends ParamGroupBase = ParamGroupBase> extends LRScheduler<TGroup> {
  readonly stepSize: number;
  readonly gamma: number;

  constructor(optimizer: Optimizer<TGroup>, stepSize: number, gamma = 0.1) {
    super(optimizer);
    this.stepSize = stepSize;
    this.gamma = gamma;
    this.step();
  }

  getLr(): number[] {
    if (this.lastEpoch <= 0) return [...this.baseLrs];
    const factor = Math.pow(this.gamma, Math.floor(this.lastEpoch / this.stepSize));
    return this.baseLrs.map((lr) => lr * factor);
  }
}

export class CosineAnnealingLR<
  TGroup extends ParamGroupBase = ParamGroupBase,
> extends LRScheduler<TGroup> {
  readonly tMax: number;
  readonly etaMin: number;

  constructor(optimizer: Optimizer<TGroup>, tMax: number, etaMin = 0) {
    super(optimizer);
    this.tMax = tMax;
    this.etaMin = etaMin;
    this.step();
  }

  getLr(): number[] {
    const progress = Math.min(this.lastEpoch, this.tMax) / Math.max(1, this.tMax);
    const cos = 0.5 * (1 + Math.cos(Math.PI * progress));
    return this.baseLrs.map((baseLr) => this.etaMin + (baseLr - this.etaMin) * cos);
  }
}

export class LinearLR<TGroup extends ParamGroupBase = ParamGroupBase> extends LRScheduler<TGroup> {
  readonly startFactor: number;
  readonly endFactor: number;
  readonly totalIters: number;

  constructor(
    optimizer: Optimizer<TGroup>,
    opts?: { startFactor?: number; endFactor?: number; totalIters?: number }
  ) {
    super(optimizer);
    this.startFactor = opts?.startFactor ?? 1 / 3;
    this.endFactor = opts?.endFactor ?? 1.0;
    this.totalIters = opts?.totalIters ?? 5;
    this.step();
  }

  getLr(): number[] {
    const t = Math.min(Math.max(0, this.lastEpoch), this.totalIters);
    const ratio = this.totalIters === 0 ? 1 : t / this.totalIters;
    const factor = this.startFactor + (this.endFactor - this.startFactor) * ratio;
    return this.baseLrs.map((lr) => lr * factor);
  }
}
