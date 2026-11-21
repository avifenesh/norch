import type { Optimizer, ParamGroupBase } from './optimizer.js';

export interface SchedulerStateDict {
  lastEpoch: number;
  baseLrs: number[];
  lastLr: number[];
  [key: string]: unknown;
}

export abstract class LRScheduler<TGroup extends ParamGroupBase = ParamGroupBase> {
  protected readonly optimizer: Optimizer<TGroup>;
  protected baseLrs: number[];
  protected lastEpoch = -1;
  protected _lastLr: number[] = [];

  constructor(optimizer: Optimizer<TGroup>) {
    this.optimizer = optimizer;
    this.baseLrs = optimizer.paramGroups.map((g) => {
      if (g.initialLr === undefined) {
        g.initialLr = g.lr;
      }
      return g.initialLr;
    });
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

  stateDict(): SchedulerStateDict {
    return {
      lastEpoch: this.lastEpoch,
      baseLrs: [...this.baseLrs],
      lastLr: [...this._lastLr],
    };
  }

  loadStateDict(state: SchedulerStateDict): void {
    this.lastEpoch = state.lastEpoch;
    this.baseLrs = [...state.baseLrs];
    this._lastLr = [...state.lastLr];
    this.optimizer.paramGroups.forEach((group, i) => {
      if (this._lastLr[i] !== undefined) {
        group.lr = this._lastLr[i]!;
      }
    });
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

export interface OneCycleLROptions {
  maxLr: number | number[];
  totalSteps: number;
  pctStart?: number;
  divFactor?: number;
  finalDivFactor?: number;
  annealStrategy?: 'cos' | 'linear';
}

export class OneCycleLR<TGroup extends ParamGroupBase = ParamGroupBase> extends LRScheduler<TGroup> {
  readonly maxLrs: number[];
  readonly totalSteps: number;
  readonly pctStart: number;
  readonly divFactor: number;
  readonly finalDivFactor: number;
  readonly annealStrategy: 'cos' | 'linear';

  constructor(optimizer: Optimizer<TGroup>, opts: OneCycleLROptions) {
    super(optimizer);
    const maxArr = Array.isArray(opts.maxLr)
      ? opts.maxLr
      : optimizer.paramGroups.map(() => opts.maxLr as number);
    if (maxArr.length !== optimizer.paramGroups.length) {
      throw new RangeError('maxLr length must match optimizer paramGroups length');
    }
    this.maxLrs = maxArr;
    this.totalSteps = Math.max(1, opts.totalSteps);
    this.pctStart = opts.pctStart ?? 0.3;
    this.divFactor = opts.divFactor ?? 25.0;
    this.finalDivFactor = opts.finalDivFactor ?? 1e4;
    this.annealStrategy = opts.annealStrategy ?? 'cos';
    this.step();
  }

  private anneal(start: number, end: number, pct: number): number {
    if (this.annealStrategy === 'linear') {
      return start + (end - start) * pct;
    }
    return end + ((start - end) / 2.0) * (Math.cos(Math.PI * pct) + 1.0);
  }

  getLr(): number[] {
    const stepNum = Math.min(this.lastEpoch, this.totalSteps - 1);
    const upSteps = Math.max(1, Math.floor(this.totalSteps * this.pctStart));
    const downSteps = Math.max(1, this.totalSteps - upSteps);

    return this.maxLrs.map((maxLr) => {
      const initLr = maxLr / this.divFactor;
      const minLr = initLr / this.finalDivFactor;
      if (stepNum <= upSteps) {
        return this.anneal(initLr, maxLr, stepNum / upSteps);
      }
      return this.anneal(maxLr, minLr, (stepNum - upSteps) / downSteps);
    });
  }
}

export class SequentialLR<
  TGroup extends ParamGroupBase = ParamGroupBase,
> extends LRScheduler<TGroup> {
  readonly schedulers: LRScheduler<TGroup>[];
  readonly milestones: number[];

  constructor(
    optimizer: Optimizer<TGroup>,
    schedulers: LRScheduler<TGroup>[],
    milestones: number[]
  ) {
    super(optimizer);
    if (milestones.length !== schedulers.length - 1) {
      throw new RangeError('milestones length must be schedulers.length - 1');
    }
    this.schedulers = schedulers;
    this.milestones = milestones;
    this.lastEpoch = 0;
    this._lastLr = schedulers[0]!.getLastLr();
    this.optimizer.paramGroups.forEach((g, i) => {
      g.lr = this._lastLr[i]!;
    });
  }

  getLr(): number[] {
    let idx = 0;
    while (idx < this.milestones.length && this.lastEpoch >= this.milestones[idx]!) {
      idx += 1;
    }
    const localEpoch = idx === 0 ? this.lastEpoch : this.lastEpoch - this.milestones[idx - 1]!;
    const active = this.schedulers[idx]!;
    active.step(localEpoch);
    return active.getLastLr();
  }
}

export interface ReduceLROnPlateauOptions {
  mode?: 'min' | 'max';
  factor?: number;
  patience?: number;
  threshold?: number;
  thresholdMode?: 'rel' | 'abs';
  cooldown?: number;
  minLr?: number;
}

export class ReduceLROnPlateau<TGroup extends ParamGroupBase = ParamGroupBase> {
  private readonly optimizer: Optimizer<TGroup>;
  readonly mode: 'min' | 'max';
  readonly factor: number;
  readonly patience: number;
  readonly threshold: number;
  readonly thresholdMode: 'rel' | 'abs';
  readonly cooldown: number;
  readonly minLr: number;
  private best: number;
  private numBadEpochs = 0;
  private cooldownCounter = 0;
  private _lastLr: number[];

  constructor(optimizer: Optimizer<TGroup>, opts: ReduceLROnPlateauOptions = {}) {
    this.optimizer = optimizer;
    this.mode = opts.mode ?? 'min';
    this.factor = opts.factor ?? 0.1;
    this.patience = opts.patience ?? 10;
    this.threshold = opts.threshold ?? 1e-4;
    this.thresholdMode = opts.thresholdMode ?? 'rel';
    this.cooldown = opts.cooldown ?? 0;
    this.minLr = opts.minLr ?? 0;
    this.best = this.mode === 'min' ? Infinity : -Infinity;
    this._lastLr = optimizer.paramGroups.map((g) => g.lr);
  }

  getLastLr(): number[] {
    return [...this._lastLr];
  }

  private isBetter(current: number): boolean {
    if (this.mode === 'min') {
      if (this.thresholdMode === 'rel') {
        return current < this.best * (1 - this.threshold);
      }
      return current < this.best - this.threshold;
    }
    if (this.thresholdMode === 'rel') {
      return current > this.best * (1 + this.threshold);
    }
    return current > this.best + this.threshold;
  }

  step(metric: number): void {
    if (this.isBetter(metric)) {
      this.best = metric;
      this.numBadEpochs = 0;
    } else {
      this.numBadEpochs += 1;
    }

    if (this.cooldownCounter > 0) {
      this.cooldownCounter -= 1;
      this.numBadEpochs = 0;
    }

    if (this.numBadEpochs > this.patience) {
      for (const group of this.optimizer.paramGroups) {
        group.lr = Math.max(group.lr * this.factor, this.minLr);
      }
      this.cooldownCounter = this.cooldown;
      this.numBadEpochs = 0;
    }
    this._lastLr = this.optimizer.paramGroups.map((g) => g.lr);
  }

  stateDict(): Record<string, unknown> {
    return {
      best: this.best,
      numBadEpochs: this.numBadEpochs,
      cooldownCounter: this.cooldownCounter,
      lastLr: [...this._lastLr],
    };
  }

  loadStateDict(state: Record<string, unknown>): void {
    if (typeof state['best'] === 'number') this.best = state['best'];
    if (typeof state['numBadEpochs'] === 'number') this.numBadEpochs = state['numBadEpochs'];
    if (typeof state['cooldownCounter'] === 'number') {
      this.cooldownCounter = state['cooldownCounter'];
    }
    if (Array.isArray(state['lastLr'])) {
      this._lastLr = [...(state['lastLr'] as number[])];
      this.optimizer.paramGroups.forEach((g, i) => {
        if (this._lastLr[i] !== undefined) g.lr = this._lastLr[i]!;
      });
    }
  }
}

export class ExponentialLR<
  TGroup extends ParamGroupBase = ParamGroupBase,
> extends LRScheduler<TGroup> {
  readonly gamma: number;

  constructor(optimizer: Optimizer<TGroup>, gamma: number) {
    super(optimizer);
    this.gamma = gamma;
    this.step();
  }

  getLr(): number[] {
    const factor = Math.pow(this.gamma, Math.max(0, this.lastEpoch));
    return this.baseLrs.map((lr) => lr * factor);
  }
}

export class MultiStepLR<
  TGroup extends ParamGroupBase = ParamGroupBase,
> extends LRScheduler<TGroup> {
  readonly milestones: number[];
  readonly gamma: number;

  constructor(optimizer: Optimizer<TGroup>, milestones: number[], gamma = 0.1) {
    super(optimizer);
    this.milestones = [...milestones].sort((a, b) => a - b);
    this.gamma = gamma;
    this.step();
  }

  getLr(): number[] {
    let passed = 0;
    for (const m of this.milestones) {
      if (this.lastEpoch >= m) passed += 1;
    }
    const factor = Math.pow(this.gamma, passed);
    return this.baseLrs.map((lr) => lr * factor);
  }
}

export class LambdaLR<
  TGroup extends ParamGroupBase = ParamGroupBase,
> extends LRScheduler<TGroup> {
  readonly lrLambdas: ((epoch: number) => number)[];

  constructor(
    optimizer: Optimizer<TGroup>,
    lrLambda: ((epoch: number) => number) | ((epoch: number) => number)[]
  ) {
    super(optimizer);
    this.lrLambdas = Array.isArray(lrLambda)
      ? lrLambda
      : optimizer.paramGroups.map(() => lrLambda);
    if (this.lrLambdas.length !== optimizer.paramGroups.length) {
      throw new RangeError('lrLambda length must match optimizer paramGroups length');
    }
    this.step();
  }

  getLr(): number[] {
    return this.baseLrs.map((base, i) => base * this.lrLambdas[i]!(Math.max(0, this.lastEpoch)));
  }
}

export class CosineAnnealingWarmRestarts<
  TGroup extends ParamGroupBase = ParamGroupBase,
> extends LRScheduler<TGroup> {
  readonly t0: number;
  readonly tMult: number;
  readonly etaMin: number;

  constructor(optimizer: Optimizer<TGroup>, t0: number, tMult = 1, etaMin = 0) {
    super(optimizer);
    if (t0 <= 0) throw new RangeError('t0 must be > 0');
    if (tMult < 1) throw new RangeError('tMult must be >= 1');
    this.t0 = t0;
    this.tMult = tMult;
    this.etaMin = etaMin;
    this.step();
  }

  getLr(): number[] {
    const epoch = Math.max(0, this.lastEpoch);
    let tCur = epoch;
    let tI = this.t0;
    if (this.tMult === 1) {
      tCur = epoch % this.t0;
    } else {
      while (tCur >= tI) {
        tCur -= tI;
        tI *= this.tMult;
      }
    }
    const cos = 0.5 * (1 + Math.cos((Math.PI * tCur) / tI));
    return this.baseLrs.map((base) => this.etaMin + (base - this.etaMin) * cos);
  }
}

export class ChainedScheduler<TGroup extends ParamGroupBase = ParamGroupBase> {
  readonly schedulers: LRScheduler<TGroup>[];

  constructor(schedulers: LRScheduler<TGroup>[]) {
    this.schedulers = schedulers;
  }

  step(): void {
    for (const sched of this.schedulers) {
      sched.step();
    }
  }

  getLastLr(): number[] {
    return this.schedulers.length > 0 ? this.schedulers[this.schedulers.length - 1]!.getLastLr() : [];
  }
}
