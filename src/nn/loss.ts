import * as F from '../functional.js';
import type { Reduction } from '../functional.js';
import type { Tensor } from '../tensor.js';
import { Module } from './module.js';

export class MSELoss extends Module {
  readonly reduction: Reduction;
  constructor(reduction: Reduction = 'mean') {
    super();
    this.reduction = reduction;
  }
  forward(input: Tensor, target: Tensor): Tensor {
    return F.mseLoss(input, target, this.reduction);
  }
}

export class L1Loss extends Module {
  readonly reduction: Reduction;
  constructor(reduction: Reduction = 'mean') {
    super();
    this.reduction = reduction;
  }
  forward(input: Tensor, target: Tensor): Tensor {
    return F.l1Loss(input, target, this.reduction);
  }
}

export class SmoothL1Loss extends Module {
  readonly reduction: Reduction;
  readonly beta: number;
  constructor(opts?: { reduction?: Reduction; beta?: number }) {
    super();
    this.reduction = opts?.reduction ?? 'mean';
    this.beta = opts?.beta ?? 1.0;
  }
  forward(input: Tensor, target: Tensor): Tensor {
    return F.smoothL1Loss(input, target, this.reduction, this.beta);
  }
}

export class CrossEntropyLoss extends Module {
  readonly weight: Tensor | null;
  readonly reduction: Reduction;
  readonly ignoreIndex: number;
  readonly labelSmoothing: number;

  constructor(opts?: {
    weight?: Tensor | null;
    reduction?: Reduction;
    ignoreIndex?: number;
    labelSmoothing?: number;
  }) {
    super();
    this.weight = opts?.weight ?? null;
    this.reduction = opts?.reduction ?? 'mean';
    this.ignoreIndex = opts?.ignoreIndex ?? -100;
    this.labelSmoothing = opts?.labelSmoothing ?? 0.0;
  }

  forward(input: Tensor, target: Tensor): Tensor {
    return F.crossEntropy(input, target, {
      weight: this.weight,
      reduction: this.reduction,
      ignoreIndex: this.ignoreIndex,
      labelSmoothing: this.labelSmoothing,
    });
  }
}

export class NLLLoss extends Module {
  readonly weight: Tensor | null;
  readonly reduction: Reduction;
  readonly ignoreIndex: number;

  constructor(opts?: { weight?: Tensor | null; reduction?: Reduction; ignoreIndex?: number }) {
    super();
    this.weight = opts?.weight ?? null;
    this.reduction = opts?.reduction ?? 'mean';
    this.ignoreIndex = opts?.ignoreIndex ?? -100;
  }

  forward(input: Tensor, target: Tensor): Tensor {
    return F.nllLoss(input, target, {
      weight: this.weight,
      reduction: this.reduction,
      ignoreIndex: this.ignoreIndex,
    });
  }
}

export class BCEWithLogitsLoss extends Module {
  readonly weight: Tensor | null;
  readonly posWeight: Tensor | null;
  readonly reduction: Reduction;

  constructor(opts?: {
    weight?: Tensor | null;
    posWeight?: Tensor | null;
    reduction?: Reduction;
  }) {
    super();
    this.weight = opts?.weight ?? null;
    this.posWeight = opts?.posWeight ?? null;
    this.reduction = opts?.reduction ?? 'mean';
  }

  forward(input: Tensor, target: Tensor): Tensor {
    return F.binaryCrossEntropyWithLogits(input, target, {
      weight: this.weight,
      posWeight: this.posWeight,
      reduction: this.reduction,
    });
  }
}

export class KLDivLoss extends Module {
  readonly reduction: Reduction;
  readonly logTarget: boolean;
  constructor(opts?: { reduction?: Reduction; logTarget?: boolean }) {
    super();
    this.reduction = opts?.reduction ?? 'mean';
    this.logTarget = opts?.logTarget ?? false;
  }
  forward(input: Tensor, target: Tensor): Tensor {
    return F.klDiv(input, target, this.reduction, this.logTarget);
  }
}
