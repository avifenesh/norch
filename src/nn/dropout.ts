import * as F from '../functional.js';
import type { Tensor } from '../tensor.js';
import { Module } from './module.js';

export class Dropout extends Module {
  readonly p: number;

  constructor(p = 0.5) {
    super();
    if (p < 0 || p > 1) {
      throw new RangeError(`Dropout probability must be in [0, 1], got ${p}`);
    }
    this.p = p;
  }

  forward(input: Tensor): Tensor {
    return F.dropout(input, this.p, this.training);
  }
}
