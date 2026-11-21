import * as F from '../functional.js';
import type { Tensor } from '../tensor.js';
import { Module } from './module.js';

export class ReLU extends Module {
  forward(input: Tensor): Tensor {
    return F.relu(input);
  }
}

export class GELU extends Module {
  readonly approximate: 'none' | 'tanh';
  constructor(approximate: 'none' | 'tanh' = 'none') {
    super();
    this.approximate = approximate;
  }
  forward(input: Tensor): Tensor {
    return F.gelu(input, this.approximate);
  }
}

export class SiLU extends Module {
  forward(input: Tensor): Tensor {
    return F.silu(input);
  }
}

export class Sigmoid extends Module {
  forward(input: Tensor): Tensor {
    return F.sigmoid(input);
  }
}

export class Tanh extends Module {
  forward(input: Tensor): Tensor {
    return F.tanh(input);
  }
}

export class LeakyReLU extends Module {
  readonly negativeSlope: number;
  constructor(negativeSlope = 0.01) {
    super();
    this.negativeSlope = negativeSlope;
  }
  forward(input: Tensor): Tensor {
    return F.leakyRelu(input, this.negativeSlope);
  }
}

export class Softmax extends Module {
  readonly dim: number;
  constructor(dim = -1) {
    super();
    this.dim = dim;
  }
  forward(input: Tensor): Tensor {
    return F.softmax(input, this.dim);
  }
}

export class LogSoftmax extends Module {
  readonly dim: number;
  constructor(dim = -1) {
    super();
    this.dim = dim;
  }
  forward(input: Tensor): Tensor {
    return F.logSoftmax(input, this.dim);
  }
}
