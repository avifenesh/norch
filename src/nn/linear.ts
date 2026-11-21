import * as F from '../functional.js';
import { empty, type TensorOptions } from '../ops.js';
import type { Tensor } from '../tensor.js';
import { calculateFanInAndFanOut, kaimingUniform_, uniform_ } from './init.js';
import { Module } from './module.js';
import { Parameter } from './parameter.js';

export class Identity extends Module {
  forward(input: Tensor): Tensor {
    return input;
  }
}

export interface LinearOptions extends TensorOptions {
  bias?: boolean;
}

export class Linear extends Module {
  readonly inFeatures: number;
  readonly outFeatures: number;
  readonly weight: Parameter;
  readonly bias: Parameter | null;

  constructor(inFeatures: number, outFeatures: number, opts?: LinearOptions) {
    super();
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    const useBias = opts?.bias ?? true;
    const factoryOpts: TensorOptions = { dtype: opts?.dtype, device: opts?.device };

    const wInit = empty([outFeatures, inFeatures], factoryOpts);
    this.weight = new Parameter(wInit, true);
    wInit.dispose();

    if (useBias) {
      const bInit = empty([outFeatures], factoryOpts);
      this.bias = new Parameter(bInit, true);
      bInit.dispose();
    } else {
      this.bias = null;
    }

    this.resetParameters();
  }

  resetParameters(): void {
    kaimingUniform_(this.weight, Math.sqrt(5));
    if (this.bias) {
      const [fanIn] = calculateFanInAndFanOut(this.weight);
      const bound = fanIn > 0 ? 1 / Math.sqrt(fanIn) : 0;
      uniform_(this.bias, -bound, bound);
    }
  }

  forward(input: Tensor): Tensor {
    return F.linear(input, this.weight, this.bias);
  }
}
