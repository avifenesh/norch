import * as F from '../functional.js';
import { empty, type TensorOptions } from '../ops.js';
import type { Tensor } from '../tensor.js';
import { calculateFanInAndFanOut, kaimingUniform_, uniform_ } from './init.js';
import { Module } from './module.js';
import { Parameter } from './parameter.js';

function pair(v: number | [number, number]): [number, number] {
  return Array.isArray(v) ? v : [v, v];
}

export interface Conv1dLayerOptions extends TensorOptions {
  stride?: number;
  padding?: number;
  dilation?: number;
  groups?: number;
  bias?: boolean;
}

export class Conv1d extends Module {
  readonly inChannels: number;
  readonly outChannels: number;
  readonly kernelSize: number;
  readonly stride: number;
  readonly padding: number;
  readonly dilation: number;
  readonly groups: number;
  readonly weight: Parameter;
  readonly bias: Parameter | null;

  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: number,
    opts?: Conv1dLayerOptions
  ) {
    super();
    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.kernelSize = kernelSize;
    this.stride = opts?.stride ?? 1;
    this.padding = opts?.padding ?? 0;
    this.dilation = opts?.dilation ?? 1;
    this.groups = opts?.groups ?? 1;
    const useBias = opts?.bias ?? true;
    const factoryOpts: TensorOptions = { dtype: opts?.dtype, device: opts?.device };

    const wInit = empty([outChannels, inChannels / this.groups, kernelSize], factoryOpts);
    this.weight = new Parameter(wInit, true);
    wInit.dispose();

    if (useBias) {
      const bInit = empty([outChannels], factoryOpts);
      this.bias = new Parameter(bInit, true);
      bInit.dispose();
    } else {
      this.bias = null;
    }

    kaimingUniform_(this.weight, Math.sqrt(5));
    if (this.bias) {
      const [fanIn] = calculateFanInAndFanOut(this.weight);
      const bound = fanIn > 0 ? 1 / Math.sqrt(fanIn) : 0;
      uniform_(this.bias, -bound, bound);
    }
  }

  forward(input: Tensor): Tensor {
    return F.conv1d(input, this.weight, {
      bias: this.bias,
      stride: [this.stride],
      padding: [this.padding],
      dilation: [this.dilation],
      groups: this.groups,
    });
  }
}

export interface Conv2dLayerOptions extends TensorOptions {
  stride?: number | [number, number];
  padding?: number | [number, number];
  dilation?: number | [number, number];
  groups?: number;
  bias?: boolean;
}

export class Conv2d extends Module {
  readonly inChannels: number;
  readonly outChannels: number;
  readonly kernelSize: [number, number];
  readonly stride: [number, number];
  readonly padding: [number, number];
  readonly dilation: [number, number];
  readonly groups: number;
  readonly weight: Parameter;
  readonly bias: Parameter | null;

  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: number | [number, number],
    opts?: Conv2dLayerOptions
  ) {
    super();
    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.kernelSize = pair(kernelSize);
    this.stride = pair(opts?.stride ?? 1);
    this.padding = pair(opts?.padding ?? 0);
    this.dilation = pair(opts?.dilation ?? 1);
    this.groups = opts?.groups ?? 1;
    const useBias = opts?.bias ?? true;
    const factoryOpts: TensorOptions = { dtype: opts?.dtype, device: opts?.device };

    const wInit = empty(
      [outChannels, inChannels / this.groups, this.kernelSize[0], this.kernelSize[1]],
      factoryOpts
    );
    this.weight = new Parameter(wInit, true);
    wInit.dispose();

    if (useBias) {
      const bInit = empty([outChannels], factoryOpts);
      this.bias = new Parameter(bInit, true);
      bInit.dispose();
    } else {
      this.bias = null;
    }

    kaimingUniform_(this.weight, Math.sqrt(5));
    if (this.bias) {
      const [fanIn] = calculateFanInAndFanOut(this.weight);
      const bound = fanIn > 0 ? 1 / Math.sqrt(fanIn) : 0;
      uniform_(this.bias, -bound, bound);
    }
  }

  forward(input: Tensor): Tensor {
    return F.conv2d(input, this.weight, {
      bias: this.bias,
      stride: this.stride,
      padding: this.padding,
      dilation: this.dilation,
      groups: this.groups,
    });
  }
}

export class MaxPool2d extends Module {
  readonly kernelSize: [number, number];
  readonly stride?: [number, number];
  readonly padding: [number, number];
  readonly dilation: [number, number];
  readonly ceilMode: boolean;

  constructor(
    kernelSize: number | [number, number],
    opts?: {
      stride?: number | [number, number];
      padding?: number | [number, number];
      dilation?: number | [number, number];
      ceilMode?: boolean;
    }
  ) {
    super();
    this.kernelSize = pair(kernelSize);
    this.stride = opts?.stride !== undefined ? pair(opts.stride) : undefined;
    this.padding = pair(opts?.padding ?? 0);
    this.dilation = pair(opts?.dilation ?? 1);
    this.ceilMode = opts?.ceilMode ?? false;
  }

  forward(input: Tensor): Tensor {
    return F.maxPool2d(input, this.kernelSize, {
      stride: this.stride,
      padding: this.padding,
      dilation: this.dilation,
      ceilMode: this.ceilMode,
    });
  }
}

export class AvgPool2d extends Module {
  readonly kernelSize: [number, number];
  readonly stride?: [number, number];
  readonly padding: [number, number];

  constructor(
    kernelSize: number | [number, number],
    opts?: { stride?: number | [number, number]; padding?: number | [number, number] }
  ) {
    super();
    this.kernelSize = pair(kernelSize);
    this.stride = opts?.stride !== undefined ? pair(opts.stride) : undefined;
    this.padding = pair(opts?.padding ?? 0);
  }

  forward(input: Tensor): Tensor {
    return F.avgPool2d(input, this.kernelSize, {
      stride: this.stride,
      padding: this.padding,
    });
  }
}

export class AdaptiveAvgPool2d extends Module {
  readonly outputSize: [number, number];
  constructor(outputSize: number | [number, number]) {
    super();
    this.outputSize = pair(outputSize);
  }
  forward(input: Tensor): Tensor {
    return F.adaptiveAvgPool2d(input, this.outputSize);
  }
}

export class Flatten extends Module {
  readonly startDim: number;
  readonly endDim: number;
  constructor(startDim = 1, endDim = -1) {
    super();
    this.startDim = startDim;
    this.endDim = endDim;
  }
  forward(input: Tensor): Tensor {
    return input.flatten(this.startDim, this.endDim);
  }
}

function triple(v: number | [number, number, number]): [number, number, number] {
  return Array.isArray(v) ? v : [v, v, v];
}

export interface Conv3dLayerOptions extends TensorOptions {
  stride?: number | [number, number, number];
  padding?: number | [number, number, number];
  dilation?: number | [number, number, number];
  groups?: number;
  bias?: boolean;
}

export class Conv3d extends Module {
  readonly inChannels: number;
  readonly outChannels: number;
  readonly kernelSize: [number, number, number];
  readonly stride: [number, number, number];
  readonly padding: [number, number, number];
  readonly dilation: [number, number, number];
  readonly groups: number;
  readonly weight: Parameter;
  readonly bias: Parameter | null;

  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: number | [number, number, number],
    opts?: Conv3dLayerOptions
  ) {
    super();
    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.kernelSize = triple(kernelSize);
    this.stride = triple(opts?.stride ?? 1);
    this.padding = triple(opts?.padding ?? 0);
    this.dilation = triple(opts?.dilation ?? 1);
    this.groups = opts?.groups ?? 1;
    const useBias = opts?.bias ?? true;
    const factoryOpts: TensorOptions = { dtype: opts?.dtype, device: opts?.device };

    const wInit = empty(
      [
        outChannels,
        inChannels / this.groups,
        this.kernelSize[0],
        this.kernelSize[1],
        this.kernelSize[2],
      ],
      factoryOpts
    );
    this.weight = new Parameter(wInit, true);
    wInit.dispose();

    if (useBias) {
      const bInit = empty([outChannels], factoryOpts);
      this.bias = new Parameter(bInit, true);
      bInit.dispose();
    } else {
      this.bias = null;
    }

    kaimingUniform_(this.weight, Math.sqrt(5));
    if (this.bias) {
      const [fanIn] = calculateFanInAndFanOut(this.weight);
      const bound = fanIn > 0 ? 1 / Math.sqrt(fanIn) : 0;
      uniform_(this.bias, -bound, bound);
    }
  }

  forward(input: Tensor): Tensor {
    return F.conv3d(input, this.weight, {
      bias: this.bias,
      stride: this.stride,
      padding: this.padding,
      dilation: this.dilation,
      groups: this.groups,
    });
  }
}

export interface ConvTranspose1dLayerOptions extends TensorOptions {
  stride?: number;
  padding?: number;
  outputPadding?: number;
  groups?: number;
  dilation?: number;
  bias?: boolean;
}

export class ConvTranspose1d extends Module {
  readonly inChannels: number;
  readonly outChannels: number;
  readonly kernelSize: number;
  readonly stride: number;
  readonly padding: number;
  readonly outputPadding: number;
  readonly groups: number;
  readonly dilation: number;
  readonly weight: Parameter;
  readonly bias: Parameter | null;

  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: number,
    opts?: ConvTranspose1dLayerOptions
  ) {
    super();
    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.kernelSize = kernelSize;
    this.stride = opts?.stride ?? 1;
    this.padding = opts?.padding ?? 0;
    this.outputPadding = opts?.outputPadding ?? 0;
    this.groups = opts?.groups ?? 1;
    this.dilation = opts?.dilation ?? 1;
    const useBias = opts?.bias ?? true;
    const factoryOpts: TensorOptions = { dtype: opts?.dtype, device: opts?.device };

    const wInit = empty([inChannels, outChannels / this.groups, kernelSize], factoryOpts);
    this.weight = new Parameter(wInit, true);
    wInit.dispose();

    if (useBias) {
      const bInit = empty([outChannels], factoryOpts);
      this.bias = new Parameter(bInit, true);
      bInit.dispose();
    } else {
      this.bias = null;
    }

    kaimingUniform_(this.weight, Math.sqrt(5));
    if (this.bias) {
      const [fanIn] = calculateFanInAndFanOut(this.weight);
      const bound = fanIn > 0 ? 1 / Math.sqrt(fanIn) : 0;
      uniform_(this.bias, -bound, bound);
    }
  }

  forward(input: Tensor): Tensor {
    return F.convTranspose1d(input, this.weight, {
      bias: this.bias,
      stride: [this.stride],
      padding: [this.padding],
      outputPadding: [this.outputPadding],
      groups: this.groups,
      dilation: [this.dilation],
    });
  }
}

export interface ConvTranspose2dLayerOptions extends TensorOptions {
  stride?: number | [number, number];
  padding?: number | [number, number];
  outputPadding?: number | [number, number];
  groups?: number;
  dilation?: number | [number, number];
  bias?: boolean;
}

export class ConvTranspose2d extends Module {
  readonly inChannels: number;
  readonly outChannels: number;
  readonly kernelSize: [number, number];
  readonly stride: [number, number];
  readonly padding: [number, number];
  readonly outputPadding: [number, number];
  readonly groups: number;
  readonly dilation: [number, number];
  readonly weight: Parameter;
  readonly bias: Parameter | null;

  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: number | [number, number],
    opts?: ConvTranspose2dLayerOptions
  ) {
    super();
    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.kernelSize = pair(kernelSize);
    this.stride = pair(opts?.stride ?? 1);
    this.padding = pair(opts?.padding ?? 0);
    this.outputPadding = pair(opts?.outputPadding ?? 0);
    this.groups = opts?.groups ?? 1;
    this.dilation = pair(opts?.dilation ?? 1);
    const useBias = opts?.bias ?? true;
    const factoryOpts: TensorOptions = { dtype: opts?.dtype, device: opts?.device };

    const wInit = empty(
      [inChannels, outChannels / this.groups, this.kernelSize[0], this.kernelSize[1]],
      factoryOpts
    );
    this.weight = new Parameter(wInit, true);
    wInit.dispose();

    if (useBias) {
      const bInit = empty([outChannels], factoryOpts);
      this.bias = new Parameter(bInit, true);
      bInit.dispose();
    } else {
      this.bias = null;
    }

    kaimingUniform_(this.weight, Math.sqrt(5));
    if (this.bias) {
      const [fanIn] = calculateFanInAndFanOut(this.weight);
      const bound = fanIn > 0 ? 1 / Math.sqrt(fanIn) : 0;
      uniform_(this.bias, -bound, bound);
    }
  }

  forward(input: Tensor): Tensor {
    return F.convTranspose2d(input, this.weight, {
      bias: this.bias,
      stride: this.stride,
      padding: this.padding,
      outputPadding: this.outputPadding,
      groups: this.groups,
      dilation: this.dilation,
    });
  }
}

export class Upsample extends Module {
  readonly opts: F.InterpolateOptions;
  constructor(opts: F.InterpolateOptions) {
    super();
    this.opts = opts;
  }
  forward(input: Tensor): Tensor {
    return F.interpolate(input, this.opts);
  }
}
