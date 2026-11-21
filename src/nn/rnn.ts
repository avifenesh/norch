import { native } from '../native.js';
import { empty, zeros, type TensorOptions } from '../ops.js';
import { Tensor, wrapTensor } from '../tensor.js';
import { uniform_ } from './init.js';
import { Module } from './module.js';
import { Parameter } from './parameter.js';

export interface RNNLayerOptions extends TensorOptions {
  numLayers?: number;
  bias?: boolean;
  batchFirst?: boolean;
  dropout?: number;
  bidirectional?: boolean;
}

abstract class RNNBase extends Module {
  readonly inputSize: number;
  readonly hiddenSize: number;
  readonly numLayers: number;
  readonly bias: boolean;
  readonly batchFirst: boolean;
  readonly dropout: number;
  readonly bidirectional: boolean;
  protected readonly gateSize: number;
  protected readonly flatParams: Parameter[] = [];

  constructor(
    inputSize: number,
    hiddenSize: number,
    gateMultiplier: number,
    opts?: RNNLayerOptions
  ) {
    super();
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.numLayers = opts?.numLayers ?? 1;
    this.bias = opts?.bias ?? true;
    this.batchFirst = opts?.batchFirst ?? false;
    this.dropout = opts?.dropout ?? 0.0;
    this.bidirectional = opts?.bidirectional ?? false;
    this.gateSize = gateMultiplier * hiddenSize;

    const factoryOpts: TensorOptions = { dtype: opts?.dtype, device: opts?.device };
    const numDirs = this.bidirectional ? 2 : 1;
    const k = 1.0 / Math.sqrt(hiddenSize);

    for (let layer = 0; layer < this.numLayers; layer++) {
      for (let dir = 0; dir < numDirs; dir++) {
        const layerInputSize = layer === 0 ? inputSize : hiddenSize * numDirs;
        const suffix = dir === 1 ? '_reverse' : '';

        const wIhInit = empty([this.gateSize, layerInputSize], factoryOpts);
        const wHhInit = empty([this.gateSize, hiddenSize], factoryOpts);
        const wIh = new Parameter(wIhInit, true);
        const wHh = new Parameter(wHhInit, true);
        wIhInit.dispose();
        wHhInit.dispose();
        uniform_(wIh, -k, k);
        uniform_(wHh, -k, k);

        this.registerParameter(`weight_ih_l${layer}${suffix}`, wIh);
        this.registerParameter(`weight_hh_l${layer}${suffix}`, wHh);
        this.flatParams.push(wIh, wHh);

        if (this.bias) {
          const bIhInit = empty([this.gateSize], factoryOpts);
          const bHhInit = empty([this.gateSize], factoryOpts);
          const bIh = new Parameter(bIhInit, true);
          const bHh = new Parameter(bHhInit, true);
          bIhInit.dispose();
          bHhInit.dispose();
          uniform_(bIh, -k, k);
          uniform_(bHh, -k, k);

          this.registerParameter(`bias_ih_l${layer}${suffix}`, bIh);
          this.registerParameter(`bias_hh_l${layer}${suffix}`, bHh);
          this.flatParams.push(bIh, bHh);
        }
      }
    }
  }

  protected getOrderedNativeParams(): Parameter[] {
    const out: Parameter[] = [];
    const numDirs = this.bidirectional ? 2 : 1;
    for (let layer = 0; layer < this.numLayers; layer++) {
      for (let dir = 0; dir < numDirs; dir++) {
        const suffix = dir === 1 ? '_reverse' : '';
        out.push(this._parameters.get(`weight_ih_l${layer}${suffix}`)!);
        out.push(this._parameters.get(`weight_hh_l${layer}${suffix}`)!);
        if (this.bias) {
          out.push(this._parameters.get(`bias_ih_l${layer}${suffix}`)!);
          out.push(this._parameters.get(`bias_hh_l${layer}${suffix}`)!);
        }
      }
    }
    return out;
  }

  protected defaultHidden(input: Tensor): Tensor {
    const numDirs = this.bidirectional ? 2 : 1;
    const isBatched = input.ndim === 3;
    const batch = isBatched ? (this.batchFirst ? input.size(0) : input.size(1)) : 1;
    const shape = isBatched
      ? [this.numLayers * numDirs, batch, this.hiddenSize]
      : [this.numLayers * numDirs, this.hiddenSize];
    return zeros(shape, { dtype: input.dtype, device: input.device });
  }
}

export class LSTM extends RNNBase {
  constructor(inputSize: number, hiddenSize: number, opts?: RNNLayerOptions) {
    super(inputSize, hiddenSize, 4, opts);
  }

  forward(
    input: Tensor,
    hx?: [Tensor, Tensor]
  ): { output: Tensor; hn: Tensor; cn: Tensor } {
    const h0 = hx ? hx[0] : this.defaultHidden(input);
    const c0 = hx ? hx[1] : this.defaultHidden(input);
    const params = this.getOrderedNativeParams().map((p) => p._native);
    const res = native.fast.lstm(
      input._native,
      [h0._native, c0._native],
      params,
      this.bias,
      this.numLayers,
      this.dropout,
      this.training,
      this.bidirectional,
      this.batchFirst
    );
    if (!hx) {
      h0.dispose();
      c0.dispose();
    }
    return {
      output: wrapTensor(res.output),
      hn: wrapTensor(res.hn),
      cn: wrapTensor(res.cn),
    };
  }
}

export class GRU extends RNNBase {
  constructor(inputSize: number, hiddenSize: number, opts?: RNNLayerOptions) {
    super(inputSize, hiddenSize, 3, opts);
  }

  forward(input: Tensor, hx?: Tensor): { output: Tensor; hn: Tensor } {
    const h0 = hx ?? this.defaultHidden(input);
    const params = this.getOrderedNativeParams().map((p) => p._native);
    const res = native.fast.gru(
      input._native,
      h0._native,
      params,
      this.bias,
      this.numLayers,
      this.dropout,
      this.training,
      this.bidirectional,
      this.batchFirst
    );
    if (!hx) h0.dispose();
    return { output: wrapTensor(res.output), hn: wrapTensor(res.hn) };
  }
}

export interface BasicRNNOptions extends RNNLayerOptions {
  nonlinearity?: 'tanh' | 'relu';
}

export class RNN extends RNNBase {
  readonly nonlinearity: 'tanh' | 'relu';

  constructor(inputSize: number, hiddenSize: number, opts?: BasicRNNOptions) {
    super(inputSize, hiddenSize, 1, opts);
    this.nonlinearity = opts?.nonlinearity ?? 'tanh';
  }

  forward(input: Tensor, hx?: Tensor): { output: Tensor; hn: Tensor } {
    const h0 = hx ?? this.defaultHidden(input);
    const params = this.getOrderedNativeParams().map((p) => p._native);
    const res = native.fast.rnn(
      input._native,
      h0._native,
      params,
      this.bias,
      this.numLayers,
      this.dropout,
      this.training,
      this.bidirectional,
      this.batchFirst,
      this.nonlinearity
    );
    if (!hx) h0.dispose();
    return { output: wrapTensor(res.output), hn: wrapTensor(res.hn) };
  }
}
