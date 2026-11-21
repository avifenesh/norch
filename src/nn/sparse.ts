import { noGrad } from '../autograd.js';
import * as F from '../functional.js';
import { empty, type TensorOptions } from '../ops.js';
import type { Tensor } from '../tensor.js';
import { normal_ } from './init.js';
import { Module } from './module.js';
import { Parameter } from './parameter.js';

export interface EmbeddingLayerOptions extends TensorOptions {
  paddingIdx?: number;
  scaleGradByFreq?: boolean;
  sparse?: boolean;
}

export class Embedding extends Module {
  readonly numEmbeddings: number;
  readonly embeddingDim: number;
  readonly paddingIdx?: number;
  readonly scaleGradByFreq: boolean;
  readonly sparse: boolean;
  readonly weight: Parameter;

  constructor(numEmbeddings: number, embeddingDim: number, opts?: EmbeddingLayerOptions) {
    super();
    this.numEmbeddings = numEmbeddings;
    this.embeddingDim = embeddingDim;
    this.paddingIdx = opts?.paddingIdx;
    this.scaleGradByFreq = opts?.scaleGradByFreq ?? false;
    this.sparse = opts?.sparse ?? false;
    const factoryOpts: TensorOptions = { dtype: opts?.dtype, device: opts?.device };

    const wInit = empty([numEmbeddings, embeddingDim], factoryOpts);
    this.weight = new Parameter(wInit, true);
    wInit.dispose();

    normal_(this.weight, 0, 1);
    if (this.paddingIdx !== undefined) {
      noGrad(() => {
        this.weight.indexPut_([this.paddingIdx!], 0);
      });
    }
  }

  forward(indices: Tensor): Tensor {
    return F.embedding(indices, this.weight, {
      paddingIdx: this.paddingIdx,
      scaleGradByFreq: this.scaleGradByFreq,
      sparse: this.sparse,
    });
  }
}

export interface EmbeddingBagLayerOptions extends TensorOptions {
  mode?: 'sum' | 'mean' | 'max';
  scaleGradByFreq?: boolean;
  sparse?: boolean;
  includeLastOffset?: boolean;
  paddingIdx?: number;
}

export class EmbeddingBag extends Module {
  readonly numEmbeddings: number;
  readonly embeddingDim: number;
  readonly mode: 'sum' | 'mean' | 'max';
  readonly scaleGradByFreq: boolean;
  readonly sparse: boolean;
  readonly includeLastOffset: boolean;
  readonly paddingIdx?: number;
  readonly weight: Parameter;

  constructor(numEmbeddings: number, embeddingDim: number, opts?: EmbeddingBagLayerOptions) {
    super();
    this.numEmbeddings = numEmbeddings;
    this.embeddingDim = embeddingDim;
    this.mode = opts?.mode ?? 'mean';
    this.scaleGradByFreq = opts?.scaleGradByFreq ?? false;
    this.sparse = opts?.sparse ?? false;
    this.includeLastOffset = opts?.includeLastOffset ?? false;
    this.paddingIdx = opts?.paddingIdx;
    const factoryOpts: TensorOptions = { dtype: opts?.dtype, device: opts?.device };

    const wInit = empty([numEmbeddings, embeddingDim], factoryOpts);
    this.weight = new Parameter(wInit, true);
    wInit.dispose();
    normal_(this.weight, 0, 1);
    if (this.paddingIdx !== undefined) {
      noGrad(() => {
        this.weight.indexPut_([this.paddingIdx!], 0);
      });
    }
  }

  forward(input: Tensor, offsets?: Tensor, perSampleWeights?: Tensor | null): Tensor {
    return F.embeddingBag(input, this.weight, {
      offsets,
      scaleGradByFreq: this.scaleGradByFreq,
      mode: this.mode,
      sparse: this.sparse,
      perSampleWeights,
      includeLastOffset: this.includeLastOffset,
      paddingIdx: this.paddingIdx,
    });
  }
}
