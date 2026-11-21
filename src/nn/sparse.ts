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
