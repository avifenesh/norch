import * as F from '../functional.js';
import type { TensorOptions } from '../ops.js';
import type { Tensor } from '../tensor.js';
import { Dropout } from './dropout.js';
import { Linear } from './linear.js';
import { Module } from './module.js';
import { LayerNorm } from './norm.js';

export interface MultiheadAttentionOptions extends TensorOptions {
  dropout?: number;
  bias?: boolean;
}

/**
 * Multi-head self/cross attention built directly on libtorch's fused scaledDotProductAttention.
 * Expects batch-first tensors of shape [B, T, embedDim].
 */
export class MultiheadAttention extends Module {
  readonly embedDim: number;
  readonly numHeads: number;
  readonly headDim: number;
  readonly dropoutP: number;
  readonly qProj: Linear;
  readonly kProj: Linear;
  readonly vProj: Linear;
  readonly outProj: Linear;

  constructor(embedDim: number, numHeads: number, opts?: MultiheadAttentionOptions) {
    super();
    if (embedDim % numHeads !== 0) {
      throw new RangeError(`embedDim (${embedDim}) must be divisible by numHeads (${numHeads})`);
    }
    this.embedDim = embedDim;
    this.numHeads = numHeads;
    this.headDim = embedDim / numHeads;
    this.dropoutP = opts?.dropout ?? 0.0;
    const linOpts = { bias: opts?.bias ?? true, dtype: opts?.dtype, device: opts?.device };
    this.qProj = new Linear(embedDim, embedDim, linOpts);
    this.kProj = new Linear(embedDim, embedDim, linOpts);
    this.vProj = new Linear(embedDim, embedDim, linOpts);
    this.outProj = new Linear(embedDim, embedDim, linOpts);
  }

  forward(
    query: Tensor,
    key: Tensor = query,
    value: Tensor = key,
    opts?: { attnMask?: Tensor | null; isCausal?: boolean }
  ): Tensor {
    const [B, Tq, D] = query.shape;
    const Tk = key.shape[1]!;
    const q = this.qProj.forward(query).view(B!, Tq!, this.numHeads, this.headDim).transpose(1, 2);
    const k = this.kProj.forward(key).view(B!, Tk, this.numHeads, this.headDim).transpose(1, 2);
    const v = this.vProj.forward(value).view(B!, Tk, this.numHeads, this.headDim).transpose(1, 2);

    const attn = F.scaledDotProductAttention(q, k, v, {
      attnMask: opts?.attnMask ?? null,
      dropoutP: this.training ? this.dropoutP : 0.0,
      isCausal: opts?.isCausal ?? false,
    });
    const merged = attn.transpose(1, 2).contiguous().view(B!, Tq!, D!);
    return this.outProj.forward(merged);
  }
}

export interface TransformerBlockOptions extends TensorOptions {
  dimFeedforward?: number;
  dropout?: number;
  normFirst?: boolean;
}

export class TransformerBlock extends Module {
  readonly norm1: LayerNorm;
  readonly norm2: LayerNorm;
  readonly attn: MultiheadAttention;
  readonly fc1: Linear;
  readonly fc2: Linear;
  readonly drop: Dropout;
  readonly normFirst: boolean;

  constructor(dModel: number, nHead: number, opts?: TransformerBlockOptions) {
    super();
    const dff = opts?.dimFeedforward ?? dModel * 4;
    const drop = opts?.dropout ?? 0.0;
    this.normFirst = opts?.normFirst ?? true;
    const common: TensorOptions = { dtype: opts?.dtype, device: opts?.device };
    this.norm1 = new LayerNorm(dModel, common);
    this.norm2 = new LayerNorm(dModel, common);
    this.attn = new MultiheadAttention(dModel, nHead, { ...common, dropout: drop });
    this.fc1 = new Linear(dModel, dff, common);
    this.fc2 = new Linear(dff, dModel, common);
    this.drop = new Dropout(drop);
  }

  forward(x: Tensor, opts?: { attnMask?: Tensor | null; isCausal?: boolean }): Tensor {
    if (this.normFirst) {
      const h1 = this.attn.forward(this.norm1.forward(x), undefined, undefined, opts);
      const r1 = x.add(this.drop.forward(h1));
      const h2 = this.fc2.forward(this.drop.forward(this.fc1.forward(this.norm2.forward(r1)).gelu()));
      return r1.add(this.drop.forward(h2));
    }
    const h1 = this.attn.forward(x, undefined, undefined, opts);
    const r1 = this.norm1.forward(x.add(this.drop.forward(h1)));
    const h2 = this.fc2.forward(this.drop.forward(this.fc1.forward(r1).gelu()));
    return this.norm2.forward(r1.add(this.drop.forward(h2)));
  }
}

export class TransformerEncoderLayer extends TransformerBlock {}

export class TransformerEncoder extends Module {
  readonly layers: TransformerEncoderLayer[] = [];
  readonly norm: LayerNorm | null;

  constructor(
    createLayer: () => TransformerEncoderLayer,
    numLayers: number,
    norm?: LayerNorm | null
  ) {
    super();
    for (let i = 0; i < numLayers; i++) {
      const layer = createLayer();
      this.layers.push(layer);
      this.addModule(`layer_${i}`, layer);
    }
    this.norm = norm ?? null;
    if (this.norm) {
      this.addModule('norm', this.norm);
    }
  }

  forward(src: Tensor, opts?: { attnMask?: Tensor | null; isCausal?: boolean }): Tensor {
    let out = src;
    for (const layer of this.layers) {
      out = layer.call(out, opts) as Tensor;
    }
    return this.norm ? (this.norm.call(out) as Tensor) : out;
  }
}

export class TransformerDecoderLayer extends Module {
  readonly selfAttn: MultiheadAttention;
  readonly multiheadAttn: MultiheadAttention;
  readonly linear1: Linear;
  readonly linear2: Linear;
  readonly norm1: LayerNorm;
  readonly norm2: LayerNorm;
  readonly norm3: LayerNorm;
  readonly drop: Dropout;
  readonly normFirst: boolean;

  constructor(dModel: number, nHead: number, opts?: TransformerBlockOptions) {
    super();
    const dff = opts?.dimFeedforward ?? dModel * 4;
    const drop = opts?.dropout ?? 0.0;
    this.normFirst = opts?.normFirst ?? true;
    const common: TensorOptions = { dtype: opts?.dtype, device: opts?.device };
    this.selfAttn = new MultiheadAttention(dModel, nHead, { ...common, dropout: drop });
    this.multiheadAttn = new MultiheadAttention(dModel, nHead, { ...common, dropout: drop });
    this.linear1 = new Linear(dModel, dff, common);
    this.linear2 = new Linear(dff, dModel, common);
    this.norm1 = new LayerNorm(dModel, common);
    this.norm2 = new LayerNorm(dModel, common);
    this.norm3 = new LayerNorm(dModel, common);
    this.drop = new Dropout(drop);
  }

  forward(
    tgt: Tensor,
    memory: Tensor,
    opts?: {
      tgtMask?: Tensor | null;
      memoryMask?: Tensor | null;
      tgtIsCausal?: boolean;
    }
  ): Tensor {
    if (this.normFirst) {
      const sa = this.selfAttn.forward(this.norm1.forward(tgt), undefined, undefined, {
        attnMask: opts?.tgtMask ?? null,
        isCausal: opts?.tgtIsCausal ?? false,
      });
      const r1 = tgt.add(this.drop.forward(sa));
      const ca = this.multiheadAttn.forward(this.norm2.forward(r1), memory, memory, {
        attnMask: opts?.memoryMask ?? null,
        isCausal: false,
      });
      const r2 = r1.add(this.drop.forward(ca));
      const ff = this.linear2.forward(this.drop.forward(this.linear1.forward(this.norm3.forward(r2)).gelu()));
      return r2.add(this.drop.forward(ff));
    }
    const sa = this.selfAttn.forward(tgt, undefined, undefined, {
      attnMask: opts?.tgtMask ?? null,
      isCausal: opts?.tgtIsCausal ?? false,
    });
    const r1 = this.norm1.forward(tgt.add(this.drop.forward(sa)));
    const ca = this.multiheadAttn.forward(r1, memory, memory, {
      attnMask: opts?.memoryMask ?? null,
      isCausal: false,
    });
    const r2 = this.norm2.forward(r1.add(this.drop.forward(ca)));
    const ff = this.linear2.forward(this.drop.forward(this.linear1.forward(r2).gelu()));
    return this.norm3.forward(r2.add(this.drop.forward(ff)));
  }
}

export class TransformerDecoder extends Module {
  readonly layers: TransformerDecoderLayer[] = [];
  readonly norm: LayerNorm | null;

  constructor(
    createLayer: () => TransformerDecoderLayer,
    numLayers: number,
    norm?: LayerNorm | null
  ) {
    super();
    for (let i = 0; i < numLayers; i++) {
      const layer = createLayer();
      this.layers.push(layer);
      this.addModule(`layer_${i}`, layer);
    }
    this.norm = norm ?? null;
    if (this.norm) {
      this.addModule('norm', this.norm);
    }
  }

  forward(
    tgt: Tensor,
    memory: Tensor,
    opts?: { tgtMask?: Tensor | null; memoryMask?: Tensor | null; tgtIsCausal?: boolean }
  ): Tensor {
    let out = tgt;
    for (const layer of this.layers) {
      out = layer.call(out, memory, opts) as Tensor;
    }
    return this.norm ? (this.norm.call(out) as Tensor) : out;
  }
}
