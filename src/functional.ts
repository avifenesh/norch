import { native } from './native.js';
import { Tensor, wrapTensor } from './tensor.js';

export type Reduction = 'none' | 'mean' | 'sum';

function pair(v: number | [number, number]): [number, number] {
  return Array.isArray(v) ? v : [v, v];
}

function single(v: number | [number]): [number] {
  return Array.isArray(v) ? v : [v];
}

export function linear(input: Tensor, weight: Tensor, bias?: Tensor | null): Tensor {
  return wrapTensor(native.fast.linear(input._native, weight._native, bias ? bias._native : null));
}

export function relu(input: Tensor): Tensor {
  return input.relu();
}

export function gelu(input: Tensor, approximate: 'none' | 'tanh' = 'none'): Tensor {
  return input.gelu(approximate);
}

export function silu(input: Tensor): Tensor {
  return input.silu();
}

export function sigmoid(input: Tensor): Tensor {
  return input.sigmoid();
}

export function tanh(input: Tensor): Tensor {
  return input.tanh();
}

export function leakyRelu(input: Tensor, negativeSlope = 0.01): Tensor {
  return input.leakyRelu(negativeSlope);
}

export function softmax(input: Tensor, dim: number): Tensor {
  return input.softmax(dim);
}

export function logSoftmax(input: Tensor, dim: number): Tensor {
  return input.logSoftmax(dim);
}

export interface Conv1dOptions {
  bias?: Tensor | null;
  stride?: number | [number];
  padding?: number | [number];
  dilation?: number | [number];
  groups?: number;
}

export function conv1d(input: Tensor, weight: Tensor, opts?: Conv1dOptions): Tensor {
  return wrapTensor(
    native.fast.conv1d(
      input._native,
      weight._native,
      opts?.bias ? opts.bias._native : null,
      single(opts?.stride ?? 1),
      single(opts?.padding ?? 0),
      single(opts?.dilation ?? 1),
      opts?.groups ?? 1
    )
  );
}

export interface Conv2dOptions {
  bias?: Tensor | null;
  stride?: number | [number, number];
  padding?: number | [number, number];
  dilation?: number | [number, number];
  groups?: number;
}

export function conv2d(input: Tensor, weight: Tensor, opts?: Conv2dOptions): Tensor {
  return wrapTensor(
    native.fast.conv2d(
      input._native,
      weight._native,
      opts?.bias ? opts.bias._native : null,
      pair(opts?.stride ?? 1),
      pair(opts?.padding ?? 0),
      pair(opts?.dilation ?? 1),
      opts?.groups ?? 1
    )
  );
}

export interface MaxPool2dOptions {
  stride?: number | [number, number];
  padding?: number | [number, number];
  dilation?: number | [number, number];
  ceilMode?: boolean;
}

export function maxPool2d(
  input: Tensor,
  kernelSize: number | [number, number],
  opts?: MaxPool2dOptions
): Tensor {
  return wrapTensor(
    native.fast.maxPool2d(
      input._native,
      pair(kernelSize),
      opts?.stride !== undefined ? pair(opts.stride) : null,
      pair(opts?.padding ?? 0),
      pair(opts?.dilation ?? 1),
      opts?.ceilMode ?? false
    )
  );
}

export interface AvgPool2dOptions {
  stride?: number | [number, number];
  padding?: number | [number, number];
  ceilMode?: boolean;
  countIncludePad?: boolean;
}

export function avgPool2d(
  input: Tensor,
  kernelSize: number | [number, number],
  opts?: AvgPool2dOptions
): Tensor {
  return wrapTensor(
    native.fast.avgPool2d(
      input._native,
      pair(kernelSize),
      opts?.stride !== undefined ? pair(opts.stride) : null,
      pair(opts?.padding ?? 0),
      opts?.ceilMode ?? false,
      opts?.countIncludePad ?? true
    )
  );
}

export function adaptiveAvgPool2d(input: Tensor, outputSize: number | [number, number]): Tensor {
  return wrapTensor(native.fast.adaptiveAvgPool2d(input._native, pair(outputSize)));
}

export function layerNorm(
  input: Tensor,
  normalizedShape: number[],
  weight?: Tensor | null,
  bias?: Tensor | null,
  eps = 1e-5
): Tensor {
  return wrapTensor(
    native.fast.layerNorm(
      input._native,
      normalizedShape,
      weight ? weight._native : null,
      bias ? bias._native : null,
      eps
    )
  );
}

export function rmsNorm(
  input: Tensor,
  normalizedShape: number[],
  weight?: Tensor | null,
  eps?: number | null
): Tensor {
  return wrapTensor(
    native.fast.rmsNorm(
      input._native,
      normalizedShape,
      weight ? weight._native : null,
      eps ?? null
    )
  );
}

export interface BatchNormOptions {
  weight?: Tensor | null;
  bias?: Tensor | null;
  training?: boolean;
  momentum?: number;
  eps?: number;
}

export function batchNorm(
  input: Tensor,
  runningMean: Tensor | null,
  runningVar: Tensor | null,
  opts?: BatchNormOptions
): Tensor {
  return wrapTensor(
    native.fast.batchNorm(
      input._native,
      opts?.weight ? opts.weight._native : null,
      opts?.bias ? opts.bias._native : null,
      runningMean ? runningMean._native : null,
      runningVar ? runningVar._native : null,
      opts?.training ?? false,
      opts?.momentum ?? 0.1,
      opts?.eps ?? 1e-5
    )
  );
}

export function groupNorm(
  input: Tensor,
  numGroups: number,
  weight?: Tensor | null,
  bias?: Tensor | null,
  eps = 1e-5
): Tensor {
  return wrapTensor(
    native.fast.groupNorm(
      input._native,
      numGroups,
      weight ? weight._native : null,
      bias ? bias._native : null,
      eps
    )
  );
}

export interface EmbeddingOptions {
  paddingIdx?: number;
  scaleGradByFreq?: boolean;
  sparse?: boolean;
}

export function embedding(input: Tensor, weight: Tensor, opts?: EmbeddingOptions): Tensor {
  return wrapTensor(
    native.fast.embedding(
      weight._native,
      input._native,
      opts?.paddingIdx ?? -1,
      opts?.scaleGradByFreq ?? false,
      opts?.sparse ?? false
    )
  );
}

export function dropout(input: Tensor, p = 0.5, training = true): Tensor {
  return wrapTensor(native.fast.dropout(input._native, p, training));
}

export interface SdpaOptions {
  attnMask?: Tensor | null;
  dropoutP?: number;
  isCausal?: boolean;
  scale?: number | null;
  enableGqa?: boolean;
}

export function scaledDotProductAttention(
  query: Tensor,
  key: Tensor,
  value: Tensor,
  opts?: SdpaOptions
): Tensor {
  return wrapTensor(
    native.fast.scaledDotProductAttention(
      query._native,
      key._native,
      value._native,
      opts?.attnMask ? opts.attnMask._native : null,
      opts?.dropoutP ?? 0,
      opts?.isCausal ?? false,
      opts?.scale ?? null,
      opts?.enableGqa ?? false
    )
  );
}

export interface CrossEntropyOptions {
  weight?: Tensor | null;
  reduction?: Reduction;
  ignoreIndex?: number;
  labelSmoothing?: number;
}

export function crossEntropy(
  input: Tensor,
  target: Tensor,
  opts?: CrossEntropyOptions
): Tensor {
  return wrapTensor(
    native.fast.crossEntropy(
      input._native,
      target._native,
      opts?.weight ? opts.weight._native : null,
      opts?.reduction ?? 'mean',
      opts?.ignoreIndex ?? -100,
      opts?.labelSmoothing ?? 0
    )
  );
}

export interface NllLossOptions {
  weight?: Tensor | null;
  reduction?: Reduction;
  ignoreIndex?: number;
}

export function nllLoss(input: Tensor, target: Tensor, opts?: NllLossOptions): Tensor {
  return wrapTensor(
    native.fast.nllLoss(
      input._native,
      target._native,
      opts?.weight ? opts.weight._native : null,
      opts?.reduction ?? 'mean',
      opts?.ignoreIndex ?? -100
    )
  );
}

export function mseLoss(input: Tensor, target: Tensor, reduction: Reduction = 'mean'): Tensor {
  return wrapTensor(native.fast.mseLoss(input._native, target._native, reduction));
}

export function l1Loss(input: Tensor, target: Tensor, reduction: Reduction = 'mean'): Tensor {
  return wrapTensor(native.fast.l1Loss(input._native, target._native, reduction));
}

export function smoothL1Loss(
  input: Tensor,
  target: Tensor,
  reduction: Reduction = 'mean',
  beta = 1.0
): Tensor {
  return wrapTensor(native.fast.smoothL1Loss(input._native, target._native, reduction, beta));
}

export interface BceWithLogitsOptions {
  weight?: Tensor | null;
  posWeight?: Tensor | null;
  reduction?: Reduction;
}

export function binaryCrossEntropyWithLogits(
  input: Tensor,
  target: Tensor,
  opts?: BceWithLogitsOptions
): Tensor {
  return wrapTensor(
    native.fast.bceWithLogitsLoss(
      input._native,
      target._native,
      opts?.weight ? opts.weight._native : null,
      opts?.posWeight ? opts.posWeight._native : null,
      opts?.reduction ?? 'mean'
    )
  );
}

export function klDiv(
  input: Tensor,
  target: Tensor,
  reduction: Reduction = 'mean',
  logTarget = false
): Tensor {
  return wrapTensor(native.fast.klDiv(input._native, target._native, reduction, logTarget));
}

export function huberLoss(
  input: Tensor,
  target: Tensor,
  reduction: Reduction = 'mean',
  delta = 1.0
): Tensor {
  return wrapTensor(native.fast.huberLoss(input._native, target._native, reduction, delta));
}

export function cosineSimilarity(x1: Tensor, x2: Tensor, dim = 1, eps = 1e-8): Tensor {
  return wrapTensor(native.fast.cosineSimilarity(x1._native, x2._native, dim, eps));
}

export interface Conv3dOptions {
  bias?: Tensor | null;
  stride?: number | [number, number, number];
  padding?: number | [number, number, number];
  dilation?: number | [number, number, number];
  groups?: number;
}

function triple(v: number | [number, number, number]): [number, number, number] {
  return Array.isArray(v) ? v : [v, v, v];
}

export function conv3d(input: Tensor, weight: Tensor, opts?: Conv3dOptions): Tensor {
  return wrapTensor(
    native.fast.conv3d(
      input._native,
      weight._native,
      opts?.bias ? opts.bias._native : null,
      triple(opts?.stride ?? 1),
      triple(opts?.padding ?? 0),
      triple(opts?.dilation ?? 1),
      opts?.groups ?? 1
    )
  );
}

export interface ConvTranspose1dOptions {
  bias?: Tensor | null;
  stride?: number | [number];
  padding?: number | [number];
  outputPadding?: number | [number];
  groups?: number;
  dilation?: number | [number];
}

export function convTranspose1d(
  input: Tensor,
  weight: Tensor,
  opts?: ConvTranspose1dOptions
): Tensor {
  return wrapTensor(
    native.fast.convTranspose1d(
      input._native,
      weight._native,
      opts?.bias ? opts.bias._native : null,
      single(opts?.stride ?? 1),
      single(opts?.padding ?? 0),
      single(opts?.outputPadding ?? 0),
      opts?.groups ?? 1,
      single(opts?.dilation ?? 1)
    )
  );
}

export interface ConvTranspose2dOptions {
  bias?: Tensor | null;
  stride?: number | [number, number];
  padding?: number | [number, number];
  outputPadding?: number | [number, number];
  groups?: number;
  dilation?: number | [number, number];
}

export function convTranspose2d(
  input: Tensor,
  weight: Tensor,
  opts?: ConvTranspose2dOptions
): Tensor {
  return wrapTensor(
    native.fast.convTranspose2d(
      input._native,
      weight._native,
      opts?.bias ? opts.bias._native : null,
      pair(opts?.stride ?? 1),
      pair(opts?.padding ?? 0),
      pair(opts?.outputPadding ?? 0),
      opts?.groups ?? 1,
      pair(opts?.dilation ?? 1)
    )
  );
}

export interface InterpolateOptions {
  size?: number | [number, number];
  scaleFactor?: number | [number, number];
  mode?: 'nearest' | 'bilinear' | 'bicubic';
  alignCorners?: boolean;
}

export function interpolate(input: Tensor, opts: InterpolateOptions): Tensor {
  let outSize: [number, number];
  if (opts.size !== undefined) {
    outSize = pair(opts.size);
  } else if (opts.scaleFactor !== undefined) {
    const [sh, sw] = pair(opts.scaleFactor);
    const h = input.size(input.ndim - 2);
    const w = input.size(input.ndim - 1);
    outSize = [Math.floor(h * sh), Math.floor(w * sw)];
  } else {
    throw new TypeError('interpolate() requires either size or scaleFactor');
  }
  return wrapTensor(
    native.fast.upsample2d(
      input._native,
      outSize,
      opts.mode ?? 'nearest',
      opts.alignCorners ?? false
    )
  );
}

export function pad(
  input: Tensor,
  padWidths: number[],
  mode: 'constant' | 'reflect' | 'replicate' | 'circular' = 'constant',
  value?: number
): Tensor {
  return wrapTensor(native.fast.pad(input._native, padWidths, mode, value ?? null));
}

export function oneHot(input: Tensor, numClasses = -1): Tensor {
  return wrapTensor(native.fast.oneHot(input._native, numClasses));
}

export const functional = {
  linear,
  relu,
  gelu,
  silu,
  sigmoid,
  tanh,
  leakyRelu,
  softmax,
  logSoftmax,
  conv1d,
  conv2d,
  conv3d,
  convTranspose1d,
  convTranspose2d,
  interpolate,
  pad,
  oneHot,
  maxPool2d,
  avgPool2d,
  adaptiveAvgPool2d,
  layerNorm,
  rmsNorm,
  batchNorm,
  groupNorm,
  embedding,
  dropout,
  scaledDotProductAttention,
  crossEntropy,
  nllLoss,
  mseLoss,
  l1Loss,
  smoothL1Loss,
  huberLoss,
  cosineSimilarity,
  binaryCrossEntropyWithLogits,
  klDiv,
};
