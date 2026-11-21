import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';
import type { DType } from './dtype.js';

export type TypedArray =
  | Float32Array
  | Float64Array
  | Int8Array
  | Uint8Array
  | Uint8ClampedArray
  | Int16Array
  | Uint16Array
  | Int32Array
  | BigInt64Array;

export type Scalar = number | boolean | bigint;

export interface NativeTensorOptions {
  dtype?: DType;
  device?: string;
  layout?: string;
  pinMemory?: boolean;
  requiresGrad?: boolean;
}

export interface NativeBackwardOptions {
  retainGraph?: boolean;
  createGraph?: boolean;
  inputs?: NativeTensor[];
}

export interface NativeGradOptions {
  retainGraph?: boolean;
  createGraph?: boolean;
  allowUnused?: boolean;
}

export interface NativeToOptions {
  device?: string;
  dtype?: DType;
  nonBlocking?: boolean;
  copy?: boolean;
  memoryFormat?: string;
}

export interface NativeTensor {
  dispose(): void;
  isDisposed(): boolean;
  shape(): number[];
  strides(): number[];
  dim(): number;
  numel(): number;
  elementSize(): number;
  nbytes(): number;
  storageOffset(): number;
  dtype(): DType;
  device(): string;
  isContiguous(): boolean;
  isCuda(): boolean;
  requiresGrad(): boolean;
  setRequiresGrad(req?: boolean): NativeTensor;
  isLeaf(): boolean;
  retainsGrad(): boolean;
  retainGrad(): void;
  grad(): NativeTensor | null;
  setGrad(g: NativeTensor | null): void;
  gradFnName(): string | null;
  dataPtr(): bigint;
  storagePtr(): bigint;
  item(): Scalar | { real: number; imag: number };
  toTypedArray(): TypedArray;
  dataView(): TypedArray;
  copyFrom(src: NativeTensor, nonBlocking?: boolean): NativeTensor;
  detach(): NativeTensor;
  detachInplace(): NativeTensor;
  clone(): NativeTensor;
  contiguous(memoryFormat?: string): NativeTensor;
  to(opts: NativeToOptions): NativeTensor;
  backward(grad?: NativeTensor | null, opts?: NativeBackwardOptions): void;
  indexGet(indices: unknown[]): NativeTensor;
  indexPut(indices: unknown[], value: NativeTensor | Scalar, accumulate?: boolean): NativeTensor;
  equal(other: NativeTensor): boolean;
  formatString(): string;
}

export interface OpSchemaArg {
  name: string;
  type: string;
  realType: string;
  kwargOnly: boolean;
  isOut: boolean;
  hasDefault: boolean;
}

export interface OpSchema {
  name: string;
  overload: string;
  isMutable: boolean;
  arguments: OpSchemaArg[];
  returns: { name: string; type: string }[];
}

export interface MemoryStats {
  liveTensors: number;
  trackedStorages: number;
  reportedExternalBytes: number;
}

export interface CudaMemoryStats {
  allocated: number;
  maxAllocated: number;
  reserved: number;
  maxReserved: number;
}

export interface NativeFastOps {
  tensor(data: unknown, opts?: NativeTensorOptions): NativeTensor;
  empty(shape: number[], opts?: NativeTensorOptions): NativeTensor;
  zeros(shape: number[], opts?: NativeTensorOptions): NativeTensor;
  ones(shape: number[], opts?: NativeTensorOptions): NativeTensor;
  full(shape: number[], fillValue: Scalar, opts?: NativeTensorOptions): NativeTensor;
  rand(shape: number[], opts?: NativeTensorOptions): NativeTensor;
  randn(shape: number[], opts?: NativeTensorOptions): NativeTensor;
  randint(low: number, high: number, shape: number[], opts?: NativeTensorOptions): NativeTensor;
  randperm(n: number, opts?: NativeTensorOptions): NativeTensor;
  arange(start: Scalar, end: Scalar, step?: Scalar, opts?: NativeTensorOptions): NativeTensor;
  linspace(start: Scalar, end: Scalar, steps: number, opts?: NativeTensorOptions): NativeTensor;
  eye(n: number, m?: number, opts?: NativeTensorOptions): NativeTensor;
  zerosLike(t: NativeTensor): NativeTensor;
  onesLike(t: NativeTensor): NativeTensor;
  emptyLike(t: NativeTensor): NativeTensor;
  fullLike(t: NativeTensor, fillValue: Scalar): NativeTensor;
  randnLike(t: NativeTensor): NativeTensor;

  add(a: NativeTensor, b: NativeTensor | Scalar, alpha?: Scalar): NativeTensor;
  sub(a: NativeTensor, b: NativeTensor | Scalar, alpha?: Scalar): NativeTensor;
  mul(a: NativeTensor, b: NativeTensor | Scalar): NativeTensor;
  div(a: NativeTensor, b: NativeTensor | Scalar): NativeTensor;
  pow(a: NativeTensor, b: NativeTensor | Scalar): NativeTensor;
  matmul(a: NativeTensor, b: NativeTensor): NativeTensor;
  bmm(a: NativeTensor, b: NativeTensor): NativeTensor;
  einsum(equation: string, tensors: NativeTensor[]): NativeTensor;
  neg(a: NativeTensor): NativeTensor;
  abs(a: NativeTensor): NativeTensor;
  exp(a: NativeTensor): NativeTensor;
  log(a: NativeTensor): NativeTensor;
  sqrt(a: NativeTensor): NativeTensor;
  rsqrt(a: NativeTensor): NativeTensor;
  sin(a: NativeTensor): NativeTensor;
  cos(a: NativeTensor): NativeTensor;
  tanh(a: NativeTensor): NativeTensor;
  sigmoid(a: NativeTensor): NativeTensor;
  relu(a: NativeTensor): NativeTensor;
  gelu(a: NativeTensor, approximate?: string): NativeTensor;
  silu(a: NativeTensor): NativeTensor;
  leakyRelu(a: NativeTensor, negativeSlope?: Scalar): NativeTensor;
  clamp(a: NativeTensor, min?: Scalar | null, max?: Scalar | null): NativeTensor;
  where(cond: NativeTensor, x: NativeTensor, y: NativeTensor): NativeTensor;
  isnan(a: NativeTensor): NativeTensor;
  isinf(a: NativeTensor): NativeTensor;
  isfinite(a: NativeTensor): NativeTensor;

  add_(a: NativeTensor, b: NativeTensor | Scalar, alpha?: Scalar): NativeTensor;
  sub_(a: NativeTensor, b: NativeTensor | Scalar, alpha?: Scalar): NativeTensor;
  mul_(a: NativeTensor, b: NativeTensor | Scalar): NativeTensor;
  div_(a: NativeTensor, b: NativeTensor | Scalar): NativeTensor;
  zero_(a: NativeTensor): NativeTensor;
  fill_(a: NativeTensor, val: Scalar): NativeTensor;
  uniform_(a: NativeTensor, from?: number, to?: number): NativeTensor;
  normal_(a: NativeTensor, mean?: number, std?: number): NativeTensor;
  addcmul_(a: NativeTensor, t1: NativeTensor, t2: NativeTensor, value?: Scalar): NativeTensor;
  addcdiv_(a: NativeTensor, t1: NativeTensor, t2: NativeTensor, value?: Scalar): NativeTensor;
  clamp_(a: NativeTensor, min?: Scalar | null, max?: Scalar | null): NativeTensor;

  sum(a: NativeTensor, dim?: number | number[] | null, keepdim?: boolean, dtype?: DType | null): NativeTensor;
  mean(a: NativeTensor, dim?: number | number[] | null, keepdim?: boolean, dtype?: DType | null): NativeTensor;
  max(a: NativeTensor): NativeTensor;
  max(a: NativeTensor, dim: number, keepdim?: boolean): { values: NativeTensor; indices: NativeTensor };
  min(a: NativeTensor): NativeTensor;
  min(a: NativeTensor, dim: number, keepdim?: boolean): { values: NativeTensor; indices: NativeTensor };
  argmax(a: NativeTensor, dim?: number | null, keepdim?: boolean): NativeTensor;
  argmin(a: NativeTensor, dim?: number | null, keepdim?: boolean): NativeTensor;
  softmax(a: NativeTensor, dim: number): NativeTensor;
  logSoftmax(a: NativeTensor, dim: number): NativeTensor;
  norm(a: NativeTensor, p?: Scalar | null, dim?: number | number[] | null, keepdim?: boolean): NativeTensor;

  reshape(a: NativeTensor, shape: number[]): NativeTensor;
  view(a: NativeTensor, shape: number[]): NativeTensor;
  transpose(a: NativeTensor, dim0: number, dim1: number): NativeTensor;
  permute(a: NativeTensor, dims: number[]): NativeTensor;
  squeeze(a: NativeTensor, dim?: number | null): NativeTensor;
  unsqueeze(a: NativeTensor, dim: number): NativeTensor;
  flatten(a: NativeTensor, startDim?: number, endDim?: number): NativeTensor;
  expand(a: NativeTensor, size: number[]): NativeTensor;
  cat(tensors: NativeTensor[], dim?: number): NativeTensor;
  stack(tensors: NativeTensor[], dim?: number): NativeTensor;
  split(a: NativeTensor, splitSizeOrSections: number | number[], dim?: number): NativeTensor[];
  chunk(a: NativeTensor, chunks: number, dim?: number): NativeTensor[];
  narrow(a: NativeTensor, dim: number, start: number, length: number): NativeTensor;
  maskedFill(a: NativeTensor, mask: NativeTensor, value: Scalar): NativeTensor;
  tril(a: NativeTensor, diagonal?: number): NativeTensor;
  triu(a: NativeTensor, diagonal?: number): NativeTensor;

  linear(input: NativeTensor, weight: NativeTensor, bias?: NativeTensor | null): NativeTensor;
  conv1d(
    input: NativeTensor,
    weight: NativeTensor,
    bias?: NativeTensor | null,
    stride?: number[],
    padding?: number[],
    dilation?: number[],
    groups?: number
  ): NativeTensor;
  conv2d(
    input: NativeTensor,
    weight: NativeTensor,
    bias?: NativeTensor | null,
    stride?: number[],
    padding?: number[],
    dilation?: number[],
    groups?: number
  ): NativeTensor;
  maxPool2d(
    input: NativeTensor,
    kernelSize: number[],
    stride?: number[] | null,
    padding?: number[],
    dilation?: number[],
    ceilMode?: boolean
  ): NativeTensor;
  avgPool2d(
    input: NativeTensor,
    kernelSize: number[],
    stride?: number[] | null,
    padding?: number[],
    ceilMode?: boolean,
    countIncludePad?: boolean
  ): NativeTensor;
  adaptiveAvgPool2d(input: NativeTensor, outputSize: number[]): NativeTensor;
  layerNorm(
    input: NativeTensor,
    normalizedShape: number[],
    weight?: NativeTensor | null,
    bias?: NativeTensor | null,
    eps?: number
  ): NativeTensor;
  rmsNorm(
    input: NativeTensor,
    normalizedShape: number[],
    weight?: NativeTensor | null,
    eps?: number | null
  ): NativeTensor;
  batchNorm(
    input: NativeTensor,
    weight: NativeTensor | null,
    bias: NativeTensor | null,
    runningMean: NativeTensor | null,
    runningVar: NativeTensor | null,
    training: boolean,
    momentum?: number,
    eps?: number
  ): NativeTensor;
  groupNorm(
    input: NativeTensor,
    numGroups: number,
    weight?: NativeTensor | null,
    bias?: NativeTensor | null,
    eps?: number
  ): NativeTensor;
  embedding(
    weight: NativeTensor,
    indices: NativeTensor,
    paddingIdx?: number,
    scaleGradByFreq?: boolean,
    sparse?: boolean
  ): NativeTensor;
  dropout(input: NativeTensor, p?: number, train?: boolean): NativeTensor;
  scaledDotProductAttention(
    q: NativeTensor,
    k: NativeTensor,
    v: NativeTensor,
    attnMask?: NativeTensor | null,
    dropoutP?: number,
    isCausal?: boolean,
    scale?: number | null,
    enableGqa?: boolean
  ): NativeTensor;
  crossEntropy(
    input: NativeTensor,
    target: NativeTensor,
    weight?: NativeTensor | null,
    reduction?: string,
    ignoreIndex?: number,
    labelSmoothing?: number
  ): NativeTensor;
  nllLoss(
    input: NativeTensor,
    target: NativeTensor,
    weight?: NativeTensor | null,
    reduction?: string,
    ignoreIndex?: number
  ): NativeTensor;
  mseLoss(input: NativeTensor, target: NativeTensor, reduction?: string): NativeTensor;
  l1Loss(input: NativeTensor, target: NativeTensor, reduction?: string): NativeTensor;
  smoothL1Loss(
    input: NativeTensor,
    target: NativeTensor,
    reduction?: string,
    beta?: number
  ): NativeTensor;
  bceWithLogitsLoss(
    input: NativeTensor,
    target: NativeTensor,
    weight?: NativeTensor | null,
    posWeight?: NativeTensor | null,
    reduction?: string
  ): NativeTensor;
  klDiv(
    input: NativeTensor,
    target: NativeTensor,
    reduction?: string,
    logTarget?: boolean
  ): NativeTensor;
}

export interface NativeAddon {
  NativeTensor: new () => NativeTensor;
  fromTypedArray(
    ta: TypedArray,
    shape?: number[] | null,
    opts?: { copy?: boolean; dtype?: DType; device?: string; requiresGrad?: boolean }
  ): NativeTensor;
  listOps(prefix?: string): { name: string; overload: string }[];
  getOpSchema(name: string, overload?: string): OpSchema | null;
  getAllSchemas(prefix?: string): OpSchema[];
  callOp(name: string, overload: string, posArgs: unknown[], kwArgs?: Record<string, unknown>): unknown;
  callOpAuto(name: string, posArgs: unknown[], kwArgs?: Record<string, unknown>): unknown;

  isGradEnabled(): boolean;
  setGradEnabled(enabled: boolean): boolean;
  isInferenceModeEnabled(): boolean;
  pushInferenceMode(enabled?: boolean): number;
  popInferenceMode(): void;
  isAutocastEnabled(deviceType?: string): boolean;
  setAutocastEnabled(deviceType: string, enabled: boolean): void;
  getAutocastDtype(deviceType?: string): DType;
  setAutocastDtype(deviceType: string, dtype: DType): void;
  autocastClearCache(): void;
  autocastIncrementNesting(): number;
  autocastDecrementNesting(): number;
  autogradBackward(
    tensors: NativeTensor[],
    gradTensors?: (NativeTensor | null)[],
    opts?: NativeBackwardOptions
  ): void;
  autogradGrad(
    outputs: NativeTensor[],
    inputs: NativeTensor[],
    gradOutputs?: (NativeTensor | null)[],
    opts?: NativeGradOptions
  ): (NativeTensor | null)[];
  applyCustomAutograd(
    name: string,
    tensorInputs: NativeTensor[],
    ctxObj: Record<string, unknown>,
    forwardFn: (ctx: Record<string, unknown>) => (NativeTensor | null)[],
    backwardFn: (ctx: Record<string, unknown>, gradOutputs: (NativeTensor | null)[]) => (NativeTensor | null)[]
  ): (NativeTensor | null)[];

  cudaIsAvailable(): boolean;
  cudaDeviceCount(): number;
  cudaCurrentDevice(): number;
  cudaSetDevice(index: number): void;
  cudaSynchronize(index?: number): void;
  cudaEmptyCache(): void;
  cudaMemoryStats(index?: number): CudaMemoryStats;
  cudaDeviceName(index?: number): string;
  manualSeed(seed: number | bigint): void;
  getNumThreads(): number;
  setNumThreads(n: number): void;
  memoryStats(): MemoryStats;
  libtorchVersion(): string;
  fast: NativeFastOps;
}

const require = createRequire(import.meta.url);
const __dirname = path.dirname(fileURLToPath(import.meta.url));

function resolveAddonPath(): string {
  const candidates = [
    process.env.NORCH_NATIVE_PATH,
    path.resolve(__dirname, '..', 'build', 'norch_native.node'),
    path.resolve(__dirname, '..', '..', 'build', 'norch_native.node'),
  ].filter((p): p is string => Boolean(p));

  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }
  throw new Error(
    `Could not find norch_native.node. Run "npm run build:native" first. Searched: ${candidates.join(', ')}`
  );
}

export const native: NativeAddon = require(resolveAddonPath()) as NativeAddon;
