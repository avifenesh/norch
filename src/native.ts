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

export interface NativeGenerator {
  manualSeed(seed: number | bigint): NativeGenerator;
  seed(): bigint;
  initialSeed(): bigint;
  device(): string;
  getState(): NativeTensor;
  setState(state: NativeTensor): void;
}

export interface NativeCudaStream {
  synchronize(): void;
  query(): boolean;
  deviceIndex(): number;
  streamId(): bigint;
  priority(): number;
  isCapturing(): boolean;
  waitEvent(event: NativeCudaEvent): void;
  waitStream(stream: NativeCudaStream): void;
}

export interface NativeCudaEvent {
  record(stream?: NativeCudaStream | null): void;
  wait(stream?: NativeCudaStream | null): void;
  synchronize(): void;
  query(): boolean;
  elapsedTime(endEvent: NativeCudaEvent): number;
}

export interface NativeCudaGraph {
  captureBegin(pool?: [bigint, bigint] | null): void;
  captureEnd(): void;
  replay(): void;
  reset(): void;
  pool(): [bigint, bigint];
}

export interface NativeAotiModel {
  run(inputs: NativeTensor[]): NativeTensor[];
  getMetadata(): Record<string, string>;
  getCallSpec(): string[];
  getConstantFqns(): string[];
  dispose(): void;
  isDisposed(): boolean;
}

export interface NativeJitModule {
  forward(inputs: unknown[]): unknown;
  runMethod(name: string, inputs: unknown[]): unknown;
  to(deviceOrDtype: string): NativeJitModule;
  eval(): NativeJitModule;
  train(mode?: boolean): NativeJitModule;
  dispose(): void;
  isDisposed(): boolean;
}

export interface NativeProcessGroupOptions {
  backend?: string;
  storeType?: 'tcp' | 'file';
  rank: number;
  worldSize: number;
  masterAddr?: string;
  masterPort?: number;
  filePath?: string;
  timeoutMs?: number;
  groupName?: string;
}

export interface NativeProcessGroup {
  rank(): number;
  worldSize(): number;
  backendName(): string;
  allReduce(tensor: NativeTensor, op?: string | number): void;
  broadcast(tensor: NativeTensor, rootRank?: number): void;
  allGather(outputTensors: NativeTensor[], inputTensor: NativeTensor): void;
  allGatherIntoTensor(outputTensor: NativeTensor, inputTensor: NativeTensor): void;
  reduceScatter(outputTensor: NativeTensor, inputTensors: NativeTensor[], op?: string | number): void;
  barrier(): void;
  send(tensor: NativeTensor, dstRank: number, tag?: number): void;
  recv(tensor: NativeTensor, srcRank: number, tag?: number): void;
  storeSet(key: string, value: string): void;
  storeGet(key: string): string;
  destroy(): void;
  isDestroyed(): boolean;
}

export interface NativeTensorOptions {
  dtype?: DType;
  device?: string;
  layout?: string;
  pinMemory?: boolean;
  requiresGrad?: boolean;
  generator?: NativeGenerator;
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
  isPinned(): boolean;
  pinMemory(): NativeTensor;
  isComplex(): boolean;
  isSparse(): boolean;
  real(): NativeTensor;
  imag(): NativeTensor;
  conj(): NativeTensor;
  toDense(): NativeTensor;
  coalesce(): NativeTensor;
  sparseIndices(): NativeTensor;
  sparseValues(): NativeTensor;
  sparseDim(): number;
  denseDim(): number;
  nnz(): number;
  requiresGrad(): boolean;
  setRequiresGrad(req?: boolean): NativeTensor;
  isLeaf(): boolean;
  retainsGrad(): boolean;
  retainGrad(): void;
  grad(): NativeTensor | null;
  setGrad(g: NativeTensor | null): void;
  gradFnName(): string | null;
  registerHook(fn: (grad: NativeTensor) => NativeTensor | null | undefined | void): number;
  removeHook(hookId: number): void;
  dataPtr(): bigint;
  storagePtr(): bigint;
  item(): Scalar | { real: number; imag: number };
  toTypedArray(): TypedArray;
  toTypedArrayAsync(): Promise<TypedArray>;
  dataView(): TypedArray;
  copyFrom(src: NativeTensor, nonBlocking?: boolean): NativeTensor;
  detach(): NativeTensor;
  detachInplace(): NativeTensor;
  clone(): NativeTensor;
  contiguous(memoryFormat?: string): NativeTensor;
  to(opts: NativeToOptions): NativeTensor;
  backward(grad?: NativeTensor | null, opts?: NativeBackwardOptions): void;
  indexGet(indices: unknown[]): NativeTensor;
  indexPut(indices: unknown[], value: NativeTensor | Scalar): NativeTensor;
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

export interface CudaDeviceProperties {
  name: string;
  major: number;
  minor: number;
  totalMemory: number;
  multiProcessorCount: number;
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
  conv3d(
    input: NativeTensor,
    weight: NativeTensor,
    bias?: NativeTensor | null,
    stride?: number[],
    padding?: number[],
    dilation?: number[],
    groups?: number
  ): NativeTensor;
  convTranspose1d(
    input: NativeTensor,
    weight: NativeTensor,
    bias?: NativeTensor | null,
    stride?: number[],
    padding?: number[],
    outputPadding?: number[],
    groups?: number,
    dilation?: number[]
  ): NativeTensor;
  convTranspose2d(
    input: NativeTensor,
    weight: NativeTensor,
    bias?: NativeTensor | null,
    stride?: number[],
    padding?: number[],
    outputPadding?: number[],
    groups?: number,
    dilation?: number[]
  ): NativeTensor;
  upsample2d(
    input: NativeTensor,
    outputSize: number[],
    mode?: string,
    alignCorners?: boolean
  ): NativeTensor;
  pad(input: NativeTensor, pad: number[], mode?: string, value?: number | null): NativeTensor;
  oneHot(input: NativeTensor, numClasses?: number): NativeTensor;
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
  huberLoss(
    input: NativeTensor,
    target: NativeTensor,
    reduction?: string,
    delta?: number
  ): NativeTensor;
  cosineSimilarity(
    x1: NativeTensor,
    x2: NativeTensor,
    dim?: number,
    eps?: number
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
  lstm(
    input: NativeTensor,
    hx: NativeTensor[],
    params: NativeTensor[],
    hasBiases: boolean,
    numLayers: number,
    dropout: number,
    train: boolean,
    bidirectional: boolean,
    batchFirst: boolean
  ): { output: NativeTensor; hn: NativeTensor; cn: NativeTensor };
  gru(
    input: NativeTensor,
    hx: NativeTensor,
    params: NativeTensor[],
    hasBiases: boolean,
    numLayers: number,
    dropout: number,
    train: boolean,
    bidirectional: boolean,
    batchFirst: boolean
  ): { output: NativeTensor; hn: NativeTensor };
  rnn(
    input: NativeTensor,
    hx: NativeTensor,
    params: NativeTensor[],
    hasBiases: boolean,
    numLayers: number,
    dropout: number,
    train: boolean,
    bidirectional: boolean,
    batchFirst: boolean,
    nonlinearity?: string
  ): { output: NativeTensor; hn: NativeTensor };
  fusedAdamStep(
    params: NativeTensor[],
    grads: NativeTensor[],
    expAvgs: NativeTensor[],
    expAvgSqs: NativeTensor[],
    maxExpAvgSqs: NativeTensor[],
    stateSteps: NativeTensor[],
    lr: number,
    beta1: number,
    beta2: number,
    weightDecay: number,
    eps: number,
    amsgrad: boolean,
    decoupled: boolean,
    gradScale?: NativeTensor | null,
    foundInf?: NativeTensor | null
  ): void;
  complex(real: NativeTensor, imag: NativeTensor): NativeTensor;
  polar(abs: NativeTensor, angle: NativeTensor): NativeTensor;
  angle(input: NativeTensor): NativeTensor;
  viewAsReal(input: NativeTensor): NativeTensor;
  viewAsComplex(input: NativeTensor): NativeTensor;
  sparseCooTensor(indices: NativeTensor, values: NativeTensor, size?: number[] | null): NativeTensor;
  sparseMm(mat1: NativeTensor, mat2: NativeTensor): NativeTensor;
}

export interface NativeAddon {
  NativeTensor: new () => NativeTensor;
  NativeGenerator: new (device?: string) => NativeGenerator;
  NativeProcessGroup: new (opts: NativeProcessGroupOptions) => NativeProcessGroup;
  NativeCudaStream?: new (device?: number, priority?: number) => NativeCudaStream;
  NativeCudaEvent?: new (
    enableTiming?: boolean,
    blocking?: boolean,
    interprocess?: boolean
  ) => NativeCudaEvent;
  NativeCudaGraph?: new (keepGraph?: boolean) => NativeCudaGraph;
  NativeAotiModel: new (
    packagePath: string,
    modelName?: string,
    deviceIndex?: number
  ) => NativeAotiModel;
  NativeJitModule: new (path: string, device?: string) => NativeJitModule;

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
  enterDualLevel(): number;
  exitDualLevel(level: number): void;
  makeDual(primal: NativeTensor, tangent: NativeTensor, level: number): NativeTensor;
  unpackDual(dual: NativeTensor, level: number): { primal: NativeTensor; tangent: NativeTensor | null };

  cudaIsAvailable(): boolean;
  cudaDeviceCount(): number;
  cudaCurrentDevice(): number;
  cudaSetDevice(index: number): void;
  cudaSynchronize(index?: number): void;
  cudaEmptyCache(): void;
  cudaSetMemoryFraction(fraction: number, index?: number): void;
  cudaResetPeakMemoryStats(index?: number): void;
  cudaMemoryStats(index?: number): CudaMemoryStats;
  cudaDeviceName(index?: number): string;
  cudaDeviceProperties(index?: number): CudaDeviceProperties;
  cudaManualSeed(seed: number | bigint): void;
  cudaManualSeedAll(seed: number | bigint): void;
  cudaCurrentStream(index?: number): NativeCudaStream;
  cudaDefaultStream(index?: number): NativeCudaStream;
  cudaSetStream(stream: NativeCudaStream): void;
  cudaGraphPoolHandle(): [bigint, bigint];
  mpsIsAvailable(): boolean;
  distIsNcclAvailable(): boolean;
  distIsGlooAvailable(): boolean;
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
  const prebuildRel = path.join('prebuilds', `${process.platform}-${process.arch}`, 'norch_native.node');
  const candidates = [
    process.env.NORCH_NATIVE_PATH,
    path.resolve(__dirname, '..', 'build', 'norch_native.node'),
    path.resolve(__dirname, '..', '..', 'build', 'norch_native.node'),
    path.resolve(__dirname, '..', prebuildRel),
    path.resolve(__dirname, '..', '..', prebuildRel),
  ].filter((p): p is string => Boolean(p));

  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }
  throw new Error(
    `Could not find norch_native.node. Run "npm run build:native" first. Searched: ${candidates.join(', ')}`
  );
}

export const native: NativeAddon = require(resolveAddonPath()) as NativeAddon;
