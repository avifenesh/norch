import * as dtypes from './dtype.js';
import * as deviceMod from './device.js';
import * as generatorMod from './generator.js';
import * as memoryMod from './memory.js';
import * as tensorMod from './tensor.js';
import * as opsMod from './ops.js';
import * as autogradMod from './autograd.js';
import * as cudaMod from './cuda.js';
import * as linalgMod from './linalg.js';
import * as fftMod from './fft.js';
import * as specialMod from './special.js';
import * as distributionsMod from './distributions.js';
import * as distributedMod from './distributed.js';
import * as exportMod from './export.js';
import * as nn from './nn/index.js';
import * as optim from './optim/index.js';
import * as utils from './utils/index.js';
import * as amp from './amp.js';

export * from './dtype.js';
export * from './device.js';
export * from './generator.js';
export * from './memory.js';
export * from './tensor.js';
export * from './ops.js';
export * from './autograd.js';
export * from './cuda.js';
export * from './linalg.js';
export * from './fft.js';
export * from './special.js';
export * from './distributions.js';
export * from './distributed.js';
export * from './export.js';
export { nn, optim, utils, amp };
export { functional as F } from './functional.js';

export const torch = {
  // Dtypes
  float32: dtypes.float32,
  float: dtypes.float,
  float64: dtypes.float64,
  double: dtypes.double,
  float16: dtypes.float16,
  half: dtypes.half,
  bfloat16: dtypes.bfloat16,
  int8: dtypes.int8,
  uint8: dtypes.uint8,
  byte: dtypes.byte,
  int16: dtypes.int16,
  short: dtypes.short,
  int32: dtypes.int32,
  int: dtypes.int,
  int64: dtypes.int64,
  long: dtypes.long,
  bool: dtypes.bool,
  complex64: dtypes.complex64,
  complex128: dtypes.complex128,
  float8_e4m3fn: dtypes.float8_e4m3fn,
  float8_e5m2: dtypes.float8_e5m2,

  // Device, Generator & Runtime
  Device: deviceMod.Device,
  device: deviceMod.device,
  Generator: generatorMod.Generator,
  cuda: cudaMod.cuda,
  mps: cudaMod.mps,
  manualSeed: cudaMod.manualSeed,
  getNumThreads: cudaMod.getNumThreads,
  setNumThreads: cudaMod.setNumThreads,
  libtorchVersion: cudaMod.libtorchVersion,

  // Memory
  tidy: memoryMod.tidy,
  keep: memoryMod.keep,
  scope: memoryMod.scope,
  dispose: memoryMod.dispose,
  memoryStats: memoryMod.memoryStats,

  // Tensor & Indexing
  Tensor: tensorMod.Tensor,
  slice: tensorMod.slice,
  Ellipsis: tensorMod.Ellipsis,

  // Factories & Ops
  tensor: opsMod.tensor,
  fromTypedArray: opsMod.fromTypedArray,
  empty: opsMod.empty,
  zeros: opsMod.zeros,
  ones: opsMod.ones,
  full: opsMod.full,
  rand: opsMod.rand,
  randn: opsMod.randn,
  randint: opsMod.randint,
  randperm: opsMod.randperm,
  arange: opsMod.arange,
  linspace: opsMod.linspace,
  eye: opsMod.eye,
  zerosLike: opsMod.zerosLike,
  onesLike: opsMod.onesLike,
  emptyLike: opsMod.emptyLike,
  fullLike: opsMod.fullLike,
  randnLike: opsMod.randnLike,
  add: opsMod.add,
  sub: opsMod.sub,
  mul: opsMod.mul,
  div: opsMod.div,
  matmul: opsMod.matmul,
  bmm: opsMod.bmm,
  einsum: opsMod.einsum,
  cat: opsMod.cat,
  stack: opsMod.stack,
  split: opsMod.split,
  chunk: opsMod.chunk,
  where: opsMod.where,
  tril: opsMod.tril,
  triu: opsMod.triu,
  sum: opsMod.sum,
  mean: opsMod.mean,
  softmax: opsMod.softmax,
  logSoftmax: opsMod.logSoftmax,
  clamp: opsMod.clamp,
  exp: opsMod.exp,
  log: opsMod.log,
  sqrt: opsMod.sqrt,
  sin: opsMod.sin,
  cos: opsMod.cos,
  tanh: opsMod.tanh,
  sigmoid: opsMod.sigmoid,
  relu: opsMod.relu,
  abs: opsMod.abs,
  allclose: opsMod.allclose,
  complex: opsMod.complex,
  polar: opsMod.polar,
  angle: opsMod.angle,
  viewAsReal: opsMod.viewAsReal,
  viewAsComplex: opsMod.viewAsComplex,
  sparseCooTensor: opsMod.sparseCooTensor,
  sparse: opsMod.sparse,
  ops: opsMod.ops,
  linalg: linalgMod.linalg,
  fft: fftMod.fft,
  special: specialMod.special,
  distributions: distributionsMod.distributions,
  distributed: distributedMod.distributed,

  // Autograd
  isGradEnabled: autogradMod.isGradEnabled,
  setGradEnabled: autogradMod.setGradEnabled,
  noGrad: autogradMod.noGrad,
  noGradScope: autogradMod.noGradScope,
  enableGrad: autogradMod.enableGrad,
  enableGradScope: autogradMod.enableGradScope,
  isInferenceModeEnabled: autogradMod.isInferenceModeEnabled,
  inferenceMode: autogradMod.inferenceMode,
  inferenceModeScope: autogradMod.inferenceModeScope,
  isAutocastEnabled: autogradMod.isAutocastEnabled,
  autocast: autogradMod.autocast,
  backward: autogradMod.backward,
  grad: autogradMod.grad,
  AutogradContext: autogradMod.AutogradContext,
  AutogradFunction: autogradMod.AutogradFunction,
  forwardAD: autogradMod.forwardAD,

  // Exported model runners
  AotiModel: exportMod.AotiModel,
  loadAotiModel: exportMod.loadAotiModel,
  JitModule: exportMod.JitModule,
  loadJitModule: exportMod.loadJitModule,

  // High-level modules & checkpoints
  nn,
  F: nn.F,
  optim,
  utils,
  amp,
  save: utils.saveSafetensors,
  load: utils.loadSafetensors,
  saveCheckpoint: utils.saveCheckpoint,
  loadCheckpoint: utils.loadCheckpoint,
};

export namespace torch {
  export type Tensor = tensorMod.Tensor;
  export type DType = dtypes.DType;
  export type Device = deviceMod.Device;
  export type DeviceLike = deviceMod.DeviceLike;
  export type Generator = generatorMod.Generator;
  export type TensorOptions = opsMod.TensorOptions;
  export type Parameter = nn.Parameter;
  export type Module = nn.Module;
  export type Optimizer = optim.Optimizer;
}

export default torch;
