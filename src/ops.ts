import type { DType } from './dtype.js';
import { normalizeDevice, type DeviceLike } from './device.js';
import type { Generator } from './generator.js';
import {
  native,
  type NativeTensor,
  type NativeTensorOptions,
  type OpSchema,
  type Scalar,
  type TypedArray,
} from './native.js';
import { Tensor, wrapResult, wrapTensor, type ReductionOptions } from './tensor.js';
import { aten, ATEN_OP_NAMES } from './generated/aten_ops.js';

export interface TensorOptions {
  dtype?: DType;
  device?: DeviceLike;
  layout?: string;
  pinMemory?: boolean;
  requiresGrad?: boolean;
  generator?: Generator;
}

function toNativeOpts(opts?: TensorOptions): NativeTensorOptions | undefined {
  if (!opts) return undefined;
  return {
    dtype: opts.dtype,
    device: normalizeDevice(opts.device),
    layout: opts.layout,
    pinMemory: opts.pinMemory,
    requiresGrad: opts.requiresGrad,
    generator: opts.generator?._native,
  };
}

// ---------- Factories ----------

export function tensor(data: unknown, opts?: TensorOptions): Tensor {
  if (ArrayBuffer.isView(data) && !(data instanceof DataView)) {
    return fromTypedArray(data as TypedArray, undefined, { ...opts, copy: true });
  }
  const arg = data instanceof Tensor ? data._native : data;
  return wrapTensor(native.fast.tensor(arg, toNativeOpts(opts)));
}

export interface FromTypedArrayOptions extends TensorOptions {
  /** If false (default for fromTypedArray), shares CPU memory zero-copy when dtype & device match. */
  copy?: boolean;
}

export function fromTypedArray(
  data: TypedArray,
  shape?: number[],
  opts?: FromTypedArrayOptions
): Tensor {
  return wrapTensor(
    native.fromTypedArray(data, shape ?? null, {
      copy: opts?.copy ?? false,
      dtype: opts?.dtype,
      device: normalizeDevice(opts?.device),
      requiresGrad: opts?.requiresGrad,
    })
  );
}

export function empty(shape: number[], opts?: TensorOptions): Tensor {
  return wrapTensor(native.fast.empty(shape, toNativeOpts(opts)));
}

export function zeros(shape: number[], opts?: TensorOptions): Tensor {
  return wrapTensor(native.fast.zeros(shape, toNativeOpts(opts)));
}

export function ones(shape: number[], opts?: TensorOptions): Tensor {
  return wrapTensor(native.fast.ones(shape, toNativeOpts(opts)));
}

export function full(shape: number[], fillValue: Scalar, opts?: TensorOptions): Tensor {
  return wrapTensor(native.fast.full(shape, fillValue, toNativeOpts(opts)));
}

export function rand(shape: number[], opts?: TensorOptions): Tensor {
  return wrapTensor(native.fast.rand(shape, toNativeOpts(opts)));
}

export function randn(shape: number[], opts?: TensorOptions): Tensor {
  return wrapTensor(native.fast.randn(shape, toNativeOpts(opts)));
}

export function randint(
  low: number,
  high: number,
  shape: number[],
  opts?: TensorOptions
): Tensor {
  return wrapTensor(native.fast.randint(low, high, shape, toNativeOpts(opts)));
}

export function randperm(n: number, opts?: TensorOptions): Tensor {
  return wrapTensor(native.fast.randperm(n, toNativeOpts(opts)));
}

export function arange(end: number, opts?: TensorOptions): Tensor;
export function arange(start: number, end: number, step?: number, opts?: TensorOptions): Tensor;
export function arange(
  a: number,
  b?: number | TensorOptions,
  c?: number | TensorOptions,
  d?: TensorOptions
): Tensor {
  if (b === undefined || typeof b === 'object') {
    return wrapTensor(native.fast.arange(0, a, 1, toNativeOpts(b)));
  }
  if (typeof c === 'object') {
    return wrapTensor(native.fast.arange(a, b, 1, toNativeOpts(c)));
  }
  return wrapTensor(native.fast.arange(a, b, c ?? 1, toNativeOpts(d)));
}

export function linspace(
  start: number,
  end: number,
  steps: number,
  opts?: TensorOptions
): Tensor {
  return wrapTensor(native.fast.linspace(start, end, steps, toNativeOpts(opts)));
}

export function eye(n: number, m?: number, opts?: TensorOptions): Tensor {
  return wrapTensor(native.fast.eye(n, m ?? n, toNativeOpts(opts)));
}

export function zerosLike(t: Tensor): Tensor {
  return wrapTensor(native.fast.zerosLike(t._native));
}

export function onesLike(t: Tensor): Tensor {
  return wrapTensor(native.fast.onesLike(t._native));
}

export function emptyLike(t: Tensor): Tensor {
  return wrapTensor(native.fast.emptyLike(t._native));
}

export function fullLike(t: Tensor, fillValue: Scalar): Tensor {
  return wrapTensor(native.fast.fullLike(t._native, fillValue));
}

export function randnLike(t: Tensor): Tensor {
  return wrapTensor(native.fast.randnLike(t._native));
}

// ---------- Functional Math & Shape ----------

export function add(a: Tensor, b: Tensor | Scalar, alpha?: Scalar): Tensor {
  return a.add(b, alpha);
}

export function sub(a: Tensor, b: Tensor | Scalar, alpha?: Scalar): Tensor {
  return a.sub(b, alpha);
}

export function mul(a: Tensor, b: Tensor | Scalar): Tensor {
  return a.mul(b);
}

export function div(a: Tensor, b: Tensor | Scalar): Tensor {
  return a.div(b);
}

export function matmul(a: Tensor, b: Tensor): Tensor {
  return a.matmul(b);
}

export function bmm(a: Tensor, b: Tensor): Tensor {
  return a.bmm(b);
}

export function einsum(equation: string, ...operands: Tensor[]): Tensor {
  return wrapTensor(
    native.fast.einsum(
      equation,
      operands.map((t) => t._native)
    )
  );
}

export function cat(tensors: Tensor[], dim = 0): Tensor {
  return wrapTensor(
    native.fast.cat(
      tensors.map((t) => t._native),
      dim
    )
  );
}

export function stack(tensors: Tensor[], dim = 0): Tensor {
  return wrapTensor(
    native.fast.stack(
      tensors.map((t) => t._native),
      dim
    )
  );
}

export function split(t: Tensor, splitSizeOrSections: number | number[], dim = 0): Tensor[] {
  return t.split(splitSizeOrSections, dim);
}

export function chunk(t: Tensor, chunks: number, dim = 0): Tensor[] {
  return t.chunk(chunks, dim);
}

export function where(condition: Tensor, x: Tensor, y: Tensor): Tensor {
  return wrapTensor(native.fast.where(condition._native, x._native, y._native));
}

export function tril(t: Tensor, diagonal = 0): Tensor {
  return t.tril(diagonal);
}

export function triu(t: Tensor, diagonal = 0): Tensor {
  return t.triu(diagonal);
}

export function sum(t: Tensor, dimOrOpts?: number | number[] | ReductionOptions, keepdim = false): Tensor {
  return t.sum(dimOrOpts, keepdim);
}

export function mean(t: Tensor, dimOrOpts?: number | number[] | ReductionOptions, keepdim = false): Tensor {
  return t.mean(dimOrOpts, keepdim);
}

export function softmax(t: Tensor, dim: number): Tensor {
  return t.softmax(dim);
}

export function logSoftmax(t: Tensor, dim: number): Tensor {
  return t.logSoftmax(dim);
}

export function clamp(t: Tensor, min?: Scalar | null, max?: Scalar | null): Tensor {
  return t.clamp(min, max);
}

export function exp(t: Tensor): Tensor {
  return t.exp();
}

export function log(t: Tensor): Tensor {
  return t.log();
}

export function sqrt(t: Tensor): Tensor {
  return t.sqrt();
}

export function sin(t: Tensor): Tensor {
  return t.sin();
}

export function cos(t: Tensor): Tensor {
  return t.cos();
}

export function tanh(t: Tensor): Tensor {
  return t.tanh();
}

export function sigmoid(t: Tensor): Tensor {
  return t.sigmoid();
}

export function relu(t: Tensor): Tensor {
  return t.relu();
}

export function abs(t: Tensor): Tensor {
  return t.abs();
}

export function allclose(a: Tensor, b: Tensor, rtol = 1e-5, atol = 1e-8, equalNan = false): boolean {
  return a.allclose(b, rtol, atol, equalNan);
}

// ---------- Generic c10::Dispatcher access ----------

export function callOp(
  name: string,
  overload: string,
  posArgs: unknown[],
  kwArgs?: Record<string, unknown>
): unknown {
  const rawPos = posArgs.map((a) => (a instanceof Tensor ? a._native : a));
  let rawKw: Record<string, unknown> | undefined;
  if (kwArgs) {
    rawKw = {};
    for (const [k, v] of Object.entries(kwArgs)) {
      rawKw[k] = v instanceof Tensor ? v._native : v;
    }
  }
  return wrapResult(native.callOp(name, overload, rawPos, rawKw));
}

export function listOps(prefix?: string): { name: string; overload: string }[] {
  return native.listOps(prefix);
}

export function getOpSchema(name: string, overload = ''): OpSchema | null {
  return native.getOpSchema(name, overload);
}

export const ops = {
  aten,
  ATEN_OP_NAMES,
  callOp,
  listOps,
  getOpSchema,
  call(name: string, posArgs: unknown[], kwArgs?: Record<string, unknown>): unknown {
    const rawPos = posArgs.map((a) => (a instanceof Tensor ? a._native : a));
    let rawKw: Record<string, unknown> | undefined;
    if (kwArgs) {
      rawKw = {};
      for (const [k, v] of Object.entries(kwArgs)) {
        rawKw[k] = v instanceof Tensor ? v._native : v;
      }
    }
    return wrapResult(native.callOpAuto(name, rawPos, rawKw) as NativeTensor);
  },
};
