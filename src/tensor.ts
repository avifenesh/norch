import { type DType } from './dtype.js';
import { Device, normalizeDevice, type DeviceLike } from './device.js';
import { trackTensor, untrackTensor } from './memory.js';
import {
  native,
  type NativeBackwardOptions,
  type NativeTensor,
  type Scalar,
  type TypedArray,
} from './native.js';

export interface SliceIndex {
  readonly __slice: true;
  readonly start: number | null;
  readonly stop: number | null;
  readonly step: number | null;
}

export const Ellipsis = '...' as const;

export function slice(
  start?: number | null,
  stop?: number | null,
  step?: number | null
): SliceIndex {
  return {
    __slice: true,
    start: start ?? null,
    stop: stop ?? null,
    step: step ?? null,
  };
}

export type TensorIndexItem =
  | number
  | bigint
  | boolean
  | null
  | typeof Ellipsis
  | ':'
  | SliceIndex
  | Tensor;

export interface ReductionOptions {
  dim?: number | number[];
  keepdim?: boolean;
  dtype?: DType;
}

export interface ToOptions {
  device?: DeviceLike;
  dtype?: DType;
  nonBlocking?: boolean;
  copy?: boolean;
  memoryFormat?: string;
}

export interface BackwardOptions {
  retainGraph?: boolean;
  createGraph?: boolean;
  inputs?: Tensor[];
}

export class Tensor implements Disposable {
  readonly _native: NativeTensor;

  constructor(handle: NativeTensor) {
    this._native = handle;
    trackTensor(this);
  }

  // ---------- Lifecycle / Disposal ----------

  get isDisposed(): boolean {
    return this._native.isDisposed();
  }

  dispose(): void {
    untrackTensor(this);
    this._native.dispose();
  }

  [Symbol.dispose](): void {
    this.dispose();
  }

  // ---------- Metadata ----------

  get shape(): number[] {
    return this._native.shape();
  }

  size(): number[];
  size(dim: number): number;
  size(dim?: number): number[] | number {
    const s = this._native.shape();
    if (dim === undefined) return s;
    const idx = dim < 0 ? s.length + dim : dim;
    const val = s[idx];
    if (val === undefined) {
      throw new RangeError(`Dimension out of range (expected to be in [-${s.length}, ${s.length - 1}], got ${dim})`);
    }
    return val;
  }

  get strides(): number[] {
    return this._native.strides();
  }

  get ndim(): number {
    return this._native.dim();
  }

  get dim(): number {
    return this._native.dim();
  }

  get numel(): number {
    return this._native.numel();
  }

  get elementSize(): number {
    return this._native.elementSize();
  }

  get nbytes(): number {
    return this._native.nbytes();
  }

  get dtype(): DType {
    return this._native.dtype();
  }

  get device(): Device {
    return new Device(this._native.device());
  }

  get layout(): string {
    return this._native.layout();
  }

  get isContiguous(): boolean {
    return this._native.isContiguous();
  }

  get isCuda(): boolean {
    return this._native.isCuda();
  }

  get isPinned(): boolean {
    return this._native.isPinned();
  }

  pinMemory(): Tensor {
    return wrapTensor(this._native.pinMemory());
  }

  get isComplex(): boolean {
    return this._native.isComplex();
  }

  get isSparse(): boolean {
    return this._native.isSparse();
  }

  get real(): Tensor {
    return wrapTensor(this._native.real());
  }

  get imag(): Tensor {
    return wrapTensor(this._native.imag());
  }

  conj(): Tensor {
    return wrapTensor(this._native.conj());
  }

  angle(): Tensor {
    return wrapTensor(native.fast.angle(this._native));
  }

  toDense(): Tensor {
    return wrapTensor(this._native.toDense());
  }

  toSparse(): Tensor {
    return wrapTensor(this._native.toSparseCoo());
  }

  toSparseCsr(): Tensor {
    return wrapTensor(this._native.toSparseCsr());
  }

  toSparseCsc(): Tensor {
    return wrapTensor(this._native.toSparseCsc());
  }

  coalesce(): Tensor {
    return wrapTensor(this._native.coalesce());
  }

  indices(): Tensor {
    return wrapTensor(this._native.sparseIndices());
  }

  values(): Tensor {
    return wrapTensor(this._native.sparseValues());
  }

  crowIndices(): Tensor {
    return wrapTensor(this._native.crowIndices());
  }

  colIndices(): Tensor {
    return wrapTensor(this._native.colIndices());
  }

  ccolIndices(): Tensor {
    return wrapTensor(this._native.ccolIndices());
  }

  rowIndices(): Tensor {
    return wrapTensor(this._native.rowIndices());
  }

  get sparseDim(): number {
    return this._native.sparseDim();
  }

  get denseDim(): number {
    return this._native.denseDim();
  }

  get nnz(): number {
    return this._native.nnz();
  }

  get requiresGrad(): boolean {
    return this._native.requiresGrad();
  }

  set requiresGrad(req: boolean) {
    this._native.setRequiresGrad(req);
  }

  requiresGrad_(req = true): this {
    this._native.setRequiresGrad(req);
    return this;
  }

  get isLeaf(): boolean {
    return this._native.isLeaf();
  }

  get retainsGrad(): boolean {
    return this._native.retainsGrad();
  }

  retainGrad(): void {
    this._native.retainGrad();
  }

  get grad(): Tensor | null {
    const g = this._native.grad();
    return g ? wrapTensor(g) : null;
  }

  set grad(value: Tensor | null) {
    this._native.setGrad(value ? value._native : null);
  }

  get gradFnName(): string | null {
    return this._native.gradFnName();
  }

  get dataPtr(): bigint {
    return this._native.dataPtr();
  }

  get storagePtr(): bigint {
    return this._native.storagePtr();
  }

  get T(): Tensor {
    const d = this.ndim;
    if (d < 2) return this;
    return this.permute(Array.from({ length: d }, (_, i) => d - 1 - i));
  }

  // ---------- Data Interop ----------

  item(): number {
    const v = this._native.item();
    if (typeof v === 'number') return v;
    if (typeof v === 'boolean') return v ? 1 : 0;
    if (typeof v === 'bigint') return Number(v);
    throw new Error('Complex tensor cannot be converted to a single JS number');
  }

  itemScalar(): Scalar | { real: number; imag: number } {
    return this._native.item();
  }

  /** Safe copy into a JS TypedArray (moves to CPU & contiguous automatically). */
  toTypedArray(): TypedArray {
    return this._native.toTypedArray();
  }

  /** Non-blocking copy to CPU + TypedArray on a libuv worker thread. */
  toTypedArrayAsync(): Promise<TypedArray> {
    return this._native.toTypedArrayAsync();
  }

  /** Zero-copy TypedArray view directly over CPU contiguous tensor storage. */
  dataView(): TypedArray {
    return this._native.dataView();
  }

  /** Convert tensor values into nested JS arrays matching tensor.shape. */
  toArray(): unknown {
    const shape = this.shape;
    if (shape.length === 0) {
      return this._native.item();
    }
    const ta = this.toTypedArray();
    const isBool = this.dtype === 'bool';
    const isComplex = this.isComplex;
    let offset = 0;
    const build = (depth: number): unknown => {
      const len = shape[depth]!;
      if (depth === shape.length - 1) {
        const out = new Array<unknown>(len);
        for (let i = 0; i < len; i++) {
          if (isComplex) {
            const re = Number(ta[offset++]);
            const im = Number(ta[offset++]);
            out[i] = { real: re, imag: im };
          } else {
            const raw = ta[offset++];
            out[i] = isBool ? Boolean(raw) : typeof raw === 'bigint' ? Number(raw) : raw;
          }
        }
        return out;
      }
      const out = new Array<unknown>(len);
      for (let i = 0; i < len; i++) {
        out[i] = build(depth + 1);
      }
      return out;
    };
    return build(0);
  }

  // ---------- Transfers / Copies / Autograd ----------

  clone(): Tensor {
    return wrapTensor(this._native.clone());
  }

  detach(): Tensor {
    return wrapTensor(this._native.detach());
  }

  detach_(): this {
    this._native.detachInplace();
    return this;
  }

  contiguous(memoryFormat?: string): Tensor {
    return wrapTensor(this._native.contiguous(memoryFormat));
  }

  copy_(src: Tensor, nonBlocking = false): this {
    this._native.copyFrom(src._native, nonBlocking);
    return this;
  }

  to(target: DeviceLike | DType | ToOptions): Tensor {
    if (typeof target === 'string') {
      if (
        target === 'cpu' ||
        target === 'cuda' ||
        target.startsWith('cuda:') ||
        target === 'mps' ||
        target.startsWith('mps:') ||
        target === 'meta'
      ) {
        return wrapTensor(this._native.to({ device: target }));
      }
      return wrapTensor(this._native.to({ dtype: target as DType }));
    }
    if (target instanceof Device) {
      return wrapTensor(this._native.to({ device: target.toString() }));
    }
    const opts = target as ToOptions;
    return wrapTensor(
      this._native.to({
        device: normalizeDevice(opts.device),
        dtype: opts.dtype,
        nonBlocking: opts.nonBlocking,
        copy: opts.copy,
        memoryFormat: opts.memoryFormat,
      })
    );
  }

  cpu(): Tensor {
    return this.to({ device: 'cpu' });
  }

  cuda(index?: number): Tensor {
    return this.to({ device: index === undefined ? 'cuda' : `cuda:${index}` });
  }

  float(): Tensor {
    return this.to({ dtype: 'float32' });
  }

  double(): Tensor {
    return this.to({ dtype: 'float64' });
  }

  half(): Tensor {
    return this.to({ dtype: 'float16' });
  }

  bfloat16(): Tensor {
    return this.to({ dtype: 'bfloat16' });
  }

  int(): Tensor {
    return this.to({ dtype: 'int32' });
  }

  long(): Tensor {
    return this.to({ dtype: 'int64' });
  }

  bool(): Tensor {
    return this.to({ dtype: 'bool' });
  }

  backward(grad?: Tensor | null, opts?: BackwardOptions): void {
    const nativeOpts: NativeBackwardOptions | undefined = opts
      ? {
          retainGraph: opts.retainGraph,
          createGraph: opts.createGraph,
          inputs: opts.inputs?.map((t) => t._native),
        }
      : undefined;
    this._native.backward(grad ? grad._native : undefined, nativeOpts);
  }

  registerHook(
    hook: (grad: Tensor) => Tensor | null | undefined | void
  ): { remove(): void; [Symbol.dispose](): void } {
    const handleId = this._native.registerHook((nativeGrad) => {
      const wrapped = wrapTensor(nativeGrad);
      const res = hook(wrapped);
      if (res instanceof Tensor) {
        return res._native;
      }
      return undefined;
    });
    let removed = false;
    const remove = (): void => {
      if (!removed && !this.isDisposed) {
        removed = true;
        this._native.removeHook(handleId);
      }
    };
    return {
      remove,
      [Symbol.dispose]: remove,
    };
  }

  // ---------- Indexing ----------

  at(...indices: TensorIndexItem[]): Tensor {
    const raw = indices.map((idx) => (idx instanceof Tensor ? idx._native : idx));
    return wrapTensor(this._native.indexGet(raw));
  }

  index(indices: TensorIndexItem[]): Tensor {
    const raw = indices.map((idx) => (idx instanceof Tensor ? idx._native : idx));
    return wrapTensor(this._native.indexGet(raw));
  }

  indexPut_(indices: TensorIndexItem[], value: Tensor | Scalar, accumulate = false): this {
    const raw = indices.map((idx) => (idx instanceof Tensor ? idx._native : idx));
    this._native.indexPut(raw, value instanceof Tensor ? value._native : value, accumulate);
    return this;
  }

  // ---------- Arithmetic / Math (unboxed C++ fast path) ----------

  add(other: Tensor | Scalar, alpha?: Scalar): Tensor {
    return wrapTensor(native.fast.add(this._native, other instanceof Tensor ? other._native : other, alpha));
  }

  sub(other: Tensor | Scalar, alpha?: Scalar): Tensor {
    return wrapTensor(native.fast.sub(this._native, other instanceof Tensor ? other._native : other, alpha));
  }

  mul(other: Tensor | Scalar): Tensor {
    return wrapTensor(native.fast.mul(this._native, other instanceof Tensor ? other._native : other));
  }

  div(other: Tensor | Scalar): Tensor {
    return wrapTensor(native.fast.div(this._native, other instanceof Tensor ? other._native : other));
  }

  pow(exponent: Tensor | Scalar): Tensor {
    return wrapTensor(
      native.fast.pow(this._native, exponent instanceof Tensor ? exponent._native : exponent)
    );
  }

  matmul(other: Tensor): Tensor {
    return wrapTensor(native.fast.matmul(this._native, other._native));
  }

  bmm(other: Tensor): Tensor {
    return wrapTensor(native.fast.bmm(this._native, other._native));
  }

  neg(): Tensor {
    return wrapTensor(native.fast.neg(this._native));
  }

  abs(): Tensor {
    return wrapTensor(native.fast.abs(this._native));
  }

  exp(): Tensor {
    return wrapTensor(native.fast.exp(this._native));
  }

  log(): Tensor {
    return wrapTensor(native.fast.log(this._native));
  }

  sqrt(): Tensor {
    return wrapTensor(native.fast.sqrt(this._native));
  }

  rsqrt(): Tensor {
    return wrapTensor(native.fast.rsqrt(this._native));
  }

  sin(): Tensor {
    return wrapTensor(native.fast.sin(this._native));
  }

  cos(): Tensor {
    return wrapTensor(native.fast.cos(this._native));
  }

  tanh(): Tensor {
    return wrapTensor(native.fast.tanh(this._native));
  }

  sigmoid(): Tensor {
    return wrapTensor(native.fast.sigmoid(this._native));
  }

  relu(): Tensor {
    return wrapTensor(native.fast.relu(this._native));
  }

  gelu(approximate: 'none' | 'tanh' = 'none'): Tensor {
    return wrapTensor(native.fast.gelu(this._native, approximate));
  }

  silu(): Tensor {
    return wrapTensor(native.fast.silu(this._native));
  }

  leakyRelu(negativeSlope = 0.01): Tensor {
    return wrapTensor(native.fast.leakyRelu(this._native, negativeSlope));
  }

  clamp(min?: Scalar | null, max?: Scalar | null): Tensor {
    return wrapTensor(native.fast.clamp(this._native, min ?? null, max ?? null));
  }

  isnan(): Tensor {
    return wrapTensor(native.fast.isnan(this._native));
  }

  isinf(): Tensor {
    return wrapTensor(native.fast.isinf(this._native));
  }

  isfinite(): Tensor {
    return wrapTensor(native.fast.isfinite(this._native));
  }

  // ---------- Comparisons (via c10::Dispatcher) ----------

  eq(other: Tensor | Scalar): Tensor {
    return wrapTensor(
      native.callOpAuto('eq', [this._native, other instanceof Tensor ? other._native : other]) as NativeTensor
    );
  }

  ne(other: Tensor | Scalar): Tensor {
    return wrapTensor(
      native.callOpAuto('ne', [this._native, other instanceof Tensor ? other._native : other]) as NativeTensor
    );
  }

  lt(other: Tensor | Scalar): Tensor {
    return wrapTensor(
      native.callOpAuto('lt', [this._native, other instanceof Tensor ? other._native : other]) as NativeTensor
    );
  }

  le(other: Tensor | Scalar): Tensor {
    return wrapTensor(
      native.callOpAuto('le', [this._native, other instanceof Tensor ? other._native : other]) as NativeTensor
    );
  }

  gt(other: Tensor | Scalar): Tensor {
    return wrapTensor(
      native.callOpAuto('gt', [this._native, other instanceof Tensor ? other._native : other]) as NativeTensor
    );
  }

  ge(other: Tensor | Scalar): Tensor {
    return wrapTensor(
      native.callOpAuto('ge', [this._native, other instanceof Tensor ? other._native : other]) as NativeTensor
    );
  }

  equal(other: Tensor): boolean {
    return this._native.equal(other._native);
  }

  allclose(other: Tensor, rtol = 1e-5, atol = 1e-8, equalNan = false): boolean {
    return Boolean(
      native.callOpAuto('allclose', [this._native, other._native, rtol, atol, equalNan])
    );
  }

  // ---------- In-place ops ----------

  add_(other: Tensor | Scalar, alpha?: Scalar): this {
    native.fast.add_(this._native, other instanceof Tensor ? other._native : other, alpha);
    return this;
  }

  sub_(other: Tensor | Scalar, alpha?: Scalar): this {
    native.fast.sub_(this._native, other instanceof Tensor ? other._native : other, alpha);
    return this;
  }

  mul_(other: Tensor | Scalar): this {
    native.fast.mul_(this._native, other instanceof Tensor ? other._native : other);
    return this;
  }

  div_(other: Tensor | Scalar): this {
    native.fast.div_(this._native, other instanceof Tensor ? other._native : other);
    return this;
  }

  zero_(): this {
    native.fast.zero_(this._native);
    return this;
  }

  fill_(value: Scalar): this {
    native.fast.fill_(this._native, value);
    return this;
  }

  uniform_(from = 0, to = 1): this {
    native.fast.uniform_(this._native, from, to);
    return this;
  }

  normal_(mean = 0, std = 1): this {
    native.fast.normal_(this._native, mean, std);
    return this;
  }

  clamp_(min?: Scalar | null, max?: Scalar | null): this {
    native.fast.clamp_(this._native, min ?? null, max ?? null);
    return this;
  }

  addcmul_(tensor1: Tensor, tensor2: Tensor, value: Scalar = 1): this {
    native.fast.addcmul_(this._native, tensor1._native, tensor2._native, value);
    return this;
  }

  addcdiv_(tensor1: Tensor, tensor2: Tensor, value: Scalar = 1): this {
    native.fast.addcdiv_(this._native, tensor1._native, tensor2._native, value);
    return this;
  }

  // ---------- Reductions ----------

  sum(dimOrOpts?: number | number[] | ReductionOptions, keepdim = false): Tensor {
    if (typeof dimOrOpts === 'object' && !Array.isArray(dimOrOpts)) {
      return wrapTensor(
        native.fast.sum(
          this._native,
          dimOrOpts.dim ?? null,
          dimOrOpts.keepdim ?? false,
          dimOrOpts.dtype ?? null
        )
      );
    }
    return wrapTensor(native.fast.sum(this._native, dimOrOpts ?? null, keepdim, null));
  }

  mean(dimOrOpts?: number | number[] | ReductionOptions, keepdim = false): Tensor {
    if (typeof dimOrOpts === 'object' && !Array.isArray(dimOrOpts)) {
      return wrapTensor(
        native.fast.mean(
          this._native,
          dimOrOpts.dim ?? null,
          dimOrOpts.keepdim ?? false,
          dimOrOpts.dtype ?? null
        )
      );
    }
    return wrapTensor(native.fast.mean(this._native, dimOrOpts ?? null, keepdim, null));
  }

  max(): Tensor;
  max(dim: number, keepdim?: boolean): { values: Tensor; indices: Tensor };
  max(dim?: number, keepdim = false): Tensor | { values: Tensor; indices: Tensor } {
    if (dim === undefined) {
      return wrapTensor(native.fast.max(this._native));
    }
    const res = native.fast.max(this._native, dim, keepdim);
    return { values: wrapTensor(res.values), indices: wrapTensor(res.indices) };
  }

  min(): Tensor;
  min(dim: number, keepdim?: boolean): { values: Tensor; indices: Tensor };
  min(dim?: number, keepdim = false): Tensor | { values: Tensor; indices: Tensor } {
    if (dim === undefined) {
      return wrapTensor(native.fast.min(this._native));
    }
    const res = native.fast.min(this._native, dim, keepdim);
    return { values: wrapTensor(res.values), indices: wrapTensor(res.indices) };
  }

  argmax(dim?: number, keepdim = false): Tensor {
    return wrapTensor(native.fast.argmax(this._native, dim ?? null, keepdim));
  }

  argmin(dim?: number, keepdim = false): Tensor {
    return wrapTensor(native.fast.argmin(this._native, dim ?? null, keepdim));
  }

  softmax(dim: number): Tensor {
    return wrapTensor(native.fast.softmax(this._native, dim));
  }

  logSoftmax(dim: number): Tensor {
    return wrapTensor(native.fast.logSoftmax(this._native, dim));
  }

  norm(p: Scalar = 2, dim?: number | number[], keepdim = false): Tensor {
    return wrapTensor(native.fast.norm(this._native, p, dim ?? null, keepdim));
  }

  all(): Tensor {
    return wrapTensor(native.callOpAuto('all', [this._native]) as NativeTensor);
  }

  any(): Tensor {
    return wrapTensor(native.callOpAuto('any', [this._native]) as NativeTensor);
  }

  var(dim?: number | number[], correction = 1, keepdim = false): Tensor {
    if (dim === undefined) {
      return wrapTensor(
        native.callOp('aten::var', 'correction', [this._native], {
          correction,
          keepdim,
        }) as NativeTensor
      );
    }
    const dims = Array.isArray(dim) ? dim : [dim];
    return wrapTensor(
      native.callOp('aten::var', 'correction', [this._native, dims], {
        correction,
        keepdim,
      }) as NativeTensor
    );
  }

  std(dim?: number | number[], correction = 1, keepdim = false): Tensor {
    if (dim === undefined) {
      return wrapTensor(
        native.callOp('aten::std', 'correction', [this._native], {
          correction,
          keepdim,
        }) as NativeTensor
      );
    }
    const dims = Array.isArray(dim) ? dim : [dim];
    return wrapTensor(
      native.callOp('aten::std', 'correction', [this._native, dims], {
        correction,
        keepdim,
      }) as NativeTensor
    );
  }

  topk(
    k: number,
    dim = -1,
    largest = true,
    sorted = true
  ): { values: Tensor; indices: Tensor } {
    const [values, indices] = native.callOp('aten::topk', '', [
      this._native,
      k,
      dim,
      largest,
      sorted,
    ]) as [NativeTensor, NativeTensor];
    return { values: wrapTensor(values), indices: wrapTensor(indices) };
  }

  sort(dim = -1, descending = false): { values: Tensor; indices: Tensor } {
    const [values, indices] = native.callOp('aten::sort', '', [
      this._native,
      dim,
      descending,
    ]) as [NativeTensor, NativeTensor];
    return { values: wrapTensor(values), indices: wrapTensor(indices) };
  }

  repeat(...repeats: number[] | [number[]]): Tensor {
    const reps = Array.isArray(repeats[0]) ? repeats[0] : (repeats as number[]);
    return wrapTensor(native.callOp('aten::repeat', '', [this._native, reps]) as NativeTensor);
  }

  gather(dim: number, index: Tensor): Tensor {
    return wrapTensor(
      native.callOp('aten::gather', '', [this._native, dim, index._native]) as NativeTensor
    );
  }

  indexSelect(dim: number, index: Tensor): Tensor {
    return wrapTensor(
      native.callOp('aten::index_select', '', [this._native, dim, index._native]) as NativeTensor
    );
  }

  // ---------- Views & Shape ----------

  reshape(...shape: number[] | [number[]]): Tensor {
    const dims = Array.isArray(shape[0]) ? shape[0] : (shape as number[]);
    return wrapTensor(native.fast.reshape(this._native, dims));
  }

  view(...shape: number[] | [number[]]): Tensor {
    const dims = Array.isArray(shape[0]) ? shape[0] : (shape as number[]);
    return wrapTensor(native.fast.view(this._native, dims));
  }

  transpose(dim0: number, dim1: number): Tensor {
    return wrapTensor(native.fast.transpose(this._native, dim0, dim1));
  }

  permute(...dims: number[] | [number[]]): Tensor {
    const order = Array.isArray(dims[0]) ? dims[0] : (dims as number[]);
    return wrapTensor(native.fast.permute(this._native, order));
  }

  squeeze(dim?: number): Tensor {
    return wrapTensor(native.fast.squeeze(this._native, dim ?? null));
  }

  unsqueeze(dim: number): Tensor {
    return wrapTensor(native.fast.unsqueeze(this._native, dim));
  }

  flatten(startDim = 0, endDim = -1): Tensor {
    return wrapTensor(native.fast.flatten(this._native, startDim, endDim));
  }

  expand(...size: number[] | [number[]]): Tensor {
    const dims = Array.isArray(size[0]) ? size[0] : (size as number[]);
    return wrapTensor(native.fast.expand(this._native, dims));
  }

  split(splitSizeOrSections: number | number[], dim = 0): Tensor[] {
    return native.fast.split(this._native, splitSizeOrSections, dim).map(wrapTensor);
  }

  chunk(chunks: number, dim = 0): Tensor[] {
    return native.fast.chunk(this._native, chunks, dim).map(wrapTensor);
  }

  narrow(dim: number, start: number, length: number): Tensor {
    return wrapTensor(native.fast.narrow(this._native, dim, start, length));
  }

  maskedFill(mask: Tensor, value: Scalar): Tensor {
    return wrapTensor(native.fast.maskedFill(this._native, mask._native, value));
  }

  tril(diagonal = 0): Tensor {
    return wrapTensor(native.fast.tril(this._native, diagonal));
  }

  triu(diagonal = 0): Tensor {
    return wrapTensor(native.fast.triu(this._native, diagonal));
  }

  // ---------- Formatting ----------

  toString(): string {
    if (this.isDisposed) return 'Tensor(<disposed>)';
    return this._native.formatString();
  }

  [Symbol.for('nodejs.util.inspect.custom')](): string {
    if (this.isDisposed) return 'Tensor(<disposed>)';
    const meta = `shape=[${this.shape.join(', ')}], dtype=${this.dtype}, device=${this.device.toString()}${
      this.requiresGrad ? ', requiresGrad=true' : ''
    }`;
    if (this.numel <= 64) {
      return `${this._native.formatString()}\n(${meta})`;
    }
    return `Tensor(${meta})`;
  }
}

export function wrapTensor(handle: NativeTensor): Tensor {
  return new Tensor(handle);
}

export function wrapResult(raw: unknown): unknown {
  if (raw === null || raw === undefined) return raw;
  if (raw instanceof native.NativeTensor) {
    return wrapTensor(raw);
  }
  if (Array.isArray(raw)) {
    return raw.map(wrapResult);
  }
  return raw;
}
