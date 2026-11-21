import type { DType } from './dtype.js';
import { normalizeDevice, type DeviceLike } from './device.js';
import { native, type NativeTensor } from './native.js';
import { Tensor, wrapTensor, type BackwardOptions } from './tensor.js';

export function isGradEnabled(): boolean {
  return native.isGradEnabled();
}

export function setGradEnabled(enabled: boolean): boolean {
  return native.setGradEnabled(enabled);
}

export function noGrad<T>(fn: () => T): T {
  const prev = native.setGradEnabled(false);
  try {
    return fn();
  } finally {
    native.setGradEnabled(prev);
  }
}

export function noGradScope(): Disposable {
  const prev = native.setGradEnabled(false);
  let closed = false;
  return {
    [Symbol.dispose](): void {
      if (!closed) {
        closed = true;
        native.setGradEnabled(prev);
      }
    },
  };
}

export function enableGrad<T>(fn: () => T): T {
  const prev = native.setGradEnabled(true);
  try {
    return fn();
  } finally {
    native.setGradEnabled(prev);
  }
}

export function enableGradScope(): Disposable {
  const prev = native.setGradEnabled(true);
  let closed = false;
  return {
    [Symbol.dispose](): void {
      if (!closed) {
        closed = true;
        native.setGradEnabled(prev);
      }
    },
  };
}

export function isInferenceModeEnabled(): boolean {
  return native.isInferenceModeEnabled();
}

export function inferenceMode<T>(fn: () => T, mode = true): T {
  native.pushInferenceMode(mode);
  try {
    return fn();
  } finally {
    native.popInferenceMode();
  }
}

export function inferenceModeScope(mode = true): Disposable {
  native.pushInferenceMode(mode);
  let closed = false;
  return {
    [Symbol.dispose](): void {
      if (!closed) {
        closed = true;
        native.popInferenceMode();
      }
    },
  };
}

export interface AutocastOptions {
  deviceType: DeviceLike;
  dtype?: DType;
  enabled?: boolean;
}

export function isAutocastEnabled(deviceType: DeviceLike = 'cuda'): boolean {
  return native.isAutocastEnabled(normalizeDevice(deviceType) ?? 'cuda');
}

export function autocast<T>(
  deviceTypeOrOpts: DeviceLike | AutocastOptions,
  fnOrDtype: (() => T) | DType,
  maybeFn?: () => T
): T {
  let dev: string;
  let dtype: DType;
  let enabled = true;
  let fn: () => T;

  if (typeof fnOrDtype === 'function') {
    fn = fnOrDtype;
    if (typeof deviceTypeOrOpts === 'object' && 'deviceType' in deviceTypeOrOpts) {
      dev = normalizeDevice(deviceTypeOrOpts.deviceType) ?? 'cuda';
      dtype = deviceTypeOrOpts.dtype ?? (dev.startsWith('cpu') ? 'bfloat16' : 'float16');
      enabled = deviceTypeOrOpts.enabled ?? true;
    } else {
      dev = normalizeDevice(deviceTypeOrOpts) ?? 'cuda';
      dtype = dev.startsWith('cpu') ? 'bfloat16' : 'float16';
    }
  } else {
    if (!maybeFn) throw new TypeError('autocast(deviceType, dtype, fn) requires a callback');
    dev = normalizeDevice(deviceTypeOrOpts as DeviceLike) ?? 'cuda';
    dtype = fnOrDtype;
    fn = maybeFn;
  }

  const prevEnabled = native.isAutocastEnabled(dev);
  const prevDtype = native.getAutocastDtype(dev);
  native.setAutocastEnabled(dev, enabled);
  native.setAutocastDtype(dev, dtype);
  native.autocastIncrementNesting();
  try {
    return fn();
  } finally {
    if (native.autocastDecrementNesting() === 0) {
      native.autocastClearCache();
    }
    native.setAutocastEnabled(dev, prevEnabled);
    native.setAutocastDtype(dev, prevDtype);
  }
}

export function backward(
  tensors: Tensor | Tensor[],
  gradTensors?: (Tensor | null)[],
  opts?: BackwardOptions
): void {
  const list = Array.isArray(tensors) ? tensors : [tensors];
  native.autogradBackward(
    list.map((t) => t._native),
    gradTensors?.map((g) => (g ? g._native : null)),
    opts
      ? {
          retainGraph: opts.retainGraph,
          createGraph: opts.createGraph,
          inputs: opts.inputs?.map((t) => t._native),
        }
      : undefined
  );
}

export interface GradOptions {
  retainGraph?: boolean;
  createGraph?: boolean;
  allowUnused?: boolean;
}

export function grad(
  outputs: Tensor | Tensor[],
  inputs: Tensor | Tensor[],
  gradOutputs?: (Tensor | null)[],
  opts?: GradOptions
): (Tensor | null)[] {
  const outList = Array.isArray(outputs) ? outputs : [outputs];
  const inList = Array.isArray(inputs) ? inputs : [inputs];
  const raw = native.autogradGrad(
    outList.map((t) => t._native),
    inList.map((t) => t._native),
    gradOutputs?.map((g) => (g ? g._native : null)),
    opts
  );
  return raw.map((g) => (g ? wrapTensor(g) : null));
}

export class AutogradContext {
  [key: string]: unknown;
  _toSave: NativeTensor[] = [];
  _nonDifferentiable: NativeTensor[] = [];
  _dirtyInputs: NativeTensor[] = [];
  _materializeGrads = true;
  _savedNativeTensors: (NativeTensor | null)[] = [];
  needsInputGrad: boolean[] = [];

  saveForBackward(...tensors: Tensor[]): void {
    this._toSave = tensors.map((t) => t._native);
  }

  markNonDifferentiable(...tensors: Tensor[]): void {
    this._nonDifferentiable = tensors.map((t) => t._native);
  }

  markDirty(...tensors: Tensor[]): void {
    this._dirtyInputs = tensors.map((t) => t._native);
  }

  setMaterializeGrads(value: boolean): void {
    this._materializeGrads = value;
  }

  get savedTensors(): Tensor[] {
    return this._savedNativeTensors.map((t) => {
      if (!t) throw new Error('Saved tensor is null/undefined');
      return wrapTensor(t);
    });
  }
}

export interface AutogradFunctionClass {
  name: string;
  forward(ctx: AutogradContext, ...args: unknown[]): Tensor | Tensor[];
  backward(ctx: AutogradContext, ...gradOutputs: Tensor[]): Tensor | null | (Tensor | null | undefined)[];
}

export abstract class AutogradFunction {
  static apply<T extends AutogradFunctionClass>(this: T, ...args: unknown[]): Tensor | Tensor[] {
    const tensorPositions: number[] = [];
    const tensorInputs: NativeTensor[] = [];
    for (let i = 0; i < args.length; i++) {
      const a = args[i];
      if (a instanceof Tensor) {
        tensorPositions.push(i);
        tensorInputs.push(a._native);
      }
    }

    const ctx = new AutogradContext();
    let multiOutput = false;

    const rawOutputs = native.applyCustomAutograd(
      this.name || 'CustomAutogradFunction',
      tensorInputs,
      ctx as unknown as Record<string, unknown>,
      () => {
        const res = this.forward(ctx, ...args);
        if (Array.isArray(res)) {
          multiOutput = true;
          return res.map((t) => t._native);
        }
        multiOutput = false;
        return [res._native];
      },
      (_ctxObj, gradOuts) => {
        const wrappedGrads = gradOuts.map((g) => (g ? wrapTensor(g) : (null as unknown as Tensor)));
        const backRes = this.backward(ctx, ...wrappedGrads);
        const allGrads = Array.isArray(backRes) ? backRes : [backRes];
        // Map either full-arg gradient list or tensor-only gradient list onto tensorInputs
        if (allGrads.length === args.length && args.length !== tensorPositions.length) {
          return tensorPositions.map((pos) => {
            const g = allGrads[pos];
            return g instanceof Tensor ? g._native : null;
          });
        }
        return allGrads.map((g) => (g instanceof Tensor ? g._native : null));
      }
    );

    const wrapped = rawOutputs.map((t) => (t ? wrapTensor(t) : (null as unknown as Tensor)));
    return multiOutput ? wrapped : wrapped[0]!;
  }
}

let activeDualLevel = -1;

export const forwardAD = {
  dualLevel<T>(fn: (level: number) => T): T {
    const level = native.enterDualLevel();
    const prev = activeDualLevel;
    activeDualLevel = level;
    try {
      return fn(level);
    } finally {
      activeDualLevel = prev;
      native.exitDualLevel(level);
    }
  },

  makeDual(primal: Tensor, tangent: Tensor, level = activeDualLevel): Tensor {
    if (level < 0) {
      throw new Error('makeDual() must be called inside forwardAD.dualLevel() or with an explicit level');
    }
    return wrapTensor(native.makeDual(primal._native, tangent._native, level));
  },

  unpackDual(dual: Tensor, level = activeDualLevel): { primal: Tensor; tangent: Tensor | null } {
    if (level < 0) {
      throw new Error('unpackDual() must be called inside forwardAD.dualLevel() or with an explicit level');
    }
    const { primal, tangent } = native.unpackDual(dual._native, level);
    return { primal: wrapTensor(primal), tangent: tangent ? wrapTensor(tangent) : null };
  },

  /** Compute exact Jacobian-vector product (f(x), J_f(x) @ v) using forward-mode AD. */
  jvp(
    fn: (...inputs: Tensor[]) => Tensor,
    primals: Tensor | Tensor[],
    tangents: Tensor | Tensor[]
  ): { output: Tensor; jvp: Tensor | null } {
    const pList = Array.isArray(primals) ? primals : [primals];
    const tList = Array.isArray(tangents) ? tangents : [tangents];
    if (pList.length !== tList.length) {
      throw new RangeError('primals and tangents must have the same length');
    }
    return forwardAD.dualLevel((level) => {
      const duals = pList.map((p, i) => forwardAD.makeDual(p, tList[i]!, level));
      const outDual = fn(...duals);
      const { primal, tangent } = forwardAD.unpackDual(outDual, level);
      for (const d of duals) d.dispose();
      outDual.dispose();
      return { output: primal, jvp: tangent };
    });
  },
};
