import { forwardAD, grad as autogradGrad, noGrad } from './autograd.js';
import type { Module } from './nn/module.js';
import { Parameter } from './nn/parameter.js';
import { cat, eye, stack, zeros } from './ops.js';
import { Tensor } from './tensor.js';

function cloneWithGrad(t: Tensor): Tensor {
  return t.detach().clone().requiresGrad_(true);
}

/**
 * Functional reverse-mode differentiation returning both the scalar output and gradient(s).
 */
export function valueAndGrad<TArgs extends unknown[]>(
  fn: (...args: TArgs) => Tensor,
  argnums: number | number[] = 0
): (...args: TArgs) => { value: Tensor; grad: Tensor | Tensor[] } {
  const indices = Array.isArray(argnums) ? argnums : [argnums];
  return (...args: TArgs) => {
    const wrappedArgs = [...args] as TArgs;
    const diffInputs: Tensor[] = [];
    for (const idx of indices) {
      const arg = args[idx];
      if (!(arg instanceof Tensor)) {
        throw new TypeError(`torch.func.grad argument at index ${idx} must be a Tensor`);
      }
      const req = cloneWithGrad(arg);
      wrappedArgs[idx] = req;
      diffInputs.push(req);
    }

    const out = fn(...wrappedArgs);
    if (out.numel !== 1) {
      throw new RangeError(
        `torch.func.grad expects a scalar output tensor (numel=1), got shape [${out.shape.join(', ')}]`
      );
    }

    const grads = autogradGrad(out, diffInputs, undefined, {
      createGraph: out.requiresGrad,
      allowUnused: true,
    });

    const resolved = grads.map((g, i) => g ?? zeros(diffInputs[i]!.shape, { dtype: diffInputs[i]!.dtype, device: diffInputs[i]!.device }));
    for (const inp of diffInputs) {
      inp.dispose();
    }
    return {
      value: out,
      grad: Array.isArray(argnums) ? resolved : resolved[0]!,
    };
  };
}

/**
 * Functional reverse-mode gradient transform (PyTorch `torch.func.grad`).
 */
export function grad<TArgs extends unknown[]>(
  fn: (...args: TArgs) => Tensor,
  argnums: number | number[] = 0
): (...args: TArgs) => Tensor | Tensor[] {
  const vg = valueAndGrad(fn, argnums);
  return (...args: TArgs) => {
    const { value, grad: g } = vg(...args);
    value.dispose();
    return g;
  };
}

/**
 * Computes the Jacobian of `fn` with respect to input `argnum` using reverse-mode AD.
 * Output shape is `[...out.shape, ...input.shape]`.
 */
export function jacrev<TArgs extends unknown[]>(
  fn: (...args: TArgs) => Tensor,
  argnum = 0
): (...args: TArgs) => Tensor {
  return (...args: TArgs) => {
    const target = args[argnum];
    if (!(target instanceof Tensor)) {
      throw new TypeError(`jacrev argument at index ${argnum} must be a Tensor`);
    }
    const req = cloneWithGrad(target);
    const callArgs = [...args] as TArgs;
    callArgs[argnum] = req;

    const out = fn(...callArgs);
    const outShape = out.shape;
    const inShape = req.shape;
    const flatOut = out.reshape(-1);
    const numOut = flatOut.numel;

    const rows: Tensor[] = [];
    const basis = eye(numOut, numOut, { dtype: flatOut.dtype, device: flatOut.device });
    try {
      for (let i = 0; i < numOut; i++) {
        const v = basis.at(i);
        const [g] = autogradGrad(flatOut, [req], [v], {
          retainGraph: i < numOut - 1,
          createGraph: true,
          allowUnused: true,
        });
        v.dispose();
        rows.push(
          g ?? zeros(inShape, { dtype: req.dtype, device: req.device })
        );
      }
      const stacked = stack(rows, 0);
      const jac = stacked.reshape([...outShape, ...inShape]);
      stacked.dispose();
      return jac;
    } finally {
      for (const r of rows) r.dispose();
      basis.dispose();
      flatOut.dispose();
      out.dispose();
      req.dispose();
    }
  };
}

/**
 * Computes the Jacobian of `fn` with respect to input `argnum` using forward-mode AD (JVP).
 * Output shape is `[...out.shape, ...input.shape]`.
 */
export function jacfwd<TArgs extends unknown[]>(
  fn: (...args: TArgs) => Tensor,
  argnum = 0
): (...args: TArgs) => Tensor {
  return (...args: TArgs) => {
    const target = args[argnum];
    if (!(target instanceof Tensor)) {
      throw new TypeError(`jacfwd argument at index ${argnum} must be a Tensor`);
    }
    const inShape = target.shape;
    const numIn = target.numel;
    const basis = eye(numIn, numIn, { dtype: target.dtype, device: target.device });
    const cols: Tensor[] = [];
    let outShape: number[] = [];

    try {
      for (let j = 0; j < numIn; j++) {
        const tangent = basis.at(j).reshape(inShape);
        const { output, jvp } = forwardAD.jvp(
          (x) => {
            const callArgs = [...args] as TArgs;
            callArgs[argnum] = x;
            return fn(...callArgs);
          },
          target,
          tangent
        );
        outShape = output.shape;
        const col =
          jvp ?? zeros(outShape, { dtype: output.dtype, device: output.device });
        cols.push(col.reshape(-1));
        if (jvp) jvp.dispose();
        output.dispose();
        tangent.dispose();
      }
      const stacked = stack(cols, 1);
      const jac = stacked.reshape([...outShape, ...inShape]);
      stacked.dispose();
      return jac;
    } finally {
      for (const c of cols) c.dispose();
      basis.dispose();
    }
  };
}

/**
 * Computes the Hessian of a scalar-valued function via `jacrev(grad(fn))`.
 */
export function hessian(fn: (x: Tensor) => Tensor): (x: Tensor) => Tensor {
  const gradFn = (x: Tensor): Tensor => {
    const out = fn(x);
    const [g] = autogradGrad(out, [x], undefined, { createGraph: true, allowUnused: true });
    out.dispose();
    return g ?? zeros(x.shape, { dtype: x.dtype, device: x.device });
  };
  return jacrev(gradFn, 0);
}

/**
 * Vectorizing map over a batch dimension (`inDims` -> `outDim`), preserving autograd.
 */
export function vmap<TArgs extends unknown[]>(
  fn: (...args: TArgs) => Tensor,
  inDims: number | (number | null)[] = 0,
  outDim = 0
): (...args: TArgs) => Tensor {
  return (...args: TArgs) => {
    const dims: (number | null)[] = Array.isArray(inDims)
      ? inDims
      : args.map((a) => (a instanceof Tensor ? inDims : null));

    let batchSize: number | null = null;
    for (let i = 0; i < args.length; i++) {
      const d = dims[i];
      if (d !== null && d !== undefined) {
        const arg = args[i];
        if (!(arg instanceof Tensor)) {
          throw new TypeError(`vmap expected Tensor at arg ${i} for inDim=${d}`);
        }
        const sz = arg.size(d);
        if (batchSize === null) {
          batchSize = sz;
        } else if (batchSize !== sz) {
          throw new RangeError(`vmap batch size mismatch: ${batchSize} vs ${sz}`);
        }
      }
    }
    if (batchSize === null) {
      throw new Error('vmap requires at least one mapped Tensor input');
    }

    const outputs: Tensor[] = [];
    const slicesToDispose: Tensor[] = [];
    try {
      for (let b = 0; b < batchSize; b++) {
        const stepArgs = args.map((arg, i) => {
          const d = dims[i];
          if (d === null || d === undefined || !(arg instanceof Tensor)) {
            return arg;
          }
          const slice = arg.narrow(d, b, 1).squeeze(d);
          slicesToDispose.push(slice);
          return slice;
        }) as TArgs;
        outputs.push(fn(...stepArgs));
      }
      return stack(outputs, outDim);
    } finally {
      for (const s of slicesToDispose) s.dispose();
      for (const o of outputs) o.dispose();
    }
  };
}

/**
 * Stateless functional invocation of an `nn.Module` using caller-supplied
 * parameters and buffers (enabling `grad`/`jacrev`/`vmap` directly over weights).
 */
export function functionalCall<R = unknown>(
  module: Module,
  paramsAndBuffers: Record<string, Tensor>,
  ...args: unknown[]
): R {
  const backups: {
    target: Module;
    key: string;
    isParam: boolean;
    hadProp: boolean;
    origProp: unknown;
    origMap: Parameter | Tensor | null | undefined;
    hadMap: boolean;
  }[] = [];

  const resolveLeaf = (fullName: string): { owner: Module; leaf: string } => {
    const parts = fullName.split('.');
    let cur: Module = module;
    for (let i = 0; i < parts.length - 1; i++) {
      const seg = parts[i]!;
      let next: Module | undefined;
      for (const [childName, childMod] of cur.namedChildren()) {
        if (childName === seg) {
          next = childMod;
          break;
        }
      }
      if (!next) {
        throw new Error(`functionalCall: submodule '${seg}' not found in path '${fullName}'`);
      }
      cur = next;
    }
    return { owner: cur, leaf: parts[parts.length - 1]! };
  };

  try {
    for (const [name, replacement] of Object.entries(paramsAndBuffers)) {
      const { owner, leaf } = resolveLeaf(name);
      const rec = owner as unknown as Record<string, unknown>;
      const paramMap = (owner as unknown as { _parameters: Map<string, Parameter | null> })._parameters;
      const bufMap = (owner as unknown as { _buffers: Map<string, Tensor | null> })._buffers;
      const isParam = paramMap.has(leaf) || rec[leaf] instanceof Parameter;
      const targetMap = isParam ? paramMap : bufMap;

      backups.push({
        target: owner,
        key: leaf,
        isParam,
        hadProp: Object.prototype.hasOwnProperty.call(owner, leaf),
        origProp: rec[leaf],
        origMap: targetMap.get(leaf),
        hadMap: targetMap.has(leaf),
      });

      if (targetMap.has(leaf)) {
        targetMap.set(leaf, replacement as Parameter);
      }
      if (Object.prototype.hasOwnProperty.call(owner, leaf)) {
        rec[leaf] = replacement;
      }
    }

    return module.call(...args) as R;
  } finally {
    noGrad(() => {
      for (let i = backups.length - 1; i >= 0; i--) {
        const b = backups[i]!;
        const rec = b.target as unknown as Record<string, unknown>;
        const targetMap = b.isParam
          ? (b.target as unknown as { _parameters: Map<string, Parameter | null> })._parameters
          : (b.target as unknown as { _buffers: Map<string, Tensor | null> })._buffers;
        if (b.hadMap) {
          targetMap.set(b.key, b.origMap as Parameter | null);
        }
        if (b.hadProp) {
          rec[b.key] = b.origProp;
        }
      }
    });
  }
}

export function flattenParams(params: Tensor[]): Tensor {
  return cat(
    params.map((p) => p.reshape(-1)),
    0
  );
}

export const func = {
  grad,
  valueAndGrad,
  jacrev,
  jacfwd,
  hessian,
  vmap,
  functionalCall,
};
