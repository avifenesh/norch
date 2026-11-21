import { noGrad } from '../autograd.js';
import { randn } from '../ops.js';
import type { Tensor } from '../tensor.js';
import type { HookHandle, Module } from './module.js';
import { Parameter } from './parameter.js';

function getParam(mod: Module, name: string): Parameter {
  const rec = mod as unknown as Record<string, unknown>;
  const map = (mod as unknown as { _parameters: Map<string, Parameter | null> })._parameters;
  const p = map.get(name) ?? (rec[name] instanceof Parameter ? (rec[name] as Parameter) : undefined);
  if (!p) {
    throw new Error(`Module has no Parameter named '${name}'`);
  }
  return p;
}

function setParamOrTensorField(mod: Module, name: string, value: Tensor | null): void {
  const rec = mod as unknown as Record<string, unknown>;
  const map = (mod as unknown as { _parameters: Map<string, Parameter | null> })._parameters;
  if (map.has(name)) {
    map.set(name, value instanceof Parameter ? value : null);
  }
  if (Object.prototype.hasOwnProperty.call(mod, name) || value !== null) {
    rec[name] = value;
  }
}

function normExceptDim(v: Tensor, dim: number): Tensor {
  if (dim === -1) {
    return v.norm(2);
  }
  const ndim = v.ndim;
  const targetDim = dim < 0 ? ndim + dim : dim;
  const reduceDims: number[] = [];
  for (let d = 0; d < ndim; d++) {
    if (d !== targetDim) reduceDims.push(d);
  }
  if (reduceDims.length === 0) {
    return v.abs();
  }
  return v.norm(2, reduceDims, true);
}

/**
 * Applies Weight Normalization (Salimans & Kingma, 2016) to `module[name]`:
 * replaces the learnable parameter with `weightG` (magnitude) and `weightV` (direction)
 * and recomputes `weight = weightG * (weightV / ||weightV||)` before every `module.call()`.
 */
export function weightNorm<T extends Module>(
  module: T,
  name = 'weight',
  dim = 0
): { module: T; remove(): void; [Symbol.dispose](): void } {
  const orig = getParam(module, name);
  const gInit = noGrad(() => normExceptDim(orig, dim));
  const weightG = new Parameter(gInit, true);
  gInit.dispose();
  const weightV = new Parameter(orig, true);

  const gName = `${name}_g`;
  const vName = `${name}_v`;
  module.registerParameter(gName, weightG);
  module.registerParameter(vName, weightV);

  const paramMap = (module as unknown as { _parameters: Map<string, Parameter | null> })._parameters;
  paramMap.delete(name);
  orig.dispose();

  let computedWeight: Tensor | null = null;
  const recompute = (): void => {
    const normV = normExceptDim(weightV, dim).clamp(1e-12, null);
    const nextW = weightV.mul(weightG.div(normV));
    normV.dispose();
    if (computedWeight && !computedWeight.isDisposed) {
      computedWeight.dispose();
    }
    computedWeight = nextW;
    setParamOrTensorField(module, name, nextW);
  };

  recompute();
  const hook: HookHandle = module.registerForwardPreHook(() => {
    recompute();
  });

  let removed = false;
  const remove = (): void => {
    if (removed) return;
    removed = true;
    hook.remove();
    const finalW = noGrad(() => {
      const normV = normExceptDim(weightV, dim).clamp(1e-12, null);
      const w = weightV.mul(weightG.div(normV));
      normV.dispose();
      return w;
    });
    const restored = new Parameter(finalW, true);
    finalW.dispose();
    if (computedWeight && !computedWeight.isDisposed) {
      computedWeight.dispose();
      computedWeight = null;
    }
    paramMap.delete(gName);
    paramMap.delete(vName);
    weightG.dispose();
    weightV.dispose();
    module.registerParameter(name, restored);
    (module as unknown as Record<string, unknown>)[name] = restored;
  };

  return { module, remove, [Symbol.dispose]: remove };
}

export interface SpectralNormOptions {
  nPowerIterations?: number;
  eps?: number;
}

/**
 * Applies Spectral Normalization (Miyato et al., 2018) to `module[name]` via power iteration,
 * bounding the top singular value of the weight matrix to <= 1.
 */
export function spectralNorm<T extends Module>(
  module: T,
  name = 'weight',
  opts?: SpectralNormOptions
): { module: T; remove(): void; [Symbol.dispose](): void } {
  const nPowerIterations = Math.max(1, opts?.nPowerIterations ?? 1);
  const eps = opts?.eps ?? 1e-12;
  const orig = getParam(module, name);
  const weightOrig = new Parameter(orig, true);
  const origName = `${name}_orig`;
  module.registerParameter(origName, weightOrig);

  const paramMap = (module as unknown as { _parameters: Map<string, Parameter | null> })._parameters;
  paramMap.delete(name);
  orig.dispose();

  const h = weightOrig.size(0);
  const wCols = Math.floor(weightOrig.numel / h);
  const u = noGrad(() => {
    const r = randn([h], { dtype: weightOrig.dtype, device: weightOrig.device });
    const n = r.norm(2).clamp(eps, null);
    const out = r.div(n);
    r.dispose();
    n.dispose();
    return out;
  });
  const v = noGrad(() => {
    const r = randn([wCols], { dtype: weightOrig.dtype, device: weightOrig.device });
    const n = r.norm(2).clamp(eps, null);
    const out = r.div(n);
    r.dispose();
    n.dispose();
    return out;
  });
  module.registerBuffer(`${name}_u`, u);
  module.registerBuffer(`${name}_v`, v);

  let computedWeight: Tensor | null = null;
  const recompute = (doPowerIter: boolean): void => {
    const wMat = weightOrig.reshape(h, wCols);
    if (doPowerIter) {
      noGrad(() => {
        for (let i = 0; i < nPowerIterations; i++) {
          const wtU = wMat.T.matmul(u);
          const vNorm = wtU.norm(2).clamp(eps, null);
          const nextV = wtU.div(vNorm);
          v.copy_(nextV);
          wtU.dispose();
          vNorm.dispose();
          nextV.dispose();

          const wV = wMat.matmul(v);
          const uNorm = wV.norm(2).clamp(eps, null);
          const nextU = wV.div(uNorm);
          u.copy_(nextU);
          wV.dispose();
          uNorm.dispose();
          nextU.dispose();
        }
      });
    }
    const wV = wMat.matmul(v);
    const sigma = u.mul(wV).sum().clamp(eps, null);
    wV.dispose();
    wMat.dispose();
    const nextW = weightOrig.div(sigma);
    sigma.dispose();
    if (computedWeight && !computedWeight.isDisposed) {
      computedWeight.dispose();
    }
    computedWeight = nextW;
    setParamOrTensorField(module, name, nextW);
  };

  recompute(true);
  const hook: HookHandle = module.registerForwardPreHook(() => {
    recompute(module.training);
  });

  let removed = false;
  const remove = (): void => {
    if (removed) return;
    removed = true;
    hook.remove();
    const restored = noGrad(() => {
      recompute(false);
      return new Parameter(computedWeight!, true);
    });
    if (computedWeight && !computedWeight.isDisposed) {
      computedWeight.dispose();
      computedWeight = null;
    }
    const bufMap = (module as unknown as { _buffers: Map<string, Tensor | null> })._buffers;
    paramMap.delete(origName);
    bufMap.delete(`${name}_u`);
    bufMap.delete(`${name}_v`);
    weightOrig.dispose();
    u.dispose();
    v.dispose();
    module.registerParameter(name, restored);
    (module as unknown as Record<string, unknown>)[name] = restored;
  };

  return { module, remove, [Symbol.dispose]: remove };
}

export const parametrizations = {
  weightNorm,
  spectralNorm,
};
