import { noGrad } from '../autograd.js';
import { untrackTensor } from '../memory.js';
import type { Parameter } from '../nn/parameter.js';
import { Tensor } from '../tensor.js';

export interface ParamGroupBase {
  params: Parameter[];
  lr: number;
  initialLr?: number;
  [key: string]: unknown;
}

export type ParamGroupInput<TGroup extends ParamGroupBase> =
  | Parameter[]
  | (Partial<TGroup> & { params: Parameter[] })[];

export interface OptimizerStateDict {
  state: Record<number, Record<string, Tensor | number>>;
  paramGroups: (Omit<ParamGroupBase, 'params'> & { params: number[] })[];
}

export abstract class Optimizer<TGroup extends ParamGroupBase = ParamGroupBase> implements Disposable {
  readonly paramGroups: TGroup[] = [];
  protected readonly state = new Map<Parameter, Record<string, Tensor | number>>();

  constructor(params: ParamGroupInput<TGroup>, defaults: Omit<TGroup, 'params'>) {
    if (params.length === 0) {
      throw new Error('Optimizer got an empty parameter list');
    }
    const first = params[0]!;
    if ('params' in first && Array.isArray(first.params)) {
      for (const rawGroup of params as (Partial<TGroup> & { params: Parameter[] })[]) {
        this.paramGroups.push({ ...defaults, ...rawGroup } as TGroup);
      }
    } else {
      this.paramGroups.push({ ...defaults, params: [...(params as Parameter[])] } as TGroup);
    }
  }

  abstract step(closure?: () => Tensor): Tensor | undefined;

  zeroGrad(setToNone = true): void {
    for (const group of this.paramGroups) {
      for (const p of group.params) {
        if (setToNone) {
          p.grad = null;
        } else {
          const g = p.grad;
          if (g) {
            g.detach_();
            g.zero_();
          }
        }
      }
    }
  }

  stateDict(): OptimizerStateDict {
    const paramToId = new Map<Parameter, number>();
    let nextId = 0;
    const packedGroups: (Omit<ParamGroupBase, 'params'> & { params: number[] })[] = [];

    for (const group of this.paramGroups) {
      const ids: number[] = [];
      for (const p of group.params) {
        let id = paramToId.get(p);
        if (id === undefined) {
          id = nextId++;
          paramToId.set(p, id);
        }
        ids.push(id);
      }
      const { params: _ignore, ...hyperparams } = group;
      packedGroups.push({ ...hyperparams, params: ids });
    }

    const packedState: Record<number, Record<string, Tensor | number>> = {};
    for (const [p, s] of this.state) {
      const id = paramToId.get(p);
      if (id === undefined) continue;
      const cloned: Record<string, Tensor | number> = {};
      for (const [k, v] of Object.entries(s)) {
        if (v instanceof Tensor) {
          cloned[k] = v.detach().clone();
        } else {
          cloned[k] = v;
        }
      }
      packedState[id] = cloned;
    }

    return { state: packedState, paramGroups: packedGroups };
  }

  loadStateDict(stateDict: OptimizerStateDict): void {
    if (stateDict.paramGroups.length !== this.paramGroups.length) {
      throw new Error(
        `Loaded state dict has ${stateDict.paramGroups.length} param groups, expected ${this.paramGroups.length}`
      );
    }

    const idToParam = new Map<number, Parameter>();
    for (let gi = 0; gi < this.paramGroups.length; gi++) {
      const targetGroup = this.paramGroups[gi]!;
      const loadedGroup = stateDict.paramGroups[gi]!;
      if (targetGroup.params.length !== loadedGroup.params.length) {
        throw new Error(
          `Loaded param group ${gi} has ${loadedGroup.params.length} params, expected ${targetGroup.params.length}`
        );
      }
      const { params: ids, ...hyperparams } = loadedGroup;
      Object.assign(targetGroup, hyperparams);
      for (let pi = 0; pi < ids.length; pi++) {
        idToParam.set(ids[pi]!, targetGroup.params[pi]!);
      }
    }

    // Clear existing owned state tensors cleanly
    this.dispose();

    noGrad(() => {
      for (const [rawId, entry] of Object.entries(stateDict.state)) {
        const p = idToParam.get(Number(rawId));
        if (!p) continue;
        const restored: Record<string, Tensor | number> = {};
        for (const [k, v] of Object.entries(entry)) {
          if (v instanceof Tensor) {
            // Keep scalar step tensors on CPU/param device as appropriate
            const targetDevice = k === 'stepTensor' && p.device.type === 'cpu' ? 'cpu' : p.device;
            const t = v.detach().to({ device: targetDevice, copy: true });
            restored[k] = this.keepStateTensor(t);
          } else {
            restored[k] = v;
          }
        }
        this.state.set(p, restored);
      }
    });
  }

  protected getState(p: Parameter): Record<string, Tensor | number> {
    let s = this.state.get(p);
    if (!s) {
      s = {};
      this.state.set(p, s);
    }
    return s;
  }

  protected keepStateTensor(t: Tensor): Tensor {
    untrackTensor(t);
    return t;
  }

  dispose(): void {
    for (const s of this.state.values()) {
      for (const v of Object.values(s)) {
        if (v instanceof Tensor && !v.isDisposed) {
          v.dispose();
        }
      }
    }
    this.state.clear();
  }

  [Symbol.dispose](): void {
    this.dispose();
  }
}
