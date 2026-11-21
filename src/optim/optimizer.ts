import { untrackTensor } from '../memory.js';
import type { Parameter } from '../nn/parameter.js';
import type { Tensor } from '../tensor.js';

export interface ParamGroupBase {
  params: Parameter[];
  lr: number;
  [key: string]: unknown;
}

export type ParamGroupInput<TGroup extends ParamGroupBase> =
  | Parameter[]
  | (Partial<TGroup> & { params: Parameter[] })[];

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
        if (typeof v !== 'number' && !v.isDisposed) {
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
