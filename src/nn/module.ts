import { noGrad } from '../autograd.js';
import type { DType } from '../dtype.js';
import type { DeviceLike } from '../device.js';
import { untrackTensor } from '../memory.js';
import { Tensor, type ToOptions } from '../tensor.js';
import { Parameter } from './parameter.js';

export interface LoadStateDictResult {
  missingKeys: string[];
  unexpectedKeys: string[];
}

export abstract class Module implements Disposable {
  protected readonly _parameters = new Map<string, Parameter | null>();
  protected readonly _buffers = new Map<string, Tensor | null>();
  protected readonly _modules = new Map<string, Module | null>();
  training = true;

  abstract forward(...args: unknown[]): unknown;

  call(...args: unknown[]): unknown {
    return this.forward(...args);
  }

  registerParameter(name: string, param: Parameter | null): void {
    this._parameters.set(name, param);
    if (param) untrackTensor(param);
  }

  registerBuffer(name: string, tensor: Tensor | null): void {
    this._buffers.set(name, tensor);
    if (tensor) untrackTensor(tensor);
  }

  addModule(name: string, module: Module | null): void {
    this._modules.set(name, module);
  }

  /**
   * Auto-discovers Parameters and child Modules assigned as normal TypeScript fields
   * (so users can simply write `this.fc = new nn.Linear(10, 5)` without manual registration!).
   */
  private discoverMembers(): {
    params: Map<string, Parameter>;
    buffers: Map<string, Tensor>;
    modules: Map<string, Module>;
  } {
    const params = new Map<string, Parameter>();
    const buffers = new Map<string, Tensor>();
    const modules = new Map<string, Module>();

    for (const [k, v] of this._parameters) {
      if (v) params.set(k, v);
    }
    for (const [k, v] of this._buffers) {
      if (v) buffers.set(k, v);
    }
    for (const [k, v] of this._modules) {
      if (v) modules.set(k, v);
    }

    for (const [key, value] of Object.entries(this)) {
      if (key === '_parameters' || key === '_buffers' || key === '_modules') continue;
      if (value instanceof Parameter) {
        if (!params.has(key)) params.set(key, value);
      } else if (value instanceof Module) {
        if (!modules.has(key)) modules.set(key, value);
      }
    }
    return { params, buffers, modules };
  }

  *namedParameters(prefix = '', recurse = true, seen = new Set<Parameter>()): Generator<[string, Parameter]> {
    const { params, modules } = this.discoverMembers();
    for (const [name, p] of params) {
      if (seen.has(p)) continue;
      seen.add(p);
      yield [prefix ? `${prefix}.${name}` : name, p];
    }
    if (recurse) {
      for (const [mName, mod] of modules) {
        const subPrefix = prefix ? `${prefix}.${mName}` : mName;
        yield* mod.namedParameters(subPrefix, true, seen);
      }
    }
  }

  parameters(recurse = true): Parameter[] {
    const out: Parameter[] = [];
    for (const [, p] of this.namedParameters('', recurse)) {
      out.push(p);
    }
    return out;
  }

  *namedBuffers(prefix = '', recurse = true, seen = new Set<Tensor>()): Generator<[string, Tensor]> {
    const { buffers, modules } = this.discoverMembers();
    for (const [name, b] of buffers) {
      if (seen.has(b)) continue;
      seen.add(b);
      yield [prefix ? `${prefix}.${name}` : name, b];
    }
    if (recurse) {
      for (const [mName, mod] of modules) {
        const subPrefix = prefix ? `${prefix}.${mName}` : mName;
        yield* mod.namedBuffers(subPrefix, true, seen);
      }
    }
  }

  buffers(recurse = true): Tensor[] {
    const out: Tensor[] = [];
    for (const [, b] of this.namedBuffers('', recurse)) {
      out.push(b);
    }
    return out;
  }

  *namedChildren(): Generator<[string, Module]> {
    const { modules } = this.discoverMembers();
    for (const entry of modules) {
      yield entry;
    }
  }

  *namedModules(prefix = '', seen = new Set<Module>()): Generator<[string, Module]> {
    if (seen.has(this)) return;
    seen.add(this);
    yield [prefix, this];
    const { modules } = this.discoverMembers();
    for (const [name, mod] of modules) {
      const subPrefix = prefix ? `${prefix}.${name}` : name;
      yield* mod.namedModules(subPrefix, seen);
    }
  }

  modules(): Module[] {
    return Array.from(this.namedModules(), ([, m]) => m);
  }

  train(mode = true): this {
    this.training = mode;
    for (const [, child] of this.namedChildren()) {
      child.train(mode);
    }
    return this;
  }

  eval(): this {
    return this.train(false);
  }

  requiresGrad_(req = true): this {
    for (const p of this.parameters()) {
      p.requiresGrad_(req);
    }
    return this;
  }

  zeroGrad(setToNone = true): void {
    for (const p of this.parameters()) {
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

  /** Moves all parameters and buffers to the given device/dtype in-place. */
  to(target: DeviceLike | DType | ToOptions): this {
    const applyToModule = (mod: Module): void => {
      const { params, buffers, modules } = mod.discoverMembers();
      noGrad(() => {
        for (const [key, p] of params) {
          const movedRaw = p.to(target);
          const moved = new Parameter(movedRaw, p.requiresGrad);
          movedRaw.dispose();
          if (mod._parameters.has(key)) mod._parameters.set(key, moved);
          if (Object.prototype.hasOwnProperty.call(mod, key)) {
            (mod as unknown as Record<string, unknown>)[key] = moved;
          }
          p.dispose();
        }
        for (const [key, b] of buffers) {
          const moved = b.to(target);
          untrackTensor(moved);
          if (mod._buffers.has(key)) mod._buffers.set(key, moved);
          if (Object.prototype.hasOwnProperty.call(mod, key)) {
            (mod as unknown as Record<string, unknown>)[key] = moved;
          }
          b.dispose();
        }
      });
      for (const child of modules.values()) {
        applyToModule(child);
      }
    };
    applyToModule(this);
    return this;
  }

  apply(fn: (module: Module) => void): this {
    for (const [, child] of this.namedChildren()) {
      child.apply(fn);
    }
    fn(this);
    return this;
  }

  stateDict(): Record<string, Tensor> {
    const out: Record<string, Tensor> = {};
    for (const [name, p] of this.namedParameters()) {
      out[name] = p.detach();
    }
    for (const [name, b] of this.namedBuffers()) {
      out[name] = b.detach();
    }
    return out;
  }

  loadStateDict(state: Record<string, Tensor>, strict = true): LoadStateDictResult {
    const missingKeys: string[] = [];
    const loaded = new Set<string>();

    noGrad(() => {
      for (const [name, p] of this.namedParameters()) {
        const incoming = state[name];
        if (!incoming) {
          missingKeys.push(name);
          continue;
        }
        if (JSON.stringify(p.shape) !== JSON.stringify(incoming.shape)) {
          throw new Error(
            `Shape mismatch for '${name}': model has [${p.shape.join(', ')}], stateDict has [${incoming.shape.join(', ')}]`
          );
        }
        p.copy_(incoming);
        loaded.add(name);
      }
      for (const [name, b] of this.namedBuffers()) {
        const incoming = state[name];
        if (!incoming) {
          missingKeys.push(name);
          continue;
        }
        if (JSON.stringify(b.shape) !== JSON.stringify(incoming.shape)) {
          throw new Error(
            `Shape mismatch for buffer '${name}': model has [${b.shape.join(', ')}], stateDict has [${incoming.shape.join(', ')}]`
          );
        }
        b.copy_(incoming);
        loaded.add(name);
      }
    });

    const unexpectedKeys = Object.keys(state).filter((k) => !loaded.has(k));
    if (strict && (missingKeys.length > 0 || unexpectedKeys.length > 0)) {
      throw new Error(
        `Error(s) in loading stateDict: missing=[${missingKeys.join(', ')}], unexpected=[${unexpectedKeys.join(', ')}]`
      );
    }
    return { missingKeys, unexpectedKeys };
  }

  numParameters(onlyTrainable = false): number {
    let total = 0;
    for (const p of this.parameters()) {
      if (!onlyTrainable || p.requiresGrad) {
        total += p.numel;
      }
    }
    return total;
  }

  dispose(): void {
    for (const p of this.parameters()) {
      if (!p.isDisposed) p.dispose();
    }
    for (const b of this.buffers()) {
      if (!b.isDisposed) b.dispose();
    }
  }

  [Symbol.dispose](): void {
    this.dispose();
  }
}
