import { AutogradContext, AutogradFunction, isGradEnabled, noGrad } from '../autograd.js';
import type { DType } from '../dtype.js';
import type { DeviceLike } from '../device.js';
import { untrackTensor } from '../memory.js';
import { Tensor, type ToOptions } from '../tensor.js';
import { Parameter } from './parameter.js';

export interface LoadStateDictResult {
  missingKeys: string[];
  unexpectedKeys: string[];
}

export interface HookHandle extends Disposable {
  remove(): void;
}

export type ForwardPreHook = (module: Module, inputs: unknown[]) => unknown[] | void;
export type ForwardHook = (module: Module, inputs: unknown[], output: unknown) => unknown | void;
export type FullBackwardHook = (
  module: Module,
  gradInput: (Tensor | null)[],
  gradOutput: (Tensor | null)[]
) => (Tensor | null)[] | void;

class BackwardHookPass extends AutogradFunction {
  static forward(
    ctx: AutogradContext,
    onBackward: (grads: (Tensor | null)[]) => (Tensor | null)[],
    ...tensors: Tensor[]
  ): Tensor | Tensor[] {
    ctx['onBackward'] = onBackward;
    ctx['count'] = tensors.length;
    const views = tensors.map((t) => t.view(t.shape));
    return views.length === 1 ? views[0]! : views;
  }

  static backward(ctx: AutogradContext, ...gradOutputs: (Tensor | null)[]): (Tensor | null)[] {
    const onBackward = ctx['onBackward'] as (grads: (Tensor | null)[]) => (Tensor | null)[];
    const modified = onBackward(gradOutputs);
    return [null, ...modified];
  }
}

export abstract class Module implements Disposable {
  protected readonly _parameters = new Map<string, Parameter | null>();
  protected readonly _buffers = new Map<string, Tensor | null>();
  protected readonly _modules = new Map<string, Module | null>();
  private readonly _forwardPreHooks = new Map<number, ForwardPreHook>();
  private readonly _forwardHooks = new Map<number, ForwardHook>();
  private readonly _backwardHooks = new Map<number, FullBackwardHook>();
  private _nextHookId = 1;
  training = true;

  abstract forward(...args: unknown[]): unknown;

  registerForwardPreHook(hook: ForwardPreHook): HookHandle {
    const id = this._nextHookId++;
    this._forwardPreHooks.set(id, hook);
    const remove = (): void => {
      this._forwardPreHooks.delete(id);
    };
    return { remove, [Symbol.dispose]: remove };
  }

  registerForwardHook(hook: ForwardHook): HookHandle {
    const id = this._nextHookId++;
    this._forwardHooks.set(id, hook);
    const remove = (): void => {
      this._forwardHooks.delete(id);
    };
    return { remove, [Symbol.dispose]: remove };
  }

  registerFullBackwardHook(hook: FullBackwardHook): HookHandle {
    const id = this._nextHookId++;
    this._backwardHooks.set(id, hook);
    const remove = (): void => {
      this._backwardHooks.delete(id);
    };
    return { remove, [Symbol.dispose]: remove };
  }

  call(...args: unknown[]): unknown {
    let inputs = args;
    for (const hook of this._forwardPreHooks.values()) {
      const modified = hook(this, inputs);
      if (modified !== undefined) {
        inputs = modified;
      }
    }

    const hasBwdHooks = this._backwardHooks.size > 0 && isGradEnabled();
    let capturedGradOutput: (Tensor | null)[] = [];

    if (hasBwdHooks) {
      const inputIndices: number[] = [];
      const inputTensors: Tensor[] = [];
      inputs.forEach((arg, idx) => {
        if (arg instanceof Tensor && arg.requiresGrad) {
          inputIndices.push(idx);
          inputTensors.push(arg);
        }
      });

      if (inputTensors.length > 0) {
        const hookedIn = BackwardHookPass.apply(
          (gradInputs: (Tensor | null)[]) => {
            let curGradIn = gradInputs;
            for (const hook of this._backwardHooks.values()) {
              const modified = hook(this, curGradIn, capturedGradOutput);
              if (modified !== undefined) {
                curGradIn = modified;
              }
            }
            return curGradIn;
          },
          ...inputTensors
        );
        const wrappedList = Array.isArray(hookedIn) ? hookedIn : [hookedIn as Tensor];
        inputs = [...inputs];
        inputIndices.forEach((origIdx, i) => {
          inputs[origIdx] = wrappedList[i]!;
        });
      }
    }

    let output = this.forward(...inputs);
    for (const hook of this._forwardHooks.values()) {
      const modified = hook(this, inputs, output);
      if (modified !== undefined) {
        output = modified;
      }
    }

    if (hasBwdHooks) {
      if (output instanceof Tensor && output.requiresGrad) {
        const hookedOut = BackwardHookPass.apply(
          (gradOutputs: (Tensor | null)[]) => {
            capturedGradOutput = gradOutputs;
            if (!inputs.some((a) => a instanceof Tensor && a.requiresGrad)) {
              for (const hook of this._backwardHooks.values()) {
                hook(this, [], gradOutputs);
              }
            }
            return gradOutputs;
          },
          output
        );
        output = hookedOut;
      }
    }
    return output;
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

  /** Moves all parameters and buffers to the given device/dtype in-place, preserving tied weights. */
  to(target: DeviceLike | DType | ToOptions): this {
    const movedParams = new Map<Parameter, Parameter>();
    const movedBuffers = new Map<Tensor, Tensor>();
    const visitedModules = new Set<Module>();

    const applyToModule = (mod: Module): void => {
      if (visitedModules.has(mod)) return;
      visitedModules.add(mod);
      const { params, buffers, modules } = mod.discoverMembers();
      noGrad(() => {
        for (const [key, p] of params) {
          let moved = movedParams.get(p);
          if (!moved) {
            const movedRaw = p.to(target);
            moved = new Parameter(movedRaw, p.requiresGrad);
            movedRaw.dispose();
            movedParams.set(p, moved);
          }
          if (mod._parameters.has(key)) mod._parameters.set(key, moved);
          if (Object.prototype.hasOwnProperty.call(mod, key)) {
            (mod as unknown as Record<string, unknown>)[key] = moved;
          }
        }
        for (const [key, b] of buffers) {
          let moved = movedBuffers.get(b);
          if (!moved) {
            moved = b.to(target);
            untrackTensor(moved);
            movedBuffers.set(b, moved);
          }
          if (mod._buffers.has(key)) mod._buffers.set(key, moved);
          if (Object.prototype.hasOwnProperty.call(mod, key)) {
            (mod as unknown as Record<string, unknown>)[key] = moved;
          }
        }
      });
      for (const child of modules.values()) {
        applyToModule(child);
      }
    };

    applyToModule(this);
    // Dispose replaced originals once (safe even when parameters/buffers were tied across submodules)
    for (const oldParam of movedParams.keys()) {
      if (!oldParam.isDisposed) oldParam.dispose();
    }
    for (const oldBuf of movedBuffers.keys()) {
      if (!oldBuf.isDisposed) oldBuf.dispose();
    }
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
