import type { DType } from './dtype.js';
import { normalizeDevice, type DeviceLike } from './device.js';
import { native, type NativeAotiModel, type NativeJitModule } from './native.js';
import { Tensor, wrapResult, wrapTensor } from './tensor.js';

export interface LoadAotiOptions {
  modelName?: string;
  deviceIndex?: number;
}

/**
 * Loads and runs an AOTInductor-compiled `.pt2` model package exported from Python
 * via `torch.export` + `torch._inductor.aoti_compile_and_package`.
 */
export class AotiModel implements Disposable {
  readonly _native: NativeAotiModel;

  constructor(packagePath: string, opts?: LoadAotiOptions) {
    this._native = new native.NativeAotiModel(
      packagePath,
      opts?.modelName ?? 'model',
      opts?.deviceIndex ?? -1
    );
  }

  run(inputs: Tensor[]): Tensor[] {
    return this._native.run(inputs.map((t) => t._native)).map(wrapTensor);
  }

  get metadata(): Record<string, string> {
    return this._native.getMetadata();
  }

  get callSpec(): string[] {
    return this._native.getCallSpec();
  }

  get constantFqns(): string[] {
    return this._native.getConstantFqns();
  }

  get isDisposed(): boolean {
    return this._native.isDisposed();
  }

  dispose(): void {
    this._native.dispose();
  }

  [Symbol.dispose](): void {
    this.dispose();
  }
}

export function loadAotiModel(packagePath: string, opts?: LoadAotiOptions): AotiModel {
  return new AotiModel(packagePath, opts);
}

/**
 * Loads and runs a TorchScript `.pt` archive saved by `torch.jit.trace(...).save(path)`
 * or any compatible libtorch serializer.
 */
export class JitModule implements Disposable {
  readonly _native: NativeJitModule;

  constructor(modelPath: string, device?: DeviceLike) {
    this._native = new native.NativeJitModule(modelPath, normalizeDevice(device));
  }

  forward(...inputs: unknown[]): unknown {
    const raw = inputs.map((a) => (a instanceof Tensor ? a._native : a));
    return wrapResult(this._native.forward(raw));
  }

  runMethod(name: string, ...inputs: unknown[]): unknown {
    const raw = inputs.map((a) => (a instanceof Tensor ? a._native : a));
    return wrapResult(this._native.runMethod(name, raw));
  }

  to(target: DeviceLike | DType): this {
    const spec = typeof target === 'string' ? target : normalizeDevice(target)!;
    this._native.to(spec);
    return this;
  }

  eval(): this {
    this._native.eval();
    return this;
  }

  train(mode = true): this {
    this._native.train(mode);
    return this;
  }

  get isDisposed(): boolean {
    return this._native.isDisposed();
  }

  dispose(): void {
    this._native.dispose();
  }

  [Symbol.dispose](): void {
    this.dispose();
  }
}

export function loadJitModule(modelPath: string, device?: DeviceLike): JitModule {
  return new JitModule(modelPath, device);
}
