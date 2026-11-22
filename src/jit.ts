import { getAutocastDtype, isAutocastEnabled, isGradEnabled, noGrad } from './autograd.js';
import { cuda, CUDAGraph, CudaStream } from './cuda.js';
import type { DeviceLike } from './device.js';
import { JitModule, loadJitModule } from './export.js';
import { functionalCall } from './func.js';
import { keep, tidy } from './memory.js';
import { native, type NativeTensor, type NativeTracedGraph } from './native.js';
import { Module } from './nn/module.js';
import { Tensor, wrapResult, wrapTensor } from './tensor.js';

export interface JitTraceOptions {
  /** Reject non-tensor Python/JS control-flow side effects during tracing (default: false). */
  strict?: boolean;
  /** Run libtorch IR passes (DCE, peephole, FuseAddMM) and enable profiling executor (default: true). */
  optimize?: boolean;
  /** Enable profiling GraphExecutor (disable for mixed-dtype autocast training to use native ATen autograd). */
  profilingExecutor?: boolean;
  /** Fold constant subexpressions (default: true for jit.trace). */
  freezeConstants?: boolean;
  /** Module class name embedded when saving to `.pt` (default: 'NorchTracedModule'). */
  name?: string;
}

export interface CompileOptions {
  /**
   * - `'default'`: trace into libtorch's C++ `GraphExecutor` (IR passes, kernel fusion, C++ autodiff).
   * - `'reduce-overhead'`: additionally capture a CUDA Graph during no-grad CUDA execution for static shapes.
   */
  mode?: 'default' | 'reduce-overhead';
  /** Enable libtorch IR optimizations and profiling executor (default: true). */
  optimize?: boolean;
  /** Freeze module weights as graph constants for inference-only deployment (default: false). */
  freeze?: boolean;
  /** Optional debug name for the compiled graph. */
  name?: string;
}

function runTraceNative(
  invokeFn: (tracedInputs: Tensor[]) => Tensor | Tensor[],
  exampleInputs: Tensor[],
  opts: JitTraceOptions
): NativeTracedGraph {
  const createdInputs: Tensor[] = [];
  const createdOutputs: Tensor[] = [];
  try {
    return native.traceGraph(
      (rawInputs: NativeTensor[]) => {
        const wrappedInputs = rawInputs.map((nt) => {
          const t = wrapTensor(nt);
          createdInputs.push(t);
          return t;
        });
        const res = tidy(() => {
          const out = invokeFn(wrappedInputs);
          if (out instanceof Tensor) {
            return keep(out);
          }
          if (Array.isArray(out)) {
            return out.map((item) => {
              if (!(item instanceof Tensor)) {
                throw new TypeError('Traced function must return a Tensor or Tensor[]');
              }
              return keep(item);
            });
          }
          throw new TypeError('Traced function must return a Tensor or Tensor[]');
        });
        if (Array.isArray(res)) {
          createdOutputs.push(...res);
          return res.map((t) => t._native);
        }
        createdOutputs.push(res);
        return res._native;
      },
      exampleInputs.map((t) => t._native),
      {
        strict: opts.strict ?? false,
        optimize: opts.optimize ?? true,
        profilingExecutor: opts.profilingExecutor,
        freezeConstants: opts.freezeConstants ?? false,
        name: opts.name,
      }
    );
  } finally {
    const disposed = new Set<Tensor>();
    for (const t of createdOutputs) {
      if (!disposed.has(t) && !t.isDisposed) {
        t.dispose();
        disposed.add(t);
      }
    }
    for (const t of createdInputs) {
      if (!disposed.has(t) && !t.isDisposed) {
        t.dispose();
        disposed.add(t);
      }
    }
  }
}

/**
 * Pure-Node traced TorchScript graph backed directly by `libtorch`'s C++ `GraphExecutor`.
 * Can execute directly in C++ (with IR optimization, kernel fusion, and autograd) or be
 * serialized to a standalone `.pt` archive via `.save(filePath)` with zero Python.
 */
export class TracedModule implements Disposable {
  readonly _native: NativeTracedGraph;

  constructor(nativeGraph: NativeTracedGraph) {
    this._native = nativeGraph;
  }

  forward(...inputs: Tensor[]): Tensor | Tensor[] {
    const out = this._native.run(inputs.map((t) => t._native));
    return wrapResult(out) as Tensor | Tensor[];
  }

  call(...inputs: Tensor[]): Tensor | Tensor[] {
    return this.forward(...inputs);
  }

  graphString(): string {
    return this._native.graphString();
  }

  optimizedGraphString(exampleInputs?: Tensor[]): string {
    return this._native.optimizedGraphString(exampleInputs?.map((t) => t._native));
  }

  get numInputs(): number {
    return this._native.numInputs();
  }

  get numOutputs(): number {
    return this._native.numOutputs();
  }

  /**
   * Serializes this traced graph (including embedded constant weights) into a standard
   * TorchScript `.pt` archive loadable via `torch.jit.load(path)` — zero Python required.
   */
  save(filePath: string): void {
    this._native.save(filePath);
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

/**
 * Traces a JS function or `nn.Module` natively in C++ `libtorch` (`torch::jit::tracer::trace`)
 * without Python. Module weights accessed during tracing are embedded as constants so the
 * returned `TracedModule` is immediately deployable and savable to `.pt` via `.save(path)`.
 */
export function jitTrace(
  fnOrModule: Module | ((...inputs: Tensor[]) => Tensor | Tensor[]),
  exampleInputs: Tensor | Tensor[],
  opts: JitTraceOptions = {}
): TracedModule {
  const examples = Array.isArray(exampleInputs) ? exampleInputs : [exampleInputs];
  if (examples.length === 0) {
    throw new TypeError('torch.jit.trace requires at least one example Tensor input');
  }

  const detachedToDispose: Tensor[] = [];
  try {
    let invoke: (xs: Tensor[]) => Tensor | Tensor[];
    if (fnOrModule instanceof Module) {
      const detachedMap: Record<string, Tensor> = {};
      for (const [name, p] of fnOrModule.namedParameters()) {
        const d = p.detach();
        detachedToDispose.push(d);
        detachedMap[name] = d;
      }
      for (const [name, b] of fnOrModule.namedBuffers()) {
        const d = b.detach();
        detachedToDispose.push(d);
        detachedMap[name] = d;
      }
      invoke = (xs: Tensor[]) => functionalCall<Tensor | Tensor[]>(fnOrModule, detachedMap, ...xs);
    } else {
      invoke = (xs: Tensor[]) => fnOrModule(...xs);
    }

    const name =
      opts.name ?? (fnOrModule instanceof Module ? fnOrModule.constructor.name || 'NorchModule' : 'NorchTracedFn');
    const nativeGraph = runTraceNative(invoke, examples, {
      strict: opts.strict ?? false,
      optimize: opts.optimize ?? true,
      freezeConstants: opts.freezeConstants ?? true,
      name,
    });
    return new TracedModule(nativeGraph);
  } finally {
    for (const d of detachedToDispose) {
      if (!d.isDisposed) d.dispose();
    }
  }
}

/**
 * Loads a TorchScript `.pt` archive saved by `TracedModule.save()` (or any libtorch serializer).
 */
export function jitLoad(modelPath: string, device?: DeviceLike): JitModule {
  return loadJitModule(modelPath, device);
}

export const jit = {
  trace: jitTrace,
  load: jitLoad,
  TracedModule,
  JitModule,
};

// ---------------- torch.compile() ----------------

interface StateBinding {
  entries: { name: string; idx: number }[];
  uniqueTensors: Tensor[];
}

function collectModuleState(mod: Module): StateBinding {
  const entries: { name: string; idx: number }[] = [];
  const uniqueTensors: Tensor[] = [];
  const seen = new Map<Tensor, number>();

  const add = (name: string, t: Tensor): void => {
    let idx = seen.get(t);
    if (idx === undefined) {
      idx = uniqueTensors.length;
      seen.set(t, idx);
      uniqueTensors.push(t);
    }
    entries.push({ name, idx });
  };

  for (const [name, p] of mod.namedParameters()) {
    add(name, p);
  }
  for (const [name, b] of mod.namedBuffers()) {
    add(name, b);
  }
  return { entries, uniqueTensors };
}

function serializeNonTensorArg(val: unknown): string {
  if (val === null) return 'null';
  if (val === undefined) return 'undefined';
  if (typeof val === 'number' || typeof val === 'boolean' || typeof val === 'string') {
    return `${typeof val}:${String(val)}`;
  }
  try {
    return JSON.stringify(val);
  } catch {
    return String(val);
  }
}

interface CudaGraphCache {
  graph: CUDAGraph;
  stream: CudaStream;
  staticInputs: Tensor[];
  staticOutputs: Tensor[];
  singleOutput: boolean;
}

interface CompiledEntry {
  traced: NativeTracedGraph;
  calls: number;
  cudaGraph?: CudaGraphCache;
  dispose(): void;
}

function runTracedToArray(traced: NativeTracedGraph, nativeInputs: NativeTensor[]): {
  tensors: Tensor[];
  singleOutput: boolean;
} {
  const raw = traced.run(nativeInputs);
  if (Array.isArray(raw)) {
    return {
      tensors: (raw as NativeTensor[]).map(wrapTensor),
      singleOutput: false,
    };
  }
  return {
    tensors: [wrapTensor(raw as NativeTensor)],
    singleOutput: true,
  };
}

class CompileEngine {
  private readonly cache = new Map<string, CompiledEntry>();
  private readonly mode: 'default' | 'reduce-overhead';
  private readonly optimize: boolean;
  private readonly freeze: boolean;
  private readonly name: string;

  constructor(
    private readonly target: Module | ((...args: unknown[]) => Tensor | Tensor[]),
    opts: CompileOptions
  ) {
    this.mode = opts.mode ?? 'default';
    this.optimize = opts.optimize ?? true;
    this.freeze = opts.freeze ?? false;
    this.name =
      opts.name ??
      (target instanceof Module ? target.constructor.name || 'CompiledModule' : target.name || 'CompiledFn');
  }

  get graphCount(): number {
    return this.cache.size;
  }

  graphStrings(): string[] {
    const out: string[] = [];
    for (const entry of this.cache.values()) {
      out.push(entry.traced.graphString());
    }
    return out;
  }

  reset(): void {
    for (const entry of this.cache.values()) {
      entry.dispose();
    }
    this.cache.clear();
  }

  invoke(args: unknown[]): Tensor | Tensor[] {
    const tensorArgs: Tensor[] = [];
    const tensorPositions: number[] = [];
    const nonTensorParts: string[] = [];

    for (let i = 0; i < args.length; i++) {
      const a = args[i];
      if (a instanceof Tensor) {
        tensorPositions.push(i);
        tensorArgs.push(a);
      } else {
        nonTensorParts.push(`${i}=${serializeNonTensorArg(a)}`);
      }
    }

    const mod = this.target instanceof Module ? this.target : null;
    const state = mod && !this.freeze ? collectModuleState(mod) : null;
    const allInputs = state ? [...tensorArgs, ...state.uniqueTensors] : tensorArgs;

    const gradEnabled = isGradEnabled();
    const firstDevType = allInputs[0]?.device.type ?? 'cpu';
    const acDevice = firstDevType === 'cuda' ? 'cuda' : 'cpu';
    const acEnabled = isAutocastEnabled(acDevice);
    const acDtype = acEnabled ? getAutocastDtype(acDevice) : 'off';

    const inputSig = allInputs
      .map((t) => `${t.shape.join('x')}:${t.dtype}:${t.device.toString()}`)
      .join('|');
    const trainSig = mod ? `tr=${mod.training ? 1 : 0}` : '';
    const key = `${inputSig};${nonTensorParts.join(',')};g=${gradEnabled ? 1 : 0};ac=${acDtype};${trainSig}`;

    let entry = this.cache.get(key);
    if (!entry) {
      const numUserTensors = tensorArgs.length;
      const invokeDuringTrace = (tracedAll: Tensor[]): Tensor | Tensor[] => {
        const callArgs = [...args];
        for (let k = 0; k < numUserTensors; k++) {
          callArgs[tensorPositions[k]!] = tracedAll[k]!;
        }
        if (mod && state) {
          const map: Record<string, Tensor> = {};
          for (const item of state.entries) {
            map[item.name] = tracedAll[numUserTensors + item.idx]!;
          }
          return functionalCall<Tensor | Tensor[]>(mod, map, ...callArgs);
        }
        if (mod) {
          const detachedMap: Record<string, Tensor> = {};
          const detachedList: Tensor[] = [];
          try {
            for (const [name, p] of mod.namedParameters()) {
              const d = p.detach();
              detachedList.push(d);
              detachedMap[name] = d;
            }
            for (const [name, b] of mod.namedBuffers()) {
              const d = b.detach();
              detachedList.push(d);
              detachedMap[name] = d;
            }
            return functionalCall<Tensor | Tensor[]>(mod, detachedMap, ...callArgs);
          } finally {
            for (const d of detachedList) {
              if (!d.isDisposed) d.dispose();
            }
          }
        }
        return (this.target as (...a: unknown[]) => Tensor | Tensor[])(...callArgs);
      };

      // Under active autocast + grad, keep IR optimizations enabled but run GraphExecutor in
      // SIMPLE mode so ATen's native autocast-aware C++ autograd handles mixed-dtype backward
      // (avoiding legacy TorchScript DifferentiableGraph fp32/bf16 matmul mismatch).
      const traced = runTraceNative(invokeDuringTrace, allInputs, {
        strict: false,
        optimize: this.optimize,
        profilingExecutor: this.optimize && !(gradEnabled && acEnabled),
        freezeConstants: this.freeze,
        name: this.name,
      });

      entry = {
        traced,
        calls: 0,
        dispose() {
          if (this.cudaGraph) {
            this.cudaGraph.graph.reset();
            for (const t of this.cudaGraph.staticInputs) {
              if (!t.isDisposed) t.dispose();
            }
            for (const t of this.cudaGraph.staticOutputs) {
              if (!t.isDisposed) t.dispose();
            }
            this.cudaGraph = undefined;
          }
          if (!this.traced.isDisposed()) {
            this.traced.dispose();
          }
        },
      };
      this.cache.set(key, entry);
    }

    entry.calls += 1;

    // Optional CUDAGraph capture & replay for mode='reduce-overhead' in inference/no-grad CUDA mode
    const canCudaGraph =
      this.mode === 'reduce-overhead' &&
      !gradEnabled &&
      tensorArgs.length > 0 &&
      cuda.isAvailable() &&
      allInputs.every((t) => t.isCuda);

    if (canCudaGraph) {
      if (entry.cudaGraph) {
        const cg = entry.cudaGraph;
        noGrad(() => {
          for (let i = 0; i < tensorArgs.length; i++) {
            cg.staticInputs[i]!.copy_(tensorArgs[i]!);
          }
          cg.graph.replay();
        });
        return cg.singleOutput
          ? cg.staticOutputs[0]!.clone()
          : cg.staticOutputs.map((o) => o.clone());
      }

      // Capture on the 3rd call (after 2 GraphExecutor profiling/fusion warmup calls)
      if (entry.calls >= 3) {
        const captureStream = new CudaStream(tensorArgs[0]!.device.index ?? 0);
        const staticInputs = noGrad(() => tensorArgs.map((t) => keep(t.detach().clone())));
        const stateNative = state ? state.uniqueTensors.map((t) => t._native) : [];
        const fullStaticNative = [...staticInputs.map((t) => t._native), ...stateNative];

        // Warmup on capture stream to stabilize allocator & GraphExecutor plan
        let sampleOutputs: Tensor[] = [];
        let singleOutput = true;
        cuda.stream(captureStream, () => {
          noGrad(() => {
            for (let w = 0; w < 2; w++) {
              const res = runTracedToArray(entry!.traced, fullStaticNative);
              singleOutput = res.singleOutput;
              if (w === 1) {
                sampleOutputs = res.tensors;
              } else {
                for (const t of res.tensors) t.dispose();
              }
            }
          });
        });
        captureStream.synchronize();

        const staticOutputs = noGrad(() => sampleOutputs.map((t) => keep(t.detach().clone())));
        for (const t of sampleOutputs) t.dispose();

        const graph = new CUDAGraph();
        cuda.graph(
          graph,
          () => {
            noGrad(() => {
              const res = runTracedToArray(entry!.traced, fullStaticNative);
              for (let i = 0; i < res.tensors.length; i++) {
                staticOutputs[i]!.copy_(res.tensors[i]!);
                res.tensors[i]!.dispose();
              }
            });
          },
          { stream: captureStream }
        );

        entry.cudaGraph = {
          graph,
          stream: captureStream,
          staticInputs,
          staticOutputs,
          singleOutput,
        };

        // Replay once for current inputs and return cloned output(s)
        noGrad(() => {
          for (let i = 0; i < tensorArgs.length; i++) {
            staticInputs[i]!.copy_(tensorArgs[i]!);
          }
          graph.replay();
        });
        return singleOutput ? staticOutputs[0]!.clone() : staticOutputs.map((o) => o.clone());
      }
    }

    const rawOut = entry.traced.run(allInputs.map((t) => t._native));
    return wrapResult(rawOut) as Tensor | Tensor[];
  }
}

/**
 * Compiled wrapper around an `nn.Module` that traces and executes via `libtorch`'s C++
 * `GraphExecutor` while keeping module parameters live for full autograd training and
 * optimizer updates. Disposing a `CompiledModule` frees compiled graphs without disposing
 * the underlying `module`.
 */
export class CompiledModule<M extends Module = Module> implements Disposable {
  readonly module: M;
  private readonly engine: CompileEngine;

  constructor(mod: M, opts: CompileOptions = {}) {
    this.module = mod;
    this.engine = new CompileEngine(mod, opts);
  }

  forward(...args: unknown[]): Tensor | Tensor[] {
    return this.engine.invoke(args);
  }

  call(...args: unknown[]): Tensor | Tensor[] {
    return this.engine.invoke(args);
  }

  get training(): boolean {
    return this.module.training;
  }

  train(mode = true): this {
    this.module.train(mode);
    return this;
  }

  eval(): this {
    this.module.eval();
    return this;
  }

  to(...args: Parameters<Module['to']>): this {
    this.module.to(...args);
    this.engine.reset();
    return this;
  }

  parameters(recurse = true): ReturnType<M['parameters']> {
    return this.module.parameters(recurse) as ReturnType<M['parameters']>;
  }

  namedParameters(prefix = '', recurse = true): ReturnType<M['namedParameters']> {
    return this.module.namedParameters(prefix, recurse) as ReturnType<M['namedParameters']>;
  }

  buffers(recurse = true): ReturnType<M['buffers']> {
    return this.module.buffers(recurse) as ReturnType<M['buffers']>;
  }

  namedBuffers(prefix = '', recurse = true): ReturnType<M['namedBuffers']> {
    return this.module.namedBuffers(prefix, recurse) as ReturnType<M['namedBuffers']>;
  }

  stateDict(): ReturnType<M['stateDict']> {
    return this.module.stateDict() as ReturnType<M['stateDict']>;
  }

  loadStateDict(...args: Parameters<Module['loadStateDict']>): ReturnType<Module['loadStateDict']> {
    return this.module.loadStateDict(...args);
  }

  zeroGrad(setToNone = true): void {
    this.module.zeroGrad(setToNone);
  }

  /** Number of cached shape/dtype/mode specializations currently compiled. */
  get graphCount(): number {
    return this.engine.graphCount;
  }

  /** TorchScript IR representations of all cached compiled graphs. */
  graphStrings(): string[] {
    return this.engine.graphStrings();
  }

  /** Clears all cached graphs and CUDA graphs without disposing the underlying module. */
  reset(): void {
    this.engine.reset();
  }

  /** Releases all compiled graphs and CUDA graph buffers (does not dispose `this.module`). */
  dispose(): void {
    this.engine.reset();
  }

  [Symbol.dispose](): void {
    this.dispose();
  }
}

export interface CompiledFunction<F extends (...args: never[]) => Tensor | Tensor[]> extends Disposable {
  (...args: Parameters<F>): ReturnType<F>;
  readonly graphCount: number;
  graphStrings(): string[];
  reset(): void;
  dispose(): void;
}

/**
 * Pure-Node `torch.compile()` powered directly by `libtorch`'s C++ tracer, IR optimizer,
 * kernel-fusing `GraphExecutor`, C++ autodiff, and optional CUDA Graphs (`mode: 'reduce-overhead'`).
 * Zero Python required.
 */
export function compile<M extends Module>(mod: M, opts?: CompileOptions): CompiledModule<M>;
export function compile<F extends (...args: never[]) => Tensor | Tensor[]>(
  fn: F,
  opts?: CompileOptions
): CompiledFunction<F>;
export function compile(
  fnOrModule: Module | ((...args: never[]) => Tensor | Tensor[]),
  opts: CompileOptions = {}
): CompiledModule | CompiledFunction<(...args: never[]) => Tensor | Tensor[]> {
  if (fnOrModule instanceof Module) {
    return new CompiledModule(fnOrModule, opts);
  }
  const engine = new CompileEngine(
    fnOrModule as (...args: unknown[]) => Tensor | Tensor[],
    opts
  );
  const wrapped = ((...args: unknown[]) => engine.invoke(args)) as unknown as CompiledFunction<
    (...args: never[]) => Tensor | Tensor[]
  >;
  Object.defineProperty(wrapped, 'graphCount', {
    get: () => engine.graphCount,
  });
  wrapped.graphStrings = () => engine.graphStrings();
  wrapped.reset = () => engine.reset();
  wrapped.dispose = () => engine.reset();
  wrapped[Symbol.dispose] = () => engine.reset();
  return wrapped;
}
