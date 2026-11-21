import { Worker } from 'node:worker_threads';
import { cuda } from '../cuda.js';
import { distributed } from '../distributed.js';
import { dtypeElementSize, type DType } from '../dtype.js';
import type { TypedArray } from '../native.js';
import { fromTypedArray, stack } from '../ops.js';
import { Tensor } from '../tensor.js';

export interface Dataset<T = unknown> {
  readonly length: number;
  getItem(index: number): T;
}

export interface Sampler {
  readonly length: number;
  [Symbol.iterator](): Iterator<number>;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class SequentialSampler implements Sampler {
  readonly length: number;
  constructor(dataSource: { readonly length: number }) {
    this.length = dataSource.length;
  }
  *[Symbol.iterator](): Iterator<number> {
    for (let i = 0; i < this.length; i++) {
      yield i;
    }
  }
}

export class RandomSampler implements Sampler {
  readonly length: number;
  private readonly rng: () => number;

  constructor(dataSource: { readonly length: number }, seed?: number) {
    this.length = dataSource.length;
    this.rng = seed !== undefined ? mulberry32(seed) : Math.random;
  }

  *[Symbol.iterator](): Iterator<number> {
    const idx = Array.from({ length: this.length }, (_, i) => i);
    for (let i = this.length - 1; i > 0; i--) {
      const j = Math.floor(this.rng() * (i + 1));
      const tmp = idx[i]!;
      idx[i] = idx[j]!;
      idx[j] = tmp;
    }
    yield* idx;
  }
}

export interface DistributedSamplerOptions {
  numReplicas?: number;
  rank?: number;
  shuffle?: boolean;
  seed?: number;
  dropLast?: boolean;
}

export class DistributedSampler implements Sampler {
  readonly datasetLength: number;
  readonly numReplicas: number;
  readonly rank: number;
  readonly shuffle: boolean;
  readonly seed: number;
  readonly dropLast: boolean;
  readonly numSamples: number;
  readonly totalSize: number;
  private epoch = 0;

  constructor(dataSource: { readonly length: number }, opts?: DistributedSamplerOptions) {
    this.datasetLength = dataSource.length;
    this.numReplicas =
      opts?.numReplicas ?? (distributed.isInitialized() ? distributed.getWorldSize() : 1);
    this.rank = opts?.rank ?? (distributed.isInitialized() ? distributed.getRank() : 0);
    if (this.rank < 0 || this.rank >= this.numReplicas) {
      throw new RangeError(
        `Invalid rank ${this.rank} for DistributedSampler with numReplicas=${this.numReplicas}`
      );
    }
    this.shuffle = opts?.shuffle ?? true;
    this.seed = opts?.seed ?? 0;
    this.dropLast = opts?.dropLast ?? false;

    if (this.dropLast && this.datasetLength % this.numReplicas !== 0) {
      this.numSamples = Math.floor(this.datasetLength / this.numReplicas);
    } else {
      this.numSamples = Math.ceil(this.datasetLength / this.numReplicas);
    }
    this.totalSize = this.numSamples * this.numReplicas;
  }

  get length(): number {
    return this.numSamples;
  }

  setEpoch(epoch: number): void {
    this.epoch = epoch;
  }

  *[Symbol.iterator](): Iterator<number> {
    const indices = Array.from({ length: this.datasetLength }, (_, i) => i);
    if (this.shuffle) {
      const rng = mulberry32(this.seed + this.epoch);
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        const tmp = indices[i]!;
        indices[i] = indices[j]!;
        indices[j] = tmp;
      }
    }

    if (!this.dropLast) {
      const padding = this.totalSize - indices.length;
      if (padding > 0 && indices.length > 0) {
        for (let i = 0; i < padding; i++) {
          indices.push(indices[i % indices.length]!);
        }
      }
    } else {
      indices.length = this.totalSize;
    }

    for (let i = this.rank; i < this.totalSize; i += this.numReplicas) {
      yield indices[i]!;
    }
  }
}

export class TensorDataset implements Dataset<Tensor[]> {
  readonly tensors: Tensor[];
  readonly length: number;

  constructor(...tensors: Tensor[]) {
    if (tensors.length === 0) {
      throw new Error('TensorDataset requires at least one tensor');
    }
    const len = tensors[0]!.size(0);
    for (const t of tensors) {
      if (t.size(0) !== len) {
        throw new Error(`Size mismatch between tensors in TensorDataset (${t.size(0)} vs ${len})`);
      }
    }
    this.tensors = tensors;
    this.length = len;
  }

  getItem(index: number): Tensor[] {
    return this.tensors.map((t) => t.at(index));
  }
}

export function defaultCollate(batch: unknown[]): unknown {
  if (batch.length === 0) return [];
  const first = batch[0];
  if (first instanceof Tensor) {
    return stack(batch as Tensor[], 0);
  }
  if (Array.isArray(first)) {
    const transposed: unknown[] = [];
    for (let col = 0; col < first.length; col++) {
      transposed.push(defaultCollate(batch.map((row) => (row as unknown[])[col])));
    }
    return transposed;
  }
  if (first !== null && typeof first === 'object') {
    const out: Record<string, unknown> = {};
    for (const key of Object.keys(first as Record<string, unknown>)) {
      out[key] = defaultCollate(batch.map((row) => (row as Record<string, unknown>)[key]));
    }
    return out;
  }
  return batch;
}

function pinBatchRecursive(value: unknown): unknown {
  if (value instanceof Tensor) {
    if (value.device.type === 'cpu' && !value.isPinned) {
      const pinned = value.pinMemory();
      value.dispose();
      return pinned;
    }
    return value;
  }
  if (Array.isArray(value)) {
    return value.map(pinBatchRecursive);
  }
  if (value !== null && typeof value === 'object') {
    const proto = Object.getPrototypeOf(value);
    if (proto === Object.prototype || proto === null) {
      const out: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
        out[k] = pinBatchRecursive(v);
      }
      return out;
    }
  }
  return value;
}

function bufferToTypedArray(buffer: ArrayBuffer, dtype: DType): TypedArray {
  switch (dtype) {
    case 'float32':
    case 'complex64':
      return new Float32Array(buffer);
    case 'float64':
    case 'complex128':
      return new Float64Array(buffer);
    case 'int8':
      return new Int8Array(buffer);
    case 'uint8':
    case 'bool':
      return new Uint8Array(buffer);
    case 'int16':
      return new Int16Array(buffer);
    case 'int32':
      return new Int32Array(buffer);
    case 'int64':
      return new BigInt64Array(buffer);
    default:
      throw new Error(`Multi-worker TensorDataset does not support dtype '${dtype}' directly`);
  }
}

export interface DataLoaderOptions<TItem, TBatch> {
  batchSize?: number;
  shuffle?: boolean;
  sampler?: Sampler;
  dropLast?: boolean;
  pinMemory?: boolean;
  collateFn?: (batch: TItem[]) => TBatch;
  seed?: number;
  /** Number of Node worker_threads for parallel async prefetching (default 0). */
  numWorkers?: number;
  /** Number of batches prefetched per worker when numWorkers > 0 (default 2). */
  prefetchFactor?: number;
  /** Optional ESM worker module path exporting getItem(index, workerData) or getBatch(indices, workerData). */
  workerScript?: string;
  /** Arbitrary structured-cloneable data passed to custom workerScript. */
  workerData?: unknown;
}

interface WorkerTensorResultMsg {
  type: 'result';
  batchId: number;
  mode: 'tensor-dataset';
  columns: { buffer: ArrayBuffer; dtype: DType; shape: number[] }[];
}

interface WorkerCustomResultMsg {
  type: 'result';
  batchId: number;
  mode: 'custom';
  payload: unknown;
}

interface WorkerErrorMsg {
  type: 'error';
  batchId: number;
  error: string;
}

type WorkerOutgoingMsg = WorkerTensorResultMsg | WorkerCustomResultMsg | WorkerErrorMsg;

export class DataLoader<TItem = Tensor[], TBatch = Tensor[]>
  implements Iterable<TBatch>, AsyncIterable<TBatch>, AsyncDisposable
{
  readonly dataset: Dataset<TItem>;
  readonly batchSize: number;
  readonly shuffle: boolean;
  readonly sampler: Sampler;
  readonly dropLast: boolean;
  readonly pinMemory: boolean;
  readonly collateFn: (batch: TItem[]) => TBatch;
  readonly numWorkers: number;
  readonly prefetchFactor: number;
  readonly workerScript: string | undefined;
  readonly workerData: unknown;
  private readonly activeWorkers = new Set<Worker>();

  constructor(dataset: Dataset<TItem>, opts?: DataLoaderOptions<TItem, TBatch>) {
    this.dataset = dataset;
    this.batchSize = opts?.batchSize ?? 1;
    this.shuffle = opts?.shuffle ?? false;
    this.sampler =
      opts?.sampler ??
      (this.shuffle ? new RandomSampler(dataset, opts?.seed) : new SequentialSampler(dataset));
    this.dropLast = opts?.dropLast ?? false;
    this.pinMemory = opts?.pinMemory ?? false;
    this.collateFn = opts?.collateFn ?? (defaultCollate as (batch: TItem[]) => TBatch);
    this.numWorkers = opts?.numWorkers ?? 0;
    this.prefetchFactor = Math.max(1, opts?.prefetchFactor ?? 2);
    this.workerScript = opts?.workerScript;
    this.workerData = opts?.workerData;
    if (this.numWorkers < 0 || !Number.isInteger(this.numWorkers)) {
      throw new RangeError(`numWorkers must be a non-negative integer, got ${this.numWorkers}`);
    }
  }

  get length(): number {
    if (this.dropLast) {
      return Math.floor(this.sampler.length / this.batchSize);
    }
    return Math.ceil(this.sampler.length / this.batchSize);
  }

  async close(): Promise<void> {
    const workers = Array.from(this.activeWorkers);
    this.activeWorkers.clear();
    await Promise.all(
      workers.map(async (w) => {
        try {
          w.postMessage({ type: 'shutdown' });
        } catch {
          // ignore already-closed worker
        }
        await w.terminate();
      })
    );
  }

  async [Symbol.asyncDispose](): Promise<void> {
    await this.close();
  }

  *[Symbol.iterator](): Iterator<TBatch> {
    if (this.numWorkers > 0) {
      throw new Error(
        'DataLoader with numWorkers > 0 uses worker_threads and must be iterated with "for await (const batch of loader)"'
      );
    }
    const shouldPin = this.pinMemory && cuda.isAvailable();
    let bucket: number[] = [];

    const flush = (): TBatch => {
      const items = bucket.map((i) => this.dataset.getItem(i));
      let batch = this.collateFn(items);
      for (const item of items) {
        if (item instanceof Tensor) {
          item.dispose();
        } else if (Array.isArray(item)) {
          for (const sub of item) {
            if (sub instanceof Tensor) sub.dispose();
          }
        }
      }
      if (shouldPin) {
        batch = pinBatchRecursive(batch) as TBatch;
      }
      bucket = [];
      return batch;
    };

    for (const idx of this.sampler) {
      bucket.push(idx);
      if (bucket.length === this.batchSize) {
        yield flush();
      }
    }
    if (bucket.length > 0 && !this.dropLast) {
      yield flush();
    }
  }

  private buildBatchIndexList(): number[][] {
    const batches: number[][] = [];
    let bucket: number[] = [];
    for (const idx of this.sampler) {
      bucket.push(idx);
      if (bucket.length === this.batchSize) {
        batches.push(bucket);
        bucket = [];
      }
    }
    if (bucket.length > 0 && !this.dropLast) {
      batches.push(bucket);
    }
    return batches;
  }

  private buildWorkerConfig(): unknown {
    if (this.workerScript) {
      return {
        mode: 'custom',
        workerScript: this.workerScript,
        userWorkerData: this.workerData,
      };
    }
    if (this.dataset instanceof TensorDataset) {
      const columns = this.dataset.tensors.map((t) => {
        const cpu = t.device.type === 'cpu' && t.isContiguous ? t : t.cpu().contiguous();
        const ta = cpu.toTypedArray();
        const elemBytes = dtypeElementSize(cpu.dtype);
        const tailShape = cpu.shape.slice(1);
        const rowElements = tailShape.reduce((a, b) => a * b, 1);
        const rowBytes = rowElements * elemBytes;
        const sab = new SharedArrayBuffer(ta.byteLength);
        new Uint8Array(sab).set(new Uint8Array(ta.buffer, ta.byteOffset, ta.byteLength));
        if (cpu !== t) cpu.dispose();
        return {
          sab,
          dtype: t.dtype,
          rowBytes,
          tailShape,
        };
      });
      return {
        mode: 'tensor-dataset',
        columns,
      };
    }
    throw new Error(
      'DataLoader with numWorkers > 0 requires either a TensorDataset or an explicit workerScript option'
    );
  }

  async *[Symbol.asyncIterator](): AsyncIterator<TBatch> {
    if (this.numWorkers === 0) {
      for (const batch of this) {
        await Promise.resolve();
        yield batch;
      }
      return;
    }

    const batches = this.buildBatchIndexList();
    if (batches.length === 0) return;

    const shouldPin = this.pinMemory && cuda.isAvailable();
    const workerConfig = this.buildWorkerConfig();
    const workerUrl = new URL('./data_worker.js', import.meta.url);
    const workerCount = Math.min(this.numWorkers, batches.length);
    const maxInFlight = workerCount * this.prefetchFactor;

    const workers: Worker[] = [];
    const idleWorkers: Worker[] = [];
    const ready = new Map<number, TBatch>();
    let workerError: Error | null = null;
    let closed = false;
    let nextDispatch = 0;
    let nextYield = 0;
    let inFlight = 0;
    let wakeup: (() => void) | null = null;

    const notify = (): void => {
      if (wakeup) {
        const fn = wakeup;
        wakeup = null;
        fn();
      }
    };

    const dispatchMore = (): void => {
      while (
        !closed &&
        !workerError &&
        nextDispatch < batches.length &&
        inFlight < maxInFlight &&
        idleWorkers.length > 0
      ) {
        const w = idleWorkers.shift()!;
        const batchId = nextDispatch++;
        inFlight++;
        w.postMessage({ type: 'batch', batchId, indices: batches[batchId]! });
      }
    };

    for (let i = 0; i < workerCount; i++) {
      const w = new Worker(workerUrl, { workerData: workerConfig });
      workers.push(w);
      this.activeWorkers.add(w);
      idleWorkers.push(w);

      w.on('message', (msg: WorkerOutgoingMsg) => {
        if (closed) return;
        inFlight--;
        idleWorkers.push(w);
        if (msg.type === 'error') {
          workerError = new Error(`DataLoader worker error on batch ${msg.batchId}: ${msg.error}`);
        } else if (msg.mode === 'tensor-dataset') {
          let tensors: unknown = msg.columns.map((col) => {
            const ta = bufferToTypedArray(col.buffer, col.dtype);
            return fromTypedArray(ta, col.shape, { copy: false, dtype: col.dtype });
          });
          if (shouldPin) {
            tensors = pinBatchRecursive(tensors);
          }
          ready.set(msg.batchId, tensors as TBatch);
        } else {
          let batch = this.collateFn(msg.payload as TItem[]);
          if (shouldPin) {
            batch = pinBatchRecursive(batch) as TBatch;
          }
          ready.set(msg.batchId, batch);
        }
        dispatchMore();
        notify();
      });

      w.on('error', (err) => {
        if (closed) return;
        workerError = err instanceof Error ? err : new Error(String(err));
        notify();
      });
    }

    try {
      dispatchMore();
      while (nextYield < batches.length) {
        if (workerError) throw workerError;
        if (ready.has(nextYield)) {
          const batch = ready.get(nextYield)!;
          ready.delete(nextYield);
          nextYield++;
          dispatchMore();
          yield batch;
        } else {
          await new Promise<void>((resolve) => {
            wakeup = resolve;
          });
        }
      }
    } finally {
      closed = true;
      for (const w of workers) {
        w.removeAllListeners('message');
      }
      // Dispose any prefetched tensors that were not yielded (e.g. consumer exited early via break)
      for (const leftover of ready.values()) {
        if (leftover instanceof Tensor) {
          leftover.dispose();
        } else if (Array.isArray(leftover)) {
          for (const sub of leftover) {
            if (sub instanceof Tensor) sub.dispose();
          }
        }
      }
      ready.clear();
      await Promise.all(
        workers.map(async (w) => {
          this.activeWorkers.delete(w);
          try {
            w.postMessage({ type: 'shutdown' });
          } catch {
            // ignore
          }
          await w.terminate();
        })
      );
    }
  }
}
