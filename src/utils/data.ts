import { cuda } from '../cuda.js';
import { stack } from '../ops.js';
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

export interface DataLoaderOptions<TItem, TBatch> {
  batchSize?: number;
  shuffle?: boolean;
  sampler?: Sampler;
  dropLast?: boolean;
  pinMemory?: boolean;
  collateFn?: (batch: TItem[]) => TBatch;
  seed?: number;
}

export class DataLoader<TItem = Tensor[], TBatch = Tensor[]>
  implements Iterable<TBatch>, AsyncIterable<TBatch>
{
  readonly dataset: Dataset<TItem>;
  readonly batchSize: number;
  readonly shuffle: boolean;
  readonly sampler: Sampler;
  readonly dropLast: boolean;
  readonly pinMemory: boolean;
  readonly collateFn: (batch: TItem[]) => TBatch;

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
  }

  get length(): number {
    if (this.dropLast) {
      return Math.floor(this.sampler.length / this.batchSize);
    }
    return Math.ceil(this.sampler.length / this.batchSize);
  }

  *[Symbol.iterator](): Iterator<TBatch> {
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

  async *[Symbol.asyncIterator](): AsyncIterator<TBatch> {
    for (const batch of this) {
      await Promise.resolve();
      yield batch;
    }
  }
}
