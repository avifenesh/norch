import { stack } from '../ops.js';
import { Tensor } from '../tensor.js';

export interface Dataset<T = unknown> {
  readonly length: number;
  getItem(index: number): T;
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

export interface DataLoaderOptions<TItem, TBatch> {
  batchSize?: number;
  shuffle?: boolean;
  dropLast?: boolean;
  collateFn?: (batch: TItem[]) => TBatch;
  seed?: number;
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

export class DataLoader<TItem = Tensor[], TBatch = Tensor[]>
  implements Iterable<TBatch>, AsyncIterable<TBatch>
{
  readonly dataset: Dataset<TItem>;
  readonly batchSize: number;
  readonly shuffle: boolean;
  readonly dropLast: boolean;
  readonly collateFn: (batch: TItem[]) => TBatch;
  private rng: () => number;

  constructor(dataset: Dataset<TItem>, opts?: DataLoaderOptions<TItem, TBatch>) {
    this.dataset = dataset;
    this.batchSize = opts?.batchSize ?? 1;
    this.shuffle = opts?.shuffle ?? false;
    this.dropLast = opts?.dropLast ?? false;
    this.collateFn = opts?.collateFn ?? (defaultCollate as (batch: TItem[]) => TBatch);
    this.rng = opts?.seed !== undefined ? mulberry32(opts.seed) : Math.random;
  }

  get length(): number {
    if (this.dropLast) {
      return Math.floor(this.dataset.length / this.batchSize);
    }
    return Math.ceil(this.dataset.length / this.batchSize);
  }

  private makeIndices(): number[] {
    const n = this.dataset.length;
    const idx = Array.from({ length: n }, (_, i) => i);
    if (this.shuffle) {
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(this.rng() * (i + 1));
        const tmp = idx[i]!;
        idx[i] = idx[j]!;
        idx[j] = tmp;
      }
    }
    return idx;
  }

  *[Symbol.iterator](): Iterator<TBatch> {
    const indices = this.makeIndices();
    for (let start = 0; start < indices.length; start += this.batchSize) {
      const end = start + this.batchSize;
      if (this.dropLast && end > indices.length) break;
      const sliceIdx = indices.slice(start, end);
      const items = sliceIdx.map((i) => this.dataset.getItem(i));
      const batch = this.collateFn(items);
      // Dispose per-sample slices created by TensorDataset.getItem() when collated into stacked batches
      for (const item of items) {
        if (item instanceof Tensor) {
          item.dispose();
        } else if (Array.isArray(item)) {
          for (const sub of item) {
            if (sub instanceof Tensor) sub.dispose();
          }
        }
      }
      yield batch;
    }
  }

  async *[Symbol.asyncIterator](): AsyncIterator<TBatch> {
    for (const batch of this) {
      await Promise.resolve();
      yield batch;
    }
  }
}
