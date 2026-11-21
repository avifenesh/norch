import { noGrad } from '../autograd.js';
import type { Device } from '../device.js';
import { distributed, ReduceOp, type DistWork, type ProcessGroup } from '../distributed.js';
import type { DType } from '../dtype.js';
import { untrackTensor } from '../memory.js';
import { empty } from '../ops.js';
import type { Tensor } from '../tensor.js';
import { Module, type HookHandle } from './module.js';
import type { Parameter } from './parameter.js';

export interface DistributedDataParallelOptions {
  processGroup?: ProcessGroup;
  broadcastBuffers?: boolean;
  /** Maximum gradient bucket size in MiB for coalesced async all-reduce (default: 25). */
  bucketCapMb?: number;
}

interface GradBucket {
  readonly dtype: DType;
  readonly device: Device;
  readonly params: Parameter[];
  readonly offsets: number[];
  totalNumel: number;
  flatBuf: Tensor | null;
  readyCount: number;
  pendingWork: DistWork | null;
}

/**
 * Wraps an nn.Module for synchronous multi-process data-parallel training.
 * Broadcasts rank-0 weights & buffers on construction and coalesces gradients
 * into contiguous buckets (`bucketCapMb`) with async `c10d::Work` all-reduce
 * overlapped with backward execution.
 */
export class DistributedDataParallel<TModule extends Module = Module> extends Module {
  readonly module: TModule;
  readonly processGroup: ProcessGroup | undefined;
  readonly broadcastBuffers: boolean;
  readonly bucketCapBytes: number;
  private syncEnabled = true;
  private readonly hookHandles: HookHandle[] = [];
  private readonly buckets: GradBucket[] = [];
  private bucketsCompletedThisStep = 0;

  constructor(module: TModule, opts?: DistributedDataParallelOptions) {
    super();
    this.module = module;
    this.processGroup = opts?.processGroup;
    this.broadcastBuffers = opts?.broadcastBuffers ?? true;
    const capMb = opts?.bucketCapMb ?? 25;
    this.bucketCapBytes = Math.max(0, Math.floor(capMb * 1024 * 1024));
    this.addModule('module', module);

    const worldSize = distributed.getWorldSize(this.processGroup);
    if (worldSize > 1) {
      noGrad(() => {
        for (const param of this.module.parameters()) {
          distributed.broadcast(param, 0, this.processGroup);
        }
        if (this.broadcastBuffers) {
          for (const buf of this.module.buffers()) {
            distributed.broadcast(buf, 0, this.processGroup);
          }
        }
      });

      this.buildBucketsAndHooks();
    }
  }

  get bucketCount(): number {
    return this.buckets.length;
  }

  private buildBucketsAndHooks(): void {
    // Reverse parameter order approximates backward readiness order (matching PyTorch DDP)
    const trainable = this.module
      .parameters()
      .filter((p) => p.requiresGrad)
      .reverse();

    let current: GradBucket | null = null;
    let currentBytes = 0;

    const flushCurrent = (): void => {
      if (current && current.params.length > 0) {
        this.buckets.push(current);
      }
      current = null;
      currentBytes = 0;
    };

    for (const p of trainable) {
      const dev = p.device;
      const devStr = dev.toString();
      const bytes = p.nbytes;
      const canJoin =
        current !== null &&
        current.dtype === p.dtype &&
        current.device.toString() === devStr &&
        this.bucketCapBytes > 0 &&
        currentBytes + bytes <= this.bucketCapBytes;

      if (!canJoin) {
        flushCurrent();
        current = {
          dtype: p.dtype,
          device: dev,
          params: [],
          offsets: [],
          totalNumel: 0,
          flatBuf: null,
          readyCount: 0,
          pendingWork: null,
        };
      }

      const bucket = current!;
      bucket.offsets.push(bucket.totalNumel);
      bucket.params.push(p);
      bucket.totalNumel += p.numel;
      currentBytes += bytes;

      if (this.bucketCapBytes === 0) {
        flushCurrent();
      }
    }
    flushCurrent();

    for (const bucket of this.buckets) {
      bucket.params.forEach((param, slotIdx) => {
        const offset = bucket.offsets[slotIdx]!;
        const numel = param.numel;
        const shape = param.shape;

        const handle = param.registerHook((grad) => {
          if (!this.syncEnabled) {
            grad.dispose();
            return undefined;
          }

          let retGrad!: Tensor;
          noGrad(() => {
            if (!bucket.flatBuf) {
              bucket.flatBuf = empty([bucket.totalNumel], {
                dtype: bucket.dtype,
                device: bucket.device,
              });
              untrackTensor(bucket.flatBuf);
            }
            const flatSlice = bucket.flatBuf.narrow(0, offset, numel);
            const flatGrad = grad.reshape(-1);
            flatSlice.copy_(flatGrad);
            flatGrad.dispose();

            // If grad accumulated from a prior noSync() microbatch, fold it before averaging
            const existingGrad = param.grad;
            if (existingGrad) {
              const existingFlat = existingGrad.reshape(-1);
              flatSlice.add_(existingFlat);
              existingFlat.dispose();
              existingGrad.dispose();
              param.grad = null;
            }
            flatSlice.dispose();
          });
          grad.dispose();

          bucket.readyCount += 1;
          if (bucket.readyCount === bucket.params.length) {
            // Launch async bucketed all-reduce overlapped with remaining backward buckets
            bucket.pendingWork =
              distributed.allReduce(bucket.flatBuf!, ReduceOp.AVG, this.processGroup, true) ?? null;
            this.bucketsCompletedThisStep += 1;

            // Once the last bucket finishes, wait on all in-flight works and write reduced grads back
            if (this.bucketsCompletedThisStep === this.buckets.length) {
              noGrad(() => {
                // Wait on all buckets and copy reduced slices into p.grad for every param except `param`
                // (whose AccumulateGrad will store `retGrad` immediately after this hook returns)
                for (const b of this.buckets) {
                  if (b.pendingWork) {
                    b.pendingWork.wait();
                    b.pendingWork = null;
                  }
                  if (b.flatBuf && !b.flatBuf.isDisposed) {
                    for (let i = 0; i < b.params.length; i++) {
                      const p = b.params[i]!;
                      const off = b.offsets[i]!;
                      const slice = b.flatBuf.narrow(0, off, p.numel).view(p.shape);
                      if (p === param) {
                        retGrad = slice.clone();
                      } else {
                        const existing = p.grad;
                        if (existing) {
                          existing.copy_(slice);
                          existing.dispose();
                        } else {
                          p.grad = slice.clone();
                        }
                      }
                      slice.dispose();
                    }
                    b.flatBuf.dispose();
                  }
                  b.flatBuf = null;
                  b.readyCount = 0;
                }
                this.bucketsCompletedThisStep = 0;
              });
              return retGrad;
            }
          }

          // Placeholder grad for AccumulateGrad; overwritten in-place once bucket work completes
          noGrad(() => {
            retGrad = empty(shape, { dtype: bucket.dtype, device: bucket.device });
          });
          return retGrad;
        });
        this.hookHandles.push(handle);
      });
    }
  }

  /**
   * Waits for any in-flight async bucket reductions to complete and copies reduced grads back.
   */
  finishGradSync(): void {
    noGrad(() => {
      for (const bucket of this.buckets) {
        if (bucket.pendingWork) {
          bucket.pendingWork.wait();
          bucket.pendingWork = null;
          if (bucket.flatBuf && !bucket.flatBuf.isDisposed) {
            for (let i = 0; i < bucket.params.length; i++) {
              const p = bucket.params[i]!;
              const off = bucket.offsets[i]!;
              const slice = bucket.flatBuf.narrow(0, off, p.numel).view(p.shape);
              const existing = p.grad;
              if (existing) {
                existing.copy_(slice);
                existing.dispose();
              } else {
                p.grad = slice.clone();
              }
              slice.dispose();
            }
          }
        }
        if (bucket.flatBuf && !bucket.flatBuf.isDisposed) {
          bucket.flatBuf.dispose();
        }
        bucket.flatBuf = null;
        bucket.readyCount = 0;
      }
      this.bucketsCompletedThisStep = 0;
    });
  }

  forward(...args: unknown[]): unknown {
    this.finishGradSync();
    if (this.training && this.broadcastBuffers && distributed.getWorldSize(this.processGroup) > 1) {
      noGrad(() => {
        for (const buf of this.module.buffers()) {
          distributed.broadcast(buf, 0, this.processGroup);
        }
      });
    }
    return this.module.call(...args);
  }

  /**
   * Temporarily disables gradient all-reduce for gradient accumulation steps.
   */
  noSync<R>(fn: () => R): R {
    const prev = this.syncEnabled;
    this.syncEnabled = false;
    try {
      const result = fn();
      if (result instanceof Promise) {
        return result.finally(() => {
          this.syncEnabled = prev;
        }) as R;
      }
      this.syncEnabled = prev;
      return result;
    } catch (err) {
      this.syncEnabled = prev;
      throw err;
    }
  }

  override dispose(): void {
    this.finishGradSync();
    for (const h of this.hookHandles) {
      h.remove();
    }
    this.hookHandles.length = 0;
    super.dispose();
  }
}
