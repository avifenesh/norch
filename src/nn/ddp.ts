import { noGrad } from '../autograd.js';
import { distributed, ReduceOp, type ProcessGroup } from '../distributed.js';
import { Module, type HookHandle } from './module.js';

export interface DistributedDataParallelOptions {
  processGroup?: ProcessGroup;
  broadcastBuffers?: boolean;
}

/**
 * Wraps an nn.Module for synchronous multi-process data-parallel training.
 * Broadcasts rank-0 weights & buffers on construction and averages gradients
 * across ranks automatically during backward() via autograd parameter hooks.
 */
export class DistributedDataParallel<TModule extends Module = Module> extends Module {
  readonly module: TModule;
  readonly processGroup: ProcessGroup | undefined;
  readonly broadcastBuffers: boolean;
  private syncEnabled = true;
  private readonly hookHandles: HookHandle[] = [];

  constructor(module: TModule, opts?: DistributedDataParallelOptions) {
    super();
    this.module = module;
    this.processGroup = opts?.processGroup;
    this.broadcastBuffers = opts?.broadcastBuffers ?? true;
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

      for (const param of this.module.parameters()) {
        if (!param.requiresGrad) continue;
        const handle = param.registerHook((grad) => {
          if (!this.syncEnabled) {
            grad.dispose();
            return undefined;
          }
          const synced = grad.clone();
          grad.dispose();
          distributed.allReduce(synced, ReduceOp.AVG, this.processGroup);
          return synced;
        });
        this.hookHandles.push(handle);
      }
    }
  }

  forward(...args: unknown[]): unknown {
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
    for (const h of this.hookHandles) {
      h.remove();
    }
    this.hookHandles.length = 0;
    super.dispose();
  }
}
