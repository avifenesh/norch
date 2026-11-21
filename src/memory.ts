import { native, type MemoryStats } from './native.js';

export interface DisposableTensor {
  dispose(): void;
  readonly isDisposed: boolean;
}

const scopeStack: Set<DisposableTensor>[] = [];

export function trackTensor<T extends DisposableTensor>(tensor: T): T {
  const top = scopeStack[scopeStack.length - 1];
  if (top) {
    top.add(tensor);
  }
  return tensor;
}

export function untrackTensor(tensor: DisposableTensor): void {
  for (let i = scopeStack.length - 1; i >= 0; i--) {
    scopeStack[i]?.delete(tensor);
  }
}

/**
 * Promote a tensor out of the current tidy()/scope() so it is not disposed when the scope exits.
 * If nested inside an outer scope, it is transferred to the parent scope.
 */
export function keep<T extends DisposableTensor>(tensor: T): T {
  if (scopeStack.length > 0) {
    scopeStack[scopeStack.length - 1]?.delete(tensor);
    if (scopeStack.length > 1) {
      scopeStack[scopeStack.length - 2]?.add(tensor);
    }
  }
  return tensor;
}

function collectReturnedTensors(value: unknown, kept: Set<DisposableTensor>, seen: Set<unknown>): void {
  if (value === null || value === undefined || typeof value !== 'object') return;
  if (seen.has(value)) return;
  seen.add(value);

  if (
    '_native' in value &&
    'dispose' in value &&
    'isDisposed' in value &&
    typeof (value as unknown as DisposableTensor).dispose === 'function'
  ) {
    kept.add(value as unknown as DisposableTensor);
    return;
  }
  if (Array.isArray(value)) {
    for (const item of value) {
      collectReturnedTensors(item, kept, seen);
    }
    return;
  }
  if (value instanceof Set) {
    for (const item of value) {
      collectReturnedTensors(item, kept, seen);
    }
    return;
  }
  if (value instanceof Map) {
    for (const v of value.values()) {
      collectReturnedTensors(v, kept, seen);
    }
    return;
  }
  const proto = Object.getPrototypeOf(value);
  if (proto === Object.prototype || proto === null) {
    for (const v of Object.values(value as Record<string, unknown>)) {
      collectReturnedTensors(v, kept, seen);
    }
  }
}

/**
 * Execute `fn` in a deterministic memory scope: all intermediate Tensors created
 * inside `fn` that are not returned (directly or inside arrays/objects/Maps/Sets)
 * or marked with `keep()` are disposed immediately on exit.
 */
export function tidy<T>(fn: () => T): T {
  const current = new Set<DisposableTensor>();
  scopeStack.push(current);
  let result: T;
  let threw = false;
  try {
    result = fn();
  } catch (err) {
    threw = true;
    throw err;
  } finally {
    scopeStack.pop();
    const kept = new Set<DisposableTensor>();
    if (!threw) {
      collectReturnedTensors(result!, kept, new Set());
    }
    const parent = scopeStack[scopeStack.length - 1];
    for (const t of current) {
      if (kept.has(t)) {
        if (parent) parent.add(t);
      } else if (!t.isDisposed) {
        t.dispose();
      }
    }
  }
  return result;
}

export class TensorScope implements Disposable {
  private readonly tracked = new Set<DisposableTensor>();
  private closed = false;

  constructor() {
    scopeStack.push(this.tracked);
  }

  keep<T extends DisposableTensor>(tensor: T): T {
    this.tracked.delete(tensor);
    const idx = scopeStack.lastIndexOf(this.tracked);
    if (idx > 0) {
      scopeStack[idx - 1]?.add(tensor);
    }
    return tensor;
  }

  dispose(): void {
    if (this.closed) return;
    this.closed = true;
    const idx = scopeStack.lastIndexOf(this.tracked);
    if (idx >= 0) {
      scopeStack.splice(idx, 1);
    }
    for (const t of this.tracked) {
      if (!t.isDisposed) {
        t.dispose();
      }
    }
    this.tracked.clear();
  }

  [Symbol.dispose](): void {
    this.dispose();
  }
}

export function scope(): TensorScope {
  return new TensorScope();
}

export function dispose(...items: (DisposableTensor | null | undefined)[]): void {
  for (const item of items) {
    if (item && !item.isDisposed) {
      item.dispose();
    }
  }
}

export function memoryStats(): MemoryStats {
  return native.memoryStats();
}
