import { native, type NativeProcessGroup } from './native.js';
import type { Tensor } from './tensor.js';

export const ReduceOp = {
  SUM: 'SUM',
  AVG: 'AVG',
  PRODUCT: 'PRODUCT',
  MIN: 'MIN',
  MAX: 'MAX',
} as const;

export type ReduceOpType = (typeof ReduceOp)[keyof typeof ReduceOp];

export interface InitProcessGroupOptions {
  backend?: 'gloo' | 'tcp' | 'nccl';
  initMethod?: string;
  rank?: number;
  worldSize?: number;
  masterAddr?: string;
  masterPort?: number;
  filePath?: string;
  timeoutMs?: number;
  groupName?: string;
}

export class ProcessGroup implements Disposable {
  readonly _native: NativeProcessGroup;

  constructor(handle: NativeProcessGroup) {
    this._native = handle;
  }

  get rank(): number {
    return this._native.rank();
  }

  get worldSize(): number {
    return this._native.worldSize();
  }

  get backend(): string {
    return this._native.backendName();
  }

  get isDestroyed(): boolean {
    return this._native.isDestroyed();
  }

  allReduce(tensor: Tensor, op: ReduceOpType = ReduceOp.SUM): void {
    this._native.allReduce(tensor._native, op);
  }

  broadcast(tensor: Tensor, src = 0): void {
    this._native.broadcast(tensor._native, src);
  }

  allGather(tensorList: Tensor[], tensor: Tensor): void {
    this._native.allGather(
      tensorList.map((t) => t._native),
      tensor._native
    );
  }

  allGatherIntoTensor(outputTensor: Tensor, inputTensor: Tensor): void {
    this._native.allGatherIntoTensor(outputTensor._native, inputTensor._native);
  }

  reduceScatter(output: Tensor, inputList: Tensor[], op: ReduceOpType = ReduceOp.SUM): void {
    this._native.reduceScatter(
      output._native,
      inputList.map((t) => t._native),
      op
    );
  }

  barrier(): void {
    this._native.barrier();
  }

  send(tensor: Tensor, dst: number, tag = 0): void {
    this._native.send(tensor._native, dst, tag);
  }

  recv(tensor: Tensor, src: number, tag = 0): void {
    this._native.recv(tensor._native, src, tag);
  }

  storeSet(key: string, value: string): void {
    this._native.storeSet(key, value);
  }

  storeGet(key: string): string {
    return this._native.storeGet(key);
  }

  destroy(): void {
    this._native.destroy();
  }

  [Symbol.dispose](): void {
    this.destroy();
  }
}

let defaultGroup: ProcessGroup | null = null;

function resolveGroup(group?: ProcessGroup): ProcessGroup {
  const g = group ?? defaultGroup;
  if (!g || g.isDestroyed) {
    throw new Error('Default process group has not been initialized. Call dist.initProcessGroup() first.');
  }
  return g;
}

function parseInitOptions(opts: InitProcessGroupOptions = {}): {
  backend: string;
  storeType: 'tcp' | 'file';
  rank: number;
  worldSize: number;
  masterAddr?: string;
  masterPort?: number;
  filePath?: string;
  timeoutMs: number;
  groupName: string;
} {
  const rank = opts.rank ?? (process.env.RANK !== undefined ? Number(process.env.RANK) : Number.NaN);
  const worldSize =
    opts.worldSize ?? (process.env.WORLD_SIZE !== undefined ? Number(process.env.WORLD_SIZE) : Number.NaN);
  if (!Number.isInteger(rank) || rank < 0) {
    throw new Error('initProcessGroup requires a valid non-negative rank (option or RANK env var)');
  }
  if (!Number.isInteger(worldSize) || worldSize < 1 || rank >= worldSize) {
    throw new Error(
      `initProcessGroup requires valid worldSize > rank (got rank=${rank}, worldSize=${worldSize})`
    );
  }

  const initMethod = opts.initMethod ?? 'env://';
  let storeType: 'tcp' | 'file' = opts.filePath ? 'file' : 'tcp';
  let masterAddr = opts.masterAddr ?? process.env.MASTER_ADDR ?? '127.0.0.1';
  let masterPort =
    opts.masterPort ?? (process.env.MASTER_PORT !== undefined ? Number(process.env.MASTER_PORT) : 29500);
  let filePath = opts.filePath;

  if (initMethod.startsWith('tcp://')) {
    storeType = 'tcp';
    const parsed = new URL(initMethod);
    masterAddr = parsed.hostname;
    masterPort = parsed.port ? Number(parsed.port) : masterPort;
  } else if (initMethod.startsWith('file://')) {
    storeType = 'file';
    filePath = initMethod.slice('file://'.length);
  } else if (initMethod !== 'env://') {
    throw new Error(`Unsupported initMethod '${initMethod}'. Use 'env://', 'tcp://host:port', or 'file:///path'.`);
  }

  return {
    backend: opts.backend ?? 'gloo',
    storeType,
    rank,
    worldSize,
    masterAddr,
    masterPort,
    filePath,
    timeoutMs: opts.timeoutMs ?? 60_000,
    groupName: opts.groupName ?? 'default',
  };
}

export function isAvailable(): boolean {
  return true;
}

export function isNcclAvailable(): boolean {
  return native.distIsNcclAvailable();
}

export function isGlooAvailable(): boolean {
  return native.distIsGlooAvailable();
}

export function isInitialized(): boolean {
  return defaultGroup !== null && !defaultGroup.isDestroyed;
}

export function initProcessGroup(opts: InitProcessGroupOptions = {}): ProcessGroup {
  if (defaultGroup && !defaultGroup.isDestroyed) {
    throw new Error('Default process group is already initialized. Call destroyProcessGroup() first.');
  }
  const resolved = parseInitOptions(opts);
  const pg = new ProcessGroup(new native.NativeProcessGroup(resolved));
  defaultGroup = pg;
  return pg;
}

export function newGroup(opts: InitProcessGroupOptions): ProcessGroup {
  const resolved = parseInitOptions(opts);
  return new ProcessGroup(new native.NativeProcessGroup(resolved));
}

export function destroyProcessGroup(group?: ProcessGroup): void {
  if (group) {
    group.destroy();
    if (group === defaultGroup) {
      defaultGroup = null;
    }
    return;
  }
  if (defaultGroup) {
    defaultGroup.destroy();
    defaultGroup = null;
  }
}

export function getRank(group?: ProcessGroup): number {
  return resolveGroup(group).rank;
}

export function getWorldSize(group?: ProcessGroup): number {
  return resolveGroup(group).worldSize;
}

export function getBackend(group?: ProcessGroup): string {
  return resolveGroup(group).backend;
}

export function allReduce(
  tensor: Tensor,
  op: ReduceOpType = ReduceOp.SUM,
  group?: ProcessGroup
): void {
  resolveGroup(group).allReduce(tensor, op);
}

export function broadcast(tensor: Tensor, src = 0, group?: ProcessGroup): void {
  resolveGroup(group).broadcast(tensor, src);
}

export function allGather(tensorList: Tensor[], tensor: Tensor, group?: ProcessGroup): void {
  resolveGroup(group).allGather(tensorList, tensor);
}

export function allGatherIntoTensor(
  outputTensor: Tensor,
  inputTensor: Tensor,
  group?: ProcessGroup
): void {
  resolveGroup(group).allGatherIntoTensor(outputTensor, inputTensor);
}

export function reduceScatter(
  output: Tensor,
  inputList: Tensor[],
  op: ReduceOpType = ReduceOp.SUM,
  group?: ProcessGroup
): void {
  resolveGroup(group).reduceScatter(output, inputList, op);
}

export function barrier(group?: ProcessGroup): void {
  resolveGroup(group).barrier();
}

export function send(tensor: Tensor, dst: number, group?: ProcessGroup, tag = 0): void {
  resolveGroup(group).send(tensor, dst, tag);
}

export function recv(tensor: Tensor, src: number, group?: ProcessGroup, tag = 0): void {
  resolveGroup(group).recv(tensor, src, tag);
}

export const distributed = {
  ReduceOp,
  ProcessGroup,
  isAvailable,
  isNcclAvailable,
  isGlooAvailable,
  isInitialized,
  initProcessGroup,
  newGroup,
  destroyProcessGroup,
  getRank,
  getWorldSize,
  getBackend,
  allReduce,
  broadcast,
  allGather,
  allGatherIntoTensor,
  reduceScatter,
  barrier,
  send,
  recv,
};
