import { native, type NativeDistWork, type NativeProcessGroup } from './native.js';
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

export class DistWork {
  readonly _native: NativeDistWork;

  constructor(handle: NativeDistWork) {
    this._native = handle;
  }

  wait(timeoutMs?: number): boolean {
    return this._native.wait(timeoutMs);
  }

  isCompleted(): boolean {
    return this._native.isCompleted();
  }

  isSuccess(): boolean {
    return this._native.isSuccess();
  }
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

  allReduce(tensor: Tensor, op: ReduceOpType = ReduceOp.SUM, asyncOp = false): DistWork | undefined {
    const w = this._native.allReduce(tensor._native, op, asyncOp);
    return w ? new DistWork(w) : undefined;
  }

  broadcast(tensor: Tensor, src = 0, asyncOp = false): DistWork | undefined {
    const w = this._native.broadcast(tensor._native, src, asyncOp);
    return w ? new DistWork(w) : undefined;
  }

  allGather(tensorList: Tensor[], tensor: Tensor, asyncOp = false): DistWork | undefined {
    const w = this._native.allGather(
      tensorList.map((t) => t._native),
      tensor._native,
      asyncOp
    );
    return w ? new DistWork(w) : undefined;
  }

  allGatherIntoTensor(
    outputTensor: Tensor,
    inputTensor: Tensor,
    asyncOp = false
  ): DistWork | undefined {
    const w = this._native.allGatherIntoTensor(outputTensor._native, inputTensor._native, asyncOp);
    return w ? new DistWork(w) : undefined;
  }

  gather(
    tensor: Tensor,
    gatherList?: Tensor[] | null,
    dst = 0,
    asyncOp = false
  ): DistWork | undefined {
    const w = this._native.gather(
      tensor._native,
      gatherList ? gatherList.map((t) => t._native) : null,
      dst,
      asyncOp
    );
    return w ? new DistWork(w) : undefined;
  }

  scatter(
    tensor: Tensor,
    scatterList?: Tensor[] | null,
    src = 0,
    asyncOp = false
  ): DistWork | undefined {
    const w = this._native.scatter(
      tensor._native,
      scatterList ? scatterList.map((t) => t._native) : null,
      src,
      asyncOp
    );
    return w ? new DistWork(w) : undefined;
  }

  allToAll(
    outputTensorList: Tensor[],
    inputTensorList: Tensor[],
    asyncOp = false
  ): DistWork | undefined {
    const w = this._native.allToAll(
      outputTensorList.map((t) => t._native),
      inputTensorList.map((t) => t._native),
      asyncOp
    );
    return w ? new DistWork(w) : undefined;
  }

  reduceScatter(
    output: Tensor,
    inputList: Tensor[],
    op: ReduceOpType = ReduceOp.SUM,
    asyncOp = false
  ): DistWork | undefined {
    const w = this._native.reduceScatter(
      output._native,
      inputList.map((t) => t._native),
      op,
      asyncOp
    );
    return w ? new DistWork(w) : undefined;
  }

  barrier(asyncOp = false): DistWork | undefined {
    const w = this._native.barrier(asyncOp);
    return w ? new DistWork(w) : undefined;
  }

  send(tensor: Tensor, dst: number, tag = 0): void {
    this._native.send(tensor._native, dst, tag);
  }

  recv(tensor: Tensor, src: number, tag = 0): void {
    this._native.recv(tensor._native, src, tag);
  }

  newSubgroup(groupName: string, ranks?: number[]): ProcessGroup | null {
    const sub = this._native.newSubgroup(groupName, ranks ?? null);
    return sub ? new ProcessGroup(sub) : null;
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
let subgroupCounter = 0;

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
  subgroupCounter = 0;
  return pg;
}

/**
 * Creates a subgroup reusing the default process group's underlying store via a
 * unique `PrefixStore` (no extra TCP sockets). Returns `null` on ranks outside `ranks`.
 */
export function newGroup(ranks?: number[], opts?: { groupName?: string }): ProcessGroup | null {
  const base = resolveGroup();
  subgroupCounter += 1;
  const name = opts?.groupName ?? `subgroup_${subgroupCounter}`;
  return base.newSubgroup(name, ranks);
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
  group?: ProcessGroup,
  asyncOp = false
): DistWork | undefined {
  return resolveGroup(group).allReduce(tensor, op, asyncOp);
}

export function broadcast(
  tensor: Tensor,
  src = 0,
  group?: ProcessGroup,
  asyncOp = false
): DistWork | undefined {
  return resolveGroup(group).broadcast(tensor, src, asyncOp);
}

export function allGather(
  tensorList: Tensor[],
  tensor: Tensor,
  group?: ProcessGroup,
  asyncOp = false
): DistWork | undefined {
  return resolveGroup(group).allGather(tensorList, tensor, asyncOp);
}

export function allGatherIntoTensor(
  outputTensor: Tensor,
  inputTensor: Tensor,
  group?: ProcessGroup,
  asyncOp = false
): DistWork | undefined {
  return resolveGroup(group).allGatherIntoTensor(outputTensor, inputTensor, asyncOp);
}

export function gather(
  tensor: Tensor,
  gatherList?: Tensor[] | null,
  dst = 0,
  group?: ProcessGroup,
  asyncOp = false
): DistWork | undefined {
  return resolveGroup(group).gather(tensor, gatherList, dst, asyncOp);
}

export function scatter(
  tensor: Tensor,
  scatterList?: Tensor[] | null,
  src = 0,
  group?: ProcessGroup,
  asyncOp = false
): DistWork | undefined {
  return resolveGroup(group).scatter(tensor, scatterList, src, asyncOp);
}

export function allToAll(
  outputTensorList: Tensor[],
  inputTensorList: Tensor[],
  group?: ProcessGroup,
  asyncOp = false
): DistWork | undefined {
  return resolveGroup(group).allToAll(outputTensorList, inputTensorList, asyncOp);
}

export function reduceScatter(
  output: Tensor,
  inputList: Tensor[],
  op: ReduceOpType = ReduceOp.SUM,
  group?: ProcessGroup,
  asyncOp = false
): DistWork | undefined {
  return resolveGroup(group).reduceScatter(output, inputList, op, asyncOp);
}

export function barrier(group?: ProcessGroup, asyncOp = false): DistWork | undefined {
  return resolveGroup(group).barrier(asyncOp);
}

export function send(tensor: Tensor, dst: number, group?: ProcessGroup, tag = 0): void {
  resolveGroup(group).send(tensor, dst, tag);
}

export function recv(tensor: Tensor, src: number, group?: ProcessGroup, tag = 0): void {
  resolveGroup(group).recv(tensor, src, tag);
}

export const distributed = {
  ReduceOp,
  DistWork,
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
  gather,
  scatter,
  allToAll,
  reduceScatter,
  barrier,
  send,
  recv,
};
