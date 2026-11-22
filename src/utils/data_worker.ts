import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { parentPort, workerData } from 'node:worker_threads';
import type { DType } from '../dtype.js';

interface TensorColumnSpec {
  sab: SharedArrayBuffer;
  dtype: DType;
  rowBytes: number;
  tailShape: number[];
}

interface TensorDatasetWorkerConfig {
  mode: 'tensor-dataset';
  columns: TensorColumnSpec[];
}

interface CustomWorkerConfig {
  mode: 'custom';
  workerScript: string;
  userWorkerData?: unknown;
}

type WorkerConfig = TensorDatasetWorkerConfig | CustomWorkerConfig;

interface BatchJobMessage {
  type: 'batch';
  batchId: number;
  indices: number[];
}

interface ShutdownMessage {
  type: 'shutdown';
}

type IncomingMessage = BatchJobMessage | ShutdownMessage;

const port = parentPort;
if (!port) {
  throw new Error('data_worker.ts must be executed inside a Worker thread');
}

const config = workerData as WorkerConfig;

let customModPromise: Promise<{
  getItem?: (index: number, data?: unknown) => unknown | Promise<unknown>;
  getBatch?: (indices: number[], data?: unknown) => unknown | Promise<unknown>;
}> | null = null;

function collectTransferables(value: unknown, out: ArrayBuffer[]): void {
  if (value instanceof ArrayBuffer) {
    out.push(value);
    return;
  }
  if (ArrayBuffer.isView(value) && value.buffer instanceof ArrayBuffer) {
    out.push(value.buffer);
    return;
  }
  if (Array.isArray(value)) {
    for (const item of value) collectTransferables(item, out);
    return;
  }
  if (value !== null && typeof value === 'object') {
    for (const v of Object.values(value as Record<string, unknown>)) {
      collectTransferables(v, out);
    }
  }
}

port.on('message', async (msg: IncomingMessage) => {
  if (msg.type === 'shutdown') {
    port.close();
    return;
  }
  const { batchId, indices } = msg;
  try {
    if (config.mode === 'tensor-dataset') {
      const transfers: ArrayBuffer[] = [];
      const columnsOut = config.columns.map((col) => {
        const src = new Uint8Array(col.sab);
        const totalBytes = indices.length * col.rowBytes;
        const dstBuf = new ArrayBuffer(totalBytes);
        const dst = new Uint8Array(dstBuf);
        for (let i = 0; i < indices.length; i++) {
          const rowIdx = indices[i]!;
          const srcOffset = rowIdx * col.rowBytes;
          dst.set(src.subarray(srcOffset, srcOffset + col.rowBytes), i * col.rowBytes);
        }
        transfers.push(dstBuf);
        return {
          buffer: dstBuf,
          dtype: col.dtype,
          shape: [indices.length, ...col.tailShape],
        };
      });
      port.postMessage({ type: 'result', batchId, mode: 'tensor-dataset', columns: columnsOut }, transfers);
    } else {
      if (!customModPromise) {
        const specifier = config.workerScript.startsWith('file://')
          ? config.workerScript
          : pathToFileURL(path.resolve(config.workerScript)).href;
        customModPromise = import(specifier);
      }
      const mod = await customModPromise;
      let payload: unknown;
      if (typeof mod.getBatch === 'function') {
        payload = await mod.getBatch(indices, config.userWorkerData);
      } else if (typeof mod.getItem === 'function') {
        const items: unknown[] = [];
        for (const idx of indices) {
          items.push(await mod.getItem(idx, config.userWorkerData));
        }
        payload = items;
      } else {
        throw new Error(
          `DataLoader workerScript '${config.workerScript}' must export getItem(index, data) or getBatch(indices, data)`
        );
      }
      const transfers: ArrayBuffer[] = [];
      collectTransferables(payload, transfers);
      port.postMessage({ type: 'result', batchId, mode: 'custom', payload }, transfers);
    }
  } catch (err) {
    const message = err instanceof Error ? err.stack ?? err.message : String(err);
    port.postMessage({ type: 'error', batchId, error: message });
  }
});
