import fs from 'node:fs';
import { type DType } from '../dtype.js';
import type { DeviceLike } from '../device.js';
import { fromTypedArray } from '../ops.js';
import type { Tensor } from '../tensor.js';

type SafeDType = 'BOOL' | 'U8' | 'I8' | 'I16' | 'I32' | 'I64' | 'F16' | 'BF16' | 'F32' | 'F64';

const DTYPE_TO_SAFE: Record<string, SafeDType> = {
  bool: 'BOOL',
  uint8: 'U8',
  int8: 'I8',
  int16: 'I16',
  int32: 'I32',
  int64: 'I64',
  float16: 'F16',
  bfloat16: 'BF16',
  float32: 'F32',
  float64: 'F64',
};

const SAFE_TO_DTYPE: Record<SafeDType, DType> = {
  BOOL: 'bool',
  U8: 'uint8',
  I8: 'int8',
  I16: 'int16',
  I32: 'int32',
  I64: 'int64',
  F16: 'float16',
  BF16: 'bfloat16',
  F32: 'float32',
  F64: 'float64',
};

interface SafeHeaderEntry {
  dtype: SafeDType;
  shape: number[];
  data_offsets: [number, number];
}

export function encodeSafetensors(
  tensors: Record<string, Tensor>,
  metadata?: Record<string, string>
): Uint8Array {
  const header: Record<string, unknown> = {};
  if (metadata && Object.keys(metadata).length > 0) {
    header['__metadata__'] = metadata;
  }

  const chunks: Uint8Array[] = [];
  let offset = 0;

  for (const name of Object.keys(tensors).sort()) {
    const t = tensors[name]!;
    const safeDtype = DTYPE_TO_SAFE[t.dtype];
    if (!safeDtype) {
      throw new Error(`Unsupported dtype for safetensors: ${t.dtype} (tensor '${name}')`);
    }
    const ta = t.toTypedArray();
    const bytes = new Uint8Array(ta.buffer, ta.byteOffset, ta.byteLength);
    const start = offset;
    const end = start + bytes.byteLength;
    header[name] = {
      dtype: safeDtype,
      shape: t.shape,
      data_offsets: [start, end],
    } satisfies SafeHeaderEntry;
    chunks.push(bytes);
    offset = end;
  }

  const headerJson = Buffer.from(JSON.stringify(header), 'utf8');
  const padLen = (8 - (headerJson.byteLength % 8)) % 8;
  const paddedHeaderLen = headerJson.byteLength + padLen;

  const totalLen = 8 + paddedHeaderLen + offset;
  const out = new Uint8Array(totalLen);
  const dv = new DataView(out.buffer);
  dv.setBigUint64(0, BigInt(paddedHeaderLen), true);
  out.set(headerJson, 8);
  for (let i = 0; i < padLen; i++) {
    out[8 + headerJson.byteLength + i] = 0x20; // ASCII space padding per safetensors spec
  }

  let cursor = 8 + paddedHeaderLen;
  for (const chunk of chunks) {
    out.set(chunk, cursor);
    cursor += chunk.byteLength;
  }
  return out;
}

export function decodeSafetensors(
  buffer: Uint8Array,
  opts?: { device?: DeviceLike }
): { tensors: Record<string, Tensor>; metadata: Record<string, string> } {
  if (buffer.byteLength < 8) {
    throw new Error('Invalid safetensors buffer: shorter than 8 bytes');
  }
  const dv = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  const headerLen = Number(dv.getBigUint64(0, true));
  const dataStart = 8 + headerLen;
  if (dataStart > buffer.byteLength) {
    throw new Error(`Invalid safetensors header length ${headerLen}`);
  }

  const headerStr = Buffer.from(buffer.buffer, buffer.byteOffset + 8, headerLen).toString('utf8');
  const parsed = JSON.parse(headerStr) as Record<string, unknown>;
  const metadata = (parsed['__metadata__'] as Record<string, string> | undefined) ?? {};
  const tensors: Record<string, Tensor> = {};

  for (const [name, rawEntry] of Object.entries(parsed)) {
    if (name === '__metadata__') continue;
    const entry = rawEntry as SafeHeaderEntry;
    const dtype = SAFE_TO_DTYPE[entry.dtype];
    if (!dtype) {
      throw new Error(`Unsupported safetensors dtype '${entry.dtype}' for tensor '${name}'`);
    }
    const [relStart, relEnd] = entry.data_offsets;
    const sliceBytes = buffer.subarray(dataStart + relStart, dataStart + relEnd);
    tensors[name] = fromTypedArray(sliceBytes, entry.shape, {
      dtype,
      device: opts?.device ?? 'cpu',
      copy: true,
    });
  }

  return { tensors, metadata };
}

export function saveSafetensors(
  filePath: string,
  tensors: Record<string, Tensor>,
  metadata?: Record<string, string>
): void {
  const bytes = encodeSafetensors(tensors, metadata);
  fs.writeFileSync(filePath, bytes);
}

export function loadSafetensors(
  filePath: string,
  opts?: { device?: DeviceLike }
): { tensors: Record<string, Tensor>; metadata: Record<string, string> } {
  const buf = fs.readFileSync(filePath);
  return decodeSafetensors(buf, opts);
}
