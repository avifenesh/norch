import fs from 'node:fs';
import { native, type NativeProfilerEvent, type NativeProfilerResult } from './native.js';

export type ProfilerActivity = 'cpu' | 'cuda';

export interface ProfileOptions {
  activities?: ProfilerActivity[];
  recordShapes?: boolean;
  profileMemory?: boolean;
  withFlops?: boolean;
}

export type ProfilerEvent = NativeProfilerEvent;

export interface ProfilerKeyAverage {
  name: string;
  count: number;
  cpuTotalUs: number;
  cpuAvgUs: number;
  cudaTotalUs: number;
  cudaAvgUs: number;
  totalBytes: number;
  totalFlops: number;
}

export class ProfilerResult implements Disposable {
  private readonly _native: NativeProfilerResult;
  private cachedEvents: ProfilerEvent[] | null = null;

  constructor(handle: NativeProfilerResult) {
    this._native = handle;
  }

  get traceStartNs(): number {
    return this._native.traceStartNs();
  }

  get isDisposed(): boolean {
    return this._native.isDisposed();
  }

  events(): ProfilerEvent[] {
    if (!this.cachedEvents) {
      this.cachedEvents = this._native.events();
    }
    return this.cachedEvents;
  }

  keyAverages(): ProfilerKeyAverage[] {
    const map = new Map<string, ProfilerKeyAverage>();
    for (const ev of this.events()) {
      const existing = map.get(ev.name);
      if (existing) {
        existing.count += 1;
        existing.cpuTotalUs += ev.durationUs;
        existing.cudaTotalUs += ev.cudaTimeUs;
        existing.totalBytes += ev.nBytes;
        existing.totalFlops += ev.flops;
      } else {
        map.set(ev.name, {
          name: ev.name,
          count: 1,
          cpuTotalUs: ev.durationUs,
          cpuAvgUs: ev.durationUs,
          cudaTotalUs: ev.cudaTimeUs,
          cudaAvgUs: ev.cudaTimeUs,
          totalBytes: ev.nBytes,
          totalFlops: ev.flops,
        });
      }
    }
    const out = Array.from(map.values());
    for (const row of out) {
      row.cpuAvgUs = row.cpuTotalUs / row.count;
      row.cudaAvgUs = row.cudaTotalUs / row.count;
    }
    return out;
  }

  table(opts?: { sortBy?: 'cpuTotalUs' | 'cudaTotalUs' | 'count' | 'totalBytes'; rowLimit?: number }): string {
    const sortBy = opts?.sortBy ?? 'cpuTotalUs';
    const limit = opts?.rowLimit ?? 25;
    const rows = this.keyAverages()
      .sort((a, b) => b[sortBy] - a[sortBy])
      .slice(0, limit);

    const header = `${'Name'.padEnd(36)} ${'Calls'.padStart(7)} ${'CPU Total (us)'.padStart(15)} ${'CPU Avg (us)'.padStart(14)} ${'CUDA (us)'.padStart(11)}`;
    const sep = '-'.repeat(header.length);
    const lines = [header, sep];
    for (const r of rows) {
      const name = r.name.length > 35 ? r.name.slice(0, 32) + '...' : r.name;
      lines.push(
        `${name.padEnd(36)} ${String(r.count).padStart(7)} ${r.cpuTotalUs.toFixed(2).padStart(15)} ${r.cpuAvgUs.toFixed(2).padStart(14)} ${r.cudaTotalUs.toFixed(2).padStart(11)}`
      );
    }
    return lines.join('\n');
  }

  /**
   * Exports a Chrome trace (`chrome://tracing` / Perfetto) JSON file.
   * Uses libtorch Kineto's native trace exporter, falling back to a direct
   * Chrome Trace Event Format serializer if Kineto trace output is unavailable.
   */
  exportChromeTrace(filePath: string): void {
    try {
      this._native.saveChromeTrace(filePath);
      if (fs.existsSync(filePath) && fs.statSync(filePath).size > 2) {
        return;
      }
    } catch {
      // Fall back to structured trace event export below
    }
    const baseNs = this.traceStartNs;
    const traceEvents = this.events().map((ev) => ({
      name: ev.name,
      cat: ev.deviceType,
      ph: 'X',
      ts: Math.max(0, (ev.startNs - baseNs) / 1000),
      dur: ev.durationUs,
      pid: process.pid,
      tid: ev.threadId,
      args: {
        overload: ev.overloadName,
        bytes: ev.nBytes,
        flops: ev.flops,
        shapes: ev.shapes,
      },
    }));
    fs.writeFileSync(filePath, JSON.stringify({ traceEvents }));
  }

  dispose(): void {
    this._native.dispose();
  }

  [Symbol.dispose](): void {
    this.dispose();
  }
}

export function startProfiler(opts: ProfileOptions = {}): void {
  native.profilerStart({ ...opts });
}

export function stopProfiler(): ProfilerResult {
  return new ProfilerResult(native.profilerStop());
}

export function isProfilerEnabled(): boolean {
  return native.profilerIsEnabled();
}

export function profile<T>(fn: () => T): { result: T; prof: ProfilerResult };
export function profile<T>(opts: ProfileOptions, fn: () => T): { result: T; prof: ProfilerResult };
export function profile<T>(
  optsOrFn: ProfileOptions | (() => T),
  maybeFn?: () => T
): { result: T; prof: ProfilerResult } {
  const opts = typeof optsOrFn === 'function' ? {} : optsOrFn;
  const fn = typeof optsOrFn === 'function' ? optsOrFn : maybeFn!;
  startProfiler(opts);
  let result: T;
  let prof: ProfilerResult;
  try {
    result = fn();
  } finally {
    prof = stopProfiler();
  }
  return { result, prof };
}

export const profiler = {
  ProfilerResult,
  startProfiler,
  stopProfiler,
  isProfilerEnabled,
  profile,
};
