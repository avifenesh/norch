import {
  native,
  type CudaDeviceProperties,
  type CudaMemoryStats,
  type NativeCudaEvent,
  type NativeCudaGraph,
  type NativeCudaStream,
} from './native.js';

export class CudaStream {
  readonly _native: NativeCudaStream;

  constructor(deviceOrNative?: number | NativeCudaStream, priority = 0) {
    if (deviceOrNative !== undefined && typeof deviceOrNative === 'object') {
      this._native = deviceOrNative;
    } else {
      if (!native.NativeCudaStream) {
        throw new Error('CUDA Stream unavailable: built against CPU-only libtorch');
      }
      this._native = new native.NativeCudaStream(deviceOrNative, priority);
    }
  }

  synchronize(): void {
    this._native.synchronize();
  }

  query(): boolean {
    return this._native.query();
  }

  get deviceIndex(): number {
    return this._native.deviceIndex();
  }

  get streamId(): bigint {
    return this._native.streamId();
  }

  get priority(): number {
    return this._native.priority();
  }

  get isCapturing(): boolean {
    return this._native.isCapturing();
  }

  waitEvent(event: CudaEvent): void {
    this._native.waitEvent(event._native);
  }

  waitStream(stream: CudaStream): void {
    this._native.waitStream(stream._native);
  }

  recordEvent(event?: CudaEvent): CudaEvent {
    const ev = event ?? new CudaEvent();
    ev.record(this);
    return ev;
  }
}

export class CudaEvent {
  readonly _native: NativeCudaEvent;

  constructor(opts?: { enableTiming?: boolean; blocking?: boolean; interprocess?: boolean }) {
    if (!native.NativeCudaEvent) {
      throw new Error('CUDA Event unavailable: built against CPU-only libtorch');
    }
    this._native = new native.NativeCudaEvent(
      opts?.enableTiming ?? false,
      opts?.blocking ?? false,
      opts?.interprocess ?? false
    );
  }

  record(stream?: CudaStream): void {
    this._native.record(stream ? stream._native : undefined);
  }

  wait(stream?: CudaStream): void {
    this._native.wait(stream ? stream._native : undefined);
  }

  synchronize(): void {
    this._native.synchronize();
  }

  query(): boolean {
    return this._native.query();
  }

  elapsedTime(endEvent: CudaEvent): number {
    return this._native.elapsedTime(endEvent._native);
  }
}

export type GraphPoolHandle = [bigint, bigint];

export class CUDAGraph {
  readonly _native: NativeCudaGraph;

  constructor(keepGraph = false) {
    if (!native.NativeCudaGraph) {
      throw new Error('CUDAGraph unavailable: built against CPU-only libtorch');
    }
    this._native = new native.NativeCudaGraph(keepGraph);
  }

  captureBegin(pool?: GraphPoolHandle | null): void {
    this._native.captureBegin(pool ?? undefined);
  }

  captureEnd(): void {
    this._native.captureEnd();
  }

  replay(): void {
    this._native.replay();
  }

  reset(): void {
    this._native.reset();
  }

  pool(): GraphPoolHandle {
    return this._native.pool();
  }
}

export const cuda = {
  Stream: CudaStream,
  Event: CudaEvent,
  CUDAGraph,

  isAvailable(): boolean {
    return native.cudaIsAvailable();
  },
  deviceCount(): number {
    return native.cudaDeviceCount();
  },
  currentDevice(): number {
    return native.cudaCurrentDevice();
  },
  setDevice(index: number): void {
    native.cudaSetDevice(index);
  },
  synchronize(index?: number): void {
    native.cudaSynchronize(index);
  },
  emptyCache(): void {
    native.cudaEmptyCache();
  },
  setMemoryFraction(fraction: number, index?: number): void {
    native.cudaSetMemoryFraction(fraction, index);
  },
  resetPeakMemoryStats(index?: number): void {
    native.cudaResetPeakMemoryStats(index);
  },
  memoryStats(index?: number): CudaMemoryStats {
    return native.cudaMemoryStats(index);
  },
  memoryAllocated(index?: number): number {
    return native.cudaMemoryStats(index).allocated;
  },
  maxMemoryAllocated(index?: number): number {
    return native.cudaMemoryStats(index).maxAllocated;
  },
  memoryReserved(index?: number): number {
    return native.cudaMemoryStats(index).reserved;
  },
  deviceName(index?: number): string {
    return native.cudaDeviceName(index);
  },
  deviceProperties(index?: number): CudaDeviceProperties {
    return native.cudaDeviceProperties(index);
  },
  deviceCapability(index?: number): [number, number] {
    const p = native.cudaDeviceProperties(index);
    return [p.major, p.minor];
  },
  manualSeed(seed: number | bigint): void {
    native.cudaManualSeed(seed);
  },
  manualSeedAll(seed: number | bigint): void {
    native.cudaManualSeedAll(seed);
  },
  currentStream(index?: number): CudaStream {
    return new CudaStream(native.cudaCurrentStream(index));
  },
  defaultStream(index?: number): CudaStream {
    return new CudaStream(native.cudaDefaultStream(index));
  },
  setStream(stream: CudaStream): void {
    native.cudaSetStream(stream._native);
  },
  stream<T>(stream: CudaStream, fn: () => T): T {
    const prev = cuda.currentStream(stream.deviceIndex);
    cuda.setStream(stream);
    try {
      return fn();
    } finally {
      cuda.setStream(prev);
    }
  },
  graphPoolHandle(): GraphPoolHandle {
    return native.cudaGraphPoolHandle();
  },
  graph<T>(
    g: CUDAGraph,
    fn: () => T,
    opts?: { pool?: GraphPoolHandle; stream?: CudaStream }
  ): T {
    const captureStream = opts?.stream ?? new CudaStream();
    const prevStream = cuda.currentStream(captureStream.deviceIndex);
    captureStream.waitStream(prevStream);
    cuda.setStream(captureStream);
    g.captureBegin(opts?.pool);
    try {
      return fn();
    } finally {
      g.captureEnd();
      prevStream.waitStream(captureStream);
      cuda.setStream(prevStream);
    }
  },
};

export const mps = {
  isAvailable(): boolean {
    return native.mpsIsAvailable();
  },
};

export function manualSeed(seed: number | bigint): void {
  native.manualSeed(seed);
}

export function getNumThreads(): number {
  return native.getNumThreads();
}

export function setNumThreads(n: number): void {
  native.setNumThreads(n);
}

export function libtorchVersion(): string {
  return native.libtorchVersion();
}
