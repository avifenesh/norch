import { native, type CudaMemoryStats } from './native.js';

export const cuda = {
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
