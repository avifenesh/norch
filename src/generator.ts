import { Device, normalizeDevice, type DeviceLike } from './device.js';
import { native, type NativeGenerator } from './native.js';
import { Tensor, wrapTensor } from './tensor.js';

export class Generator {
  readonly _native: NativeGenerator;

  constructor(device: DeviceLike = 'cpu') {
    this._native = new native.NativeGenerator(normalizeDevice(device) ?? 'cpu');
  }

  manualSeed(seed: number | bigint): this {
    this._native.manualSeed(seed);
    return this;
  }

  seed(): bigint {
    return this._native.seed();
  }

  initialSeed(): bigint {
    return this._native.initialSeed();
  }

  get device(): Device {
    return new Device(this._native.device());
  }

  getState(): Tensor {
    return wrapTensor(this._native.getState());
  }

  setState(state: Tensor): void {
    this._native.setState(state._native);
  }
}
