export type DeviceType = 'cpu' | 'cuda' | 'mps' | 'meta';

export type DeviceLike =
  | DeviceType
  | `cuda:${number}`
  | `mps:${number}`
  | Device
  | { type: string; index?: number };

export class Device {
  readonly type: string;
  readonly index: number | undefined;

  constructor(typeOrSpec: DeviceLike | string, index?: number) {
    if (typeof typeOrSpec === 'string') {
      const parts = typeOrSpec.split(':');
      this.type = parts[0] ?? 'cpu';
      if (parts.length > 1 && parts[1] !== undefined) {
        this.index = Number.parseInt(parts[1], 10);
      } else {
        this.index = index;
      }
    } else if (typeOrSpec instanceof Device) {
      this.type = typeOrSpec.type;
      this.index = index ?? typeOrSpec.index;
    } else {
      this.type = typeOrSpec.type;
      this.index = index ?? typeOrSpec.index;
    }
  }

  toString(): string {
    return this.index !== undefined ? `${this.type}:${this.index}` : this.type;
  }

  equals(other: DeviceLike): boolean {
    const d = other instanceof Device ? other : new Device(other);
    return this.type === d.type && (this.index ?? 0) === (d.index ?? 0);
  }
}

export function device(spec: DeviceLike, index?: number): Device {
  return new Device(spec, index);
}

export function normalizeDevice(spec: DeviceLike | undefined): string | undefined {
  if (spec === undefined) return undefined;
  if (typeof spec === 'string') return spec;
  if (spec instanceof Device) return spec.toString();
  return spec.index !== undefined ? `${spec.type}:${spec.index}` : spec.type;
}
