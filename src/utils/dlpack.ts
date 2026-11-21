import { native, type NativeDlpackCapsule } from '../native.js';
import { Tensor, wrapTensor } from '../tensor.js';

export type DlpackCapsule = NativeDlpackCapsule;

/**
 * Exports a Tensor as a zero-copy DLPack capsule (DLManagedTensor*).
 * The capsule can be consumed once by `fromDlpack()` or passed to external
 * native addons via `dlpackPointer()`. Unconsumed capsules free themselves on GC.
 */
export function toDlpack(tensor: Tensor): DlpackCapsule {
  return native.toDlpack(tensor._native);
}

/**
 * Imports a zero-copy Tensor from a DLPack capsule, transferring ownership of
 * the underlying DLManagedTensor memory to libtorch.
 */
export function fromDlpack(capsule: DlpackCapsule): Tensor {
  return wrapTensor(native.fromDlpack(capsule));
}

/**
 * Returns the DLPack `(deviceType, deviceId)` tuple for a tensor.
 */
export function dlpackDevice(tensor: Tensor): [number, number] {
  return native.dlpackDevice(tensor._native);
}

/**
 * Returns the raw pointer address (`bigint`) of the underlying `DLManagedTensor*`
 * for interop with other native Node addons or FFI bindings.
 */
export function dlpackPointer(capsule: DlpackCapsule): bigint {
  return native.dlpackPointer(capsule);
}

export const dlpack = {
  toDlpack,
  fromDlpack,
  dlpackDevice,
  dlpackPointer,
};
