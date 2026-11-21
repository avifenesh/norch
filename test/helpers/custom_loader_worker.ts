export function getItem(index: number, data?: { scale?: number }): Float32Array {
  const scale = data?.scale ?? 1;
  return new Float32Array([index * scale, index * scale + 1]);
}
