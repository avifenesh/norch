export type DType =
  | 'float32'
  | 'float64'
  | 'float16'
  | 'bfloat16'
  | 'int8'
  | 'uint8'
  | 'int16'
  | 'int32'
  | 'int64'
  | 'bool'
  | 'complex64'
  | 'complex128'
  | 'float8_e4m3fn'
  | 'float8_e5m2';

export const float32: DType = 'float32';
export const float: DType = 'float32';
export const float64: DType = 'float64';
export const double: DType = 'float64';
export const float16: DType = 'float16';
export const half: DType = 'float16';
export const bfloat16: DType = 'bfloat16';
export const int8: DType = 'int8';
export const uint8: DType = 'uint8';
export const byte: DType = 'uint8';
export const int16: DType = 'int16';
export const short: DType = 'int16';
export const int32: DType = 'int32';
export const int: DType = 'int32';
export const int64: DType = 'int64';
export const long: DType = 'int64';
export const bool: DType = 'bool';
export const complex64: DType = 'complex64';
export const complex128: DType = 'complex128';
export const float8_e4m3fn: DType = 'float8_e4m3fn';
export const float8_e5m2: DType = 'float8_e5m2';

export function isFloatingPointDType(dtype: DType): boolean {
  return (
    dtype === 'float32' ||
    dtype === 'float64' ||
    dtype === 'float16' ||
    dtype === 'bfloat16' ||
    dtype === 'float8_e4m3fn' ||
    dtype === 'float8_e5m2'
  );
}

export function isIntegerDType(dtype: DType): boolean {
  return (
    dtype === 'int8' ||
    dtype === 'uint8' ||
    dtype === 'int16' ||
    dtype === 'int32' ||
    dtype === 'int64'
  );
}

export function dtypeElementSize(dtype: DType): number {
  switch (dtype) {
    case 'bool':
    case 'int8':
    case 'uint8':
    case 'float8_e4m3fn':
    case 'float8_e5m2':
      return 1;
    case 'int16':
    case 'float16':
    case 'bfloat16':
      return 2;
    case 'int32':
    case 'float32':
      return 4;
    case 'int64':
    case 'float64':
    case 'complex64':
      return 8;
    case 'complex128':
      return 16;
  }
}
