import { noGrad } from '../autograd.js';
import type { Tensor } from '../tensor.js';

export type Nonlinearity = 'linear' | 'conv1d' | 'conv2d' | 'sigmoid' | 'tanh' | 'relu' | 'leaky_relu';

export function calculateGain(nonlinearity: Nonlinearity, param = 0.01): number {
  switch (nonlinearity) {
    case 'linear':
    case 'conv1d':
    case 'conv2d':
    case 'sigmoid':
      return 1.0;
    case 'tanh':
      return 5.0 / 3.0;
    case 'relu':
      return Math.SQRT2;
    case 'leaky_relu':
      return Math.sqrt(2.0 / (1 + param * param));
  }
}

export function calculateFanInAndFanOut(tensor: Tensor): [number, number] {
  const shape = tensor.shape;
  if (shape.length < 2) {
    throw new Error(`Fan in and fan out cannot be computed for tensor with fewer than 2 dims (got ${shape.length})`);
  }
  const numInputFmaps = shape[1]!;
  const numOutputFmaps = shape[0]!;
  let receptiveFieldSize = 1;
  for (let i = 2; i < shape.length; i++) {
    receptiveFieldSize *= shape[i]!;
  }
  return [numInputFmaps * receptiveFieldSize, numOutputFmaps * receptiveFieldSize];
}

export function uniform_(tensor: Tensor, a = 0, b = 1): Tensor {
  return noGrad(() => tensor.uniform_(a, b));
}

export function normal_(tensor: Tensor, mean = 0, std = 1): Tensor {
  return noGrad(() => tensor.normal_(mean, std));
}

export function constant_(tensor: Tensor, val: number): Tensor {
  return noGrad(() => tensor.fill_(val));
}

export function zeros_(tensor: Tensor): Tensor {
  return noGrad(() => tensor.zero_());
}

export function ones_(tensor: Tensor): Tensor {
  return noGrad(() => tensor.fill_(1));
}

export function xavierUniform_(tensor: Tensor, gain = 1.0): Tensor {
  const [fanIn, fanOut] = calculateFanInAndFanOut(tensor);
  const std = gain * Math.sqrt(2.0 / (fanIn + fanOut));
  const bound = Math.sqrt(3.0) * std;
  return uniform_(tensor, -bound, bound);
}

export function xavierNormal_(tensor: Tensor, gain = 1.0): Tensor {
  const [fanIn, fanOut] = calculateFanInAndFanOut(tensor);
  const std = gain * Math.sqrt(2.0 / (fanIn + fanOut));
  return normal_(tensor, 0, std);
}

export function kaimingUniform_(
  tensor: Tensor,
  a = 0,
  mode: 'fan_in' | 'fan_out' = 'fan_in',
  nonlinearity: Nonlinearity = 'leaky_relu'
): Tensor {
  const [fanIn, fanOut] = calculateFanInAndFanOut(tensor);
  const fan = mode === 'fan_in' ? fanIn : fanOut;
  const gain = calculateGain(nonlinearity, a);
  const std = gain / Math.sqrt(fan);
  const bound = Math.sqrt(3.0) * std;
  return uniform_(tensor, -bound, bound);
}

export function kaimingNormal_(
  tensor: Tensor,
  a = 0,
  mode: 'fan_in' | 'fan_out' = 'fan_in',
  nonlinearity: Nonlinearity = 'leaky_relu'
): Tensor {
  const [fanIn, fanOut] = calculateFanInAndFanOut(tensor);
  const fan = mode === 'fan_in' ? fanIn : fanOut;
  const gain = calculateGain(nonlinearity, a);
  const std = gain / Math.sqrt(fan);
  return normal_(tensor, 0, std);
}

export const init = {
  calculateGain,
  calculateFanInAndFanOut,
  uniform_,
  normal_,
  constant_,
  zeros_,
  ones_,
  xavierUniform_,
  xavierNormal_,
  kaimingUniform_,
  kaimingNormal_,
};
