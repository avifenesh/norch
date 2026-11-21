import { noGrad } from '../autograd.js';
import { tidy } from '../memory.js';
import { native, type NativeTensor } from '../native.js';
import { randn } from '../ops.js';
import { wrapTensor, type Tensor } from '../tensor.js';

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

/**
 * Truncated normal initialization using inverse-CDF sampling (matches torch.nn.init.trunc_normal_).
 */
export function truncNormal_(tensor: Tensor, mean = 0, std = 1, a = -2, b = 2): Tensor {
  return noGrad(() =>
    tidy(() => {
      const normCdf = (x: number): number => {
        // Abramowitz & Stegun 7.1.26 erf approximation (max err 1.5e-7)
        const z = x / Math.SQRT2;
        const sign = z < 0 ? -1 : 1;
        const ax = Math.abs(z);
        const t = 1.0 / (1.0 + 0.3275911 * ax);
        const y =
          1.0 -
          (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
            0.254829592) *
            t) *
            Math.exp(-ax * ax);
        return 0.5 * (1.0 + sign * y);
      };
      const l = normCdf((a - mean) / std);
      const u = normCdf((b - mean) / std);
      tensor.uniform_(2 * l - 1, 2 * u - 1);
      const erfinv = wrapTensor(native.callOp('aten::erfinv', '', [tensor._native]) as NativeTensor);
      tensor.copy_(erfinv.mul(std * Math.SQRT2).add(mean).clamp(a, b));
      return tensor;
    })
  );
}

/**
 * Fills tensor with a (semi-)orthogonal matrix (matches torch.nn.init.orthogonal_).
 */
export function orthogonal_(tensor: Tensor, gain = 1.0): Tensor {
  if (tensor.ndim < 2) {
    throw new Error('Only tensors with 2 or more dimensions are supported for orthogonal_');
  }
  return noGrad(() =>
    tidy(() => {
      const rows = tensor.size(0);
      const cols = tensor.numel / rows;
      const flat = randn([Math.max(rows, cols), Math.min(rows, cols)], {
        dtype: tensor.dtype,
        device: tensor.device,
      });
      const [qRaw, rRaw] = native.callOp('aten::linalg_qr', '', [flat._native, 'reduced']) as [
        NativeTensor,
        NativeTensor,
      ];
      let q = wrapTensor(qRaw);
      const r = wrapTensor(rRaw);
      const d = wrapTensor(native.callOp('aten::diagonal', '', [r._native, 0, -2, -1]) as NativeTensor);
      const ph = wrapTensor(native.callOp('aten::sign', '', [d._native]) as NativeTensor);
      q = q.mul(ph);
      if (rows < cols) {
        q = q.T.contiguous();
      }
      tensor.copy_(q.reshape(tensor.shape).mul(gain));
      return tensor;
    })
  );
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
  truncNormal_,
  orthogonal_,
};
