import { native, type NativeTensor } from './native.js';
import { Tensor, wrapTensor } from './tensor.js';

export type FftNorm = 'forward' | 'backward' | 'ortho';

export const fft = {
  fft(input: Tensor, n?: number | null, dim = -1, norm?: FftNorm | null): Tensor {
    return wrapTensor(
      native.callOp('aten::fft_fft', '', [input._native, n ?? null, dim, norm ?? null]) as NativeTensor
    );
  },

  ifft(input: Tensor, n?: number | null, dim = -1, norm?: FftNorm | null): Tensor {
    return wrapTensor(
      native.callOp('aten::fft_ifft', '', [input._native, n ?? null, dim, norm ?? null]) as NativeTensor
    );
  },

  rfft(input: Tensor, n?: number | null, dim = -1, norm?: FftNorm | null): Tensor {
    return wrapTensor(
      native.callOp('aten::fft_rfft', '', [input._native, n ?? null, dim, norm ?? null]) as NativeTensor
    );
  },

  irfft(input: Tensor, n?: number | null, dim = -1, norm?: FftNorm | null): Tensor {
    return wrapTensor(
      native.callOp('aten::fft_irfft', '', [input._native, n ?? null, dim, norm ?? null]) as NativeTensor
    );
  },

  fft2(
    input: Tensor,
    s?: number[] | null,
    dim: [number, number] = [-2, -1],
    norm?: FftNorm | null
  ): Tensor {
    return wrapTensor(
      native.callOp('aten::fft_fft2', '', [input._native, s ?? null, dim, norm ?? null]) as NativeTensor
    );
  },

  ifft2(
    input: Tensor,
    s?: number[] | null,
    dim: [number, number] = [-2, -1],
    norm?: FftNorm | null
  ): Tensor {
    return wrapTensor(
      native.callOp('aten::fft_ifft2', '', [input._native, s ?? null, dim, norm ?? null]) as NativeTensor
    );
  },

  rfft2(
    input: Tensor,
    s?: number[] | null,
    dim: [number, number] = [-2, -1],
    norm?: FftNorm | null
  ): Tensor {
    return wrapTensor(
      native.callOp('aten::fft_rfft2', '', [input._native, s ?? null, dim, norm ?? null]) as NativeTensor
    );
  },

  irfft2(
    input: Tensor,
    s?: number[] | null,
    dim: [number, number] = [-2, -1],
    norm?: FftNorm | null
  ): Tensor {
    return wrapTensor(
      native.callOp('aten::fft_irfft2', '', [input._native, s ?? null, dim, norm ?? null]) as NativeTensor
    );
  },

  fftn(input: Tensor, s?: number[] | null, dim?: number[] | null, norm?: FftNorm | null): Tensor {
    return wrapTensor(
      native.callOp('aten::fft_fftn', '', [input._native, s ?? null, dim ?? null, norm ?? null]) as NativeTensor
    );
  },

  ifftn(input: Tensor, s?: number[] | null, dim?: number[] | null, norm?: FftNorm | null): Tensor {
    return wrapTensor(
      native.callOp('aten::fft_ifftn', '', [input._native, s ?? null, dim ?? null, norm ?? null]) as NativeTensor
    );
  },

  fftshift(input: Tensor, dim?: number[] | null): Tensor {
    return wrapTensor(
      native.callOp('aten::fft_fftshift', '', [input._native, dim ?? null]) as NativeTensor
    );
  },

  ifftshift(input: Tensor, dim?: number[] | null): Tensor {
    return wrapTensor(
      native.callOp('aten::fft_ifftshift', '', [input._native, dim ?? null]) as NativeTensor
    );
  },
};
