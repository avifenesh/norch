import { native, type NativeTensor, type Scalar } from './native.js';
import { Tensor, wrapTensor } from './tensor.js';

function callUnary(op: string, x: Tensor): Tensor {
  return wrapTensor(native.callOpAuto(op, [x._native]) as NativeTensor);
}

export function erf(input: Tensor): Tensor {
  return callUnary('special_erf', input);
}

export function erfc(input: Tensor): Tensor {
  return callUnary('special_erfc', input);
}

export function erfcx(input: Tensor): Tensor {
  return callUnary('special_erfcx', input);
}

export function erfinv(input: Tensor): Tensor {
  return callUnary('special_erfinv', input);
}

export function gammaln(input: Tensor): Tensor {
  return callUnary('special_gammaln', input);
}

export function digamma(input: Tensor): Tensor {
  return callUnary('special_digamma', input);
}

export function psi(input: Tensor): Tensor {
  return callUnary('special_psi', input);
}

export function polygamma(n: number, input: Tensor): Tensor {
  return wrapTensor(native.callOpAuto('special_polygamma', [n, input._native]) as NativeTensor);
}

export function logsumexp(input: Tensor, dim: number | number[], keepdim = false): Tensor {
  const dims = Array.isArray(dim) ? dim : [dim];
  return wrapTensor(native.callOpAuto('special_logsumexp', [input._native, dims, keepdim]) as NativeTensor);
}

export function sinc(input: Tensor): Tensor {
  return callUnary('special_sinc', input);
}

export function ndtr(input: Tensor): Tensor {
  return callUnary('special_ndtr', input);
}

export function ndtri(input: Tensor): Tensor {
  return callUnary('special_ndtri', input);
}

export function logNdtr(input: Tensor): Tensor {
  return callUnary('special_log_ndtr', input);
}

export function expit(input: Tensor): Tensor {
  return callUnary('special_expit', input);
}

export function logit(input: Tensor, eps?: number): Tensor {
  return wrapTensor(
    native.callOpAuto('special_logit', eps === undefined ? [input._native] : [input._native, eps]) as NativeTensor
  );
}

export function entr(input: Tensor): Tensor {
  return callUnary('special_entr', input);
}

export function xlogy(input: Tensor | Scalar, other: Tensor | Scalar): Tensor {
  const a = input instanceof Tensor ? input._native : input;
  const b = other instanceof Tensor ? other._native : other;
  return wrapTensor(native.callOpAuto('special_xlogy', [a, b]) as NativeTensor);
}

export function xlog1py(input: Tensor | Scalar, other: Tensor | Scalar): Tensor {
  const a = input instanceof Tensor ? input._native : input;
  const b = other instanceof Tensor ? other._native : other;
  return wrapTensor(native.callOpAuto('special_xlog1py', [a, b]) as NativeTensor);
}

export function i0(input: Tensor): Tensor {
  return callUnary('special_i0', input);
}

export function i1(input: Tensor): Tensor {
  return callUnary('special_i1', input);
}

export function gammainc(input: Tensor, other: Tensor): Tensor {
  return wrapTensor(native.callOpAuto('special_gammainc', [input._native, other._native]) as NativeTensor);
}

export function gammaincc(input: Tensor, other: Tensor): Tensor {
  return wrapTensor(native.callOpAuto('special_gammaincc', [input._native, other._native]) as NativeTensor);
}

export function zeta(input: Tensor | Scalar, other: Tensor | Scalar): Tensor {
  const a = input instanceof Tensor ? input._native : input;
  const b = other instanceof Tensor ? other._native : other;
  return wrapTensor(native.callOpAuto('special_zeta', [a, b]) as NativeTensor);
}

export const special = {
  erf,
  erfc,
  erfcx,
  erfinv,
  gammaln,
  digamma,
  psi,
  polygamma,
  logsumexp,
  sinc,
  ndtr,
  ndtri,
  logNdtr,
  expit,
  logit,
  entr,
  xlogy,
  xlog1py,
  i0,
  i1,
  gammainc,
  gammaincc,
  zeta,
};
