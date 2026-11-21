import { native, type NativeTensor, type Scalar } from './native.js';
import { Tensor, wrapTensor } from './tensor.js';

export const linalg = {
  norm(a: Tensor, ord?: Scalar | string | null, dim?: number | number[] | null, keepdim = false): Tensor {
    if (typeof ord === 'string') {
      const dims = Array.isArray(dim) ? dim : dim !== undefined && dim !== null ? [dim] : [-2, -1];
      return wrapTensor(
        native.callOp('aten::linalg_matrix_norm', 'str_ord', [a._native, ord, dims, keepdim]) as NativeTensor
      );
    }
    return a.norm(ord ?? 2, dim ?? undefined, keepdim);
  },

  vectorNorm(a: Tensor, ord: Scalar = 2, dim?: number | number[] | null, keepdim = false): Tensor {
    return a.norm(ord, dim ?? undefined, keepdim);
  },

  matrixNorm(a: Tensor, ord: string | number = 'fro', dim: [number, number] = [-2, -1], keepdim = false): Tensor {
    if (typeof ord === 'string') {
      return wrapTensor(
        native.callOp('aten::linalg_matrix_norm', 'str_ord', [a._native, ord, dim, keepdim]) as NativeTensor
      );
    }
    return wrapTensor(
      native.callOp('aten::linalg_matrix_norm', '', [a._native, ord, dim, keepdim]) as NativeTensor
    );
  },

  qr(a: Tensor, mode: 'reduced' | 'complete' | 'r' = 'reduced'): { Q: Tensor; R: Tensor } {
    const [q, r] = native.callOp('aten::linalg_qr', '', [a._native, mode]) as [
      NativeTensor,
      NativeTensor,
    ];
    return { Q: wrapTensor(q), R: wrapTensor(r) };
  },

  svd(
    a: Tensor,
    fullMatrices = true
  ): { U: Tensor; S: Tensor; Vh: Tensor } {
    const [u, s, vh] = native.callOp('aten::linalg_svd', '', [a._native, fullMatrices]) as [
      NativeTensor,
      NativeTensor,
      NativeTensor,
    ];
    return { U: wrapTensor(u), S: wrapTensor(s), Vh: wrapTensor(vh) };
  },

  eigh(a: Tensor, UPLO: 'L' | 'U' = 'L'): { eigenvalues: Tensor; eigenvectors: Tensor } {
    const [vals, vecs] = native.callOp('aten::linalg_eigh', '', [a._native, UPLO]) as [
      NativeTensor,
      NativeTensor,
    ];
    return { eigenvalues: wrapTensor(vals), eigenvectors: wrapTensor(vecs) };
  },

  eigvalsh(a: Tensor, UPLO: 'L' | 'U' = 'L'): Tensor {
    return wrapTensor(native.callOp('aten::linalg_eigvalsh', '', [a._native, UPLO]) as NativeTensor);
  },

  cholesky(a: Tensor, upper = false): Tensor {
    return wrapTensor(
      native.callOp('aten::linalg_cholesky', '', [a._native], { upper }) as NativeTensor
    );
  },

  solve(a: Tensor, b: Tensor, left = true): Tensor {
    return wrapTensor(
      native.callOp('aten::linalg_solve', '', [a._native, b._native], { left }) as NativeTensor
    );
  },

  inv(a: Tensor): Tensor {
    return wrapTensor(native.callOp('aten::linalg_inv', '', [a._native]) as NativeTensor);
  },

  pinv(a: Tensor, rcond = 1e-15, hermitian = false): Tensor {
    return wrapTensor(native.callOp('aten::linalg_pinv', '', [a._native, rcond, hermitian]) as NativeTensor);
  },

  det(a: Tensor): Tensor {
    return wrapTensor(native.callOp('aten::linalg_det', '', [a._native]) as NativeTensor);
  },

  slogdet(a: Tensor): { sign: Tensor; logabsdet: Tensor } {
    const [sign, logabsdet] = native.callOp('aten::linalg_slogdet', '', [a._native]) as [
      NativeTensor,
      NativeTensor,
    ];
    return { sign: wrapTensor(sign), logabsdet: wrapTensor(logabsdet) };
  },

  lstsq(
    a: Tensor,
    b: Tensor,
    rcond?: number | null,
    driver?: string | null
  ): { solution: Tensor; residuals: Tensor; rank: Tensor; singularValues: Tensor } {
    const [sol, res, rank, sv] = native.callOp(
      'aten::linalg_lstsq',
      '',
      [a._native, b._native, rcond ?? null],
      { driver: driver ?? null }
    ) as [NativeTensor, NativeTensor, NativeTensor, NativeTensor];
    return {
      solution: wrapTensor(sol),
      residuals: wrapTensor(res),
      rank: wrapTensor(rank),
      singularValues: wrapTensor(sv),
    };
  },

  multiDot(tensors: Tensor[]): Tensor {
    return wrapTensor(
      native.callOp('aten::linalg_multi_dot', '', [tensors.map((t) => t._native)]) as NativeTensor
    );
  },
};
