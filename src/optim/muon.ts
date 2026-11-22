import { noGrad } from '../autograd.js';
import { tidy } from '../memory.js';
import { native, type NativeTensor } from '../native.js';
import { zerosLike } from '../ops.js';
import { Tensor, wrapTensor } from '../tensor.js';
import { Optimizer, type ParamGroupBase, type ParamGroupInput } from './optimizer.js';

export const MUON_DEFAULT_NS_COEFFICIENTS: readonly [number, number, number] = [
  3.4445, -4.775, 2.0315,
];
export const MUON_DEFAULT_NS_STEPS = 5;
export const MUON_DEFAULT_EPS = 1e-7;

export type MuonAdjustLrFn = 'original' | 'match_rms_adamw' | 'spectral_unclamped';

export interface MuonOptions {
  /** Learning rate (default: 1e-3). */
  lr?: number;
  /** Decoupled weight decay coefficient (default: 0.1). */
  weightDecay?: number;
  /** Momentum factor (default: 0.95). */
  momentum?: number;
  /** Enable Nesterov momentum (default: true). */
  nesterov?: boolean;
  /** Quintic Newton-Schulz polynomial coefficients `(a, b, c)` (default: `[3.4445, -4.775, 2.0315]`). */
  nsCoefficients?: [number, number, number];
  /** Numerical stability epsilon for spectral norm scaling (default: 1e-7). */
  eps?: number;
  /** Number of Newton-Schulz orthogonalization iterations (default: 5). */
  nsSteps?: number;
  /**
   * Shape-based LR adjustment strategy:
   * - `'original'` (default when `null`/omitted): `sqrt(max(1, rows / cols))` (Keller Jordan)
   * - `'match_rms_adamw'`: `0.2 * sqrt(max(rows, cols))` (Moonshot / Kimi)
   * - `'spectral_unclamped'`: `sqrt(rows / cols)` (Jeremy Bernstein)
   */
  adjustLrFn?: MuonAdjustLrFn | null;
}

export interface MuonParamGroup extends ParamGroupBase {
  weightDecay: number;
  momentum: number;
  nesterov: boolean;
  nsCoefficients: [number, number, number];
  eps: number;
  nsSteps: number;
  adjustLrFn: MuonAdjustLrFn | null;
}

export interface NewtonSchulzOptions {
  nsCoefficients?: [number, number, number];
  nsSteps?: number;
  eps?: number;
}

/**
 * Computes the approximate orthogonalization (zeroth power / polar factor) of a 2D matrix
 * via quintic Newton-Schulz iterations in bfloat16 (matching `torch.optim._muon._zeropower_via_newtonschulz`).
 */
export function zeropowerViaNewtonSchulz(
  grad: Tensor,
  opts: NewtonSchulzOptions = {}
): Tensor {
  const [a, b, c] = opts.nsCoefficients ?? MUON_DEFAULT_NS_COEFFICIENTS;
  const steps = opts.nsSteps ?? MUON_DEFAULT_NS_STEPS;
  const eps = opts.eps ?? MUON_DEFAULT_EPS;
  return wrapTensor(
    native.fast.zeropowerViaNewtonSchulz(grad._native, a, b, c, steps, eps)
  );
}

const VALID_ADJUST_LR_FNS = new Set<string>([
  'original',
  'match_rms_adamw',
  'spectral_unclamped',
]);

/**
 * Muon (MomentUm Orthogonalized by Newton-Schulz) optimizer for 2D hidden-layer weight matrices,
 * matching PyTorch's `torch.optim.Muon` with a single-call C++ ATen fast path.
 *
 * Pair with `optim.AdamW` for 1D parameters (biases, norms) and embeddings/LM heads.
 */
export class Muon extends Optimizer<MuonParamGroup> {
  constructor(params: ParamGroupInput<MuonParamGroup>, opts: MuonOptions = {}) {
    const lr = opts.lr ?? 1e-3;
    const weightDecay = opts.weightDecay ?? 0.1;
    const momentum = opts.momentum ?? 0.95;
    const nesterov = opts.nesterov ?? true;
    const nsCoefficients: [number, number, number] = opts.nsCoefficients
      ? [...opts.nsCoefficients]
      : [...MUON_DEFAULT_NS_COEFFICIENTS];
    const eps = opts.eps ?? MUON_DEFAULT_EPS;
    const nsSteps = opts.nsSteps ?? MUON_DEFAULT_NS_STEPS;
    const adjustLrFn = opts.adjustLrFn ?? null;

    if (lr < 0) throw new RangeError(`Invalid learning rate: ${lr}`);
    if (momentum < 0) throw new RangeError(`Invalid momentum: ${momentum}`);
    if (weightDecay < 0) throw new RangeError(`Invalid weightDecay: ${weightDecay}`);
    if (nsCoefficients.length !== 3) {
      throw new RangeError('nsCoefficients must contain exactly 3 numbers');
    }
    if (nsSteps < 0 || nsSteps >= 100) {
      throw new RangeError(`nsSteps must be in [0, 99], got ${nsSteps}`);
    }
    if (adjustLrFn !== null && !VALID_ADJUST_LR_FNS.has(adjustLrFn)) {
      throw new RangeError(`Unsupported adjustLrFn: ${String(adjustLrFn)}`);
    }

    super(params, {
      lr,
      weightDecay,
      momentum,
      nesterov,
      nsCoefficients,
      eps,
      nsSteps,
      adjustLrFn,
    });

    for (const group of this.paramGroups) {
      for (const p of group.params) {
        if (p.dim !== 2) {
          throw new RangeError(
            `Muon only supports 2D parameters, got shape [${p.shape.join(', ')}]`
          );
        }
      }
    }
  }

  step(closure?: () => Tensor): Tensor | undefined {
    const loss = closure ? closure() : undefined;
    noGrad(() => {
      tidy(() => {
        for (const group of this.paramGroups) {
          const { lr, weightDecay, momentum, nesterov, nsCoefficients, eps, nsSteps, adjustLrFn } =
            group;
          const nativeParams: NativeTensor[] = [];
          const nativeGrads: NativeTensor[] = [];
          const nativeBufs: NativeTensor[] = [];

          for (const p of group.params) {
            const g = p.grad;
            if (!g) continue;
            const st = this.getState(p);
            let buf = st['momentumBuffer'] as Tensor | undefined;
            if (!buf) {
              buf = this.keepStateTensor(zerosLike(g));
              st['momentumBuffer'] = buf;
            }
            nativeParams.push(p._native);
            nativeGrads.push(g._native);
            nativeBufs.push(buf._native);
          }

          if (nativeParams.length === 0) continue;

          native.fast.muonStep(
            nativeParams,
            nativeGrads,
            nativeBufs,
            lr,
            weightDecay,
            momentum,
            nesterov,
            nsCoefficients[0],
            nsCoefficients[1],
            nsCoefficients[2],
            nsSteps,
            eps,
            adjustLrFn ?? 'original'
          );
        }
      });
    });
    return loss;
  }
}
