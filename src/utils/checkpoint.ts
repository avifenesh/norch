import {
  AutogradContext,
  AutogradFunction,
  enableGrad,
  grad,
  noGrad,
} from '../autograd.js';
import type { Tensor } from '../tensor.js';

class CheckpointFunction extends AutogradFunction {
  static forward(
    ctx: AutogradContext,
    runFn: (...inputs: Tensor[]) => Tensor,
    ...inputs: Tensor[]
  ): Tensor {
    ctx['runFn'] = runFn;
    ctx.saveForBackward(...inputs);
    return noGrad(() => runFn(...inputs));
  }

  static backward(ctx: AutogradContext, gradOut: Tensor): (Tensor | null)[] {
    const runFn = ctx['runFn'] as (...inputs: Tensor[]) => Tensor;
    const saved = ctx.savedTensors;
    const detachedInputs = saved.map((t) => {
      const d = t.detach();
      if (t.requiresGrad) d.requiresGrad_(true);
      return d;
    });

    const output = enableGrad(() => runFn(...detachedInputs));
    const reqInputs = detachedInputs.filter((t) => t.requiresGrad);
    const grads =
      reqInputs.length > 0 ? grad([output], reqInputs, [gradOut], { allowUnused: true }) : [];

    let gIdx = 0;
    const result: (Tensor | null)[] = detachedInputs.map((t) =>
      t.requiresGrad ? (grads[gIdx++] ?? null) : null
    );
    return result;
  }
}

/**
 * Activation checkpointing (gradient checkpointing): runs `fn` without saving
 * intermediate activations in forward, and recomputes `fn` during backward.
 */
export function checkpoint(fn: (...inputs: Tensor[]) => Tensor, ...inputs: Tensor[]): Tensor {
  return CheckpointFunction.apply(fn, ...inputs) as Tensor;
}
