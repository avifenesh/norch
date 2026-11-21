import { noGrad } from '../autograd.js';
import { Module } from '../nn/module.js';
import type { Tensor } from '../tensor.js';

export type AvgFn = (averagedParam: Tensor, modelParam: Tensor, numAveraged: number) => Tensor;

/**
 * Returns an Exponential Moving Average update function for `AveragedModel`.
 */
export function getEmaAvgFn(decay = 0.999): AvgFn {
  return (avgParam, modelParam, _numAveraged) => {
    const scaledAvg = avgParam.mul(decay);
    const out = scaledAvg.add(modelParam, 1 - decay);
    scaledAvg.dispose();
    return out;
  };
}

/**
 * Stochastic Weight Averaging (SWA) / Exponential Moving Average (EMA) wrapper
 * matching `torch.optim.swa_utils.AveragedModel`.
 */
export class AveragedModel<TModule extends Module = Module> extends Module {
  readonly module: TModule;
  readonly avgFn: AvgFn;
  readonly useBuffers: boolean;
  private nAveraged = 0;

  constructor(
    averagedModuleClone: TModule,
    opts?: { avgFn?: AvgFn; useBuffers?: boolean }
  ) {
    super();
    this.module = averagedModuleClone;
    this.module.requiresGrad_(false);
    this.addModule('module', averagedModuleClone);
    this.avgFn =
      opts?.avgFn ??
      ((avgP, modelP, n) => {
        const diff = modelP.sub(avgP);
        const out = avgP.add(diff, 1 / (n + 1));
        diff.dispose();
        return out;
      });
    this.useBuffers = opts?.useBuffers ?? false;
  }

  get numAveraged(): number {
    return this.nAveraged;
  }

  updateParameters(model: TModule): void {
    noGrad(() => {
      const avgParams = this.module.parameters();
      const srcParams = model.parameters();
      if (avgParams.length !== srcParams.length) {
        throw new Error(
          `AveragedModel parameter count mismatch (${avgParams.length} vs ${srcParams.length})`
        );
      }
      for (let i = 0; i < avgParams.length; i++) {
        const dst = avgParams[i]!;
        const src = srcParams[i]!;
        if (this.nAveraged === 0) {
          dst.copy_(src);
        } else {
          const updated = this.avgFn(dst, src, this.nAveraged);
          dst.copy_(updated);
          updated.dispose();
        }
      }
      if (this.useBuffers) {
        const avgBufs = this.module.buffers();
        const srcBufs = model.buffers();
        for (let i = 0; i < Math.min(avgBufs.length, srcBufs.length); i++) {
          const dst = avgBufs[i]!;
          const src = srcBufs[i]!;
          if (this.nAveraged === 0) {
            dst.copy_(src);
          } else {
            const updated = this.avgFn(dst, src, this.nAveraged);
            dst.copy_(updated);
            updated.dispose();
          }
        }
      }
      this.nAveraged += 1;
    });
  }

  forward(...args: unknown[]): unknown {
    return this.module.call(...args);
  }
}

export const swaUtils = {
  AveragedModel,
  getEmaAvgFn,
};
