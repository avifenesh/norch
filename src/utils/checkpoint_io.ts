import type { DeviceLike } from '../device.js';
import type { Module } from '../nn/module.js';
import type { LRScheduler, Optimizer, OptimizerStateDict, ReduceLROnPlateau } from '../optim/index.js';
import { Tensor } from '../tensor.js';
import { loadSafetensors, saveSafetensors } from './safetensors.js';

export interface SaveCheckpointOptions {
  model: Module;
  optimizer?: Optimizer;
  scheduler?: LRScheduler | ReduceLROnPlateau;
  step?: number;
  epoch?: number;
  extra?: Record<string, unknown>;
}

export interface LoadCheckpointOptions {
  model: Module;
  optimizer?: Optimizer;
  scheduler?: LRScheduler | ReduceLROnPlateau;
  device?: DeviceLike;
  strict?: boolean;
}

export interface LoadedCheckpointMeta {
  step?: number;
  epoch?: number;
  extra?: Record<string, unknown>;
}

const MODEL_PREFIX = 'model.';
const OPT_PREFIX = 'optim.';

/**
 * Saves model weights, optimizer state tensors/hyperparams, scheduler state,
 * and step/epoch metadata into a single binary `.safetensors` checkpoint file.
 */
export function saveCheckpoint(filePath: string, opts: SaveCheckpointOptions): void {
  const packedTensors: Record<string, Tensor> = {};
  const ownedToDispose: Tensor[] = [];

  try {
    const modelSd = opts.model.stateDict();
    for (const [k, t] of Object.entries(modelSd)) {
      packedTensors[`${MODEL_PREFIX}${k}`] = t;
      ownedToDispose.push(t);
    }

    const metaJson: Record<string, unknown> = {
      format: 'norch.checkpoint.v1',
    };
    if (opts.step !== undefined) metaJson['step'] = opts.step;
    if (opts.epoch !== undefined) metaJson['epoch'] = opts.epoch;
    if (opts.extra !== undefined) metaJson['extra'] = opts.extra;

    if (opts.optimizer) {
      const optSd = opts.optimizer.stateDict();
      const scalarState: Record<number, Record<string, number>> = {};
      const tensorStateKeys: Record<number, Record<string, string>> = {};

      for (const [rawId, entry] of Object.entries(optSd.state)) {
        const id = Number(rawId);
        for (const [field, val] of Object.entries(entry)) {
          if (val instanceof Tensor) {
            const key = `${OPT_PREFIX}${id}.${field}`;
            packedTensors[key] = val;
            ownedToDispose.push(val);
            if (!tensorStateKeys[id]) tensorStateKeys[id] = {};
            tensorStateKeys[id]![field] = key;
          } else {
            if (!scalarState[id]) scalarState[id] = {};
            scalarState[id]![field] = val;
          }
        }
      }
      metaJson['optimizer'] = {
        paramGroups: optSd.paramGroups,
        scalarState,
        tensorStateKeys,
      };
    }

    if (opts.scheduler) {
      metaJson['scheduler'] = opts.scheduler.stateDict();
    }

    saveSafetensors(filePath, packedTensors, {
      norch_checkpoint: JSON.stringify(metaJson),
    });
  } finally {
    for (const t of ownedToDispose) {
      if (!t.isDisposed) t.dispose();
    }
  }
}

/**
 * Loads a checkpoint created by `saveCheckpoint()`, restoring model weights,
 * optimizer moments/hyperparams, and scheduler state cleanly.
 */
export function loadCheckpoint(filePath: string, opts: LoadCheckpointOptions): LoadedCheckpointMeta {
  const { tensors, metadata } = loadSafetensors(filePath, { device: opts.device });
  try {
    const modelSd: Record<string, Tensor> = {};
    for (const [k, t] of Object.entries(tensors)) {
      if (k.startsWith(MODEL_PREFIX)) {
        modelSd[k.slice(MODEL_PREFIX.length)] = t;
      }
    }
    opts.model.loadStateDict(modelSd, opts.strict ?? true);

    const rawMeta = metadata['norch_checkpoint'];
    const parsed = rawMeta ? (JSON.parse(rawMeta) as Record<string, unknown>) : {};

    if (opts.optimizer && parsed['optimizer']) {
      const optMeta = parsed['optimizer'] as {
        paramGroups: OptimizerStateDict['paramGroups'];
        scalarState: Record<number, Record<string, number>>;
        tensorStateKeys: Record<number, Record<string, string>>;
      };
      const restoredState: OptimizerStateDict['state'] = {};
      for (const [rawId, scalars] of Object.entries(optMeta.scalarState ?? {})) {
        const id = Number(rawId);
        restoredState[id] = { ...(restoredState[id] ?? {}), ...scalars };
      }
      for (const [rawId, keyMap] of Object.entries(optMeta.tensorStateKeys ?? {})) {
        const id = Number(rawId);
        if (!restoredState[id]) restoredState[id] = {};
        for (const [field, key] of Object.entries(keyMap)) {
          const t = tensors[key];
          if (t) restoredState[id]![field] = t;
        }
      }
      opts.optimizer.loadStateDict({
        paramGroups: optMeta.paramGroups,
        state: restoredState,
      });
    }

    if (opts.scheduler && parsed['scheduler']) {
      opts.scheduler.loadStateDict(parsed['scheduler'] as never);
    }

    return {
      step: typeof parsed['step'] === 'number' ? parsed['step'] : undefined,
      epoch: typeof parsed['epoch'] === 'number' ? parsed['epoch'] : undefined,
      extra: parsed['extra'] as Record<string, unknown> | undefined,
    };
  } finally {
    for (const t of Object.values(tensors)) {
      if (!t.isDisposed) t.dispose();
    }
  }
}
