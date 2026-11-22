export {
  Optimizer,
  type ParamGroupBase,
  type ParamGroupInput,
  type OptimizerStateDict,
} from './optimizer.js';
export { SGD, type SGDOptions, type SGDParamGroup } from './sgd.js';
export { Adam, AdamW, type AdamOptions, type AdamParamGroup } from './adam.js';
export {
  Muon,
  zeropowerViaNewtonSchulz,
  MUON_DEFAULT_NS_COEFFICIENTS,
  MUON_DEFAULT_NS_STEPS,
  MUON_DEFAULT_EPS,
  type MuonOptions,
  type MuonParamGroup,
  type MuonAdjustLrFn,
  type NewtonSchulzOptions,
} from './muon.js';
export { RMSprop, type RMSpropOptions, type RMSpropParamGroup } from './rmsprop.js';
export {
  Adagrad,
  Adadelta,
  Adamax,
  NAdam,
  RAdam,
  LBFGS,
  type AdagradOptions,
  type AdadeltaOptions,
  type AdamaxOptions,
  type NAdamOptions,
  type RAdamOptions,
  type LBFGSOptions,
} from './extra_optimizers.js';
export { AveragedModel, getEmaAvgFn, swaUtils, type AvgFn } from './swa_utils.js';
export {
  LRScheduler,
  StepLR,
  CosineAnnealingLR,
  LinearLR,
  OneCycleLR,
  SequentialLR,
  ReduceLROnPlateau,
  ExponentialLR,
  MultiStepLR,
  LambdaLR,
  CosineAnnealingWarmRestarts,
  ChainedScheduler,
  type SchedulerStateDict,
  type OneCycleLROptions,
  type ReduceLROnPlateauOptions,
} from './lr_scheduler.js';
