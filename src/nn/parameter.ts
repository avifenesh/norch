import { untrackTensor } from '../memory.js';
import { Tensor } from '../tensor.js';

/**
 * A Tensor marked as a learnable Module parameter (requiresGrad=true by default).
 * Parameters are automatically excluded from enclosing tidy()/scope() auto-disposal.
 */
export class Parameter extends Tensor {
  constructor(data: Tensor, requiresGrad = true) {
    const det = data._native.detach();
    const cloned = det.clone();
    det.dispose();
    cloned.setRequiresGrad(requiresGrad);
    super(cloned);
    // Parameters live with their Module, not transient tidy() scopes
    untrackTensor(this);
  }
}
