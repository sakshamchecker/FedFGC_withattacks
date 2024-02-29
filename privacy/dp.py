from typing import Tuple

import numpy as np
from flwr.common.typing import NDArrays


# Calculates the L2-norm of a potentially ragged array
def _get_update_norm(update: NDArrays) -> float:
    flattened_update = update[0]
    for i in range(1, len(update)):
        flattened_update = np.append(flattened_update, update[i])
    return float(np.sqrt(np.sum(np.square(flattened_update))))


def add_gaussian_noise(update: NDArrays, std_dev: float) -> NDArrays:
    """Add iid Gaussian noise to each floating point value in the update."""
    update_noised = [
        layer + np.random.normal(0, std_dev, layer.shape) for layer in update
    ]
    return update_noised


def clip_by_l2(update: NDArrays, threshold: float) -> Tuple[NDArrays, bool]:
    """Scales the update so thats its L2 norm is upper-bound to threshold."""
    update_norm = _get_update_norm(update)
    scaling_factor = min(1, threshold / update_norm)
    update_clipped: NDArrays = [layer * scaling_factor for layer in update]
    return update_clipped, (scaling_factor < 1)

def dp(original_params, updated_params, noise_multiplier, max_grad_norm):
    update = [np.subtract(x, y) for (x, y) in zip(updated_params, original_params)]
    # update, clipped = clip_by_l2(update, 0.2)
    # update = add_gaussian_noise(update, 0.1)
    update, clipped = clip_by_l2(update, max_grad_norm)
    update = add_gaussian_noise(update, noise_multiplier)
    for i, _ in enumerate(original_params):
        updated_params[i] = original_params[i] + update[i]
    return updated_params