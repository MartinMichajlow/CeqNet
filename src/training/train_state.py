# Taken from mlff (https://github.com/thorben-frank/mlff, commit 99dbf76)
# Original author: Thorben Frank et al.
# Modifications: renamed imports (mlff.src → src)

from flax.linen import Module
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from optax import GradientTransformation


def create_train_state(module: Module, params: FrozenDict, tx: GradientTransformation) -> TrainState:
    """
    Creates an initial TrainState.

    Args:
        module (Module): A FLAX module.
        params (FrozenDict): A FrozenDict with the model parameters.
        tx (GradientTransformation): An optax GradientTransformation.

    Returns: A FLAX TrainState.

    """
    return TrainState.create(apply_fn=module.apply, params=params, tx=tx)