# Partially adapted from mlff (https://github.com/thorben-frank/mlff, commit 99dbf76)
# Original author (mlff): Thorben Frank et al.
# Adapted from mlff: Coach class structure, run method, __dict_repr__
# Original contributions: qtot_loss and qlambda fields

import logging
logging.basicConfig(level=logging.INFO)
import jax.numpy as jnp

from dataclasses import dataclass
from typing import (Any, Callable, Dict, Sequence, Tuple)
from flax.core.frozen_dict import FrozenDict

from .run import run_training


Array = Any
StackNet = Any
LossFn = Callable[[FrozenDict, Dict[str, jnp.ndarray]], jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
DataTupleT = Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]
Derivative = Tuple[str, Tuple[str, str, Callable]]
ObservableFn = Callable[[FrozenDict, Dict[str, Array]], Dict[str, Array]]


@dataclass
class Coach:
    input_keys: Sequence[str]
    target_keys: Sequence[str]
    epochs: int
    training_batch_size: int
    validation_batch_size: int
    loss_weights: Dict[str, float]
    ckpt_dir: str
    data_path: str
    net_seed: int = 0
    training_seed: int = 0
    qtot_loss: bool = False  # [Original contribution]
    qlambda: float = None    # [Original contribution]

    def run(self, train_state, train_ds, valid_ds, loss_fn, metric_fn=None, use_wandb: bool = True, **kwargs):
        logging.info('started coach.run()')
        run_training(state=train_state,
                     loss_fn=loss_fn,
                     metric_fn=metric_fn,
                     train_ds=train_ds,
                     valid_ds=valid_ds,
                     epochs=self.epochs,
                     train_bs=self.training_batch_size,
                     valid_bs=self.validation_batch_size,
                     ckpt_dir=self.ckpt_dir,
                     seed=self.training_seed,
                     use_wandb=use_wandb,
                     **kwargs
                     )

    def __dict_repr__(self):
        return {'coach': {'input_keys': self.input_keys,
                          'target_keys': self.target_keys,
                          'epochs': self.epochs,
                          'training_batch_size': self.training_batch_size,
                          'validation_batch_size': self.validation_batch_size,
                          'loss_weights': self.loss_weights,
                          'ckpt_dir': self.ckpt_dir,
                          'data_path': self.data_path,
                          'training_seed': self.training_seed,
                          'net_seed': self.net_seed,
                          'qtot_loss': self.qtot_loss}}
