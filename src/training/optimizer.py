# Taken from mlff (https://github.com/thorben-frank/mlff, commit 99dbf76)
# Original author: Thorben Frank et al.
# Modifications: renamed imports (mlff.src → src)

import optax
from optax import constant_schedule
from dataclasses import dataclass

import logging
from typing import (Dict)
from optax import exponential_decay


@dataclass
class Optimizer:
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    eps_root: float = 0.0
    transition_steps: int = None
    decay_rate: float = None
    weight_decay: float = None

    def get(self,
            learning_rate: float,
            *args,
            **kwargs) -> optax.GradientTransformation:
        """
        Get the optax optimizer, for a specified learning rate.

        Args:
            learning_rate (float): The learning rate
            *args ():
            **kwargs ():

        Returns:

        """
        self.learning_rate = learning_rate
        weight_decay = 0. if self.weight_decay is None else self.weight_decay
        mask = None if self.weight_decay is None else flattened_traversal(lambda path, _: path[-1] != 'bias')

        if self.transition_steps is None or self.decay_rate is None:
            step_size_fn = None
        else:
            step_size_fn = exponential_decay(1.,
                                             transition_steps=self.transition_steps,
                                             decay_rate=self.decay_rate
                                             )

        return optimizer(learning_rate=self.learning_rate,
                         b1=self.b1,
                         b2=self.b2,
                         eps=self.eps,
                         eps_root=self.eps_root,
                         weight_decay=weight_decay,
                         mask=mask,
                         step_size_fn=step_size_fn)

    def __dict_repr__(self):
        return {'optimizer': {'learning_rate': self.learning_rate,
                             'transition_steps': self.transition_steps,
                             'decay_rate': self.decay_rate,
                             'weight_decay': self.weight_decay}}

@dataclass
class Optimizer_amsgrad:

    def get(self,
            learning_rate: float,
            *args,
            **kwargs) -> optax.GradientTransformation:
        """
        Get the optax optimizer, for a specified learning rate.

        Args:
            learning_rate (float): The learning rate
            *args ():
            **kwargs ():

        Returns:
        """

        self.learning_rate = learning_rate
        return optimizer_amsgrad(learning_rate=self.learning_rate)

    def __dict_repr__(self):
        return {'optimizer_amsgrad': {'learning_rate': self.learning_rate}}


def optimizer_amsgrad(learning_rate):
    return optax.chain(
        optax.scale_by_amsgrad(),
        optax.scale(-learning_rate),
        optax.scale_by_schedule(optax.constant_schedule(1)),
    )


def optimizer(learning_rate,
              b1: float = 0.9,
              b2: float = 0.999,
              eps: float = 1e-8,
              eps_root: float = 0.0,
              weight_decay: float = 0.0,
              mask=None,
              step_size_fn=None):

    if step_size_fn is None:
        step_size_fn = constant_schedule(1.)

    return optax.chain(
        optax.scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
        optax.add_decayed_weights(weight_decay, mask),
        optax.scale(-learning_rate),
        optax.scale_by_schedule(step_size_fn),
    )



