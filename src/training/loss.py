# Partially adapted from mlff (https://github.com/thorben-frank/mlff, commit 99dbf76)
# Original author (mlff): Thorben Frank et al.
# Adapted from mlff: mse_loss, energy_loss, force_loss
# Original contributions: get_loss_fn (rewritten to support CeqNet observables and charge conservation loss)

import jax.numpy as jnp
from typing import (Dict, Tuple)

DataTupleT = Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]


def mse_loss(*, y, y_true): return jnp.mean((y_true - y)**2)


def energy_loss(*, y, y_true):
    """

    Args:
        y ():
        y_true ():

    Returns:

    """
    return mse_loss(y=y, y_true=y_true)


def force_loss(*, y, y_true):
    """
    Force loss that takes differently sized molecules across a single batch into account.

    Args:
        y_pred (): predicted forces of shape [...,n,3]
        y_true (): expected forces of shape [...,n,3]

    Returns:

    """

    # number of force components per batch dimension
    n_fc = (y_true != 0).sum(-1).sum(-1)  # shape: (...)
    return jnp.mean(1/n_fc*((y - y_true)**2).sum(-1).sum(-1))


# [Original contribution]
def get_loss_fn(obs_fn, weights, qtot: bool=False, qlambda:float=0.0):
    """
    Returns loss function for given ceqnet. If qtot=False, loss_fn gets only generated as weighted mse sum of loss
    observables. Otherwise, we incorporate loss term for Qtot.

    :param obs_fn:
    :param weights:
    :param qtot:
    :return:
    """

    if qtot == False:
        def loss_fn(params, batch: DataTupleT):
            inputs, targets = batch
            outputs = obs_fn(params, inputs)
            loss = jnp.zeros(1)
            train_metrics = {}
            for name, target in targets.items():
                _l = mse_loss(y_true=outputs[name], y=targets[name])
                loss += weights[name] * _l
                train_metrics.update({name: _l})
            loss = jnp.reshape(loss, ())
            train_metrics.update({'loss': loss})

            return loss, train_metrics
        return loss_fn

    if qtot == True:
        def loss_fn(params, batch: DataTupleT):
            inputs, targets = batch
            outputs = obs_fn(params, inputs)
            loss = jnp.zeros(1)
            train_metrics = {}
            for name, target in targets.items():
                _l = mse_loss(y_true=outputs[name], y=targets[name])
                loss += weights[name] * _l
                train_metrics.update({name: _l})

            qtot_pred = jnp.squeeze(outputs['q']).sum(axis=-1)
            qtot = inputs['Q']
            eps = 1e-10
            loss += qlambda * jnp.sum(jnp.abs(qtot_pred - qtot)/(qtot+eps))
            loss = jnp.reshape(loss, ())
            train_metrics.update({'loss': loss})

            return loss, train_metrics

        return loss_fn