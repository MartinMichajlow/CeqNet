# Taken from mlff (https://github.com/thorben-frank/mlff, commit 99dbf76)
# Original author: Thorben Frank et al.
# Modifications: renamed imports (mlff.src → src)

import jax.numpy as jnp
import numpy as np
import jax
import logging
import time

from functools import partial
from pprint import pformat
from typing import (Any, Callable, Dict, Sequence, Tuple)
from flax.training.train_state import TrainState
from flax.core.frozen_dict import FrozenDict
from flax.training import checkpoints

logging.basicConfig(level=logging.INFO)

Array = Any
StackNet = Any
LossFn = Callable[[FrozenDict, Dict[str, jnp.ndarray]], jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
DataTupleT = Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]
Derivative = Tuple[str, Tuple[str, str, Callable]]
ObservableFn = Callable[[FrozenDict, Dict[str, Array]], Dict[str, Array]]



#####################################################################################################################
# OLD TRAINING ROUTINE:
#####################################################################################################################

def train_epoch(state: TrainState,
                ds: DataTupleT,
                loss_fn: LossFn,
                bs: int,
                rng) -> Tuple[TrainState, Dict[str, float]]:
    """
    Arbitrary epoch for NN training.
    Args:
        state (TrainState): Flax train state.
        ds (Tuple): Tuple of data, where first entry is input and second the expected output
        loss_fn (Callable): Loss function. Gradient is computed wrt it
        bs (int): Batch size
        rng (RandomKey): JAX PRNGKey
    Returns: Updated optimizer state and the training loss.
    """
    inputs, targets = ds
    _k = list(inputs.keys())
    n_data = len(inputs[_k[0]])

    steps_per_epoch = n_data // bs
    perms = jax.random.permutation(rng, n_data)
    perms = perms[:steps_per_epoch * bs]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, bs))
    batch_metrics = []
    for perm in perms:
        batch = jax.tree.map(lambda y: y[perm, ...], ds)
        # batch = (Dict[str, Array[perm, ...]], Dict[str, Array[perm, ...]])
        state, metrics = train_step_fn(state, batch, loss_fn)
        batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch.
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {k: np.mean([metrics[k] for metrics in batch_metrics_np]) for k in batch_metrics_np[0]}
    return state, epoch_metrics_np


@partial(jax.jit, static_argnums=2)
def train_step_fn(state: TrainState,
                  batch: Dict,
                  loss_fn: Callable) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """
    Training step.
        state (TrainState): Flax train state.
        batch (Tuple): Batch of validation data
        loss_fn (Callable): Loss function
    Returns: Updated optimizer state and loss for current batch.
    """
    (loss, train_metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, batch)
    state = state.apply_gradients(grads=grads)
    return state, train_metrics


def valid_epoch(state: TrainState,
                ds: DataTupleT,
                metric_fn: LossFn,
                bs: int) -> Dict[str, float]:
    """
    Validation epoch for NN training.
    Args:
        state (TrainState): Flax train state.
        ds (Tuple): Validation data. First entry is input, second is expected output
        metric_fn (Callable): Function that evaluates the model wrt some metric fn
        bs (int): Batch size.
    Returns: Validation metric.
    """
    inputs, targets = ds
    _k = list(inputs.keys())
    n_data = len(inputs[_k[0]])

    steps_per_epoch = n_data // bs
    batch_metrics = []
    idxs = jnp.arange(n_data)
    idxs = idxs[:steps_per_epoch * bs]  # skip incomplete batch
    idxs = idxs.reshape((steps_per_epoch, bs))
    for idx in idxs:
        batch = jax.tree.map(lambda y: y[idx, ...], ds)
        # batch = (Dict[str, Array[perm, ...]], Dict[str, Array[perm, ...]])
        metrics = valid_step_fn(state, batch, metric_fn)
        batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch.
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {k: np.mean([metrics[k] for metrics in batch_metrics_np]) for k in batch_metrics_np[0]}
    return epoch_metrics_np


@partial(jax.jit, static_argnums=2)
def valid_step_fn(state: TrainState,
                  batch: DataTupleT,
                  metric_fn: Callable) -> Dict[str, jnp.ndarray]:
    """
    Validation step.
    Args:
        state (TrainState): Flax train state.
        batch (Tuple): Batch of validation data.
        metric_fn (Callable): Function that evaluates the model wrt some metric fn
    Returns: Validation metrics on the batch.
    """
    _, metrics = metric_fn(state.params, batch)
    return metrics


def run_training(state: TrainState,
                 loss_fn: LossFn,
                 train_ds: DataTupleT,
                 valid_ds: DataTupleT,
                 train_bs: int,
                 valid_bs: int,
                 metric_fn: LossFn = None,
                 epochs: int = 100,
                 ckpt_dir: str = None,
                 save_every_t: int = None,
                 eval_every_t: int = 1,
                 print_every_t: int = 1,
                 save_best: Sequence[str] = None,
                 ckpt_overwrite: bool = False,
                 seed: int = 0,
                 use_wandb: bool = True
                 ):
    """
    Run training for a NN. The checkpoints are saved, such that _n corresponds to the case of n optimizer updates. By
    doing so, it can be directly compared to number steps that can be specified for the neural tangent function on
    NNGPs where a time step can be specified. Also, checkpoints saved with _0 at the end, correspond to the model at
    initialization which also might be interesting for our case.
    Args:
        state (TrainState): Flax train state.
        loss_fn (Callable): The loss function. Gradient is computed wrt to this function.
        train_ds (Tuple): Tuple of training data. First entry is input, second is expected output.
        valid_ds (Tuple): Tuple of validation data. First entry is input, second is expected output.
        train_bs (int): Training batch size.
        valid_bs (int): Validation batch size.
        metric_fn (Callable): Dictionary of functions, which are evaluated on the validation set and logged.
        epochs (int): Number of training epochs.
        ckpt_dir (str): Checkpoint path.
        save_every_t (int): Save the model every t-th epoch
        eval_every_t (int): Evaluate the metrics every t-th epoch
        print_every_t (int): Print the training loss every t-th epoch
        save_best (List): Save the model based on evaluation metric. Each entry must one key in the metric_fns
        ckpt_overwrite (bool): Whether overwriting of existing checkpoints is allowed.
        seed (int): Random seed.
    Returns:
    """
    rng = jax.random.PRNGKey(seed)

    if metric_fn is None:
        metric_fn = loss_fn
        logging.info('no metrics functions defined, default to loss function.')

    tot_time = 0
    valid_metrics = {}
    best_valid_metrics = {}
    for i in range(epochs):
        epoch_start = time.time()
        if save_every_t is not None:
            if i % save_every_t == 0:
                checkpoints.save_checkpoint(ckpt_dir,
                                            state,
                                            i,
                                            keep=int(epochs//save_every_t)+1,
                                            prefix='checkpoint_epoch_',
                                            overwrite=ckpt_overwrite)

        rng, input_rng = jax.random.split(rng)
        train_start = time.time()
        state, train_metrics = train_epoch(state, train_ds, loss_fn, train_bs, input_rng)
        train_end = time.time()

        # check for NaN
        train_metrics_np = jax.device_get(train_metrics)

        if (np.isnan(train_metrics_np['loss']) or
            np.isinf(train_metrics_np['loss']) or
            (np.abs(train_metrics_np['loss']) > 2 and i > 5)
        ):

            if np.isnan(train_metrics_np['loss']):
                logging.warning(f'NaN detected during training in step {i} in the loss function value. Reload the '
                                'last checkpoint.')
            elif np.isinf(train_metrics_np['loss']):
                logging.warning(f'Inf detected during training in step {i} in the loss function value. Reload the '
                                'last checkpoint.')
            elif (np.abs(train_metrics_np['loss']) > 2):
                logging.warning(f'Value greater than 2 detected during training in step {i} in the loss function value. Reload the '
                                'last checkpoint.')
            if np.isinf(train_metrics_np['loss']) or np.isnan(train_metrics_np['loss']):
                def reset_records():
                    return jax.tree.map(lambda x: jnp.zeros(x.shape), state.params['record'])

                state_dict = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None, step=None,
                                                            prefix='checkpoint_loss_')
                try:
                    state_dict['params']['record'] = reset_records()
                except KeyError:
                    pass
                state = state.replace(params=FrozenDict(state_dict['params']))
                opt_state = state.tx.init(state.params)
                state.replace(opt_state=opt_state)


        valid_start, valid_end = (0., 0.)
        if i % eval_every_t == 0:
            valid_start = time.time()
            valid_metrics.update(valid_epoch(state, valid_ds, metric_fn, bs=valid_bs))
            valid_end = time.time()
            if i == 0:
                best_valid_metrics.update(valid_metrics)
            # loop over all metrics and compare
            for _k, _v in best_valid_metrics.items():
                if valid_metrics[_k] < _v or i == 0: # save the first checkpoint to debug test error
                    best_valid_metrics[_k] = valid_metrics[_k]
                    checkpoints.save_checkpoint(ckpt_dir,
                                                state,
                                                i,
                                                keep=1,
                                                prefix='checkpoint_{}_'.format(_k),
                                                overwrite=ckpt_overwrite)
        epoch_end = time.time()

        e_time = epoch_end - epoch_start
        t_time = train_end - train_start
        v_time = valid_end - valid_start
        tot_time += e_time
        times = {'Epoch time (s)': e_time,
                 'Training epoch time (s)': t_time,
                 'Validation epoch time (s)': v_time,
                 'Total time (s)': tot_time}

        if i % print_every_t == 0:
            logging.info('Epoch: {}'.format(i))
            logging.info('Times: Epoch: {} s, Training: {} s, Validation: {}'.format(e_time, t_time, v_time))
            logging.info('Training metrics: {}'.format(pformat(train_metrics)))
            logging.info('Evaluation metrics: {}'.format(pformat(valid_metrics)))

        if use_wandb:
            import wandb
            if i > 0:
                wandb.log(times, step=i)
                log_train_metrics = {key + '_train': train_metrics[key] for (key, item) in train_metrics.items()}
                log_valid_metrics = {key + '_valid': valid_metrics[key] for (key, item) in valid_metrics.items()}
                wandb.log(log_train_metrics, step=i)
                wandb.log(log_valid_metrics, step=i)