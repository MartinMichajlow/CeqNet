"""
CeqNet training example on ANI-1x.

Trains CeqNet to predict dipole or quadrupole moments from the ANI-1x dataset.
Hyperparameters are read from examples/config.yaml via --run_idx.

Run from the repository root:
    python examples/train.py --run_idx 5      # pick sweep config 5 from config.yaml
"""

import argparse
import os
import sys
import logging

import jax
import jax.numpy as jnp
import numpy as np

# Allow imports from the repo root regardless of install state.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.training import Coach, get_loss_fn, create_train_state, Optimizer
from src.indexing import get_indices
from src.data import DataTuple
from src.nn.ceqnet import CeqNet, get_obs_fn
from src.nn.embed import AtomTypeEmbed, ChargeEmbed, CeqEmbed, GeometryEmbed
from src.nn.layer import So3kratesLayer
from src.nn.observable import Dipole, Quadrupole
from src.io import bundle_dicts, save_json

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--run_idx', type=int, required=True,
                    help='1-indexed sweep config from config.yaml (for SLURM array jobs)')
args = parser.parse_args()

HERE = os.path.dirname(__file__)

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
import yaml
with open(os.path.join(HERE, 'config.yaml')) as f:
    cfg = yaml.safe_load(f)

sweep = cfg['sweep'][args.run_idx - 1]
model_cfg = cfg['model']
train_cfg = cfg['training']
data_cfg = cfg['data']
out_cfg = cfg['output']

obs_type   = sweep['obs_type']
obs_mode   = sweep['obs_mode']
obs_alpha  = sweep.get('alpha', None)
r_cut      = sweep['r_cut']
n_layers   = sweep['n_layers']

F                     = model_cfg['F']
degrees               = model_cfg['degrees']
n_rbf                 = model_cfg['n_rbf']
num_embeddings        = model_cfg['num_embeddings']
activation_name       = model_cfg['activation']
radial_basis_function = model_cfg['radial_basis_function']
radial_cutoff_function= model_cfg['radial_cutoff_function']
hardness_mode         = model_cfg['hardness_mode']
radii_mode            = model_cfg['radii_mode']

n_epochs              = train_cfg['n_epochs']
training_batch_size   = train_cfg['batch_size']
validation_batch_size = train_cfg['batch_size']
learning_rate         = train_cfg['learning_rate']
decay_rate            = train_cfg['decay_rate']
transition_steps      = train_cfg['transition_steps']
net_seed              = train_cfg['net_seed']
training_seed         = train_cfg['training_seed']

loss_weights = sweep['loss_weights']

run_tag  = f"{args.run_idx:02d}_{obs_mode}_rc{r_cut}_nl{n_layers}"
DATA_DIR = os.path.dirname(data_cfg['train'])
CKPT_DIR = os.path.join(out_cfg['base_dir'], run_tag, 'module')
PROJECT_DIR = os.path.join(out_cfg['base_dir'], run_tag)
data_files = (data_cfg['train'], data_cfg['valid'], data_cfg['test'])

os.makedirs(CKPT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Property key mapping  (dataset array name → model internal name)
# ---------------------------------------------------------------------------
prop_keys = {
    'energy':           'E',
    'force':            'F',
    'atomic_type':      'z',
    'atomic_position':  'R',
    'hirshfeld_volume': None,
    'total_charge':     'Q',
    'total_spin':       None,
    'partial_charge':   'q',
    'properties':       'p',
    'dipole':           'mu',
    'quadrupole':       'quad',
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
d_train = dict(np.load(data_files[0]))
d_valid = dict(np.load(data_files[1]))
d_test  = dict(np.load(data_files[2]))

# Compute neighbour index lists for each split
for split in [d_train, d_valid, d_test]:
    idx = get_indices(split['R'], split['z'], r_cut)
    split['idx_i'], split['idx_j'] = idx['idx_i'], idx['idx_j']

# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------
geometry_embeddings = [
    GeometryEmbed(
        degrees=degrees,
        radial_basis_function=radial_basis_function,
        n_rbf=n_rbf,
        radial_cutoff_fn=radial_cutoff_function,
        r_cut=r_cut,
        prop_keys=prop_keys,
        sphc=True,
    )
]

feature_embeddings = [
    ChargeEmbed(num_embeddings=num_embeddings, features=F, prop_keys=prop_keys),
    AtomTypeEmbed(num_embeddings=num_embeddings, features=F, prop_keys=prop_keys),
]

charge_embeddings = [
    CeqEmbed(num_embeddings=num_embeddings, prop_keys=prop_keys,
             hardness_mode=hardness_mode, radii_mode=radii_mode),
]

so3krates_defaults = {
    'degrees':                   degrees,
    'fb_rad_filter_features':    [F, F],
    'fb_sph_filter_features':    [F // 4, F],
    'gb_rad_filter_features':    [F, F],
    'gb_sph_filter_features':    [F // 4, F],
}
atomic_layers = [So3kratesLayer(**so3krates_defaults) for _ in range(n_layers)]

_OBS_CLS = {'dipole': Dipole, 'quadrupole': Quadrupole}
if obs_type not in _OBS_CLS:
    raise ValueError(f"Unknown obs_type {obs_type!r}. Choose 'dipole' or 'quadrupole'.")
observables = [_OBS_CLS[obs_type](mode=obs_mode, prop_keys=prop_keys, degrees=degrees, alpha=obs_alpha)]

target_keys = list(loss_weights.keys())

net = CeqNet(
    feature_embeddings=feature_embeddings,
    geometry_embeddings=geometry_embeddings,
    charge_embeddings=charge_embeddings,
    layers=atomic_layers,
    observables=observables,
    prop_keys=prop_keys,
    mode=obs_mode,
    eval_keys=target_keys,
)

# ---------------------------------------------------------------------------
# Initialise parameters
# ---------------------------------------------------------------------------
obs_fn = jax.vmap(get_obs_fn(net), in_axes=(None, 0))

input_keys = [prop_keys['atomic_position'], prop_keys['atomic_type'],
              'idx_i', 'idx_j', prop_keys['total_charge']]

data_tuple = DataTuple(input_keys=input_keys, target_keys=target_keys)
train_ds = data_tuple(d_train)
valid_ds = data_tuple(d_valid)
test_ds  = data_tuple(d_test)

init_input = jax.tree.map(lambda x: jnp.array(x[0, ...]), train_ds[0])
params = net.init(jax.random.PRNGKey(net_seed), init_input)

# ---------------------------------------------------------------------------
# Optimizer & train state
# ---------------------------------------------------------------------------
opt = Optimizer(decay_rate=decay_rate, transition_steps=transition_steps)
tx = opt.get(learning_rate=learning_rate)
train_state = create_train_state(net, params, tx)

# ---------------------------------------------------------------------------
# Save hyperparameters
# ---------------------------------------------------------------------------
h = bundle_dicts([net.__dict_repr__(), opt.__dict_repr__()])
save_json(path=CKPT_DIR, filename='hyperparameters.json', data=h, overwrite=True)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
loss_fn = get_loss_fn(obs_fn=obs_fn, weights=loss_weights)

coach = Coach(
    input_keys=input_keys,
    target_keys=target_keys,
    epochs=n_epochs,
    training_batch_size=training_batch_size,
    validation_batch_size=validation_batch_size,
    loss_weights=loss_weights,
    ckpt_dir=CKPT_DIR,
    data_path=DATA_DIR,
    net_seed=net_seed,
    training_seed=training_seed,
)

coach.run(
    train_state=train_state,
    train_ds=train_ds,
    valid_ds=valid_ds,
    loss_fn=loss_fn,
    print_every_t=1,
    ckpt_overwrite=True,
    use_wandb=False,
)

# ---------------------------------------------------------------------------
# Evaluate on test set
# ---------------------------------------------------------------------------
from flax.training import checkpoints
from src.inference import evaluate_model, mae_metric, rmse_metric
from pprint import pprint

print('\n=== Test evaluation ===')
for obs_key in target_keys:
    ckpt = checkpoints.restore_checkpoint(CKPT_DIR, prefix=f'checkpoint_{obs_key}_', target=None)
    metric_fn = {
        'mae':  mae_metric,
        'rmse': rmse_metric,
    }
    metrics, _ = evaluate_model(ckpt['params'], obs_fn, test_ds, validation_batch_size, metric_fn)
    pprint({obs_key + ' test ' + k: v[obs_key] for k, v in metrics.items()})
