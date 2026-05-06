# CeqNet — Charge Equilibration Network

## What this is 
Implementation of an equivariant self-attention message passing neural network for molecular multipole prediction in [JAX](https://github.com/google/jax) / [Flax](https://github.com/google/flax). Possible prediction modes are partial atomic charges, molecular dipoles and molecular multipoles. Per mode multiple prediction methods are available (see [Observable Modes](#observable-modes) for a list and [Methods](#methods) for details).


The network backbone is the SO3Krates architecture [^2]. Code was taken from the [mlff](https://github.com/thorben-frank/mlff) repository. See [below](#repository-structure-and-attribution) for an overview of the repository structure and attribution (`mlff` vs. `CeqNet`). Developed as part of a master's thesis project (2024)

## Installation
Download the repo
```bash
git clone https://github.com/martinmichajlow/ceqnet 
```
JAX with cuda support has to be installed separately
```
pip install --upgrade pip

# CUDA 12 installation
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# alternatively CUDA 11 installation
# pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda
```

Then install the repo locally

```
cd ceqnet
pip install -e .
```

Alternatively, build the provided Apptainer container which bundles all dependencies:

```bash
apptainer build .container/base.sif .container/container.def
```

## Quick start

A training example on a 100-molecule ANI-1x subset is included in `examples/` — no downloads needed. Edit the data and output paths in `examples/config.yaml`, then run:

```bash
python examples/train.py --run_idx 1   # run sweep configuration 1
```

To launch a full sweep on a SLURM cluster:

```bash
# Edit CONTAINER and path settings in examples/train.sh, then:
sbatch examples/train.sh
```

Sweep configurations (dipole and quadrupole runs across cutoff/layer combinations) are defined in `examples/config.yaml`.

## Data format

Training data is supplied as NumPy `.npz` files, one per split (train / val / test). Each file contains the following arrays, padded to a fixed maximum atom count `n_max`:

| Array | Shape | dtype | Description |
|-------|-------|-------|-------------|
| `R` | `(N, n_max, 3)` | float32 | Atomic positions in Å |
| `z` | `(N, n_max)` | int64 | Atomic numbers (0 for padding) |
| `Q` | `(N, 1)` | int64 | Total molecular charge |
| `q` | `(N, n_max, 1)` | float32 | Partial charges (e.g. Hirshfeld) |
| `mu` | `(N, 3, 1)` | float32 | Dipole moment in Debye |
| `quad` | `(N, 6, 1)` | float32 | Quadrupole tensor (upper triangle: xx, yy, zz, xy, xz, yz) |
| `E` | `(N, 1)` | float64 | Total energy |
| `F` | `(N, n_max, 3)` | float32 | Atomic forces |

Not all arrays are required — only those referenced by the configured `loss_weights` and `prop_keys` need to be present.

## Observable modes

The observable used for training and inference is controlled by the `obs_mode` field in `config.yaml`. Modes are one or more tokens joined by `+`. See the [Methods](#methods) for details on how they are computed.

| Charge | Dipole | Quadrupole |
|--------|--------|------------|
| `ceqcha` | `ceqdip` | `ceqquad` |
| `trivcha` | `trivdip` | `trivquad` |
| `redischa` | `redisdip` | `redisquad` |
| | `atomicdip` | `atomicquad` |
| | | `atomicdipquad` |

### Combining tokens

For a fixed prediction modality (`PartialCharge`, `Dipole` or `Quadrupole`) multiple tokens can be blended using the `alpha` field:

```yaml
# Single token — no blending
obs_mode: ceqdip

# Multiple tokens — fixed weights (normalized to sum to 1)
obs_mode: ceqdip+redisdip+atomicdip
alpha: [1.0, 2.0, 1.0]   # effective weights: 0.25 / 0.5 / 0.25

# or alternatively learned blend (softmax over logits)
alpha: learnable
```

## CeQ Embedding Modes

`CeqEmbed` produces per-atom atomic hardnesses $J_i$ and Gaussian widths $\sigma_i$ used
in the charge-equilibration layer. Both are configured independently via `hardness_mode`
and `radii_mode` in the model config.

### Hardness modes (`hardness_mode`)

| Token | Formula | Notes |
|-------|---------|-------|
| `learnable` | $e_z$ | Unconstrained; can be negative |
| `zero` | $0$ | Disables the on-site diagonal term |
| `exp` | $\exp(e_z)$ | Positive, positive, unbounded |
| `softplus` | $\log(1 + \exp(e_z))$ | Softplus; positive, unbounded |
| `{N}sgm` | $N \cdot S(e_z)$ | Positive, bounded in $(0, N)$; e.g. `20sgm` |
| `a_sgm` | $\|e_z\| \cdot S(e_z)$ | Non-negative, data-dependent scale |
| `a_abs` | $\|e_z\|$ | Non-negative absolute value |


### Atomic width modes (`radii_mode`)
All hardness tokens are also valid for `radii_mode`. In addition, fixed and scaled table-based modes are available:

| Token | Formula | Notes |
|-------|---------|-------|
| `ase` | $r^\text{cov}_z$ | Fixed ASE covalent radii per element |
| `qeq` | $r^\text{qeq}_z$ | Fixed QeQ radii (Rappé & Goddard, 1991) |
| `ase_scaled` | $f_z \cdot r^\text{cov}_z$ | ASE radii scaled by a learnable per-element factor $f_z$ |
| `qeq_scaled` | $f_z \cdot r^\text{qeq}_z$ | QeQ radii scaled by a learnable per-element factor $f_z$ |

Here $e_z \in\mathbb{R}$ is a learnable per-element embedding scalar and $S$ is the sigmoid function.

## Methods

> **Note on math rendering:** This section contains LaTeX math blocks. Rendering depends on GitHub's MathML support, which varies by browser — Safari renders all expressions correctly; Chromium-based browsers (Chrome, Brave, Edge) may display some blocks incorrectly.

Here we describe how the different partial charge, dipole and quadrupole quantities are computed from the invariant features $\boldsymbol{f}_i$ and the equivariant output features $\boldsymbol{\chi}_i^{l=1}$ and $\boldsymbol{\chi}_i^{l=2}$ of the network (with $i=1,\ldots,n_a$).

### Partial Charges
For a system of $n_a$ atoms with types $z_1,\dots z_{n_a}$ and total charge $Q$ we predict partial charges with the following methods:

#### $\text{Triv}$
**Token:** `trivcha`

$$q_i = \boldsymbol{w}_q^\top \boldsymbol{f}_i + q_{z}$$

with shared, learnable weight vector ${\boldsymbol{w}}_q$ and atomtype specific bias term ${q}_z$.

#### $\text{ReDis}$
**Token:** `redischa`

$$q_i = \text{Triv}(\boldsymbol{f}_i) + \frac{1}{n_a} \left(Q-\sum_{j=1}^{n_a}\text{Triv}(\boldsymbol{f}_j)\right)$$

#### $\text{CeQ}$
**Token:** `ceqcha`

Inspired by charge equilibration [^1]. Compute electronegativities $(\chi_i)_{i=1}^{n_a}$ from invariant features

$$\chi_i = \text{Silu}(\boldsymbol{w}^\top \boldsymbol{f}_i)$$

with learnable regression weights $\boldsymbol{w}\in\mathbb{R}^F$ and atomic hardness $J_i$ and atomic widths $\sigma_i$ from atomic embeddings

$$J_i = 20\cdot S(f_{\text{emb},1}(z_i))$$

$$\sigma_i = 20 \cdot S(f_{\text{emb},2}(z_i))$$

with $f_{\text{emb},1},f_{\text{emb},2}:\{0,\dots,z_{\max}\}\to \mathbb{R}$ learnable and sigmoid function $S$. The scaling factors 20 are somewhat arbitrarily chosen to enforce values ranges akin to the original charge equilibration setting (see [CeQ Embedding Modes](#ceq-embedding-modes) for other embedding options). Build the matrix

$$
A_{ij} = \begin{cases}
J_i + \frac{1}{\sigma_i \sqrt{\pi}} & \text{if } i=j, \\
\frac{\text{erf}\left(\frac{r_{ij}}{\sqrt{2}\gamma_{ij}}\right)}{r_{ij}} & \text{otherwise,}
\end{cases}
$$

with $\gamma_{ij} = \sqrt{\sigma_i^2 + \sigma_j^2}$, interatomic distances $r_{ij}$ and error function $\text{erf}$, and solve

$$\begin{pmatrix}
  \begin{array}{c|c}
    \begin{matrix} & & \\\ & A & \\\ & & \end{matrix} & \begin{matrix} 1 \\\ \vdots \\\ 1 \end{matrix} \\\
    \hline
    \begin{matrix} 1 & \cdots & 1 \end{matrix} & 0
  \end{array}
\end{pmatrix}
\begin{pmatrix}
  \begin{array}{c}
    q_1 \\\
    \vdots \\\
    q_{n_a} \\\
    \hline
    \lambda
  \end{array}
\end{pmatrix} = \begin{pmatrix} \begin{array}{c} -\chi_1 \\\
    \vdots \\\
    -\chi_{n_a} \\\
    \hline
    Q
  \end{array}
\end{pmatrix}$$

### Dipoles

#### From Partial Charges
**Tokens:** `ceqdip` (uses `ceqcha`), `trivdip` (uses `trivcha`), `redisdip` (uses `redischa`)

Use one of the partial charge methods from above and compute

$$\boldsymbol{\mu} = \sum_{i=1}^{n_a} q_i \boldsymbol{r}_i$$,

where $\boldsymbol{r}_i$, $i=1,\dots,n_a$ are the atomic positions.

#### From Equivariant Features
**Token:** `atomicdip`

Compute atomic dipole $\boldsymbol{\mu}_i$ as

$$\boldsymbol{\mu}_i = \phi(\boldsymbol{f}_i) P \boldsymbol{\chi}_i^{(l=1)}$$

and molecular dipole as

$$\boldsymbol{\mu} = \sum_{i=1}^{n_a} \boldsymbol{\mu}_i$$

where the matrix

$$
P = \begin{pmatrix} 0 & 0 & 1 \\\ 1 & 0 & 0 \\\ 0 & 1 & 0 \end{pmatrix}
$$

accounts for reordering spherical harmonics in spherical coordinates to cartesian order and a one-layer MLP $\phi$ with $\text{Silu}$ activation.

### Quadrupoles
We predict the upper triangular of the traceless quadrupole tensor $\mathcal{Q}$ as the vector

$$(\mathcal{Q}_{11}-\tfrac{1}{3}\text{tr}(\mathcal{Q}), \mathcal{Q}_{22}-\tfrac{1}{3}\text{tr}(\mathcal{Q}), \mathcal{Q}_{33}-\tfrac{1}{3}\text{tr}(\mathcal{Q}), \mathcal{Q}_{12}, \mathcal{Q}_{13}, \mathcal{Q}_{23})^\top.$$

Hence, for the following computations of $3\times 3$ symmetric matrices we post-hoc vectorize the upper triangular and subtract the trace accordingly.


#### From Partial Charges
**Tokens:** `ceqquad` (uses `ceqcha`), `trivquad` (uses `trivcha`), `redisquad` (uses `redischa`)

Use one of the partial charge methods from above and compute

$$\tilde{\mathcal{Q}} = \sum_{i=1}^{n_a} q_i (\boldsymbol{r}_i \otimes \boldsymbol{r}_i)$$

#### From Equivariant Features $(l=1)$
**Token:** `atomicdipquad`

Compute atomic dipoles $\mu_i$ as above and

$$\tilde{\mathcal{Q}} = \sum_{i=1}^{n_a} \boldsymbol{r}_i \otimes \boldsymbol{\mu}_i + \boldsymbol{\mu}_i \otimes \boldsymbol{r}_i$$

#### From Equivariant Features $(l=2)$
**Token:** `atomicquad`

Predict atomic quadrupole $\hat{\mathcal{Q}}_i$ as

$$\hat{\mathcal{Q}}_i = \text{detrace}\left[Q^{-1}\left[D\text{pad}_1\left(\frac{\boldsymbol{\chi}_i^{l=2}}{\lVert\boldsymbol{\chi}_i^{l=2}\rVert_2}\right)+\boldsymbol{e_3}\right]\right]$$

where

$$\text{pad}_1:\mathbb{R}^5\to\mathbb{R}^6, \quad (x_1,x_2,x_3,x_4,x_5)^\top\mapsto (x_1,x_2,x_3,x_4,x_5,1)^\top$$

$$D=\text{diag}\left(2\sqrt{\tfrac{\pi}{15}},2\sqrt{\tfrac{\pi}{15}},4\sqrt{\tfrac{\pi}{5}},2\sqrt{\tfrac{\pi}{15}},4\sqrt{\tfrac{\pi}{15}},1\right) \in \mathbb{R}^{6\times 6}$$

the canonical unit vector $\boldsymbol{e_3}\in\mathbb{R}^6$ and

$$
Q=\begin{pmatrix} 0 & 0 & 0 & 1 & 0 & 0 \\\ 0 & 0 & 0 & 0 & 0 & 1 \\\ 0 & 0 & 3 & 0 & 0 & 0 \\\ 0 & 0 & 0 & 0 & 1 & 0 \\\ 1 & -1 & 0 & 0 & 0 & 0 \\\ 1 & 1 & 1 & 0 & 0 & 0 \end{pmatrix}
$$

The rationale for this is that we assume the equivariant feature $\boldsymbol{\chi}_i^{l=2}$ to encode angular information of an atomic quadrupole contribution $q_i (\boldsymbol{r}_i\otimes \boldsymbol{r}_i)$, whose vectorized upper triangular up to scaling with $q_i$ can be expressed as $\boldsymbol{v}=(x^2,y^2,z^2,xy,xz,yz)^\top$. Assuming a unit vector $\boldsymbol{r}=(x,y,z)^\top$ (since we are only interested in direction), the real spherical harmonic of degree $2$ can be written as

$$
Y^2(\boldsymbol{r})=\begin{pmatrix} \frac{1}{2}\sqrt{\frac{15}{\pi}}\cdot xy \\\ \frac{1}{2}\sqrt{\frac{15}{\pi}}\cdot yz \\\ \frac{1}{4}\sqrt{\frac{5}{\pi}}\cdot(3z^2-1) \\\ \frac{1}{2}\sqrt{\frac{15}{\pi}}\cdot xz \\\ \frac{1}{4}\sqrt{\frac{15}{\pi}}\cdot(x^2-y^2) \end{pmatrix}.
$$

It can be easily confirmed that it holds

$$Q\boldsymbol{v} = D\text{pad}_1(Y^2(\boldsymbol{r}))+\boldsymbol{e_3}$$
## Repository Structure and Attribution

The equivariant message passing network architecture (So3krates [^2]) and much of the infrastructure originates from the [mlff](https://github.com/thorben-frank/mlff) library by Thorben Frank et al.:

> Thorben Frank et al., *mlff — Machine Learning Force Fields*,
> https://github.com/thorben-frank/mlff, commit `99dbf76` (Sep 30, 2022)

Within this project file headers show two levels of attribution:

| Header | Tag | Meaning |
|--------|-----|---------|
| `# Taken from mlff` | [mlff] | File is reproduced as-is; only import paths were updated (`mlff.src → src`) |
| `# Adapted from mlff` | [mlff + CeQ] | File was substantively modified (classes replaced, code removed, logic changed) and/or mixes mlff-derived and original code; the header lists which parts are which |

Files with **none** of the above headers are original contributions of this project. An overview of the repository structure is

```
src/
├── nn/
│   ├── ceqnet/                 # CeqNet model, forward pass, observable dispatch   [CeQ]
│   ├── layer/
│   │   ├── so3krates_layer.py  # SO3Krates equivariant message-passing layer       [mlff]
│   │   └── schnet_layer.py     # SchNet layer                                      [mlff]
│   ├── embed/
│   │   ├── embed.py            # Geometry, atom-type, charge, Qeq embeddings       [mlff + CeQ]
│   │   └── h_register.py       # Embedding module registry                         [mlff]
│   ├── observable/             # Multipole observables              [mlff + CeQ]
│   ├── fast_attention/         # FAVOR+ fast attention                              [mlff]
│   ├── mlp/                    # MLP building block                                 [mlff]
│   ├── base/                   # Base module class                                  [mlff]
│   └── activation_function/    # Activation functions                               [mlff]
├── sph_ops/                    # Spherical harmonic contractions, CG coefficients   [mlff]
├── basis_function/             # Radial and spherical basis functions                [mlff]
├── cutoff_function/            # Radial cutoff and PBC utilities                    [mlff]
├── training/
│   ├── coach.py                # Training orchestration                             [mlff + CeQ]
│   ├── loss.py                 # Loss functions                                     [mlff + CeQ]
│   ├── run.py                  # Low-level train/valid epoch loops                  [mlff]
│   ├── optimizer.py            # Optimizer with exponential LR decay                [mlff]
│   └── train_state.py          # Flax TrainState wrapper                            [mlff]
├── inference/                  # Evaluation metrics and model evaluation             [mlff + CeQ]
├── data/                       # DataSet, DataTuple, preprocessing                  [mlff]
├── indexing/                   # Neighbour index computation                        [mlff]
├── io/                         # Data loaders: QM9, 4th-generation NNP data         [mlff + CeQ]
├── padding/                    # Padding for batched inference                       [mlff + CeQ]
├── geometric/                  # Rotation matrices, SPHC metric                     [mlff]
├── masking/                    # Safe scale/mask operations                         [mlff]
├── random/                     # Seed utilities                                     [mlff]
└── debugging/                  # NaN catching utilities                             [CeQ]

examples/
├── train.py                    # Minimal end-to-end training + evaluation example
├── train.sh                    # SLURM array job script (8-run sweep)
├── config.yaml                 # Sweep hyperparameters
└── data/                       # QM9 subset (100 molecules, no download needed)

.container/
└── container.def               # Apptainer/Singularity container definition (Python 3.12 + JAX CUDA)
```



## Dependencies

| Package | Purpose |
|---------|---------|
| `jax`, `jaxlib` | Numerical backend, JIT, autodiff (install separately with CUDA support — see above) |
| `flax` | Neural network layers and training state |
| `optax` | Optimizers |
| `orbax-checkpoint == 0.5.23` | Checkpoint saving and restoring |
| `jaxopt` | Implicit differentiation utilities |
| `jraph` | Graph neural network primitives |
| `numpy` | Numerics |
| `ase` | Covalent radii for charge embedding |
| `scikit-learn` | Stratified data splitting |
| `tqdm` | Progress bars |
| `wandb` | Experiment tracking (optional) |
| `pyyaml` | Config file parsing |
| `h5py` | HDF5 data loading |

## References
[^1]: Rappé, A. K., & Goddard III, W. A. (1991). Charge equilibration for molecular dynamics simulations.
[^2]: Frank, J.T., Unke, O.T., Müller, KR. et al. A Euclidean transformer for fast and stable machine learned force fields. Nat Commun 15, 6539 (2024).
