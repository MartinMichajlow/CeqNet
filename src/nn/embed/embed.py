# Partially adapted from mlff (https://github.com/thorben-frank/mlff, commit 99dbf76)
# Original author (mlff): Thorben Frank et al.
# Adapted from mlff: AtomTypeEmbed, ChargeSpinEmbed, GeometryEmbed (base structure), _init_sphc
# Adapted from ChargeSpinEmbed (mlff): ChargeEmbed (wraps inputs via prop_keys; adds safe_mask
#   for padded atoms and zeroes output when total charge Q=0)
# Original contributions: CeqEmbed, HardnessEmbed

import re

import jax
import jax.numpy as jnp
from jax.nn import silu
import flax.linen as nn

from typing import (Any, Dict, Sequence)

import numpy as np
import logging

from src.nn.base.sub_module import BaseSubModule
from src.masking.mask import safe_scale
from src.masking.mask import safe_mask
from src.basis_function.radial import get_rbf_fn
from src.cutoff_function.pbc import add_cell_offsets
from src.cutoff_function.radial import get_cutoff_fn
from functools import partial
from jax.ops import segment_sum
from ase.data import covalent_radii

from src.nn.mlp import MLP
from src.sph_ops.spherical_harmonics import init_sph_fn

class AtomTypeEmbed(BaseSubModule):
    num_embeddings: int
    features: int
    prop_keys: Dict
    module_name: str = 'atom_type_embed'

    def setup(self):
        self.atomic_type_key = self.prop_keys.get('atomic_type')

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs) -> jnp.ndarray:
        """
        Create atomic embeddings based on the atomic types.

        Args:
            inputs (Dict):
                z (Array): atomic types, shape: (n)
                point_mask (Array): Mask for atom-wise operations, shape: (n)
            *args (Tuple):
            **kwargs (Dict):

        Returns: Atomic embeddings, shape: (n,F)

        """
        z = inputs[self.atomic_type_key]
        point_mask = inputs['point_mask']

        z = z.astype(jnp.int32)  # shape: (n)
        return safe_scale(nn.Embed(num_embeddings=self.num_embeddings, features=self.features)(z),
                          scale=point_mask[:, None])

    def __dict_repr__(self):
        return {self.module_name: {'num_embeddings': self.num_embeddings,
                                   'features': self.features,
                                   'prop_keys': self.prop_keys}}



qeq_radii = np.array([np.inf, 0.371, 1.3, 1.557, 1.24, 0.822, 0.759, 0.715, 0.669, 0.706, 1.768, 2.085, 1.5, 1.201, 1.176, 1.102, 1.047])


def _parse_sgm_scale(mode: str):
    """Return the float scale from a '{N}sgm' mode string, or None if not that pattern."""
    m = re.match(r'^(\d+(?:\.\d+)?)sgm$', mode)
    return float(m.group(1)) if m else None


# [Original contribution]
class CeqEmbed(BaseSubModule):
    """
    Produces per-atom atomic hardnesses J_i and Gaussian widths σ_i for the
    charge-equilibration (CeQ) layer.

    hardness_mode and radii_mode accept the same set of tokens:

      'learnable'   — unconstrained learnable embedding; output can be negative.
      'zero'        — fixed at zero (disables the on-site diagonal in the CeQ matrix).
      'exp'         — exp(e_i); always positive, unbounded.
      '{N}sgm'      — N · σ(e_i); positive, bounded in (0, N).  N is any positive
                      number, e.g. '20sgm'.
      'a_sgm'       — |e_i| · σ(e_i); non-negative, data-dependent upper bound.
      'a_abs'       — |e_i|; non-negative absolute value.
      'softplus'      — log(1 + exp(e_z)); always positive, smooth.

    Additional tokens for radii_mode only (fixed, non-learnable tables):
      'ase'         — ASE covalent radii (fixed per element).
      'qeq'         — QeQ radii from Rappé & Goddard (1991) (fixed per element).
      'ase_scaled'  — ASE radii multiplied by a learnable per-element scale factor.
      'qeq_scaled'  — QeQ radii multiplied by a learnable per-element scale factor.

    Outputs: {'J': Array (n,), 'sigma': Array (n,)}
    """
    num_embeddings: int
    prop_keys: Dict
    hardness_mode: str
    radii_mode: str
    module_name: str = 'qeq_embed'

    def setup(self):
        self.total_charge_key = self.prop_keys.get('total_charge')
        self.atomic_type_key = self.prop_keys.get('atomic_type')

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs):
        """
        Args:
            inputs (Dict):
                z (Array): atomic types, shape: (n)
                point_mask (Array): mask for atom-wise operations, shape: (n)
        Returns: {'J': Array (n,), 'sigma': Array (n,)}
        """
        z = inputs[self.atomic_type_key].astype(jnp.int32)
        point_mask = inputs['point_mask']

        def _embed1():
            return nn.Embed(num_embeddings=self.num_embeddings, features=1)(z)

        def _scalar(raw):
            return jnp.squeeze(safe_scale(raw, scale=point_mask[:, None]))

        # --- hardness J ---
        sgm = _parse_sgm_scale(self.hardness_mode)
        if sgm is not None:
            J = _scalar(sgm * jax.nn.sigmoid(_embed1()))
        elif self.hardness_mode == 'learnable':
            J = _scalar(_embed1())
        elif self.hardness_mode == 'zero':
            J = jnp.zeros(z.shape, dtype=jnp.float32)
        elif self.hardness_mode == 'exp':
            J = _scalar(jnp.exp(_embed1()))
        elif self.hardness_mode == 'a_sgm':
            e = _embed1()
            J = _scalar(jnp.abs(e) * jax.nn.sigmoid(e))
        elif self.hardness_mode == 'a_abs':
            J = _scalar(jnp.abs(_embed1()))
        elif self.hardness_mode == 'softplus':
            J = _scalar(jnp.log(1 + jnp.exp(_embed1())))
        else:
            raise ValueError(f"Unknown hardness_mode: {self.hardness_mode!r}")

        # --- Gaussian widths sigma ---
        sgm = _parse_sgm_scale(self.radii_mode)
        if sgm is not None:
            sigma = _scalar(sgm * jax.nn.sigmoid(_embed1()))
        elif self.radii_mode == 'learnable':
            sigma = _scalar(_embed1())
        elif self.radii_mode == 'ase':
            sigma = safe_scale(jnp.take(covalent_radii, z), scale=point_mask)
        elif self.radii_mode == 'qeq':
            sigma = safe_scale(jnp.take(qeq_radii, z), scale=point_mask)
        elif self.radii_mode == 'ase_scaled':
            base = safe_scale(jnp.take(covalent_radii, z), scale=point_mask)
            sigma = _scalar(_embed1()) * base
        elif self.radii_mode == 'qeq_scaled':
            base = safe_scale(jnp.take(qeq_radii, z), scale=point_mask)
            sigma = _scalar(_embed1()) * base
        elif self.radii_mode == 'exp':
            sigma = _scalar(jnp.exp(_embed1()))
        elif self.radii_mode == 'a_sgm':
            e = _embed1()
            sigma = _scalar(jnp.abs(e) * jax.nn.sigmoid(e))
        elif self.radii_mode == 'a_abs':
            sigma = _scalar(jnp.abs(_embed1()))
        elif self.radii_mode == 'softplus':
            sigma = _scalar(jnp.log(1 + jnp.exp(_embed1())))
        else:
            raise ValueError(f"Unknown radii_mode: {self.radii_mode!r}")

        return {'J': J, 'sigma': sigma}

    def __dict_repr__(self):
        return {self.module_name: {'num_embeddings': self.num_embeddings,
                                   'prop_keys': self.prop_keys,
                                   'hardness_mode': self.hardness_mode,
                                   'radii_mode': self.radii_mode}}

# [Original contribution]
class HardnessEmbed(BaseSubModule):
    num_embeddings: int
    prop_keys: Dict
    module_name: str = 'hardness_embed'

    """
    Returns atomic hardness as in "Ko, T.W., Finkler, J.A., Goedecker, S. et al. A fourth-generation high-dimensional 
    neural network potential with accurate electrostatics including non-local charge transfer. Nat Commun 12, 398
    (2021). https://doi.org/10.1038/s41467-020-20427-2"
    """

    def setup(self):
        self.total_charge_key = self.prop_keys.get('total_charge')
        self.atomic_type_key = self.prop_keys.get('atomic_type')

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """

        Args:
           inputs (Dict):
                z (Array): atomic types, shape: (n)
                Q (Array): total charge, shape: (1)
                point_mask (Array): Mask for atom-wise operations, shape: (n)
            *args ():
            **kwargs ():

        Returns:

        """
        z = inputs[self.atomic_type_key]
        point_mask = inputs['point_mask']
        cov_r_std = jnp.array([0, 5, 0, 7, 3, 3, 1, 1, 2, 3, 0, 9, 7])
        sigma = safe_scale(jnp.take(cov_r_std, z), scale=point_mask)
        z = z.astype(jnp.int32)  # shape: (n)
        J = safe_scale(nn.Embed(num_embeddings=self.num_embeddings, features=1)(z), scale=point_mask[:, None])  # shape: (n_atoms,1)
        J = jnp.squeeze(J)
        return {'J': J, 'sigma': sigma}

    def __dict_repr__(self):
        return {self.module_name: {'num_embeddings': self.num_embeddings,
                                   'prop_keys': self.prop_keys}}


class GeometryEmbed(BaseSubModule):
    prop_keys: Dict
    degrees: Sequence[int]
    radial_basis_function: str
    n_rbf: int
    radial_cutoff_fn: str
    r_cut: float
    sphc: bool
    sphc_normalization: float = None
    mic: bool = False
    solid_harmonic: bool = False
    input_convention: str = 'positions'
    module_name: str = 'geometry_embed'

    def setup(self):
        if self.input_convention == 'positions':
            self.atomic_position_key = self.prop_keys.get('atomic_position')
            if self.mic == 'bins':
                logging.warning(f'mic={self.mic} is deprecated in favor of mic=True.')
            if self.mic == 'naive':
                raise DeprecationWarning(f'mic={self.mic} is not longer supported.')
            if self.mic:
                self.unit_cell_key = self.prop_keys.get('unit_cell')
                self.cell_offset_key = self.prop_keys.get('cell_offset')

        elif self.input_convention == 'displacements':
            self.displacement_vector_key = self.prop_keys.get('displacement_vector')
        else:
            raise ValueError(f"{self.input_convention} is not a valid argument for `input_convention`.")

        self.atomic_type_key = self.prop_keys.get('atomic_type')

        self.sph_fns = [init_sph_fn(y) for y in self.degrees]

        _rbf_fn = get_rbf_fn(self.radial_basis_function)
        self.rbf_fn = _rbf_fn(n_rbf=self.n_rbf, r_cut=self.r_cut)

        _cut_fn = get_cutoff_fn(self.radial_cutoff_fn)
        self.cut_fn = partial(_cut_fn, r_cut=self.r_cut)
        self._lambda = jnp.float32(self.sphc_normalization) if self.sphc_normalization is not None else None

    def __call__(self, inputs: Dict, *args, **kwargs):
        """
        Embed geometric information from the atomic positions and its neighboring atoms.
        Args:
            inputs (Dict):
                R (Array): atomic positions, shape: (n,3)
                idx_i (Array): index centering atom, shape: (n_pairs)
                idx_j (Array): index neighboring atom, shape: (n_pairs)
                pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pairs)
            *args ():
            **kwargs ():

        Returns:
        """
        idx_i = inputs['idx_i']  # shape: (n_pairs)
        idx_j = inputs['idx_j']  # shape: (n_pairs)
        pair_mask = inputs['pair_mask']  # shape: (n_pairs)

        # depending on the input convention, calculate the displacement vectors or load them from input
        if self.input_convention == 'positions':
            R = inputs[self.atomic_position_key]  # shape: (n,3)
            # Calculate pairwise distance vectors
            r_ij = safe_scale(jax.vmap(lambda i, j: R[j] - R[i])(idx_i, idx_j), scale=pair_mask[:, None])
            # shape: (n_pairs,3)

            # Apply minimal image convention if needed
            if self.mic:
                cell = inputs[self.unit_cell_key]  # shape: (3,3)
                cell_offsets = inputs[self.cell_offset_key]  # shape: (n_pairs,3)
                r_ij = add_cell_offsets(r_ij=r_ij, cell=cell, cell_offsets=cell_offsets)  # shape: (n_pairs,3)

        elif self.input_convention == 'displacements':
            R = None
            r_ij = inputs[self.displacement_vector_key]
        else:
            raise ValueError(f"{self.input_convention} is not a valid argument for `input_convention`.")

        # Scale pairwise distance vectors with pairwise mask
        r_ij = safe_scale(r_ij, scale=pair_mask[:, None])

        # Calculate pairwise distances
        d_ij = safe_scale(jnp.linalg.norm(r_ij, axis=-1), scale=pair_mask)  # shape : (n_pairs)

        # Gaussian basis expansion of distances
        rbf_ij = safe_scale(self.rbf_fn(d_ij[:, None]), scale=pair_mask[:, None])  # shape: (n_pairs,K)
        phi_r_cut = safe_scale(self.cut_fn(d_ij), scale=pair_mask)  # shape: (n_pairs)

        # Normalized distance vectors
        unit_r_ij = safe_mask(mask=d_ij[:, None] != 0,
                              operand=r_ij,
                              fn=lambda y: y / d_ij[:, None],
                              placeholder=0
                              )  # shape: (n_pairs, 3)
        unit_r_ij = safe_scale(unit_r_ij, scale=pair_mask[:, None])  # shape: (n_pairs, 3)

        # Spherical harmonics
        sph_harms_ij = []
        for sph_fn in self.sph_fns:
            sph_ij = safe_scale(sph_fn(unit_r_ij), scale=pair_mask[:, None])  # shape: (n_pairs,2l+1)
            sph_harms_ij += [sph_ij]  # len: |L| / shape: (n_pairs,2l+1)

        sph_harms_ij = jnp.concatenate(sph_harms_ij, axis=-1) if len(self.degrees) > 0 else None
        # shape: (n_pairs,m_tot)

        geometric_data = {'R': R,
                          'r_ij': r_ij,
                          'unit_r_ij': unit_r_ij,
                          'd_ij': d_ij,
                          'rbf_ij': rbf_ij,
                          'phi_r_cut': phi_r_cut,
                          'sph_ij': sph_harms_ij,
                          }

        # Spherical harmonic coordinates (SPHCs)
        if self.sphc:
            z = inputs[self.atomic_type_key]
            point_mask = inputs['point_mask']
            if self.sphc_normalization is None:
                # Initialize SPHCs to zero
                geometric_data.update(_init_sphc_zeros(z=z,
                                                       sph_ij=sph_harms_ij,
                                                       phi_r_cut=phi_r_cut,
                                                       idx_i=idx_i,
                                                       point_mask=point_mask,
                                                       mp_normalization=self._lambda)
                                      )
            else:
                # Initialize SPHCs with a neighborhood dependent embedding
                geometric_data.update(_init_sphc(z=z,
                                                 sph_ij=sph_harms_ij,
                                                 phi_r_cut=phi_r_cut,
                                                 idx_i=idx_i,
                                                 point_mask=point_mask,
                                                 mp_normalization=self._lambda)
                                      )

        # Solid harmonics (Spherical harmonics + radial part)
        if self.solid_harmonic:
            rbf_ij = safe_scale(rbf_ij, scale=phi_r_cut[:, None])  # shape: (n_pairs,K)
            g_ij = sph_harms_ij[:, :, None] * rbf_ij[:, None, :]  # shape: (n_pair,m_tot,K)
            g_ij = safe_scale(g_ij, scale=pair_mask[:, None, None], placeholder=0)  # shape: (n_pair,m_tot,K)
            geometric_data.update({'g_ij': g_ij})

        return geometric_data

    def reset_input_convention(self, input_convention: str) -> None:
        self.input_convention = input_convention

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'degrees': self.degrees,
                                   'radial_basis_function': self.radial_basis_function,
                                   'n_rbf': self.n_rbf,
                                   'radial_cutoff_fn': self.radial_cutoff_fn,
                                   'r_cut': self.r_cut,
                                   'sphc': self.sphc,
                                   'sphc_normalization': self.sphc_normalization,
                                   'solid_harmonic': self.solid_harmonic,
                                   'mic': self.mic,
                                   'input_convention': self.input_convention,
                                   'prop_keys': self.prop_keys}
                }


def _init_sphc(z, sph_ij, phi_r_cut, idx_i, point_mask, mp_normalization, *args, **kwargs):
    _sph_harms_ij = safe_scale(sph_ij, phi_r_cut[:, None])  # shape: (n_pairs,m_tot)
    chi = segment_sum(_sph_harms_ij, segment_ids=idx_i, num_segments=len(z))
    chi = safe_scale(chi, scale=point_mask[:, None])  # shape: (n,m_tot)
    return {'chi': chi / mp_normalization}


def _init_sphc_zeros(z, sph_ij, *args, **kwargs):
    return {'chi': jnp.zeros((z.shape[-1], sph_ij.shape[-1]), dtype=sph_ij.dtype)}


# [Adapted from ChargeSpinEmbed (mlff)]
class ChargeEmbed(BaseSubModule):
    features: int
    prop_keys: Dict
    num_embeddings: int = 100
    module_name: str = 'tot_charge_embed'

    def setup(self):
        self.total_charge_key = self.prop_keys.get('total_charge')
        self.atomic_type_key = self.prop_keys.get('atomic_type')

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """
        Args:
           inputs (Dict):
                z (Array): atomic types, shape: (n)
                Q (Array): total charge, shape: (1)
                point_mask (Array): Mask for atom-wise operations, shape: (n)
            *args ():
            **kwargs ():
        Returns:
        """
        z = inputs[self.atomic_type_key]
        Q = inputs[self.total_charge_key]
        point_mask = inputs['point_mask']

        z = z.astype(jnp.int32)
        q = nn.Embed(num_embeddings=self.num_embeddings, features=self.features)(z)  # shape: (n,F)
        Q_ = Q // jnp.inf  # -1 if Q < 0 and 0 otherwise
        Q_ = Q_.astype(jnp.int32)  # shape: (1)
        k = nn.Embed(num_embeddings=2, features=self.features)(Q_)  # shape: (1,F)
        v = nn.Embed(num_embeddings=2, features=self.features)(Q_)  # shape: (1,F)
        q_x_k = (q * k).sum(axis=-1) / jnp.sqrt(self.features)  # shape: (n)
        q_x_k = safe_scale(q_x_k,
                           scale=point_mask,
                           placeholder=-1e10)  # shape: (n)

        def calculate_numerator(u):
            w = jnp.log(1 + jnp.exp(u))
            return w

        numerator = safe_mask(mask=point_mask, fn=calculate_numerator, operand=q_x_k)
        a = safe_mask(mask=numerator != 0, fn=lambda x: Q * x / x.sum(axis=-1), operand=numerator, placeholder=0)
        e_Q = MLP(features=[self.features, self.features],
                    activation_fn=silu,
                    use_bias=False)(a[:, None] * v)  # shape: (n,F)
        Q_mask = jnp.ones_like(e_Q) * (Q!=0)
        e_Q = jnp.where(Q_mask, e_Q, jnp.zeros_like(e_Q))
        return safe_scale(e_Q, scale=point_mask[:, None])  # shape: (n,F)

    def __dict_repr__(self):
        return {self.module_name: {'num_embeddings': self.num_embeddings,
                                   'features': self.features,
                                   'prop_keys': self.prop_keys}}

class ChargeSpinEmbed(nn.Module):
    num_embeddings: int
    features: int

    @nn.compact
    def __call__(self,
                 z: jnp.ndarray,
                 psi: jnp.ndarray,
                 point_mask: jnp.ndarray,
                 *args,
                 **kwargs) -> jnp.ndarray:
        """
        Create atomic embeddings based on the total charge or the number of unpaired spins in the system, following the
        embedding procedure introduced in SpookyNet. Returns per atom embeddings of dimension F.
        Args:
            z (Array): Atomic types, shape: (n)
            psi (Array): Total charge or number of unpaired spins, shape: (1)
            point_mask (Array): Mask for atom-wise operations, shape: (n)
            *args ():
            **kwargs ():
        Returns: Per atom embedding, shape: (n,F)
        """
        z = z.astype(jnp.int32)  # shape: (n)
        q = nn.Embed(num_embeddings=self.num_embeddings, features=self.features)(z)  # shape: (n,F)
        psi_ = psi // jnp.inf  # -1 if psi < 0 and 0 otherwise
        psi_ = psi_.astype(jnp.int32)  # shape: (1)
        k = nn.Embed(num_embeddings=2, features=self.features)(psi_)  # shape: (1,F)
        v = nn.Embed(num_embeddings=2, features=self.features)(psi_)  # shape: (1,F)
        q_x_k = (q*k).sum(axis=-1) / jnp.sqrt(self.features)  # shape: (n)
        q_x_k = safe_scale(q_x_k,
                           scale=point_mask,
                           placeholder=-1e10)  # shape: (n)

        numerator = jnp.log(1 + jnp.exp(q_x_k))  # shape: (n)
        a = psi * numerator / numerator.sum(axis=-1)  # shape: (n)

        e_psi = MLP(features=[self.features, self.features],
                    activation_fn=silu,
                    use_bias=False)(a[:, None] * v)  # shape: (n,F)

        e_psi = jnp.where(psi != 0, e_psi, jnp.zeros_like(e_psi))
        return safe_scale(e_psi, scale=point_mask[:, None])  # shape: (n,F)

