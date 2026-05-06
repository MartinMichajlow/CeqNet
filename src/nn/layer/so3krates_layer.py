# Adapted from mlff (https://github.com/thorben-frank/mlff, commit 99dbf76)
# Original author: Thorben Frank et al.
# Modifications: renamed imports (mlff.src → src); replaced SPHCLayerNorm with FAVOR+ fast attention hook

import jax.numpy as jnp
import flax.linen as nn
import jax
import logging

from jax.ops import segment_sum
from functools import partial
from typing import (Any, Callable, Dict, Sequence)
from itertools import chain

from src.nn.base.sub_module import BaseSubModule
from src.masking.mask import safe_scale, safe_mask
from src.nn.fast_attention import make_fast_generalized_attention
from src.nn.mlp import MLP
from src.nn.activation_function.activation_function import get_activation_fn
from src.nn.activation_function.activation_function import silu
from src.cutoff_function import polynomial_cutoff_fn
from src.sph_ops.contract import init_clebsch_gordan_matrix, init_expansion_fn
from src.sph_ops import make_l0_contraction_fn
from src.geometric.sphc_metric import euclidean


class So3kratesLayer(BaseSubModule):
    fb_rad_filter_features: Sequence[int]
    gb_rad_filter_features: Sequence[int]
    fb_sph_filter_features: Sequence[int]
    gb_sph_filter_features: Sequence[int]
    degrees: Sequence[int]
    fb_attention: str = 'conv_att'
    gb_attention: str = 'conv_att'
    fb_filter: str = 'radial_spherical'
    gb_filter: str = 'radial_spherical'
    n_heads: int = 4
    final_layer: bool = False
    non_local_sphc: bool = False
    non_local_feature: bool = False
    fast_attention_kwargs: Dict = None
    chi_cut: float = None
    chi_cut_dynamic: bool = False
    parity: bool = True
    layer_normalization: bool = False
    sphc_normalization: bool = False
    neighborhood_normalization: bool = False
    module_name: str = 'so3krates_layer'

    def setup(self):
        if self.chi_cut:
            self.chi_cut_fn = partial(polynomial_cutoff_fn,
                                      r_cut=self.chi_cut,
                                      p=6)
        elif self.chi_cut_dynamic:
            self.chi_cut_fn = partial(polynomial_cutoff_fn,
                                      p=6)
        else:
            self.chi_cut_fn = lambda y, *args, **kwargs: jnp.zeros(1)

        if self.chi_cut is not None or self.chi_cut_dynamic is True:
            logging.warning('Localization in SPHC is used: Make sure that your neighborhood lists are global. Future'
                            ' implementation will work with two different index lists, but for now make sure you pass '
                            'global neighborhood lists for things to work correctly.')

        if self.chi_cut is not None and self.chi_cut_dynamic is True:
            msg = "chi_cut_dynamic is set to True and a manual chi_cut is specified. Use only one of the two."
            raise ValueError(msg)

        if self.neighborhood_normalization:
            def neigh_normalization_fn(x, point_mask, idx_i, phi_r_cut):
                c = segment_sum(phi_r_cut, segment_ids=idx_i, num_segments=len(x))[:, None]  # shape: (n,1)
                x_normalized = safe_mask(mask=c != 0,
                                         operand=x,
                                         fn=lambda y: y / c,
                                         placeholder=0)  # shape: (n,m_tot)
                return safe_scale(x_normalized, scale=point_mask[:, None] != 0)
        else:
            neigh_normalization_fn = lambda x, *args, **kwargs: x

        self.neigh_normalization_fn = neigh_normalization_fn

        _fast_attention = get_default_fast_attention_kwargs()
        if self.fast_attention_kwargs is not None:
            _fast_attention.update(self.fast_attention_kwargs)
        self.fast_attention = _fast_attention

        self.contraction_fn = make_l0_contraction_fn(self.degrees)

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 chi: jnp.ndarray,
                 rbf_ij: jnp.ndarray,
                 sph_ij: jnp.ndarray,
                 phi_r_cut: jnp.ndarray,
                 idx_i: jnp.ndarray,
                 idx_j: jnp.ndarray,
                 pair_mask: jnp.ndarray,
                 point_mask: jnp.ndarray,
                 *args,
                 **kwargs):
        """

        Args:
            x (Array): Atomic features, shape: (n,F)
            chi (Array): Spherical harmonic coordinates, shape: (n,m_tot)
            rbf_ij (Array): RBF expanded distances, shape: (n_pairs,K)
            sph_ij (Array): Spherical harmonics from i to j, shape: (n_pairs,m_tot)
            phi_r_cut (Array): Output of the cutoff function feature block, shape: (n_pairs)
            idx_i (Array): index centering atom, shape: (n_pairs)
            idx_j (Array): index neighboring atom, shape: (n_pairs)
            pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pairs)
            point_mask (Array): index based mask to exclude nodes that come from padding, shape: (n)
            *args ():
            **kwargs ():

        Returns:

        """
        self.sow('record', 'chi_in', chi)

        chi_ij = safe_scale(jax.vmap(lambda i, j: chi[j] - chi[i])(idx_i, idx_j),
                            scale=pair_mask[:, None])  # shape: (P,m_tot)

        m_chi_ij = self.contraction_fn(chi_ij)  # shape: (P,|l|)

        def segment_softmax(y):
            y_ = safe_scale(y - jax.ops.segment_max(y, segment_ids=idx_i, num_segments=x.shape[0])[idx_i],
                            scale=pair_mask,
                            placeholder=0)
            a = jnp.exp(y_)
            b = segment_sum(jnp.exp(y_), segment_ids=idx_i, num_segments=x.shape[0])
            return a / b[idx_i]

        if self.chi_cut_dynamic:
            d_chi_ij = safe_scale(euclidean(chi, idx_i=idx_i, idx_j=idx_j).squeeze(axis=-1),
                                  scale=pair_mask)  # shape: (n_pairs)
            n_atoms = point_mask.sum(axis=-1)
            r_cut = 1. / n_atoms
            phi_chi_cut = safe_scale(self.chi_cut_fn(segment_softmax(d_chi_ij), r_cut=r_cut),
                                     scale=pair_mask)
            phi_chi_cut = safe_mask(d_chi_ij > 0, fn=lambda u: u, operand=phi_chi_cut, placeholder=0)  # shape: (P)
        else:
            d_chi_ij = safe_scale(euclidean(chi, idx_i=idx_i, idx_j=idx_j).squeeze(axis=-1),
                                  scale=pair_mask)  # shape: (n_pairs)
            phi_chi_cut = safe_scale(self.chi_cut_fn(segment_softmax(d_chi_ij)),
                                     scale=pair_mask)  # shape: (n_pairs)
            phi_chi_cut = safe_mask(d_chi_ij > 0, fn=lambda u: u, operand=phi_chi_cut, placeholder=0)  # shape: (P)

        # pre layer-normalization
        if self.layer_normalization:
            x_pre_1 = safe_mask(point_mask[:, None] != 0, fn=nn.LayerNorm(), operand=x)
        else:
            x_pre_1 = x

        x_local = FeatureBlock(filter=self.fb_filter,
                               rad_filter_features=self.fb_rad_filter_features,
                               sph_filter_features=self.fb_sph_filter_features,
                               attention=self.fb_attention,
                               n_heads=self.n_heads)(x=x_pre_1,
                                                     rbf_ij=rbf_ij,
                                                     d_chi_ij_l=m_chi_ij,
                                                     phi_r_cut=phi_r_cut,
                                                     idx_i=idx_i,
                                                     idx_j=idx_j,
                                                     pair_mask=pair_mask)  # shape: (n,F)

        chi_local = GeometricBlock(filter=self.gb_filter,
                                   rad_filter_features=self.gb_rad_filter_features,
                                   sph_filter_features=self.gb_sph_filter_features,
                                   attention=self.gb_attention,
                                   degrees=self.degrees)(chi=chi,
                                                         sph_ij=sph_ij,
                                                         x=x_pre_1,
                                                         rbf_ij=rbf_ij,
                                                         d_chi_ij_l=m_chi_ij,
                                                         phi_r_cut=phi_r_cut,
                                                         phi_chi_cut=phi_chi_cut,
                                                         idx_i=idx_i,
                                                         idx_j=idx_j,
                                                         pair_mask=pair_mask)  # shape: (n,m_tot)

        x_local = self.neigh_normalization_fn(x_local,
                                              point_mask=point_mask,
                                              idx_i=idx_i,
                                              phi_r_cut=phi_r_cut)  # shape: (n,F)

        chi_local = self.neigh_normalization_fn(chi_local,
                                                point_mask=point_mask,
                                                idx_i=idx_i,
                                                phi_r_cut=phi_r_cut)  # shape: (n,m_tot)

        # global attention updates, using the FAVOR+ algorithm
        if self.non_local_feature:
            x_non_local = NonLocalFeatureBlock(**self.fast_attention)(x=x_pre_1)
        else:
            x_non_local = jnp.float32(0.)

        if self.non_local_sphc:
            chi_non_local = NonLocalGeometricBlock(**self.fast_attention)(chi=chi, x=x_pre_1)
        else:
            chi_non_local = jnp.float32(0.)

        # add local and potential non local features and sphc, respectively and first skip connection
        x_skip_1 = x + x_local + x_non_local
        chi_skip_1 = chi + chi_local + chi_non_local

        # x_skip_1 = nn.Dense(x_skip_1.shape[-1])(x_skip_1)  # shape: (n,F)

        # second pre layer-normalization
        if self.layer_normalization:
            x_pre_2 = safe_mask(point_mask[:, None] != 0, fn=nn.LayerNorm(), operand=x_skip_1)
        else:
            x_pre_2 = x_skip_1

        # feature <-> sphc interaction layer
        delta_x, delta_chi = InteractionBlock(self.degrees,
                                              parity=self.parity)(x_pre_2, chi_skip_1, point_mask)

        # second skip connection
        x_skip_2 = (x_skip_1 + delta_x)
        chi_skip_2 = (chi_skip_1 + delta_chi)

        # in the final layer apply post layer-normalization
        if self.final_layer:
            if self.layer_normalization:
                x_skip_2 = safe_mask(point_mask[:, None] != 0, fn=nn.LayerNorm(), operand=x_skip_2)
            else:
                x_skip_2 = x_skip_2

        self.sow('record', 'chi_out', chi_skip_2)

        return {'x': x_skip_2, 'chi': chi_skip_2}

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'fb_filter': self.fb_filter,
                                   'fb_rad_filter_features': self.fb_rad_filter_features,
                                   'fb_sph_filter_features': self.fb_sph_filter_features,
                                   'fb_attention': self.fb_attention,
                                   'gb_filter': self.gb_filter,
                                   'gb_rad_filter_features': self.gb_rad_filter_features,
                                   'gb_sph_filter_features': self.gb_sph_filter_features,
                                   'gb_attention': self.gb_attention,
                                   'n_heads': self.n_heads,
                                   'non_local_sphc': self.non_local_sphc,
                                   'non_local_feature': self.non_local_feature,
                                   'fast_attention_kwargs': self.fast_attention_kwargs,
                                   'chi_cut': self.chi_cut,
                                   'chi_cut_dynamic': self.chi_cut_dynamic,
                                   'degrees': self.degrees,
                                   'parity': self.parity,
                                   'layer_normalization': self.layer_normalization,
                                   'sphc_normalization': self.sphc_normalization,
                                   'neighborhood_normalization': self.neighborhood_normalization,
                                   'final_layer': self.final_layer
                                   }
                }


class FeatureBlock(nn.Module):
    filter: str
    rad_filter_features: Sequence[int]
    sph_filter_features: Sequence[int]
    attention: str
    n_heads: int

    def setup(self):
        if self.filter == 'radial':
            self.filter_fn = InvariantFilter(n_heads=1,
                                             features=self.rad_filter_features,
                                             activation_fn=silu)
        elif self.filter == 'radial_spherical':
            self.filter_fn = RadialSphericalFilter(rad_n_heads=1,
                                                   rad_features=self.rad_filter_features,
                                                   sph_n_heads=1,
                                                   sph_features=self.sph_filter_features,
                                                   activation_fn=silu)
        else:
            msg = "Filter argument `{}` is not a valid value.".format(self.filter)
            raise ValueError(msg)

        if self.attention == 'conv_att':
            self.attention_fn = ConvAttention(n_heads=self.n_heads)
        elif self.attention == 'self_att':
            self.attention_fn = SelfAttention(n_heads=self.n_heads)
        else:
            msg = "Attention argument `{}` is not a valid value.".format(self.filter)
            raise ValueError(msg)

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 rbf_ij: jnp.ndarray,
                 d_chi_ij_l: jnp.ndarray,
                 phi_r_cut: jnp.ndarray,
                 idx_i: jnp.ndarray,
                 idx_j: jnp.ndarray,
                 pair_mask: jnp.ndarray,
                 *args,
                 **kwargs):
        """

        Args:
            x (Array): Atomic features, shape: (n,F)
            rbf_ij (Array): RBF expanded distances, shape: (n_pairs,K)
            d_chi_ij_l (Array): Per degree distances of SPHCs, shape: (n_all_pairs,|L|)
            phi_r_cut (Array): Output of the cutoff function, shape: (n_pairs)
            idx_i (Array): index centering atom, shape: (n_pairs)
            idx_j (Array): index neighboring atom, shape: (n_pairs)
            pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pairs)
            *args ():
            **kwargs ():

        Returns:

        """
        w_ij = self.filter_fn(rbf=rbf_ij, d_gamma=d_chi_ij_l)  # shape: (n_pairs,F)
        x_ = self.attention_fn(x=x,
                               w_ij=w_ij,
                               phi_r_cut=phi_r_cut,
                               idx_i=idx_i,
                               idx_j=idx_j,
                               pair_mask=pair_mask)  # shape: (n,F)
        return x_


class GeometricBlock(nn.Module):
    degrees: Sequence[int]
    filter: str
    rad_filter_features: Sequence[int]
    sph_filter_features: Sequence[int]
    attention: str

    def setup(self):
        if self.filter == 'radial':
            self.filter_fn = InvariantFilter(n_heads=1,
                                             features=self.rad_filter_features,
                                             activation_fn=silu)
        elif self.filter == 'radial_spherical':
            self.filter_fn = RadialSphericalFilter(rad_n_heads=1,
                                                   rad_features=self.rad_filter_features,
                                                   sph_n_heads=1,
                                                   sph_features=self.sph_filter_features,
                                                   activation_fn=silu)
        else:
            msg = "Filter argument `{}` is not a valid value.".format(self.filter)
            raise ValueError(msg)

        if self.attention == 'conv_att':
            self.attention_fn = SphConvAttention(n_heads=len(self.degrees), harmonic_orders=self.degrees)
        elif self.attention == 'self_att':
            self.attention_fn = SphSelfAttention(n_heads=len(self.degrees), harmonic_orders=self.degrees)
        else:
            msg = "Attention argument `{}` is not a valid value.".format(self.filter)
            raise ValueError(msg)

    @nn.compact
    def __call__(self,
                 chi: jnp.ndarray,
                 sph_ij: jnp.ndarray,
                 x: jnp.ndarray,
                 rbf_ij: jnp.ndarray,
                 d_chi_ij_l: jnp.ndarray,
                 phi_r_cut: jnp.ndarray,
                 phi_chi_cut: jnp.ndarray,
                 idx_i: jnp.ndarray,
                 idx_j: jnp.ndarray,
                 pair_mask: jnp.ndarray,
                 *args,
                 **kwargs):
        """

        Args:
            chi (array): spherical coordinates for all orders l, shape: (n,m_tot)
            sph_ij (array): spherical harmonics for all orders l, shape: (n_all_pairs,n,m_tot)
            x (array): atomic embeddings, shape: (n,F)
            rbf_ij (array): radial basis expansion of distances, shape: (n_pairs,K)
            d_chi_ij_l (array): pairwise distance between spherical coordinates, shape: (n_all_pairs,|L|)
            phi_r_cut (array): filter cutoff, shape: (n_pairs,L)
            phi_chi_cut (array): cutoff that scales filter values based on distance in Spherical space,
                shape: (n_all_pairs,|L|)
            idx_i (Array): index centering atom, shape: (n_pairs)
            idx_j (Array): index neighboring atom, shape: (n_pairs)
            pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pairs)
            *args ():
            **kwargs ():

        Returns:

        """
        w_ij = safe_scale(self.filter_fn(rbf=rbf_ij, d_gamma=d_chi_ij_l),
                          scale=pair_mask[:, None])  # shape: (n_pairs,F)
        chi_ = self.attention_fn(chi=chi,
                                 sph_ij=sph_ij,
                                 x=x,
                                 w_ij=w_ij,
                                 phi_r_cut=phi_r_cut,
                                 phi_chi_cut=phi_chi_cut,
                                 idx_i=idx_i,
                                 idx_j=idx_j,
                                 pair_mask=pair_mask)  # shape: (n,m_tot)
        return chi_  # shape: (n,m_tot)


class NonLocalFeatureBlock(nn.Module):
    renormalize_attention: bool
    numerical_stabilizer: float
    nb_features: int
    features_type: str
    kernel_fn: str
    kernel_epsilon: float
    redraw_features: bool
    unidirectional: bool
    lax_scan_unroll: int

    def setup(self) -> None:
        self._kernel_fn = get_activation_fn(self.kernel_fn)

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        nb_features = x.shape[-1] if self.nb_features is None else self.nb_features
        fast_attention_fn = make_fast_generalized_attention(qkv_dim=x.shape[-1],
                                                            renormalize_attention=self.renormalize_attention,
                                                            nb_features=nb_features,
                                                            features_type=self.features_type,
                                                            kernel_fn=self._kernel_fn,
                                                            kernel_epsilon=self.kernel_epsilon,
                                                            numerical_stabilizer=self.numerical_stabilizer,
                                                            redraw_features=self.redraw_features,
                                                            unidirectional=self.unidirectional,
                                                            lax_scan_unroll=self.lax_scan_unroll
                                                            )

        q = nn.Dense(x.shape[-1])(x)[None, :, None, :]  # shape: (1,n,1,F)
        k = nn.Dense(x.shape[-1])(x)[None, :, None, :]  # shape: (1,n,1,F)
        v = nn.Dense(x.shape[-1])(x)[None, :, None, :]  # shape: (1,n,1,F)

        x_attended = fast_attention_fn(q, k, v)  # shape: (1,n,1,F)
        x_attended = x_attended[0, :, 0, :]  # shape: (n,F)
        return x_attended


class NonLocalGeometricBlock(nn.Module):
    renormalize_attention: bool
    numerical_stabilizer: float
    nb_features: int
    features_type: str
    kernel_fn: str
    kernel_epsilon: float
    redraw_features: bool
    unidirectional: bool
    lax_scan_unroll: int

    def setup(self) -> None:
        self._kernel_fn = get_activation_fn(self.kernel_fn)

    @nn.compact
    def __call__(self, chi, x, *args, **kwargs):
        nb_features = x.shape[-1] if self.nb_features is None else self.nb_features
        fast_attention_fn = make_fast_generalized_attention(qkv_dim=x.shape[-1],
                                                            renormalize_attention=self.renormalize_attention,
                                                            nb_features=nb_features,
                                                            features_type=self.features_type,
                                                            kernel_fn=self._kernel_fn,
                                                            kernel_epsilon=self.kernel_epsilon,
                                                            numerical_stabilizer=self.numerical_stabilizer,
                                                            redraw_features=self.redraw_features,
                                                            unidirectional=self.unidirectional,
                                                            lax_scan_unroll=self.lax_scan_unroll
                                                            )

        q = nn.Dense(x.shape[-1])(x)[None, :, None, :]  # shape: (1,n,1,F)
        k = nn.Dense(x.shape[-1])(x)[None, :, None, :]  # shape: (1,n,1,F)
        v = chi[None, :, None, :]  # shape: (1,n,1,m_tot)

        chi_attended = fast_attention_fn(q, k, v)  # shape: (1,n,1,m_tot)
        chi_attended = chi_attended[0, :, 0, :]  # shape: (n,m_tot)
        return chi_attended


class InteractionBlock(nn.Module):
    degrees: Sequence[int]
    parity: bool

    def setup(self):
        segment_ids = jnp.array(
            [y for y in chain(*[[n] * (2 * self.degrees[n] + 1) for n in range(len(self.degrees))])])
        num_segments = len(self.degrees)
        self.v_segment_sum = jax.vmap(partial(segment_sum, segment_ids=segment_ids, num_segments=num_segments))
        self.selfmix = SelfMixLayer(self.degrees, parity=self.parity)

        _repeats = [2 * y + 1 for y in self.degrees]
        self.repeat_fn = partial(jnp.repeat, repeats=jnp.array(_repeats), axis=-1, total_repeat_length=sum(_repeats))

        self.contraction_fn = make_l0_contraction_fn(degrees=self.degrees)

    @nn.compact
    def __call__(self, x, chi, point_mask, *args, **kwargs):
        """

        Args:
            x (): shape: (n,F)
            chi (Array): shape: (n,m_tot)
            *args ():
            **kwargs ():

        Returns:

        """
        F = x.shape[-1]
        nl = len(self.degrees)

        d_chi = self.contraction_fn(chi)  # shape: (n,|l|)

        y = jnp.concatenate([x, d_chi], axis=-1)  # shape: (n,F+|l|)
        a1, b1 = jnp.split(MLP(features=[int(F + nl)],
                               activation_fn=silu)(y),
                           indices_or_sections=[F], axis=-1)
        # shape: (n,F) / shape: (n,n_l) / shape: (n,n_l)
        return a1, self.repeat_fn(b1) * chi  # + self.repeat_fn(b2) * chi_nl


class InvariantFilter(nn.Module):
    n_heads: int
    features: Sequence[int]
    activation_fn: Callable = silu

    def setup(self):
        assert self.features[-1] % self.n_heads == 0

        f_out = int(self.features[-1] / self.n_heads)
        self._features = [*self.features[:-1], f_out]
        self.filter_fn = nn.vmap(MLP,
                                 in_axes=None, out_axes=-2,
                                 axis_size=self.n_heads,
                                 variable_axes={'params': 0},
                                 split_rngs={'params': True}
                                 )

    @nn.compact
    def __call__(self, rbf, *args, **kwargs):
        """
        Filter build from invariant geometric features.

        Args:
            rbf (Array): pairwise geometric features, shape: (...,K)
            *args ():
            **kwargs ():

        Returns: filter values, shape: (...,F)

        """
        w = self.filter_fn(self._features, self.activation_fn)(rbf)  # shape: (...,n_heads,F_head)
        w = w.reshape(*rbf.shape[:-1], -1)  # shape: (...,n,F)
        return w


class RadialSphericalFilter(nn.Module):
    rad_n_heads: int
    rad_features: Sequence[int]
    sph_n_heads: int
    sph_features: Sequence[int]
    activation_fn: Callable = silu

    def setup(self):
        assert self.rad_features[-1] % self.rad_n_heads == 0
        assert self.sph_features[-1] % self.sph_n_heads == 0

        f_out_rad = int(self.rad_features[-1] / self.rad_n_heads)
        f_out_sph = int(self.sph_features[-1] / self.sph_n_heads)

        self._rad_features = [*self.rad_features[:-1], f_out_rad]
        self._sph_features = [*self.sph_features[:-1], f_out_sph]

        self.rad_filter_fn = nn.vmap(MLP,
                                     in_axes=None, out_axes=-2,
                                     axis_size=self.rad_n_heads,
                                     variable_axes={'params': 0},
                                     split_rngs={'params': True}
                                     )

        self.sph_filter_fn = nn.vmap(MLP,
                                     in_axes=None, out_axes=-2,
                                     axis_size=self.sph_n_heads,
                                     variable_axes={'params': 0},
                                     split_rngs={'params': True}
                                     )

    @nn.compact
    def __call__(self, rbf, d_gamma, *args, **kwargs):
        """
        Filter build from invariant geometric features.

        Args:
            rbf (Array): pairwise, radial basis expansion, shape: (...,K)
            d_gamma (Array): pairwise distance of spherical coordinates, shape: (...,n_l)
            *args ():
            **kwargs ():

        Returns: filter values, shape: (...,F)

        """
        w = self.rad_filter_fn(self._rad_features, self.activation_fn)(rbf)  # shape: (...,n_heads,F_head)
        w += self.sph_filter_fn(self._sph_features, self.activation_fn)(d_gamma)  # shape: (...,n_heads,F_head)
        w = w.reshape(*rbf.shape[:-1], -1)  # shape: (...,n,n,F)
        return w


class SelfMixLayer(nn.Module):
    """
    SelfMix layer as implemented in PhisNet but only trainable scalars.
    """
    harmonic_orders: Sequence[int]
    parity: bool

    def setup(self):
        _l_out_max = max(self.harmonic_orders)
        _cg = init_clebsch_gordan_matrix(degrees=self.harmonic_orders, l_out_max=_l_out_max)
        self.expansion_fn = init_expansion_fn(degrees=self.harmonic_orders, cg=_cg)

        _nl = len(self.harmonic_orders)
        _repeats = [2 * y + 1 for y in self.harmonic_orders]
        self.repeat_fn = partial(jnp.repeat, repeats=jnp.array(_repeats), axis=-3, total_repeat_length=sum(_repeats))
        self.coefficients = self.param('params', nn.initializers.normal(stddev=.1), (_nl, _nl, _nl))
        # shape: (n_l,n_l,n_l)
        if self.parity:
            p = (-1) ** jnp.arange(min(self.harmonic_orders), max(self.harmonic_orders) + 1)
            pxp = jnp.einsum('i,j->ij', p, p)[None, ...]  # shape: (1, n_l, n_l)
            lxl_parity_filter = (pxp == p[..., None, None])  # shape: (n_l, n_l, n_l)
            self.parity_filter = jnp.repeat(lxl_parity_filter,
                                            repeats=jnp.array(_repeats),
                                            axis=0,
                                            total_repeat_length=sum(_repeats)).astype(int)
            # shape: (m_tot, n_l, n_l)
        else:
            self.parity_filter = jnp.ones((sum(_repeats), _nl, _nl))

    def __call__(self, gamma, *args, **kwargs):
        """
        Spherical non-linear layer which mixes all valid combinations of the harmonic orders.

        Args:
            gamma (Array): spherical coordinates, shape: (n,m_tot)
            *args ():
            **kwargs ():

        Returns:

        """
        gxg = jnp.einsum('...nm, ...nl -> ...nml', gamma, gamma)[:, None, :, :]  # shape: (n, 1, m_tot, m_tot)
        jm = self.expansion_fn(gxg) * self.parity_filter  # shape: (n,m_tot,n_l,n_l)
        jm = jm * self.repeat_fn(self.coefficients)  # shape: (n,m_tot,n_l,n_l)
        gamma_ = jnp.triu(jm, k=1).sum(axis=(-2, -1))
        return gamma_


def equal_head_split(x: jnp.ndarray, n_heads: int) -> (Callable, jnp.ndarray):
    def inv_split(inputs):
        return inputs.reshape(*x.shape[:-1], -1)

    return inv_split, x.reshape(*x.shape[:-1], n_heads, -1)


class SelfAttention(nn.Module):
    n_heads: int

    def setup(self):
        self.coeff_fn = nn.vmap(AttentionCoefficients,
                                in_axes=(-2, None, None), out_axes=-1,
                                axis_size=self.n_heads,
                                variable_axes={'params': 0},
                                split_rngs={'params': True}
                                )
        self.aggregate_fn = nn.vmap(AttentionAggregation,
                                    in_axes=(-2, -1, None, None), out_axes=-2,
                                    axis_size=self.n_heads,
                                    variable_axes={'params': 0},
                                    split_rngs={'params': True}
                                    )

    @nn.compact
    def __call__(self, x, phi_r_cut, idx_i, idx_j, pair_mask, *args, **kwargs):
        """

        Args:
            x (Array): atomic embeddings, shape: (n,F)
            phi_r_cut (Array): cutoff that scales attention coefficients, shape: (n_pairs)
            idx_i (Array): index centering atom, shape: (n_pairs)
            idx_j (Array): index neighboring atom, shape: (n_pairs)
            pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pairs)
            kwargs:

        Returns:

        """
        inv_head_split, x_heads = equal_head_split(x, n_heads=self.n_heads)  # shape: (n,n_heads,F_head)
        alpha = self.coeff_fn()(x_heads, idx_j, idx_j)  # shape: (n_pairs,n_heads)
        alpha = safe_scale(alpha, scale=pair_mask[:, None] * phi_r_cut[:, None])  # shape: (n_pairs,n_heads)
        # Note: here is scaling with pair_mask not really necessary, since phi_r_cut has been already scaled by it.
        #       However, for completeness, we do it here again.

        # save attention values for later analysis
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self.sow('record', 'alpha', alpha)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        x_ = inv_head_split(self.aggregate_fn()(x_heads, alpha, idx_i, idx_j))  # shape: (n,F)
        return x_


class ConvAttention(nn.Module):
    n_heads: int

    def setup(self):
        self.coeff_fn = nn.vmap(ConvAttentionCoefficients,
                                in_axes=(-2, -2, None, None), out_axes=-1,
                                axis_size=self.n_heads,
                                variable_axes={'params': 0},
                                split_rngs={'params': True}
                                )

        self.aggregate_fn = nn.vmap(AttentionAggregation,
                                    in_axes=(-2, -1, None, None), out_axes=-2,
                                    axis_size=self.n_heads,
                                    variable_axes={'params': 0},
                                    split_rngs={'params': True}
                                    )

    @nn.compact
    def __call__(self, x, w_ij, phi_r_cut, idx_i, idx_j, pair_mask, *args, **kwargs):
        """

        Args:
            x (Array): atomic embeddings, shape: (n,F)
            w_ij (Array): filter, shape: (n_pairs,F)
            phi_r_cut (Array): cutoff that scales attention coefficients, shape: (n_pairs)

        Returns:

        """
        inv_x_head_split, x_heads = equal_head_split(x, n_heads=self.n_heads)  # shape: (n,n_heads,F_head)
        _, w_heads = equal_head_split(w_ij, n_heads=self.n_heads)  # shape: (n_pairs,n_heads,F_head)
        alpha = self.coeff_fn()(x_heads, w_heads, idx_i, idx_j)  # shape: (n_pairs,n_heads)
        alpha = safe_scale(alpha, scale=pair_mask[:, None] * phi_r_cut[:, None])  # shape: (n_pairs,n_heads)

        # save attention values for later analysis
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self.sow('record', 'alpha', alpha)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        x_ = inv_x_head_split(self.aggregate_fn()(x_heads, alpha, idx_i, idx_j))  # shape: (n,F)
        return x_


class SphSelfAttention(nn.Module):
    n_heads: int
    harmonic_orders: Sequence[int]

    def setup(self):
        _repeats = [2 * y + 1 for y in self.harmonic_orders]
        self.repeat_fn = partial(jnp.repeat, repeats=jnp.array(_repeats), axis=-1, total_repeat_length=sum(_repeats))
        self.coeff_fn = nn.vmap(AttentionCoefficients,
                                in_axes=(-2, None, None), out_axes=-1,
                                axis_size=self.n_heads,
                                variable_axes={'params': 0},
                                split_rngs={'params': True}
                                )

    @nn.compact
    def __call__(self,
                 chi: jnp.ndarray,
                 sph_ij: jnp.ndarray,
                 x: jnp.ndarray,
                 phi_r_cut: jnp.ndarray,
                 phi_chi_cut: jnp.ndarray,
                 idx_i: jnp.ndarray,
                 idx_j: jnp.ndarray,
                 pair_mask: jnp.ndarray,
                 *args,
                 **kwargs) -> jnp.ndarray:
        """

        Args:
            chi (Array): spherical coordinates for all degrees l, shape: (n,m_tot)
            sph_ij (Array): spherical harmonics for all degrees l, shape: (n_pairs,m_tot)
            x (Array): atomic embeddings, shape: (n,F)
            phi_r_cut (Array): cutoff that scales attention coefficients, shape: (n_pairs)
            phi_chi_cut (Array): cutoff that scales filter values based on distance in Spherical space,
                shape: (n_pairs,n_l)
            idx_i (Array): index centering atom, shape: (n_pairs)
            idx_j (Array): index neighboring atom, shape: (n_pairs)
            pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pairs)
            args:
            kwargs:

        Returns:

        """

        # number of heads equals number of degrees, i.e. n_heads = n_l
        inv_head_split, x_heads = equal_head_split(x, n_heads=self.n_heads)  # shape: (n,n_heads,F_head)
        alpha_ij = self.coeff_fn()(x_heads, idx_i, idx_j)  # shape: (n_pairs,n_heads)
        alpha_r_ij = safe_scale(alpha_ij, scale=pair_mask[:, None] * phi_r_cut[:, None])  # shape: (n_pairs,n_heads)
        alpha_s_ij = safe_scale(alpha_ij, scale=pair_mask[:, None] * phi_chi_cut[:, None])  # shape: (n_pairs,n_heads)
        alpha_ij = alpha_r_ij + alpha_s_ij  # shape: (n_pairs,n_heads)

        # save attention values for later analysis
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self.sow('record', 'alpha_r', alpha_r_ij)
        self.sow('record', 'alpha_s', alpha_s_ij)
        self.sow('record', 'alpha', alpha_ij)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        alpha_ij = self.repeat_fn(alpha_ij)  # shape: (n_pairs,m_tot)
        chi_ = segment_sum(alpha_ij * sph_ij, segment_ids=idx_i, num_segments=x.shape[0])  # shape: (n,m_tot)
        return chi_


class SphConvAttention(nn.Module):
    n_heads: int
    harmonic_orders: Sequence[int]

    def setup(self):
        _repeats = [2 * y + 1 for y in self.harmonic_orders]
        self.repeat_fn = partial(jnp.repeat, repeats=jnp.array(_repeats), axis=-1, total_repeat_length=sum(_repeats))
        self.coeff_fn = nn.vmap(ConvAttentionCoefficients,
                                in_axes=(-2, -2, None, None), out_axes=-1,
                                axis_size=self.n_heads,
                                variable_axes={'params': 0},
                                split_rngs={'params': True}
                                )

    @nn.compact
    def __call__(self, chi, sph_ij, x, w_ij, phi_r_cut, phi_chi_cut, idx_i, idx_j, pair_mask, *args, **kwargs):
        """

        Args:
            chi (Array): spherical coordinates for all degrees l, shape: (n,m_tot)
            sph_ij (Array): spherical harmonics for all degrees l, shape: (n_pairs,m_tot)
            x (Array): atomic embeddings, shape: (n,F)
            w_ij (Array): filter, shape: (n_pairs,F)
            phi_r_cut (Array): cutoff that scales attention coefficients, shape: (n_pairs)
            phi_chi_cut (Array): cutoff that scales filter values based on distance in spherical space,
                shape: (n_pairs,n_l)
            idx_i (Array): index centering atom, shape: (n_pairs)
            idx_j (Array): index neighboring atom, shape: (n_pairs)
            pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pairs)
            args:
            kwargs:

        Returns:

        """

        # number of heads equals number of harmonics, i.e. n_heads = n_l
        inv_x_head_split, x_heads = equal_head_split(x, n_heads=self.n_heads)  # shape: (n,n_heads,F_head)
        _, w_ij_heads = equal_head_split(w_ij, n_heads=self.n_heads)  # shape: (n_pairs,n_heads,F_head)
        alpha_ij = self.coeff_fn()(x_heads, w_ij_heads, idx_i, idx_j)  # shape: (n_pairs,n_heads)
        alpha_r_ij = safe_scale(alpha_ij, scale=pair_mask[:, None] * phi_r_cut[:, None])  # shape: (n_pairs,n_heads)
        alpha_s_ij = safe_scale(alpha_ij, scale=pair_mask[:, None] * phi_chi_cut[:, None])  # shape: (n_pairs,n_heads)
        alpha_ij = alpha_r_ij + alpha_s_ij  # shape: (n_pairs,n_heads)

        # save attention values for later analysis
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self.sow('record', 'alpha_r', alpha_r_ij)
        self.sow('record', 'alpha_s', alpha_s_ij)
        self.sow('record', 'alpha', alpha_ij)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        alpha_ij = self.repeat_fn(alpha_ij)  # shape: (n_pairs,m_tot)
        chi_ = segment_sum(alpha_ij * sph_ij, segment_ids=idx_i, num_segments=x.shape[0])  # shape: (n,m_tot)
        return chi_


class AttentionCoefficients(nn.Module):
    @nn.compact
    def __call__(self, x, idx_i, idx_j):
        """

        Args:
            x (Array): atomic embeddings, shape: (n,F)
            idx_i (Array): index centering atom, shape: (n_pairs)
            idx_j (Array): index neighboring atom, shape: (n_pairs)

        Returns: Geometric attention coefficients, shape: (n_pairs)

        """

        q_i = nn.Dense(x.shape[-1], use_bias=False)(x)[idx_i]  # shape: (n_pairs,F)
        k_j = nn.Dense(x.shape[-1], use_bias=False)(x)[idx_j]  # shape: (n_pairs,F)

        return (q_i * k_j).sum(axis=-1) / jnp.sqrt(x.shape[-1])  # shape: (n_pairs)


class ConvAttentionCoefficients(nn.Module):
    @nn.compact
    def __call__(self, x, w_ij, idx_i, idx_j):
        """

        Args:
            x (Array): atomic embeddings, shape: (n,F)
            w_ij (Array): filter, shape: (n_pairs,F)
            idx_i (Array): index centering atom, shape: (n_pairs)
            idx_j (Array): index neighboring atom, shape: (n_pairs)

        Returns: Geometric attention coefficients, shape: (n_pairs)

        """

        q_i = nn.Dense(x.shape[-1], use_bias=False)(x)[idx_i]  # shape: (n_pairs,F)
        k_j = nn.Dense(x.shape[-1], use_bias=False)(x)[idx_j]  # shape: (n_pairs,F)

        return (q_i * w_ij * k_j).sum(axis=-1) / jnp.sqrt(x.shape[-1])


class _ConvAttentionCoefficients(nn.Module):
    @nn.compact
    def __call__(self, x, w_ij, idx_i, idx_j):
        """

        Args:
            x (Array): atomic embeddings, shape: (n,F)
            w_ij (Array): filter, shape: (n_pairs,F)
            idx_i (Array): index centering atom, shape: (n_pairs)
            idx_j (Array): index neighboring atom, shape: (n_pairs)

        Returns: Geometric attention coefficients, shape: (n_pairs)

        """
        F = x.shape[-1]

        q_i = x[idx_i]  # shape: (n_pairs,F)
        k_j = x[idx_j]  # shape: (n_pairs,F)

        y = jnp.concatenate([q_i, k_j, w_ij], axis=-1)
        return MLP(features=[F, 1], activation_fn=silu)(y).squeeze(axis=-1)


class AttentionAggregation(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray,
                 alpha_ij: jnp.ndarray,
                 idx_i: jnp.ndarray,
                 idx_j: jnp.ndarray) -> jnp.ndarray:
        """

        Args:
            x (Array): atomic embeddings, shape: (n,F)
            alpha_ij (Array): attention coefficients, shape: (n_pairs)
            idx_i (Array): index centering atom, shape: (n_pairs)
            idx_j (Array): index neighboring atom, shape: (n_pairs)

        Returns:

        """

        v_j = nn.Dense(x.shape[-1], use_bias=False)(x)[idx_j]  # shape: (n_pairs,F)
        return segment_sum(alpha_ij[:, None] * v_j, segment_ids=idx_i, num_segments=x.shape[0])  # shape: (n,F)


def get_default_fast_attention_kwargs() -> Dict:
    return {'renormalize_attention': True,
            'numerical_stabilizer': 0.0,
            'nb_features': None,
            'features_type': 'ortho',
            'kernel_fn': 'silu',
            'kernel_epsilon': 0.001,
            'redraw_features': False,
            'unidirectional': False,
            'lax_scan_unroll': 16}