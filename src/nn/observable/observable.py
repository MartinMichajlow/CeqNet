# Adapted from mlff (https://github.com/thorben-frank/mlff, commit 99dbf76)
# Original author: Thorben Frank et al.
# Modifications: renamed imports (mlff.src → src); replaced monolithic observable classes
#                with configurable PartialCharge, Dipole, Quadrupole modules

from typing import Any, Dict, Sequence
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from jax.scipy.special import erf

from src.masking.mask import safe_scale, safe_mask
from src.nn.base.sub_module import BaseSubModule
from src.geometric.metric import coordinates_to_distance_matrix
from src.padding import pad_ceq
from src.nn.mlp import MLP
from src.nn.activation_function.activation_function import silu


def get_observable_module(name, h):
    if name == 'partial_charge':
        return PartialCharge(**h)
    elif name == 'dipole':
        return Dipole(**h)
    elif name == 'quadrupole':
        return Quadrupole(**h)
    else:
        raise ValueError(f"No observable module for name={name!r}. "
                         f"Legacy classes are on the legacy-observables branch.")


# ---------------------------------------------------------------------------
# Shared helpers (no Flax parameters)
# ---------------------------------------------------------------------------

_QUAD_IDX = ([0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2])


def _detrace(q6):
    """Remove trace from a 6-vector symmetric quadrupole [xx,yy,zz,xy,xz,yz]."""
    tr = q6[0] + q6[1] + q6[2]
    return q6 - (tr / 3) * jnp.array([1, 1, 1, 0, 0, 0])


def _charge_quad_ntl(q_i, R):
    """Non-traceless quadrupole Q̃ = Σ_i q_i R_i⊗R_i, shape (6,)."""
    Q3 = (jax.vmap(jnp.outer, in_axes=(-2, -2))(R, R) * q_i[:, None, None]).sum(axis=-3)
    return Q3[_QUAD_IDX]


def _build_ceq_A(R, J, sigma, point_mask):
    """Assemble the augmented CeQ system matrix A_aug (Ko et al. 2021 Nat. Comm., eq. 6)."""
    pair_mask = jnp.expand_dims(point_mask, 0) * jnp.expand_dims(point_mask, 1)
    r_ij = safe_scale(jnp.squeeze(coordinates_to_distance_matrix(R)), pair_mask)
    gamma_ij = safe_scale(
        jnp.sqrt(jnp.expand_dims(sigma, 0) ** 2 + jnp.expand_dims(sigma, 1) ** 2), pair_mask)
    A_off = safe_mask(mask=(gamma_ij != 0), fn=lambda g: 1 / (g * jnp.sqrt(2)),
                      operand=gamma_ij, placeholder=0)
    A_off = erf(A_off * r_ij)
    A_off = safe_mask(mask=(r_ij != 0), fn=lambda a: a / r_ij, operand=A_off, placeholder=0)
    J_diag = safe_scale(J, point_mask)
    sigma_inv = safe_mask(mask=point_mask, fn=lambda s: 1 / s, operand=sigma)
    A_diag = jnp.diag(J_diag) + jnp.diag(sigma_inv) / jnp.sqrt(jnp.pi)
    return pad_ceq(safe_scale(A_diag + A_off, pair_mask, 0), point_mask)


def _ceq_q(inputs, f, pm, chi_name):
    """Solve CeQ system → partial charges q_i. Must be called inside @nn.compact."""
    A_aug = _build_ceq_A(inputs['R'], inputs['J'], inputs['sigma'], pm)
    chi_i = MLP(features=[f.shape[-1], 1], use_bias=False, name=chi_name)(f).squeeze(-1)
    chi_i = safe_scale(chi_i, pm)
    b = jnp.pad(-chi_i, (0, 1), constant_values=(0, jnp.squeeze(inputs['Q'])))
    return safe_scale(jax.scipy.linalg.solve(A_aug, b)[:-1], pm)


def _redis_q(inputs, f, pm, q_name):
    """ReDis partial charges. Must be called inside @nn.compact."""
    q_triv = MLP(features=[f.shape[-1], 1], use_bias=False, name=q_name)(f).squeeze(-1)
    n_a = jnp.sum(pm)
    return safe_scale(q_triv - (jnp.sum(q_triv) - inputs['Q']) / n_a, pm)


def _triv_q(f, pm, q_name):
    """Triv partial charges. Must be called inside @nn.compact."""
    return safe_scale(MLP(features=[f.shape[-1], 1], name=q_name)(f).squeeze(-1), pm)


def _validate_alpha(alpha, n):
    """Check alpha is consistent with the number of tokens."""
    if alpha is None or alpha == 'learnable':
        return
    if isinstance(alpha, (list, tuple)):
        if len(alpha) != n:
            raise ValueError(
                f"alpha has {len(alpha)} entries but mode has {n} tokens")
    elif n != 2:
        raise ValueError(
            f"Scalar alpha is only valid for exactly 2 tokens, got {n}")


def _get_weights(self_obj, n):
    """Return normalized weight array of length n. Must be called inside @nn.compact."""
    alpha = self_obj.alpha
    if alpha is None:
        return jnp.ones(n) / n
    elif alpha == 'learnable':
        logits = self_obj.param('alpha_logits', nn.initializers.zeros, (n,))
        return jax.nn.softmax(logits)
    elif isinstance(alpha, (list, tuple)):
        w = jnp.array([float(a) for a in alpha])
        return w / w.sum()
    else:
        a = float(alpha)
        return jnp.array([a, 1.0 - a])


def _blend(parts, weights):
    """Weighted sum: Σ_k weights[k] * parts[k]."""
    out = parts[0] * weights[0]
    for w, p in zip(weights[1:], parts[1:]):
        out = out + w * p
    return out


# ---------------------------------------------------------------------------
# PartialCharge
# ---------------------------------------------------------------------------

class PartialCharge(BaseSubModule):
    """
    Configurable partial charge prediction.

    mode: one or more tokens from {'ceqcha', 'trivcha', 'redischa'} separated by '+'.
    alpha: blend weights — see _get_weights for semantics.
        None        → equal weights (1/N each)
        list[float] → normalized by their sum to weights summing to 1
        float       → only for 2 tokens: [alpha, 1-alpha]
        'learnable' → softmax over N trainable logits

    Outputs: {'q': Array (n,)}
    """
    mode: str
    prop_keys: Dict
    alpha: Any = None
    module_name: str = 'partial_charge'

    def setup(self):
        tokens = [t.strip() for t in self.mode.split('+')]
        bad = [t for t in tokens if t not in ('ceqcha', 'trivcha', 'redischa')]
        if bad:
            raise ValueError(f"Unknown PartialCharge token(s): {bad}")
        _validate_alpha(self.alpha, len(tokens))
        self._tokens = tokens

    @nn.compact
    def __call__(self, inputs, *args, **kwargs):
        f, pm = inputs['x'], inputs['point_mask']
        qs = []
        for tok in self._tokens:
            if tok == 'ceqcha':
                qs.append(_ceq_q(inputs, f, pm, 'ceq_chi'))
            elif tok == 'trivcha':
                qs.append(_triv_q(f, pm, 'triv_q'))
            elif tok == 'redischa':
                qs.append(_redis_q(inputs, f, pm, 'redis_q'))
        if len(qs) == 1:
            return {'q': qs[0]}
        return {'q': _blend(qs, _get_weights(self, len(qs)))}

    def __dict_repr__(self):
        return {self.module_name: {'mode': self.mode, 'alpha': self.alpha,
                                   'prop_keys': self.prop_keys}}


# ---------------------------------------------------------------------------
# Dipole
# ---------------------------------------------------------------------------

class Dipole(BaseSubModule):
    """
    Configurable dipole prediction.

    mode: one or more tokens from {'ceqdip', 'trivdip', 'redisdip', 'atomicdip'} separated by '+'.
    alpha: blend weights (see PartialCharge).
    degrees: spherical harmonic degrees in inputs['chi'] (required for 'atomicdip').

    Outputs: {'mu': Array (3,)} plus {'q': Array (n,)} when a charge-based token is used.
    """
    mode: str
    prop_keys: Dict
    degrees: Sequence[int] = None
    alpha: Any = None
    module_name: str = 'dipole'

    _ALL_TOKENS = frozenset({'ceqdip', 'trivdip', 'redisdip', 'atomicdip'})

    def setup(self):
        tokens = [t.strip() for t in self.mode.split('+')]
        bad = [t for t in tokens if t not in self._ALL_TOKENS]
        if bad:
            raise ValueError(f"Unknown Dipole token(s): {bad}")
        _validate_alpha(self.alpha, len(tokens))
        self._tokens = tokens
        if 'atomicdip' in tokens:
            if self.degrees is None:
                raise ValueError("'degrees' required when using 'atomicdip'")
            self._l1_idxs = jnp.array([1, 2, 3]) if 0 in self.degrees else jnp.array([0, 1, 2])
            # P: reorders spherical harmonics (sph. coords) to cartesian order
            self._P = jnp.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=jnp.float32)
            inv_c = 1 / jnp.sqrt(3 / (4 * jnp.pi))
            self._inv_c_l1 = jnp.array([inv_c, inv_c, inv_c])

    @nn.compact
    def __call__(self, inputs, *args, **kwargs):
        f, pm, R = inputs['x'], inputs['point_mask'], inputs['R']
        mu_parts = []
        q_out = None

        for tok in self._tokens:
            if tok == 'ceqdip':
                q_i = _ceq_q(inputs, f, pm, 'ceqdip_chi')
                mu_parts.append(jnp.sum(q_i[:, None] * R, axis=0))
                if q_out is None:
                    q_out = q_i
            elif tok == 'trivdip':
                q_i = _triv_q(f, pm, 'trivdip_q')
                mu_parts.append(jnp.sum(q_i[:, None] * R, axis=0))
                if q_out is None:
                    q_out = q_i
            elif tok == 'redisdip':
                q_i = _redis_q(inputs, f, pm, 'redisdip_q')
                mu_parts.append(jnp.sum(q_i[:, None] * R, axis=0))
                if q_out is None:
                    q_out = q_i
            elif tok == 'atomicdip':
                chi_norm = safe_mask(mask=pm[:, None],
                                     fn=partial(jnp.linalg.norm, axis=-1, keepdims=True),
                                     operand=inputs['chi'], placeholder=0)
                chi_hat = safe_mask(mask=chi_norm != 0, fn=lambda y: y / chi_norm,
                                    operand=inputs['chi'], placeholder=0)
                chi_i_l1 = chi_hat[:, self._l1_idxs] * self._inv_c_l1[None, :]
                P_chi_l1 = jnp.einsum('ij,...j->...i', self._P, chi_i_l1)
                phi_fi = MLP(features=[f.shape[-1], 1], name='atomicdip_amp')(f)
                mu_i = safe_scale(phi_fi * P_chi_l1, pm[:, None])
                mu_parts.append(mu_i.sum(axis=-2))

        mu = mu_parts[0] if len(mu_parts) == 1 else _blend(mu_parts, _get_weights(self, len(mu_parts)))
        out = {'mu': mu}
        if q_out is not None:
            out['q'] = q_out
        return out

    def __dict_repr__(self):
        return {self.module_name: {'mode': self.mode, 'degrees': list(self.degrees or []),
                                   'alpha': self.alpha, 'prop_keys': self.prop_keys}}


# ---------------------------------------------------------------------------
# Quadrupole
# ---------------------------------------------------------------------------

class Quadrupole(BaseSubModule):
    """
    Configurable quadrupole (and dipole) prediction.

    mode: one or more tokens from
          {'ceqquad', 'trivquad', 'redisquad', 'atomicquad', 'atomicdipquad'} separated by '+'.

    Token descriptions
    ------------------
    ceqquad / trivquad / redisquad:
        Non-traceless quadrupole from point charges:  Q̃ = Σ_i q_i R_i⊗R_i
        Dipole: μ = Σ_i q_i R_i

    atomicquad:
        Per-atom contribution Q̂_i = φ(f_i) Q_inv(D pad_1(χ̂_i^{l=2}) + e_3),
        summed over atoms. No dipole contribution (μ = 0).

    atomicdipquad:
        Quadrupole from atomic dipoles: Q̃ = Σ_i (R_i⊗μ_i + μ_i⊗R_i),
        where μ_i = φ(f_i) P χ̂_i^{l=1}.
        Dipole: μ = Σ_i μ_i

    alpha: blend weights (see PartialCharge).
    degrees: required when using 'atomicquad' or 'atomicdipquad'.

    All tokens are detraced before blending. Outputs:
        {'quad': Array (6,), 'quad_ntl': Array (6,), 'mu': Array (3,)}
        plus {'q': Array (n,)} when a charge-based token is used.
    """
    mode: str
    prop_keys: Dict
    degrees: Sequence[int] = None
    alpha: Any = None
    module_name: str = 'quadrupole'

    _ALL_TOKENS = frozenset({'ceqquad', 'trivquad', 'redisquad', 'atomicquad', 'atomicdipquad'})

    def setup(self):
        tokens = [t.strip() for t in self.mode.split('+')]
        bad = [t for t in tokens if t not in self._ALL_TOKENS]
        if bad:
            raise ValueError(f"Unknown Quadrupole token(s): {bad}")
        _validate_alpha(self.alpha, len(tokens))
        self._tokens = tokens

        needs_l1 = 'atomicdipquad' in tokens
        needs_l2 = 'atomicquad' in tokens
        if needs_l1 or needs_l2:
            if self.degrees is None:
                raise ValueError("'degrees' required when using 'atomicquad' or 'atomicdipquad'")
        if needs_l1:
            degs = self.degrees
            self._l1_idxs = jnp.array([1, 2, 3]) if 0 in degs else jnp.array([0, 1, 2])
            # P: reorders spherical harmonics (sph. coords) to cartesian order
            self._P = jnp.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=jnp.float32)
            inv_c = 1 / jnp.sqrt(3 / (4 * jnp.pi))
            self._inv_c_l1 = jnp.array([inv_c, inv_c, inv_c])

        if needs_l2:
            degs = self.degrees
            if (0 in degs) and (1 in degs):
                self._l2_idxs = jnp.array([4, 5, 6, 7, 8])
            elif (0 not in degs) and (1 in degs):
                self._l2_idxs = jnp.array([3, 4, 5, 6, 7])
            else:
                self._l2_idxs = jnp.array([0, 1, 2, 3, 4])
            Q = jnp.array([[0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 1],
                            [0, 0, 3, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [1, -1, 0, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0]], dtype=jnp.float32)
            D = jnp.diag(jnp.array([2 * np.sqrt(np.pi / 15), 2 * np.sqrt(np.pi / 15),
                                     4 * np.sqrt(np.pi / 5), 2 * np.sqrt(np.pi / 15),
                                     4 * np.sqrt(np.pi / 15), 1]))
            self._D = D
            self._Q_inv = jnp.linalg.inv(Q)

    def _chi_normalized(self, chi, pm):
        """Normalise the spherical feature vector chi per atom: χ̂_i = χ_i / ‖χ_i‖."""
        chi_norm = safe_mask(mask=pm[:, None],
                             fn=partial(jnp.linalg.norm, axis=-1, keepdims=True),
                             operand=chi, placeholder=0)
        return safe_mask(mask=chi_norm != 0, fn=lambda y: y / chi_norm,
                         operand=chi, placeholder=0)

    def _atomic_dipole(self, chi_hat, f, pm, amp_name):
        """Per-atom dipoles μ_i = φ(f_i) P χ̂_i^{l=1}, shape (n, 3)."""
        chi_i_l1 = chi_hat[:, self._l1_idxs] * self._inv_c_l1[None, :]
        P_chi_l1 = jnp.einsum('ij,...j->...i', self._P, chi_i_l1)
        phi_fi = MLP(features=[f.shape[-1], 1], name=amp_name)(f)
        return safe_scale(phi_fi * P_chi_l1, pm[:, None])

    @nn.compact
    def __call__(self, inputs, *args, **kwargs):
        f, pm, R = inputs['x'], inputs['point_mask'], inputs['R']
        quad_ntl_parts = []
        mu_parts = []
        q_out = None

        for tok in self._tokens:
            if tok == 'ceqquad':
                q_i = _ceq_q(inputs, f, pm, 'ceqquad_chi')
                q_out = q_i
                quad_ntl_parts.append(_charge_quad_ntl(q_i, R))
                mu_parts.append(jnp.sum(q_i[:, None] * R, axis=0))

            elif tok == 'trivquad':
                q_i = _triv_q(f, pm, 'trivquad_q')
                if q_out is None:
                    q_out = q_i
                quad_ntl_parts.append(_charge_quad_ntl(q_i, R))
                mu_parts.append(jnp.sum(q_i[:, None] * R, axis=0))

            elif tok == 'redisquad':
                q_i = _redis_q(inputs, f, pm, 'redisquad_q')
                if q_out is None:
                    q_out = q_i
                quad_ntl_parts.append(_charge_quad_ntl(q_i, R))
                mu_parts.append(jnp.sum(q_i[:, None] * R, axis=0))

            elif tok == 'atomicquad':
                # Q̂_i = Q_inv(D pad_1(χ̂_i^{l=2}) + e_3),  scaled by φ(f_i)
                chi_hat = self._chi_normalized(inputs['chi'], pm)
                chi_i_l2 = chi_hat[:, self._l2_idxs]
                pad1_chi_i_l2 = jnp.concatenate(
                    [chi_i_l2, jnp.ones((chi_i_l2.shape[0], 1))], axis=-1)   # pad_1: append 1
                D_pad1_chi_i = jnp.einsum('ij,...j->...i', self._D, pad1_chi_i_l2)
                D_pad1_chi_i = D_pad1_chi_i.at[..., 2].add(1.0)              # + e_3
                Q_hat_i = jnp.einsum('ij,...j->...i', self._Q_inv, D_pad1_chi_i)
                phi_fi = MLP(features=[f.shape[-1], 1], name='atomicquad_amp')(f)
                quad_ntl_parts.append(safe_scale(phi_fi * Q_hat_i, pm[:, None]).sum(axis=-2))
                mu_parts.append(jnp.zeros(3))

            elif tok == 'atomicdipquad':
                # Q̃ = Σ_i (R_i⊗μ_i + μ_i⊗R_i)
                chi_hat = self._chi_normalized(inputs['chi'], pm)
                mu_i = self._atomic_dipole(chi_hat, f, pm, 'atomicdipquad_amp')
                Q_tilde = (jax.vmap(jnp.outer, in_axes=(-2, -2))(mu_i, R) +
                           jax.vmap(jnp.outer, in_axes=(-2, -2))(R, mu_i))
                Q_tilde = safe_scale(Q_tilde, pm[:, None, None]).sum(axis=-3)
                quad_ntl_parts.append(Q_tilde[_QUAD_IDX])
                mu_parts.append(safe_scale(mu_i, pm[:, None]).sum(axis=-2))

        n = len(quad_ntl_parts)
        if n == 1:
            quad_ntl = quad_ntl_parts[0]
            mu = mu_parts[0]
        else:
            w = _get_weights(self, n)
            quad_ntl = _blend(quad_ntl_parts, w)
            mu = _blend(mu_parts, w)

        quad_tl = _detrace(quad_ntl)
        out = {'quad': quad_tl, 'quad_ntl': quad_ntl, 'mu': mu}
        if q_out is not None:
            out['q'] = q_out
        return out

    def __dict_repr__(self):
        return {self.module_name: {'mode': self.mode, 'degrees': list(self.degrees or []),
                                   'alpha': self.alpha, 'prop_keys': self.prop_keys}}
