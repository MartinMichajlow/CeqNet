# Partially adapted from mlff (https://github.com/thorben-frank/mlff, commit 99dbf76)
# Original author (mlff): Thorben Frank et al.
# Modifications: renamed imports (mlff.src → src)
# Original contributions: pad_ceq, pad_coordinates_single, pad_partial_charges,
#   pad_partial_charges_single, pad_atomic_types_single

import numpy as np
import jax.numpy as jnp
from src.geometric.metric import coordinates_to_distance_matrix
from jax.ops import segment_sum
from src.masking import safe_scale
from functools import partial

# [Original contribution]
def pad_ceq(A, point_mask):
    """
    Filling a 0-padded matrix A from [42], eq. (6), with ones at proper positions, to imitate eq. (6) for padded atom
    vectors.

    :param A: matrix A as in [42], but padded with 0's, to have shape: (len(point_mask),len(point_mask))=(n,n)
    :param point_mask: shape (n,n)
    :return: matrix from [42] eq. (6), but for padded atom vectors, i.e. shape: (n+1,n+1), last row left to right,
    last column top to bottom: 1 (entries 1 to n_atoms), 0 (entries n_atoms+1 to n), 1 (entry n+1). Otherwise, the
    rows/columns n_atoms+1 to n are 0, except for diagonals, which are 1.
    """
    # padding last row and column with 1's and setting position (n+1,n+1) to 0
    A_ = jnp.pad(A, ((0, 1), (0, 1)), mode='constant', constant_values=((0, 1), (0, 1)))  # shape: (n+1,n+1)
    A_ = A_.at[-1, -1].set(0)  # shape: (n+1,n+1)

    # setting 0's at positions (n+1,n_atoms+1), ..., (n+1,n), and (n_atoms+1,n+1), ...,(n,n+1)
    mask_1 = jnp.pad(jnp.expand_dims(point_mask, axis=0), ((len(point_mask), 0), (0, 0)), mode='constant',
                 constant_values=((1, 0), (0, 0)))
    mask_2 = jnp.pad(point_mask, ((0, 1)), mode='constant', constant_values=((0, 0)))
    mask_2_expanded = jnp.expand_dims(mask_2, axis=1)
    mask = jnp.append(mask_1, mask_2_expanded, axis=1)
    A_ = safe_scale(A_, mask)

    # we need to regularize the Matrix A, so we just add ones on the diagonal in the rows n_atoms+1,...n
    point_mask_inv = jnp.ones(len(point_mask)) - point_mask
    point_mask_inv = jnp.pad(point_mask_inv, ((0, 1)), mode='constant', constant_values=((0, 0)))
    A_ = A_ + jnp.diag(point_mask_inv)

    return A_


def index_padding_length(R, z, r_cut):
    """
        For the coordinates of the same molecule, determine the padding length for each of the `n_data` frames given some
        cutoff radius `r_cut`. As atoms may leave or enter the neighborhood for a given atom, one can have different lengths
        for the index lists even for the same molecule. The suggested padding length is the difference between the
        maximal number of indices over the whole training set and the number of indices for each frame.

        Args:
            R (Array): Atomic coordinates, shape: (n_data,n,3)
            z (Array): Atomic types, shape: (n_data,n)
            r_cut (float): Cutoff distance

        Returns: Padding lengths, shape: (n_data)

        """
    n = R.shape[-2]
    n_data = R.shape[0]
    idx = np.indices((n_data, n, n))
    msk_ij = (np.einsum('...i, ...j -> ...ij', z, z) != 0).astype(np.int16)
    Dij = coordinates_to_distance_matrix(R).squeeze()
    idx_seg, _, _ = np.split(idx[:, np.where((msk_ij*Dij <= r_cut) & (msk_ij*Dij > 0), True, False)],
                             indices_or_sections=3,
                             axis=0)

    segment_length = segment_sum(np.ones(len(idx_seg),), segment_ids=idx_seg)
    pad_length = np.array((max(segment_length) - segment_length))
    return pad_length

def pad_coordinates(R, n_max, pad_value=0):
    n = R.shape[-2]

    pad_length = n_max - n
    assert pad_length >= 0

    return np.pad(R, ((0, 0), (0, pad_length), (0, 0)), mode='constant',
                  constant_values=((0, 0), (0, pad_value), (0, 0)))

# [Original contribution]
def pad_coordinates_single(R, n_max, pad_value=0):
    n = R.shape[-2]

    pad_length = n_max - n
    assert pad_length >= 0

    return np.pad(R, ((0, pad_length), (0, 0)), mode='constant',
                  constant_values=((0, 0), (0, pad_value)))

# [Original contribution]
def pad_partial_charges(q, n_max, pad_value=0):
    n = q.shape[-2]

    pad_length = n_max - n
    assert pad_length >= 0

    return np.pad(q, ((0, 0), (0, pad_length), (0, 0)), mode='constant', constant_values=((0, 0), (0, pad_value), (0, 0)))

# [Original contribution]
def pad_partial_charges_single(q, n_max, pad_value=0):
    n = q.shape[-1]

    pad_length = n_max - n
    assert pad_length >= 0

    return np.pad(q, ((0, pad_length)), mode='constant', constant_values=((0, pad_value)))

def pad_atomic_types(z, n_max, pad_value=0):
    n = z.shape[-1]

    pad_length = n_max - n
    assert pad_length >= 0

    return np.pad(z, ((0, 0), (0, pad_length)), mode='constant', constant_values=((0, 0), (0, pad_value)))


# [Original contribution]
def pad_atomic_types_single(z, n_max, pad_value=0):
    n = z.shape[-1]

    pad_length = n_max - n
    assert pad_length >= 0

    return np.pad(z, ((0, pad_length)), mode='constant', constant_values=((0, pad_value)))


def pad_indices(idx_i, idx_j, n_pair_max, pad_value=-1):
    n_pair = idx_i.shape[-1]
    assert idx_j.shape[-1] == n_pair

    pad_length = n_pair_max - n_pair
    assert pad_length >= 0

    pad = partial(np.pad, pad_width=((0, 0), (0, pad_length)), mode='constant',
                  constant_values=((0, 0), (0, pad_value)))
    pad_idx_i, pad_idx_j = map(pad, [idx_i, idx_j])
    return pad_idx_i, pad_idx_j