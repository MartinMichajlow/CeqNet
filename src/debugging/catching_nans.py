import jax
import numpy as np

def check_leaves_nan(p):
    """
    gives out a pytree with the same structure, for which the leaves are 0 if the contain a nan, 1 else.
    :param p:
    :return:
    """
    return jax.tree.map(lambda y: np.sum(np.isnan(np.array(y))), p)