# Taken from mlff (https://github.com/thorben-frank/mlff, commit 99dbf76)
# Original author: Thorben Frank et al.
# Modifications: renamed imports (mlff.src → src)

import numpy as np


def set_seeds(seed=0):
    try:
        random.seed(seed)
    except NameError:
        import random
        random.seed(seed)

    np.random.seed(seed)
