# Taken from mlff (https://github.com/thorben-frank/mlff, commit 99dbf76)
# Original author: Thorben Frank et al.
# Modifications: renamed imports (mlff.src → src)

from abc import abstractmethod

import flax.linen as nn


class BaseSubModule(nn.Module):

    @abstractmethod
    def __dict_repr__(self):
        pass

    def reset_prop_keys(self, prop_keys):
        self.prop_keys.update(prop_keys)
