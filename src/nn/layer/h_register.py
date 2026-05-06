# Taken from mlff (https://github.com/thorben-frank/mlff, commit 99dbf76)
# Original author: Thorben Frank et al.
# Modifications: renamed imports (mlff.src → src)

from typing import Dict
from .schnet_layer import SchNetLayer
from .so3krates_layer import So3kratesLayer

def get_layer(name: str, h: Dict):
    if name == 'schnet_layer':
        return SchNetLayer(**h)
    elif name == 'so3krates_layer':
        return So3kratesLayer(**h)
    else:
        msg = "Layer with `module_name={}` is not implemented.".format(name)
        raise NotImplementedError(msg)
