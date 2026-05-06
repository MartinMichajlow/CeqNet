# Taken from mlff (https://github.com/thorben-frank/mlff, commit 99dbf76)
# Original author: Thorben Frank et al.
# Modifications: renamed imports (mlff.src → src)

from typing import Dict
from .embed import (AtomTypeEmbed,
                    HardnessEmbed,
                    GeometryEmbed,
                    ChargeEmbed,
                    CeqEmbed)



def get_embedding_module(name: str, h: Dict):
    if name == 'atom_type_embed':
        return AtomTypeEmbed(**h)
    elif name == 'hardness_embed':
        return HardnessEmbed(**h)
    elif name == 'tot_charge_embed':
        return ChargeEmbed(**h)
    elif name == 'geometry_embed':
        return GeometryEmbed(**h)
    elif name == 'qeq_embed':
        return CeqEmbed(**h)
    else:
        msg = "No embedding module implemented for `module_name={}`".format(name)
        raise ValueError(msg)
