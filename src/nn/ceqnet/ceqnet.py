import flax.linen as nn
import jax
import jax.numpy as jnp
import json
import os

from typing import (Any, Callable, Dict, Sequence, Tuple)

from src.nn.layer import get_layer
from src.nn.embed import get_embedding_module
from src.nn.observable import get_observable_module


Array = Any


class CeqNet(nn.Module):
    feature_embeddings: Sequence[Callable]
    geometry_embeddings: Sequence[Callable]
    charge_embeddings: Sequence[Callable]
    layers: Sequence[Callable]
    observables: Sequence[Callable]
    prop_keys: Dict
    mode: str
    eval_keys: Sequence[str] = None

    def setup(self):
        if len(self.feature_embeddings) == 0:
            msg = "At least one embedding module in `feature_embeddings` is required."
            raise ValueError(msg)
        if len(self.observables) == 0:
            msg = "At least one observable module in `observables` is required."
            raise ValueError(msg)

    @nn.compact
    def __call__(self,
                 inputs,
                 *args,
                 **kwargs) -> Dict[str, jnp.ndarray]:
        """
        Energy function of the NN.

        Args:
            inputs (Dict):
            args (Tuple):
            kwargs (Dict):

        Returns: energy, shape: (1)

        """

        quantities = {}
        quantities.update(inputs)

        # Initialize masks
        quantities.update(init_masks(z=inputs[self.prop_keys['atomic_type']],
                                     idx_i=inputs['idx_i'])
                          )

        # Initialize the geometric quantities
        for geom_emb in self.geometry_embeddings:
            geom_quantities = geom_emb(quantities)
            quantities.update(geom_quantities)

        # Initialize the per atom embedding
        embeds = []
        for embed_fn in self.feature_embeddings:
            embeds += [embed_fn(quantities)]  # len: n_embeds, shape: (n,F)
        x = jnp.stack(embeds, axis=-1).sum(axis=-1) / jnp.sqrt(len(embeds))  # shape: (n,F)
        quantities.update({'x': x})

        # Initialize hardness when any ceq-based observable is used
        if 'ceq' in self.mode:
            for charge_emb in self.charge_embeddings:
                charge_quantities = charge_emb(quantities)
                quantities.update(charge_quantities)

        for (n, layer) in enumerate(self.layers):

            updated_quantities = layer(**quantities)
            quantities.update(updated_quantities)

        observables = {}
        for o_fn in self.observables:
            o_dict = o_fn(quantities)
            observables.update(o_dict)
            quantities.update(o_dict)

        eval_observables = {}
        if self.eval_keys == None:
            eval_observables.update(observables)
        else:
            for key in observables.keys():
                if key in self.eval_keys:
                    eval_observables.update({key: observables[key]})
        return jax.tree.map(lambda y: y[..., None], eval_observables)


    def __dict_repr__(self):
            feature_embeddings = []
            geometry_embeddings = []
            charge_embeddings = []
            layers = []
            observables = []
            eval_keys = []
            for x in self.geometry_embeddings:
                geometry_embeddings += [x.__dict_repr__()]
            for x in self.feature_embeddings:
                feature_embeddings += [x.__dict_repr__()]
            for x in self.charge_embeddings:
                charge_embeddings += [x.__dict_repr__()]
            for (n, x) in enumerate(self.layers):
                layers += [x.__dict_repr__()]
            for x in self.observables:
                observables += [x.__dict_repr__()]
            if self.eval_keys == None:
                eval_keys = None
            else:
                eval_keys = []
                for x in self.eval_keys:
                    eval_keys += [x]


            return {'ceq_net': {'feature_embeddings': feature_embeddings,
                                'geometry_embeddings': geometry_embeddings,
                                'charge_embeddings': charge_embeddings,
                                'layers': layers,
                                'observables': observables,
                                'prop_keys': self.prop_keys,
                                'mode': self.mode,
                                'eval_keys': eval_keys}}

    def to_json(self, ckpt_dir, name='hyperparameters.json'):
        j = self.__dict_repr__()
        with open(os.path.join(ckpt_dir, name), 'w', encoding='utf-8') as f:
            json.dump(j, f, ensure_ascii=False, indent=4)

    def reset_prop_keys(self, prop_keys, sub_modules=True) -> None:
        self.prop_keys.update(prop_keys)
        if sub_modules:
            all_modules = self.geometry_embeddings + self.feature_embeddings + self.observables
            for m in all_modules:
                m.reset_prop_keys(prop_keys=prop_keys)


def init_masks(z, idx_i):
    point_mask = (z != 0).astype(jnp.float32)  # shape: (n)
    pair_mask = (idx_i != -1).astype(jnp.float32)  # shape: (n_pairs)
    return {'point_mask': point_mask, 'pair_mask': pair_mask}


def init_ceq_net(h):
    """
    initializes a ceq net from a given hyperparameter file
    :param h: dict of hyperparameters
    :return:
    """
    _h = h['ceq_net']
    feature_embs = [get_embedding_module(*tuple(x.items())[0]) for x in _h['feature_embeddings']]
    geometry_embs = [get_embedding_module(*tuple(x.items())[0]) for x in _h['geometry_embeddings']]
    charge_embs = [get_embedding_module(*tuple(x.items())[0]) for x in _h['charge_embeddings']]
    lays = [get_layer(*tuple(x.items())[0]) for x in _h['layers']]

    obs = [get_observable_module(*tuple(x.items())[0]) for x in _h['observables']]
    eval_keys = None
    if 'eval_keys' in _h.keys():
        eval_keys = _h['eval_keys']
    return CeqNet(**{'feature_embeddings': feature_embs,
                     'geometry_embeddings': geometry_embs,
                     'charge_embeddings': charge_embs,
                     'layers': lays,
                     'observables': obs,
                     'prop_keys': _h['prop_keys'],
                     'mode': _h['mode'],
                     'eval_keys': eval_keys
                     })



