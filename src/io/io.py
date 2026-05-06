# Taken from mlff (https://github.com/thorben-frank/mlff, commit 99dbf76)
# Original author: Thorben Frank et al.
# Modifications: renamed imports (mlff.src → src)

import itertools
import os
import pathlib
import json
from typing import (Dict, Sequence)
import logging

import numpy as np


def read_json(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + "_" + str(counter) + extension
        counter += 1

    return path


def last_module(path):
    # returns path to last module of list of the form module, module_1, module_2, ..., module_n
    # (no gaps in the sequence 0,1,2,... allowed)
    filename, extension = os.path.splitext(path)
    counter = 1

    if os.path.exists(path) == False:
        logging.warning("Module path does not exist")
        raise FileNotFoundError
    next_path = filename + "_" + str(counter) + extension

    while os.path.exists(next_path):
        path = filename + "_" + str(counter) + extension
        counter += 1
        next_path = filename + "_" + str(counter) + extension

    return path

def cond_create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_directory(path, overwrite=False):
    if not overwrite:
        path = uniquify(path)
    pathlib.Path(path).mkdir(parents=True, exist_ok=overwrite)
    return path


def save_json(path, filename, data, overwrite=False):
    path = create_directory(path, overwrite=overwrite)
    save_path = os.path.join(path, filename)
    with open(save_path, 'w') as f:
        json.dump(data, f)

def update_json(path, filename, data, overwrite=False):
    path = create_directory(path, overwrite=overwrite)
    save_path = os.path.join(path, filename)
    if os.path.exists(save_path):
        d = read_json(save_path)
        with open(save_path, 'w') as f:
            d.update(data)
            json.dump(d, f)
    else:
        with open(save_path, 'w') as f:
            json.dump(data, f)

def update_npz(path, filename, overwrite=False, **kwargs):
    path = create_directory(path, overwrite=overwrite)
    save_path = os.path.join(path, filename+'.npz')
    if os.path.exists(save_path):
        d = dict(np.load(save_path, allow_pickle=True))
        d.update(kwargs)
        np.savez(save_path, **d)
    else:
        np.savez(save_path, **kwargs)

def bundle_dicts(x: Sequence[Dict]) -> Dict:
    """
    Bundles a list of dictionaries into one.

    Args:
        x (Sequence): List of dictionaries.

    Returns: The bundled dictionary.

    """

    bd = {}
    for d in x:
        bd.update(d)
    return bd


def merge_dicts(x, y):
    x.update(y)
    return x

def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))