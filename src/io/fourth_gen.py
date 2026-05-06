# data loader for .npz-files created from the data from "Ko, Tsz Wai, et al. "A fourth-generation high-dimensional neural network potential with accurate electrostatics including non-local charge transfer." Nature communications 12.1 (2021): 1-11."
import logging

import numpy as np
import jax

def load_fg_data(data_path: str, prop_keys: dict, n_data = None):
    """
    loads the and [:n_data]-chunk of the fourth gen data set, which is stored at data_path.
    """
    from src.data.dataset import DataSet
    data_tot = dict(np.load(data_path))

    # loading the complete dataset might lead to memory issues, so we slice n_data many datapoint off
    if n_data != None:
        if type(n_data) != int:
            logging.warning("n_data needs to be of type int.")
        data = {}
        data[prop_keys['atomic_type']] = data_tot['z'][:n_data]
        data[prop_keys['partial_charge']] = data_tot['q'][:n_data]
        data[prop_keys['atomic_position']] = data_tot['R'][:n_data]
        data[prop_keys['total_charge']] = data_tot['Q'][:n_data]
        data[prop_keys['energy']] = data_tot['E'][:n_data]
        data[prop_keys['force']] = data_tot['F'][:n_data]

    else:
        data = {}
        data[prop_keys['atomic_type']] = data_tot['z']
        data[prop_keys['partial_charge']] = data_tot['q']
        data[prop_keys['atomic_position']] = data_tot['R']
        data[prop_keys['total_charge']] = data_tot['Q']
        data[prop_keys['energy']] = data_tot['E']
        data[prop_keys['force']] = data_tot['F']

    data = DataSet(data=data, prop_keys=prop_keys)
    return data

def load_fg_test_data(data_path, train_split, prop_keys, n_data = 10000):
    """
    loads the test_data set specified by train_split. n_data needs to be big enough, such that all indices in train_split
    are in the scope of the loaded qm9 data.
    :param data_path:
    :param train_split:
    :param r_cut:
    :param prop_keys:
    :param n_data:
    :return:
    """
    data = load_fg_data(data_path, prop_keys, n_data)
    test_data = jax.tree.map(lambda y: y[train_split['random_split']['data_idx_test']], data.data)
    return test_data
