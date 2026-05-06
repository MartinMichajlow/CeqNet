import jax
import numpy as np

def load_qm9(data_path: str, prop_keys: dict, n_data = None):
    """
    loads [:n_data]-chunk of the qm9 data set, which is stored at data_path.
    """
    from src.data.dataset import DataSet
    data_tot = np.load(data_path)

    # loading the complete dataset might lead to memory issues, so we slice n_data many datapoint off
    data = {}
    if n_data is not None:
        data[prop_keys['atomic_type']] = data_tot['atomic_types'][:n_data]
        data[prop_keys['partial_charge']] = data_tot['partial_charges'][:n_data]
        data[prop_keys['atomic_position']] = data_tot['coordinates'][:n_data]
        data[prop_keys['properties']] = data_tot['properties'][:n_data]
        data[prop_keys['total_charge']] = data_tot['total_charges'][:n_data]
        data['n_atoms'] = data_tot['n_atoms'][:n_data]

    else:
        data[prop_keys['atomic_type']] = data_tot['atomic_types']
        data[prop_keys['partial_charge']] = data_tot['partial_charges']
        data[prop_keys['atomic_position']] = data_tot['coordinates']
        data[prop_keys['properties']] = data_tot['properties']
        data[prop_keys['total_charge']] = data_tot['total_charges']
        data['n_atoms'] = data_tot['n_atoms']


    data = DataSet(data=data, prop_keys=prop_keys)

    return data

def load_qm9_test_data(data_path, train_split, prop_keys, n_data = 10000):
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
    data = load_qm9(data_path, prop_keys, n_data)
    test_data = jax.tree.map(lambda y: y[train_split['random_split']['data_idx_test']], data.data)
    return test_data