""" Collection of helper functions and classes for the dataset package """

import pickle
import json
from typing import Union
import os
from pathlib import Path

import numpy as np


# Class definitions ####################################################################################################
########################################################################################################################


class SampleProcessor(object):
    """ Used to iterate over list of samples via multiprocessing """

    def __init__(self, func, **kwargs):
        """
        Args:
            func: Function to call for every sample (sample must be the first argument)
            kwargs: all arguments needed to pass onto the given function (except the iterating argument)
        """
        self.func = func
        self.args = kwargs

    def __call__(self, sample):
        return self.func(sample, **self.args)


# Module functions #####################################################################################################
########################################################################################################################


def convert_to_numpy(out_dict: dict) -> dict:
    """ This function converts a data stream dictionary to numpy. """
    for k, v in out_dict.items():
        if k == 'Timestamp':
            out_dict[k] = np.array(v, dtype=np.int64)
        else:
            out_dict[k] = np.array(v, dtype=np.float64)

    return out_dict


def load_from_json(json_path: Union[str, Path]):
    """ This function loads data from a json file. """
    with open(json_path, encoding='utf-8') as jf:
        data = json.load(jf)

    return data


def save_data_to_json(data, path: Union[str, Path], filename: str):
    """ Helper function for saving data to json. """
    Path(path).mkdir(parents=True, exist_ok=True)

    class NpEncoder(json.JSONEncoder):
        """ Overrides JSONEncoder to handle data formatting of certain numpy data types. """
        def default(self, o):
            if isinstance(o, np.integer):
                return int(o)
            elif isinstance(o, np.floating):
                return float(o)
            elif isinstance(o, np.ndarray):
                return o.tolist()
            else:
                return super(NpEncoder, self).default(o)

    with open(os.path.join(path, filename), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1, cls=NpEncoder)


def save_data_as_pickle(data, target_dir: Union[str, Path], file_name: str = "data.pickle"):
    """ Function for saving data in pickle format. """
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(target_dir, file_name + '.pickle'), 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data, f, protocol=4)


def load_data_from_pickle(file_path: Union[str, Path]):
    """ Loads a pickle file. """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    return data
