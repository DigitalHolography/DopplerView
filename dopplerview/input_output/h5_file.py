import h5py
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

def h5_group_to_dict(group, string_dtype=True):
    result = {}
    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            result[key] = item[()].decode() if string_dtype and isinstance(item[()], bytes) else item[()]
        elif isinstance(item, h5py.Group):
            result[key] = h5_group_to_dict(item, string_dtype=string_dtype)
    return result

def read_h5_to_dict(h5_path):
    """Reads an .h5 file and returns its contents as a dictionary."""
    cache = {}
    metadata = {}
    with h5py.File(h5_path, "r") as h5:
        for key in h5.keys():
            if key == "metadata":
                metadata = h5_group_to_dict(h5[key])
            else:
                cache[key] = h5[key][()]
    return cache, metadata

def hdf5_safe(x):
    if isinstance(x, Path):
        return str(x)
    return x

def write_dict_to_h5(data_dict, h5_path, overwrite=True):
    with h5py.File(h5_path, "a") as h5_cache:
        for key, value in data_dict.items():
            if key in h5_cache:
                if overwrite:
                    del h5_cache[key]
                else:
                    continue
            h5_cache.create_dataset(key, data=hdf5_safe(value))