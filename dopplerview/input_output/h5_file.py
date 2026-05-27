import h5py
from pathlib import Path

def read_h5_to_dict(h5_path):
    """Reads an .h5 file and returns its contents as a dictionary."""
    cache = {}
    with h5py.File(h5_path, "r") as input_file:
        for key in input_file.keys():
            cache[key] = input_file[key][()]
    return cache

def hdf5_safe(x):
    if isinstance(x, Path):
        return str(x)
    return x

def write_dict_to_h5(data_dict, h5_path):
    with h5py.File(h5_path, "w") as h5_cache:
        for key, value in data_dict.items():
            if key in h5_cache:
                    del h5_cache[key]
            h5_cache.create_dataset(key, data=hdf5_safe(value))