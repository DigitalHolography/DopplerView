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

def write_dict_to_h5(data_dict, h5_path, overwrite=True):
    open_mode = "w" if overwrite else "a"
    with h5py.File(h5_path, open_mode) as h5_cache:
        for key, value in data_dict.items():
            if key in h5_cache:
                continue
            h5_cache.create_dataset(key, data=hdf5_safe(value))