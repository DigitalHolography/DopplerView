import h5py
from pathlib import Path
import re

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
            try:
                h5_cache.create_dataset(key, data=hdf5_safe(value))
            except Exception as e:
                logger.error(f"Failed to write key '{key}' to HDF5 file: {e}")

def read_bands(h5, generic_band_names=["LF_M0", "HF_M0"]):
    """
    Reads the signal bands from the HDF5 file. Bands are expected to be named as "LF_M0", "HF_M0", or follow the patterns "band_<start_freq>_<end_freq>" or "band_<band_number>_<start_freq>_<end_freq>".
    If the genereic band names are not found, it will attempt to read the bands based on the patterns.
	Returns a list of tuples containing the band name and its corresponding data, ordered by the band_number if present, otherwise by the start frequency.
    """
    
	bands = {}
    
	# First, try to read the generic band names
	for band_name in generic_band_names:
		if band_name in h5:
			bands[band_name] = h5[band_name][()]

	# If no generic bands were found, look for bands with specific patterns
	if not bands:
		pattern = re.compile(r"band_(\d+)_(\d+)")
		for key in h5.keys():
			match = pattern.match(key)
			if match:
				start_freq = int(match.group(1))
				end_freq = int(match.group(2))
				bands[key] = h5[key][()]

	# Sort the bands by their start frequency or band number
	sorted_bands = sorted(bands.items(), key=lambda x: (int(re.search(r"band_(\d+)_", x[0]).group(1)) if re.search(r"band_(\d+)_", x[0]) else float('inf')))
	
	return sorted_bands