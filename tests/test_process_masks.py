import numpy as np
from skimage.measure import label

from dopplerview.segmentation.process_masks import connect_components


def test_connect_components_bridges_components_within_distance():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:8, 3:6] = True
    mask[5:8, 10:13] = True

    connected = connect_components(mask, max_distance=5)

    assert label(connected).max() == 1
    assert np.all(connected[mask])


def test_connect_components_leaves_distant_components_separate():
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:5, 2:5] = True
    mask[14:17, 14:17] = True

    connected = connect_components(mask, max_distance=5)

    assert np.array_equal(connected, mask)
