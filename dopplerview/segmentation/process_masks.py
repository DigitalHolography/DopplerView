"""
Process binary masks
"""

import numpy as np
from skimage.morphology import skeletonize, disk
from skimage.measure import label
from skimage.segmentation import watershed, find_boundaries
from scipy.ndimage import distance_transform_edt, binary_dilation, convolve
from skimage import morphology
from scipy.ndimage import rotate
from skimage.measure import label
from skimage import exposure

import logging
logger = logging.getLogger(__name__)

def disk_mask(numX, numY, R1, center=(0.5, 0.5), R2=None):
    """
    Creates a binary disk-shaped mask on a normalized grid.

    Parameters:
        numX (int): number of rows (height)
        numY (int): number of columns (width)
        R1 (float): inner radius
        R2 (float): outer radius (default 2)
        center (tuple): center of the disk in normalized coordinates (default (0.5, 0.5))

    Returns:
        mask (np.ndarray): binary mask of shape (numX, numY)
    """
    if R2 is not None and R1 > R2:
        raise ValueError("R1 must be less than or equal to R2")

    x_center, y_center= center

    # normalized grid from 0 to 1
    x = np.linspace(0, 1, numX)
    y = np.linspace(0, 1, numY)
    Y, X = np.meshgrid(y, x)  # X = cols, Y = rows (MATLAB style)

    R = (X - x_center) ** 2 + (Y - y_center) ** 2

    if R2 is None:
        mask = R <= R1**2
    else:
        mask = (R > R1**2) & (R <= R2**2)

    return mask.astype(bool)

def elliptical_mask(ny, nx, radius_frac, center = None):
    radius_frac = max(0.0, min(1.0, float(radius_frac)))
    a = (nx / 2) * radius_frac
    b = (ny / 2) * radius_frac

    Y, X = np.ogrid[:ny, :nx]

    if center is None:
        cy, cx = ny / 2, nx / 2
    else:
        cy, cx = center

    mask = ((X - cx) / a) ** 2 + ((Y - cy) / b) ** 2 <= 1.0
    return mask

def bwareafilt_largest(binary_mask, connectivity=2):
    """
    Equivalent to MATLAB: bwareafilt(binary_mask, 1, 8)

    connectivity:
        1 → 4-connectivity
        2 → 8-connectivity (MATLAB 8)
    """
    labeled = label(binary_mask, connectivity=connectivity)
    
    if labeled.max() == 0:
        return np.zeros_like(binary_mask, dtype=bool)

    # Count pixels per label
    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # ignore background

    largest_label = counts.argmax()
    return labeled == largest_label

def get_labeled_vesselness(mask, x_center, y_center, r1=0.1, r2=0.35, numCircles=10):
    numX, numY = mask.shape
    dr = (r2 - r1) / numCircles

    # Skeletonize and remove central circle
    skel = skeletonize(mask)
    circle_mask = disk_mask(numX, numY, R1=r1, center=(y_center / numY, x_center/ numX))
    skel = skel & ~circle_mask

    # Remove branch points
    neigh = np.array([[1,1,1],[1,10,1],[1,1,1]])
    bp_map = convolve(skel.astype(int), neigh, mode='constant') >= 13  # heuristic
    skel_no_branches = skel & ~binary_dilation(bp_map, disk(2))

    # Label branches
    label_skel = label(skel_no_branches)
    n = label_skel.max()

    # Distance transform (negative for watershed)
    D = -distance_transform_edt(mask)
    D[~(mask)] = -np.inf

    # Markers from skeleton
    markers = label_skel > 0
    markers = binary_dilation(markers, disk(1))

    # Watershed
    L = watershed(~markers)
    edges = find_boundaries(L, mode='outer')
        
    L[edges] = 0
    # L[L>1] = 1

    L = L * mask

    labeled_vessels = np.zeros_like(mask, dtype=int)
    for i in range(1, n + 1):
        branch_pixels = (L == i)
        labeled_vessels[branch_pixels] = i
    
    labeled_vessels *= ~circle_mask

    return labeled_vessels, edges

# mask_diaphragm = process_masks.disk_mask(mask_vessel.shape[0], mask_vessel.shape[1], diaphragm_radius)
# mask_circle = process_masks.disk_mask(mask_vessel.shape[0], mask_vessel.shape[1], diaphragm_radius, center = (x_center, y_center))


# def clean_vessel_mask(mask_vessel, mask_circle=None, mask_diaphragm=None):
#     if mask_circle is not None:
#         mask_vessel = mask_vessel & ~mask_circle
#     largest_connected_components = bwareafilt_largest(
#             mask_vessel,
#             connectivity=2  # 8-connectivity
#         )
#     if mask_diaphragm is not None:
#         mask_vessel = mask_vessel & mask_diaphragm
#     return mask_vessel & largest_connected_components

def clean_vessel_mask(
    raw_mask,
    image_shape,
    optic_disc_center=None,
    diaphragm_radius=None,
    crop_radius=None,
):
    height, width = image_shape

    if diaphragm_radius is not None:
        logger.info(f"    - Applying diaphragm mask with radius {diaphragm_radius}")
        mask_diaphragm = disk_mask(
            height, width, R1=diaphragm_radius
        )

    if crop_radius is not None:
        optic_disc_center = optic_disc_center if optic_disc_center is not None else (width // 2, height // 2)
        mask_center = disk_mask(
            height, width,
            R1=crop_radius,
            center=optic_disc_center
        )

    mask = raw_mask & ~mask_center if crop_radius is not None else raw_mask
    largest_component = bwareafilt_largest(
        mask,
        connectivity=2,
    )

    clean = raw_mask & largest_component & mask_diaphragm if diaphragm_radius is not None else raw_mask & largest_component

    return clean