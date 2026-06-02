"""
Process binary masks
"""

import numpy as np
from skimage.morphology import closing, skeletonize, disk
from skimage.measure import label
from skimage.segmentation import watershed, find_boundaries
from scipy.ndimage import distance_transform_edt, binary_dilation, convolve

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

    y_center, x_center = center

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

def clean_vessel_mask(
    raw_mask,
    image_shape,
    optic_disc_center=None,
    diaphragm_radius=None,
    connect_radius=None,
    tolerance=0,
):
    """
    Clean vessel mask by:
      1. Optionally reconnecting vessels through the optic disc.
      2. Optionally bridging small gaps (tolerance).
      3. Keeping only the largest connected component.
      4. Optionally restricting to the diaphragm mask.

    Parameters
    ----------
    raw_mask : ndarray[bool]
        Input vessel mask.
    image_shape : tuple[int, int]
        (height, width).
    optic_disc_center : tuple[int, int], optional
        Center of the optic disc.
    diaphragm_radius : int, optional
        Radius of the diaphragm mask.
    connect_radius : int, optional
        Radius added temporarily for connectivity analysis.
        Defaults to crop_radius if not provided.
    tolerance : int, default=0
        Radius (pixels) used for binary closing before
        connected-component analysis.
    """
    height, width = image_shape

    if optic_disc_center is None:
        optic_disc_center = (width // 2, height // 2)

    connect_mask = None

    connect_mask = disk_mask(
        height,
        width,
        R1=connect_radius,
        center=optic_disc_center,
    )

    mask_for_cc = raw_mask.copy()

    if connect_mask is not None:
        mask_for_cc |= connect_mask

    if tolerance > 0:
        mask_for_cc = closing(
            mask_for_cc,
            footprint=disk(tolerance),
        )

    # Largest connected component
    largest_component = bwareafilt_largest(
        mask_for_cc,
        connectivity=2,
    )

    # Keep only original vessel pixels
    clean = raw_mask & largest_component

    # Apply diaphragm mask
    if diaphragm_radius is not None:
        diaphragm_mask = disk_mask(
            height,
            width,
            R1=diaphragm_radius,
        )

        clean &= diaphragm_mask

    # Reconnect vessels through optic disc
    if connect_mask is not None:
        clean = clean & ~connect_mask | (raw_mask & connect_mask)
    return clean

def mask_to_bbox(mask):
    """
    Compute bounding box from a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask of shape (H, W)

    Returns
    -------
    x_min, y_min, x_max, y_max : int
        Bounding box coordinates

    Returns None if mask is empty.
    """

    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min = xs.min()
    x_max = xs.max()

    y_min = ys.min()
    y_max = ys.max()

    return x_min, y_min, x_max, y_max

def bbox_to_mask(center, width, height, target_shape):
        x_center, y_center = center

        h, w = target_shape

        # Coordinate grid
        X, Y = np.meshgrid(
            np.arange(1, w + 1),
            np.arange(1, h + 1)
        )

        # Ellipse radii
        rx = width / 2
        ry = height / 2

        if rx > 0 and ry > 0:

            norm_dist = (
                ((X - x_center) ** 2) / (rx ** 2)
                + ((Y - y_center) ** 2) / (ry ** 2)
            )

            optic_disk_mask = norm_dist <= 1

        else:
            optic_disk_mask = np.zeros((h, w), dtype=bool)

        return optic_disk_mask