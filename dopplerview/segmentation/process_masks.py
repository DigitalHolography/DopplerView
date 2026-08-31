"""
Process binary masks
"""

import numpy as np
from skimage.morphology import closing, skeletonize, disk
from skimage.measure import label, regionprops
from skimage.segmentation import watershed, find_boundaries
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_erosion, convolve
from scipy.spatial import cKDTree

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

def get_labeled_vessels(mask, mask_optic_disc=True, x_center=None, y_center=None, r1=0.1):
    numX, numY = mask.shape

    if mask_optic_disc:
        if x_center is None:
            x_center = numY / 2
        if y_center is None:
            y_center = numX / 2

    # Skeletonize and remove central circle
    skel = skeletonize(mask)
    if mask_optic_disc:
        circle_mask = disk_mask(numX, numY, R1=r1, center=(x_center / numY, y_center/ numX))
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
    
    if mask_optic_disc:
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

def draw_connection(mask, point1, point2, thickness=1):
    """
    Draw a line connecting two points in a binary mask.

    Parameters:
    - mask: 2D numpy array (binary mask)
    - point1: Tuple (x1, y1) for the first point
    - point2: Tuple (x2, y2) for the second point
    - thickness: Thickness of the line

    Returns:
    - mask_with_line: 2D numpy array with the line drawn
    """
    from skimage.draw import line

    rr, cc = line(point1[1], point1[0], point2[1], point2[0])
    
    # Ensure the coordinates are within the mask bounds
    rr = np.clip(rr, 0, mask.shape[0] - 1)
    cc = np.clip(cc, 0, mask.shape[1] - 1)

    mask_with_line = mask.copy()
    
    for t in range(-thickness // 2, thickness // 2 + 1):
        rr_offset = np.clip(rr + t, 0, mask.shape[0] - 1)
        cc_offset = np.clip(cc + t, 0, mask.shape[1] - 1)
        mask_with_line[rr_offset, cc_offset] = True

    return mask_with_line

def get_all_points_in_radius(mask, center, radius):
    """
    Get all points in the mask that are within a given radius from a center point.

    Parameters:
    - mask: 2D numpy array (binary mask)
    - center: Tuple (x, y) for the center point
    - radius: Radius in pixels

    Returns:
    - points_in_radius: List of tuples (x, y) of points within the radius
    """
    y_indices, x_indices = np.where(mask)
    points_in_radius = []
    
    for x, y in zip(x_indices, y_indices):
        if np.sqrt((x - center[0])**2 + (y - center[1])**2) <= radius:
            points_in_radius.append((x, y))
    
    return points_in_radius

def connect_components(mask, max_distance=5):
    """
    Connect components in a binary mask that are within a certain distance of each other.

    Parameters:
    - mask: 2D numpy array (binary mask)
    - max_distance: Maximum distance to connect components

    Returns:
    - connected_mask: 2D numpy array (binary mask) with connected components
    """
    mask = np.asarray(mask, dtype=bool)
    if max_distance <= 0 or not mask.any():
        return mask.copy()

    labeled_mask = label(mask)
    if labeled_mask.max() < 2:
        return mask.copy()

    # Only component boundary pixels can form a shortest bridge. A spatial
    # index avoids the previous implementation's full foreground scan for
    # every pixel (and its full-image copy per connected component).
    boundary = mask & ~binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    coords = np.column_stack(np.nonzero(boundary))  # (row, column)
    component_labels = labeled_mask[boundary]
    pairs = cKDTree(coords).query_pairs(max_distance, output_type="ndarray")
    if pairs.size == 0:
        return mask.copy()

    pairs = pairs[component_labels[pairs[:, 0]] != component_labels[pairs[:, 1]]]
    if pairs.size == 0:
        return mask.copy()

    # One shortest bridge is sufficient for each pair of components. Drawing
    # every nearby pixel pair only thickens the same bridge and dominates the
    # runtime on dense masks.
    component_pairs = np.sort(component_labels[pairs], axis=1)
    distances_sq = np.sum((coords[pairs[:, 0]] - coords[pairs[:, 1]]) ** 2, axis=1)
    order = np.lexsort((distances_sq, component_pairs[:, 1], component_pairs[:, 0]))
    sorted_component_pairs = component_pairs[order]
    first_for_pair = np.ones(order.size, dtype=bool)
    first_for_pair[1:] = np.any(
        sorted_component_pairs[1:] != sorted_component_pairs[:-1], axis=1
    )

    connected_mask = mask.copy()
    for pair_index in order[first_for_pair]:
        first, second = coords[pairs[pair_index]]
        connected_mask = draw_connection(
            connected_mask,
            (first[1], first[0]),
            (second[1], second[0]),
            thickness=2,
        )

    return connected_mask

def keep_connected_components(mask, anchor, negative=False):
    """
    Keep only the connected components in the mask that are connected to the anchor.

    Parameters:
    - mask: 2D numpy array (binary mask)
    - anchor: 2D numpy array (binary mask) indicating the anchor points

    Returns:
    - cleaned_mask: 2D numpy array (binary mask) with only the connected components
      that are connected to the anchor.
    """
    labeled_mask = label(mask)
    anchor_labels = np.unique(labeled_mask[anchor > 0])
    
    negative_mask = np.isin(labeled_mask, anchor_labels) if negative else ~np.isin(labeled_mask, anchor_labels)
    
    cleaned_mask = mask & ~negative_mask

    return cleaned_mask

def remove_small_vessels(labeled_vessels, min_size=10):
    unique_labels, counts = np.unique(labeled_vessels, return_counts=True)
    small_labels = unique_labels[counts < min_size]
    if small_labels.size:
        labeled_vessels[np.isin(labeled_vessels, small_labels)] = 0
    return labeled_vessels


def compute_branch_overlaps(labels, artery_mask, vein_mask, choroid_mask=None):
    artery_mask = artery_mask.astype(bool)
    vein_mask = vein_mask.astype(bool)
    if choroid_mask is not None:
        choroid_mask = choroid_mask.astype(bool)

    max_label = labels.max()

    sizes = np.bincount(labels.ravel(), minlength=max_label + 1)

    artery_counts = np.bincount(
        labels[artery_mask].ravel(),
        minlength=max_label + 1,
    )

    vein_counts = np.bincount(
        labels[vein_mask].ravel(),
        minlength=max_label + 1,
    )

    if choroid_mask is not None:
        choroid_counts = np.bincount(
            labels[choroid_mask].ravel(),
            minlength=max_label + 1,
        )

    branch_ids = np.unique(labels)
    branch_ids = branch_ids[branch_ids > 0]

    artery_ratio = artery_counts[branch_ids] / sizes[branch_ids]
    vein_ratio = vein_counts[branch_ids] / sizes[branch_ids]

    if choroid_mask is not None:
        choroid_ratio = choroid_counts[branch_ids] / sizes[branch_ids]

    return {
        "branch_ids": branch_ids,
        "size": sizes[branch_ids],
        "artery_pixels": artery_counts[branch_ids],
        "vein_pixels": vein_counts[branch_ids],
        "choroid_pixels": choroid_counts[branch_ids] if choroid_mask is not None else None,
        "artery_ratio": artery_ratio,
        "vein_ratio": vein_ratio,
        "choroid_ratio": choroid_ratio if choroid_mask is not None else None,
    }

def compute_branch_label(overlaps, artery_threshold=0.4, vein_threshold=0.4, choroid_threshold=0.4, remove_overlaps=False, labeled_vessels=None):
    artery_mask = overlaps["artery_ratio"] >= artery_threshold
    vein_mask = overlaps["vein_ratio"] >= vein_threshold
    if overlaps["choroid_ratio"] is not None:
        choroid_mask = overlaps["choroid_ratio"] >= choroid_threshold

    labels = np.zeros_like(overlaps["branch_ids"], dtype=int)
    labels[artery_mask] += 1
    labels[vein_mask] += 2
    if overlaps["choroid_ratio"] is not None:
        labels[choroid_mask] += 4 if not remove_overlaps else 3
    if remove_overlaps:
        if labeled_vessels is None:
            raise ValueError("labeled_vessels must be provided when remove_overlaps is True")
        # Remove overlapping labels
        labels[artery_mask & vein_mask] = 0
        if overlaps["choroid_ratio"] is not None:
            labels[artery_mask & choroid_mask] = 0
            labels[vein_mask & choroid_mask] = 0

        to_remove = np.where(labels == 0)[0]
        labeled_vessels[
            np.isin(labeled_vessels, overlaps["branch_ids"][to_remove])
        ] = 0
        labels = labels[labels != 0]

        # Relabel the clean mask
        l = np.unique(labeled_vessels)
        l = l[l != 0]

        relabel_map = np.zeros(labeled_vessels.max() + 1, dtype=labeled_vessels.dtype)
        relabel_map[l] = np.arange(1, len(l) + 1)

        labeled_vessels = relabel_map[labeled_vessels]

        return labels, labeled_vessels

    return labels

def remove_small_vessels(labeled_vessels, min_size=10):
    unique_labels, counts = np.unique(labeled_vessels, return_counts=True)
    small_labels = unique_labels[counts < min_size]
    for label in small_labels:
        labeled_vessels[labeled_vessels == label] = 0
    return labeled_vessels

def get_branch_differences(pred_labels, gt_labels, labeled_vessels):
    differences = np.zeros_like(labeled_vessels, dtype=object)
    branch_ids = np.unique(labeled_vessels)
    branch_ids = branch_ids[branch_ids > 0]
    if len(pred_labels) != branch_ids.size or len(gt_labels) != branch_ids.size:
        raise ValueError("pred_labels and gt_labels must contain one value per branch")
    for label_idx, branch_idx in enumerate(branch_ids):
        if pred_labels[label_idx] != gt_labels[label_idx]:
            print(f"Branch {branch_idx}: Predicted label = {pred_labels[label_idx]}, Ground truth label = {gt_labels[label_idx]}")
            branch_mask = labeled_vessels == branch_idx
            differences[branch_mask] = branch_idx
            
    return differences
