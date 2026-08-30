import numpy as np
from scipy.ndimage import binary_erosion


def fit_base_plane(
    depth_map: np.ndarray,
    mask: np.ndarray
) -> tuple[float, float, float]:
    """
    Gonzalez Phase 2: Fit plane z = ax + by + c to plate surface.
    Boundary pixels touch the plate. Their depth = plate depth.
    """
    mask_bool = mask.astype(bool)

    eroded = binary_erosion(mask_bool)
    boundary = mask_bool & (eroded == False)

    y_coords, x_coords = np.where(boundary)

    if len(y_coords) < 3:
        return 0.0, 0.0, float(np.mean(depth_map))

    z_values = depth_map[y_coords, x_coords]

    A = np.column_stack([
        x_coords,
        y_coords,
        np.ones_like(x_coords)
    ])

    a, b, c = np.linalg.lstsq(
        A,
        z_values,
        rcond=None
    )[0]

    return float(a), float(b), float(c)