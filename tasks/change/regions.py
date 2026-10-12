"""Pixel-area measurements for declared screen-coordinate region partitions."""
from __future__ import annotations

import numpy as np


def mask_region_pixels(mask: np.ndarray) -> dict[str, int]:
    """Partition every foreground pixel once in each complete 2x2/3x3 grid.

    These are deployment geometry conventions, not geographic directions.
    Counts measure absolute predicted area, never object counts or density.
    """
    if (not isinstance(mask, np.ndarray) or mask.ndim != 2
            or min(mask.shape) < 3 or mask.dtype.kind not in "biuf"
            or not np.isfinite(mask).all()):
        raise ValueError("Invalid region foreground mask")
    foreground = mask != 0
    height, width = foreground.shape
    rows = np.linspace(0, height, 4, dtype=int)
    columns = np.linspace(0, width, 4, dtype=int)
    names = ("top_left", "top_center", "top_right", "middle_left", "center",
             "middle_right", "bottom_left", "bottom_center", "bottom_right")
    counts = {}
    for index, name in enumerate(names):
        row, column = divmod(index, 3)
        counts[name] = int(foreground[rows[row]:rows[row + 1],
                                     columns[column]:columns[column + 1]].sum())
    middle_y, middle_x = height // 2, width // 2
    counts.update(upper_left=int(foreground[:middle_y, :middle_x].sum()),
                  upper_right=int(foreground[:middle_y, middle_x:].sum()),
                  lower_left=int(foreground[middle_y:, :middle_x].sum()),
                  lower_right=int(foreground[middle_y:, middle_x:].sum()))
    return counts
