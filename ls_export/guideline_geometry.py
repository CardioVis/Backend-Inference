"""Centerline from a binary pericardium mask (column-wise midpoints). Pure NumPy."""

from __future__ import annotations

import numpy as np


def get_pericardium_centerline(
    mask: np.ndarray,
    *,
    gap_threshold: int = 5,
    smooth_win: int = 3,
) -> list[list[list[float]]]:
    """For each column x where mask>0, take y_mid = (y_min+y_max)/2; split into segments on x-gaps.

    `mask` is 2D (H, W), truthy where foreground (e.g. pericardium class).
    Returns list of segments; each segment is [[x, y], ...] in pixel coordinates.
    """
    h, w = mask.shape[:2]
    m = (mask > 0).astype(np.uint8)
    points_by_x: list[tuple[float, float]] = []
    for x in range(w):
        col = m[:, x]
        ys = np.where(col > 0)[0]
        if len(ys) > 0:
            y_mid = (float(ys.min()) + float(ys.max())) / 2.0
            points_by_x.append((float(x), y_mid))
    if not points_by_x:
        return []
    points_by_x.sort(key=lambda p: p[0])
    if smooth_win >= 3 and len(points_by_x) >= smooth_win:
        xs = np.array([p[0] for p in points_by_x], dtype=np.float64)
        ys = np.array([p[1] for p in points_by_x], dtype=np.float64)
        kernel = np.ones(smooth_win, dtype=np.float64) / float(smooth_win)
        ys_smooth = np.convolve(ys, kernel, mode="same")
        points_by_x = list(zip(xs.tolist(), ys_smooth.tolist()))
    segments: list[list[list[float]]] = []
    seg: list[list[float]] = [[points_by_x[0][0], points_by_x[0][1]]]
    for i in range(1, len(points_by_x)):
        if points_by_x[i][0] - points_by_x[i - 1][0] > gap_threshold:
            if len(seg) >= 2:
                segments.append(seg)
            seg = [[points_by_x[i][0], points_by_x[i][1]]]
        else:
            seg.append([points_by_x[i][0], points_by_x[i][1]])
    if len(seg) >= 2:
        segments.append(seg)
    return segments
