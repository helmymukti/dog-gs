"""Small projection helpers (WGS84 geographic -> UTM) so that no GDAL/PROJ is needed."""
from __future__ import annotations

import numpy as np


def utm_zone_from_epsg(epsg: int):
    if 32601 <= epsg <= 32660:
        return epsg - 32600, False
    if 32701 <= epsg <= 32760:
        return epsg - 32700, True
    raise ValueError(f"EPSG:{epsg} is not a WGS84/UTM code")


def ll2utm(lat, lon, epsg: int):
    """Transverse Mercator (Snyder 1987) on WGS84; sub-millimetre within a UTM zone."""
    zone, south = utm_zone_from_epsg(epsg)
    a = 6378137.0
    f = 1 / 298.257223563
    k0 = 0.9996
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    lat = np.radians(np.asarray(lat, dtype=np.float64))
    lon = np.radians(np.asarray(lon, dtype=np.float64))
    lon0 = np.radians((zone - 1) * 6 - 180 + 3)
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    T = np.tan(lat) ** 2
    C = ep2 * np.cos(lat) ** 2
    A = np.cos(lat) * (lon - lon0)
    e4, e6 = e2 * e2, e2 ** 3
    M = a * ((1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * lat - (3 * e2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * np.sin(2 * lat)
             + (15 * e4 / 256 + 45 * e6 / 1024) * np.sin(4 * lat) - (35 * e6 / 3072) * np.sin(6 * lat))
    E = k0 * N * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T * T + 72 * C - 58 * ep2) * A ** 5 / 120) + 500000.0
    Nn = k0 * (M + N * np.tan(lat) * (A * A / 2 + (5 - T + 9 * C + 4 * C * C) * A ** 4 / 24
                                       + (61 - 58 * T + T * T + 600 * C - 330 * ep2) * A ** 6 / 720))
    return E, Nn + (10000000.0 if south else 0.0)


def pixel_to_target_affine(gt_src, shape_src, epsg_target):
    """Affine (3x2) mapping source pixel (col,row) -> target projected coords, fitted on a 5x5 grid.

    Returns (coef_x, coef_y, max_residual_m). Exact when CRSs are equal; for EPSG:4326 sources
    the residual quantifies the (tiny, for UAV-sized areas) linearisation error.
    """
    H, W = shape_src[:2]
    cols, rows = np.meshgrid(np.linspace(0, W, 5), np.linspace(0, H, 5))
    cols, rows = cols.ravel(), rows.ravel()
    x = gt_src.x0 + cols * gt_src.gsd_x
    y = gt_src.y_top - rows * gt_src.gsd_y
    if gt_src.epsg in (4326, 0) and abs(x).max() <= 180:
        X, Y = ll2utm(y, x, epsg_target)
    elif gt_src.epsg in (epsg_target, 0):
        X, Y = x, y
    else:
        raise ValueError(f"cannot transform EPSG:{gt_src.epsg} to EPSG:{epsg_target} (only 4326 or identical CRS)")
    A = np.c_[cols, rows, np.ones_like(cols)]
    cx = np.linalg.lstsq(A, X, rcond=None)[0]
    cy = np.linalg.lstsq(A, Y, rcond=None)[0]
    res = max(np.abs(A @ cx - X).max(), np.abs(A @ cy - Y).max())
    return cx, cy, float(res)
