"""Per-Gaussian pre-processing for orthographic rendering."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io_ply import GaussianModel

# Real spherical-harmonics constants (same as graphdeco-inria 3DGS)
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
      -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def sh_basis(direction, degree: int) -> np.ndarray:
    """SH basis values (excluding degree 0) for a unit direction, 3DGS convention."""
    x, y, z = np.asarray(direction, dtype=np.float64) / np.linalg.norm(direction)
    b = []
    if degree >= 1:
        b += [-C1 * y, C1 * z, -C1 * x]
    if degree >= 2:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        b += [C2[0] * xy, C2[1] * yz, C2[2] * (2 * zz - xx - yy), C2[3] * xz, C2[4] * (xx - yy)]
    if degree >= 3:
        b += [C3[0] * y * (3 * xx - yy), C3[1] * xy * z, C3[2] * y * (4 * zz - xx - yy),
              C3[3] * z * (2 * zz - 3 * xx - 3 * yy), C3[4] * x * (4 * zz - xx - yy),
              C3[5] * z * (xx - yy), C3[6] * x * (xx - 3 * yy)]
    return np.asarray(b, dtype=np.float64)


def colours(g: GaussianModel, mode: str = "sh", direction=(0.0, 0.0, -1.0)) -> np.ndarray:
    """RGB colour of every Gaussian.

    mode "sh": SH evaluated for one viewing direction (default nadir, d = (0,0,-1),
               i.e. from a camera at infinity above the scene, z up).
    mode "dc": view-independent colour from the degree-0 term only.
    """
    c = C0 * g.f_dc.astype(np.float64)
    if mode == "sh" and g.f_rest.shape[2] > 0:
        b = sh_basis(direction, g.sh_degree)
        c = c + np.einsum("nck,k->nc", g.f_rest.astype(np.float64), b)
    elif mode not in ("sh", "dc"):
        raise ValueError("colour mode must be 'sh' or 'dc'")
    return np.clip(c + 0.5, 0.0, None)


def rotation_matrices(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q.T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)


@dataclass
class Splats:
    """Screen-space (pixel) parameters of the orthographic splats, sorted top-down."""

    px: np.ndarray      # column of centre (pixel units, continuous, 0 = left edge)
    py: np.ndarray      # row of centre (pixel units, continuous, 0 = top edge)
    z: np.ndarray       # height (PLY frame)
    a: np.ndarray       # conic coefficients: q = a dx^2 + 2 b dx dy + c dy^2
    b: np.ndarray
    c: np.ndarray
    rad: np.ndarray     # 3-sigma footprint radius (pixels)
    opacity: np.ndarray
    rgb: np.ndarray     # (N,3)

    @property
    def n(self):
        return int(self.px.shape[0])


def orthographic_splats(g: GaussianModel, rgb: np.ndarray, x0: float, y_top: float, gsd: float,
                        dilation: float = 0.3) -> Splats:
    """Project 3D Gaussians orthographically onto the horizontal plane.

    For a nadir orthographic camera the projected 2D covariance is the marginal of the
    3D covariance on (X, Y): Sigma_2D = [Sigma]_{xy}; no perspective Jacobian is needed.
    Rows grow southwards, so the XY covariance term changes sign. ``dilation`` (px^2) is
    the low-pass filter used by 3DGS for anti-aliasing.
    """
    R = rotation_matrices(g.rot)
    M = R * g.scale[:, None, :]
    S = M @ M.transpose(0, 2, 1)
    sxx = S[:, 0, 0] / gsd ** 2 + dilation
    syy = S[:, 1, 1] / gsd ** 2 + dilation
    sxy = -S[:, 0, 1] / gsd ** 2
    det = sxx * syy - sxy * sxy
    a, b, c = syy / det, -sxy / det, sxx / det
    lam = 0.5 * (sxx + syy) + np.sqrt(np.maximum(0.25 * (sxx - syy) ** 2 + sxy ** 2, 0.0))
    rad = np.ceil(3.0 * np.sqrt(lam))
    px = (g.xyz[:, 0] - x0) / gsd
    py = (y_top - g.xyz[:, 1]) / gsd
    order = np.argsort(-g.xyz[:, 2], kind="stable")      # highest first = front-to-back for nadir
    f = lambda v: np.ascontiguousarray(v[order])
    return Splats(f(px), f(py), f(g.xyz[:, 2]), f(a), f(b), f(c), f(rad), f(g.opacity), f(rgb))


def filter_gaussians(g: GaussianModel, zmin=None, zmax=None, max_scale=None, min_opacity=1.0 / 255,
                     centre=None, radius=None):
    """Boolean mask of Gaussians kept for rendering, plus a dict of how many each rule removed."""
    keep = np.ones(g.n, bool)
    stats = {}

    def apply(name, m):
        nonlocal keep
        stats[name] = int((keep & ~m).sum())
        keep &= m

    if centre is not None and radius is not None:
        apply("outside_region", np.hypot(g.xyz[:, 0] - centre[0], g.xyz[:, 1] - centre[1]) < radius)
    if zmin is not None:
        apply("below_zmin", g.xyz[:, 2] > zmin)
    if zmax is not None:
        apply("above_zmax", g.xyz[:, 2] < zmax)
    if max_scale is not None:
        apply("too_large", g.scale.max(1) < max_scale)
    if min_opacity is not None:
        apply("transparent", g.opacity > min_opacity)
    stats["kept"] = int(keep.sum())
    return keep, stats
