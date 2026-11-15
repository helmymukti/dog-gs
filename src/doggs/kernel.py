"""Front-to-back alpha compositing of orthographic Gaussian splats.

The same kernel source runs under three backends:
  * "numba"  - JIT-compiled and parallel (recommended; ``pip install numba``)
  * "c"      - the reference C implementation in ``_ortho.c`` (compiled on demand with a C compiler)
  * "python" - plain Python (very slow; used for testing and as a last resort)

For each pixel p, with Gaussians sorted by decreasing height,
    alpha_i(p) = min(0.99, o_i * exp(-0.5 * d^T Sigma_i^-1 d)),   skipped if < 1/255
    C(p)       = sum_i c_i alpha_i T_i,   T_i = prod_{j<i} (1 - alpha_j),  stop when T < 1e-4
The median-depth surface is the height of the Gaussian at which T first drops below 0.5.
"""
from __future__ import annotations

import ctypes
import math
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import numpy as np

T_MIN = 1e-4
A_MIN = 1.0 / 255.0
A_MAX = 0.99
Q_MAX = 18.0


def _composite(nbands, px, py, z, a, b, c, rad, op, rgb, x0, y0, H, W,
               out_rgb, out_T, out_zw, out_w, out_zmed):
    for band in _prange(nbands):
        r0 = (H * band) // nbands
        r1 = (H * (band + 1)) // nbands
        for i in range(px.shape[0]):
            cx = px[i] - x0 - 0.5
            cy = py[i] - y0 - 0.5
            R = rad[i]
            ya = int(math.floor(cy - R))
            yb = int(math.ceil(cy + R))
            if yb < r0 or ya >= r1:
                continue
            xa = int(math.floor(cx - R))
            xb = int(math.ceil(cx + R))
            if xb < 0 or xa >= W:
                continue
            if ya < r0:
                ya = r0
            if yb > r1 - 1:
                yb = r1 - 1
            if xa < 0:
                xa = 0
            if xb > W - 1:
                xb = W - 1
            A = a[i]; B = b[i]; C = c[i]; o = op[i]; zi = z[i]
            cr = rgb[i, 0]; cg = rgb[i, 1]; cbl = rgb[i, 2]
            for y in range(ya, yb + 1):
                dy = y - cy
                for x in range(xa, xb + 1):
                    T = out_T[y, x]
                    if T < T_MIN:
                        continue
                    dx = x - cx
                    q = A * dx * dx + 2.0 * B * dx * dy + C * dy * dy
                    if q > Q_MAX:
                        continue
                    al = o * math.exp(-0.5 * q)
                    if al < A_MIN:
                        continue
                    if al > A_MAX:
                        al = A_MAX
                    w = al * T
                    out_rgb[y, x, 0] += w * cr
                    out_rgb[y, x, 1] += w * cg
                    out_rgb[y, x, 2] += w * cbl
                    out_zw[y, x] += w * zi
                    out_w[y, x] += w
                    Tn = T * (1.0 - al)
                    if T >= 0.5 and Tn < 0.5:
                        out_zmed[y, x] = zi
                    out_T[y, x] = Tn


_prange = range
_numba_fn = None
_c_lib = None
_PY_SOURCE_FN = _composite


def numba_available() -> bool:
    try:
        import numba  # noqa: F401
        return True
    except Exception:
        return False


def _get_numba():
    global _numba_fn
    if _numba_fn is None:
        import numba
        g = dict(_composite.__globals__)
        g["_prange"] = numba.prange
        fn = type(_composite)(_composite.__code__, g, "_composite_nb")
        _numba_fn = numba.njit(parallel=True, fastmath=True, cache=False)(fn)
    return _numba_fn


def _c_source() -> Path:
    return Path(__file__).with_name("_ortho.c")


def _c_lib_path() -> Path:
    ext = ".dll" if sys.platform == "win32" else (".dylib" if sys.platform == "darwin" else ".so")
    return Path(__file__).with_name("_ortho" + ext)


def build_c(verbose: bool = True) -> Path:
    """Compile the C backend with the first available compiler (gcc/clang/cc)."""
    cc = os.environ.get("CC") or next((c for c in ("gcc", "clang", "cc") if shutil.which(c)), None)
    if cc is None:
        raise RuntimeError("no C compiler found (set CC or install gcc/clang); or use the numba backend")
    out = _c_lib_path()
    cmd = [cc, "-O3", "-ffast-math", "-shared", "-fPIC", str(_c_source()), "-o", str(out), "-lm"]
    omp = cmd[:1] + ["-fopenmp"] + cmd[1:]
    for attempt in (omp, cmd):
        r = subprocess.run(attempt, capture_output=True, text=True)
        if r.returncode == 0:
            if verbose:
                print("built", out, "(OpenMP)" if attempt is omp else "(single-threaded)")
            return out
    raise RuntimeError("C build failed:\n" + r.stderr)


def c_available() -> bool:
    return _c_lib_path().exists()


def _get_c():
    global _c_lib
    if _c_lib is None:
        lib = ctypes.CDLL(str(_c_lib_path()))
        lib.composite.restype = None
        _c_lib = lib
    return _c_lib


def pick_backend(name: str = "auto") -> str:
    if name != "auto":
        return name
    if numba_available():
        return "numba"
    if c_available():
        return "c"
    return "python"


def composite(backend, splats_idx, S, x0, y0, H, W, threads=None):
    """Composite the splats with indices ``splats_idx`` into an H x W tile at (x0, y0).

    Returns (rgb, T, zw, w, zmed) accumulation arrays.
    """
    out_rgb = np.zeros((H, W, 3), np.float32)
    out_T = np.ones((H, W), np.float32)
    out_zw = np.zeros((H, W), np.float32)
    out_w = np.zeros((H, W), np.float32)
    out_zmed = np.full((H, W), np.nan, np.float32)
    if splats_idx.size == 0:
        return out_rgb, out_T, out_zw, out_w, out_zmed
    f32 = lambda v: np.ascontiguousarray(v[splats_idx], dtype=np.float32)
    f64 = lambda v: np.ascontiguousarray(v[splats_idx], dtype=np.float64)
    px, py = f64(S.px), f64(S.py)
    z, a, b, c, rad, op = f32(S.z), f32(S.a), f32(S.b), f32(S.c), f32(S.rad), f32(S.opacity)
    rgb = np.ascontiguousarray(S.rgb[splats_idx], dtype=np.float32)
    if backend == "numba":
        import numba
        nb = numba.get_num_threads() if threads is None else threads
        _get_numba()(max(1, nb), px, py, z, a, b, c, rad, op, rgb, float(x0), float(y0), H, W,
                     out_rgb, out_T, out_zw, out_w, out_zmed)
    elif backend == "c":
        lib = _get_c()
        P = lambda v: v.ctypes.data_as(ctypes.c_void_p)
        lib.composite(ctypes.c_int(int(px.shape[0])), P(px), P(py), P(z), P(a), P(b), P(c), P(rad), P(op), P(rgb),
                      ctypes.c_double(x0), ctypes.c_double(y0), ctypes.c_int(H), ctypes.c_int(W),
                      P(out_rgb), P(out_T), P(out_zw), P(out_w), P(out_zmed))
    elif backend == "python":
        _PY_SOURCE_FN(1, px, py, z, a, b, c, rad, op, rgb, float(x0), float(y0), H, W,
                      out_rgb, out_T, out_zw, out_w, out_zmed)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return out_rgb, out_T, out_zw, out_w, out_zmed
