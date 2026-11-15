"""Reader for 3D Gaussian Splatting PLY files (graphdeco-inria layout)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PLY_TYPES = {
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "ushort": "u2", "uint16": "u2", "short": "i2", "int16": "i2",
    "uint": "u4", "uint32": "u4", "int": "i4", "int32": "i4",
}


@dataclass
class GaussianModel:
    """Gaussian parameters in the PLY frame (activated where noted).

    xyz      (N,3) float64  means
    f_dc     (N,3) float32  SH degree-0 coefficients
    f_rest   (N,3,K) float32 higher-order SH coefficients, K = (deg+1)^2 - 1
    opacity  (N,)  float64  sigmoid-activated opacity in [0,1]
    scale    (N,3) float64  exp-activated standard deviations (metres)
    rot      (N,4) float64  normalised quaternions (w, x, y, z)
    comments list[str]      PLY header comments
    """

    xyz: np.ndarray
    f_dc: np.ndarray
    f_rest: np.ndarray
    opacity: np.ndarray
    scale: np.ndarray
    rot: np.ndarray
    comments: list

    @property
    def n(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def sh_degree(self) -> int:
        k = self.f_rest.shape[2] + 1
        return int(round(np.sqrt(k))) - 1

    def subset(self, mask: np.ndarray) -> "GaussianModel":
        return GaussianModel(self.xyz[mask], self.f_dc[mask], self.f_rest[mask], self.opacity[mask],
                             self.scale[mask], self.rot[mask], self.comments)


def _parse_header(raw: bytes):
    end = raw.index(b"end_header")
    end = raw.index(b"\n", end) + 1
    lines = raw[:end].decode("ascii", "replace").splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError("not a PLY file")
    fmt, n, props, comments, in_vertex = None, None, [], [], False
    for ln in lines[1:]:
        tok = ln.split()
        if not tok:
            continue
        if tok[0] == "format":
            fmt = tok[1]
        elif tok[0] == "comment":
            comments.append(ln[len("comment "):])
        elif tok[0] == "element":
            in_vertex = tok[1] == "vertex"
            if in_vertex:
                n = int(tok[2])
        elif tok[0] == "property" and in_vertex:
            if tok[1] == "list":
                raise ValueError("list properties are not supported in the vertex element")
            props.append((tok[2], _PLY_TYPES[tok[1]]))
    if fmt != "binary_little_endian":
        raise ValueError(f"unsupported PLY format {fmt!r}; expected binary_little_endian")
    return end, n, props, comments


def read_gaussians(path: str | Path) -> GaussianModel:
    """Read a 3DGS PLY and return activated Gaussian parameters."""
    raw = Path(path).read_bytes()
    start, n, props, comments = _parse_header(raw)
    dt = np.dtype([(name, "<" + t) for name, t in props])
    v = np.frombuffer(raw, dtype=dt, count=n, offset=start)
    names = set(dt.names)
    need = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
            "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}
    missing = need - names
    if missing:
        raise ValueError(f"PLY is missing Gaussian properties: {sorted(missing)}")
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)
    n_rest = len([p for p in names if p.startswith("f_rest_")])
    if n_rest % 3:
        raise ValueError("number of f_rest properties is not divisible by 3")
    k = n_rest // 3
    # graphdeco layout: f_rest_[c*K + j] for channel c, coefficient j
    rest = np.stack([v[f"f_rest_{i}"] for i in range(n_rest)], 1).astype(np.float32) if n_rest else np.zeros((n, 0), np.float32)
    f_rest = rest.reshape(n, 3, k)
    opacity = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))
    scale = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float64))
    q = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return GaussianModel(xyz, f_dc, f_rest, opacity, scale, q, comments)
