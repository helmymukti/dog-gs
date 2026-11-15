"""Georeferencing metadata of GeoSplat exports (gaussians_crs*.json)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class GeoFrame:
    """How PLY coordinates map to a projected CRS.

    true_coord = ply_coord + offset ; EPSG code of the projected CRS.
    region_centre / region_radius: valid region in PLY coordinates (optional).
    """

    epsg: int
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    region_centre: np.ndarray | None = None
    region_radius: float | None = None
    source: str = "user"
    notes: dict = field(default_factory=dict)


def _find_entry(meta: dict, ply_name: str):
    files = meta.get("files", {})
    if ply_name in files:
        return ply_name, files[ply_name]
    # fall back to any UTM entry
    for k, v in files.items():
        if "utm" in k.lower() or "EPSG" in str(v.get("frame", "")):
            return k, v
    raise KeyError(f"no matching entry for {ply_name!r} in GeoSplat CRS json")


def load_geosplat_crs(json_path: str | Path, ply_name: str) -> GeoFrame:
    """Parse a GeoSplat ``gaussians_crs*.json`` for a given UTM PLY file."""
    meta = json.loads(Path(json_path).read_text(encoding="utf-8"))
    name, e = _find_entry(meta, Path(ply_name).name)
    m = re.search(r"EPSG:(\d+)", e.get("frame", "")) or re.search(r"epsg(\d{4,5})", name.lower())
    if not m:
        raise ValueError("EPSG code not found in GeoSplat CRS json")
    epsg = int(m.group(1))
    off = np.array(e.get("add_offset_to_get_utm", [0.0, 0.0, 0.0]), dtype=np.float64)
    gf = GeoFrame(epsg=epsg, offset=off, source=f"{Path(json_path).name}:{name}")
    vr, fe = e.get("valid_region"), e.get("from_enu")
    if vr and fe:
        s = float(fe["scale"])
        R = np.asarray(fe["R"], dtype=np.float64)
        t = np.asarray(fe["t"], dtype=np.float64)
        c = np.array([vr["centre_enu_en"][0], vr["centre_enu_en"][1], 0.0])
        gf.region_centre = (s * R @ c + t)[:2]
        gf.region_radius = float(vr["radius_m"]) * s
    gf.notes = {k: meta.get(k) for k in ("software", "n_gaussians", "height_reference", "accuracy", "vertical_correction") if k in meta}
    return gf


def autodetect_crs_json(ply_path: str | Path) -> Path | None:
    """Look for a GeoSplat CRS json next to the PLY (prefers *_hcorr when the PLY is hcorr)."""
    p = Path(ply_path)
    cands = sorted(p.parent.glob("gaussians_crs*.json"))
    if not cands:
        return None
    if "hcorr" in p.name:
        for c in cands:
            if "hcorr" in c.name:
                return c
    return cands[0]
