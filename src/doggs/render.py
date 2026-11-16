"""Direct orthophoto generation: GeoSplat/3DGS PLY -> georeferenced orthophoto + DSM."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import __version__, about
from .gaussians import colours, filter_gaussians, orthographic_splats
from .geosplat import GeoFrame, autodetect_crs_json, load_geosplat_crs
from .geotiff import GeoTransform, write_geotiff
from .io_ply import read_gaussians
from .kernel import composite, pick_backend


@dataclass
class OrthoResult:
    rgba: np.ndarray            # (H,W,4) uint8; alpha = accumulated opacity
    dsm: np.ndarray             # (H,W) float32, median-depth surface height (true vertical datum), NaN = no data
    dsm_mean: np.ndarray        # (H,W) float32, opacity-weighted mean height
    gt: GeoTransform
    meta: dict = field(default_factory=dict)

    def save(self, out_dir, stem="doggs", min_alpha=0.5, write_mean=False):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        desc = json.dumps({k: self.meta[k] for k in ("software", "gsd", "colour_mode", "epsg") if k in self.meta})
        files = {}
        files["ortho"] = out / f"{stem}_ortho.tif"
        write_geotiff(files["ortho"], self.rgba, self.gt, rgba=True, description=desc)
        d = self.dsm.copy()
        d[self.rgba[..., 3] < int(round(min_alpha * 255))] = np.nan
        files["dsm"] = out / f"{stem}_dsm.tif"
        write_geotiff(files["dsm"], np.nan_to_num(d, nan=-9999.0).astype(np.float32), self.gt, nodata=-9999)
        if write_mean:
            m = self.dsm_mean.copy()
            m[self.rgba[..., 3] < int(round(min_alpha * 255))] = np.nan
            files["dsm_mean"] = out / f"{stem}_dsm_mean.tif"
            write_geotiff(files["dsm_mean"], np.nan_to_num(m, nan=-9999.0).astype(np.float32), self.gt, nodata=-9999)
        meta = dict(self.meta, files={k: str(v.name) for k, v in files.items()})
        files["meta"] = out / f"{stem}_meta.json"
        files["meta"].write_text(json.dumps(meta, indent=2))
        return files


def read_camera_heights(ply_path):
    """Camera heights from a GeoSplat camera_trajectory*.csv next to the PLY (column h_m / h / alt)."""
    import csv
    p = Path(ply_path).parent
    for c in sorted(p.glob("camera_trajectory*.csv")):
        try:
            rows = list(csv.DictReader(open(c, newline="", encoding="utf-8-sig")))
            for col in ("h_m", "h", "alt", "abs_alt_m", "height"):
                if rows and col in rows[0]:
                    return np.array([float(r[col]) for r in rows if r[col] not in ("", None)])
        except (OSError, ValueError):
            continue
    return None


def resolve_frame(ply_path, crs_json=None, epsg=None, offset=None) -> GeoFrame:
    if crs_json is None and epsg is None:
        crs_json = autodetect_crs_json(ply_path)
    if crs_json is not None:
        gf = load_geosplat_crs(crs_json, Path(ply_path).name)
        if epsg is not None:
            gf.epsg = int(epsg)
        if offset is not None:
            gf.offset = np.asarray(offset, dtype=np.float64)
        return gf
    if epsg is None:
        raise ValueError("no GeoSplat CRS json found; pass --epsg (and --offset if coordinates are shifted)")
    return GeoFrame(epsg=int(epsg), offset=np.asarray(offset if offset is not None else (0, 0, 0), dtype=np.float64))


def render_orthophoto(ply_path, gsd=0.06, colour_mode="sh", crs_json=None, epsg=None, offset=None,
                      bounds=None, use_region=True, region_scale=1.0, mask_outside_region=False,
                      zmin=None, zmax=None, max_scale=20.0, min_opacity=1.0 / 255,
                      backend="auto", tile=1024, threads=None, verbose=True, camera_heights=None,
                      log=None, progress=None, should_stop=None, on_stage=None) -> OrthoResult:
    """Render a georeferenced orthophoto and median-depth DSM from a 3DGS PLY.

    bounds: (xmin, ymin, xmax, ymax) in the true projected CRS; default = valid region of
            the GeoSplat export (or the 0.5-99.5 percentile extent of the Gaussians).
    zmin/zmax: height window in the true vertical datum (removes floaters near the cameras).
               "auto" uses the ground level (5th percentile of Gaussian heights in the region) and the
               median camera height H_cam:  zmin = ground - 10 m,  zmax = ground + 0.6 (H_cam - ground).
    camera_heights: camera heights in the true datum (default: read camera_trajectory*.csv next to the PLY).
    log / progress / should_stop / on_stage: optional callbacks used by the desktop app.
    """
    t_all = time.perf_counter()
    if log is None:
        log = print if verbose else (lambda *a, **k: None)
    stage = on_stage or (lambda *a, **k: None)
    stop = should_stop or (lambda: False)
    stage("read", "running")
    gf = resolve_frame(ply_path, crs_json, epsg, offset)
    t0 = time.perf_counter()
    g = read_gaussians(ply_path)
    t_read = time.perf_counter() - t0
    log(f"read {g.n:,} Gaussians (SH degree {g.sh_degree}) in {t_read:.1f} s; EPSG:{gf.epsg}, offset {gf.offset.tolist()}")
    stage("read", "done", f"{g.n:,} Gaussians · EPSG:{gf.epsg}")
    stage("filter", "running")

    off = gf.offset
    centre = radius = None
    if use_region and gf.region_centre is not None:
        centre, radius = gf.region_centre, gf.region_radius * region_scale
    auto_info = None
    if zmin == "auto" or zmax == "auto":
        sel = np.ones(g.n, bool) if centre is None else np.hypot(g.xyz[:, 0] - centre[0], g.xyz[:, 1] - centre[1]) < radius
        ground = float(np.percentile(g.xyz[sel, 2], 5)) + off[2]
        if camera_heights is None:
            camera_heights = read_camera_heights(ply_path)
        hc = float(np.median(camera_heights)) if camera_heights is not None and len(camera_heights) else None
        if zmin == "auto":
            zmin = ground - 10.0
        if zmax == "auto":
            zmax = ground + 0.6 * (hc - ground) if hc is not None else None
            if hc is None:
                log("auto height window: no camera trajectory found; upper limit disabled")
        auto_info = dict(ground=round(ground, 3), camera_height=None if hc is None else round(hc, 3),
                         zmin=None if zmin is None else round(zmin, 3), zmax=None if zmax is None else round(zmax, 3))
        log(f"auto height window: ground {ground:.2f} m, camera {hc if hc is None else round(hc, 2)} m -> "
            f"zmin {zmin if zmin is None else round(zmin, 2)}, zmax {zmax if zmax is None else round(zmax, 2)}")
    keep, fstats = filter_gaussians(g, None if zmin is None else zmin - off[2], None if zmax is None else zmax - off[2],
                                    max_scale, min_opacity, centre, radius)
    g = g.subset(keep)
    log(f"filter: {fstats}")
    stage("filter", "done", f"{fstats['kept']:,} kept")

    if bounds is None:
        if centre is not None:
            xmin, xmax = centre[0] - radius + off[0], centre[0] + radius + off[0]
            ymin, ymax = centre[1] - radius + off[1], centre[1] + radius + off[1]
        else:
            lo = np.percentile(g.xyz[:, :2], 0.5, axis=0) + off[:2]
            hi = np.percentile(g.xyz[:, :2], 99.5, axis=0) + off[:2]
            (xmin, ymin), (xmax, ymax) = lo, hi
    else:
        xmin, ymin, xmax, ymax = bounds
    X0 = math.floor(xmin / gsd) * gsd
    YT = math.ceil(ymax / gsd) * gsd
    W = int(math.ceil((xmax - X0) / gsd))
    H = int(math.ceil((YT - ymin) / gsd))
    gt = GeoTransform(X0, YT, gsd, gsd, gf.epsg)
    log(f"grid {W} x {H} px at {gsd} m; upper-left ({X0:.3f}, {YT:.3f})")

    rgb = colours(g, colour_mode)
    S = orthographic_splats(g, rgb, X0 - off[0], YT - off[1], gsd)
    be = pick_backend(backend)
    log(f"backend: {be}")
    stage("render", "running")

    acc_rgb = np.zeros((H, W, 3), np.uint8)
    alpha = np.zeros((H, W), np.uint8)
    dsm = np.full((H, W), np.nan, np.float32)
    dsm_mean = np.full((H, W), np.nan, np.float32)
    lox, hix, loy, hiy = S.px - S.rad, S.px + S.rad, S.py - S.rad, S.py + S.rad
    t0 = time.perf_counter()
    ntiles = math.ceil(H / tile) * math.ceil(W / tile)
    k = 0
    for ty in range(0, H, tile):
        for tx in range(0, W, tile):
            if stop():
                stage("render", "cancelled")
                raise RuntimeError("cancelled")
            h, w = min(tile, H - ty), min(tile, W - tx)
            idx = np.nonzero((hix >= tx) & (lox < tx + w) & (hiy >= ty) & (loy < ty + h))[0]
            r, T, zw, ws, zm = composite(be, idx, S, tx, ty, h, w, threads)
            acc_rgb[ty:ty + h, tx:tx + w] = (np.clip(r, 0, 1) * 255 + 0.5).astype(np.uint8)
            alpha[ty:ty + h, tx:tx + w] = ((1 - T) * 255 + 0.5).astype(np.uint8)
            dsm[ty:ty + h, tx:tx + w] = zm + off[2]
            with np.errstate(invalid="ignore", divide="ignore"):
                dsm_mean[ty:ty + h, tx:tx + w] = np.where(ws > 1e-3, zw / ws, np.nan) + off[2]
            k += 1
            if progress:
                progress(k / ntiles)
            if k % max(1, ntiles // 10) == 0 or k == ntiles:
                log(f"  tile {k}/{ntiles}")
    t_render = time.perf_counter() - t0
    stage("render", "done", f"{W} x {H} px · {t_render:.1f} s")

    if mask_outside_region and centre is not None:
        cols, rows = np.meshgrid(np.arange(W), np.arange(H))
        xs, ys = gt.coord(cols, rows)
        out = np.hypot(xs - (centre[0] + off[0]), ys - (centre[1] + off[1])) > radius
        alpha[out] = 0
        acc_rgb[out] = 0
        dsm[out] = np.nan
        dsm_mean[out] = np.nan

    meta = dict(software=f"dog-gs {__version__}", input=str(Path(ply_path).name), crs_source=gf.source, epsg=gf.epsg,
                offset=gf.offset.tolist(), gsd=gsd, width=W, height=H, upper_left=[X0, YT], colour_mode=colour_mode,
                sh_degree=g.sh_degree, filters=dict(zmin=zmin, zmax=zmax, max_scale=max_scale, min_opacity=min_opacity,
                                                    region_centre=None if centre is None else (np.asarray(centre) + off[:2]).tolist(),
                                                    region_radius=radius, counts=fstats),
                backend=be, tile=tile, time_read_s=round(t_read, 2), time_render_s=round(t_render, 2),
                time_total_s=round(time.perf_counter() - t_all, 2),
                coverage_alpha_gt_0p5=float((alpha > 127).mean()), geosplat_notes=gf.notes,
                provenance=about.provenance())
    log(f"rendered in {t_render:.1f} s (total {meta['time_total_s']} s)")
    return OrthoResult(np.dstack([acc_rgb, alpha]), dsm, dsm_mean, gt, meta)
