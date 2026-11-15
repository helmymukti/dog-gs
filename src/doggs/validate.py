"""Accuracy assessment of DOG-GS products (requires opencv-python for check-xy / compare).

check_z   : DSM heights vs surveyed points (e.g. GNSS RTK)
check_xy  : horizontal residuals at checkpoints, locating each surveyed feature in the
            orthophoto by gradient template matching against a reference orthophoto in which
            the checkpoint coordinates fall on the surveyed feature
compare   : relative geometry (displacement field, similarity/affine fit) and radiometry
            between the orthophoto and a reference orthophoto
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from .geotiff import GeoTransform, read_geotiff
from .proj import pixel_to_target_affine


def _cv2():
    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        raise ImportError("this command needs OpenCV: pip install opencv-python") from e
    return cv2


# ---------------------------------------------------------------- statistics
def stats_xy(dE, dN):
    """Horizontal error statistics (NSSDA / Greenwalt-Schultz)."""
    dE, dN = np.asarray(dE, float), np.asarray(dN, float)
    n = len(dE)
    rE, rN = math.sqrt(np.mean(dE ** 2)), math.sqrt(np.mean(dN ** 2))
    rH = math.hypot(rE, rN)
    mE, mN = dE.mean(), dN.mean()
    loo = []
    for k in range(n):
        o = np.ones(n, bool)
        o[k] = False
        if o.sum():
            loo.append(math.hypot(dE[k] - dE[o].mean(), dN[k] - dN[o].mean()))
    return dict(n=n, rmse_e=rE, rmse_n=rN, rmse_h=rH,
                ce90=2.1460 * (rE + rN) / 2, nssda_h95=1.7308 * rH if min(rE, rN) / max(rE, rN, 1e-12) >= 0.6 else None,
                bias=math.hypot(mE, mN), bias_azimuth_deg=math.degrees(math.atan2(mE, mN)) % 360,
                mean_e=mE, mean_n=mN, sd_about_mean=math.sqrt(np.mean((dE - mE) ** 2 + (dN - mN) ** 2)),
                r95=float(np.percentile(np.hypot(dE, dN), 95)),
                loo_translation_corrected_rmse_h=math.sqrt(np.mean(np.square(loo))) if loo else None)


def stats_z(dz):
    dz = np.asarray(dz, float)
    return dict(n=len(dz), mean=float(dz.mean()), rmse=float(math.sqrt(np.mean(dz ** 2))),
                sd=float(dz.std(ddof=1)) if len(dz) > 1 else None, max_abs=float(np.abs(dz).max()),
                nmad=float(1.4826 * np.median(np.abs(dz - np.median(dz)))))


def read_points(path, id_col=None):
    """CSV with columns id/label, E/easting/x, N/northing/y, optional h/z/height, optional class/role."""
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8-sig")))
    def pick(r, *names):
        for n in names:
            for k in r:
                if k.strip().lower() == n:
                    return r[k]
        return None
    pts = []
    for r in rows:
        e = pick(r, "e", "easting", "x")
        nn = pick(r, "n", "northing", "y")
        if e in (None, "") or nn in (None, ""):
            continue
        h = pick(r, "h", "h_ellipsoidal", "h_checkpoint", "z", "height", "ellipsoidal_height")
        pts.append(dict(id=pick(r, *( [id_col] if id_col else []), "id", "label", "name") or str(len(pts) + 1),
                        E=float(e), N=float(nn), h=None if h in (None, "") else float(h),
                        cls=pick(r, "class", "role", "role_suggested") or ""))
    return pts


# ---------------------------------------------------------------- resampling
def resample_to_grid(src, gt_src, gt_dst, shape_dst, antialias=True):
    """Resample a reference raster onto the DOG-GS grid. Returns (array, valid_mask, max_lin_err_m)."""
    cv2 = _cv2()
    cx, cy, err = pixel_to_target_affine(gt_src, src.shape, gt_dst.epsg)
    M = np.array([[cx[0], cx[1]], [cy[0], cy[1]]])
    Mi = np.linalg.inv(M)
    H, W = shape_dst[:2]
    cols, rows = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    X, Y = gt_dst.coord(cols, rows)
    dx, dy = X - cx[2], Y - cy[2]
    sc = (Mi[0, 0] * dx + Mi[0, 1] * dy - 0.5).astype(np.float32)
    sr = (Mi[1, 0] * dx + Mi[1, 1] * dy - 0.5).astype(np.float32)
    del X, Y, dx, dy, cols, rows
    img = src
    ratio = gt_dst.gsd_x / max(abs(np.hypot(cx[0], cy[0])), 1e-12)
    if antialias and ratio > 1.2:
        img = cv2.GaussianBlur(src, (0, 0), 0.5 * math.sqrt(ratio ** 2 - 1))
    out = cv2.remap(img, sc, sr, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = (sc >= 0) & (sr >= 0) & (sc <= src.shape[1] - 1) & (sr <= src.shape[0] - 1)
    return out, valid, err


def _gray(a):
    cv2 = _cv2()
    a = a[..., :3] if a.ndim == 3 else a
    return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) if a.ndim == 3 else a


def _load_pair(ortho_path, reference_path):
    o, gto = read_geotiff(ortho_path)
    r, gtr = read_geotiff(reference_path)
    rr, valid, err = resample_to_grid(r, gtr, gto, o.shape)
    alpha = o[..., 3] if (o.ndim == 3 and o.shape[2] == 4) else np.full(o.shape[:2], 255, np.uint8)
    return o, gto, rr, valid, alpha, err


# ---------------------------------------------------------------- vertical
def check_z(dsm_path, points, radius=0.5, classes=None):
    dsm, gt = read_geotiff(dsm_path)
    dsm = dsm.astype(np.float64)
    dsm[dsm <= -9998] = np.nan
    out = []
    rpx = int(math.ceil(radius / gt.gsd_x))
    yy, xx = np.mgrid[-rpx:rpx + 1, -rpx:rpx + 1]
    disk = np.hypot(xx, yy) * gt.gsd_x <= radius
    for p in points:
        if p["h"] is None or (classes and p["cls"] not in classes):
            continue
        c, r = gt.pixel(p["E"], p["N"])
        c, r = int(round(float(c))), int(round(float(r)))
        win = dsm[r - rpx:r + rpx + 1, c - rpx:c + rpx + 1]
        z = float(np.nanmedian(win[disk])) if win.shape == disk.shape and np.isfinite(win[disk]).any() else float("nan")
        out.append(dict(id=p["id"], cls=p["cls"], E=p["E"], N=p["N"], h=p["h"], z_dsm=z, dz=z - p["h"]))
    good = [o["dz"] for o in out if np.isfinite(o["dz"])]
    return out, (stats_z(good) if good else None)


# ---------------------------------------------------------------- horizontal
def check_xy(ortho_path, reference_path, points, windows_m=(6, 9, 12), search_m=8.0, min_ncc=0.5):
    """Horizontal residual (ortho - checkpoint) at each checkpoint.

    The checkpoint coordinates must fall on the surveyed feature in the reference orthophoto;
    the feature is then located in the DOG-GS orthophoto by normalised cross-correlation of
    gradient magnitudes. Points with NCC < min_ncc are reported but excluded from the statistics.
    """
    cv2 = _cv2()
    o, gt, r, valid, alpha, err = _load_pair(ortho_path, reference_path)

    def grad(a):
        a = cv2.GaussianBlur(_gray(a).astype(np.float32), (0, 0), 1.5)
        return np.hypot(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1))

    gR, gO = grad(r), grad(o)
    g = gt.gsd_x
    s = int(round(search_m / g))
    out = []
    for p in points:
        c, rr = gt.pixel(p["E"], p["N"])
        c, rr = int(round(float(c))), int(round(float(rr)))
        best = None
        for wm in windows_m:
            h = int(round(wm / 2 / g))
            T = gR[rr - h:rr + h, c - h:c + h]
            Sx = gO[rr - h - s:rr + h + s, c - h - s:c + h + s]
            if T.shape != (2 * h, 2 * h) or Sx.shape != (2 * h + 2 * s, 2 * h + 2 * s) or T.std() == 0:
                continue
            res = cv2.matchTemplate(Sx, T, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if best is None or mx > best[0]:
                best = (mx, (loc[0] - s) * g, -(loc[1] - s) * g, wm)
        if best is None:
            out.append(dict(id=p["id"], ncc=float("nan"), dE=float("nan"), dN=float("nan"), window_m=None, used=False))
        else:
            out.append(dict(id=p["id"], ncc=float(best[0]), dE=float(best[1]), dN=float(best[2]), window_m=best[3],
                            used=bool(best[0] >= min_ncc)))
    used = [x for x in out if x["used"]]
    st = stats_xy([x["dE"] for x in used], [x["dN"] for x in used]) if len(used) >= 2 else None
    return out, st, dict(reference_linearisation_error_m=err)


# ---------------------------------------------------------------- relative comparison
def _detectors(cv2):
    akaze = cv2.xfeatures2d.AKAZE_create(threshold=0.0005) if hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "AKAZE_create") \
        else cv2.AKAZE_create(threshold=0.0005)
    return [(akaze, cv2.NORM_HAMMING), (cv2.SIFT_create(), cv2.NORM_L2)]


def _ssim_map(a, b, sigma=1.5):
    cv2 = _cv2()
    a, b = a.astype(np.float64), b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    f = lambda x: cv2.GaussianBlur(x, (11, 11), sigma)
    ma, mb = f(a), f(b)
    va, vb, cab = f(a * a) - ma * ma, f(b * b) - mb * mb, f(a * b) - ma * mb
    return ((2 * ma * mb + C1) * (2 * cab + C2)) / ((ma * ma + mb * mb + C1) * (va + vb + C2))


def fit_transforms(xy_ref, d):
    """Similarity (LSQ) and affine (IRLS, Huber 1 m) from reference coords and displacements (ortho - ref)."""
    x = xy_ref[:, 0] - xy_ref[:, 0].mean()
    y = xy_ref[:, 1] - xy_ref[:, 1].mean()
    Eg, Ng = x + d[:, 0], y + d[:, 1]
    n = len(x)
    Z = np.c_[np.r_[x, y], np.r_[-y, x], np.r_[np.ones(n), np.zeros(n)], np.r_[np.zeros(n), np.ones(n)]]
    p = np.linalg.lstsq(Z, np.r_[Eg, Ng], rcond=None)[0]
    rs = np.hypot(*(Z @ p - np.r_[Eg, Ng]).reshape(2, -1))
    sim = dict(scale=float(math.hypot(p[0], p[1])), rotation_deg=float(math.degrees(math.atan2(p[1], p[0]))),
               t_e=float(p[2]), t_n=float(p[3]), resid_rmse=float(math.sqrt(np.mean(rs ** 2))), resid_median=float(np.median(rs)))
    X = np.c_[x, y, np.ones(n)]
    w = np.ones(n)
    for _ in range(20):
        W = np.sqrt(w)
        a = np.linalg.lstsq(X * W[:, None], Eg * W, rcond=None)[0]
        b = np.linalg.lstsq(X * W[:, None], Ng * W, rcond=None)[0]
        r = np.hypot(X @ a - Eg, X @ b - Ng)
        w = np.where(r < 1.0, 1.0, 1.0 / np.maximum(r, 1e-9))
    aff = dict(scale_e=float(math.hypot(a[0], b[0])), scale_n=float(math.hypot(a[1], b[1])),
               rotation_deg=float(math.degrees(math.atan2(b[0], a[0]))), t_e=float(a[2]), t_n=float(b[2]),
               resid_rmse=float(math.sqrt(np.mean(r ** 2))), resid_median=float(np.median(r)))
    return sim, aff


def compare(ortho_path, reference_path, scales=(2, 3), ratio=0.75, nb_radius_m=40.0, agree_m=1.5, min_agree=2,
            dedupe_m=2.0, min_alpha=0.8, out_npz=None):
    """Relative geometric and radiometric comparison with a reference orthophoto."""
    cv2 = _cv2()
    o, gt, r, valid, alpha, err = _load_pair(ortho_path, reference_path)
    g = gt.gsd_x
    mask = (valid & (alpha >= int(min_alpha * 255))).astype(np.uint8) * 255
    gR, gO = _gray(r), _gray(o)
    cl = cv2.createCLAHE(3.0, (8, 8))
    P, Q = [], []
    for f in scales:
        rs = cl.apply(cv2.resize(gR, None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA))
        os_ = cl.apply(cv2.resize(gO, None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA))
        m = cv2.resize(mask, (rs.shape[1], rs.shape[0]), interpolation=cv2.INTER_NEAREST)
        for det, norm in _detectors(cv2):
            kr, dr = det.detectAndCompute(rs, m)
            ko, do = det.detectAndCompute(os_, m)
            if dr is None or do is None:
                continue
            mm = cv2.BFMatcher(norm).knnMatch(do, dr, k=2)
            good = [a for a, b in (x for x in mm if len(x) == 2) if a.distance < ratio * b.distance]
            P.append(np.float32([ko[x.queryIdx].pt for x in good]) * f)
            Q.append(np.float32([kr[x.trainIdx].pt for x in good]) * f)
    po, pr = np.concatenate(P), np.concatenate(Q)
    d = np.c_[(po[:, 0] - pr[:, 0]) * g, -(po[:, 1] - pr[:, 1]) * g]          # ortho - reference (E, N)
    xy = np.stack(gt.coord(pr[:, 0], pr[:, 1]), 1)
    keep = np.zeros(len(d), bool)
    for i in range(len(d)):
        nb = np.hypot(*(xy - xy[i]).T) < nb_radius_m
        nb[i] = False
        if nb.sum() >= min_agree:
            keep[i] = (np.hypot(*(d[nb] - d[i]).T) < agree_m).sum() >= min_agree
    sel = []
    for i in np.where(keep)[0]:
        if all(math.hypot(*(xy[i] - xy[j])) > dedupe_m for j in sel):
            sel.append(i)
    sel = np.array(sel, int)
    dd, P2 = d[sel], xy[sel]
    rel = stats_xy(dd[:, 0], dd[:, 1]) if len(sel) >= 3 else None
    sim, aff = fit_transforms(P2, dd) if len(sel) >= 6 else (None, None)

    radio = {}
    # radiometry as georeferenced and after similarity co-registration
    for lab in ("as_georeferenced", "after_similarity"):
        oo = o[..., :3]
        al = alpha
        if lab == "after_similarity" and len(sel) >= 6:
            dst = np.stack(gt.pixel(P2[:, 0], P2[:, 1]), 1).astype(np.float32)
            srcp = np.stack(gt.pixel(P2[:, 0] + dd[:, 0], P2[:, 1] + dd[:, 1]), 1).astype(np.float32)
            M, _ = cv2.estimateAffinePartial2D(srcp, dst, method=cv2.LMEDS)
            oo = cv2.warpAffine(np.ascontiguousarray(oo), M, (o.shape[1], o.shape[0]))
            al = cv2.warpAffine(alpha, M, (o.shape[1], o.shape[0]))
        elif lab == "after_similarity":
            continue
        mk = valid & (al >= int(min_alpha * 255))
        ys, xs = np.where(mk)
        sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
        a, b, mm_ = _gray(r)[sl].astype(np.float64), _gray(oo)[sl].astype(np.float64), mk[sl]
        S = _ssim_map(a, b)
        av, bv = a[mm_], b[mm_]
        mse = float(np.mean((av - bv) ** 2))
        radio[lab] = dict(ssim=float(S[mm_].mean()), psnr_db=float(10 * math.log10(255 ** 2 / mse)),
                          ncc=float(np.corrcoef(av, bv)[0, 1]), mse=mse, mae=float(np.mean(np.abs(av - bv))),
                          pixels=int(mm_.sum()))
    ref_area = valid.sum() * g * g
    cov = dict(reference_area_m2=float(ref_area), ortho_area_alpha_gt_0p5_m2=float((alpha > 127).sum() * g * g),
               overlap_pct_of_reference=float(100 * (valid & (alpha > 127)).sum() / max(valid.sum(), 1)))
    pp = pr[sel]
    vy, vx = np.where(valid & (alpha > 127))
    cw, ch = (vx.max() - vx.min() + 1) / 6, (vy.max() - vy.min() + 1) / 6
    cells = len(set(zip(((pp[:, 0] - vx.min()) // cw).astype(int), ((pp[:, 1] - vy.min()) // ch).astype(int)))) if len(sel) else 0
    if out_npz:
        np.savez(out_npz, xy=P2, d=dd)
    return dict(matches_pooled=int(len(d)), matches_consistent=int(len(sel)), grid_cells_6x6=cells,
                relative=rel, similarity=sim, affine=aff, radiometry=radio, coverage=cov,
                reference_linearisation_error_m=err)


def to_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def dump_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
