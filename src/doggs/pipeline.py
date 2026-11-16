"""Processing pipeline used by the desktop app (and usable from Python).

Stages: read -> filter -> render -> write -> checkz (optional) -> compare (optional).
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import __version__, about

STAGES = [
    ("read", "Read Gaussian model"),
    ("filter", "Filter Gaussians & height window"),
    ("render", "Orthographic Gaussian rendering"),
    ("write", "Write GeoTIFF orthophoto & DSM"),
    ("checkz", "Height check vs RTK checkpoints"),
    ("compare", "Comparison with reference orthophoto"),
]
SETTINGS_FILE = Path.home() / ".doggs_settings.json"


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(d: dict):
    try:
        SETTINGS_FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except OSError:
        pass


def find_model(path):
    """Resolve a GeoSplat project folder or a PLY file -> (ply_path, project_root_or_None)."""
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".ply":
        root = next((q for q in p.parents if (q / "logs").is_dir() or (q / "08_Exports").is_dir()), None)
        return p, root
    if p.is_dir():
        for pattern in ("gaussians_utm*_hcorr.ply", "gaussians_utm*.ply"):
            hits = sorted(p.rglob(pattern), key=lambda q: (len(q.parts), str(q)))
            if hits:
                return hits[0], p
        hits = sorted(p.rglob("point_cloud.ply"), key=str, reverse=True)
        if hits:
            return hits[0], p
    return None, None


def default_output(ply, root):
    if root is not None:
        return str(Path(root) / "11_DOG-GS")
    return str(Path(ply).parent / "DOG-GS_output")


def default_stem(ply, root):
    base = Path(root).name if root is not None else Path(ply).stem
    base = re.sub(r"_GeoSplat$", "", base)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", base) or "doggs"


@dataclass
class Job:
    ply: str
    out_dir: str
    stem: str = "doggs"
    gsd: float = 0.06
    colour: str = "sh"
    zmin: object = "auto"          # float, "auto" or None
    zmax: object = "auto"
    backend: str = "auto"
    checkpoints: str | None = None
    reference: str | None = None
    mask_region: bool = False
    extra: dict = field(default_factory=dict)


class Pipeline:
    def __init__(self, job: Job, on_log=print, on_stage=None, on_progress=None):
        self.job = job
        self.log = on_log
        self.stage = on_stage or (lambda *a, **k: None)
        self.progress = on_progress or (lambda f: None)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self) -> bool:
        from .render import render_orthophoto
        j = self.job
        t0 = time.time()
        self.log(about.banner())
        self.log(f"input : {j.ply}\noutput: {j.out_dir}")
        report = dict(job=asdict(j), software=about.provenance())
        try:
            res = render_orthophoto(j.ply, gsd=j.gsd, colour_mode=j.colour, zmin=j.zmin, zmax=j.zmax,
                                    backend=j.backend, mask_outside_region=j.mask_region, log=self.log,
                                    progress=lambda f: self.progress(0.1 + 0.75 * f),
                                    should_stop=lambda: self._stop, on_stage=self.stage, **j.extra)
        except RuntimeError as e:
            if str(e) == "cancelled":
                self.log("cancelled by user")
                return False
            raise
        self.stage("write", "running")
        files = res.save(j.out_dir, stem=j.stem)
        report["render"] = res.meta
        report["files"] = {k: str(v) for k, v in files.items()}
        self.stage("write", "done", Path(files["ortho"]).name)
        self.progress(0.9)

        if j.checkpoints:
            self.stage("checkz", "running")
            try:
                from .validate import check_z, read_points, to_csv
                pts = [p for p in read_points(j.checkpoints) if not re.search(r"skip|excl", p["cls"] or "", re.I)]
                rows, st = check_z(files["dsm"], pts)
                to_csv(rows, Path(j.out_dir) / f"{j.stem}_check_z.csv")
                report["check_z"] = dict(points=rows, stats=st)
                for r in rows:
                    self.log(f"  {r['id']:>8} {r['cls']:>6}  h={r['h']:.3f}  DSM={r['z_dsm']:.3f}  dZ={r['dz']:+.3f}")
                if st:
                    self.log(f"  dZ: n={st['n']} mean={st['mean']:+.3f} RMSE={st['rmse']:.3f} NMAD={st['nmad']:.3f} m")
                    self.stage("checkz", "done", f"RMSE {st['rmse']:.2f} m · NMAD {st['nmad']:.2f} m (n={st['n']})")
                else:
                    self.stage("checkz", "done", "no valid points")
            except Exception as e:  # noqa: BLE001
                self.log(f"height check failed: {e}")
                self.stage("checkz", "failed", str(e)[:80])
        else:
            self.stage("checkz", "skipped", "no checkpoint CSV")
        self.progress(0.95)

        if j.reference:
            self.stage("compare", "running")
            try:
                from .validate import compare
                c = compare(files["ortho"], j.reference)
                report["compare"] = c
                rel, sim = c["relative"], c["similarity"]
                if rel and sim:
                    self.log(f"  relative RMSE_H {rel['rmse_h']:.3f} m, bias {rel['bias']:.3f} m @ {rel['bias_azimuth_deg']:.1f}°, "
                             f"similarity scale {sim['scale']:.4f}, residual {sim['resid_rmse']:.3f} m")
                    self.stage("compare", "done", f"RMSE {rel['rmse_h']:.2f} m · after similarity {sim['resid_rmse']:.2f} m")
                else:
                    self.stage("compare", "done", "too few correspondences")
            except Exception as e:  # noqa: BLE001
                self.log(f"comparison failed: {e}")
                self.stage("compare", "failed", str(e)[:80])
        else:
            self.stage("compare", "skipped", "no reference orthophoto")

        report["elapsed_s"] = round(time.time() - t0, 1)
        rp = Path(j.out_dir) / f"{j.stem}_report.json"
        rp.write_text(json.dumps(report, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
        self.log(f"report: {rp}")
        self.progress(1.0)
        self.log(f"done in {report['elapsed_s']} s")
        return True


def environment() -> dict:
    """Which optional components are installed (shown in the app header)."""
    from .kernel import c_available, numba_available
    try:
        import cv2  # noqa: F401
        cv = True
    except ImportError:
        cv = False
    return dict(numba=numba_available(), c=c_available(), opencv=cv, version=__version__)


def open_path(path):
    if os.name == "nt":
        os.startfile(path)  # noqa: S606
    else:
        import subprocess
        subprocess.Popen(["xdg-open", str(path)])
