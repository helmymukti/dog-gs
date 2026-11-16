import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from doggs.pipeline import Job, Pipeline, STAGES, find_model, default_stem
from test_render import scene


def test_pipeline_runs_and_reports(tmp_path):
    ply = scene(tmp_path)
    # emulate a GeoSplat export: CRS json next to the PLY
    (tmp_path / "gaussians_crs.json").write_text(json.dumps({"files": {"s.ply": {
        "frame": "EPSG:32749 with false origin removed", "add_offset_to_get_utm": [698000.0, 9194000.0, 0.0]}}}))
    events = []
    job = Job(ply=str(ply), out_dir=str(tmp_path / "out"), stem="t", gsd=0.5, colour="dc",
              zmin=None, zmax=None, backend="python", extra=dict(max_scale=None, bounds=(698000, 9194000, 698030, 9194030)))
    ok = Pipeline(job, on_log=lambda s: None, on_stage=lambda k, s, i="": events.append((k, s))).run()
    assert ok
    keys = {k for k, _ in STAGES}
    assert {k for k, s in events if s in ("done", "skipped")} == keys
    rep = json.loads((tmp_path / "out" / "t_report.json").read_text())
    assert rep["software"]["software"] == "DOG-GS" and len(rep["software"]["authors"]) == 4


def test_find_model_and_stem(tmp_path):
    proj = tmp_path / "DJI_0001_V_GeoSplat"
    (proj / "10_Vertical_Correction").mkdir(parents=True)
    (proj / "08_Exports").mkdir()
    (proj / "08_Exports" / "gaussians_utm_epsg32749_offset.ply").write_bytes(b"")
    (proj / "10_Vertical_Correction" / "gaussians_utm_epsg32749_offset_hcorr.ply").write_bytes(b"")
    ply, root = find_model(proj)
    assert ply.name.endswith("_hcorr.ply") and root == proj
    assert default_stem(ply, root) == "DJI_0001_V"
