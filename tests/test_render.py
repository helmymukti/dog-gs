import json
import math

import numpy as np
import pytest

from doggs import render_orthophoto
from doggs.geotiff import read_geotiff
from doggs.kernel import build_c, c_available
from doggs.gaussians import sh_basis, C1, C2, C3

SH_C0 = 0.28209479177387814


def write_ply(path, xyz, rgb, opacity, scale, quat, sh_rest=None):
    n = len(xyz)
    k = 15
    props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(3 * k)] + \
            ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    a = np.zeros((n, len(props)), np.float32)
    a[:, 0:3] = xyz
    a[:, 6:9] = (np.asarray(rgb) - 0.5) / SH_C0
    if sh_rest is not None:
        a[:, 9:9 + 3 * k] = sh_rest
    o = np.clip(opacity, 1e-6, 1 - 1e-6)
    a[:, 9 + 3 * k] = np.log(o / (1 - o))
    a[:, 10 + 3 * k:13 + 3 * k] = np.log(scale)
    a[:, 13 + 3 * k:17 + 3 * k] = quat
    hdr = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n + "".join(f"property float {p}\n" for p in props) + "end_header\n"
    with open(path, "wb") as f:
        f.write(hdr.encode())
        f.write(a.tobytes())


def scene(tmp_path):
    # ground: grey disc gaussians; a red "roof" gaussian above; a green elongated gaussian at 45 deg
    xyz = [[10.0, 10.0, 0.0], [10.0, 10.0, 5.0], [20.0, 20.0, 0.5]]
    rgb = [[0.5, 0.5, 0.5], [0.9, 0.1, 0.1], [0.1, 0.8, 0.1]]
    op = [0.99, 0.99, 0.99]
    sc = [[3.0, 3.0, 0.05], [0.6, 0.6, 0.05], [2.0, 0.2, 0.05]]
    h = math.sqrt(0.5)
    q = [[1, 0, 0, 0], [1, 0, 0, 0], [math.cos(math.pi / 8), 0, 0, math.sin(math.pi / 8)]]   # 45 deg about z
    p = tmp_path / "s.ply"
    write_ply(p, np.array(xyz), np.array(rgb), np.array(op), np.array(sc), np.array(q, float))
    return p


def test_geometry_and_occlusion(tmp_path):
    p = scene(tmp_path)
    res = render_orthophoto(p, gsd=0.1, colour_mode="dc", epsg=32749, offset=(698000, 9194000, 0),
                            bounds=(698000, 9194000, 698030, 9194030), backend="python", max_scale=None, verbose=False)
    gt = res.gt
    c, r = gt.pixel(698010.0, 9194010.0)
    c, r = int(round(float(c))), int(round(float(r)))
    px = res.rgba[r, c]
    assert px[0] > 200 and px[1] < 60, px                    # roof (red) occludes ground at its centre
    assert abs(res.dsm[r, c] - 5.0) < 1e-3                   # DSM = roof height
    c2, r2 = gt.pixel(698012.5, 9194012.5)                   # off-roof, ground visible
    c2, r2 = int(round(float(c2))), int(round(float(r2)))
    a2 = res.rgba[r2, c2, 3] / 255
    assert 0.3 < a2 < 0.7                                    # soft ground splat, alpha = o*exp(-q/2)
    assert abs(res.rgba[r2, c2, 0] / a2 - 128) < 6           # premultiplied grey
    assert np.isnan(res.dsm[r2, c2]) and abs(res.dsm_mean[r2, c2]) < 1e-3   # T never < 0.5 -> no median surface
    # elongated gaussian along the NE diagonal: bright NE (col+, row-), dark SE (col+, row+)
    c3, r3 = gt.pixel(698020.0, 9194020.0)
    c3, r3 = int(round(float(c3))), int(round(float(r3)))
    k = 12
    assert res.rgba[r3 - k, c3 + k, 1] > 100 and res.rgba[r3 + k, c3 + k, 1] < 60


def test_sh_nadir_basis():
    b = sh_basis((0, 0, -1), 3)
    expect = np.zeros(15)
    expect[1] = -C1
    expect[5] = 2 * C2[2]
    expect[11] = -2 * C3[3]
    assert np.allclose(b, expect)


@pytest.mark.skipif(not (c_available() or __import__("shutil").which("gcc")), reason="no C compiler")
def test_backends_agree(tmp_path):
    if not c_available():
        try:
            build_c(verbose=False)
        except RuntimeError as e:
            pytest.skip(f"C backend could not be built: {e}")
    p = scene(tmp_path)
    kw = dict(gsd=0.2, colour_mode="dc", epsg=32749, offset=(698000, 9194000, 0),
              bounds=(698000, 9194000, 698030, 9194030), max_scale=None, verbose=False, tile=64)
    a = render_orthophoto(p, backend="python", **kw)
    b = render_orthophoto(p, backend="c", **kw)
    assert np.abs(a.rgba.astype(int) - b.rgba.astype(int)).max() <= 1
    m = np.isfinite(a.dsm) & np.isfinite(b.dsm)
    assert np.allclose(a.dsm[m], b.dsm[m], atol=1e-4)


def test_geotiff_roundtrip(tmp_path):
    p = scene(tmp_path)
    res = render_orthophoto(p, gsd=0.5, colour_mode="dc", epsg=32749, offset=(698000, 9194000, 0),
                            bounds=(698000, 9194000, 698030, 9194030), backend="python", max_scale=None, verbose=False)
    files = res.save(tmp_path / "out")
    arr, gt = read_geotiff(files["ortho"])
    assert arr.shape == res.rgba.shape and gt.epsg == 32749
    assert abs(gt.x0 - 698000) < 1e-6 and abs(gt.y_top - 9194030) < 1e-6 and gt.gsd_x == 0.5
    meta = json.loads(files["meta"].read_text())
    assert meta["epsg"] == 32749
