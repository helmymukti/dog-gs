"""Command-line interface:  doggs {render,check-z,check-xy,compare,build-c,info}."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _floats(s, n):
    v = [float(x) for x in s.replace(",", " ").split()]
    if len(v) != n:
        raise argparse.ArgumentTypeError(f"expected {n} numbers")
    return v


def resolve_input(path):
    """Accept a PLY file or a GeoSplat project folder; return (ply_path, default_output_dir)."""
    p = Path(path)
    if p.is_file():
        return p, p.parent / "DOG-GS_output"
    if not p.is_dir():
        raise SystemExit(f"input not found: {p}")
    for pattern in ("gaussians_utm*_hcorr.ply", "gaussians_utm*.ply", "point_cloud.ply"):
        hits = sorted(p.rglob(pattern), key=lambda q: (len(q.parts), str(q)))
        if pattern == "point_cloud.ply":
            hits = sorted(hits, key=lambda q: str(q), reverse=True)     # prefer the latest iteration
        if hits:
            print("input PLY:", hits[0])
            return hits[0], p / "11_DOG-GS"
    raise SystemExit(f"no Gaussian PLY found in {p}")


def cmd_render(a):
    from .render import render_orthophoto
    ply, default_out = resolve_input(a.ply)
    a.ply = str(ply)
    if a.out is None:
        a.out = str(default_out)
    res = render_orthophoto(a.ply, gsd=a.gsd, colour_mode=a.colour, crs_json=a.crs_json, epsg=a.epsg,
                            offset=_floats(a.offset, 3) if a.offset else None,
                            bounds=_floats(a.bounds, 4) if a.bounds else None,
                            use_region=not a.no_region, region_scale=a.region_scale,
                            mask_outside_region=a.mask_region, zmin=a.zmin, zmax=a.zmax,
                            max_scale=a.max_scale, min_opacity=a.min_opacity, backend=a.backend,
                            tile=a.tile, threads=a.threads, verbose=not a.quiet)
    files = res.save(a.out, stem=a.stem, min_alpha=a.min_alpha, write_mean=a.dsm_mean)
    for k, v in files.items():
        print(f"{k:9s} {v}")


def cmd_check_z(a):
    from .validate import check_z, read_points, to_csv
    pts = read_points(a.points)
    rows, st = check_z(a.dsm, pts, radius=a.radius, classes=a.classes.split(",") if a.classes else None)
    for r in rows:
        print(f"{r['id']:>8} {r['cls']:>10} h={r['h']:.3f} z={r['z_dsm']:.3f} dz={r['dz']:+.3f}")
    print(json.dumps(st, indent=2))
    if a.csv:
        to_csv(rows, a.csv)


def cmd_check_xy(a):
    from .validate import check_xy, read_points, to_csv, dump_json
    pts = read_points(a.points)
    rows, st, info = check_xy(a.ortho, a.reference, pts, search_m=a.search, min_ncc=a.min_ncc)
    for r in rows:
        print(f"{r['id']:>8} ncc={r['ncc']:.2f} dE={r['dE']:+.3f} dN={r['dN']:+.3f} {'used' if r['used'] else 'excluded'}")
    print(json.dumps(dict(stats=st, **info), indent=2))
    if a.csv:
        to_csv(rows, a.csv)
    if a.json:
        dump_json(dict(points=rows, stats=st, **info), a.json)


def cmd_compare(a):
    from .validate import compare, dump_json
    r = compare(a.ortho, a.reference, min_alpha=a.min_alpha, out_npz=a.npz)
    print(json.dumps(r, indent=2, default=float))
    if a.json:
        dump_json(r, a.json)


def cmd_build_c(a):
    from .kernel import build_c
    build_c()


def cmd_info(a):
    from .kernel import c_available, numba_available, pick_backend
    print(f"dog-gs {__version__}; numba: {numba_available()}; C backend built: {c_available()}; auto backend: {pick_backend()}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="doggs", description="DOG-GS: direct orthophoto generation from georeferenced 3D Gaussian Splatting")
    p.add_argument("--version", action="version", version=f"dog-gs {__version__}")
    sp = p.add_subparsers(dest="cmd", required=True)

    r = sp.add_parser("render", help="render orthophoto + DSM GeoTIFFs from a georeferenced 3DGS PLY")
    r.add_argument("ply", help="Gaussian PLY file, or a GeoSplat project folder")
    r.add_argument("-o", "--out", help="output folder (default: <project>/11_DOG-GS or <ply folder>/DOG-GS_output)")
    r.add_argument("--stem", default="doggs")
    r.add_argument("--gsd", type=float, default=0.06, help="ground sampling distance in metres (default 0.06)")
    r.add_argument("--colour", "--color", choices=["sh", "dc"], default="sh", help="SH at nadir (sh) or view-independent DC")
    r.add_argument("--crs-json", help="GeoSplat gaussians_crs*.json (auto-detected next to the PLY)")
    r.add_argument("--epsg", type=int, help="EPSG of the PLY frame (if no GeoSplat json)")
    r.add_argument("--offset", help="'dx dy dz' added to PLY coordinates to obtain true coordinates")
    r.add_argument("--bounds", help="'xmin ymin xmax ymax' in the true CRS")
    r.add_argument("--no-region", action="store_true", help="ignore the GeoSplat valid region")
    r.add_argument("--region-scale", type=float, default=1.0)
    r.add_argument("--mask-region", action="store_true", help="set alpha=0 outside the valid region")
    r.add_argument("--zmin", type=float, help="drop Gaussians below this height (true datum)")
    r.add_argument("--zmax", type=float, help="drop Gaussians above this height (true datum), e.g. floaters")
    r.add_argument("--max-scale", type=float, default=20.0, help="drop Gaussians with an axis std > this (m)")
    r.add_argument("--min-opacity", type=float, default=1 / 255)
    r.add_argument("--min-alpha", type=float, default=0.5, help="DSM nodata where accumulated opacity < this")
    r.add_argument("--dsm-mean", action="store_true", help="also write the opacity-weighted mean-height DSM")
    r.add_argument("--backend", choices=["auto", "numba", "c", "python"], default="auto")
    r.add_argument("--tile", type=int, default=1024)
    r.add_argument("--threads", type=int)
    r.add_argument("-q", "--quiet", action="store_true")
    r.set_defaults(fn=cmd_render)

    z = sp.add_parser("check-z", help="DSM heights vs surveyed points")
    z.add_argument("dsm"); z.add_argument("points")
    z.add_argument("--radius", type=float, default=0.5)
    z.add_argument("--classes", help="comma-separated classes/roles to include")
    z.add_argument("--csv")
    z.set_defaults(fn=cmd_check_z)

    x = sp.add_parser("check-xy", help="horizontal residuals at checkpoints via a reference orthophoto")
    x.add_argument("ortho"); x.add_argument("reference"); x.add_argument("points")
    x.add_argument("--search", type=float, default=8.0)
    x.add_argument("--min-ncc", type=float, default=0.5)
    x.add_argument("--csv"); x.add_argument("--json")
    x.set_defaults(fn=cmd_check_xy)

    c = sp.add_parser("compare", help="relative geometry and radiometry vs a reference orthophoto")
    c.add_argument("ortho"); c.add_argument("reference")
    c.add_argument("--min-alpha", type=float, default=0.8)
    c.add_argument("--json"); c.add_argument("--npz")
    c.set_defaults(fn=cmd_compare)

    b = sp.add_parser("build-c", help="compile the optional C backend")
    b.set_defaults(fn=cmd_build_c)
    i = sp.add_parser("info", help="show available backends")
    i.set_defaults(fn=cmd_info)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
