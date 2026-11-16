# DOG-GS — Direct Orthophoto Generation from Georeferenced 3D Gaussian Splatting

DOG-GS renders a **true orthophoto** and a **median-depth DSM** directly from a georeferenced
3D Gaussian Splatting (3DGS) model — for example the UTM export of
[GeoSplat](https://github.com/helmymukti/GeoSplat) v0.2.0 — with an orthographic (nadir) Gaussian
rasterization operator. No dense point cloud, mesh, DSM interpolation or orthorectification
step is needed: one rendering pass produces georeferenced GeoTIFFs.

On a 1.07 M-Gaussian model of the ITS Research Center (Surabaya), a 8455 × 8455 px orthophoto at
0.06 m GSD is rendered in about 35 s on a 2-core CPU (C backend, no GPU).

## Method

For a nadir orthographic camera, each Gaussian with mean **μ** and covariance
**Σ = R S Sᵀ Rᵀ** projects onto the horizontal plane as a 2-D Gaussian whose covariance is the
marginal of **Σ** on (E, N); no perspective Jacobian is involved:

```
Σ₂D = [Σ]_EN / gsd² + 0.3 I          (pixel units, 0.3 px² anti-aliasing low-pass)
```

Gaussians are sorted by decreasing height (front-to-back as seen from above) and composited:

```
α_i(p) = min(0.99, o_i · exp(−½ dᵀ Σ₂D⁻¹ d)),   skipped if α_i < 1/255
C(p)   = Σ_i c_i α_i T_i,   T_i = Π_{j<i} (1 − α_j),   stop when T < 10⁻⁴
DSM(p) = z_k   where T first drops below 0.5   (median-depth surface)
```

Colours `c_i` are spherical harmonics evaluated for the nadir viewing direction
d = (0, 0, −1) (`--colour sh`), or the view-independent DC term (`--colour dc`).
Pixel (row, col) is centred at `E = E₀ + (col + ½)·gsd`, `N = N_top − (row + ½)·gsd`.

## Installation

```bash
pip install .                 # numpy + tifffile only
pip install ".[all]"          # + numba (fast parallel backend) and opencv (validation tools)
pip install ".[all,gui]"      # + PySide6 desktop app  (run: doggs-gui  or  python -m doggs.gui)
```

Backends (`--backend auto` picks the first available):

| backend | requirement | notes |
|---|---|---|
| `numba` | `pip install numba` (included in Anaconda) | parallel JIT, recommended on Windows |
| `c` | a C compiler; run `doggs build-c` once | reference implementation, OpenMP |
| `python` | none | identical kernel in pure Python; for testing only (slow) |

All three run the same compositing kernel; the test-suite checks that they agree.

## Desktop app (Windows)

1. Double-click **`DOG-GS.bat`**. The first run creates a private `.venv` in this folder and installs
   DOG-GS with Numba, OpenCV and PySide6 (Python 3.9+ and an internet connection are needed once).
2. **Drop a GeoSplat project folder** (or any georeferenced Gaussian `.ply`) into the window.
   From a GeoSplat folder the vertically corrected UTM model `…/10_Vertical_Correction/gaussians_utm_…_hcorr.ply`
   and the GeoSplat checkpoint file are picked automatically. Optionally drop an RTK checkpoint `.csv`
   and a reference orthophoto `.tif` as well.
3. Check the settings (GSD, colour mode, backend, height window — *Automatic* removes floaters near
   the camera path) and press **Start**.

The app shows the six processing stages, a log and a preview, and writes everything to
`<project>\11_DOG-GS\` (or `DOG-GS_output\` next to a single PLY):
`*_ortho.tif`, `*_dsm.tif`, `*_meta.json`, `*_report.json` and, when checkpoints were given, `*_check_z.csv`.
You can also drag a folder or PLY onto `DOG-GS.bat` itself, or use `windows/DOG-GS_render.bat` for a
command-line run without the window.

## Usage

```bash
# render orthophoto + DSM from a GeoSplat UTM export (gaussians_crs*.json is auto-detected)
doggs render gaussians_utm_epsg32749_offset_hcorr.ply -o out --stem ITS \
      --gsd 0.06 --zmin 20 --zmax 110

# any 3DGS PLY in a projected CRS
doggs render point_cloud.ply --epsg 32749 --offset "698000 9194000 0" --gsd 0.05 --zmin auto --zmax auto

# validation (needs opencv-python)
doggs check-z  out/ITS_dsm.tif   rtk_points.csv --classes NVA,VVA --csv dz.csv
doggs check-xy out/ITS_ortho.tif reference_ortho.tif gnss_checkpoints.csv --json xy.json
doggs compare  out/ITS_ortho.tif reference_ortho.tif --json compare.json
doggs info
```

Outputs: `<stem>_ortho.tif` (RGBA, band 4 = accumulated opacity), `<stem>_dsm.tif`
(float32, nodata −9999 where opacity < 0.5), `<stem>_meta.json` (all parameters, filter counts,
timings and the GeoSplat provenance). GeoTIFFs carry EPSG georeferencing and open directly in
QGIS/ArcGIS; GDAL is not required.

`render` options of note: `--zmin/--zmax` remove floaters (e.g. Gaussians near the camera
trajectory), `--max-scale` drops oversized primitives, `--mask-region` clips to the GeoSplat valid
region, `--bounds` sets an explicit extent, `--tile` controls memory use.

Point CSVs need columns `id`, `E`/`easting`, `N`/`northing`, and for `check-z` a height column
(`h`, `h_ellipsoidal`, `h_checkpoint`, `z`) plus optional `class`/`role`.

### Validation tools

* **check-z** – median DSM height within a radius (default 0.5 m) at each surveyed point; reports
  mean, RMSE, SD, NMAD.
* **check-xy** – horizontal residuals at checkpoints whose coordinates fall on the surveyed feature
  in a reference orthophoto; each feature is located in the DOG-GS orthophoto by NCC of gradient
  magnitudes (6/9/12 m templates, ±8 m search). Reports RMSE_E/N/H, CE90
  (= 2.1460·(RMSE_E+RMSE_N)/2), NSSDA 95 % (= 1.7308·RMSE_H), bias magnitude/azimuth, dispersion and
  leave-one-out translation-corrected RMSE.
* **compare** – relative geometry against a reference orthophoto (AKAZE + SIFT on CLAHE images at
  1/2 and 1/3 resolution, ratio test 0.75, local-consensus filtering), similarity and affine fits,
  coverage, and SSIM/PSNR/NCC/MSE/MAE before and after similarity co-registration. References in
  EPSG:4326 (including palette TIFFs) are reprojected on the fly.

## Python API

```python
from doggs import render_orthophoto
res = render_orthophoto("gaussians_utm_epsg32749_offset_hcorr.ply", gsd=0.06, zmin=20, zmax=110)
res.save("out", stem="ITS")
```

## Tests

```bash
pip install ".[test]" && pytest
```

## Authors

| Name | Affiliation |
|---|---|
| **Helmy Mukti Wijaya**<sup>1,*</sup> | Department of Geomatics Engineering, Institut Teknologi Sepuluh Nopember (ITS), Surabaya, Indonesia |
| Hepi Hapsari Handayani<sup>1</sup> | Department of Geomatics Engineering, Institut Teknologi Sepuluh Nopember (ITS), Surabaya, Indonesia |
| Teguh Hariyanto<sup>1</sup> | Department of Geomatics Engineering, Institut Teknologi Sepuluh Nopember (ITS), Surabaya, Indonesia |
| Harintaka<sup>2</sup> | Department of Geomatics Engineering, Universitas Gadjah Mada (UGM), Yogyakarta, Indonesia |

<sup>*</sup> Corresponding author: helmy.mukti.wijaya@gmail.com

## How to cite

> Wijaya, H. M., Handayani, H. H., Hariyanto, T., & Harintaka (2026). DOG-GS: Direct Orthophoto Generation
> from Georeferenced 3D Gaussian Splatting (Version 0.1.0) [Computer software]. https://github.com/helmymukti/dog-gs

BibTeX and the full citation metadata are in `CITATION.cff` (also available from *Help → How to cite* in
the app). Please also cite 3D Gaussian Splatting (Kerbl et al., 2023, *ACM TOG* 42(4)) and, when the
model was produced with it, GeoSplat (https://github.com/helmymukti/GeoSplat).

## License

MIT — see `LICENSE`.
