# Changelog

## [0.1.0] - 2026-11-16
### Added
- Orthographic Gaussian rasterization operator: marginal (E, N) covariance projection, height-sorted
  front-to-back alpha compositing, SH colour at nadir or DC colour, median-depth DSM.
- Three interchangeable backends running the same kernel: Numba (parallel), C/OpenMP and pure Python.
- GeoSplat v0.2.0 support: `gaussians_crs*.json` (UTM offset, EPSG, valid region) and
  `camera_trajectory*.csv` (automatic height window against floaters).
- GeoTIFF output without GDAL (RGBA orthophoto, float32 DSM, JSON metadata with provenance).
- Validation tools: `check-z`, `check-xy`, `compare`.
- Desktop app (PySide6) with drag-and-drop, settings, stage list, preview, About / How to cite.
- Windows one-click launcher `DOG-GS.bat` and command-line drag-and-drop `windows/DOG-GS_render.bat`.
