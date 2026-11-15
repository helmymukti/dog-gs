# Example: ITS Research Center (Surabaya), GeoSplat v0.2.0

Input: `gaussians_utm_epsg32749_offset_hcorr.ply` (1,065,378 Gaussians, SH degree 3, 30k iterations)
from a single DJI Mavic 3E orbit video (12 June 2025, 216 frames at 2 fps), plus `gaussians_crs_hcorr.json`.

```bash
doggs render gaussians_utm_epsg32749_offset_hcorr.ply -o its --stem ITS_DOG-GS --gsd 0.06 --zmin 20 --zmax 110
doggs check-z  its/ITS_DOG-GS_dsm.tif   checkpoints_dz.csv --classes NVA,VVA
doggs check-xy its/ITS_DOG-GS_ortho.tif reference_ortho.tif gnss_cp.csv
doggs compare  its/ITS_DOG-GS_ortho.tif reference_ortho.tif
```

Result (C backend, 2 CPU cores): 948,811 Gaussians kept, 8455 x 8455 px at 0.06 m, render 34.5 s.
Horizontal (7 GNSS checkpoints): RMSE_H 2.52 m, dominated by a 2.43 m systematic offset (azimuth 321 deg)
from the non-RTK UAV GNSS track; leave-one-out translation-corrected RMSE_H 0.76 m.
Relative to an independent Metashape orthophoto: similarity scale 0.989, residual RMSE 0.44 m.
