"""Minimal GeoTIFF reading/writing with tifffile (no GDAL required)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tifffile


@dataclass
class GeoTransform:
    """North-up grid: x = x0 + (col + 0.5) * gsd ; y = y_top - (row + 0.5) * gsd (pixel centres)."""

    x0: float
    y_top: float
    gsd_x: float
    gsd_y: float
    epsg: int

    def pixel(self, x, y):
        return (np.asarray(x) - self.x0) / self.gsd_x - 0.5, (self.y_top - np.asarray(y)) / self.gsd_y - 0.5

    def coord(self, col, row):
        return self.x0 + (np.asarray(col) + 0.5) * self.gsd_x, self.y_top - (np.asarray(row) + 0.5) * self.gsd_y


def _geokeys(epsg: int):
    if epsg == 4326:
        return (1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326)
    return (1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, int(epsg))


def write_geotiff(path, arr, gt: GeoTransform, nodata=None, rgba=False, tile=512, description=None):
    tags = [(33550, "d", 3, (gt.gsd_x, gt.gsd_y, 0.0), False),
            (33922, "d", 6, (0.0, 0.0, 0.0, gt.x0, gt.y_top, 0.0), False),
            (34735, "H", 16, _geokeys(gt.epsg), False)]
    if nodata is not None:
        tags.append((42113, "s", 0, str(nodata), False))
    kw = dict(compression="deflate", tile=(tile, tile), extratags=tags, description=description)
    if rgba:
        kw.update(photometric="rgb", extrasamples=[2])
    tifffile.imwrite(str(path), arr, **kw)


def read_geotiff(path):
    """Read a north-up GeoTIFF. Palette images are expanded to RGB. Returns (array, GeoTransform)."""
    with tifffile.TiffFile(str(path)) as t:
        page = t.pages[0]
        arr = page.asarray()
        tags = page.tags
        scale = tags["ModelPixelScaleTag"].value
        tie = tags["ModelTiepointTag"].value
        epsg = None
        if t.is_geotiff:
            gm = t.geotiff_metadata or {}
            for k in ("ProjectedCSTypeGeoKey", "GeographicTypeGeoKey"):
                v = gm.get(k)
                if v is not None:
                    try:
                        epsg = int(getattr(v, "value", v))
                        break
                    except (TypeError, ValueError):
                        pass
        if page.photometric == 3 and page.colormap is not None:      # palette
            cm = page.colormap
            cm = (cm >> 8).astype(np.uint8) if cm.max() > 255 else cm.astype(np.uint8)
            arr = cm.T[arr]
    if tie[0] != 0 or tie[1] != 0:
        x0 = tie[3] - tie[0] * scale[0]
        yt = tie[4] + tie[1] * scale[1]
    else:
        x0, yt = tie[3], tie[4]
    return arr, GeoTransform(float(x0), float(yt), float(scale[0]), float(scale[1]), epsg or 0)
