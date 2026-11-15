"""DOG-GS: Direct Orthophoto Generation from Georeferenced 3D Gaussian Splatting.

Renders a true orthophoto (and a median-depth DSM) directly from a georeferenced
3D Gaussian Splatting model, e.g. the UTM export of GeoSplat v0.2.0, using an
orthographic (nadir) Gaussian rasterization operator with front-to-back alpha
compositing. No dense point cloud, mesh or DSM interpolation is required.
"""

__version__ = "0.1.0"

from .io_ply import read_gaussians, GaussianModel  # noqa: F401
from .geosplat import load_geosplat_crs  # noqa: F401
from .render import render_orthophoto, OrthoResult  # noqa: F401
