"""The catalog stars that a camera attitude sees, in their apparent places.

`catalog_field` takes an attitude and the time of a frame, finds the catalog stars inside the
field (a cone query on the catalog positions, padded for the proper motion that the catalog
epoch leaves out), and returns their rows and apparent unit vectors. The tracker, the
analyzer, and the sky quality step all start from it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from seeingmon.survey import apparent
from seeingmon.survey.apparent import ObservationEpoch
from seeingmon.survey.catalog import CapCatalog
from seeingmon.survey.geometry import FloatArray
from seeingmon.survey.wcs_fit import CameraAttitude

# Catalog stars this much farther than the half diagonal of the frame can still match.
FIELD_MARGIN_DEG = 0.15


def half_diagonal_deg(attitude: CameraAttitude, width_px: int, height_px: int) -> float:
    """Half the diagonal of a frame, in degrees, for an attitude's plate scale."""
    return float(np.degrees(0.5 * np.hypot(width_px, height_px) * attitude.scale_rad_px))


def catalog_field(
    catalog: CapCatalog,
    attitude: CameraAttitude,
    epoch: ObservationEpoch,
    *,
    width_px: int,
    height_px: int,
    max_g_mag: float | None = None,
    margin_deg: float = FIELD_MARGIN_DEG,
) -> tuple[npt.NDArray[np.intp], FloatArray]:
    """The catalog rows near the field of an attitude, and their apparent CIRS vectors.

    The rows are indices into the catalog, brightest first. The vectors have shape `(N, 3)`
    and hold the places at `epoch` after proper motion, parallax, light deflection,
    aberration, precession, and nutation.
    """
    center = apparent.astrometric_from_apparent(attitude.boresight(), epoch)
    radius = half_diagonal_deg(attitude, width_px, height_px) + margin_deg
    rows = catalog.cone(center, radius, max_g_mag=max_g_mag)
    if rows.size == 0:
        return rows, np.zeros((0, 3))
    vectors = apparent.apparent_vectors(
        catalog.ra_deg[rows],
        catalog.dec_deg[rows],
        catalog.pm_ra_mas_yr[rows],
        catalog.pm_dec_mas_yr[rows],
        catalog.parallax_mas[rows],
        epoch,
        catalog_epoch_jyear=catalog.epoch_jyear,
    )
    return rows, vectors
