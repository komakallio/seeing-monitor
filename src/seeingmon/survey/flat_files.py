"""Read and write the flat files that `[survey] flat_file` names.

A flat file is a 2-D image of the whole sensor of the survey mode, scaled to a median of 1. It is
a NumPy file (`.npy`) or a FITS image, and `seeingmon.survey.sky.load_flat` reads both. The
writer here makes either one atomically (`seeingmon.store.layout.write_atomic`), so a reader never
sees a half-written flat.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import numpy.typing as npt

from seeingmon.solvers import fitsio
from seeingmon.store.layout import write_atomic
from seeingmon.survey.sky import ArrayFlat, SkyError, load_flat

NPY_SUFFIXES = frozenset({".npy"})
FITS_SUFFIXES = frozenset({".fits", ".fit", ".fts"})


class FlatFileError(Exception):
    """A flat file cannot be written or read. The message never names a path."""


def write_flat(path: Path, flat: npt.NDArray[np.float32]) -> None:
    """Write a flat as `float32` to a `.npy` or a FITS file, depending on the suffix.

    The write is atomic. A suffix that is neither raises `FlatFileError`.
    """
    suffix = path.suffix.lower()
    array = np.ascontiguousarray(flat, dtype=np.float32)
    try:
        if suffix in NPY_SUFFIXES:
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            write_atomic(path, buffer.getvalue())
        elif suffix in FITS_SUFFIXES:
            write_atomic(path, fitsio.image_bytes(array))
        else:
            raise FlatFileError("the output file must end in .npy, .fits, or .fit")
    except OSError as error:
        raise FlatFileError(f"cannot write the flat: {error.strerror or error}") from None


def read_flat_image(path: Path) -> npt.NDArray[np.float32]:
    """Read a flat file and return its image, scaled to a median of 1.

    The result is the image that `[survey] flat_file` would give the pipeline.
    """
    try:
        flat = load_flat(path)
    except SkyError as error:
        raise FlatFileError(str(error)) from None
    if not isinstance(flat, ArrayFlat):
        raise FlatFileError("the flat file is empty")
    return np.asarray(flat.image(flat.shape), dtype=np.float32)
