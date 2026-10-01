"""The simulated camera: turbulence, optics, stars, a sensor model, and faults.

Build a camera with `create` (the factory that `seeingmon.drivers.create_driver("sim", ...)`
calls) or with `sim_camera`, which sets up one turbulent layer of a known strength. Read the
injected truth from `driver.truth`.

    clock = VirtualClock()
    driver = sim_camera(clock, r0_m=0.10, wind_speed_m_s=8.0, outer_scale_m=20.0, seed=3)
    driver.open()
    driver.configure(StreamConfig("bin1", 2000, 120, roi=Roi(4000, 2700, 128, 128)))
    driver.start()
    frame = driver.read_frame(timeout_s=1.0)
    truth = driver.truth.frames[-1]  # the true image motion of this frame, in arcseconds

Importing this package needs SciPy (the `fast` extra).
"""

from __future__ import annotations

from seeingmon.drivers.sim.detector import HotPixelConfig
from seeingmon.drivers.sim.driver import SimDriver, create, sim_camera
from seeingmon.drivers.sim.faults import GeometryChange, SimFaults
from seeingmon.drivers.sim.optics import PsfConfig
from seeingmon.drivers.sim.options import SimOptions
from seeingmon.drivers.sim.params import GainPoint, SimParams
from seeingmon.drivers.sim.sky import (
    SYNTHETIC_SITE,
    CloudEvent,
    Clouds,
    ScintillationConfig,
    Site,
)
from seeingmon.drivers.sim.stars import (
    Pointing,
    SkyProjector,
    StarField,
    make_polar_field,
    polaris_field,
)
from seeingmon.drivers.sim.truth import FrameTruth, SimTruth, StarPositions
from seeingmon.drivers.sim.turbulence import Layer, TurbulenceConfig, TurbulenceModel

__all__ = [
    "SYNTHETIC_SITE",
    "CloudEvent",
    "Clouds",
    "FrameTruth",
    "GainPoint",
    "GeometryChange",
    "HotPixelConfig",
    "Layer",
    "Pointing",
    "PsfConfig",
    "ScintillationConfig",
    "SimDriver",
    "SimFaults",
    "SimOptions",
    "SimParams",
    "SimTruth",
    "Site",
    "SkyProjector",
    "StarField",
    "StarPositions",
    "TurbulenceConfig",
    "TurbulenceModel",
    "create",
    "make_polar_field",
    "polaris_field",
    "sim_camera",
]
