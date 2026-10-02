"""The flat commands: `seeingmon flat make` and `seeingmon flat build`.

`seeingmon flat make` combines frames of a lit panel into the master flat for `[survey] flat_file`
(`seeingmon.survey.flat_make`). `seeingmon flat build` builds a flat from the survey frames of
the night sky (`seeingmon.survey.flat_sky`), or compares the sky with a panel flat and updates
it (`--base-flat`, `seeingmon.survey.flat_base`). Both run offline, on one frame at a time, and
print a plain-text report that names no path of the machine.

This module only defines the arguments and reads the configuration. The handlers import the heavy
modules when they run, so `seeingmon --help` stays fast.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from seeingmon.cli import CliError, Subparsers, add_command

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from seeingmon.config import Config
    from seeingmon.profile import Profile
    from seeingmon.survey.config import SurveyConfig


def register_flat(subparsers: Subparsers) -> None:
    parser = add_command(
        subparsers,
        "flat",
        help="Make or build the flat field of the survey frames.",
        handler=_missing_subcommand,
    )
    commands = parser.add_subparsers(dest="flat_command", metavar="<subcommand>", required=True)
    _register_make(commands)
    _register_build(commands)


def _missing_subcommand(args: argparse.Namespace) -> int:
    # `required=True` makes argparse reject a missing subcommand before this runs.
    raise CliError("choose a subcommand: make or build", exit_code=2)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--high-pass-px",
        type=float,
        default=40.0,
        help="the width of the Gaussian that separates the smooth part from the fine part, in "
        "binned pixels (default 40)",
    )
    parser.add_argument(
        "--bin", type=int, default=4, help="the binning of the analysis, in pixels (default 4)"
    )
    parser.add_argument(
        "--center-x",
        type=float,
        help="the optical center, x in sensor pixels (default: the middle of the frame)",
    )
    parser.add_argument(
        "--center-y",
        type=float,
        help="the optical center, y in sensor pixels (default: the middle of the frame)",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        help="the calibration folder with the dark library in darks/ (default: calibration_dir in "
        "[survey], or calibration/ in the data directory)",
    )
    parser.add_argument(
        "--local-config",
        type=Path,
        help="read this file instead of local/config.toml (an absent file is ignored)",
    )


def _register_make(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    make = commands.add_parser(
        "make",
        help="Combine frames of a lit panel into the master flat.",
        description=(
            "Combine panel frames into the master flat for [survey] flat_file. Pass a SER file "
            "or a folder of FITS files with --frames, once for each set of frames. Each frame "
            "needs a mean level of 20 to 80% of full scale within 5% of the median of its set, "
            "and the frames must be in the survey mode with no region of interest. The command "
            "subtracts the bias level from the dark library (or --bias-level, or bias frames "
            "that pass a check), averages each set with a sigma clip, averages the sets, and "
            "writes a float32 image with a median of 1. It prints the vignetting, the tilt, the "
            "shadows, and how well the sets agree. A light source has a gradient of its own: "
            "take a second set with the source turned by 180 degrees, pass it as a second "
            "--frames, and add --source-turned."
        ),
    )
    make.add_argument(
        "--frames",
        action="append",
        required=True,
        type=Path,
        help="a SER file or a folder of FITS files (repeat it for each set of frames)",
    )
    make.add_argument(
        "--out", required=True, type=Path, help="the flat to write (.npy, or FITS: .fits)"
    )
    make.add_argument(
        "--source-turned",
        action="store_true",
        help="you turned the light source by 180 degrees between the sets",
    )
    make.add_argument(
        "--bias-level",
        type=float,
        help="the bias level in the counts of the frames (default: the dark library)",
    )
    make.add_argument(
        "--bias",
        type=Path,
        help="bias frames (a SER file or a folder of FITS files) taken with the lens covered. "
        "The command checks them, and uses another bias level when they hold light.",
    )
    make.add_argument(
        "--temperature-c",
        type=float,
        help="the sensor temperature of the frames, for the bias of the dark library (default: "
        "CCD-TEMP of the FITS headers)",
    )
    make.add_argument(
        "--gain",
        type=int,
        help="the gain of the frames, for the bias of the dark library (default: GAIN of the "
        "FITS headers, or the gain in [survey.dark])",
    )
    make.add_argument(
        "--full-scale",
        type=float,
        help="the count where the ADC saturates (default: found from the frames)",
    )
    make.add_argument(
        "--byte-order",
        choices=("little", "big"),
        help="the byte order of the 16-bit pixels of a SER file (default: its header says)",
    )
    make.add_argument("--min-level-percent", type=float, default=20.0, help="default 20")
    make.add_argument("--max-level-percent", type=float, default=80.0, help="default 80")
    make.add_argument(
        "--flicker-percent",
        type=float,
        default=5.0,
        help="drop a frame whose level differs from the median of its set by more than this "
        "(default 5)",
    )
    make.add_argument(
        "--min-frames",
        type=int,
        default=15,
        help="warn when a set has fewer frames that pass (default 15)",
    )
    _add_common(make)
    make.set_defaults(handler=_make)


def _register_build(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    build = commands.add_parser(
        "build",
        help="Build a flat from the survey frames of the night sky.",
        description=(
            "Build a flat from the survey frames that core keeps as FITS files (under survey/ of "
            "the data directory). The camera is fixed to the ground and points at the pole, so "
            "the sky turns about the middle of the frame, and the mean of many star-masked frames "
            "holds the vignetting and the shadows of the dust. The sky cannot give the tilt of "
            "the flat, so the result has none. A frame counts when the Sun is below -18 degrees, "
            "the Moon is down or under 25% lit, the cloud fraction is under 0.1, the "
            "transparency is at least 0.95, and the sky level lies within 10% of the median. "
            "The Sun and the Moon come from [site] in the configuration. --accumulator keeps the "
            "sums in a file, so that a later run adds only the new frames. The command writes a "
            "float32 image with a median of 1 and prints a report. With --base-flat, a flat that "
            "you took with a panel, the command divides the mean sky by it and reports what "
            "changed: new shadows, a change of the vignetting of more than 1%, and the plane. It "
            "writes nothing then, unless you add --update: the new flat is the base flat with the "
            "changes that exceed their limits, and its tilt is always the tilt of the base flat."
        ),
    )
    build.add_argument(
        "frames",
        nargs="?",
        type=Path,
        help="the folder with the survey frames (searched for FITS files, also in subfolders)",
    )
    build.add_argument(
        "--out",
        type=Path,
        help="the flat to write (.npy, or FITS: .fits). With --base-flat, give it with --update",
    )
    build.add_argument(
        "--base-flat",
        type=Path,
        help="a flat that you took with a panel (.npy or FITS): the command divides the mean sky "
        "by it and reports what changed. The tilt comes from this flat",
    )
    build.add_argument(
        "--update",
        action="store_true",
        help="with --base-flat, write the base flat with the changes that the sky shows and that "
        "exceed their limits (new shadows, bright patches, and a radial change of more than 1%%) "
        "to --out. Without it, the command only reports",
    )
    build.add_argument(
        "--radial-limit-percent",
        type=float,
        default=1.0,
        help="with --base-flat, the change of the radial profile at one of the five radii that "
        "the update takes from the sky (default 1)",
    )
    build.add_argument(
        "--accumulator",
        type=Path,
        help="the file that keeps the running sums and the frames already added (default: none)",
    )
    build.add_argument(
        "--min-frames",
        type=int,
        default=20,
        help="warn when fewer frames than this went in (default 20)",
    )
    build.add_argument(
        "--min-roll-deg",
        type=float,
        default=60.0,
        help="warn when the frames cover less roll than this (default 60)",
    )
    build.add_argument(
        "--polaris-mask-px",
        type=float,
        default=400.0,
        help="the radius of the disk that hides Polaris and its halo, in pixels (default 400)",
    )
    build.add_argument(
        "--max-sun-elevation",
        type=float,
        default=-18.0,
        help="take only frames with the Sun below this elevation, in degrees (default -18)",
    )
    build.add_argument(
        "--max-moon-illumination",
        type=float,
        default=0.25,
        help="take a frame with the Moon up only when its lit fraction is at most this "
        "(default 0.25)",
    )
    build.add_argument(
        "--moon-min-elevation",
        type=float,
        default=0.0,
        help="the Moon counts as up above this elevation, in degrees (default 0)",
    )
    build.add_argument(
        "--max-cloud-fraction",
        type=float,
        default=0.1,
        help="take only frames with a smaller cloud fraction (default 0.1)",
    )
    build.add_argument(
        "--min-transparency",
        type=float,
        default=0.95,
        help="take only frames with at least this transparency (default 0.95)",
    )
    build.add_argument(
        "--sky-tolerance-percent",
        type=float,
        default=10.0,
        help="take only frames whose sky level lies within this percentage of the median of the "
        "frames (default 10)",
    )
    build.add_argument(
        "--min-exposure-s",
        type=float,
        default=5.0,
        help="leave out frames with a shorter exposure, in seconds (default 5)",
    )
    build.add_argument(
        "--accept-unchecked",
        action="store_true",
        help="take a frame whose header lacks the cloud fraction or the transparency, or when "
        "the configuration has no [site] to rule out the Sun and the Moon",
    )
    _add_common(build)
    build.set_defaults(handler=_build)


# --- Reading the configuration --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FlatContext:
    """What the commands read from the configuration: the profile and the survey settings."""

    profile: Profile
    survey: SurveyConfig
    calibration_dir: Path | None
    config: Config


def _load_context(args: argparse.Namespace) -> FlatContext:
    """Read the profile, the `[survey]` table, and the location of the dark library."""
    from seeingmon.config import ConfigError, load_config
    from seeingmon.profile import ProfileError
    from seeingmon.store.layout import DataLayout
    from seeingmon.survey.config import SurveyConfig
    from seeingmon.survey.dark import CALIBRATION_DIRNAME

    try:
        config = load_config(local_file=args.local_config)
        survey = config.section("survey", SurveyConfig)
        profile = config.profile
    except (ConfigError, ProfileError) as exc:
        raise CliError(str(exc)) from None
    calibration: Path | None = None
    if args.calibration_dir is not None:
        calibration = Path(args.calibration_dir)
    elif survey.calibration_dir:
        calibration = Path(survey.calibration_dir)
    else:
        try:
            calibration = DataLayout.from_config(config).root / CALIBRATION_DIRNAME
        except ConfigError:
            calibration = None
    return FlatContext(profile, survey, calibration, config)


def _center(args: argparse.Namespace) -> tuple[float, float] | None:
    if (args.center_x is None) != (args.center_y is None):
        raise CliError("pass --center-x and --center-y together", exit_code=2)
    if args.center_x is None:
        return None
    return (float(args.center_x), float(args.center_y))


def _check_output_name(path: Path) -> None:
    from seeingmon.survey.flat_files import FITS_SUFFIXES, NPY_SUFFIXES

    if path.suffix.lower() not in NPY_SUFFIXES | FITS_SUFFIXES:
        raise CliError("--out must end in .npy, .fits, or .fit", exit_code=2)


# --- flat make ------------------------------------------------------------------------------


def _make(args: argparse.Namespace) -> int:
    from seeingmon.clock import SystemClock
    from seeingmon.survey import flat_make as fm
    from seeingmon.survey.dark import DARKS_DIRNAME, DarkLibrary
    from seeingmon.survey.flat_files import FlatFileError, write_flat

    out = Path(args.out)
    _check_output_name(out)
    center = _center(args)
    try:
        options = fm.MakeOptions(
            min_level_percent=args.min_level_percent,
            max_level_percent=args.max_level_percent,
            flicker_percent=args.flicker_percent,
            min_frames=args.min_frames,
            full_scale=args.full_scale,
            bin_factor=args.bin,
            high_pass_px=args.high_pass_px,
            center_xy=center,
        )
    except ValueError as exc:
        raise CliError(str(exc), exit_code=2) from None
    context = _load_context(args)
    readout = context.profile.survey_readout
    geometry = fm.Geometry(
        shape=(readout.height_px, readout.width_px),
        scale_arcsec_px=context.profile.plate_scale_arcsec_per_px(readout),
        adc_bits=readout.adc_bits,
    )
    sources: list[fm.FrameSource] = []
    bias_frames: fm.FrameSource | None = None
    try:
        sources.extend(fm.open_frames(Path(p), byte_order=args.byte_order) for p in args.frames)
        if args.bias is not None:
            bias_frames = fm.open_frames(Path(args.bias), byte_order=args.byte_order)
        temperatures = [s.temperature_c for s in sources if s.temperature_c is not None]
        temperature = args.temperature_c
        if temperature is None and temperatures:
            temperature = sum(temperatures) / len(temperatures)
        gains = [s.gain for s in sources if s.gain is not None]
        gain = (
            args.gain
            if args.gain is not None
            else (gains[0] if gains else context.survey.dark.gain)
        )
        library = None
        if context.calibration_dir is not None:
            library = fm.library_bias(
                DarkLibrary(context.calibration_dir / DARKS_DIRNAME),
                mode=context.profile.survey_mode.mode,
                gain=int(gain),
                temperature_c=temperature,
                doubling_c=context.survey.dark.doubling_c,
            )
        result = fm.make_flat(
            sources,
            bias=fm.BiasInput(level=args.bias_level, frames=bias_frames, library=library),
            geometry=geometry,
            options=options,
            source_turned=args.source_turned,
            clock=SystemClock(),
        )
        try:
            write_flat(out, result.flat)
        except FlatFileError as exc:
            raise CliError(str(exc)) from None
    except fm.FlatError as exc:
        raise CliError(str(exc)) from None
    finally:
        for source in sources:
            source.close()
        if bias_frames is not None:
            bias_frames.close()
    for line in fm.format_make_report(result, name=out.name):
        print(line)
    return 0


# --- flat build -----------------------------------------------------------------------------


def _load_hot_pixels(survey: SurveyConfig) -> npt.NDArray[np.bool_] | None:
    """The hot-pixel mask that `[survey] hot_pixel_file` names, or `None`."""
    if not survey.hot_pixel_file:
        return None
    from seeingmon.survey.pipeline import load_hot_pixels

    try:
        return load_hot_pixels(survey.hot_pixel_file)
    except (OSError, ValueError) as exc:
        raise CliError(f"cannot read the hot-pixel file: {type(exc).__name__}") from None


def _build(args: argparse.Namespace) -> int:
    from seeingmon.clock import SystemClock
    from seeingmon.config import ConfigError
    from seeingmon.scheduler.config import load_site
    from seeingmon.survey import flat_sky as fs
    from seeingmon.survey.dark import DARKS_DIRNAME, DarkLibrary
    from seeingmon.survey.flat_files import FlatFileError, read_flat_image, write_flat

    base_path = None if args.base_flat is None else Path(args.base_flat)
    if args.update and base_path is None:
        raise CliError("--update needs --base-flat, the flat to update", exit_code=2)
    if base_path is not None and not args.update and args.out is not None:
        raise CliError(
            "without --update the command only reports: add --update, or leave out --out",
            exit_code=2,
        )
    if args.out is None and (base_path is None or args.update):
        raise CliError("give --out, the flat to write", exit_code=2)
    out = None if args.out is None else Path(args.out)
    if out is not None:
        _check_output_name(out)
        if base_path is not None and out.resolve() == base_path.resolve():
            raise CliError(
                "--out names the base flat: write the new flat to another file, so that the "
                "panel flat stays",
                exit_code=2,
            )
    if args.frames is None and args.accumulator is None:
        raise CliError(
            "give the folder of frames, or an accumulator that holds frames", exit_code=2
        )
    center = _center(args)
    try:
        options = fs.BuildOptions(
            bin_factor=args.bin,
            high_pass_px=args.high_pass_px,
            polaris_mask_px=args.polaris_mask_px,
            center_xy=center,
            min_frames=args.min_frames,
            min_roll_deg=args.min_roll_deg,
            max_sun_elevation_deg=args.max_sun_elevation,
            max_moon_illumination=args.max_moon_illumination,
            moon_min_elevation_deg=args.moon_min_elevation,
            max_cloud_fraction=args.max_cloud_fraction,
            min_transparency=args.min_transparency,
            sky_tolerance=args.sky_tolerance_percent / 100.0,
            min_exposure_s=args.min_exposure_s,
            accept_unchecked=args.accept_unchecked,
            radial_limit=args.radial_limit_percent / 100.0,
        )
    except ValueError as exc:
        raise CliError(str(exc), exit_code=2) from None
    base_flat = None
    if base_path is not None:
        try:
            base_flat = read_flat_image(base_path)
        except FlatFileError as exc:
            raise CliError(str(exc)) from None
    context = _load_context(args)
    try:
        site = load_site(context.config)
    except ConfigError as exc:
        raise CliError(str(exc)) from None
    library = None
    if context.calibration_dir is not None:
        library = DarkLibrary(context.calibration_dir / DARKS_DIRNAME)

    def progress(message: str) -> None:
        print(message, flush=True)

    try:
        result = fs.build_sky_flat(
            None if args.frames is None else Path(args.frames),
            profile=context.profile,
            survey=context.survey,
            library=library,
            site=site,
            hot_mask=_load_hot_pixels(context.survey),
            accumulator_path=None if args.accumulator is None else Path(args.accumulator),
            options=options,
            clock=SystemClock(),
            progress=progress,
            base_flat=base_flat,
            update=args.update,
        )
        if out is not None:
            try:
                write_flat(out, result.flat)
            except FlatFileError as exc:
                raise CliError(str(exc)) from None
    except fs.SkyFlatError as exc:
        raise CliError(str(exc)) from None
    report = fs.format_sky_report(
        result,
        name=None if out is None else out.name,
        base_name=None if base_path is None else base_path.name,
    )
    for line in report:
        print(line)
    return 0
