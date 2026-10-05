# Twilight: Polaris detection and the star count

Status: proposal. The owner asked for it on October 5, 2026, and has not approved it yet. When the owner approves it, the approved parts move into `docs/architecture.md`, and this file then holds only the background. Every number here is provisional, and commissioning (phase 3) sets the final values.

The lane brief is [`twilight-brief.md`](twilight-brief.md).

## Goals

1. Detect Polaris as soon as the camera can see it in the evening twilight, and note when it is last seen in the morning.
2. Collect statistics of the Sun's elevation at the first and the last detection, by night, so that you can see how they change with the season and the transparency.
3. Use the number of stars in the survey frames to tell when the sky is dark, and whether it is clear tonight.

## What the system does now

| Sun elevation | State | What the camera does | What the system records |
|---|---|---|---|
| Above −3° | `safe` | A 1 ms watch frame from a central 20′ region, every 60 s | The median, for the daylight gate only. No star detection. |
| Below −4°, and the watch median below 35 % of saturation | `auto` | A survey step: a 1 ms frame at gain 0, then a 30 s frame at gain 120 | `survey_frame`, `pointing`, and `sky_quality` records, flagged `twilight` until −18° |
| Below −4°, with no pointing solution | `auto` | Survey steps, and a solve attempt every 60 s (`solve_retry_s`) | Unsolved `pointing` records |
| Below −4°, with a valid solution | `auto` | 2 ms fast frames on a 4.1′ region around Polaris | Fast windows. After 450 frames without the star, a new solve, at most once a minute. |

The current design has four gaps:

- **The gate settings decide when the first measurement happens, not the sky.** Polaris (G ≈ 1.9) behind the telescope is bright enough to detect in short frames well before the Sun reaches −4°, maybe before sunset. Nothing looks for it before `auto` starts, so data from the current system would measure the −4° gate, the 60 s retry, and the 180 s survey cadence.
- **The long survey frame saturates in nautical twilight.** The daylight gate checks only the 1 ms frame, and the long frame keeps its 30 s exposure. The frame does not solve, so it has no cloud fraction, no limiting magnitude, and no sky brightness.
- **A saturated frame can look cloudy.** The cloud fraction takes its expected stars from the noise of the frame (`SurveyPipeline` in `src/seeingmon/survey/pipeline.py`). A clipped background has a low measured noise, so the pipeline expects many stars, misses them, and reports clouds.
- **Nothing summarizes a twilight.** No record or event says when Polaris appeared, when the sky became dark, or whether the evening is clear.

## Design

### The Polaris probe

The probe is a short frame on the predicted position of Polaris, taken in `safe`, that runs a star detection.

- **When it runs.** In `safe`, while the site is configured, the clock is synchronized, and the Sun is below `probe.max_sun_elevation_deg` (+2°). It runs between watch frames and does not replace them, so the daylight gate keeps its meaning. In the morning, `auto` stops at −3°, and the probe runs until the Sun rises above the same limit, so it also sees the last detection.
- **Where it looks.** At the position of Polaris that the latest pointing solution predicts, whatever the solution's age. The mount does not move, so an old solution still predicts Polaris within a few arcminutes, even though it is too old for the fast stream (`pointing.validity_s`, 12 hours). Without any solution, the probe uses the reference solution that commissioning saves (`survey.pointing.reference_file`). Without either, the probe stays off, and the scheduler writes one event that says why.
- **The frame.** The fast readout mode (bin1), gain 0, and a region of `probe.roi_arcmin` (10′) around the predicted position. The region is wider than the fast region because an old solution has a larger error.
- **The exposure.** A closed loop holds the background near `probe.target_background_fraction` (0.3) of saturation. Each exposure is the previous one times the ratio of the target to the measured background, at most 4 times larger or smaller per step, between the profile's shortest exposure and `probe.max_exposure_us` (2 ms, where Polaris starts to saturate in bin1).
- **The measurement.** The background median and its robust noise, then the brightest source within `probe.search_radius_arcmin` (1′) of the prediction: its position, peak, aperture flux, and signal-to-noise ratio (SNR). A frame counts as a detection at an SNR of `probe.detect_snr` (10) or more.
- **The cadence.** One probe frame every `probe.interval_s` (15 s). A probe frame takes a few milliseconds of exposure and a small region, so its cost is small. The lane measures it against the performance budgets.
- **First and last detection.** `probe.confirm_frames` (3) detections in a row confirm Polaris, and the time of the first of them is the first detection. The scheduler writes the event `polaris.first_seen` with the time and the Sun's elevation. The same number of misses in a row in the morning, after detections, gives `polaris.last_seen`, at the time of the last detection.
- **Censoring.** If the probe detects Polaris on its first frame of the evening, the first detection happened before the probe started, and the summary marks it `left_censored`. If the probe never loses Polaris in the morning before the Sun reaches the limit, the summary marks the last detection `right_censored`. Statistics must handle both, because a mean that ignores them is biased.

### An earlier start of `auto` (an option, off by default)

With `probe.start_auto_on_detection = true`, a confirmed detection lets `auto` start before the Sun reaches −4°. It stays off until the adaptive survey exposure below works and phase 3 shows that the fast stream measures usefully in bright twilight. The daylight gate keeps its measured override in every case.

### The survey exposure in twilight

The long survey exposure becomes adaptive while the `twilight` flag applies:

- The first long frame of an evening takes its exposure from the 1 ms frame. The pipeline converts the background of the 1 ms frame to a sky rate in electrons per second with the profile's gain and bias, and chooses the exposure that puts the background at `survey.twilight.target_background_fraction` (0.3) of saturation at the long gain.
- Each later long frame scales the previous exposure by the ratio of the target to the measured background, at most 4 times per step, between `survey.twilight.min_exposure_s` (1 s) and `long_exposure_s` (30 s).
- Below −18°, or when the exposure reaches `long_exposure_s`, the survey uses `long_exposure_s` again.

The dark model must scale with the exposure. The lane checks that the dark library's model does.

**A saturation guard.** A survey frame with more than `survey.twilight.max_saturated_fraction` (1 %) of its pixels saturated, or a background above 80 % of saturation, gets the flag `saturated_sky`. Such a frame gets no cloud fraction, limiting magnitude, or sky brightness, and the cloud tracker ignores it.

### The star count, darkness, and clear skies

These quantities already exist on a solved frame: `n_detected` in `survey_frame`, and `cloud_fraction`, `limiting_mag`, and `sky_mag_arcsec2` in `sky_quality`. The cloud fraction already takes its expected stars from the noise of the frame, so it separates a bright sky from a cloudy one. The design adds the following:

- **`n_expected` in `sky_quality`.** The number of catalog stars that the cloud fraction expects. With `n_detected`, it gives the star count against what a clear sky would show at that sky brightness.
- **Darkness.** The event `sky.dark` fires when the sky brightness changes by less than `darkness.max_slope_mag_per_deg` (0.05 mag per degree of Sun elevation) over `darkness.frames` (3) solved survey frames in a row. This finds the end of twilight from the measurement, which also works in summer, when the Sun never reaches −18°.
- **Clear tonight.** The cloud tracker already follows the cloud fraction with hysteresis. The event `sky.clear_verdict` fires once per evening, `darkness.verdict_frames` (5) solved frames after `sky.dark`, with the share of those frames that have a cloud fraction at or below `cloud.clear_threshold`.

### The twilight summary

`core` writes one `twilight_summary` record for each evening and each morning, next to the nightly star summary (`src/seeingmon/services/core/nightly.py`). An evening closes at `sky.dark`, at the night's split hour, or at shutdown, whichever comes first. A morning closes when the Sun rises above `probe.max_sun_elevation_deg`.

| Field | Meaning |
|---|---|
| `night` | The night, by `night_split_utc_hour` |
| `kind` | `evening` or `morning` |
| `probe_start_utc`, `probe_start_sun_deg` | The first probe frame of the twilight |
| `polaris_utc`, `polaris_sun_deg` | The first detection (evening) or the last detection (morning) |
| `polaris_censored` | `none`, `left`, or `right` |
| `first_solve_utc`, `first_solve_sun_deg` | The first solved survey frame |
| `dark_utc`, `dark_sun_deg`, `dark_sky_mag_arcsec2` | The `sky.dark` event and the sky brightness at it |
| `clear_share` | The share of clear frames in the verdict |
| `transparency_median` | The median transparency of the verdict frames |
| `flags` | `moon`, `time_invalid`, and `no_pointing` |

Every probe frame also writes a `polaris_probe` record (time, Sun elevation, exposure, gain, background fraction, background noise, SNR, detection, position, and its offset from the prediction). Its retention is 30 days, because the summaries keep what the statistics need.

### Statistics

`seeingmon twilight stats` prints the Sun elevation at the first and the last detection, by month and by transparency bin, from the `twilight_summary` records. It counts censored nights separately instead of dropping them. A plot on the History page can follow later.

## Settings

All keys are new, and all values are provisional.

| Key | Default | Meaning |
|---|---|---|
| `scheduler.probe.enabled` | `true` | Run the probe in `safe` |
| `scheduler.probe.max_sun_elevation_deg` | 2.0 | The probe runs while the Sun is below this |
| `scheduler.probe.interval_s` | 15.0 | The time between probe frames |
| `scheduler.probe.roi_arcmin` | 10.0 | The size of the probe region |
| `scheduler.probe.search_radius_arcmin` | 1.0 | How far from the prediction a source counts as Polaris |
| `scheduler.probe.target_background_fraction` | 0.3 | The background that the exposure loop aims for |
| `scheduler.probe.max_exposure_us` | 2000 | The longest probe exposure |
| `scheduler.probe.detect_snr` | 10.0 | The SNR of a detection |
| `scheduler.probe.confirm_frames` | 3 | Detections or misses in a row that change the state |
| `scheduler.probe.start_auto_on_detection` | `false` | Start `auto` at a confirmed detection |
| `survey.twilight.target_background_fraction` | 0.3 | The background that the long exposure aims for |
| `survey.twilight.min_exposure_s` | 1.0 | The shortest adaptive long exposure |
| `survey.twilight.max_saturated_fraction` | 0.01 | The share of saturated pixels that sets `saturated_sky` |
| `survey.darkness.max_slope_mag_per_deg` | 0.05 | The slope of the sky brightness that counts as dark |
| `survey.darkness.frames` | 3 | Solved frames in a row for `sky.dark` |
| `survey.darkness.verdict_frames` | 5 | Solved frames after `sky.dark` for the clear verdict |

## Open questions

- **How early is Polaris detectable?** The lane computes the predicted probe SNR against the Sun's elevation from the simulator's twilight model (`src/seeingmon/drivers/sim/sky.py`) and the profile, and adds the result to `docs/research-notes.md`. If Polaris is detectable above +2°, the default of `probe.max_sun_elevation_deg` should rise. The simulator's twilight curve is a typical one, so the real sky decides in phase 3.
- **The twilight gradient.** The sky near the pole is brighter toward the Sun's side. The survey background mesh (`mesh_px` 64) should follow it, and the lane checks that on a simulated frame with a gradient.
- **The morning.** Dawn is the same physics in reverse, but `auto` stops at −3° and leaves a gap for the probe to fill. The lane confirms that the hand-over from `auto` to `safe` does not lose the probe's state.

## For phase 3

The final probe interval, exposure limit, and detection SNR; whether `start_auto_on_detection` turns on; and the darkness slope, all from real twilights.
