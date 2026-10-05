# Seeing whenever Polaris is visible

Status: approved by the owner on October 5, 2026, and in implementation since the same day. When the lane builds it, the built parts move into `docs/architecture.md`, and this file then holds only the background. Every number here is provisional, and commissioning (phase 3) sets the final values.

The lane brief is [`visibility-brief.md`](visibility-brief.md).

## Goals

1. **Seeing whenever Polaris is visible.** If the camera can detect Polaris, the system measures seeing, at any Sun elevation: in daylight, in twilight, on bright summer nights, and in gaps between clouds. The Sun's elevation sets flags on the readings and never stops them.
2. **Pointing that does not expire.** The mount is rigid, so the last pointing solution always says where Polaris is. Only a solve that succeeds replaces it.
3. **Visibility statistics.** The Sun's elevation at the first and the last detection of each night, by season and transparency.
4. **Darkness and clear skies from the star count.** The survey frames tell when the sky is dark and whether the night is clear.

## What the system does now, and why it misses readings

| Condition | What happens now |
|---|---|
| The Sun above −3° | `safe`: one 1 ms brightness frame per minute. No seeing, even when Polaris is visible. |
| The Sun below −4° | `auto` starts. The fast stream that measures seeing starts only with a pointing solution younger than 12 hours (`[survey.pointing] validity_s`). |
| No solution younger than 12 hours | The scheduler waits for a survey frame to solve. In twilight the 30 s survey frame saturates, so the solve fails, and it retries every minute. |
| Polaris hidden for 450 frames | The scheduler stops the fast stream and runs a survey step to solve again. |

The result:

- **No seeing in daylight or bright twilight**, because of the −3° and −4° limits.
- **No seeing on summer nights in the north.** The lowest Sun elevation at midsummer is your latitude + 23.4° − 90°. North of about 62.6° the Sun never gets below −4° then, and at 60° it gets only to −6.6°.
- **No seeing early on the first clear night after a cloudy one.** That night starts with an expired solution and has to solve again, and in twilight the solve fails.
- **Wasted solves during clouds.** Each time clouds hide Polaris, the scheduler runs a survey step, although the mount has not moved.

The 12-hour limit has no physical reason. A `PointingSolution` stores the camera's attitude in an Earth-fixed frame (`src/seeingmon/survey/pointing.py`), and the tracker turns it with the Earth's rotation and applies precession, nutation, and aberration. On a mount that does not move, a solution from last month predicts Polaris as well as one from a minute ago.

## Design

### Pointing without an age limit

- `[survey.pointing] validity_s` gets the default 0, which means no limit. A positive value still works, for a mount that is not rigid.
- When `core` starts, it seeds the tracker with the newest usable stored solution of any age. The checks of the matched stars and the residual stay.
- A failed solve never clears the solution. Only a solve that succeeds replaces it, and a solve that sets the `moved` flag (5′ or 0.5° of roll from the reference) also writes a warning event.
- A missing star never starts a solve by itself. Clouds and bright skies hide Polaris often, and a hidden star says nothing about the mount. The survey steps keep solving at their normal cadence whenever the sky allows, which catches a real move or a drift of a few pixels.
- Without any solution (a new installation), the system behaves as now: survey frames at night until one solves, or the alignment helper.

### Two modes of the fast stream: search and measure

The fast stream replaces the Sun's elevation as the gate. It runs in one of two modes whenever a solution exists and the sky is not saturated:

- **Search.** A burst of `search.burst_frames` (50) fast frames every `search.interval_s` (15 s), on the region where the solution predicts Polaris. A detection in `search.confirm_bursts` (2) bursts in a row switches to measure. A burst keeps the camera busy for about 3 % of the time, and the camera is idle between bursts.
- **Measure.** The fast stream as it runs now: continuous frames, seeing windows, and recentering at the ROI edge. When the star is missing for `fast.missing_star_frames` (450) frames in a row, the stream returns to search. It no longer starts a solve.

The scheduler writes the event `polaris.visible` when the stream switches to measure and `polaris.hidden` when it switches back, each with the Sun's elevation. These events give the visibility statistics.

**When to search.** Search runs only where Polaris can appear: while the Sun is below `search.max_sun_elevation_deg`, and at night under clouds. The default is 90°, which means no limit. The detection estimate (`docs/research-notes.md`, "Polaris in a bright sky") takes the SNR that a matched filter reaches, which depends only on the star, the sky, and the size of the star's image. In the estimate's daylight sky near the pole (4.2 mag/arcsec² in V), the median frame of a burst holds 7,980 e⁻ of Polaris in 1.23 ms on a sky of 4,310 e⁻² per pixel, and the image covers about 6 px², so the matched SNR is 41. It never falls to 10 between −18° and +90°, and it would take a sky 1.7 mag brighter than the model's daylight, and 0.7 mag brighter than the brightest daylight sky measured near the pole, to bring it there. Across the measured daylight skies, it ranges from 18 to 61. The fast path's centroid aperture (201 px²) adds the noise of about 200 pixels of empty sky: in daylight it gives 8.3, and it falls to 10 at +8.9°, which set the first default of +12°. That limit came from the method, not from the sky. With a value below 90, one burst every `search.probe_interval_s` (600 s) above the limit checks that the limit is not too low, so that the statistics are not cut off by the system's own setting. When a probe burst finds Polaris, the system measures, and it writes a warning event that the limit is too low.

**The detection.** The fast analyzer already reports whether it found the star. Search adds the signal-to-noise ratio (SNR) of the star in each frame of a burst: the SNR of a filter matched to the image of the star, at the brightest place of the filtered image within `search.radius_px` of the prediction, with the sky noise measured on the ROI border and the star's own photon noise. A burst counts as a detection when the median of its frames reaches `search.detect_snr` (10). In measure, the same filter around the centroid decides whether the star is missing. On frames without a star, the noise reaches 10 with a chance below 2 × 10⁻¹⁹ per frame, so a false detection does not happen. The window keeps the SNR of the centroid aperture as `star_snr`, because that SNR tells the noise of the centroids.

### The fast exposure in a bright sky

In a bright sky, a 2 ms frame at gain 0 can saturate its background. The fast stream adapts its exposure between windows, never within one:

- Before each window, the scheduler picks the exposure that puts the background at `fast.target_background_fraction` (0.3) of saturation, from the previous window, between the profile's shortest exposure and `fast.exposure_us` (2 ms).
- Each window record already carries its exposure, and the seeing estimator already corrects for the exposure.
- The window record gets the background level and the star's SNR, so the noise of a reading can be judged later.

**The daylight gate.** The Sun's elevation no longer gates the camera. The measured gate stays: when the background exceeds `saturation_limit` (50 %) of saturation even at the shortest exposure, nothing can be measured, and the scheduler waits in `safe` with its brightness watch. The Sun cannot enter the field, because the celestial pole is always at least 66.5° from the Sun.

### Can a reading in a bright sky be trusted?

Two effects need checks:

- **Centroid noise.** A bright background adds photon noise to each centroid, and the estimator subtracts the noise from the variance (architecture, "Reported quantities"). That works only while the noise estimate is right. The centroids come from the wide aperture, so a daylight window is noisy: in the simulator's daylight, a window reads an `r0` of 3.4 cm against the injected 10 cm. The lane measures the bias of the seeing against the simulator's truth across background levels, and a window gets the flag `noisy` where the bias exceeds `fast.max_noise_bias` (5 %).
- **A sunlit telescope.** A tube that the Sun has heated adds its own turbulence. That turbulence is real but local. The flags let you filter it: `daylight` while the Sun is above 0°, and `twilight` from 0° to −18° as now.

Neither effect stops a reading. The flags let a user decide.

### Survey frames in a bright sky

- **An adaptive long exposure.** While `twilight` applies, the long exposure takes its value from the 1 ms frame first, and from the previous long frame after that, so that the background sits at `survey.twilight.target_background_fraction` (0.3) of saturation. It stays between `survey.twilight.min_exposure_s` (1 s) and `long_exposure_s` (30 s), and changes by at most 4 times per step.
- **No survey in daylight.** When even the shortest long exposure would pass the target, the scheduler skips the survey step. The pointing does not need it.
- **A saturation guard.** A survey frame with more than `survey.twilight.max_saturated_fraction` (1 %) of its pixels saturated, or a background above 80 % of saturation, gets the flag `saturated_sky` and no cloud fraction, limiting magnitude, or sky brightness. Today a clipped background looks quiet, so the pipeline expects many stars, misses them, and reports clouds.
- The dark model must scale with the exposure. The lane checks that it does.

### Darkness and clear skies

These already exist on a solved survey frame: `n_detected` in `survey_frame`, and `cloud_fraction`, `limiting_mag`, and `sky_mag_arcsec2` in `sky_quality`. The cloud fraction takes its expected stars from the noise of the frame, so it already tells a bright sky from a cloudy one. The design adds:

- **`n_expected` and `n_expected_found` in `sky_quality`.** The number of catalog stars that the cloud fraction expects, and the number of them that detection found. The cloud fraction is 1 minus their ratio. `n_detected` counts every detection, hot pixels and faint stars included, so it does not compare with `n_expected`.
- **`sky.dark`.** An event when a line fitted to the sky brightness of the last `survey.darkness.frames` (5) solved frames changes by less than `survey.darkness.max_slope_mag_per_hour` (0.3 mag per hour). A rule per degree of Sun elevation would never fire, because near the Sun's lowest point the elevation barely changes. The rule per hour also works in summer, when the Sun never reaches −18°, and fires near the darkest time of the night.
- **`sky.clear_verdict`.** An event once per evening, `survey.darkness.verdict_frames` (5) solved frames after `sky.dark`, with the share of frames at or below the cloud tracker's `clear_threshold`.

### The visibility summary

`core` writes one `visibility_summary` record per night, next to the nightly star summary (`src/seeingmon/services/core/nightly.py`), when the night's split hour passes.

| Field | Meaning |
|---|---|
| `night` | The night, by `night_split_utc_hour` |
| `first_visible_utc`, `first_visible_sun_deg` | The first `polaris.visible` of the evening |
| `last_visible_utc`, `last_visible_sun_deg` | The last `polaris.hidden` of the morning |
| `visible_hours`, `seeing_hours` | The time in measure mode, and the time that produced seeing windows |
| `first_censored`, `last_censored` | Whether Polaris was already visible when the night started, or still visible when it ended |
| `dark_utc`, `dark_sun_deg`, `dark_sky_mag_arcsec2` | The `sky.dark` event |
| `clear_share`, `transparency_median` | The clear verdict |
| `flags` | `moon`, `time_invalid`, `no_pointing` |

Censored values stay in the statistics as censored, because dropping them biases the result. `seeingmon visibility stats` prints the Sun elevation at the first and the last detection by month and by transparency bin, with censored nights counted separately.

## Settings

All values are provisional.

| Key | Default | Meaning |
|---|---|---|
| `survey.pointing.validity_s` | 0 (changed) | No age limit. A positive value restores one. |
| `scheduler.search.burst_frames` | 50 | Frames in one search burst |
| `scheduler.search.interval_s` | 15.0 | The time between search bursts |
| `scheduler.search.detect_snr` | 10.0 | The median matched SNR of a detection |
| `scheduler.search.radius_px` | 20.0 | How far from the prediction a detection may lie, in fast-mode pixels |
| `scheduler.search.confirm_bursts` | 2 | Bursts with a detection in a row that start measure |
| `scheduler.search.max_sun_elevation_deg` | 90.0 | Search runs while the Sun is below this. 90 or more means always, the default: in the detection estimate, the matched SNR of Polaris stays above 10 in full daylight. |
| `scheduler.search.probe_interval_s` | 600.0 | The time between check bursts above the limit |
| `fastpath.matched_fwhm_airy_widths` | 1.0 (new) | The FWHM of the matched filter, in Airy FWHM of the readout mode |
| `fastpath.min_star_snr` | 6.0 (now matched) | The matched SNR below which measure counts the star as missing. It was the SNR of the centroid aperture. |
| `scheduler.fast.target_background_fraction` | 0.3 | The background that the fast exposure aims for |
| `scheduler.fast.max_noise_bias` | 0.05 | The seeing bias that sets `noisy` |
| `survey.twilight.target_background_fraction` | 0.3 | The background that the long exposure aims for |
| `survey.twilight.min_exposure_s` | 1.0 | The shortest adaptive long exposure |
| `survey.twilight.max_saturated_fraction` | 0.01 | The share of saturated pixels that sets `saturated_sky` |
| `survey.darkness.max_slope_mag_per_hour` | 0.3 | The change of the sky brightness per hour that counts as dark |
| `survey.darkness.frames` | 5 | Solved frames in the fit for `sky.dark` |
| `survey.darkness.max_gap_s` | 600.0 | The longest time between two frames of the fit. A longer gap starts the fit again. |
| `survey.darkness.verdict_frames` | 5 | Solved frames after `sky.dark` for the clear verdict |

`scheduler.daylight.sun_elevation_limit_deg` and `sun_resume_margin_deg` no longer gate the camera, so they go away. `twilight_elevation_deg` stays for the flag.

## Open questions

- **How bright a sky still shows Polaris?** The lane computed the SNR of Polaris in a fast frame against the Sun's elevation, from +60° to −18°, and extended the simulator's sky above +10° with a measured daylight sky near the pole (`docs/research-notes.md`, "Polaris in a bright sky"). In the model, the matched SNR of the median frame stays at 41 or more at every Sun elevation, so Polaris is detectable in full daylight. The measured daylight sky scatters by 1.5 mag, and even its brightest value gives 18. The model leaves out the color of the sky, haze, and how the sky near the pole darkens as the Sun sinks toward +10°, so the real sky decides in phase 3.
- **The cost of measuring in daylight.** Searching costs little. In the model Polaris is visible in daylight, though, so the system measures all day: the camera and the CPU work continuously, and the sensor warms in the sun. The lane measures the CPU load, the memory, and the sensor temperature of a day of measuring against the performance budgets. A warmer sensor also affects the dark library.
- **Where the seeing is measured.** The readings describe the line of sight to Polaris, corrected to the zenith. A planet low in the south looks through more air. The architecture already reports the zenith value, and the History page should say so.

## For phase 3

The search interval and SNR, the exposure targets, the noise-bias limit, and the darkness slope, all from real skies.
