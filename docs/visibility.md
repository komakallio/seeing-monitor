# Seeing whenever Polaris is visible

Status: built, except the cost measurement (step 9 of [visibility-brief.md](visibility-brief.md)), which is in progress. You approved this design on October 5, 2026, and the visibility lane built it on October 5 and 6. This file keeps the goals and the reasoning, and [architecture.md](architecture.md) holds the built design:

- [Scheduler](architecture.md#scheduler): the search and measure modes of the fast stream, the search limit and its probes, the events `polaris.visible`, `polaris.hidden`, and `polaris.search_limit_low`, the measured daylight gate, and the adaptive exposures of the fast stream and of the survey.
- [Seeing (fast)](architecture.md#seeing-fast): the missing-star test of measure, the noise model with the sky, the `noisy` flag, and the optional weighted centroid.
- [Sky quality](architecture.md#sky-quality), under "Transparency and clouds" and "The visibility of Polaris": `n_expected`, `saturated_sky`, `sky.dark`, `sky.clear_verdict`, the nightly `visibility_summary`, and `seeingmon visibility stats`.
- [Pointing](architecture.md#pointing) and [Plate solvers](architecture.md#plate-solvers): a solution without an age limit, and the retry around the pole.
- [Reported quantities](architecture.md#reported-quantities): the seeing in a bright sky, with the flags `noisy` and `daylight`, and the Sun's elevations of the visibility summary.

Ten departures from the approved text wait for your decision. [Approved text that waits for a decision](#approved-text-that-waits-for-a-decision) keeps the approved rule of each. Every number is provisional, and commissioning (phase 3) sets the final values.

## Goals

1. **Seeing whenever Polaris is visible.** If the camera can detect Polaris, the system measures seeing, at any Sun elevation: in daylight, in twilight, on bright summer nights, and in gaps between clouds. The Sun's elevation sets flags on the readings and never stops them.
2. **Pointing that does not expire.** The mount is rigid, so the last pointing solution always says where Polaris is. Only a solve that succeeds replaces it.
3. **Visibility statistics.** The Sun's elevation at the first and the last detection of each night, by season and transparency.
4. **Darkness and clear skies from the star count.** The survey frames tell when the sky is dark and whether the night is clear.

## Why the system missed readings

Before this design, the Sun's elevation and the age of the pointing solution gated the fast stream:

| Condition | What happened |
|---|---|
| The Sun above −3° | `safe`: one 1 ms brightness frame per minute. No seeing, even when Polaris was visible. |
| The Sun below −4° | `auto` started. The fast stream that measures seeing started only with a pointing solution younger than 12 hours (`[survey.pointing] validity_s`). |
| No solution younger than 12 hours | The scheduler waited for a survey frame to solve. In twilight the 30 s survey frame saturated, so the solve failed, and it retried every minute. |
| Polaris hidden for 450 frames | The scheduler stopped the fast stream and ran a survey step to solve again. |

The result:

- **No seeing in daylight or bright twilight**, because of the −3° and −4° limits.
- **No seeing on summer nights in the north.** The lowest Sun elevation at midsummer is your latitude + 23.4° − 90°. North of about 62.6° the Sun never gets below −4° then, and at 60° it gets only to −6.6°.
- **No seeing early on the first clear night after a cloudy one.** That night started with an expired solution and had to solve again, and in twilight the solve failed.
- **Wasted solves during clouds.** Each time clouds hid Polaris, the scheduler ran a survey step, although the mount had not moved.

The 12-hour limit had no physical reason. A `PointingSolution` stores the camera's attitude in an Earth-fixed frame (`src/seeingmon/survey/pointing.py`), and the tracker turns it with the Earth's rotation and applies precession, nutation, and aberration. On a mount that does not move, a solution from last month predicts Polaris as well as one from a minute ago.

## The reasoning behind the design

### Pointing without an age limit

Only a solve that succeeds replaces a solution, and a missing star never starts a solve. Clouds and bright skies hide Polaris often, and a hidden star says nothing about the mount. The survey steps keep solving at their normal cadence whenever the sky allows, which catches a real move or a drift of a few pixels. Without an age limit, though, the solvers always get the hint of 2° around the prediction, so a mount that moved further would never solve again. A retry with the hint of 15° around the pole closes that gap (fix 1 of the brief).

### The measured sky as the gate

The Sun cannot enter the field, because the celestial pole is always at least 66.5° from the Sun. So the Sun's elevation sets flags, and only a sky that the fast stream cannot take, even at its shortest exposure, holds the camera. The brightness frame (1 ms, bin2) saturates long before the fast stream does at its shortest exposure (32 µs, bin1), so the gate judges the background that the fast stream would see (fix 2).

Searching costs little. A burst of 50 fast frames takes about 0.6 s, so a burst every 15 s keeps the camera busy for about 4% of the time. The median of the SNR of the frames decides, and not the SNR of their sum, because measure needs a centroid in every frame (fix 5).

### What decides whether Polaris is detectable

The SNR that decides is a property of the star, the sky, and the size of the star's image, and a filter matched to the image reaches it (fix 7). The detection estimate (`docs/research-notes.md`, "Polaris in a bright sky") takes a daylight sky near the pole of 4.2 mag/arcsec² in V. The median frame of a burst then holds 7,980 e⁻ of Polaris in 1.23 ms on a sky of 4,310 e⁻² per pixel. The simulator's image is as sharp as the 50 mm aperture allows and covers about 6 px² (21.5 arcsec²), so the matched SNR is 41. It never falls to 10 between −18° and +90°, and it would take a sky 1.7 mag brighter than the model's daylight, and 0.7 mag brighter than the brightest daylight sky measured near the pole, to bring it there. Across the measured daylight skies, it ranges from 18 to 61. The real image is wider: in your recordings it covers about 280 arcsec², 13 times as much, and the same daylight sky then gives about 13, which falls to 10 in a sky of about 3.9 mag/arcsec², within the measured daylight skies. So the search has no Sun limit by default, and the focus and the sky of the day decide how much of the daylight shows Polaris.

The size of the image depends on the focus and the optics, so each frame tries Gaussian filters of 1, 2, and 4 Airy FWHM and keeps the best, which stays within 5% of the best weighting of the pixels for an image up to 13″ wide. On frames without a star, the noise reaches 10 with a chance below 6 × 10⁻¹⁹ per frame, so a false detection does not happen.

The fast path's centroid aperture (201 px²) adds the noise of about 200 pixels of empty sky. In daylight it gives 8.3, and it falls to 10 at +8.9°, which set the first default of +12°. That limit came from the method, not from the sky. With a limit below 90°, one probe burst every 10 minutes above it checks that the limit is not too low, so that the statistics are not cut off by the system's own setting.

### Exposures in a bright sky

A 2 ms fast frame at gain 0 saturates its background in a bright sky long before the sky hides Polaris, and a 30 s survey frame saturates in twilight long before its stars fade. Both exposures therefore follow the sky. The fast exposure changes only between windows, so that each window has one exposure, for which the seeing estimator corrects. The pointing does not need the long survey frame, so a step can skip it in daylight.

A clipped background looks quiet. A pipeline that trusted it would expect many stars, miss them, and report clouds, so a survey frame with a saturated sky gives no photometry.

### Can a reading in a bright sky be trusted?

Two effects need checks:

- **Centroid noise.** A bright background adds photon noise to each centroid, and the estimator subtracts the modeled noise from the variance. That works only while the model is right. The centroid aperture sums the noise of about 200 pixels of sky: in the simulator's daylight, a window read an `r0` of 3.3 cm against the injected 10 cm while the noise model left out the sky. With the sky in the model, it reads about 10% low, and a centroid weighted by the star's image reads the truth (`docs/research-notes.md`, "The seeing in a bright sky").
- **A sunlit telescope.** A tube that the Sun has heated adds its own turbulence. That turbulence is real but local.

Neither effect stops a reading. The flags `noisy` and `daylight` let a user decide.

### Darkness and clear skies from the star count

A solved survey frame already has `n_detected`, and `cloud_fraction`, `limiting_mag`, and `sky_mag_arcsec2`. The cloud fraction takes its expected stars from the noise of the frame, so it already tells a bright sky from a cloudy one. `n_detected` counts every detection, hot pixels and faint stars included, so it does not compare with the expected stars. `sky_quality` therefore keeps the counts behind the cloud fraction, `n_expected` and `n_expected_found`.

A rule for `sky.dark` per degree of Sun elevation never fires. The twilight sky changes by about 1 mag per degree at −12° and still by about 0.2 near −18°, and near the Sun's lowest point on a summer night a change per degree is noise. A rule per hour fires near the darkest time of every night, also in summer, when the Sun never reaches −18° (fix 6).

### The visibility statistics

A night has a first and a last detection, and either can be a bound rather than a moment: Polaris was already visible when the night started, or the station did not see it appear. Censored values stay in the statistics as censored, because dropping them biases the result toward the nights that the station watched from end to end.

## Approved text that waits for a decision

The lane built these ten points differently from the approved text, for the reasons that [visibility-brief.md](visibility-brief.md) gives under "Departures awaiting a decision". Each keeps its approved rule here until you decide.

| Departure | The approved text | What the code does |
|---|---|---|
| 1. The fast exposure | "Before each window, the scheduler picks the exposure that puts the background at `fast.target_background_fraction` (0.3) of saturation, from the previous window, between the profile's shortest exposure and `fast.exposure_us` (2 ms)." | The scheduler picks the exposure before each fast period and each search burst, and both windows of a period share it. |
| 2. The clear verdict | "An event once per evening, `survey.darkness.verdict_frames` (5) solved frames after `sky.dark`, with the share of frames at or below the cloud tracker's `clear_threshold`." | The verdict counts the long frames with a cloud fraction after `sky.dark`, solved or not. |
| 3. The missing star in measure | "In measure, the first filter around the centroid decides whether the star is missing." | The star counts as found when the first matched filter or the centroid aperture reaches `[fastpath] min_star_snr` (6). |
| 4. The long survey exposure | "While `twilight` applies, the long exposure takes its value from the 1 ms frame first, and from the previous long frame after that." "When even the shortest long exposure would pass the target, the scheduler skips the survey step." | The long exposure follows the measured background at any Sun. The first long frame after a start takes 1 s and grows by 4 times a step, and two rules skip the long exposure, while the step keeps its 1 ms frame. |
| 5. The saturation guard | "A survey frame with more than `survey.twilight.max_saturated_fraction` (1 %) of its pixels saturated, or a background above 80 % of saturation, gets the flag `saturated_sky` and no cloud fraction, limiting magnitude, or sky brightness." | Such a frame also gets no zero point, transparency, `n_expected`, or `n_expected_found`, and the 80% is the setting `[survey.twilight] max_background_fraction`. |
| 6. Censored detections | `first_censored` and `last_censored` say "whether Polaris was already visible when the night started, or still visible when it ended". | A detection is also censored when the station did not watch the sky for longer than `[survey.visibility] max_gap_s` (300 s) before the first or after the last detection. |
| 7. The values and the names of the summary | The fields `first_visible_utc`, `last_visible_utc`, and `dark_utc`, and `visible_hours` and `seeing_hours` without a unit. The last detection is "the last `polaris.hidden` of the morning". The `moon` flag has no rule. | The names carry their units (`first_visible_utc_ns`, `last_visible_utc_ns`, `dark_utc_ns`, and hours in `h`). A detection censored at an edge of the night takes the time of the edge and the Sun's elevation there. `moon` applies when the Moon was up and lit at a detection or at `sky.dark`, and the statistics bin the transparency at 0.6, 0.8, and 0.9. |
| 8. The key of the noise limit | `scheduler.fast.max_noise_bias` (0.05). | The key is `[fastpath] max_noise_bias`, because the fast analyzer sets the flag. |
| 9. The `noisy` flag | "A window gets the flag `noisy` where the bias exceeds `fast.max_noise_bias` (5 %)", from the bias measured against the simulator's truth across background levels. | The analyzer predicts the bias from the window's share of noise in the variance and the measured error of the noise model. |
| 10. A second centroid | The design has none. | `[fastpath] centroid = "gaussian"` takes the position from a centroid weighted by a Gaussian of 3 Airy FWHM, and the default stays `"aperture"`. |

## Settings

The settings live in the default files, each with a comment, and every value is provisional. The lane added or changed these:

- `config/default.d/scheduler.toml`: `[scheduler.search]`, `[scheduler.fast]`, `[scheduler.daylight]`, and `[scheduler.watch] bright_exposure_us`.
- `config/default.d/survey.toml`: `[survey.pointing]`, `[survey.solve]`, `[survey.twilight]`, `[survey.darkness]`, `[survey.visibility]`, `[survey.cloud] min_completeness`, and `[survey.sky] min_exposure_s`.
- `config/default.d/fastpath.toml`: `matched_fwhm_airy_widths`, `min_star_snr`, `max_noise_bias`, `centroid`, and `centroid_fwhm_airy_widths`.
- `config/default.d/web.toml`: `[web] withhold_fields`, which now also holds the Sun's elevations of the visibility summary.

The Sun no longer gates the camera, so `[scheduler.daylight] sun_elevation_limit_deg` and `sun_resume_margin_deg` are gone. `twilight_elevation_deg` stays for the flag, and `daylight_elevation_deg` (0°) sets `daylight`.

## Open questions

- **How bright a sky still shows Polaris?** The detection estimate computed the SNR of Polaris in a fast frame against the Sun's elevation, from +60° to −18°, and extended the simulator's sky above +10° with a measured daylight sky near the pole (`docs/research-notes.md`, "Polaris in a bright sky"). In the model, the matched SNR of the median frame stays at 41 or more at every Sun elevation, so Polaris is detectable in full daylight. The measured daylight sky scatters by 1.5 mag, and even its brightest value gives 18. The model leaves out the color of the sky, haze, and how the sky near the pole darkens as the Sun sinks toward +10°, so the real sky decides in phase 3.
- **How wide is the real image?** The model's image is as sharp as the aperture allows. Your recordings show an image of about 280 arcsec², 13 times the model's, which lowers the daylight SNR to about 13, and the bright third of the measured daylight skies would hide Polaris. Phase 3 measures the image of the fast stream in focus, and whether a sharper focus or a color filter narrows it.
- **The cost of measuring in daylight.** Searching costs little. In the model Polaris is visible in daylight, though, so the system measures all day: the camera and the CPU work continuously, and the sensor warms in the sun. Step 9 of the lane measures the CPU load, the memory, and the sensor temperature of a simulated day of measuring against the performance budgets. A warmer sensor also needs dark sets at its temperatures.
- **Which centroid in a bright sky?** The aperture reads `r0` about 10% low in the simulator's daylight, and the weighted centroid reads the truth there, but on your 8-bit bin2 recordings of a dark sky it reads `r0` 10% below the aperture, because its gain does not hold for the wider real image in bin2 pixels. Its gain needs a calibration on real frames before it can become the default.
- **Where the seeing is measured.** The readings describe the line of sight to Polaris, corrected to the zenith. A planet low in the south looks through more air. The architecture already reports the zenith value, and the History page should say so.

## For phase 3

The search interval and SNR, the exposure targets, the noise-bias limit, and the darkness slope, all from real skies. The runbook lists what to look at on the first sunny day and the first clear night, and what to measure: the in-focus image of the fast stream, the daylight sky near the pole, the noise model on real frames, the gain of the weighted centroid, the sensor temperature in sunlight, and the cost of the search on the Pi 4 ([Check the visibility of Polaris](runbook.md#check-the-visibility-of-polaris)).
