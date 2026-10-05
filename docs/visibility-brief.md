# Visibility lane brief

Status: in progress. The owner approved [`visibility.md`](visibility.md) on October 5, 2026, and said to start the same day. The owner also approved five fixes to the design, listed under "Fixes approved at the start".

You are the visibility lane. You make the system measure seeing whenever Polaris is visible: pointing without an age limit, the search and measure modes of the fast stream, the adaptive exposures, the darkness and clear-sky events, and the visibility summary that `docs/visibility.md` describes. That file is the design for this lane. Where it and `docs/architecture.md` disagree, the visibility design wins for this feature, and you update the architecture to match.

## Read first

`CLAUDE.md`, `docs/visibility.md`, `docs/architecture.md` (the "Scheduler", "Pointing", "Reported quantities", and "Transparency and clouds" sections), `docs/development.md`, and `docs/phase2-status.md`.

## Rules

- Work in your own lane clone, set up as `docs/development.md` describes. Commit small, push every commit to `main`, and add no co-authors.
- The repository is public. Before each commit, run `tools/check_repo.py --staged` and `tools/scan_secrets.py --staged`, and review `git diff --staged`. No site coordinates, host names, or paths from your machine enter a tracked file, a test, or a commit message. Tests use the simulator's synthetic site.
- Do not edit `uv.lock`. If you need a new dependency, stop and tell the lead.
- Keep every new value in `config/default.d/scheduler.toml` or `config/default.d/survey.toml`, with a comment, and mark it provisional. Do not tune on the sky.
- Never leave `main` red.

## Steps

Each step ends with tests that pass and a push.

1. **Pointing without an age limit.** Make `validity_s = 0` mean no limit and the default, in the tracker and in the seeding of `core` from the store. Make sure that a failed solve never clears the solution, and that a missing star no longer starts a solve. Push this step on its own, because it helps every clear night after a cloudy one.
   Done when: tests cover a solution that is 30 days old (tracker and seeding), a positive limit that still expires, a failed solve after a good one, and 450 missing frames without a solve.
2. **The detection estimate.** Compute the SNR of Polaris in a fast frame (bin1, gain 0, at most 2 ms) against the Sun's elevation from +60° to −18°. Extend the simulator's sky model above +10° with a published daylight sky brightness near the pole, and cite it. Add a section to `docs/research-notes.md` with the table and the elevation where the SNR crosses 10. Then replace the placeholder default of `search.max_sun_elevation_deg` (+3°) with that elevation plus a margin of a few degrees, or with no limit if Polaris stays detectable in full daylight, and say why in the commit message.
   Done when: the section exists, a test reproduces its numbers, and the default follows them.
3. **Search and measure.** Add `[scheduler.search]`, the search bursts, the search limit from step 2 with its check bursts above it, the switch between the modes, and the events `polaris.visible` and `polaris.hidden`. Remove the Sun's elevation as a gate, and keep the measured saturation gate and the brightness watch.
   Done when: scheduler tests with the fake driver cover a day with and without a visible Polaris, no search above the limit except the check bursts, a check burst that finds Polaris, a cloud gap at night, a saturated sky, no solution, an unsynchronized clock, and a missing site. A simulated day and night with `seeingmon dev --driver sim` measures seeing from the elevation that step 2 predicts, within 1°.
4. **The adaptive fast exposure.** Add the exposure loop between windows, and the background level and the SNR in the window record. Regenerate `docs/quantities.md` and `docs/openapi.json` as `docs/development.md` describes.
   Done when: the window exposure follows a simulated twilight without saturating, and the record and generated-file tests pass.
5. **Bias in a bright sky.** Measure the bias of the seeing against the simulator's truth across background levels and exposures. Add the flags `noisy` and `daylight`.
   Done when: a table of the bias is in `docs/research-notes.md`, and tests show the flags on the right windows.
6. **The survey in a bright sky.** Add `[survey.twilight]`, the adaptive long exposure, the skip in daylight, and the `saturated_sky` guard on the cloud fraction, the limiting magnitude, the sky brightness, and the cloud tracker. Check that the dark model scales with the exposure, and fix it if it does not.
   Done when: a simulated evening solves its first survey frame at an elevation that you report, and a test shows that a saturated frame no longer reports clouds.
7. **Darkness and the clear verdict.** Add `n_expected` to `sky_quality`, the `[survey.darkness]` settings, and the events `sky.dark` and `sky.clear_verdict`.
   Done when: simulated nights with and without clouds give the right verdict, and a simulated summer night, where the Sun stays above −18°, still fires `sky.dark`.
8. **The visibility summary.** Declare `visibility_summary`, let `core` write it at the split hour, and add `seeingmon visibility stats`.
   Done when: the end-to-end night test (`tests/services/e2e/`) produces a summary with every field set or explained, and the command prints it, with censored nights counted separately.
9. **The cost.** Measure the CPU load, the memory, and the sensor temperature of a simulated day of measuring, and of a cloudy night of searching, with the performance harness, and add budget lines to `docs/performance.md`.
   Done when: the budget tests pass on the dev machine.
10. **The documentation.** Move the built design into `docs/architecture.md` (the "Scheduler", "Pointing", and "Transparency and clouds" sections, and the decisions table), leave `docs/visibility.md` as background with a pointer, and add a visibility check to the commissioning steps in `docs/runbook.md`.

## Fixes approved at the start

A read of the code before step 1 found five gaps in the design. The owner approved these fixes on October 5, 2026:

1. **A moved mount solves again.** Without an age limit, the solvers always get the hint of 2° around the prediction and never the hint of 15° around the pole, so a mount that moved more than about 2° never solves again. When the hinted solvers fail on a frame with enough stars, the pipeline tries again with the pole hint (step 1).
2. **The daylight gate measures the fast stream.** The gate reads a 1 ms bin2 frame, which saturates long before the fast stream does at its shortest exposure. The gate decides from the background that the fast stream would have at its shortest exposure (step 3).
3. **The brightness frame stays when the survey skips.** In `auto`, only the 1 ms frame of the survey step updates the background. When the survey skips its long exposure in daylight, it still takes the 1 ms frame, so the gate keeps working (step 6).
4. **`polaris.hidden` at every end of measure.** The scheduler writes `polaris.hidden` whenever measure ends, also when it ends because the state changes, such as a move to `safe` (step 3).
5. **The open details.**
   - The SNR of a burst is the median of the SNR of the star in its frames. Measure needs a centroid in every frame, so the SNR of the summed frames would switch to measure where the centroids are noise. (The proposal that the owner approved said the sum, and the lead corrected it the same day.)
   - A check burst above the limit needs `search.confirm_bursts` detections in a row, as a search does.
   - The warning that the limit is too low is the event `polaris.search_limit_low`.
   - TOML has no null, so the default file cannot write `None` for `search.max_sun_elevation_deg`. A value of 90 or more means no limit.
6. **`sky.dark` from the change per hour.** The design's rule, less than 0.05 mag per degree of Sun elevation, never fires: the twilight sky changes by about 1 mag per degree at −12° and still by about 0.2 near −18°, and near the Sun's lowest point on a summer night a change per degree is noise. The sky counts as dark when a line fitted to the last `survey.darkness.frames` (5) solved frames changes by less than `survey.darkness.max_slope_mag_per_hour` (0.3 mag per hour). The owner approved this on October 5, 2026 (step 7).
7. **The detection decides by a matched filter.** The first detection estimate (step 2) and the search (step 3) took the SNR of the fast path's centroid aperture, 201 px² around a star image of about 1.3 px FWHM, which adds the noise of about 200 pixels of empty sky. The owner corrected this on October 5, 2026: the SNR that decides whether Polaris is detectable is a property of the star, the sky, and the image size. The search bursts and the missing-star test of measure take the SNR of a matched filter (`seeingmon.fastpath.matched`), and the estimate decides by it. A search frame tries Gaussian filters of 1, 2, and 4 Airy FWHM and keeps the best, because the size of the real image is not known. In the model's daylight sky, the median frame gives 41 against the aperture's 8.3, so the estimate has no crossing of 10, and `search.max_sun_elevation_deg` is 90 (no limit) instead of 12. With the wider image of the owner's recordings (13 times the model's area), the same sky gives about 13, and the SNR falls to 10 in a sky of about 3.9 mag/arcsec², within the measured daylight skies, so the focus decides how much of the daylight shows Polaris. The window's `star_snr` stays the SNR of the centroid aperture, which step 5 needs. The centroids still come from the wide aperture, so daylight windows are biased until step 5: a simulated daylight window reads an `r0` of 3.4 cm against the injected 10 cm.

## Departures awaiting a decision

The lane built these differently from the design. Each waits for the owner's decision, and [`visibility.md`](visibility.md) keeps the approved text until then.

1. **The exposure adapts at the start of each fast period (step 4).** The design picks the exposure before each window. A fast period holds whole analysis windows on one stream (two windows of 60 s by default), and a new exposure needs a new stream, so the scheduler picks the exposure before each fast period and each search burst, from the last window or burst. The second window of a period keeps the exposure of the first, so the background lags the sky by up to a cycle. In the simulator's dawn, where the sky brightens by up to one magnitude a degree, a cycle of 120 s keeps the background below 0.40 of saturation (0.391 at most, `tests/scheduler/test_exposure.py`), and the default cycle of 180 s would keep it below about 0.45. Adapting before each window would end the stream in the middle of a period, which the lead decided against.
2. **The clear verdict counts every frame with a cloud fraction (step 7).** The design counts `verdict_frames` solved frames after `sky.dark`. Since step 1, a frame that does not solve still gets a cloud fraction from the stored pointing, and thick clouds are what keeps a frame from solving. A verdict over solved frames only waits for the gaps between the clouds and calls a cloudy night clear. The code counts the long frames with a cloud fraction after `sky.dark`, solved or not, so an overcast after dark gives a clear share of 0 at once. Going back to solved frames only is a one-line change in `DarknessWatch.update` (`src/seeingmon/services/core/darkness.py`).
3. **The aperture also keeps a star in measure (step 3b).** Fix 7 lets the matched filter decide whether the star is missing in measure. A filter of the Airy FWHM sees little of a defocused or wide image, though, and would lose a bright star that the centroid aperture finds, for example in the rapid focus mode, which uses the same kernel. The kernel therefore counts the star as found when either the first matched filter or the centroid aperture reaches `[fastpath] min_star_snr` (6), so measure never loses a frame that the aperture kept before. Going back to the matched filter alone is a one-line change in `_measure_at` (`src/seeingmon/fastpath/kernel.py`).

## Out of scope

- Web UI changes, beyond what the new fields and flags need to show up where records already show.
- Final values for any setting. That is phase 3.

## Stop and ask

Stop and tell the lead when the design looks wrong or contradictory (propose a fix and continue with the next step), when a step needs a dependency, or when step 2 or step 5 shows that readings in a bright sky are not usable at all.

## Reporting

Update the visibility row under "Requests added during phase 2" in `docs/phase2-status.md` when you start and after each step, with the commit hashes. At the end, report in plain language: from which Sun elevation Polaris is visible in the simulation, how large the bias of a bright-sky reading is, and what phase 3 must measure.
