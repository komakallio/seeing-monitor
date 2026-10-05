# Twilight lane brief

Status: waiting for the owner's approval of [`twilight.md`](twilight.md). Start when the owner approves it, and build only the approved parts.

You are the twilight lane. You build the Polaris probe, the adaptive survey exposure in twilight, the darkness and clear-sky events, and the twilight summary that `docs/twilight.md` describes. That file is the design for this lane. Where it and `docs/architecture.md` disagree, the twilight design wins for this feature, and you update the architecture to match.

## Read first

`CLAUDE.md`, `docs/twilight.md`, `docs/architecture.md` (the "Scheduler", "Pointing", and "Transparency and clouds" sections), `docs/development.md`, and `docs/phase2-status.md`.

## Rules

- Work in your own lane clone, set up as `docs/development.md` describes. Commit small, push every commit to `main`, and add no co-authors.
- The repository is public. Before each commit, run `tools/check_repo.py --staged` and `tools/scan_secrets.py --staged`, and review `git diff --staged`. No site coordinates, host names, or paths from your machine enter a tracked file, a test, or a commit message. Tests use the simulator's synthetic site.
- Do not edit `uv.lock`. If you need a new dependency, stop and tell the lead.
- Keep every new value in `config/default.d/scheduler.toml` or `config/default.d/survey.toml`, with a comment, and mark it provisional. Do not tune on the sky.
- Never leave `main` red.

## Steps

Each step ends with tests that pass and a push.

1. **The detection estimate.** Compute the predicted probe SNR of Polaris against the Sun's elevation from +5° to −18°, with the simulator's twilight model and the profile's fast mode at gain 0, with an exposure of at most 2 ms. Add a short section to `docs/research-notes.md` with the table and the elevation where the SNR crosses 10. If that elevation is above +2°, raise the default of `probe.max_sun_elevation_deg` and say why in the commit message.
   Done when: the section exists and a test reproduces its numbers.
2. **The prediction from an old solution.** Add a way for the tracker to predict Polaris from its latest solution at any age, and from the reference solution when it has no solution. Keep `valid_at` and the fast stream's 12-hour rule unchanged.
   Done when: unit tests cover a fresh solution, a 20-hour-old solution, the reference only, and neither.
3. **The probe in the scheduler.** Add `[scheduler.probe]`, the probe step in `safe` (between watch frames, at its own interval), the exposure loop, the measurement, the confirmation, and the events `polaris.first_seen` and `polaris.last_seen`. The watch frame and the daylight gate do not change.
   Done when: scheduler tests with the fake driver cover the evening (first detection, with and without censoring), the morning (last detection across the hand-over from `auto` to `safe`), a missing site, an unsynchronized clock, and no solution. A simulated evening with `seeingmon dev --driver sim` writes `polaris.first_seen` at the elevation that step 1 predicts, within 1°.
4. **The `polaris_probe` record.** Declare it in `src/seeingmon/records/`, with a retention of 30 days, and write one per probe frame. Regenerate `docs/quantities.md` and `docs/openapi.json` as `docs/development.md` describes.
   Done when: the record tests and the generated-file tests pass.
5. **The adaptive survey exposure.** Add `[survey.twilight]`, the first long exposure from the 1 ms frame, the closed loop, the return to `long_exposure_s`, and the `saturated_sky` flag with its guard on the cloud fraction, the limiting magnitude, the sky brightness, and the cloud tracker. Check that the dark model scales with the exposure, and fix it if it does not.
   Done when: a simulated evening solves its first survey frame at a Sun elevation that you report, and no frame with `saturated_sky` gets a cloud fraction. A test shows that a saturated frame no longer reports clouds.
6. **Darkness and the clear verdict.** Add `n_expected` to `sky_quality`, the `[survey.darkness]` settings, and the events `sky.dark` and `sky.clear_verdict`.
   Done when: simulated nights with and without clouds give the right verdict, and a simulated summer night, where the Sun stays above −18°, still fires `sky.dark`.
7. **The twilight summary.** Declare `twilight_summary`, and let `core` write one per evening and per morning next to the nightly star summary. Add `seeingmon twilight stats`.
   Done when: the end-to-end night test (`tests/services/e2e/`) produces an evening and a morning summary with every field set or explained, and `seeingmon twilight stats` prints them, with censored nights counted separately.
8. **The cost.** Measure the probe's CPU and memory cost with the performance harness, and add a budget line to `docs/performance.md`.
   Done when: the budget test passes on the dev machine.
9. **The documentation.** Move the built design from `docs/twilight.md` into `docs/architecture.md` (the "Scheduler" and "Transparency and clouds" sections, and the decisions table), leave `docs/twilight.md` as background with a pointer, and add a twilight check to the commissioning steps in `docs/runbook.md`.

## Out of scope

- A plot on the History page or any other web UI change.
- Turning on `probe.start_auto_on_detection`. Build it, test it, and leave it off.
- Final values for any setting. That is phase 3.

## Stop and ask

Stop and tell the lead when the design looks wrong or contradictory (propose a fix and continue with the next step), when a step needs a dependency, or when step 1 shows that the probe cannot detect Polaris before the Sun reaches −4°, because then most of the probe has no purpose.

## Reporting

Update the twilight row under "Requests added during phase 2" in `docs/phase2-status.md` when you start and after each step, with the commit hashes. At the end, report in plain language: the predicted elevation of the first detection, what the simulated evening showed, and what phase 3 must measure.
