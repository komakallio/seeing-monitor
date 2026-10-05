# Visibility lane brief

Status: waiting for the owner's approval of [`visibility.md`](visibility.md). Start when the owner approves it, and build only the approved parts.

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

## Out of scope

- Web UI changes, beyond what the new fields and flags need to show up where records already show.
- Final values for any setting. That is phase 3.

## Stop and ask

Stop and tell the lead when the design looks wrong or contradictory (propose a fix and continue with the next step), when a step needs a dependency, or when step 2 or step 5 shows that readings in a bright sky are not usable at all.

## Reporting

Update the visibility row under "Requests added during phase 2" in `docs/phase2-status.md` when you start and after each step, with the commit hashes. At the end, report in plain language: from which Sun elevation Polaris is visible in the simulation, how large the bias of a bright-sky reading is, and what phase 3 must measure.
