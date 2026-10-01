# Seeing monitor: kickoff prompt

You are the lead engineer for a seeing monitor: a fixed-mount, monochrome camera that points at Polaris and measures atmospheric seeing and sky quality. The software runs on a Raspberry Pi (the Pi). This prompt starts phase 1, the architecture. In every phase, follow `CLAUDE.md`: the repository is public, so commit no secrets and no deployment-specific values, push every commit, and add no co-authors.

## Reference setup

- **Camera:** ZWO ASI294MM.
- **Optics:** ToupTek GS-250 PAPO guide scope: 250 mm focal length and 50 mm aperture (f/5). The vendor's "1-inch" figure follows the vidicon-tube sensor-format convention, not the diameter of the corrected image circle. I tested it: image quality is good to the edges of the ASI294MM sensor, and Polaris should always stay inside the corrected field. Verify the remaining specifications in vendor documentation.
- **Computer:** Raspberry Pi, headless, unattended every night. Confirm the model with me. Assume a Pi 4 or Pi 5 class device with USB 3 and a 64-bit OS.
- **Mount:** fixed, aimed at Polaris. Polaris moves on a small circle around the celestial pole, so the fast-mode region of interest (ROI) must follow it.

This setup is one configuration, not a design assumption. The software must also work with other sensor sizes, pixel sizes, bit depths, and optics.

- Describe sensor and optics in a profile: resolution, pixel size, bit depth, binning and readout modes, focal length, aperture, usable image circle, and limits for ROI, gain, and exposure. Derive everything else from the profile, for example plate scale, field of view, and ROI size in pixels. Hard-code nothing from the reference setup.
- Put camera access behind a driver interface. Include a simulated camera (synthetic stars with turbulence) and a replay driver, so the software runs without hardware or sky.
- Develop and test on Windows and Linux. Deploy on Raspberry Pi OS.

## Measurement modes

One camera serves all modes, so design a scheduler that shares its time.

1. **Seeing (fast).** Use short exposures (about 10 ms, configurable), a small ROI around Polaris, and the highest practical frame rate. Measure centroid, width, peak, and flux for each frame. Derive seeing and related statistics over time windows, for example image-motion variance, a seeing estimate in arcseconds, scintillation, and the motion spectrum.
2. **Sky quality (long exposures).** Measure sky background brightness in magnitudes per square arcsecond, transparency, and cloud presence, calibrated against a star catalog.
3. **Pointing (long exposures).** Plate-solve the field locally. Track pointing, rotation, and focus stability over days and seasons. Supply the Polaris position to the fast mode.
4. **Alignment helper.** While someone adjusts the mount, show a low-latency live view with the solved pointing against the target, offset and rotation, and aids for focus, histogram, and saturation.

## Interfaces and constraints

- **Web UI**, served from the Pi: current seeing, history, latest images, and the alignment helper. Keep it simple. Make it work on a phone, with a red night mode.
- **REST API:** latest and historical results, status, and health. Document and version it.
- **Database:** undecided. Define a pluggable sink with a local store and store-and-forward buffering, so you can add a remote database later without changing the core.
- **Timing:** timestamp every frame in UTC with a known accuracy. Report dropped frames; never drop them silently.
- **Resources:** plan for the Pi's CPU, USB, and storage limits. High frame rates produce more data than you can keep, so define what you store: per-frame metrics, raw bursts, or aggregates.
- **Operation:** the system runs unattended for months. It starts automatically, recovers from faults, reports its health, and handles daylight and clouds safely.
- **Commissioning:** support it from the start. Capture raw bursts on demand, run parameter sweeps, and replay recordings through the production analysis code.
- **Simplicity:** one maintainer runs the system. Prefer a few proven components over a large stack.

## Phases

### Architecture (phase 1, now)

Write `docs/architecture.md`, short enough for me to review in about 15 minutes. Cover:

- Components and data flow, with a Mermaid diagram.
- Technology choices with reasons: language, libraries, local storage, and web stack.
- Process and thread model, data rates, storage, and retention.
- Data model, API sketch, scheduler, and configuration.
- The method for each measurement mode and how you validate it.
- Test strategy, deployment, and security.
- Risks and open questions.

Define each reported quantity: definition, units, estimator, uncertainty, and known failure modes (for example, mount vibration, undersampling, and saturation). Look up the camera and optics specifications in vendor documentation. Compare camera access options (vendor SDK, INDI, ASCOM Alpaca) and plate solvers.

Ask me, in one batch before you write, about decisions that are mine to make, such as the database, the license, and the language. Record the other decisions with their reasons. Stop for my review, and write no implementation code.

### Implementation (phase 2)

After I approve the design, build in small, tested steps. Start with the simulator and the hardware-independent core. Commit and push each step.

When you reach the analysis code, ask me for the recorded data: a few gigabytes of real 10 ms exposure video. Use it to build the replay driver and to validate the algorithms against real data. Never write its location to the repository.

### Commissioning (phase 3)

After the system is installed, you get access to it. I provide the access details at that time; never write them to the repository. Under a real sky, find good exposure, gain, ROI, and cadence values, and validate the measurements. Record the findings in `docs/` and update the profile defaults. Keep site-specific values in local configuration.
