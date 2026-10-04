"use strict";

/*
 * The scenarios of the logic of the Flat page (static/js/flattext.js): the words, the status line of
 * the library, the hint about when a session starts, the phases of a session and the gauge of its
 * level, what a finished session leaves, the verdicts on a new flat, the rows of the library, the
 * check of the form, and how often to poll. The function takes `FlatText`, a `test(name, fn)`
 * function, and an `assert` object, so that Node (flattext.test.js) and a browser console can both
 * run it.
 */
module.exports = function scenarios(FlatText, test, assert) {
  const DASH = "—";
  const MINUS = "−";

  /** The task of an idle core, changed by `extra`. */
  function task(extra) {
    return Object.assign(
      {
        state: "idle",
        task_id: null,
        phase: null,
        step: 0,
        steps: 0,
        message: "",
        set_number: 1,
        frames: null,
        target_fraction: null,
        exposure_s: null,
        level_fraction: null,
        saturated_fraction: null,
        warnings: [],
        pause_after: true,
        started_utc: null,
        finished_utc: null,
        summary: "",
        version: null,
      },
      extra
    );
  }

  /** A task that was queued with 32 frames and a target of 50 %. */
  function session(extra) {
    return task(Object.assign({ task_id: 3, frames: 32, target_fraction: 0.5 }, extra));
  }

  /** A flat like the ones of the owner's lens: the corners lose 9.6 %, and a little dust. */
  function flat(extra) {
    return Object.assign(
      {
        version: "flat-5e6f7a8b",
        t_utc: "2026-10-04T20:15:00Z",
        age_days: 0.02,
        state: "pending",
        active: false,
        pending: true,
        mode: "bin2",
        gain: 120,
        width_px: 4144,
        height_px: 2822,
        sensor_temperature_c: 12.4,
        exposure_s: 0.0390625,
        target_fraction: 0.5,
        second_set: false,
        source_turned: false,
        frames_taken: 32,
        frames_used: 31,
        noise_percent: 0.11,
        bias_source: "dark library",
        bias_note: "from the dark library",
        corner_percent: -9.6,
        vignetting: [],
        tilt: { width_percent: -0.62, height_percent: 0.41 },
        optics_tilt: null,
        source_tilt: null,
        shadows: 3,
        shadow_min_depth_percent: 1.0,
        shadow_items: [],
        edge_artifacts: 2,
        agreement: null,
        sets: [],
        warnings: [],
        has_image: true,
        image_url: "/api/v1/flat/flat-5e6f7a8b/image",
        activated_utc: null,
      },
      extra
    );
  }

  function library(extra) {
    return Object.assign(
      {
        mode: "bin2",
        gain: 120,
        sensor_temperature_c: 12.3,
        active_version: null,
        pending_version: null,
        flat_file_pinned: false,
        library_overrides: false,
        blocker: null,
        flats: [],
        session: null,
        task: task(),
      },
      extra
    );
  }

  // --- Words ------------------------------------------------------------------------------------

  test("a clause of core becomes a sentence", () => {
    assert.equal(FlatText.sentence("the library holds no dark set"), "The library holds no dark set.");
    assert.equal(FlatText.sentence("Already a sentence."), "Already a sentence.");
    assert.equal(FlatText.sentence("What now?"), "What now?");
    assert.equal(FlatText.sentence("  "), "");
    assert.equal(FlatText.sentence(null), "");
  });

  test("a plain C after a number becomes degrees Celsius", () => {
    assert.equal(FlatText.degrees("interpolated to 12.3 C"), "interpolated to 12.3 °C");
    assert.equal(FlatText.degrees("a Cat"), "a Cat");
    assert.equal(FlatText.degrees(""), "");
  });

  test("an exposure reads in seconds, milliseconds, or microseconds", () => {
    assert.equal(FlatText.exposureText(0.0390625), "39.1 ms");
    assert.equal(FlatText.exposureText(0.02), "20 ms");
    assert.equal(FlatText.exposureText(1.5), "1.5 s");
    assert.equal(FlatText.exposureText(1), "1 s");
    assert.equal(FlatText.exposureText(120), "120 s");
    assert.equal(FlatText.exposureText(32e-6), "32 µs");
    assert.equal(FlatText.exposureText(null), DASH);
  });

  test("a level is a whole percent of the full scale", () => {
    assert.equal(FlatText.levelText(0.498), "50 %");
    assert.equal(FlatText.levelText(0.256), "26 %");
    assert.equal(FlatText.levelText(null), DASH);
  });

  test("a tilt reads across the width and across the height, with the sign that the page uses", () => {
    assert.equal(FlatText.tiltText({ width_percent: -0.62, height_percent: 0.41 }), MINUS + "0.62 % across the width, +0.41 % across the height");
    assert.equal(FlatText.tiltText({ width_percent: null, height_percent: null }), DASH);
    assert.equal(FlatText.tiltText(null), DASH);
  });

  test("the age of a flat reads in days", () => {
    assert.equal(FlatText.ageText(0.2), "today");
    assert.equal(FlatText.ageText(1.4), "yesterday");
    assert.equal(FlatText.ageText(12.7), "12 days ago");
    assert.equal(FlatText.ageText(null), DASH);
    assert.equal(FlatText.day("2026-10-04T20:15:00Z"), "2026-10-04");
    assert.equal(FlatText.day(null), DASH);
  });

  // --- The library --------------------------------------------------------------------------------

  test("a library without a flat in use says that the survey applies no correction", () => {
    const line = FlatText.libraryLine(library());
    assert.equal(line.word, "None");
    assert.equal(line.level, "warn");
    assert.ok(line.text.includes("does not correct for the lens"));
  });

  test("a flat in use is named with its age", () => {
    const old = flat({ state: "approved", active: true, pending: false, age_days: 12.4 });
    const line = FlatText.libraryLine(library({ active_version: old.version, flats: [old] }));
    assert.equal(line.word, "In use");
    assert.equal(line.level, "good");
    assert.equal(line.text, "The survey divides by the flat flat-5e6f7a8b, made 12 days ago.");
  });

  test("the flat of the library replaces the flat file of the settings, and the line says so", () => {
    const old = flat({ active: true, pending: false, state: "approved", age_days: 3 });
    const line = FlatText.libraryLine(library({ active_version: old.version, flats: [old], flat_file_pinned: true, library_overrides: true }));
    assert.ok(line.text.endsWith("It replaces the flat of the setting flat_file."));
  });

  test("a flat file in the settings stands in when the library has no flat in use", () => {
    const line = FlatText.libraryLine(library({ flat_file_pinned: true }));
    assert.equal(line.word, "From settings");
    assert.equal(line.level, "good");
  });

  test("the sensor temperature has a line, and a missing one says so", () => {
    assert.equal(FlatText.temperatureLine(library()), "Sensor temperature now: 12.3 °C.");
    assert.equal(FlatText.temperatureLine(library({ sensor_temperature_c: null })), "The sensor temperature is not reported.");
  });

  test("the library finds the pending flat and the active one by their versions", () => {
    const first = flat({ version: "flat-11111111" });
    const second = flat({ version: "flat-22222222", active: true, pending: false, state: "approved" });
    const lib = library({ flats: [first, second], pending_version: first.version, active_version: second.version });
    assert.equal(FlatText.pendingFlat(lib).version, "flat-11111111");
    assert.equal(FlatText.activeFlat(lib).version, "flat-22222222");
    assert.equal(FlatText.pendingFlat(library()), null);
  });

  // --- Polling and hints ------------------------------------------------------------------------

  test("the page polls fast while a session is queued or running, and slowly otherwise", () => {
    assert.equal(FlatText.pollInterval(session({ state: "queued" })), FlatText.limits.FAST_POLL_MS);
    assert.equal(FlatText.pollInterval(session({ state: "running" })), FlatText.limits.FAST_POLL_MS);
    for (const state of ["idle", "ok", "failed", "aborted"]) {
      assert.equal(FlatText.pollInterval(session({ state })), FlatText.limits.SLOW_POLL_MS);
    }
    assert.equal(FlatText.pollInterval(null), FlatText.limits.SLOW_POLL_MS);
  });

  test("the hint says when a session starts, by the state of the scheduler", () => {
    const idle = task();
    assert.equal(FlatText.startHint(undefined, idle), "");
    assert.ok(FlatText.startHint(null, idle).includes("cannot reach core"));
    assert.ok(FlatText.startHint("paused", idle).includes("resumes it"));
    assert.ok(FlatText.startHint("paused", idle).includes("pauses again"));
    assert.ok(FlatText.startHint("align", idle).includes("alignment helper"));
    assert.ok(FlatText.startHint("safe", idle).includes("starts at once"));
    assert.ok(FlatText.startHint("commission", idle).includes("Another task"));
    assert.ok(FlatText.startHint("auto", idle).includes("next step"));
    assert.equal(FlatText.startHint("safe", session({ state: "running" })), "");
  });

  // --- The session --------------------------------------------------------------------------------

  test("a session that waits has no active phase", () => {
    const list = FlatText.phases(session({ state: "queued", message: "The scheduler is paused." }));
    assert.deepEqual(list.map((item) => item.id), ["setup", "exposure", "capture", "build"]);
    assert.deepEqual(list.map((item) => item.state), ["pending", "pending", "pending", "pending"]);
    assert.equal(list[2].detail, "32 frames");
  });

  test("the search for the exposure shows the try, the exposure, and the level", () => {
    const list = FlatText.phases(session({ state: "running", phase: "exposure", step: 2, steps: 8, exposure_s: 0.0390625, level_fraction: 0.499 }));
    assert.deepEqual(list.map((item) => item.state), ["done", "active", "pending", "pending"]);
    assert.equal(list[0].detail, "The camera and the library are ready.");
    assert.equal(list[1].detail, "Try 2 (at most 8): 39.1 ms gives 50 % of full scale.");
  });

  test("the frames show the number of the frame", () => {
    const list = FlatText.phases(session({ state: "running", phase: "capture", step: 12, steps: 32, exposure_s: 0.0390625 }));
    assert.deepEqual(list.map((item) => item.state), ["done", "done", "active", "pending"]);
    assert.equal(list[1].detail, "Exposure 39.1 ms.");
    assert.equal(list[2].detail, "Frame 12 of 32.");
  });

  test("a session that ended ok has every phase done", () => {
    const list = FlatText.phases(session({ state: "ok", frames: 32, exposure_s: 0.0390625 }));
    assert.deepEqual(list.map((item) => item.state), ["done", "done", "done", "done"]);
    assert.equal(list[2].detail, "32 frames of 39.1 ms.");
  });

  test("the moment after the flat is stored shows every phase done, whatever the state says", () => {
    const list = FlatText.phases(session({ state: "running", phase: "done", step: 1, steps: 1, frames: 32, exposure_s: 0.0390625 }));
    assert.deepEqual(list.map((item) => item.state), ["done", "done", "done", "done"]);
  });

  test("a finished session that lost its exposure still names its frames", () => {
    const list = FlatText.phases(session({ state: "ok", frames: 32, exposure_s: null }));
    assert.equal(list[1].detail, "");
    assert.equal(list[2].detail, "32 frames.");
  });

  test("the progress message is the message of core while a session waits or runs, and the summary after", () => {
    assert.equal(FlatText.progressMessage(session({ state: "queued", message: "The scheduler is paused." })), "The scheduler is paused.");
    assert.equal(FlatText.progressMessage(session({ state: "queued" })), "Waiting for the next step of the scheduler.");
    assert.equal(FlatText.progressMessage(session({ state: "running", message: "Frame 3 of 32: 50 % of full scale." })), "Frame 3 of 32: 50 % of full scale.");
    assert.equal(FlatText.progressMessage(session({ state: "ok", summary: "Made the flat." })), "Made the flat.");
  });

  test("the gauge shows the level against the target with a word", () => {
    const on = FlatText.levelGauge(session({ state: "running", level_fraction: 0.51, saturated_fraction: 0 }));
    assert.equal(on.word, "On target");
    assert.equal(on.level, "good");
    assert.equal(on.fill, 51);
    assert.equal(on.mark, 50);
    assert.equal(on.text, "Level 51 % of full scale. Aim: 50 %.");
    const dark = FlatText.levelGauge(session({ state: "running", level_fraction: 0.4, saturated_fraction: 0 }));
    assert.deepEqual([dark.word, dark.level], ["Too dark", "warn"]);
    const darker = FlatText.levelGauge(session({ state: "running", level_fraction: 0.2, saturated_fraction: 0 }));
    assert.deepEqual([darker.word, darker.level], ["Too dark", "bad"]);
    const bright = FlatText.levelGauge(session({ state: "running", level_fraction: 0.62, saturated_fraction: 0 }));
    assert.deepEqual([bright.word, bright.level], ["Too bright", "warn"]);
    const saturated = FlatText.levelGauge(session({ state: "running", level_fraction: 0.5, saturated_fraction: 0.02 }));
    assert.deepEqual([saturated.word, saturated.level], ["Too bright", "bad"]);
  });

  test("the gauge stays inside the bar, and it hides when there is no level or no session", () => {
    assert.equal(FlatText.levelGauge(session({ state: "running", level_fraction: 1.7 })).fill, 100);
    assert.equal(FlatText.levelGauge(session({ state: "running", level_fraction: -0.1 })).fill, 0);
    assert.equal(FlatText.levelGauge(session({ state: "running", level_fraction: null })), null);
    assert.equal(FlatText.levelGauge(session({ state: "ok", level_fraction: 0.5 })), null);
    assert.equal(FlatText.levelGauge(task()), null);
  });

  test("the notes of a session are sentences, and empty ones drop out", () => {
    assert.deepEqual(FlatText.notes(task({ warnings: ["the light drifts", "", "Already a sentence."] })), ["The light drifts.", "Already a sentence."]);
    assert.deepEqual(FlatText.notes(task()), []);
  });

  test("a finished session leaves a banner, and a session under way or none leaves nothing", () => {
    assert.equal(FlatText.result(library({ task: task() })), null);
    assert.equal(FlatText.result(library({ task: session({ state: "running" }) })), null);
    assert.equal(FlatText.result(library({ task: session({ state: "queued" }) })), null);
    const ok = FlatText.result(library({ task: session({ state: "ok", summary: "Made the flat flat-5e6f7a8b.", version: "flat-5e6f7a8b" }) }));
    assert.deepEqual([ok.level, ok.title, ok.text, ok.version], ["good", "The session ended", "Made the flat flat-5e6f7a8b.", "flat-5e6f7a8b"]);
    const failed = FlatText.result(library({ task: session({ state: "failed", summary: "Not enough light: the frame reaches 6 % of full scale at the longest exposure of 1 s. Use a brighter source, or hold it closer to the lens. The library is unchanged." }) }));
    assert.deepEqual([failed.level, failed.title], ["bad", "The session failed"]);
    assert.ok(failed.text.startsWith("Not enough light:"));
    const aborted = FlatText.result(library({ task: session({ state: "aborted", summary: "The flat session stopped before it added a flat, and the library is unchanged." }) }));
    assert.deepEqual([aborted.level, aborted.title], ["warn", "The session was stopped"]);
    assert.equal(FlatText.result(library({ task: session({ state: "failed" }) })).text, "The session ended without a summary.");
  });

  test("the notice about the paused scheduler fits what the session did", () => {
    const waiting = { version: "flat-5e6f7a8b", t_utc: "2026-10-04T20:15:00Z", expires_utc: "2026-10-05T20:15:00Z", frames: 32, exposure_s: 0.039 };
    const ok = FlatText.resumeNotice(library({ task: session({ state: "ok" }), session: waiting }));
    assert.equal(ok.title, "Remove the light, then press Resume.");
    assert.ok(ok.text.startsWith("The scheduler is paused, so the station records nothing"));
    assert.ok(ok.text.endsWith("To take a second set, leave the light on."));
    const done = FlatText.resumeNotice(library({ task: session({ state: "ok" }) }));
    assert.equal(done.text.includes("second set"), false);
    for (const state of ["failed", "aborted"]) {
      const failed = FlatText.resumeNotice(library({ task: session({ state }), session: waiting }));
      assert.ok(failed.title.startsWith("Press Take flat to try again"));
      assert.equal(failed.text.includes("second set"), false);
    }
  });

  // --- The review ---------------------------------------------------------------------------------

  function byLabel(review, label) {
    return review.items.find((item) => item.label === label);
  }

  test("a flat like the ones of the owner's lens gets good verdicts with a sentence each", () => {
    const review = FlatText.verdicts(flat());
    assert.equal(review.level, "good");
    assert.equal(review.word, "Good");
    assert.deepEqual(review.items.map((item) => item.label), ["Corners", "Tilt", "Dust", "Noise", "Frames"]);
    assert.equal(byLabel(review, "Corners").value, "9.6 % less light than the center");
    assert.equal(byLabel(review, "Dust").value, "3 dust shadows deeper than 1.0 %");
    assert.equal(byLabel(review, "Noise").value, "0.11 % per pixel");
    assert.equal(byLabel(review, "Frames").value, "31 of 32 frames used");
    for (const item of review.items) {
      assert.equal(item.word, "Good");
      assert.ok(item.why.length > 20, item.label);
    }
  });

  test("the tilt of one set may include the light source, and the verdict says so", () => {
    const small = byLabel(FlatText.verdicts(flat()), "Tilt");
    assert.equal(small.level, "good");
    assert.equal(small.value, MINUS + "0.62 % across the width, +0.41 % across the height");
    assert.ok(small.why.includes("gradient of your light source"));
    assert.ok(small.why.includes("180 degrees"));
    const large = byLabel(FlatText.verdicts(flat({ tilt: { width_percent: 1.8, height_percent: 0.1 } })), "Tilt");
    assert.equal(large.level, "warn");
    assert.equal(large.word, "Check");
  });

  test("a flat of two sets shows the tilt of the optics apart from the light source", () => {
    const two = flat({
      second_set: true,
      source_turned: true,
      optics_tilt: { width_percent: -0.4, height_percent: 0.3 },
      source_tilt: { width_percent: -0.22, height_percent: 0.11 },
      agreement: { smooth_rms_percent: 0.09, fine_rms_percent: 0.13, expected_fine_rms_percent: 0.12, plane: null },
    });
    const review = FlatText.verdicts(two);
    const tilt = byLabel(review, "Tilt");
    assert.ok(tilt.value.startsWith("optics " + MINUS + "0.40 % across the width"));
    assert.ok(tilt.value.includes("; light source " + MINUS + "0.22 %"));
    assert.equal(tilt.level, "good");
    const agree = byLabel(review, "Two sets");
    assert.equal(agree.value, "fine part 0.13 % (the noise predicts 0.12 %)");
    assert.equal(agree.level, "good");
    assert.equal(review.level, "good");
  });

  test("two sets that differ more than their noise explains get a warning, and a large difference a problem", () => {
    const base = { second_set: true, optics_tilt: { width_percent: 0, height_percent: 0 }, source_tilt: { width_percent: 0, height_percent: 0 } };
    const warn = FlatText.verdicts(flat(Object.assign({}, base, { agreement: { fine_rms_percent: 0.3, expected_fine_rms_percent: 0.12, smooth_rms_percent: 0.1, plane: null } })));
    assert.equal(byLabel(warn, "Two sets").level, "warn");
    assert.equal(warn.level, "warn");
    const bad = FlatText.verdicts(flat(Object.assign({}, base, { agreement: { fine_rms_percent: 0.9, expected_fine_rms_percent: 0.12, smooth_rms_percent: 0.1, plane: null } })));
    assert.equal(byLabel(bad, "Two sets").level, "bad");
    assert.equal(bad.level, "bad");
    assert.equal(bad.word, "Problem");
  });

  test("corners that lose a lot of light mean that the light does not cover the lens", () => {
    assert.equal(byLabel(FlatText.verdicts(flat({ corner_percent: -2 })), "Corners").level, "good");
    assert.equal(byLabel(FlatText.verdicts(flat({ corner_percent: -25 })), "Corners").level, "good");
    const warn = byLabel(FlatText.verdicts(flat({ corner_percent: -33 })), "Corners");
    assert.equal(warn.level, "warn");
    assert.ok(warn.why.includes("covers the whole lens"));
    const bad = FlatText.verdicts(flat({ corner_percent: -62 }));
    assert.equal(byLabel(bad, "Corners").level, "bad");
    assert.equal(bad.level, "bad");
    assert.ok(byLabel(bad, "Corners").why.includes("does not cover the whole lens"));
  });

  test("corners that get more light than the center are worth a look", () => {
    const corner = byLabel(FlatText.verdicts(flat({ corner_percent: 3 })), "Corners");
    assert.equal(corner.value, "3.0 % more light than the center");
    assert.equal(corner.level, "warn");
    assert.equal(byLabel(FlatText.verdicts(flat({ corner_percent: 12 })), "Corners").level, "bad");
    assert.equal(byLabel(FlatText.verdicts(flat({ corner_percent: null })), "Corners").level, "warn");
  });

  test("noise, dust, and used frames have thresholds that name the next step", () => {
    assert.equal(byLabel(FlatText.verdicts(flat({ noise_percent: 0.4 })), "Noise").level, "warn");
    assert.ok(byLabel(FlatText.verdicts(flat({ noise_percent: 0.4 })), "Noise").why.includes("More frames"));
    assert.equal(byLabel(FlatText.verdicts(flat({ noise_percent: 0.9 })), "Noise").level, "bad");
    assert.equal(byLabel(FlatText.verdicts(flat({ shadows: 0 })), "Dust").value, "No dust shadows deeper than 1.0 %");
    assert.equal(byLabel(FlatText.verdicts(flat({ shadows: 1 })), "Dust").value, "1 dust shadow deeper than 1.0 %");
    assert.equal(byLabel(FlatText.verdicts(flat({ shadows: 12 })), "Dust").level, "warn");
    assert.equal(byLabel(FlatText.verdicts(flat({ shadows: 40 })), "Dust").level, "bad");
    assert.equal(byLabel(FlatText.verdicts(flat({ frames_taken: 32, frames_used: 26 })), "Frames").level, "warn");
    assert.equal(byLabel(FlatText.verdicts(flat({ frames_taken: 32, frames_used: 10 })), "Frames").level, "bad");
  });

  test("the verdict on the whole flat is the worst of the verdicts", () => {
    const review = FlatText.verdicts(flat({ noise_percent: 0.4, corner_percent: -62 }));
    assert.equal(review.level, "bad");
    assert.ok(review.text.includes("Take the flat again"));
    const check = FlatText.verdicts(flat({ noise_percent: 0.4 }));
    assert.equal(check.level, "warn");
    assert.ok(check.text.includes("Check"));
  });

  test("the notes of a flat are sentences, and the one that the tilt verdict says already drops out", () => {
    assert.deepEqual(FlatText.flatNotes(flat({ warnings: ["the light drifts", ""] })), ["The light drifts."]);
    assert.deepEqual(FlatText.flatNotes(flat()), []);
    const tilt = "The tilt may include the gradient of your light source, up to about 1% for a phone screen. A second set with the source turned by 180 degrees separates the two.";
    assert.deepEqual(FlatText.flatNotes(flat({ warnings: ["The light drifts: a frame is 3.4 % above the median level.", tilt] })), ["The light drifts: a frame is 3.4 % above the median level."]);
  });

  test("the second set is offered only for the first set of a session, when no session runs", () => {
    const pending = flat();
    const waiting = { version: pending.version, t_utc: "2026-10-04T20:15:00Z", expires_utc: "2026-10-05T20:15:00Z", frames: 32, exposure_s: 0.039 };
    const base = { flats: [pending], pending_version: pending.version, session: waiting };
    assert.equal(FlatText.canTakeSecondSet(library(base)), true);
    assert.equal(FlatText.canTakeSecondSet(library(Object.assign({}, base, { session: null }))), false);
    assert.equal(FlatText.canTakeSecondSet(library(Object.assign({}, base, { task: session({ state: "running" }) }))), false);
    assert.equal(FlatText.canTakeSecondSet(library(Object.assign({}, base, { blocker: "Record a dark set first (Dark page)." }))), false);
    assert.equal(FlatText.canTakeSecondSet(library(Object.assign({}, base, { flats: [flat({ second_set: true })] }))), false);
    assert.equal(FlatText.canTakeSecondSet(library(Object.assign({}, base, { session: Object.assign({}, waiting, { version: "flat-00000000" }) }))), false);
  });

  test("the sentence about the second set says why, and until when the first set stays", () => {
    const waiting = { version: "flat-5e6f7a8b", t_utc: "2026-10-04T20:15:00Z", expires_utc: "2026-10-05T20:15:00Z", frames: 32, exposure_s: 0.039 };
    const text = FlatText.secondSetText(library({ session: waiting }));
    assert.ok(text.startsWith(FlatText.secondSetWhy));
    assert.ok(text.endsWith("The first set stays until 2026-10-05 20:15 UTC."));
    assert.equal(FlatText.secondSetText(library()), FlatText.secondSetWhy);
  });

  // --- The library --------------------------------------------------------------------------------

  test("a flat in use, a flat that waits, and a flat used before have their own words", () => {
    assert.deepEqual(FlatText.stateWord(flat({ active: true, pending: false })), { word: "In use", level: "good" });
    assert.deepEqual(FlatText.stateWord(flat()), { word: "Waiting for you", level: "warn" });
    assert.deepEqual(FlatText.stateWord(flat({ pending: false, state: "approved" })), { word: "Used before", level: "good" });
  });

  test("the rows list the numbers of each flat and the actions that a person may take", () => {
    const pending = flat({ version: "flat-11111111" });
    const active = flat({ version: "flat-22222222", active: true, pending: false, state: "approved", age_days: 12.3, second_set: true, frames_used: 63, corner_percent: -9.9, shadows: 4, noise_percent: 0.088 });
    const older = flat({ version: "flat-33333333", pending: false, state: "approved", age_days: 47.2, t_utc: "2026-08-18T01:00:00Z" });
    const rows = FlatText.rows(library({ flats: [pending, active, older] }));
    assert.deepEqual(rows.map((row) => row.version), ["flat-11111111", "flat-22222222", "flat-33333333"]);
    assert.deepEqual(rows[0].actions, ["use", "discard"]);
    assert.deepEqual(rows[1].actions, []);
    assert.deepEqual(rows[2].actions, ["again", "delete"]);
    assert.equal(rows[1].made, "2026-10-04, 12 days ago");
    assert.equal(rows[1].corners, MINUS + "9.9 %");
    assert.equal(rows[1].shadows, "4");
    assert.equal(rows[1].noise, "0.09 %");
    assert.equal(rows[1].sets, "2 sets, 63 frames");
    assert.equal(rows[2].sets, "1 set, 31 frames");
    assert.equal(rows[1].state.word, "In use");
    assert.equal(FlatText.rows(library()).length, 0);
  });

  // --- The form -----------------------------------------------------------------------------------

  test("an empty form takes the defaults of the server and keeps the pause", () => {
    const checked = FlatText.validate({ frames: "", target: "", pauseAfter: true });
    assert.equal(checked.ok, true);
    assert.deepEqual(checked.body, { pause_after: true });
    assert.deepEqual(FlatText.validate({ frames: " ", target: " ", pauseAfter: false }).body, { pause_after: false });
  });

  test("the values of the form become the body of the request", () => {
    const checked = FlatText.validate({ frames: "24", target: "42.5", pauseAfter: true });
    assert.equal(checked.ok, true);
    assert.deepEqual(checked.body, { pause_after: true, frames: 24, target_fraction: 0.425 });
    assert.equal(FlatText.validate({ frames: "8", target: "30" }).body.target_fraction, 0.3);
    assert.equal(FlatText.validate({ frames: "64", target: "70" }).body.frames, 64);
    assert.equal(FlatText.defaults.frames, 32);
    assert.equal(FlatText.defaults.targetPercent, 50);
  });

  test("a value out of range or not a number gets a sentence for its field", () => {
    for (const frames of ["7", "65", "12.5", "many", "-3", "1e2"]) {
      const checked = FlatText.validate({ frames, target: "", pauseAfter: true });
      assert.equal(checked.ok, false, frames);
      assert.equal(checked.errors.frames, "Enter a whole number of frames from 8 to 64.");
      assert.equal("frames" in checked.body, false);
    }
    for (const target of ["29.9", "70.1", "half", "-5", "5e1", "50%", "1.234"]) {
      const checked = FlatText.validate({ frames: "", target, pauseAfter: true });
      assert.equal(checked.ok, false, target);
      assert.equal(checked.errors.target, "Enter a brightness target from 30 to 70 percent, such as 50.");
    }
  });

  test("the answer of the server names the field that it refused", () => {
    const found = FlatText.serverErrors({
      details: [
        { field: "body.frames", message: "Input should be greater than or equal to 8", type: "greater_than_equal" },
        { field: "body.target_fraction", message: "Input should be less than or equal to 0.7", type: "less_than_equal" },
        { field: "body.set_number", message: "Input should be 1 or 2", type: "literal_error" },
      ],
    });
    assert.deepEqual(found, {
      frames: "Input should be greater than or equal to 8.",
      target: "Input should be less than or equal to 0.7.",
      set_number: "Input should be 1 or 2.",
    });
    assert.deepEqual(FlatText.serverErrors(null), {});
    assert.deepEqual(FlatText.serverErrors({ details: null }), {});
  });
};
