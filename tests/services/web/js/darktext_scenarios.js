"use strict";

/*
 * The scenarios of the logic of the Dark page (static/js/darktext.js): the status line of the
 * library, the hint about when a session starts, the phases of a session, the check of the form,
 * how often to poll, and the numbers of the chart. The function takes `DarkText`, a `test(name, fn)`
 * function, and an `assert` object, so that Node (darktext.test.js) and a browser console can both
 * run it.
 */
module.exports = function scenarios(DarkText, test, assert) {
  const DASH = "—";

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
        covered: null,
        level_dn: null,
        reason: "",
        exposure_s: null,
        frames: null,
        bias_frames: null,
        wait_for_cover: true,
        pause_after: true,
        started_utc: null,
        finished_utc: null,
        summary: "",
        set_name: null,
      },
      extra
    );
  }

  /** A task that was queued with 9 dark frames and 6 bias frames. */
  function session(extra) {
    return task(Object.assign({ task_id: 3, exposure_s: 30, frames: 9, bias_frames: 6 }, extra));
  }

  function set(name, temperatureC, rate, extra) {
    return Object.assign(
      {
        name,
        t_utc: "2026-09-28T03:12:00Z",
        age_days: 4,
        temperature_c: temperatureC,
        temperature_spread_c: 0.3,
        exposure_s: 30,
        n_frames: 9,
        n_bias_frames: 9,
        rate_e_per_s: rate,
        hot_pixels: 187,
      },
      extra
    );
  }

  function library(extra) {
    return Object.assign(
      {
        mode: "bin2",
        gain: 120,
        exposure_s: 30,
        sensor_temperature_c: 12.3,
        status: { due: true, reason: "no recent set within 3.0 C of 12.3 C (the nearest is 4.2 C away)", tolerance_c: 3, max_age_days: 183 },
        model: null,
        sets: [],
        task: task(),
      },
      extra
    );
  }

  // --- Words ------------------------------------------------------------------------------------

  test("a clause of core becomes a sentence, and a degree sign follows the number", () => {
    assert.equal(DarkText.sentence("the library holds no dark set"), "The library holds no dark set.");
    assert.equal(DarkText.sentence("already a sentence."), "Already a sentence.");
    assert.equal(DarkText.sentence("  "), "");
    assert.equal(DarkText.sentence(null), "");
    assert.equal(DarkText.degrees("within 3.0 C of 18.4 C (6.2 C away)"), "within 3.0 °C of 18.4 °C (6.2 °C away)");
    assert.equal(DarkText.degrees("a Cold frame, 5 Celsius"), "a Cold frame, 5 Celsius");
  });

  test("the dark current has three significant digits, and a missing value is a dash", () => {
    assert.equal(DarkText.rate(0.04931), "0.0493");
    assert.equal(DarkText.rate(0.118), "0.118");
    assert.equal(DarkText.rate(1.2345), "1.23");
    assert.equal(DarkText.rate(12.345), "12.3");
    assert.equal(DarkText.rate(0), "0");
    assert.equal(DarkText.rate(null), DASH);
    assert.equal(DarkText.temperature(12.34, 1), "12.3 °C");
    assert.equal(DarkText.temperature(null, 1), DASH);
    assert.equal(DarkText.day("2026-09-28T03:12:00Z"), "2026-09-28");
  });

  // --- The status line --------------------------------------------------------------------------

  test("a library that is due is amber, and it gives the reason as a sentence", () => {
    const line = DarkText.dueLine(library({ sets: [set("a", 20, 0.1)] }));
    assert.deepEqual(line, {
      word: "Due",
      level: "warn",
      text: "No recent set within 3.0 °C of 12.3 °C (the nearest is 4.2 °C away).",
    });
  });

  test("a library that is up to date is green", () => {
    const line = DarkText.dueLine(
      library({ sets: [set("a", 12, 0.05)], status: { due: false, reason: "a set lies within 3.0 C of 12.3 C", tolerance_c: 3, max_age_days: 183 } })
    );
    assert.equal(line.word, "Up to date");
    assert.equal(line.level, "good");
    assert.equal(line.text, "A set lies within 3.0 °C of 12.3 °C.");
  });

  test("an empty library says so, even when core calls it due", () => {
    const line = DarkText.dueLine(library({ status: { due: true, reason: "the library holds no dark set", tolerance_c: 3, max_age_days: 183 } }));
    assert.equal(line.word, "Empty");
    assert.equal(line.level, "warn");
    assert.equal(line.text, "The library holds no dark set.");
    assert.equal(DarkText.dueLine(library({ status: { due: true, reason: "", tolerance_c: 3, max_age_days: 183 } })).text, "The library holds no dark set.");
  });

  test("the sensor temperature is in the status line, or the page says that core does not report it", () => {
    assert.equal(DarkText.temperatureLine(library()), "Sensor temperature now: 12.3 °C.");
    assert.equal(DarkText.temperatureLine(library({ sensor_temperature_c: null })), "The sensor temperature is not reported.");
  });

  // --- When a session starts ----------------------------------------------------------------------

  test("the hint says when a session starts for each state of the scheduler", () => {
    const idle = task();
    assert.equal(DarkText.startHint("auto", idle), "A session starts at the next step of the scheduler. An exposure in progress finishes first.");
    assert.equal(DarkText.startHint("safe", idle), "The scheduler is in safe, so a session starts at once.");
    assert.equal(DarkText.startHint("paused", idle), "The scheduler is paused. A session starts after you press Resume.");
    assert.equal(DarkText.startHint("align", idle), "The alignment helper runs. A session starts after it ends.");
    assert.equal(DarkText.startHint("commission", idle), "Another task runs. A session starts after it.");
    assert.equal(DarkText.startHint(null, idle), "The server cannot reach core, so it cannot start a session now.");
    assert.equal(DarkText.startHint(undefined, idle), "");
  });

  test("the hint is silent while a session is queued or running, and it is back after the session", () => {
    assert.equal(DarkText.startHint("auto", session({ state: "queued" })), "");
    assert.equal(DarkText.startHint("auto", session({ state: "running", phase: "bias" })), "");
    assert.notEqual(DarkText.startHint("auto", session({ state: "ok" })), "");
    assert.notEqual(DarkText.startHint("auto", session({ state: "failed" })), "");
  });

  // --- The phases -------------------------------------------------------------------------------

  function states(list) {
    return list.map((phase) => phase.id + ":" + phase.state).join(" ");
  }

  test("a queued session lists every phase as pending, with the frames that it will take", () => {
    const list = DarkText.phases(session({ state: "queued", message: "Waiting for the next step of the scheduler." }));
    assert.equal(states(list), "bias:pending cover:pending dark:pending build:pending");
    assert.equal(list[0].detail, "6 frames");
    assert.equal(list[1].detail, "");
    assert.equal(list[2].detail, "9 frames");
  });

  test("the bias phase is active with its step", () => {
    const list = DarkText.phases(session({ state: "running", phase: "bias", step: 2, steps: 6 }));
    assert.equal(states(list), "bias:active cover:pending dark:pending build:pending");
    assert.equal(list[0].detail, "2 of 6");
  });

  test("the cover phase says in words that the frame is not dark yet, with the reason and the level", () => {
    const list = DarkText.phases(
      session({ state: "running", phase: "cover", covered: false, level_dn: 2412.5, reason: "the median is 2400 counts above the expected level" })
    );
    assert.equal(states(list), "bias:done cover:active dark:pending build:pending");
    assert.equal(
      list[1].detail,
      "The frame is not dark yet: the median is 2400 counts above the expected level (level 2413 DN). Cover the camera."
    );
    assert.equal(list[0].detail, "6 frames");
  });

  test("the cover phase says that the camera is covered once a test frame is dark", () => {
    const list = DarkText.phases(session({ state: "running", phase: "cover", covered: true, level_dn: 11.8 }));
    assert.equal(list[1].detail, "The frame is dark, so the camera is covered.");
    assert.equal(DarkText.coverText(session({ covered: null })), "Waiting for the first check of a frame.");
    assert.equal(DarkText.coverText(session({ covered: false, level_dn: null, reason: "" })), "The frame is not dark yet. Cover the camera.");
  });

  test("the dark and the build phases follow, and the earlier phases are done", () => {
    const dark = DarkText.phases(session({ state: "running", phase: "dark", step: 4, steps: 9, covered: true }));
    assert.equal(states(dark), "bias:done cover:done dark:active build:pending");
    assert.equal(dark[1].detail, "The camera was covered.");
    assert.equal(dark[2].detail, "4 of 9");
    const build = DarkText.phases(session({ state: "running", phase: "build" }));
    assert.equal(states(build), "bias:done cover:done dark:done build:active");
    assert.equal(build[2].detail, "9 frames");
  });

  test("a session that does not wait for the cover has no cover phase", () => {
    const list = DarkText.phases(session({ state: "running", phase: "dark", step: 1, steps: 9, wait_for_cover: false }));
    assert.equal(states(list), "bias:done dark:active build:pending");
  });

  test("a finished session has every phase done", () => {
    assert.equal(states(DarkText.phases(session({ state: "ok" }))), "bias:done cover:done dark:done build:done");
  });

  test("the message of the progress is core's own words, with a fallback", () => {
    assert.equal(DarkText.progressMessage(session({ state: "queued", message: "The scheduler is paused. The dark session starts after you resume it." })), "The scheduler is paused. The dark session starts after you resume it.");
    assert.equal(DarkText.progressMessage(session({ state: "queued", message: "" })), "Waiting for the next step of the scheduler.");
    assert.equal(DarkText.progressMessage(session({ state: "running", message: "Dark frame 4 of 9." })), "Dark frame 4 of 9.");
    assert.equal(DarkText.progressMessage(session({ state: "ok", summary: "Added the set x." })), "Added the set x.");
  });

  test("a degree in the sentence of core gets its sign, in the progress and in the result", () => {
    const ended = session({ state: "ok", summary: "Added the set x at 12.3 C: 9 dark frames." });
    assert.equal(DarkText.progressMessage(ended), "Added the set x at 12.3 °C: 9 dark frames.");
    assert.equal(DarkText.result(library({ task: ended })).text, "Added the set x at 12.3 °C: 9 dark frames.");
  });

  test("a running session has no result yet, and an idle one has none either", () => {
    assert.equal(DarkText.result(library({ task: session({ state: "running" }) })), null);
    assert.equal(DarkText.result(library({ task: session({ state: "queued" }) })), null);
    assert.equal(DarkText.result(library()), null);
  });

  test("a session that ended ok shows its sentence and the set that it added", () => {
    const added = set("dark-20261001T030000Z-bin2-g120.fits", 12.3, 0.0493, { hot_pixels: 201 });
    const result = DarkText.result(
      library({
        sets: [added, set("old", 4, 0.01)],
        task: session({ state: "ok", summary: "Added the set dark-20261001T030000Z-bin2-g120.fits.", set_name: added.name }),
      })
    );
    assert.equal(result.level, "good");
    assert.equal(result.title, "The session ended");
    assert.equal(result.text, "Added the set dark-20261001T030000Z-bin2-g120.fits.");
    assert.deepEqual(result.set, { name: added.name, temperature: "12.3 °C", rate: "0.0493 e⁻/s", hotPixels: "201" });
  });

  test("a session that failed or was aborted gives its reason and adds no set", () => {
    const failed = DarkText.result(library({ task: session({ state: "failed", summary: "The first dark frame was not dark." }) }));
    assert.equal(failed.level, "bad");
    assert.equal(failed.title, "The session failed");
    assert.equal(failed.text, "The first dark frame was not dark.");
    assert.equal(failed.set, null);
    const aborted = DarkText.result(library({ task: session({ state: "aborted", summary: "The scheduler paused." }) }));
    assert.equal(aborted.level, "warn");
    assert.equal(aborted.title, "The session was aborted");
    assert.equal(aborted.set, null);
    assert.equal(DarkText.result(library({ task: session({ state: "failed", summary: "" }) })).text, "The session ended without a summary.");
  });

  test("a finished session whose set the list lacks still shows its sentence", () => {
    const result = DarkText.result(library({ task: session({ state: "ok", summary: "Added x.", set_name: "gone.fits" }) }));
    assert.equal(result.set, null);
    assert.equal(result.text, "Added x.");
  });

  // --- Polling ----------------------------------------------------------------------------------

  test("the page reads every 2 seconds while a session is queued or running, and every 10 otherwise", () => {
    assert.equal(DarkText.pollInterval(session({ state: "queued" })), 2000);
    assert.equal(DarkText.pollInterval(session({ state: "running" })), 2000);
    for (const state of ["idle", "ok", "failed", "aborted"]) {
      assert.equal(DarkText.pollInterval(task({ state })), 10000);
    }
    assert.equal(DarkText.pollInterval(null), 10000);
    assert.equal(DarkText.isActive(null), false);
  });

  // --- The form ---------------------------------------------------------------------------------

  const BLANK = { exposure: "", frames: "", biasFrames: "", waitForCover: true, pauseAfter: true };

  test("an empty form asks for the configured values and keeps both boxes", () => {
    const checked = DarkText.validate(BLANK);
    assert.equal(checked.ok, true);
    assert.deepEqual(checked.body, { wait_for_cover: true, pause_after: true });
    assert.deepEqual(checked.errors, {});
  });

  test("values in range go into the body as numbers, and an unchecked box says false", () => {
    const checked = DarkText.validate({ exposure: " 45.5 ", frames: "12", biasFrames: "3", waitForCover: false, pauseAfter: false });
    assert.equal(checked.ok, true);
    assert.deepEqual(checked.body, { wait_for_cover: false, pause_after: false, exposure_s: 45.5, frames: 12, bias_frames: 3 });
  });

  test("the bounds are accepted", () => {
    const low = DarkText.validate(Object.assign({}, BLANK, { exposure: "0.001", frames: "3", biasFrames: "3" }));
    assert.equal(low.ok, true);
    const high = DarkText.validate(Object.assign({}, BLANK, { exposure: "600", frames: "50", biasFrames: "50" }));
    assert.equal(high.ok, true);
    assert.equal(high.body.exposure_s, 600);
  });

  test("an exposure that is not a positive number up to 600 is refused with a sentence", () => {
    for (const bad of ["0", "-5", "abc", "601", "1e2", ".", "12,5", "30 s", "0.0"]) {
      const checked = DarkText.validate(Object.assign({}, BLANK, { exposure: bad }));
      assert.equal(checked.ok, false, bad);
      assert.equal(checked.errors.exposure, "Enter an exposure between 0 and 600 seconds, such as 30.", bad);
      assert.equal("exposure_s" in checked.body, false);
    }
  });

  test("a frame count that is not a whole number from 3 to 50 is refused, for each of the two fields", () => {
    for (const bad of ["2", "51", "3.5", "-4", "many", "1e1"]) {
      const frames = DarkText.validate(Object.assign({}, BLANK, { frames: bad }));
      assert.equal(frames.ok, false, bad);
      assert.equal(frames.errors.frames, "Enter a whole number of dark frames from 3 to 50.", bad);
      const bias = DarkText.validate(Object.assign({}, BLANK, { biasFrames: bad }));
      assert.equal(bias.ok, false, bad);
      assert.equal(bias.errors.biasFrames, "Enter a whole number of bias frames from 3 to 50.", bad);
    }
  });

  test("every wrong field is named at once", () => {
    const checked = DarkText.validate({ exposure: "x", frames: "1", biasFrames: "99", waitForCover: true, pauseAfter: true });
    assert.deepEqual(Object.keys(checked.errors).sort(), ["biasFrames", "exposure", "frames"]);
  });

  test("the sentences of a 422 answer land on the fields that the server names", () => {
    const found = DarkText.serverErrors({
      details: [
        { field: "body.exposure_s", message: "Input should be less than or equal to 120", type: "less_than_equal" },
        { field: "body.bias_frames", message: "Input should be greater than or equal to 3", type: "greater_than_equal" },
        { field: "body.label", message: "String should have at most 10 characters", type: "string_too_long" },
      ],
    });
    assert.deepEqual(found, {
      exposure: "Input should be less than or equal to 120.",
      biasFrames: "Input should be greater than or equal to 3.",
      label: "String should have at most 10 characters.",
    });
    assert.deepEqual(DarkText.serverErrors(null), {});
    assert.deepEqual(DarkText.serverErrors({ details: null }), {});
  });

  // --- The chart --------------------------------------------------------------------------------

  const MODEL = { reference_c: 20, rate_ref_e_per_s: 0.12, doubling_c: 6, doubling_fitted: true, rms_log2: 0.05, n_sets: 3 };

  test("the ticks of the rate axis are 1, 2, and 5 of each power of ten", () => {
    assert.deepEqual(
      DarkText.rateTicks(0.004, 0.3).map((tick) => tick.label),
      ["0.005", "0.01", "0.02", "0.05", "0.1", "0.2"]
    );
    assert.deepEqual(
      DarkText.rateTicks(0.9, 12).map((tick) => tick.label),
      ["1", "2", "5", "10"]
    );
  });

  test("the model doubles the rate every doubling step from the reference", () => {
    assert.equal(DarkText.modelRate(MODEL, 20), 0.12);
    assert.equal(DarkText.modelRate(MODEL, 26), 0.24);
    assert.equal(DarkText.modelRate(MODEL, 14), 0.06);
  });

  test("the temperature axis always shows 0 to 25 degrees, and it grows to hold the sets and now", () => {
    const narrow = DarkText.chartData(library({ sets: [set("a", 10, 0.03), set("b", 15, 0.06)], sensor_temperature_c: 12 }));
    assert.equal(narrow.xMin, -5);
    assert.equal(narrow.xMax, 30);
    assert.deepEqual(narrow.band, { from: 0, to: 25 });
    assert.deepEqual(narrow.xTicks, [-5, 0, 5, 10, 15, 20, 25, 30]);
    const wide = DarkText.chartData(library({ sets: [set("a", -12, 0.001), set("b", 41, 3)], sensor_temperature_c: 12 }));
    assert.equal(wide.xMin, -15);
    assert.equal(wide.xMax, 45);
    const hot = DarkText.chartData(library({ sets: [set("a", 10, 0.03)], sensor_temperature_c: 33.5 }));
    assert.equal(hot.xMax, 35);
  });

  test("the chart holds a point for each set, the newest first, and the sensor temperature", () => {
    const chart = DarkText.chartData(library({ sets: [set("new", 12.3, 0.05), set("old", 4, 0.01)], sensor_temperature_c: 14 }));
    assert.deepEqual(
      chart.points.map((point) => [point.name, point.x, point.rate, point.newest]),
      [["new", 12.3, 0.05, true], ["old", 4, 0.01, false]]
    );
    assert.equal(chart.points[0].text, "12.3 °C, 0.0500 e⁻/s, 187 hot pixels, 2026-09-28");
    assert.equal(chart.sensor, 14);
    assert.equal(chart.curve, null);
  });

  test("the rate axis holds every set and, with a model, the line at both ends", () => {
    const sets = [set("a", 5, 0.02), set("b", 20, 0.12)];
    const without = DarkText.chartData(library({ sets }));
    assert.ok(Math.abs(without.yMin - 0.02 / 1.6) < 1e-12);
    assert.ok(Math.abs(without.yMax - 0.12 * 1.6) < 1e-12);
    const withModel = DarkText.chartData(library({ sets, model: MODEL }));
    assert.deepEqual(withModel.curve.map((point) => point.x), [-5, 30]);
    assert.ok(Math.abs(withModel.curve[0].rate - 0.12 * Math.pow(2, -25 / 6)) < 1e-12);
    assert.ok(withModel.yMin <= withModel.curve[0].rate);
    assert.ok(withModel.yMax >= withModel.curve[1].rate);
  });

  test("an empty library still has a chart with axes, and a set without a rate stays out", () => {
    const empty = DarkText.chartData(library({ sensor_temperature_c: null }));
    assert.equal(empty.points.length, 0);
    assert.equal(empty.sensor, null);
    assert.equal(empty.xMin, -5);
    assert.ok(empty.yMax > empty.yMin);
    assert.ok(empty.yTicks.length > 0);
    assert.equal(DarkText.chartData(library({ sets: [set("zero", 10, 0)] })).points.length, 0);
  });

  test("the summary of the chart says what it shows, for a screen reader", () => {
    const chart = DarkText.chartData(library({ sets: [set("a", 4, 0.01), set("b", 12.3, 0.05)], model: MODEL, sensor_temperature_c: 14 }));
    assert.equal(
      chart.summary,
      "Dark rate against sensor temperature on a logarithmic scale: 2 sets from 4.0 °C to 12.3 °C, a model that doubles every 6.0 °C, the sensor is at 14.0 °C."
    );
    assert.equal(
      DarkText.chartData(library({ sensor_temperature_c: null })).summary,
      "Dark rate against sensor temperature on a logarithmic scale: 0 sets."
    );
  });

  test("the note under the chart says that the model needs sets across 0 to 25 degrees", () => {
    assert.equal(
      DarkText.modelNote(library()),
      "There is no model yet. The model needs sets across 0 to 25 °C."
    );
    assert.equal(
      DarkText.modelNote(library({ model: MODEL })),
      "The line is the model: the dark current doubles every 6.0 °C (fitted to the sets). The model needs sets across 0 to 25 °C."
    );
    const assumed = DarkText.modelNote(library({ model: Object.assign({}, MODEL, { doubling_fitted: false }) }));
    assert.ok(assumed.includes("assumed, because the sets are too few to fit it"));
  });
};
