"use strict";

/*
 * The logic of the Flat page that needs no page: the status line of the library, the hint about when
 * a session starts, the phases of a session and the level of its frames, what a finished session
 * leaves, the verdicts on a new flat, the rows of the library, the check of the form, and how often
 * to poll. The functions are pure (they read values and return values), so Node can run them
 * (tests/services/web/js/). `flat.js` puts the results into the page.
 *
 * A flat shows how much light each pixel gets when the whole field is lit evenly. The owner lights
 * the lens with an even light source, the station takes frames, and it combines them into a pending
 * flat. The owner looks at it and decides: use it, or discard it. The verdicts here are rules of
 * thumb, and each one says why in a sentence.
 */
(function () {
  const { fmt } = window.Seeing;

  const ACTIVE = ["queued", "running"];
  const FAST_POLL_MS = 2000;
  const SLOW_POLL_MS = 10000;
  const MIN_FRAMES = 8;
  const MAX_FRAMES = 64;
  const MIN_TARGET_PERCENT = 30;
  const MAX_TARGET_PERCENT = 70;
  const DEFAULT_FRAMES = 32;
  const DEFAULT_TARGET_PERCENT = 50;
  const ON_TARGET = 0.1; // a level within 10 % of the target is on target
  const NEAR_TARGET = 0.25;
  const CORNER_GOOD_PERCENT = 25; // a lens loses up to this much light in the corners
  const CORNER_WARN_PERCENT = 40;
  const TILT_GOOD_PERCENT = 1;
  const NOISE_GOOD_PERCENT = 0.25;
  const NOISE_WARN_PERCENT = 0.5;
  const USED_GOOD = 0.9;
  const USED_WARN = 0.7;
  const SHADOWS_GOOD = 8;
  const SHADOWS_WARN = 20;
  const AGREE_GOOD = 1.5;
  const AGREE_WARN = 3;
  const DEGREE_C = "°C";
  const WORDS = { good: "Good", warn: "Check", bad: "Problem" };
  const PHASES = [
    { id: "setup", label: "Setting up" },
    { id: "exposure", label: "Finding the exposure" },
    { id: "capture", label: "Taking frames" },
    { id: "build", label: "Combining the frames" },
  ];
  const SECOND_SET_WHY =
    "A second set with the light source turned by 180 degrees separates the gradient of the light source from the tilt of the lens, so the flat comes out more accurate.";

  // --- Words ------------------------------------------------------------------------------------

  /** A sentence from a clause of core: the first letter is a capital, and a period ends it. */
  function sentence(text) {
    const clean = String(text || "").trim();
    if (clean === "") {
      return "";
    }
    const capital = clean.charAt(0).toUpperCase() + clean.slice(1);
    return /[.!?]$/.test(capital) ? capital : capital + ".";
  }

  /** "6.2 C" becomes "6.2 °C". Core writes the unit as a plain C. */
  function degrees(text) {
    return String(text || "").replace(/(\d) C(?![A-Za-z])/g, "$1 " + DEGREE_C);
  }

  function missing(value) {
    return value === null || value === undefined || Number.isNaN(value);
  }

  /** An exposure in words: "1.5 s", "39.1 ms", or "32 µs". */
  function exposureText(seconds) {
    if (missing(seconds)) {
      return fmt.dash;
    }
    if (seconds >= 100) {
      return Math.round(seconds) + " s";
    }
    if (seconds >= 1) {
      return Number(seconds.toPrecision(3)) + " s";
    }
    if (seconds >= 0.001) {
      return Number((seconds * 1000).toPrecision(3)) + " ms";
    }
    return Number((seconds * 1e6).toPrecision(3)) + " µs";
  }

  /** A fraction of the full scale as a whole percent: 0.498 becomes "50 %". */
  function levelText(fraction) {
    return missing(fraction) ? fmt.dash : fmt.percent(fraction, 0);
  }

  /** A value that is a percent already, with a sign and a unit: "−0.62 %". */
  function signedPercent(value, digits) {
    return missing(value) ? fmt.dash : fmt.signed(value, digits === undefined ? 2 : digits) + " %";
  }

  function tiltText(tilt) {
    if (!tilt || (missing(tilt.width_percent) && missing(tilt.height_percent))) {
      return fmt.dash;
    }
    return signedPercent(tilt.width_percent) + " across the width, " + signedPercent(tilt.height_percent) + " across the height";
  }

  /** The date of an ISO time: "2026-10-01". */
  function day(iso) {
    return iso ? iso.slice(0, 10) : fmt.dash;
  }

  /** The age of a flat in words: "today", "yesterday", or "12 days ago". */
  function ageText(days) {
    if (missing(days)) {
      return fmt.dash;
    }
    if (days < 1) {
      return "today";
    }
    const whole = Math.floor(days);
    return whole === 1 ? "yesterday" : whole + " days ago";
  }

  // --- The state of the library -------------------------------------------------------------------

  function isActive(task) {
    return Boolean(task) && ACTIVE.includes(task.state);
  }

  /** How long to wait before the next read, in milliseconds: fast while a session is under way. */
  function pollInterval(task) {
    return isActive(task) ? FAST_POLL_MS : SLOW_POLL_MS;
  }

  function findFlat(library, version) {
    return version ? library.flats.find((item) => item.version === version) || null : null;
  }

  /** The newest flat that waits for a decision, or `null`. */
  function pendingFlat(library) {
    return findFlat(library, library.pending_version);
  }

  /** The flat that the survey divides by, or `null`. */
  function activeFlat(library) {
    return findFlat(library, library.active_version);
  }

  /**
   * The status line of the library: a word, a level for its color, and a sentence. A flat in use
   * is good. Without one, the survey either uses the flat file of its settings or applies no
   * correction.
   */
  function libraryLine(library) {
    const flat = activeFlat(library);
    if (flat) {
      let text = "The survey divides by the flat " + flat.version + ", made " + ageText(flat.age_days) + ".";
      if (library.library_overrides) {
        text += " It replaces the flat of the setting flat_file.";
      }
      return { word: "In use", level: "good", text };
    }
    if (library.flat_file_pinned) {
      return { word: "From settings", level: "good", text: "No flat of the library is in use. The survey uses the flat file of its settings." };
    }
    return {
      word: "None",
      level: "warn",
      text: "No flat is in use, so the survey does not correct for the lens, and a star at the edge of the frame looks fainter than one in the middle.",
    };
  }

  function temperatureLine(library) {
    const value = library.sensor_temperature_c;
    if (missing(value)) {
      return "The sensor temperature is not reported.";
    }
    return "Sensor temperature now: " + fmt.num(value, 1) + " " + DEGREE_C + ".";
  }

  /**
   * What to tell a person who is about to start a session: when it begins. `state` is the state of
   * the scheduler (`null` when the server cannot reach core). A paused scheduler resumes when the
   * page starts a session, and it pauses again at the end.
   */
  function startHint(state, task) {
    if (isActive(task)) {
      return "";
    }
    switch (state) {
      case undefined:
        return ""; // the status has not arrived yet
      case null:
        return "The server cannot reach core, so it cannot start a session now.";
      case "paused":
        return "The scheduler is paused. Taking a flat resumes it, and it pauses again when the session ends.";
      case "align":
        return "The alignment helper runs. A session starts after it ends.";
      case "safe":
        return "The scheduler is in safe, so a session starts at once.";
      case "commission":
        return "Another task runs. A session starts after it.";
      default:
        return "A session starts at the next step of the scheduler. An exposure in progress finishes first.";
    }
  }

  // --- The session --------------------------------------------------------------------------------

  /**
   * The phases of a session with their states, for the list of progress. A phase is `done`,
   * `active`, or `pending`. A session that waits in the queue has no active phase, and a session
   * that ended `ok` has every phase done.
   */
  function phases(task) {
    const running = PHASES.findIndex((phase) => phase.id === task.phase);
    const finishing = task.phase === "done"; // the flat is stored, and the task ends a moment later
    return PHASES.map((phase, index) => {
      let state = "pending";
      if (task.state === "ok" || finishing) {
        state = "done";
      } else if (task.state === "running" && running >= 0) {
        state = index < running ? "done" : index === running ? "active" : "pending";
      }
      return { id: phase.id, label: phase.label, state, detail: phaseDetail(phase.id, state, task) };
    });
  }

  function phaseDetail(id, state, task) {
    if (state === "pending") {
      return id === "capture" && task.frames ? task.frames + " frames" : "";
    }
    switch (id) {
      case "setup":
        return state === "active" ? "Checking the camera and the dark library." : "The camera and the library are ready.";
      case "exposure": {
        if (state === "done") {
          return missing(task.exposure_s) ? "" : "Exposure " + exposureText(task.exposure_s) + ".";
        }
        const tries = task.step > 0 ? "Try " + task.step + " (at most " + task.steps + ")" : "Starting the search";
        const found = missing(task.exposure_s) ? "" : ": " + exposureText(task.exposure_s) + " gives " + levelText(task.level_fraction) + " of full scale";
        return tries + found + ".";
      }
      case "capture":
        if (state === "done") {
          return task.frames + " frames" + (missing(task.exposure_s) ? "" : " of " + exposureText(task.exposure_s)) + ".";
        }
        return task.step > 0 ? "Frame " + task.step + " of " + task.steps + "." : "Starting the frames.";
      default:
        return state === "active" ? "Combining the frames into a flat. This takes a while." : "The flat is stored.";
    }
  }

  /** The progress as one sentence for the live region: core's message, or what the queue means. */
  function progressMessage(task) {
    if (task.state === "queued") {
      return degrees(task.message) || "Waiting for the next step of the scheduler.";
    }
    if (task.state === "running") {
      return degrees(task.message) || "The session runs.";
    }
    return degrees(task.summary);
  }

  /**
   * The level of the frames against the target, for the gauge: how full the bar is, where the target
   * mark sits (both in percent of the full scale), and a word, a level, and a sentence. Core reports
   * the level while it searches for the exposure and while it takes frames.
   */
  function levelGauge(task) {
    if (!isActive(task) || missing(task.level_fraction) || missing(task.target_fraction)) {
      return null;
    }
    const level = task.level_fraction;
    const target = task.target_fraction;
    const deviation = (level - target) / target;
    let word = "On target";
    let tone = "good";
    if (task.saturated_fraction > 0.001) {
      word = "Too bright";
      tone = "bad";
    } else if (Math.abs(deviation) > ON_TARGET) {
      word = deviation < 0 ? "Too dark" : "Too bright";
      tone = Math.abs(deviation) > NEAR_TARGET ? "bad" : "warn";
    }
    const clamp = (value) => Number(Math.max(0, Math.min(100, value * 100)).toFixed(2));
    return {
      fill: clamp(level),
      mark: clamp(target),
      word,
      level: tone,
      text: "Level " + levelText(level) + " of full scale. Aim: " + levelText(target) + ".",
    };
  }

  /** The notes of a running or finished session, as sentences. */
  function notes(task) {
    return (task.warnings || []).map(sentence).filter((text) => text !== "");
  }

  /** What a finished session leaves: its sentence, and the flat that it added (when it added one). */
  function result(library) {
    const task = library.task;
    if (isActive(task) || task.state === "idle") {
      return null;
    }
    const titles = { ok: "The session ended", aborted: "The session was stopped", failed: "The session failed" };
    return {
      state: task.state,
      level: task.state === "ok" ? "good" : task.state === "aborted" ? "warn" : "bad",
      title: titles[task.state] || "The session ended",
      text: degrees(task.summary) || "The session ended without a summary.",
      version: task.version || null,
    };
  }

  /**
   * What to tell a person when the scheduler is paused after a session: a title and a sentence. A
   * session that ended ok leaves a flat to look at, and the first set of a session may get a second
   * set while the light stays on. After a failure or a stop, the person may try again at once.
   */
  function resumeNotice(library) {
    const paused = "The scheduler is paused, so the station records nothing until you resume it.";
    if (library.task.state === "ok") {
      return {
        title: "Remove the light, then press Resume.",
        text: library.session ? paused + " To take a second set, leave the light on." : paused,
      };
    }
    return { title: "Press Take flat to try again, or remove the light and press Resume.", text: paused };
  }

  // --- The review of a new flat -------------------------------------------------------------------

  function verdict(level, label, value, why) {
    return { label, value, level, word: WORDS[level], why };
  }

  function cornerVerdict(flat) {
    const value = flat.corner_percent;
    if (missing(value)) {
      return verdict("warn", "Corners", fmt.dash, "The report has no number for the corners.");
    }
    const text = fmt.num(Math.abs(value), 1) + " % " + (value <= 0 ? "less" : "more") + " light than the center";
    const loss = -value;
    if (loss <= CORNER_GOOD_PERCENT && value <= 1) {
      return verdict("good", "Corners", text, "A lens loses some light toward the edge. The flat corrects it.");
    }
    if (value > 1) {
      return verdict(value <= 5 ? "warn" : "bad", "Corners", text, "The corners get more light than the center. Check that the light is even.");
    }
    if (loss <= CORNER_WARN_PERCENT) {
      return verdict("warn", "Corners", text, "The corners lose more light than a lens usually does. Check that the light covers the whole lens.");
    }
    return verdict("bad", "Corners", text, "The light probably does not cover the whole lens. Hold the light source closer, or use a larger one.");
  }

  function tiltVerdict(flat) {
    if (flat.second_set && flat.optics_tilt) {
      const text = "optics " + tiltText(flat.optics_tilt) + "; light source " + tiltText(flat.source_tilt);
      return verdict("good", "Tilt", text, "The second set separates the tilt of the lens from the gradient of the light source. The flat keeps the first.");
    }
    const tilt = flat.tilt || {};
    const size = Math.max(Math.abs(tilt.width_percent || 0), Math.abs(tilt.height_percent || 0));
    const why = "This tilt may include the gradient of your light source. " + SECOND_SET_WHY;
    return verdict(size <= TILT_GOOD_PERCENT ? "good" : "warn", "Tilt", tiltText(tilt), why);
  }

  function shadowVerdict(flat) {
    const count = flat.shadows || 0;
    const depth = missing(flat.shadow_min_depth_percent) ? "" : " deeper than " + fmt.num(flat.shadow_min_depth_percent, 1) + " %";
    const text = count === 0 ? "No dust shadows" + depth : count + (count === 1 ? " dust shadow" : " dust shadows") + depth;
    if (count <= SHADOWS_GOOD) {
      return verdict("good", "Dust", text, "A dust shadow is a faint ring. The flat corrects it while the dust stays where it is.");
    }
    return verdict(count <= SHADOWS_WARN ? "warn" : "bad", "Dust", text, "Many shadows. Clean the lens and the window of the camera, then take a new flat.");
  }

  function noiseVerdict(flat) {
    const value = flat.noise_percent;
    if (missing(value)) {
      return verdict("warn", "Noise", fmt.dash, "The report has no number for the noise.");
    }
    const text = fmt.num(value, 2) + " % per pixel";
    if (value <= NOISE_GOOD_PERCENT) {
      return verdict("good", "Noise", text, "Low. The noise of the flat adds to the noise of every measurement, and this adds little.");
    }
    if (value <= NOISE_WARN_PERCENT) {
      return verdict("warn", "Noise", text, "A little high. More frames make it lower.");
    }
    return verdict("bad", "Noise", text, "High. Take more frames, or use a brighter light.");
  }

  function framesVerdict(flat) {
    const taken = flat.frames_taken || 0;
    const used = flat.frames_used || 0;
    const text = used + " of " + taken + " frames used";
    const share = taken > 0 ? used / taken : 0;
    if (share >= USED_GOOD) {
      return verdict("good", "Frames", text, "A frame that fails a check is left out of the flat. Leaving a few out is normal.");
    }
    return verdict(share >= USED_WARN ? "warn" : "bad", "Frames", text, "Many frames failed a check, so the light was not steady. Check the notes below.");
  }

  function agreementVerdict(flat) {
    const agreement = flat.agreement;
    if (!agreement || missing(agreement.fine_rms_percent)) {
      return null;
    }
    const expected = agreement.expected_fine_rms_percent;
    const text = "fine part " + fmt.num(agreement.fine_rms_percent, 2) + " %" + (missing(expected) ? "" : " (the noise predicts " + fmt.num(expected, 2) + " %)");
    const ratio = missing(expected) || expected <= 0 ? 1 : agreement.fine_rms_percent / expected;
    if (ratio <= AGREE_GOOD) {
      return verdict("good", "Two sets", text, "The two sets agree within their noise.");
    }
    return verdict(ratio <= AGREE_WARN ? "warn" : "bad", "Two sets", text, "The sets differ more than their noise explains. The light may have drifted between them.");
  }

  /**
   * The verdicts on a new flat: the numbers that matter, each with a level, a word, and a sentence
   * that says why. The worst level is the verdict on the whole flat.
   */
  function verdicts(flat) {
    const found = [cornerVerdict(flat), tiltVerdict(flat), shadowVerdict(flat), noiseVerdict(flat), framesVerdict(flat), agreementVerdict(flat)].filter(Boolean);
    const order = ["good", "warn", "bad"];
    const worst = found.reduce((level, item) => (order.indexOf(item.level) > order.indexOf(level) ? item.level : level), "good");
    const overall = {
      good: { word: "Good", text: "The numbers look right. You can use this flat." },
      warn: { word: "Check", text: "Look at the numbers that say Check before you use this flat." },
      bad: { word: "Problem", text: "Something is wrong. Take the flat again, or read the notes first." },
    }[worst];
    return { items: found, level: worst, word: overall.word, text: overall.text };
  }

  // The verdict on the tilt says this already, with its own words.
  const SAID_BY_A_VERDICT = ["The tilt may include the gradient of your light source"];

  /** The notes of a flat: what the session and the combination noticed, apart from what a verdict says. */
  function flatNotes(flat) {
    return (flat.warnings || []).map(sentence).filter((text) => text !== "" && !SAID_BY_A_VERDICT.some((start) => text.startsWith(start)));
  }

  /**
   * Whether the page offers a second set: the first set of a session waits, it made the flat under
   * review, and no session runs.
   */
  function canTakeSecondSet(library) {
    const flat = pendingFlat(library);
    return Boolean(library.session) && Boolean(flat) && library.session.version === flat.version && !flat.second_set && !isActive(library.task) && library.blocker === null;
  }

  /** The sentence about the second set, with the time when the first set goes. */
  function secondSetText(library) {
    const session = library.session;
    const until = session ? " The first set stays until " + session.expires_utc.slice(0, 16).replace("T", " ") + " UTC." : "";
    return SECOND_SET_WHY + until;
  }

  // --- The library --------------------------------------------------------------------------------

  /** The state of a flat in words, with a level. */
  function stateWord(flat) {
    if (flat.active) {
      return { word: "In use", level: "good" };
    }
    if (flat.pending) {
      return { word: "Waiting for you", level: "warn" };
    }
    return { word: "Used before", level: "good" };
  }

  /** The rows of the table of flats, newest first. `actions` names what a person may do with it. */
  function rows(library) {
    return library.flats.map((flat) => ({
      version: flat.version,
      made: day(flat.t_utc) + ", " + ageText(flat.age_days),
      state: stateWord(flat),
      corners: missing(flat.corner_percent) ? fmt.dash : fmt.signed(flat.corner_percent, 1) + " %",
      shadows: String(flat.shadows || 0),
      noise: missing(flat.noise_percent) ? fmt.dash : fmt.num(flat.noise_percent, 2) + " %",
      sets: (flat.second_set ? "2 sets, " : "1 set, ") + flat.frames_used + " frames",
      actions: flat.active ? [] : [flat.pending ? "use" : "again", flat.pending ? "discard" : "delete"],
    }));
  }

  // --- The form -----------------------------------------------------------------------------------

  function wholeNumber(text) {
    return /^\d{1,9}$/.test(text) ? Number(text) : null;
  }

  /**
   * Check the values of the form and build the body of the request. `values` holds the text of each
   * field. An empty field leaves its key out, which takes the default. The brightness target is a
   * percent in the form and a fraction in the request. Returns `{ ok, body, errors }`, where
   * `errors` maps a field to a sentence.
   */
  function validate(values) {
    const errors = {};
    const body = { pause_after: Boolean(values.pauseAfter) };
    const framesText = String(values.frames || "").trim();
    if (framesText !== "") {
      const count = wholeNumber(framesText);
      if (count === null || count < MIN_FRAMES || count > MAX_FRAMES) {
        errors.frames = "Enter a whole number of frames from " + MIN_FRAMES + " to " + MAX_FRAMES + ".";
      } else {
        body.frames = count;
      }
    }
    const targetText = String(values.target || "").trim();
    if (targetText !== "") {
      const percent = /^\d{1,3}(\.\d{0,2})?$/.test(targetText) ? Number(targetText) : NaN;
      if (!Number.isFinite(percent) || percent < MIN_TARGET_PERCENT || percent > MAX_TARGET_PERCENT) {
        errors.target = "Enter a brightness target from " + MIN_TARGET_PERCENT + " to " + MAX_TARGET_PERCENT + " percent, such as 50.";
      } else {
        body.target_fraction = Number((percent / 100).toFixed(4));
      }
    }
    return { ok: Object.keys(errors).length === 0, body, errors };
  }

  /** The sentences of a 422 answer of the server, one for each field that it names. */
  function serverErrors(error) {
    const found = {};
    for (const detail of (error && error.details) || []) {
      const field = String(detail.field || "").replace(/^body\./, "");
      const key = field === "target_fraction" ? "target" : field;
      found[key] = sentence(detail.message);
    }
    return found;
  }

  window.Seeing.FlatText = {
    sentence,
    degrees,
    exposureText,
    levelText,
    tiltText,
    day,
    ageText,
    isActive,
    pollInterval,
    pendingFlat,
    activeFlat,
    libraryLine,
    temperatureLine,
    startHint,
    phases,
    progressMessage,
    levelGauge,
    notes,
    result,
    resumeNotice,
    verdicts,
    flatNotes,
    canTakeSecondSet,
    secondSetText,
    stateWord,
    rows,
    validate,
    serverErrors,
    secondSetWhy: SECOND_SET_WHY,
    defaults: { frames: DEFAULT_FRAMES, targetPercent: DEFAULT_TARGET_PERCENT },
    limits: { MIN_FRAMES, MAX_FRAMES, MIN_TARGET_PERCENT, MAX_TARGET_PERCENT, FAST_POLL_MS, SLOW_POLL_MS },
  };
})();
