"use strict";

/*
 * The logic of the Dark page that needs no page: the status line of the library, the phases of a
 * dark session, the hint about when a session starts, the check of the form, how often to poll, and
 * the numbers of the chart (the axes, the points, and the line of the model). The functions are pure
 * (they read values and return values), so Node can run them (tests/services/web/js/).
 * `dark.js` puts the results into the page.
 *
 * A dark set is recorded with the camera covered. The library holds sets at different sensor
 * temperatures, because the dark current doubles about every 6 degrees. The model needs sets across
 * 0 to 25 degrees C to fit that doubling.
 */
(function () {
  const { fmt } = window.Seeing;

  const DEGREE_C = "°C";
  const ACTIVE = ["queued", "running"];
  const FAST_POLL_MS = 2000;
  const SLOW_POLL_MS = 10000;
  const MIN_FRAMES = 3;
  const MAX_FRAMES = 50;
  const MAX_EXPOSURE_S = 600;
  const MODEL_FROM_C = 0;
  const MODEL_TO_C = 25;
  const TEMPERATURE_STEP_C = 5;
  const PHASES = [
    { id: "bias", label: "Bias frames" },
    { id: "cover", label: "Waiting for the cover" },
    { id: "dark", label: "Dark frames" },
    { id: "build", label: "Building the master dark" },
  ];

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

  /** The dark current with three significant digits: 0.0493, 0.118, 1.23, and 12.3. */
  function rate(value) {
    if (value === null || value === undefined || Number.isNaN(value)) {
      return fmt.dash;
    }
    if (value === 0) {
      return "0";
    }
    const digits = Math.min(6, Math.max(0, 2 - Math.floor(Math.log10(Math.abs(value)))));
    return value.toFixed(digits);
  }

  function temperature(value, digits) {
    const text = fmt.num(value, digits === undefined ? 1 : digits);
    return text === fmt.dash ? text : text + " " + DEGREE_C;
  }

  /** The date of an ISO time: "2026-10-01". */
  function day(iso) {
    return iso ? iso.slice(0, 10) : fmt.dash;
  }

  // --- The state of the library -------------------------------------------------------------------

  function isActive(task) {
    return Boolean(task) && ACTIVE.includes(task.state);
  }

  /** How long to wait before the next read, in milliseconds: fast while a session is under way. */
  function pollInterval(task) {
    return isActive(task) ? FAST_POLL_MS : SLOW_POLL_MS;
  }

  /**
   * The status line of the library: a word, a level for its color, and the reason in plain words.
   * An empty library is due, and it says so with its own word.
   */
  function dueLine(library) {
    const status = library.status;
    const reason = sentence(degrees(status.reason));
    if (library.sets.length === 0) {
      return { word: "Empty", level: "warn", text: reason || "The library holds no dark set." };
    }
    if (status.due) {
      return { word: "Due", level: "warn", text: reason };
    }
    return { word: "Up to date", level: "good", text: reason };
  }

  function temperatureLine(library) {
    const value = library.sensor_temperature_c;
    if (value === null || value === undefined) {
      return "The sensor temperature is not reported.";
    }
    return "Sensor temperature now: " + temperature(value, 1) + ".";
  }

  /**
   * What to tell a person who is about to start a session: when it begins. `state` is the state of
   * the scheduler (`null` when the server cannot reach core).
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
        return "The scheduler is paused. A session starts after you press Resume.";
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

  /** The words about the latest check of a frame while the session waits for the cover. */
  function coverText(task) {
    if (task.covered === true) {
      return "The frame is dark, so the camera is covered.";
    }
    if (task.covered === false) {
      const level = task.level_dn === null || task.level_dn === undefined ? "" : " (level " + fmt.num(task.level_dn, 0) + " DN)";
      const why = task.reason ? ": " + task.reason : "";
      return "The frame is not dark yet" + why + level + ". Cover the camera.";
    }
    return "Waiting for the first check of a frame.";
  }

  function countText(count) {
    return count === null || count === undefined ? "" : count + (count === 1 ? " frame" : " frames");
  }

  /**
   * The phases of a session with their states, for the list of progress. A phase is `done`,
   * `active`, or `pending`. A session that waits in the queue has no active phase, and a session
   * that ended `ok` has every phase done. The cover phase is there only when the session waits for
   * the cover.
   */
  function phases(task) {
    const running = PHASES.findIndex((phase) => phase.id === task.phase);
    return PHASES.filter((phase) => phase.id !== "cover" || task.wait_for_cover).map((phase) => {
      const index = PHASES.indexOf(phase);
      let state = "pending";
      if (task.state === "ok") {
        state = "done";
      } else if (task.state === "running" && running >= 0) {
        state = index < running ? "done" : index === running ? "active" : "pending";
      }
      let detail = "";
      if (phase.id === "bias" || phase.id === "dark") {
        const total = phase.id === "bias" ? task.bias_frames : task.frames;
        detail = state === "active" && task.steps > 0 ? task.step + " of " + task.steps : countText(total);
      } else if (phase.id === "cover" && state === "active") {
        detail = coverText(task);
      } else if (phase.id === "cover" && state === "done") {
        detail = "The camera was covered.";
      }
      return { id: phase.id, label: phase.label, state, detail };
    });
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

  /** What a finished session leaves: its sentence, and the set that it added (when it added one). */
  function result(library) {
    const task = library.task;
    if (isActive(task) || task.state === "idle") {
      return null;
    }
    const added = task.set_name ? library.sets.find((item) => item.name === task.set_name) : null;
    return {
      state: task.state,
      level: task.state === "ok" ? "good" : task.state === "aborted" ? "warn" : "bad",
      title: task.state === "ok" ? "The session ended" : task.state === "aborted" ? "The session was aborted" : "The session failed",
      text: degrees(task.summary) || "The session ended without a summary.",
      set: added
        ? {
            name: added.name,
            temperature: temperature(added.temperature_c, 1),
            rate: rate(added.rate_e_per_s) + " e⁻/s",
            hotPixels: String(added.hot_pixels),
          }
        : null,
    };
  }

  // --- The form -----------------------------------------------------------------------------------

  function wholeNumber(text) {
    return /^\d{1,9}$/.test(text) ? Number(text) : null;
  }

  /**
   * Check the values of the form and build the body of the request. `values` holds the text of each
   * field. An empty field means "use the configured value" and leaves the key out of the body, or
   * sends `null`. The server holds the exposure to its own limit, and it answers 422 above it.
   * Returns `{ ok, body, errors }`, where `errors` maps a field to a sentence.
   */
  function validate(values) {
    const errors = {};
    const body = { wait_for_cover: Boolean(values.waitForCover), pause_after: Boolean(values.pauseAfter) };
    const exposureText = String(values.exposure || "").trim();
    if (exposureText !== "") {
      const exposure = /^\d{0,6}(\.\d{0,3})?$/.test(exposureText) ? Number(exposureText) : NaN;
      if (!Number.isFinite(exposure) || exposure <= 0 || exposure > MAX_EXPOSURE_S) {
        errors.exposure = "Enter an exposure between 0 and " + MAX_EXPOSURE_S + " seconds, such as 30.";
      } else {
        body.exposure_s = exposure;
      }
    }
    for (const [key, field, name] of [["frames", "frames", "dark frames"], ["biasFrames", "bias_frames", "bias frames"]]) {
      const text = String(values[key] || "").trim();
      if (text === "") {
        continue;
      }
      const count = wholeNumber(text);
      if (count === null || count < MIN_FRAMES || count > MAX_FRAMES) {
        errors[key] = "Enter a whole number of " + name + " from " + MIN_FRAMES + " to " + MAX_FRAMES + ".";
      } else {
        body[field] = count;
      }
    }
    return { ok: Object.keys(errors).length === 0, body, errors };
  }

  /** The sentences of a 422 answer of the server, one for each field that it names. */
  function serverErrors(error) {
    const found = {};
    for (const detail of (error && error.details) || []) {
      const field = String(detail.field || "").replace(/^body\./, "");
      const key = field === "bias_frames" ? "biasFrames" : field === "exposure_s" ? "exposure" : field;
      found[key] = sentence(detail.message);
    }
    return found;
  }

  // --- The chart ----------------------------------------------------------------------------------

  /** The shortest decimal text of a round value: 0.005, 0.1, 1, and 20. */
  function tickLabel(value) {
    const text = String(Number(value.toPrecision(6)));
    return text.includes("e") ? value.toFixed(8).replace(/0+$/, "") : text;
  }

  /** Round values 1, 2, and 5 of each power of ten between `min` and `max` (both positive). */
  function rateTicks(min, max) {
    const ticks = [];
    for (let power = Math.floor(Math.log10(min)); power <= Math.ceil(Math.log10(max)); power += 1) {
      for (const base of [1, 2, 5]) {
        const value = base * Math.pow(10, power);
        if (value >= min * 0.999999 && value <= max * 1.000001) {
          ticks.push({ value, label: tickLabel(value) });
        }
      }
    }
    return ticks;
  }

  /** The rate of the model at a temperature: the rate at the reference, doubled every `doubling_c`. */
  function modelRate(model, temperatureC) {
    return model.rate_ref_e_per_s * Math.pow(2, (temperatureC - model.reference_c) / model.doubling_c);
  }

  /**
   * The numbers of the chart of the dark rate against the temperature. The temperature axis always
   * shows 0 to 25 degrees C, which the model needs, and it grows to hold every set and the sensor
   * temperature. The rate axis is logarithmic and holds every set and the line of the model. The
   * line is straight on that axis, so two points draw it.
   */
  function chartData(library) {
    const sets = library.sets.filter((item) => item.rate_e_per_s > 0);
    const sensor = library.sensor_temperature_c === undefined ? null : library.sensor_temperature_c;
    const temperatures = sets.map((item) => item.temperature_c).concat(sensor === null ? [] : [sensor]);
    const low = Math.min(MODEL_FROM_C, ...temperatures);
    const high = Math.max(MODEL_TO_C, ...temperatures);
    const xMin = Math.floor((low - 1) / TEMPERATURE_STEP_C) * TEMPERATURE_STEP_C;
    const xMax = Math.ceil((high + 1) / TEMPERATURE_STEP_C) * TEMPERATURE_STEP_C;
    const model = library.model || null;
    const curve = model
      ? [xMin, xMax].map((x) => ({ x, rate: modelRate(model, x) }))
      : null;
    const rates = sets.map((item) => item.rate_e_per_s).concat(curve ? curve.map((p) => p.rate) : []);
    const empty = rates.length === 0;
    const yMin = empty ? 0.01 : Math.min(...rates) / 1.6;
    const yMax = empty ? 1 : Math.max(...rates) * 1.6;
    const xTicks = [];
    for (let x = xMin; x <= xMax; x += TEMPERATURE_STEP_C) {
      xTicks.push(x);
    }
    const newest = sets.length > 0 ? sets[0].name : null;
    return {
      xMin,
      xMax,
      yMin,
      yMax,
      xTicks,
      yTicks: rateTicks(yMin, yMax),
      band: { from: MODEL_FROM_C, to: MODEL_TO_C },
      points: sets.map((item) => ({
        x: item.temperature_c,
        rate: item.rate_e_per_s,
        name: item.name,
        newest: item.name === newest,
        text: temperature(item.temperature_c, 1) + ", " + rate(item.rate_e_per_s) + " e⁻/s, " + item.hot_pixels + " hot pixels, " + day(item.t_utc),
      })),
      curve,
      sensor,
      summary: chartSummary(sets, sensor, model),
    };
  }

  function chartSummary(sets, sensor, model) {
    const parts = ["Dark rate against sensor temperature on a logarithmic scale: " + sets.length + (sets.length === 1 ? " set" : " sets")];
    if (sets.length > 0) {
      const temps = sets.map((item) => item.temperature_c);
      parts[0] += " from " + temperature(Math.min(...temps), 1) + " to " + temperature(Math.max(...temps), 1);
    }
    if (model) {
      parts.push("a model that doubles every " + fmt.num(model.doubling_c, 1) + " " + DEGREE_C);
    }
    if (sensor !== null) {
      parts.push("the sensor is at " + temperature(sensor, 1));
    }
    return parts.join(", ") + ".";
  }

  /** The sentence under the chart: what the model needs, and what it has. */
  function modelNote(library) {
    const needs = "The model needs sets across " + MODEL_FROM_C + " to " + MODEL_TO_C + " " + DEGREE_C + ".";
    const model = library.model;
    if (!model) {
      return "There is no model yet. " + needs;
    }
    const kind = model.doubling_fitted ? "fitted to the sets" : "assumed, because the sets are too few to fit it";
    return "The line is the model: the dark current doubles every " + fmt.num(model.doubling_c, 1) + " " + DEGREE_C + " (" + kind + "). " + needs;
  }

  window.Seeing.DarkText = {
    sentence,
    degrees,
    rate,
    temperature,
    day,
    isActive,
    pollInterval,
    dueLine,
    temperatureLine,
    startHint,
    coverText,
    phases,
    progressMessage,
    result,
    validate,
    serverErrors,
    rateTicks,
    modelRate,
    chartData,
    modelNote,
    limits: { MIN_FRAMES, MAX_FRAMES, MAX_EXPOSURE_S, FAST_POLL_MS, SLOW_POLL_MS },
  };
})();
