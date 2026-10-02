"use strict";

/*
 * The Dark page: the dark library with a chart of the dark rate against the temperature, a form that
 * starts a dark session, and the progress of the session. The owner covers the camera by hand,
 * presses Start, waits, removes the cover, and presses Resume, because the station pauses at the end
 * so that nothing records data with the camera covered. Cancel is the pause command.
 *
 * `darktext.js` holds the logic that needs no page (the words, the phases, the check of the form,
 * and the numbers of the chart). This file reads the API, puts the results into the page, and draws
 * the chart. The page reads `GET /dark` every 2 seconds while a session is queued or running, and
 * every 10 seconds otherwise, and a hidden page waits.
 */
(function () {
  const { h, $, clear, fmt, api, Status, Token, poller } = window.Seeing;
  const { DarkText } = window.Seeing;

  const state = {
    library: null,
    failed: null, // the error of the last read, or null
    sending: false,
    readCount: 0, // numbers the reads, so that an older answer never replaces a newer one
    applied: 0,
    previousTask: null, // the state of the task at the last read, to notice that it ended
    exposureTouched: false,
  };
  let plot = null;
  let loop = null;

  function token(name, fallback) {
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
  }

  function schedulerState() {
    const status = Status.current;
    if (!status || !status.core.reachable || !status.scheduler) {
      return status ? null : undefined;
    }
    return status.scheduler.state;
  }

  function commandsEnabled() {
    const status = Status.current;
    return !status || status.ui.commands_enabled;
  }

  // --- The chart --------------------------------------------------------------------------------

  /** A log-scale chart of the dark rate against the temperature: the sets, the model, and now. */
  class DarkPlot {
    constructor(container) {
      this.chart = null;
      this.hover = null;
      this.box = { left: 52, top: 22, width: 1, height: 1 };
      container.classList.add("plot", "chart");
      this.canvas = h("canvas", { role: "img", "aria-label": "Dark rate against sensor temperature" });
      this.tip = h("div", { class: "tip", hidden: true });
      container.append(this.canvas, this.tip);
      this.ctx = this.canvas.getContext("2d");
      if (typeof ResizeObserver === "function") {
        new ResizeObserver(() => this.draw()).observe(container);
      } else {
        window.addEventListener("resize", () => this.draw());
      }
      window.addEventListener("seeing:theme", () => this.draw());
      this.canvas.addEventListener("pointermove", (event) => this.pointer(event));
      this.canvas.addEventListener("pointerdown", (event) => this.pointer(event));
      this.canvas.addEventListener("pointerleave", () => this.leave());
      this.canvas.addEventListener("pointercancel", () => this.leave());
    }

    setData(chart) {
      this.chart = chart;
      this.hover = null;
      this.canvas.setAttribute("aria-label", chart.summary);
      this.draw();
    }

    setup() {
      const rect = this.canvas.getBoundingClientRect();
      const width = Math.max(1, Math.round(rect.width));
      const height = Math.max(1, Math.round(rect.height));
      const dpr = window.devicePixelRatio || 1;
      if (this.canvas.width !== Math.round(width * dpr) || this.canvas.height !== Math.round(height * dpr)) {
        this.canvas.width = Math.round(width * dpr);
        this.canvas.height = Math.round(height * dpr);
      }
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.box = { left: 52, top: 22, width: Math.max(1, width - 52 - 12), height: Math.max(1, height - 22 - 44), full: { width, height } };
      return this.box;
    }

    x(value) {
      const { xMin, xMax } = this.chart;
      return this.box.left + ((value - xMin) / (xMax - xMin)) * this.box.width;
    }

    y(rate) {
      const { yMin, yMax } = this.chart;
      const span = Math.log(yMax) - Math.log(yMin);
      return this.box.top + (1 - (Math.log(rate) - Math.log(yMin)) / span) * this.box.height;
    }

    draw() {
      if (!this.canvas.isConnected || !this.chart) {
        return;
      }
      const box = this.setup();
      const ctx = this.ctx;
      const chart = this.chart;
      const { width, height } = box.full;
      const muted = token("--muted", "#566176");
      const grid = token("--grid", "#e1e6ee");
      ctx.clearRect(0, 0, width, height);
      ctx.font = "12px system-ui, sans-serif";
      // The band of the temperatures that the model needs.
      ctx.fillStyle = token("--surface-2", "#eef1f6");
      ctx.fillRect(this.x(chart.band.from), box.top, this.x(chart.band.to) - this.x(chart.band.from), box.height);
      ctx.lineWidth = 1;
      ctx.strokeStyle = grid;
      ctx.fillStyle = muted;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (const tick of chart.yTicks) {
        const y = Math.round(this.y(tick.value)) + 0.5;
        ctx.beginPath();
        ctx.moveTo(box.left, y);
        ctx.lineTo(box.left + box.width, y);
        ctx.stroke();
        ctx.fillText(tick.label, box.left - 6, y);
      }
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      const skip = box.width / chart.xTicks.length < 30 ? 2 : 1;
      chart.xTicks.forEach((value, index) => {
        const x = Math.round(this.x(value)) + 0.5;
        ctx.beginPath();
        ctx.moveTo(x, box.top);
        ctx.lineTo(x, box.top + box.height);
        ctx.stroke();
        if (index % skip === 0) {
          ctx.fillText(fmt.num(value, 0), x, box.top + box.height + 6);
        }
      });
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText("e⁻/s per pixel", 2, 2);
      ctx.textAlign = "center";
      ctx.fillText("sensor temperature, °C", box.left + box.width / 2, box.top + box.height + 24);
      this.drawModel();
      this.drawSensor();
      this.drawPoints();
    }

    drawModel() {
      const chart = this.chart;
      if (!chart.curve) {
        return;
      }
      const ctx = this.ctx;
      ctx.save();
      ctx.beginPath();
      ctx.rect(this.box.left, this.box.top, this.box.width, this.box.height);
      ctx.clip();
      ctx.strokeStyle = token("--series-2", "#c2410c");
      ctx.lineWidth = 1.75;
      ctx.beginPath();
      chart.curve.forEach((point, index) => {
        const x = this.x(point.x);
        const y = this.y(point.rate);
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
      ctx.restore();
    }

    drawSensor() {
      const sensor = this.chart.sensor;
      if (sensor === null) {
        return;
      }
      const ctx = this.ctx;
      const x = Math.round(this.x(sensor)) + 0.5;
      ctx.strokeStyle = token("--text", "#17202e");
      ctx.fillStyle = token("--text", "#17202e");
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 3]);
      ctx.beginPath();
      ctx.moveTo(x, this.box.top);
      ctx.lineTo(x, this.box.top + this.box.height);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.textBaseline = "bottom";
      ctx.textAlign = x > this.box.left + this.box.width - 70 ? "right" : "left";
      ctx.fillText("now " + fmt.num(sensor, 1) + " °C", x + (ctx.textAlign === "left" ? 4 : -4), this.box.top - 2);
    }

    drawPoints() {
      const ctx = this.ctx;
      const color = token("--series-1", "#1f55c4");
      const surface = token("--surface", "#ffffff");
      for (const point of this.chart.points) {
        const x = this.x(point.x);
        const y = this.y(point.rate);
        ctx.fillStyle = color;
        ctx.strokeStyle = surface;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(x, y, point.newest ? 6 : 4.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        if (this.hover === point) {
          ctx.strokeStyle = token("--text", "#17202e");
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(x, y, 8, 0, Math.PI * 2);
          ctx.stroke();
        }
      }
    }

    pointer(event) {
      if (!this.chart) {
        return;
      }
      const rect = this.canvas.getBoundingClientRect();
      const px = event.clientX - rect.left;
      const py = event.clientY - rect.top;
      let best = null;
      let distance = 20;
      for (const point of this.chart.points) {
        const d = Math.hypot(this.x(point.x) - px, this.y(point.rate) - py);
        if (d < distance) {
          best = point;
          distance = d;
        }
      }
      this.hover = best;
      this.draw();
      if (!best) {
        this.tip.hidden = true;
        return;
      }
      this.tip.replaceChildren(h("div", { text: best.name }), h("div", { text: best.text }));
      this.tip.hidden = false;
      const tipWidth = this.tip.offsetWidth;
      const x = this.x(best.x);
      const flip = x + tipWidth + 14 > rect.width;
      this.tip.style.left = Math.max(0, flip ? x - tipWidth - 10 : x + 10) + "px";
      this.tip.style.top = Math.max(0, this.y(best.rate) - 30) + "px";
    }

    leave() {
      this.hover = null;
      this.tip.hidden = true;
      this.draw();
    }
  }

  // --- Rendering --------------------------------------------------------------------------------

  function showError(error) {
    const banner = $("dark-error");
    banner.hidden = !error;
    banner.dataset.level = "bad";
    banner.textContent = error ? error.message : "";
  }

  function renderStatus(library) {
    const line = DarkText.dueLine(library);
    const pill = $("due-pill");
    pill.textContent = line.word;
    pill.dataset.level = line.level;
    $("due-reason").textContent = line.text;
    $("temperature-line").textContent = DarkText.temperatureLine(library);
  }

  function row(set) {
    const cell = (label, text, title) => h("td", { text, title, role: "cell", dataset: { label } });
    const exposure = Number(set.exposure_s.toFixed(1)).toString() + " s";
    return h(
      "tr",
      { role: "row" },
      cell("Date", DarkText.day(set.t_utc), set.name),
      cell("Temperature", DarkText.temperature(set.temperature_c, 1), "Spread " + fmt.num(set.temperature_spread_c, 1) + " °C during the set"),
      cell("Frames", set.n_frames + " + " + set.n_bias_frames + " bias"),
      cell("Exposure", exposure),
      cell("Dark rate", DarkText.rate(set.rate_e_per_s) + " e⁻/s"),
      cell("Hot pixels", String(set.hot_pixels))
    );
  }

  function renderSets(library) {
    const empty = library.sets.length === 0;
    $("sets-empty").hidden = !empty;
    $("sets-wrap").hidden = empty;
    const body = clear($("sets-body"));
    for (const set of library.sets) {
      body.append(row(set));
    }
  }

  function legend(entries) {
    const box = clear($("legend-dark"));
    for (const [text, color, kind] of entries) {
      // The swatch has the shape of the mark in the chart, because the night mode paints every color red.
      const item = h("span", { text, dataset: { kind } });
      item.style.setProperty("--swatch", "var(" + color + ")");
      box.append(item);
    }
  }

  function renderChart(library) {
    if (!plot) {
      plot = new DarkPlot($("plot-dark"));
    }
    const chart = DarkText.chartData(library);
    plot.setData(chart);
    legend([["dark set", "--series-1", "dot"], ...(chart.curve ? [["model", "--series-2", "line"]] : []), ...(chart.sensor === null ? [] : [["sensor now", "--text", "dash"]])]);
    $("model-note").textContent = DarkText.modelNote(library);
  }

  function renderPhases(task) {
    const list = clear($("phases"));
    const show = DarkText.isActive(task) || task.state === "ok";
    list.hidden = !show;
    if (!show) {
      return;
    }
    for (const phase of DarkText.phases(task)) {
      const item = h(
        "li",
        { class: "phase", dataset: { state: phase.state }, "aria-current": phase.state === "active" ? "step" : null },
        h("span", { class: "mark", "aria-hidden": "true" }),
        h("span", { class: "phase-text" }, h("span", { class: "phase-name", text: phase.label }), phase.detail ? h("span", { class: "phase-detail", text: phase.detail }) : null)
      );
      list.append(item);
    }
  }

  function renderResult(library) {
    const result = DarkText.result(library);
    const banner = $("result");
    banner.hidden = result === null;
    if (result === null) {
      return;
    }
    banner.dataset.level = result.level;
    $("result-title").textContent = result.title;
    $("result-text").textContent = result.text;
    const facts = clear($("result-facts"));
    facts.hidden = result.set === null;
    if (result.set) {
      for (const [name, value] of [["Set", result.set.name], ["Temperature", result.set.temperature], ["Dark rate", result.set.rate], ["Hot pixels", result.set.hotPixels]]) {
        facts.append(h("dt", { text: name }), h("dd", { text: value }));
      }
    }
  }

  function renderTask(library) {
    const task = library.task;
    const active = DarkText.isActive(task);
    $("progress-empty").hidden = task.state !== "idle";
    renderPhases(task);
    $("task-message").textContent = active ? DarkText.progressMessage(task) : "";
    renderResult(library);
  }

  /** The buttons and the hints, which depend on the library, the status of the server, and the token. */
  function renderControls() {
    const library = state.library;
    const task = library ? library.task : null;
    const active = DarkText.isActive(task);
    const paused = schedulerState() === "paused";
    const enabled = commandsEnabled();
    const down = Boolean(state.failed) && state.failed.status === 503;
    $("dark-start").disabled = active || state.sending || !library || !enabled || down;
    $("dark-start").title = enabled ? "" : "The server has no token hash, so it refuses every command.";
    if (!enabled && $("command-note").textContent === "") {
      $("command-note").textContent = "The server has no API token configured, so it refuses every command.";
    }
    $("dark-cancel").hidden = !active;
    $("dark-cancel").disabled = state.sending;
    const ended = Boolean(task) && !active && task.state !== "idle";
    $("resume-notice").hidden = !(paused && ended);
    $("dark-resume").hidden = !paused || ended; // the notice below has its own button
    $("dark-resume").disabled = state.sending;
    $("start-hint").textContent = down ? "The server cannot reach core, so it cannot start a session now." : DarkText.startHint(schedulerState(), task);
    if (library) {
      $("survey-note").textContent = "Leave a field empty to use the setting of the survey. The survey records in mode " + library.mode + " at gain " + library.gain + " with an exposure of " + Number(library.exposure_s.toFixed(1)) + " s.";
      const field = $("f-exposure");
      if (!state.exposureTouched && document.activeElement !== field) {
        field.value = String(Number(library.exposure_s.toFixed(1)));
      }
    }
    $("dark-note").textContent = active ? "A session is under way. This page reads it every 2 s." : "This page reads the library every 10 s.";
  }

  function render(library) {
    renderStatus(library);
    renderSets(library);
    renderChart(library);
    renderTask(library);
    renderControls();
  }

  // --- Reading ----------------------------------------------------------------------------------

  async function read() {
    let library;
    const mine = (state.readCount += 1);
    try {
      library = await api.get("dark");
    } catch (error) {
      state.failed = error;
      if (error.status === 503) {
        showError({ message: "The server cannot reach core, so it cannot show the dark library. Trying again." });
      } else if (error.status !== 401) {
        showError(error);
      }
      renderControls();
      return;
    }
    if (mine < state.applied) {
      return; // a newer answer arrived first
    }
    state.applied = mine;
    state.failed = null;
    showError(null);
    const before = state.previousTask;
    state.library = library;
    state.previousTask = library.task.state;
    if (before !== null && before !== library.task.state && library.task.state !== "queued") {
      $("command-note").textContent = ""; // the answer to the last command is old news now
    }
    render(library);
    if (before !== null && DarkText.isActive({ state: before }) && !DarkText.isActive(library.task)) {
      // The session ended, and the scheduler may have paused. Ask for its state now, not in 10 s.
      Status.load().catch(() => undefined);
    }
  }

  // --- Commands ---------------------------------------------------------------------------------

  function formValues() {
    return {
      exposure: $("f-exposure").value,
      frames: $("f-frames").value,
      biasFrames: $("f-bias").value,
      waitForCover: $("f-cover").checked,
      pauseAfter: $("f-pause").checked,
    };
  }

  const FIELD_IDS = { exposure: ["f-exposure", "e-exposure"], frames: ["f-frames", "e-frames"], biasFrames: ["f-bias", "e-bias"] };

  function showFieldErrors(errors) {
    let first = null;
    for (const [key, [inputId, errorId]] of Object.entries(FIELD_IDS)) {
      const message = errors[key] || "";
      $(errorId).hidden = message === "";
      $(errorId).textContent = message;
      $(inputId).setAttribute("aria-invalid", message === "" ? "false" : "true");
      if (message !== "" && first === null) {
        first = inputId;
      }
    }
    if (first !== null) {
      $("advanced").open = true;
      $(first).focus();
    }
  }

  /** Send a command. The commands need the token, and the answer says what changed. */
  async function command(path, body, busyText) {
    const note = $("command-note");
    if (!Token.has()) {
      note.textContent = "Enter the token first: the commands need it.";
      window.Seeing.openTokenPanel();
      return null;
    }
    state.sending = true;
    renderControls();
    note.textContent = busyText;
    try {
      const answer = await api.post(path, body);
      note.textContent = DarkText.sentence(answer.message) + (answer.accepted ? "" : " (not accepted)");
      return answer;
    } catch (error) {
      if (error.status === 409) {
        note.textContent = DarkText.sentence(error.message);
      } else if (error.status === 429) {
        note.textContent = "Too many requests. Wait " + (error.retryAfter || 60) + " s and try again.";
      } else if (error.status === 401) {
        note.textContent = "The token is not right.";
      } else if (error.status === 403) {
        note.textContent = error.message;
      } else if (error.status === 503) {
        note.textContent = "Core does not answer, so nothing changed.";
      } else if (error.status === 422) {
        note.textContent = "The server did not accept the values.";
        showFieldErrors(DarkText.serverErrors(error));
      } else {
        note.textContent = error.message;
      }
      return null;
    } finally {
      state.sending = false;
      await Promise.all([read(), Status.load().catch(() => undefined)]);
      renderControls();
    }
  }

  async function start(event) {
    event.preventDefault();
    const checked = DarkText.validate(formValues());
    showFieldErrors(checked.errors);
    if (!checked.ok) {
      $("command-note").textContent = "Fix the values first.";
      return;
    }
    await command("commands/dark", checked.body, "Starting the session.");
  }

  function cancel() {
    return command("mode", { mode: "paused" }, "Cancelling the session.").then((answer) => {
      if (answer) {
        $("command-note").textContent = "The session is cancelled, and the station stays paused. Press Resume when you want it to measure again.";
      }
    });
  }

  function resume() {
    return command("mode", { mode: "auto" }, "Resuming the scheduler.");
  }

  function buildControls() {
    $("dark-form").addEventListener("submit", start);
    $("dark-cancel").addEventListener("click", cancel);
    $("dark-resume").addEventListener("click", resume);
    $("notice-resume").addEventListener("click", resume);
    $("f-exposure").addEventListener("input", () => {
      state.exposureTouched = true;
    });
    window.addEventListener("seeing:status", renderControls);
    window.addEventListener("seeing:token", () => loop.now());
  }

  window.addEventListener("DOMContentLoaded", () => {
    window.Seeing.boot("dark");
    buildControls();
    loop = poller(read, () => DarkText.pollInterval(state.library ? state.library.task : null));
    loop.start();
  });
})();
