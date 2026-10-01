"use strict";

/*
 * The Align page: the live view of the star field with the target and the solved position drawn
 * over it, the offset and the roll, a focus bar, a histogram of the pixel values, and a warning for
 * saturation. The page starts and stops the alignment (the commands need the token).
 *
 * `LiveLink` (live.js) keeps the connection to the server: a WebSocket with reconnects, and the
 * polling of the newest frame as a fallback. This file draws what the link delivers. The page
 * closes the connection while its tab stays hidden.
 */
(function () {
  const { h, $, fmt, api, ApiError, Status, Token, poller, LiveLink } = window.Seeing;

  const HIDDEN_CLOSE_MS = 20000;
  const FAMILY = "system-ui, sans-serif";

  const view = { canvas: null, ctx: null, hist: null, overlays: true, lastState: null, picture: null, bitmapOk: typeof createImageBitmap === "function" };
  const align = { active: false, known: false };
  let live = null;
  let hiddenTimer = null;

  function token(name, fallback) {
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
  }

  function showError(error) {
    const banner = $("align-error");
    banner.hidden = !error;
    banner.dataset.level = "bad";
    banner.textContent = error ? (error.status === 401 ? "The server asks for the token. Enter it above." : error.message) : "";
  }

  // --- Link state -------------------------------------------------------------------------------

  const LINKS = {
    idle: ["Live view: off", "warn", ""],
    connecting: ["Live view: connecting", "warn", "Connecting to the live view."],
    live: ["Live view: receiving", "good", ""],
    waiting: ["Live view: connected, no frame", "warn", ""],
    stalled: ["Live view: stalled", "bad", "No frame arrived for a while."],
    reconnecting: ["Live view: reconnecting", "warn", "The connection broke. Trying again."],
    polling: ["Live view: polling", "warn", ""],
    denied: ["Live view: needs the token", "bad", "The server asks for the token on every read. Enter it above."],
    busy: ["Live view: server is busy", "warn", "The server shows the live view to as many people as it allows. Trying again later."],
    unavailable: ["Live view: core does not answer", "bad", "The server cannot reach core."],
  };

  function onLink(name) {
    const [text, level] = LINKS[name];
    const pill = $("link-state");
    pill.textContent = text;
    pill.dataset.level = level;
    updateCover();
  }

  function updateCover() {
    const cover = $("cover");
    const name = live ? live.name : "idle";
    let text = "";
    if (!(live && live.fresh())) {
      if (name === "live" || name === "waiting" || name === "polling") {
        text = align.known && !align.active ? "Alignment is not running. Start it to see the live view." : "Waiting for the next frame.";
      } else {
        text = LINKS[name][2];
      }
    }
    cover.textContent = text;
    cover.hidden = text === "";
  }

  function staleMs() {
    const status = Status.current;
    return (status ? status.ui.alignment_stall_s : 5) * 1000;
  }

  // --- Drawing ----------------------------------------------------------------------------------

  function setupCanvases() {
    view.canvas = $("view");
    view.ctx = view.canvas.getContext("2d");
    view.hist = $("hist");
    $("overlays").addEventListener("change", (event) => {
      view.overlays = event.target.checked;
      redraw();
    });
    window.addEventListener("seeing:theme", () => {
      redraw();
      drawHistogram(view.lastState);
    });
    if (typeof ResizeObserver === "function") {
      new ResizeObserver(() => drawHistogram(view.lastState)).observe(view.hist.parentElement);
      new ResizeObserver(() => requestAnimationFrame(redraw)).observe(view.canvas);
    }
    clearView();
  }

  function clearView() {
    const { canvas, ctx } = view;
    ctx.fillStyle = "#05070a";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  async function decode(blob) {
    if (view.bitmapOk) {
      return createImageBitmap(blob);
    }
    const url = URL.createObjectURL(blob);
    try {
      const image = new Image();
      image.src = url;
      await image.decode();
      return image;
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  async function showFrame(blob, state) {
    const picture = await decode(blob);
    const width = picture.width;
    const height = picture.height;
    if (view.canvas.width !== width || view.canvas.height !== height) {
      view.canvas.width = width;
      view.canvas.height = height;
    }
    view.ctx.drawImage(picture, 0, 0);
    if (view.picture && typeof view.picture.close === "function") {
      view.picture.close();
    }
    view.picture = picture;
    view.lastState = state;
    drawOverlays(state);
    renderState(state);
    updateCover();
  }

  function redraw() {
    if (view.picture) {
      view.ctx.drawImage(view.picture, 0, 0);
      drawOverlays(view.lastState);
    }
  }

  function drawOverlays(state) {
    if (!view.overlays || !state || !state.frame || !state.target) {
      return;
    }
    const { canvas, ctx } = view;
    const sx = canvas.width / state.frame.width_px;
    const sy = canvas.height / state.frame.height_px;
    // Sizes are in CSS pixels on the screen, so the marks keep their size when the image scales.
    const shown = canvas.getBoundingClientRect().width;
    const unit = shown > 0 ? canvas.width / shown : 1;
    const target = [state.target.x_px * sx, state.target.y_px * sy];
    const green = token("--overlay-target", "#0a7a3d");
    const red = token("--overlay-solved", "#b3261e");
    ctx.save();
    ctx.lineWidth = Math.max(1.5, 1.6 * unit);
    ctx.font = "600 " + Math.round(12 * unit) + "px " + FAMILY;
    ctx.textBaseline = "top";
    // The target: a cross with a gap in the middle, and a ring.
    ctx.strokeStyle = green;
    ctx.fillStyle = green;
    const gap = 6 * unit;
    const arm = 28 * unit;
    ctx.beginPath();
    ctx.moveTo(target[0] - arm, target[1]);
    ctx.lineTo(target[0] - gap, target[1]);
    ctx.moveTo(target[0] + gap, target[1]);
    ctx.lineTo(target[0] + arm, target[1]);
    ctx.moveTo(target[0], target[1] - arm);
    ctx.lineTo(target[0], target[1] - gap);
    ctx.moveTo(target[0], target[1] + gap);
    ctx.lineTo(target[0], target[1] + arm);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(target[0], target[1], 16 * unit, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillText("target", target[0] + 20 * unit, target[1] - 26 * unit);
    if (state.solved) {
      const solved = [state.solved.x_px * sx, state.solved.y_px * sy];
      ctx.strokeStyle = red;
      ctx.fillStyle = red;
      const distance = Math.hypot(solved[0] - target[0], solved[1] - target[1]);
      if (distance > 4 * unit) {
        const angle = Math.atan2(solved[1] - target[1], solved[0] - target[0]);
        const start = [target[0] + Math.cos(angle) * 16 * unit, target[1] + Math.sin(angle) * 16 * unit];
        const end = [solved[0] - Math.cos(angle) * 10 * unit, solved[1] - Math.sin(angle) * 10 * unit];
        if (Math.hypot(end[0] - start[0], end[1] - start[1]) > 2) {
          ctx.beginPath();
          ctx.moveTo(start[0], start[1]);
          ctx.lineTo(end[0], end[1]);
          ctx.stroke();
          const head = 7 * unit;
          ctx.beginPath();
          ctx.moveTo(end[0], end[1]);
          ctx.lineTo(end[0] - Math.cos(angle - 0.45) * head, end[1] - Math.sin(angle - 0.45) * head);
          ctx.lineTo(end[0] - Math.cos(angle + 0.45) * head, end[1] - Math.sin(angle + 0.45) * head);
          ctx.closePath();
          ctx.fill();
        }
      }
      ctx.beginPath();
      ctx.arc(solved[0], solved[1], 10 * unit, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(solved[0], solved[1], 2 * unit, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillText("Polaris", solved[0] + 14 * unit, solved[1] + 8 * unit);
    }
    ctx.restore();
  }

  function drawHistogram(state) {
    const canvas = view.hist;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const histogram = state && state.histogram;
    if (!histogram || histogram.counts.length === 0) {
      ctx.fillStyle = token("--muted", "#566176");
      ctx.font = "12px " + FAMILY;
      ctx.textAlign = "center";
      ctx.fillText("No histogram yet", width / 2, height / 2);
      return;
    }
    const counts = histogram.counts;
    const top = Math.log10(Math.max(...counts) + 1) || 1;
    const bar = width / counts.length;
    const saturated = state.saturation && state.saturation.warning;
    ctx.strokeStyle = token("--grid", "#e1e6ee");
    ctx.lineWidth = 1;
    for (let decade = 0; decade <= top; decade += 1) {
      const y = Math.round(height - (decade / top) * (height - 4)) - 0.5;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    counts.forEach((count, index) => {
      const fraction = Math.log10(count + 1) / top;
      const barHeight = Math.max(count > 0 ? 1 : 0, fraction * (height - 4));
      const last = index >= counts.length - 2;
      ctx.fillStyle = saturated && last ? token("--bad", "#b3261e") : token("--accent", "#1f55c4");
      ctx.fillRect(index * bar + 1, height - barHeight, Math.max(1, bar - 2), barHeight);
    });
    $("hist-min").textContent = String(Math.round(histogram.min_dn));
    $("hist-max").textContent = String(Math.round(histogram.max_dn));
    canvas.setAttribute("aria-label", "Histogram of the pixel values, " + counts.length + " bins from " + Math.round(histogram.min_dn) + " to " + Math.round(histogram.max_dn) + " DN, log scale.");
  }

  // --- Readouts ---------------------------------------------------------------------------------

  function setFacts(id, rows) {
    const list = $(id);
    list.replaceChildren();
    for (const [name, value] of rows) {
      list.append(h("dt", { text: name }), h("dd", { text: value }));
    }
  }

  function renderState(state) {
    const offset = state.offset;
    const solved = state.solved;
    const frame = state.frame;
    if (offset) {
      setFacts("offset-facts", [
        ["Horizontal (x)", fmt.signed(offset.dx_px, 1) + " px, " + fmt.signed(offset.dx_arcsec, 1) + "″"],
        ["Vertical (y)", fmt.signed(offset.dy_px, 1) + " px, " + fmt.signed(offset.dy_arcsec, 1) + "″"],
        ["Distance", fmt.num(offset.distance_px, 1) + " px, " + fmt.num(offset.distance_arcsec, 1) + "″"],
        ["Roll", fmt.signed(offset.roll_deg, 2) + "°"],
      ]);
      const limit = 2;
      const roll = Math.max(-limit, Math.min(limit, offset.roll_deg || 0));
      $("roll-mark").style.left = "calc(" + ((roll + limit) / (2 * limit)) * 100 + "% - 1px)";
      $("roll-min").textContent = "−" + limit + "°";
      $("roll-max").textContent = "+" + limit + "°";
      $("roll-note").textContent = Math.abs(offset.roll_deg || 0) < 0.1 ? "The roll is within 0.1 degrees of the target." : "Rotate the camera by " + fmt.signed(-(offset.roll_deg || 0), 2) + " degrees to match the target roll.";
    } else {
      setFacts("offset-facts", [["Offset", "no solution yet"]]);
    }
    const rows = [];
    if (solved) {
      rows.push(["Matched stars", String(solved.n_matched)]);
      rows.push(["Solution rms", fmt.arcsec(solved.rms_arcsec, 1)]);
      rows.push(["Solution age", fmt.num(solved.age_s, 1) + " s"]);
    }
    if (frame) {
      rows.push(["Frame", "#" + frame.seq + ", " + frame.width_px + " × " + frame.height_px + " px"]);
      rows.push(["Readout mode", frame.readout_mode || fmt.dash]);
      rows.push(["Exposure, gain", (frame.exposure_s === null ? fmt.dash : frame.exposure_s + " s") + ", " + (frame.gain === null ? fmt.dash : frame.gain)]);
      rows.push(["Plate scale", fmt.num(frame.plate_scale_arcsec_px, 2) + "″/px"]);
    }
    if (state.t_utc) {
      rows.push(["Frame time", fmt.clock(state.t_utc) + " UTC"]);
    }
    if (state.saturation) {
      rows.push(["Saturated pixels", fmt.num(state.saturation.fraction * 100, 3) + " %"]);
    }
    setFacts("solution-facts", rows);
    renderFocus(state.focus);
    renderSaturation(state.saturation);
    drawHistogram(state);
    $("frame-info").textContent = frame ? "Frame " + frame.seq + " at " + fmt.clock(state.t_utc) + " UTC" : "";
  }

  function renderFocus(focus) {
    const fill = $("focus-fill");
    if (!focus || focus.fwhm_px === null) {
      fill.style.width = "0";
      $("focus-note").textContent = "No focus measure yet.";
      return;
    }
    const ratio = focus.best_fwhm_px ? Math.max(0, Math.min(1, focus.best_fwhm_px / focus.fwhm_px)) : Math.max(0, Math.min(1, 3 / focus.fwhm_px));
    fill.style.width = Math.round(ratio * 100) + "%";
    fill.dataset.level = ratio >= 0.9 ? "good" : ratio >= 0.7 ? "warn" : "bad";
    $("focus-note").textContent =
      "Star width " + fmt.num(focus.fwhm_px, 2) + " px" + (focus.best_fwhm_px ? " (best " + fmt.num(focus.best_fwhm_px, 2) + " px)" : "") + (focus.n_stars ? ", " + focus.n_stars + " stars" : "") + ". A narrower star is better.";
  }

  function renderSaturation(saturation) {
    const banner = $("saturation-banner");
    if (saturation && saturation.warning) {
      banner.hidden = false;
      banner.textContent = "Saturation: " + fmt.num(saturation.fraction * 100, 2) + " % of the pixels are at full scale. Polaris reads wrong while it saturates. Shorten the exposure or lower the gain.";
    } else {
      banner.hidden = true;
    }
  }

  // --- The link ---------------------------------------------------------------------------------

  function socketUrl() {
    const scheme = window.location.protocol === "https:" ? "wss" : "ws";
    return scheme + "://" + window.location.host + api.url("alignment/stream");
  }

  function pollInterval() {
    const status = Status.current;
    const fps = status ? status.ui.alignment_max_fps : 2;
    return Math.max(500, 1000 / fps);
  }

  /** Read the newest frame and its state, for the polling fallback. */
  async function pollFrame(after) {
    try {
      const response = await api.raw("alignment/frame", { after }, { accept: "image/jpeg" });
      if (response.status === 204) {
        return null;
      }
      const seq = Number(response.headers.get("X-Frame-Seq")) || after;
      const blob = await response.blob();
      const state = await api.get("alignment/state");
      return { blob, state, seq };
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        return { denied: true };
      }
      throw error;
    }
  }

  function createLink() {
    return new LiveLink({
      url: socketUrl,
      tokenRequired: () => Boolean(Status.current && Status.current.ui.token_required_for_reads),
      token: () => Token.get(),
      isActive: () => align.active,
      staleMs,
      pollIntervalMs: pollInterval,
      pollFrame,
      onFrame: (blob, state) => {
        showFrame(blob, state).catch(() => undefined);
      },
      onLink,
    });
  }

  // --- State and commands -----------------------------------------------------------------------

  function commandsEnabled() {
    const status = Status.current;
    return !status || status.ui.commands_enabled;
  }

  async function refreshState() {
    try {
      const state = await api.get("alignment/state");
      align.active = state.active;
      align.known = true;
      showError(null);
      const pill = $("run-state");
      pill.textContent = state.active ? "Alignment: running" : "Alignment: not running";
      pill.dataset.level = state.active ? "good" : "warn";
      $("start").disabled = !commandsEnabled();
      $("stop").disabled = !commandsEnabled() || !state.active;
      $("align-note").textContent = state.active ? "Alignment is running." : "Alignment is not running.";
      if (state.active && state.frame && live && live.name !== "live" && (!view.lastState || state.frame.seq !== view.lastState.frame.seq)) {
        renderState(state);
      }
      updateCover();
    } catch (error) {
      if (error.status === 503) {
        $("run-state").textContent = "Alignment: core does not answer";
        $("run-state").dataset.level = "bad";
      }
      if (error.status !== 401) {
        showError(error);
      }
    }
  }

  async function command(path, busyText) {
    const note = $("command-note");
    if (!Token.has()) {
      note.textContent = "Enter the token first: the commands need it.";
      window.Seeing.openTokenPanel();
      return;
    }
    note.textContent = busyText;
    try {
      const answer = await api.post(path, {});
      note.textContent = answer.message + (answer.accepted ? "" : " (not accepted)");
      await refreshState();
    } catch (error) {
      if (error.status === 429) {
        note.textContent = "Too many requests. Wait " + (error.retryAfter || 60) + " s and try again.";
      } else if (error.status === 401) {
        note.textContent = "The token is not right.";
      } else if (error.status === 403) {
        note.textContent = error.message;
      } else if (error.status === 503) {
        note.textContent = "Core does not answer, so nothing changed.";
      } else {
        note.textContent = error.message;
      }
    }
  }

  function buildControls() {
    $("start").addEventListener("click", () => command("alignment/start", "Starting the alignment."));
    $("stop").addEventListener("click", () => command("alignment/stop", "Stopping the alignment."));
    window.addEventListener("seeing:status", () => {
      const enabled = commandsEnabled();
      $("start").disabled = !enabled;
      $("start").title = enabled ? "" : "The server has no token hash, so it refuses every command.";
      if (!enabled) {
        $("command-note").textContent = "The server has no API token configured, so it refuses every command.";
      }
    });
    window.addEventListener("seeing:token", () => {
      if (live && (live.name === "denied" || live.name === "reconnecting")) {
        live.restart();
      }
    });
    document.addEventListener("visibilitychange", () => {
      clearTimeout(hiddenTimer);
      if (document.hidden) {
        hiddenTimer = setTimeout(() => live.stop(), HIDDEN_CLOSE_MS);
      } else if (live.stopped) {
        live.start();
      }
    });
    setInterval(() => {
      live.tick();
      updateCover();
    }, 1000);
  }

  window.addEventListener("DOMContentLoaded", async () => {
    window.Seeing.boot("align");
    live = createLink();
    setupCanvases();
    buildControls();
    drawHistogram(null);
    try {
      await Status.load();
    } catch (error) {
      showError(error);
    }
    poller(refreshState, () => (align.active ? 3000 : 2000)).start();
    live.start();
  });
})();
