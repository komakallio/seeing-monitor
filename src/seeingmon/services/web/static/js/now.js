"use strict";

/*
 * The Now page: what the system is doing, the newest seeing, sky, and pointing records with a
 * sparkline of the last hours, Polaris on its orbit, the state of the system, the latest image,
 * and the latest events.
 *
 * The page polls the status every 2 seconds, and a card loads again only when a new record has
 * arrived. Every age ticks each second on the clock of the server, and it turns "late" and then
 * "stale" when the next record is overdue. The strip at the top shows that the page still hears
 * the station. A card that fails shows its own message while the others go on.
 */
(function () {
  const { h, $, clear, fmt, api, Status, Plot, poller, flagChip, explainReason, showImage, Ticker, watchAge } = window.Seeing;
  const SPARK_HOURS = 3;
  const POLL_MS = 2000;
  const plots = {};

  // How often a new record should arrive, in seconds. An age turns "late" at 1.5 times that and
  // "stale" at 3 times. A seeing window is dated at its start, and the windows come in bursts, so
  // the newest record is up to 80 s old between two bursts without anything being wrong.
  const EXPECTED = { seeing: 70, sky: 200, pointing: 200, health: 75, image: 200 };
  const ALWAYS = new Set(["health"]);
  const KIND_OF = { seeing_window: "seeing", sky_quality: "sky", pointing: "pointing", health: "health" };

  let activity = null;

  /**
   * How often a record of this kind should arrive now, as a function for `watchAge`. Only the `auto`
   * state measures, so while the scheduler is known to be in another state the ages tick without a
   * verdict, and the activity banner says why nothing new arrives. When the state is unknown (core
   * does not answer), the ages count as if the system measured. The health record comes in every state.
   */
  function expect(kind) {
    return () => (EXPECTED[kind] && (ALWAYS.has(kind) || !activity || activity.state === "auto") ? EXPECTED[kind] : 0);
  }

  // --- Helpers ----------------------------------------------------------------------------------

  function setBig(id, text, unit, missing) {
    const node = $(id);
    clear(node);
    node.classList.toggle("missing", Boolean(missing));
    node.append(text);
    if (unit && !missing) {
      node.append(h("span", { class: "unit", text: unit }));
    }
  }

  function setFacts(id, rows) {
    const list = clear($(id));
    for (const [name, value] of rows) {
      list.append(h("dt", { text: name }), h("dd", { text: value }));
    }
  }

  function setChips(id, flags) {
    const box = clear($(id));
    for (const flag of flags || []) {
      box.append(flagChip(flag));
    }
  }

  function why(record, field) {
    return record.quality && record.quality[field] ? record.quality[field] : "";
  }

  function noData(prefix, error) {
    const empty = error && error.code === "no_data";
    setBig(prefix + "-value", empty ? "No data yet" : "Not available", "", true);
    $(prefix + "-note").textContent = empty ? "The store holds no record of this kind yet." : error.message;
    setFacts(prefix + "-facts", []);
    setChips(prefix + "-chips", []);
    watchAge($(prefix + "-when"), null, 0);
    if (plots[prefix]) {
      plots[prefix].setData({ t0: 0, t1: 1, series: [] });
    }
  }

  function sparkRange(status) {
    const to = Date.parse(status.now);
    return { t0: to - SPARK_HOURS * 3600e3, t1: to };
  }

  async function spark(prefix, path, field, range, options) {
    if (!plots[prefix]) {
      plots[prefix] = new Plot($(prefix + "-plot"), Object.assign({ spark: true }, options));
    }
    const page = await api.get(path, {
      from: new Date(range.t0).toISOString(),
      step: "1m",
      fields: field + ",flags",
      limit: 500,
    });
    const points = page.items.map((item) => ({
      t: Date.parse(item.t_utc),
      v: item[field],
      mark: (item.flags || []).includes("cloud") ? "--warn" : null,
    }));
    plots[prefix].setData({ t0: range.t0, t1: range.t1, series: [{ name: options.label, color: "--series-1", points }] });
  }

  // --- Cards ------------------------------------------------------------------------------------

  async function loadSeeing(status) {
    let record;
    try {
      record = await api.get("seeing/latest");
    } catch (error) {
      noData("seeing", error);
      return;
    }
    const value = record.seeing_fwhm_arcsec;
    setBig("seeing-value", fmt.num(value, 2), "″ FWHM", value === null);
    $("seeing-note").textContent = value === null ? why(record, "seeing_fwhm_arcsec") || "No value in the newest window." : "";
    watchAge($("seeing-when"), Date.parse(record.t_utc), expect("seeing"));
    setChips("seeing-chips", record.flags);
    setFacts("seeing-facts", [
      ["Fried parameter r0", value === null ? fmt.dash : fmt.num(record.r0_cm, 1) + " cm"],
      ["Image motion rms", fmt.arcsec(record.image_motion_rms_x_arcsec, 2) + " x, " + fmt.arcsec(record.image_motion_rms_y_arcsec, 2) + " y"],
      ["Star width", fmt.arcsec(record.width_fwhm_arcsec, 2)],
      ["Scintillation index", fmt.num(record.scintillation_index, 4)],
      ["Valid frames", fmt.percent(record.valid_fraction, 1) + " of " + record.n_frames],
      ["Window", fmt.stamp(record.t_utc) + ", " + record.duration_s + " s"],
      ["Camera", record.readout_mode + ", " + record.exposure_us / 1000 + " ms, gain " + record.gain],
    ]);
    await spark("seeing", "seeing", "seeing_fwhm_arcsec", sparkRange(status), { label: "Seeing FWHM, last " + SPARK_HOURS + " h", unit: "″", digits: 2 });
  }

  async function loadSky(status) {
    let record;
    try {
      record = await api.get("sky/latest");
    } catch (error) {
      noData("sky", error);
      return;
    }
    const value = record.sky_mag_arcsec2;
    setBig("sky-value", fmt.num(value, 2), "mag/″²", value === null);
    $("sky-note").textContent = value === null ? why(record, "sky_mag_arcsec2") || "No value in the newest record." : "";
    watchAge($("sky-when"), Date.parse(record.t_utc), expect("sky"));
    setChips("sky-chips", record.flags);
    setFacts("sky-facts", [
      ["Cloud cover", fmt.percent(record.cloud_fraction, 0)],
      ["Transparency", fmt.num(record.transparency, 2)],
      ["Limiting magnitude", fmt.num(record.limiting_mag, 1)],
      ["Zero point", fmt.num(record.zero_point_mag, 2) + " mag"],
      ["Stars used", record.n_stars_used === null ? fmt.dash : String(record.n_stars_used)],
      ["Time", fmt.stamp(record.t_utc)],
    ]);
    await spark("sky", "sky", "sky_mag_arcsec2", sparkRange(status), { label: "Sky brightness, last " + SPARK_HOURS + " h", unit: " mag", digits: 2 });
  }

  async function loadPointing(status) {
    let record;
    try {
      record = await api.get("pointing/latest");
    } catch (error) {
      noData("pointing", error);
      return;
    }
    const value = record.offset_arcmin;
    setBig("pointing-value", fmt.num(value, 2), "′ off target", value === null);
    $("pointing-note").textContent = value === null ? why(record, "offset_arcmin") || "No solution in the newest record." : "";
    watchAge($("pointing-when"), Date.parse(record.t_utc), expect("pointing"));
    setChips("pointing-chips", record.flags);
    setFacts("pointing-facts", [
      ["Roll", fmt.num(record.roll_deg, 2) + "°"],
      ["Matched stars", record.n_matched === null ? fmt.dash : String(record.n_matched)],
      ["Solution rms", fmt.arcsec(record.solve_rms_arcsec, 1)],
      ["Focus FWHM", fmt.num(record.focus_fwhm_px, 2) + " px"],
      ["Plate scale", fmt.num(record.plate_scale_arcsec_px, 2) + "″/px in " + record.readout_mode],
      ["Solver", record.solver],
      ["Time", fmt.stamp(record.t_utc)],
    ]);
    await spark("pointing", "pointing", "offset_arcmin", sparkRange(status), { label: "Pointing offset, last " + SPARK_HOURS + " h", unit: "′", digits: 2 });
  }

  function loadSystem(status) {
    const box = clear($("system-chips"));
    const components = status.health.components || {};
    for (const name of Object.keys(components).sort()) {
      box.append(h("span", { class: "chip", text: name + ": " + components[name], dataset: { level: fmt.level(components[name]) } }));
    }
    const healthAt = status.data.health && status.data.health.t_utc ? Date.parse(status.data.health.t_utc) : null;
    watchAge($("system-when"), healthAt, expect("health"), "health ");
    const rows = [];
    const scheduler = status.scheduler;
    if (scheduler) {
      rows.push(["State", scheduler.state + (scheduler.degraded ? " (degraded)" : "")]);
      rows.push(["In this state since", fmt.stamp(scheduler.state_since)]);
      rows.push(["Cloud", scheduler.cloud ? "yes, " + fmt.percent(scheduler.cloud_fraction, 0) : "no"]);
      rows.push(["Twilight", scheduler.twilight ? "yes" : "no"]);
      rows.push(["Queued tasks", String(scheduler.queued_tasks)]);
      if (scheduler.sensor_temperature_c !== null) {
        rows.push(["Sensor", fmt.num(scheduler.sensor_temperature_c, 1) + " °C"]);
      }
    } else {
      rows.push(["Scheduler", "core does not answer"]);
    }
    rows.push(["Station", status.station_id || fmt.dash]);
    rows.push(["Core", status.core.reachable ? "reachable" : "not reachable"]);
    setFacts("system-facts", rows);
    const note = clear($("system-note"));
    note.append("Newest records: ");
    const kinds = Object.entries(status.data);
    kinds.forEach(([name, item], index) => {
      const age = h("span", { class: "age" });
      watchAge(age, item.t_utc ? Date.parse(item.t_utc) : null, expect(KIND_OF[name]));
      note.append(name.replace(/_/g, " ") + " ", age, index < kinds.length - 1 ? ", " : ".");
    });
  }

  let shownImage = null;

  async function loadImage() {
    const body = $("image-body");
    let item;
    try {
      item = await api.get("images/latest", { format: "json" });
    } catch (error) {
      clear(body).append(h("p", { class: "empty", text: error.code === "no_data" || error.status === 404 ? "No image yet." : error.message }));
      shownImage = null;
      watchAge($("image-when"), null, 0);
      return;
    }
    watchAge($("image-when"), item.t_utc_ns / 1e6, expect("image"));
    const key = item.preview_url + "|" + item.t_utc_ns;
    if (key === shownImage) {
      return;
    }
    clear(body);
    const image = h("img", { alt: "The newest " + item.kind + " image", loading: "lazy", width: 480, height: 327 });
    image.style.width = "100%";
    image.style.height = "auto";
    await showImage(image, item.preview_url);
    body.append(h("a", { href: "images.html" }, image));
    body.append(h("p", { class: "note", text: item.kind + ", " + fmt.stamp(item.t_utc) + " UTC, " + fmt.bytes(item.size_bytes) }));
    shownImage = key;
  }

  async function loadEvents() {
    const list = clear($("events-list"));
    let page;
    try {
      page = await api.get("events", { limit: 8, order: "desc" });
    } catch (error) {
      list.append(h("li", { class: "empty", text: error.message }));
      return;
    }
    if (page.items.length === 0) {
      list.append(h("li", { class: "empty", text: "No events in the last 24 hours." }));
      return;
    }
    for (const event of page.items) {
      list.append(
        h(
          "li",
          {},
          h("time", { datetime: event.t_utc, text: fmt.stamp(event.t_utc), title: event.t_utc }),
          h("span", { class: "chip level", text: event.level, dataset: { level: fmt.level(event.level) } }),
          h("span", { class: "msg" }, event.message, h("span", { class: "kind", text: event.kind }))
        )
      );
    }
  }

  // --- The live strip ---------------------------------------------------------------------------

  let lastOk = null;
  let failure = null;
  let failingSince = null;
  let coreDown = false;

  function pingLive() {
    const dot = $("live-dot");
    dot.classList.remove("ping");
    void dot.offsetWidth;
    dot.classList.add("ping");
  }

  function renderLive() {
    const live = $("live");
    const text = $("live-text");
    if (lastOk === null && failure === null) {
      live.dataset.level = "warn";
      text.textContent = "Connecting";
      return;
    }
    if (failure !== null) {
      const since = (performance.now() - (lastOk === null ? failingSince : lastOk)) / 1000;
      live.dataset.level = since > 15 ? "bad" : "warn";
      text.textContent = "No link to the station" + (lastOk === null ? "" : " for " + fmt.duration(since));
      return;
    }
    const since = (performance.now() - lastOk) / 1000;
    if (coreDown) {
      live.dataset.level = "bad";
      text.textContent = "The web server answers, but core does not";
    } else if (since > 8) {
      live.dataset.level = "warn";
      text.textContent = "Late: the last update was " + fmt.duration(since) + " ago";
    } else {
      live.dataset.level = "good";
      text.textContent = "Live, updated " + Math.round(since) + " s ago";
    }
    $("live-clock").textContent = new Date(Status.nowMs()).toISOString().slice(11, 19) + " UTC";
  }

  // --- What the system is doing -----------------------------------------------------------------

  const STATE_WORDS = { safe: "Waiting", auto: "Measuring", align: "Aligning", commission: "Commissioning", paused: "Paused" };
  const PURPOSE_LABELS = {
    fast: "Fast stream: the seeing windows",
    survey: "Survey exposure for the pointing and the sky",
    watch: "Brightness watch",
    align: "Alignment live view",
    commission: "Commissioning exposure",
  };

  function takeActivity(status) {
    const s = status.scheduler;
    if (!s) {
      activity = null;
      return;
    }
    const a = s.activity || null;
    const ms = (iso) => (iso ? Date.parse(iso) : null);
    const purpose = s.stream && s.stream.purpose;
    const waiting = s.state === "safe" || s.state === "paused";
    activity = {
      state: (a && a.state) || s.state,
      phase: a ? a.phase : null,
      label: (a && a.label) || (waiting ? (s.state === "paused" ? "Paused" : "Waiting") + (s.state_reason ? ": " + s.state_reason : "") : PURPOSE_LABELS[purpose] || STATE_WORDS[s.state] || s.state),
      sinceWord: a ? "In this step for " : "In this mode for ",
      since: ms(a ? a.since_utc : s.state_since),
      ends: ms(a && a.ends_utc),
      nextLabel: (a && a.next_label) || null,
      next: ms(a && a.next_utc),
      detail: (a && (a.detail || a.reason)) || (waiting ? "" : s.state_reason) || "",
      degraded: Boolean(s.degraded),
    };
  }

  /** Seconds as "m:ss", or "h:mm:ss" from an hour. */
  function span(seconds) {
    const total = Math.max(0, Math.round(seconds));
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const rest = String(total % 60).padStart(2, "0");
    return hours ? hours + ":" + String(minutes).padStart(2, "0") + ":" + rest : minutes + ":" + rest;
  }

  function renderActivity() {
    const box = $("activity");
    if (!activity) {
      box.hidden = true;
      return;
    }
    box.hidden = false;
    const now = Status.nowMs();
    const bad = activity.phase === "camera_fault";
    box.dataset.level = bad ? "bad" : activity.degraded || activity.state === "safe" || activity.state === "paused" ? "warn" : "good";
    $("act-mode").textContent = (STATE_WORDS[activity.state] || activity.state) + (activity.degraded ? ", degraded" : "");
    $("act-label").textContent = activity.label;
    $("act-since").textContent = activity.since === null ? "" : activity.sinceWord + span((now - activity.since) / 1000);
    const bar = $("act-bar");
    if (activity.since !== null && activity.ends !== null && activity.ends > activity.since) {
      bar.hidden = false;
      const fraction = Math.min(1, Math.max(0, (now - activity.since) / (activity.ends - activity.since)));
      $("act-fill").style.width = (fraction * 100).toFixed(1) + "%";
      $("act-left").textContent = activity.ends > now ? span((activity.ends - now) / 1000) + " left" : "ending now";
    } else {
      bar.hidden = true;
      $("act-left").textContent = "";
    }
    const next = clear($("act-next"));
    if (activity.nextLabel) {
      next.append(h("strong", { text: "Next: " }), activity.nextLabel);
      if (activity.next !== null) {
        next.append(", " + fmt.clock(new Date(activity.next).toISOString()) + " UTC, in " + span((activity.next - now) / 1000));
      }
    }
    $("act-detail").textContent = activity.detail || "";
  }

  // --- Polaris ----------------------------------------------------------------------------------

  let polaris = null;
  let polarisData = null;
  let profileModes = null;
  let poleFields = true;

  async function loadPolaris(status) {
    if (!profileModes) {
      const profile = await api.get("profile");
      profileModes = Object.fromEntries(profile.readout_modes.map((mode) => [mode.name, mode]));
    }
    const from = new Date(Status.nowMs() - 3 * 3600e3).toISOString();
    const base = "polaris_x_px,polaris_y_px,plate_scale_arcsec_px,readout_mode";
    let page;
    try {
      page = await api.get("pointing", { from, fields: poleFields ? base + ",pole_x_px,pole_y_px" : base, limit: 300 });
    } catch (error) {
      if (!poleFields || error.code !== "invalid_request") {
        throw error;
      }
      // A server of an older release: its pointing record has no pole, so the widget extrapolates.
      poleFields = false;
      page = await api.get("pointing", { from, fields: base, limit: 300 });
    }
    const rows = page.items
      .filter((item) => item.polaris_x_px !== null && item.polaris_x_px !== undefined && profileModes[item.readout_mode])
      .sort((a, b) => Date.parse(a.t_utc) - Date.parse(b.t_utc));
    if (rows.length === 0) {
      polarisData = null;
      polaris.update(null);
      watchAge($("polaris-when"), null, 0);
      $("polaris-note").textContent = "No solved pointing in the last 3 hours.";
      return;
    }
    const last = rows[rows.length - 1];
    const mode = profileModes[last.readout_mode];
    const pole = (value) => (value === null || value === undefined ? null : value);
    polarisData = {
      frame: { width: mode.width_px, height: mode.height_px },
      scale: last.plate_scale_arcsec_px,
      fast: Boolean(status.scheduler && status.scheduler.stream && status.scheduler.stream.purpose === "fast"),
      points: rows.map((row) => ({
        t: Date.parse(row.t_utc),
        x: row.polaris_x_px,
        y: row.polaris_y_px,
        px: pole(row.pole_x_px),
        py: pole(row.pole_y_px),
      })),
    };
    polaris.update(polarisData);
    tickPolaris();
    watchAge($("polaris-when"), Date.parse(last.t_utc), expect("pointing"));
  }

  function tickPolaris() {
    if (!polaris || !polarisData) {
      return;
    }
    const at = polaris.tick(Status.nowMs());
    if (!at) {
      return;
    }
    const orbit = polaris.orbit;
    const perMinute = orbit ? orbit.radius * window.Polaris.SIDEREAL_RAD_PER_S * 60 : null;
    const old = at.ageS > 600 ? " The marker rests on a solution that is " + fmt.duration(at.ageS) + " old." : "";
    $("polaris-note").textContent =
      "Polaris circles the pole at 15° an hour" +
      (perMinute === null ? "" : ", " + fmt.num(perMinute, 1) + " px a minute in this frame") +
      ". The green square is the ROI of the fast stream." +
      old;
  }

  // --- The banner and the refresh ---------------------------------------------------------------

  function showBanner(status) {
    const banner = $("banner");
    const health = status.health;
    if (health.status === "healthy" && !status.demo) {
      banner.hidden = true;
      return;
    }
    clear(banner);
    banner.hidden = false;
    banner.dataset.level = health.status === "healthy" ? "warn" : fmt.level(health.status);
    if (status.demo) {
      banner.append(h("p", { text: "This is the demo. The data are synthetic, and the station does not exist." }));
    }
    if (health.status !== "healthy") {
      const verdict = health.status === "failed" ? "The system has failed. " : "The system is " + health.status + ". ";
      banner.append(h("strong", { text: verdict }), health.reasons.map(explainReason).join(" "));
    }
  }

  // What each card showed last: a card loads again only when its key changes. The key of a card
  // with a sparkline also holds the minute, so that the right edge of the plot keeps moving.
  const seen = {};
  const inFlight = new Set();

  function reload(kind, key, job) {
    if (seen[kind] === key || inFlight.has(kind)) {
      return;
    }
    seen[kind] = key;
    inFlight.add(kind);
    job()
      .catch(() => {
        seen[kind] = undefined;
      })
      .finally(() => inFlight.delete(kind));
  }

  const stampOf = (entry) => (entry && entry.t_utc ? entry.t_utc : "none");

  async function refresh() {
    let status;
    try {
      status = await Status.load();
    } catch (error) {
      failure = error;
      if (failingSince === null) {
        failingSince = performance.now();
      }
      const banner = $("banner");
      banner.hidden = false;
      banner.dataset.level = "bad";
      banner.textContent = error.status === 401 ? "The server asks for the token. Enter it above." : "Cannot reach the server: " + error.message;
      renderLive();
      return;
    }
    failure = null;
    failingSince = null;
    lastOk = performance.now();
    coreDown = !status.core.reachable;
    pingLive();
    showBanner(status);
    takeActivity(status);
    renderLive();
    renderActivity();
    loadSystem(status);
    const nowMs = Status.nowMs();
    const minute = Math.floor(nowMs / 60000);
    reload("seeing", stampOf(status.data.seeing_window) + "|" + minute, () => loadSeeing(status));
    reload("sky", stampOf(status.data.sky_quality) + "|" + minute, () => loadSky(status));
    reload("pointing", stampOf(status.data.pointing) + "|" + minute, () => loadPointing(status));
    reload("events", stampOf(status.data.event) + "|" + minute, loadEvents);
    reload("image", String(Math.floor(nowMs / 30000)), loadImage);
    const fast = Boolean(status.scheduler && status.scheduler.stream && status.scheduler.stream.purpose === "fast");
    reload("polaris", stampOf(status.data.pointing) + "|" + Math.floor(nowMs / 20000) + "|" + fast, () => loadPolaris(status));
  }

  window.addEventListener("DOMContentLoaded", () => {
    window.Seeing.boot("now", { statusEvery: false });
    polaris = window.Polaris.create($("polaris-plot"));
    Ticker.add(renderLive);
    Ticker.add(renderActivity);
    Ticker.add(tickPolaris);
    renderLive();
    const loop = poller(refresh, () => POLL_MS);
    window.addEventListener("seeing:token", () => {
      for (const key of Object.keys(seen)) {
        delete seen[key];
      }
      loop.now();
    });
    loop.start();
  });
})();
