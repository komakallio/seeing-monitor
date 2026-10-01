"use strict";

/*
 * The Now page: the newest seeing, sky, and pointing records with a sparkline of the last hours,
 * the state of the system, the latest image, and the latest events. One refresh loads the status
 * and then every card, and a card that fails shows its own message while the others go on.
 */
(function () {
  const { h, $, clear, fmt, api, Status, Plot, poller, flagChip, explainReason, showImage } = window.Seeing;
  const SPARK_HOURS = 3;
  const plots = {};

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
    $(prefix + "-when").textContent = "";
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
    $("seeing-when").textContent = fmt.age(status.data.seeing_window.age_s);
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
    $("sky-when").textContent = fmt.age(status.data.sky_quality.age_s);
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
    $("pointing-when").textContent = fmt.age(status.data.pointing.age_s);
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
    $("system-when").textContent = status.health.age_s === null ? "" : "health " + fmt.age(status.health.age_s);
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
    const ages = Object.entries(status.data)
      .map(([name, item]) => name.replace(/_/g, " ") + " " + fmt.age(item.age_s))
      .join(", ");
    $("system-note").textContent = "Newest records: " + ages + ".";
  }

  async function loadImage() {
    const body = clear($("image-body"));
    let item;
    try {
      item = await api.get("images/latest", { format: "json" });
    } catch (error) {
      body.append(h("p", { class: "empty", text: error.code === "no_data" || error.status === 404 ? "No image yet." : error.message }));
      $("image-when").textContent = "";
      return;
    }
    const image = h("img", { alt: "The newest " + item.kind + " image", loading: "lazy", width: 480, height: 327 });
    image.style.width = "100%";
    image.style.height = "auto";
    await showImage(image, item.preview_url);
    body.append(h("a", { href: "images.html" }, image));
    body.append(h("p", { class: "note", text: item.kind + ", " + fmt.stamp(item.t_utc) + " UTC, " + fmt.bytes(item.size_bytes) }));
    $("image-when").textContent = fmt.age((Date.parse(Status.current.now) - item.t_utc_ns / 1e6) / 1000);
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

  async function refresh() {
    let status;
    try {
      status = await Status.load();
    } catch (error) {
      const banner = $("banner");
      banner.hidden = false;
      banner.dataset.level = "bad";
      banner.textContent = error.status === 401 ? "The server asks for the token. Enter it above." : "Cannot reach the server: " + error.message;
      $("updated").textContent = "No update";
      return;
    }
    showBanner(status);
    loadSystem(status);
    await Promise.all([loadSeeing(status), loadSky(status), loadPointing(status), loadImage(), loadEvents()].map((p) => p.catch(() => undefined)));
    $("updated").textContent = "Updated " + fmt.clock(status.now) + " UTC, every " + status.ui.refresh_s + " s";
  }

  window.addEventListener("DOMContentLoaded", () => {
    window.Seeing.boot("now", { statusEvery: false });
    const loop = poller(refresh, () => (Status.current ? Status.current.ui.refresh_s * 1000 : 10000));
    window.addEventListener("seeing:token", () => loop.now());
    loop.start();
  });
})();
