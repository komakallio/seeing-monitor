"use strict";

/*
 * The History page: plots of seeing, r0, sky brightness, cloud cover, and pointing over a range of
 * time, and the events of the same range. The range ends at the time of the server, not at the
 * time of this browser. A long range reads coarser buckets (the API averages the numbers of a
 * bucket and joins its flags), so a plot never holds more than a few hundred points.
 */
(function () {
  const { h, $, clear, fmt, api, Status, Plot, flagChip } = window.Seeing;

  const HOUR = 3600e3;
  const RANGES = [
    { id: "6h", label: "6 h", ms: 6 * HOUR, step: "raw" },
    { id: "24h", label: "24 h", ms: 24 * HOUR, step: "10m" },
    { id: "3d", label: "3 days", ms: 72 * HOUR, step: "10m" },
    { id: "7d", label: "7 days", ms: 168 * HOUR, step: "1h" },
    { id: "30d", label: "30 days", ms: 720 * HOUR, step: "1h" },
  ];
  const STEP_NOTES = { raw: "one point for each window", "1m": "one point per minute", "10m": "one point per 10 minutes", "1h": "one point per hour" };
  const MAX_PAGES = 12;
  // A point gets the mark of its first flag in MARK_ORDER. `noisy` and `vibration` both say that
  // the seeing can read high, and `daylight` and `twilight` both say that the Sun was up or near.
  const MARKS = { cloud: "--series-4", vibration: "--series-2", noisy: "--series-2", daylight: "--series-3", twilight: "--series-3" };
  const MARK_TEXT = { cloud: "clouds", vibration: "vibration", noisy: "noisy", daylight: "daylight", twilight: "twilight" };
  const MARK_ORDER = ["cloud", "vibration", "noisy", "daylight", "twilight"];

  const state = { range: RANGES[1], step: "auto", level: "info", loadId: 0, eventCursor: null, eventCount: 0 };
  const plots = {};

  function plotFor(key, options) {
    if (!plots[key]) {
      plots[key] = new Plot($("plot-" + key), options);
    }
    return plots[key];
  }

  function iso(ms) {
    return new Date(ms).toISOString();
  }

  function stepFor(range) {
    return state.step === "auto" ? range.step : state.step;
  }

  /** Read every page of a history route. Returns the items and whether pages remain. */
  async function readAll(path, fields, span, step) {
    const items = [];
    let cursor = null;
    for (let page = 0; page < MAX_PAGES; page += 1) {
      const data = await api.get(path, { from: iso(span.t0), to: iso(span.t1), step, fields, cursor });
      items.push(...data.items);
      cursor = data.next_cursor;
      if (!cursor) {
        return { items, more: false };
      }
    }
    return { items, more: true };
  }

  function markOf(flags) {
    for (const flag of MARK_ORDER) {
      if ((flags || []).includes(flag)) {
        return MARKS[flag];
      }
    }
    return null;
  }

  function notes(item) {
    const parts = [];
    if (item.n_samples && item.n_samples > 1) {
      parts.push(item.n_samples + " windows");
    }
    if ((item.flags || []).length) {
      parts.push(item.flags.join(", "));
    }
    return parts.join("; ");
  }

  function points(items, field, withMarks) {
    return items.map((item) => ({
      t: Date.parse(item.t_utc),
      v: item[field],
      note: notes(item),
      mark: withMarks ? markOf(item.flags) : null,
    }));
  }

  function count(id, items, field, more) {
    const values = items.filter((item) => item[field] !== null && item[field] !== undefined).length;
    $(id).textContent = values + " values" + (more ? ", more than shown" : "");
  }

  function legend(id, entries) {
    const box = clear($(id));
    for (const [text, color] of entries) {
      const item = h("span", { text });
      item.style.setProperty("--swatch", "var(" + color + ")");
      box.append(item);
    }
  }

  async function drawSeeing(span, step) {
    const seeing = await readAll("seeing", "seeing_fwhm_arcsec,r0_cm,flags", span, step);
    plotFor("seeing", { label: "Seeing FWHM", unit: "″", digits: 2, zeroBased: false }).setData({
      t0: span.t0,
      t1: span.t1,
      series: [{ name: "FWHM", color: "--series-1", points: points(seeing.items, "seeing_fwhm_arcsec", true) }],
    });
    plotFor("r0", { label: "Fried parameter r0", unit: " cm", digits: 1 }).setData({
      t0: span.t0,
      t1: span.t1,
      series: [{ name: "r0", color: "--series-3", points: points(seeing.items, "r0_cm", false) }],
    });
    count("seeing-count", seeing.items, "seeing_fwhm_arcsec", seeing.more);
    count("r0-count", seeing.items, "r0_cm", seeing.more);
    const used = new Set();
    for (const item of seeing.items) {
      for (const flag of item.flags || []) {
        if (MARKS[flag]) {
          used.add(flag);
        }
      }
    }
    legend("legend-seeing", [["seeing FWHM", "--series-1"], ...[...used].map((flag) => [MARK_TEXT[flag], MARKS[flag]])]);
    return seeing.items.length;
  }

  async function drawSky(span, step) {
    const sky = await readAll("sky", "sky_mag_arcsec2,cloud_fraction,transparency,flags", span, step);
    plotFor("sky", { label: "Sky brightness", unit: " mag/″²", digits: 2 }).setData({
      t0: span.t0,
      t1: span.t1,
      series: [{ name: "Sky", color: "--series-1", points: points(sky.items, "sky_mag_arcsec2", true) }],
    });
    plotFor("cloud", { label: "Cloud cover and transparency", digits: 2, yMin: 0, yMax: 1 }).setData({
      t0: span.t0,
      t1: span.t1,
      series: [
        { name: "Cloud cover", color: "--series-2", points: points(sky.items, "cloud_fraction", false) },
        { name: "Transparency", color: "--series-3", points: points(sky.items, "transparency", false) },
      ],
    });
    count("sky-count", sky.items, "sky_mag_arcsec2", sky.more);
    count("cloud-count", sky.items, "cloud_fraction", sky.more);
    legend("legend-cloud", [["cloud cover (0 clear, 1 overcast)", "--series-2"], ["transparency", "--series-3"]]);
  }

  async function drawPointing(span, step) {
    const pointing = await readAll("pointing", "offset_arcmin,flags", span, step);
    plotFor("pointing", { label: "Pointing offset", unit: "′", digits: 2, zeroBased: true }).setData({
      t0: span.t0,
      t1: span.t1,
      series: [{ name: "Offset", color: "--series-4", points: points(pointing.items, "offset_arcmin", false) }],
    });
    count("pointing-count", pointing.items, "offset_arcmin", pointing.more);
  }

  // --- Events -----------------------------------------------------------------------------------

  function eventItem(event) {
    return h(
      "li",
      {},
      h("time", { datetime: event.t_utc, text: fmt.stamp(event.t_utc), title: event.t_utc }),
      h("span", { class: "chip level", text: event.level, dataset: { level: fmt.level(event.level) } }),
      h("span", { class: "msg" }, event.message, h("span", { class: "kind", text: event.kind }))
    );
  }

  async function loadEvents(span, more) {
    const list = $("events-list");
    if (!more) {
      clear(list);
      state.eventCursor = null;
      state.eventCount = 0;
    }
    const data = await api.get("events", {
      from: iso(span.t0),
      to: iso(span.t1),
      level: state.level,
      order: "desc",
      limit: 50,
      cursor: more ? state.eventCursor : null,
    });
    for (const event of data.items) {
      list.append(eventItem(event));
    }
    state.eventCount += data.items.length;
    state.eventCursor = data.next_cursor;
    $("events-more").hidden = !data.next_cursor;
    $("events-count").textContent = state.eventCount + " shown";
    if (state.eventCount === 0) {
      list.append(h("li", { class: "empty", text: "No events in this range." }));
    }
  }

  // --- Loading ----------------------------------------------------------------------------------

  function showError(error) {
    const banner = $("history-error");
    banner.hidden = !error;
    banner.dataset.level = "bad";
    banner.textContent = error ? (error.status === 401 ? "The server asks for the token. Enter it above." : error.message) : "";
  }

  async function load() {
    const id = ++state.loadId;
    showError(null);
    $("range-note").textContent = "Loading.";
    let status;
    try {
      status = await Status.load();
    } catch (error) {
      showError(error);
      $("range-note").textContent = "";
      return;
    }
    const t1 = Date.parse(status.now) + 1000;
    const span = { t0: t1 - state.range.ms, t1 };
    const step = stepFor(state.range);
    try {
      await Promise.all([drawSeeing(span, step), drawSky(span, step), drawPointing(span, step), loadEvents(span, false)]);
    } catch (error) {
      if (id === state.loadId) {
        showError(error);
        $("range-note").textContent = "";
      }
      return;
    }
    if (id === state.loadId) {
      $("range-note").textContent =
        fmt.stamp(iso(span.t0)) + " to " + fmt.stamp(iso(span.t1)) + " UTC, " + STEP_NOTES[step];
      state.span = span;
    }
  }

  function buildControls() {
    const box = $("ranges");
    for (const range of RANGES) {
      box.append(
        h("button", {
          type: "button",
          text: range.label,
          "aria-pressed": range === state.range ? "true" : "false",
          onclick: (event) => {
            state.range = range;
            for (const button of box.children) {
              button.setAttribute("aria-pressed", button === event.currentTarget ? "true" : "false");
            }
            load();
          },
        })
      );
    }
    $("step").addEventListener("change", (event) => {
      state.step = event.target.value;
      load();
    });
    $("reload").addEventListener("click", () => load());
    $("level").addEventListener("change", (event) => {
      state.level = event.target.value;
      if (state.span) {
        loadEvents(state.span, false).catch(showError);
      }
    });
    $("events-more").addEventListener("click", () => {
      if (state.span) {
        loadEvents(state.span, true).catch(showError);
      }
    });
  }

  window.addEventListener("DOMContentLoaded", () => {
    window.Seeing.boot("history");
    buildControls();
    window.addEventListener("seeing:token", () => load());
    load();
  });
})();
