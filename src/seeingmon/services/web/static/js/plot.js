"use strict";

/*
 * A small canvas plotter for time series: lines and dots over a time axis in UTC, a value axis with
 * round ticks, gaps where the data have none, colored marks for flagged points, and a read-out
 * that follows the pointer or the finger. No library and no external asset.
 *
 * Colors come from the CSS tokens (--series-1 and so on), so the dark scheme and the night mode
 * apply by themselves. The plot draws again when the theme changes or its container resizes.
 *
 *     const plot = new Seeing.Plot(container, { label: "Seeing FWHM", unit: "″", digits: 2 });
 *     plot.setData({
 *       t0, t1,                                    // the time range, in milliseconds since the epoch
 *       series: [{ name: "FWHM", color: "--series-1", points: [{ t, v, note, mark }] }],
 *     });
 *
 * A point has a time `t`, a value `v` (or `null` for a gap), an optional `note` that the read-out
 * shows, and an optional `mark` (a CSS token) that draws a dot of that color on the point.
 */
(function () {
  const { h, fmt } = window.Seeing;

  const TIME_STEPS_MS = [
    60e3, 120e3, 300e3, 600e3, 900e3, 1800e3, 3600e3, 2 * 3600e3, 3 * 3600e3, 6 * 3600e3,
    12 * 3600e3, 86400e3, 2 * 86400e3, 7 * 86400e3,
  ];
  const DAY_MS = 86400e3;

  function token(name, fallback) {
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
  }

  function niceStep(span, count) {
    const raw = span / count;
    const power = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / power;
    const nice = norm < 1.5 ? 1 : norm < 3.5 ? 2 : norm < 7.5 ? 5 : 10;
    return nice * power;
  }

  /** Round tick values between `min` and `max`, about `count` of them. */
  function valueTicks(min, max, count) {
    const step = niceStep(max - min || 1, count);
    const ticks = [];
    for (let v = Math.ceil(min / step - 1e-9) * step; v <= max + step * 1e-9; v += step) {
      ticks.push(Number(v.toFixed(10)));
    }
    return { ticks, step };
  }

  /** Tick times (milliseconds) between `t0` and `t1`, aligned to UTC, about `count` of them. */
  function timeTicks(t0, t1, count) {
    const span = t1 - t0;
    const step = TIME_STEPS_MS.find((s) => span / s <= count) || TIME_STEPS_MS[TIME_STEPS_MS.length - 1];
    const ticks = [];
    for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) {
      ticks.push(t);
    }
    return { ticks, step };
  }

  function timeLabel(t, step) {
    const iso = new Date(t).toISOString();
    if (step >= DAY_MS || t % DAY_MS === 0) {
      return iso.slice(5, 10);
    }
    return iso.slice(11, 16);
  }

  function median(values) {
    if (values.length === 0) {
      return 0;
    }
    const sorted = values.slice().sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
  }

  /** The index of the point nearest in time to `t`, by bisection. `points` are sorted. */
  function nearest(points, t) {
    let low = 0;
    let high = points.length - 1;
    while (low < high) {
      const mid = (low + high) >> 1;
      if (points[mid].t < t) {
        low = mid + 1;
      } else {
        high = mid;
      }
    }
    if (low > 0 && Math.abs(points[low - 1].t - t) <= Math.abs(points[low].t - t)) {
      return low - 1;
    }
    return low;
  }

  class Plot {
    constructor(container, options) {
      this.options = Object.assign(
        { label: "", unit: "", digits: 2, spark: false, zeroBased: false, yMin: null, yMax: null, gapFactor: 3.5 },
        options
      );
      this.container = container;
      this.container.classList.add("plot", this.options.spark ? "spark" : "chart");
      this.canvas = h("canvas", { role: "img", "aria-label": this.options.label });
      this.tip = h("div", { class: "tip", hidden: true });
      this.container.append(this.canvas, this.tip);
      this.ctx = this.canvas.getContext("2d");
      this.data = { t0: 0, t1: 1, series: [] };
      this.hoverT = null;
      this.box = { left: 0, top: 0, width: 1, height: 1 };

      if (typeof ResizeObserver === "function") {
        new ResizeObserver(() => this.draw()).observe(this.container);
      } else {
        window.addEventListener("resize", () => this.draw());
      }
      window.addEventListener("seeing:theme", () => this.draw());
      if (window.matchMedia) {
        const query = window.matchMedia("(prefers-color-scheme: dark)");
        if (query.addEventListener) {
          query.addEventListener("change", () => this.draw());
        }
      }
      if (!this.options.spark) {
        const move = (event) => this.pointer(event);
        this.canvas.addEventListener("pointermove", move);
        this.canvas.addEventListener("pointerdown", move);
        this.canvas.addEventListener("pointerleave", () => this.leave());
        this.canvas.addEventListener("pointercancel", () => this.leave());
      }
    }

    /** Give the plot its data, and draw. `t0` and `t1` bound the time axis. */
    setData(data) {
      this.data = {
        t0: data.t0,
        t1: data.t1,
        series: data.series.map((series) => ({
          name: series.name,
          color: series.color || "--series-1",
          dots: Boolean(series.dots),
          points: series.points.filter((p) => Number.isFinite(p.t)),
        })),
      };
      this.hoverT = null;
      this.canvas.setAttribute("aria-label", this.summary());
      this.draw();
    }

    summary() {
      const values = [];
      for (const series of this.data.series) {
        for (const p of series.points) {
          if (p.v !== null && p.v !== undefined && Number.isFinite(p.v)) {
            values.push(p.v);
          }
        }
      }
      const label = this.options.label;
      if (values.length === 0) {
        return label + ": no data in this range.";
      }
      const last = this.data.series[0] ? this.data.series[0].points.filter((p) => p.v !== null).pop() : null;
      const unit = this.options.unit;
      const digits = this.options.digits;
      return (
        label + ": " + values.length + " values, from " + fmt.num(Math.min(...values), digits) + unit +
        " to " + fmt.num(Math.max(...values), digits) + unit +
        (last ? ", latest " + fmt.num(last.v, digits) + unit : "") + "."
      );
    }

    yRange() {
      const { yMin, yMax, zeroBased } = this.options;
      let min = Infinity;
      let max = -Infinity;
      for (const series of this.data.series) {
        for (const p of series.points) {
          if (p.v !== null && p.v !== undefined && Number.isFinite(p.v)) {
            min = Math.min(min, p.v);
            max = Math.max(max, p.v);
          }
        }
      }
      if (!Number.isFinite(min)) {
        return null;
      }
      if (zeroBased) {
        min = Math.min(0, min);
      }
      if (max - min < 1e-9) {
        const pad = Math.abs(max) * 0.1 || 1;
        min -= pad;
        max += pad;
      }
      const pad = (max - min) * 0.08;
      return {
        min: yMin !== null ? yMin : zeroBased ? min : min - pad,
        max: yMax !== null ? yMax : max + pad,
      };
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
      const spark = this.options.spark;
      this.box = spark
        ? { left: 2, top: 4, width: width - 4, height: height - 8, full: { width, height } }
        : { left: 48, top: 20, width: width - 48 - 12, height: height - 20 - 26, full: { width, height } };
      return this.box;
    }

    x(t) {
      const { t0, t1 } = this.data;
      return this.box.left + ((t - t0) / (t1 - t0 || 1)) * this.box.width;
    }

    y(v, range) {
      return this.box.top + (1 - (v - range.min) / (range.max - range.min)) * this.box.height;
    }

    draw() {
      if (!this.canvas.isConnected) {
        return;
      }
      const box = this.setup();
      const ctx = this.ctx;
      const { width, height } = box.full;
      ctx.clearRect(0, 0, width, height);
      const range = this.yRange();
      const text = token("--muted", "#566176");
      const grid = token("--grid", "#e1e6ee");
      ctx.font = "12px system-ui, sans-serif";
      if (range === null) {
        ctx.fillStyle = text;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("No data in this range", width / 2, height / 2);
        return;
      }
      if (!this.options.spark) {
        this.drawAxes(range, text, grid);
      }
      for (const series of this.data.series) {
        this.drawSeries(series, range);
      }
      if (this.hoverT !== null && !this.options.spark) {
        this.drawHover(range);
      }
    }

    drawAxes(range, text, grid) {
      const ctx = this.ctx;
      const box = this.box;
      ctx.lineWidth = 1;
      ctx.strokeStyle = grid;
      ctx.fillStyle = text;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      const values = valueTicks(range.min, range.max, 4);
      const decimals = Math.max(0, -Math.floor(Math.log10(values.step)));
      for (const v of values.ticks) {
        const y = Math.round(this.y(v, range)) + 0.5;
        ctx.beginPath();
        ctx.moveTo(box.left, y);
        ctx.lineTo(box.left + box.width, y);
        ctx.stroke();
        ctx.fillText(fmt.num(v, Math.min(decimals, 4)), box.left - 6, y);
      }
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      const times = timeTicks(this.data.t0, this.data.t1, Math.max(2, Math.floor(box.width / 56)));
      for (const t of times.ticks) {
        const x = Math.round(this.x(t)) + 0.5;
        ctx.beginPath();
        ctx.moveTo(x, box.top);
        ctx.lineTo(x, box.top + box.height);
        ctx.stroke();
        ctx.fillText(timeLabel(t, times.step), x, box.top + box.height + 6);
      }
      if (this.options.unit) {
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.fillText(this.options.unit, 2, 2);
      }
    }

    drawSeries(series, range) {
      const ctx = this.ctx;
      const color = token(series.color, "#1f55c4");
      const points = series.points;
      const spacing = median(points.slice(1).map((p, i) => p.t - points[i].t));
      const gap = spacing * this.options.gapFactor;
      ctx.lineWidth = this.options.spark ? 1.5 : 1.75;
      ctx.lineJoin = "round";
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      if (!series.dots) {
        let open = false;
        let previous = null;
        ctx.beginPath();
        for (const p of points) {
          const missing = p.v === null || p.v === undefined || !Number.isFinite(p.v);
          if (missing || (previous && gap > 0 && p.t - previous.t > gap)) {
            open = false;
          }
          if (missing) {
            previous = null;
            continue;
          }
          const x = this.x(p.t);
          const y = this.y(p.v, range);
          if (open) {
            ctx.lineTo(x, y);
          } else {
            ctx.moveTo(x, y);
            open = true;
          }
          previous = p;
        }
        ctx.stroke();
      }
      const sparse = points.length <= 120;
      for (let i = 0; i < points.length; i += 1) {
        const p = points[i];
        if (p.v === null || p.v === undefined || !Number.isFinite(p.v)) {
          continue;
        }
        const x = this.x(p.t);
        const y = this.y(p.v, range);
        const isolated =
          (i === 0 || points[i - 1].v === null || p.t - points[i - 1].t > gap) &&
          (i === points.length - 1 || points[i + 1].v === null || points[i + 1].t - p.t > gap);
        if (p.mark) {
          ctx.fillStyle = token(p.mark, color);
          ctx.beginPath();
          ctx.arc(x, y, this.options.spark ? 2.5 : 3.5, 0, Math.PI * 2);
          ctx.fill();
        } else if (series.dots || isolated || (sparse && !this.options.spark)) {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(x, y, series.dots ? 2.5 : 2, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      if (this.options.spark) {
        const last = points.filter((p) => p.v !== null && Number.isFinite(p.v)).pop();
        if (last) {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(this.x(last.t), this.y(last.v, range), 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    drawHover(range) {
      const ctx = this.ctx;
      const box = this.box;
      const x = this.x(this.hoverT);
      ctx.strokeStyle = token("--muted", "#566176");
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(Math.round(x) + 0.5, box.top);
      ctx.lineTo(Math.round(x) + 0.5, box.top + box.height);
      ctx.stroke();
      ctx.setLineDash([]);
      for (const series of this.data.series) {
        const p = this.pick(series);
        if (p && p.v !== null && Number.isFinite(p.v)) {
          ctx.fillStyle = token(series.color, "#1f55c4");
          ctx.strokeStyle = token("--surface", "#ffffff");
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(this.x(p.t), this.y(p.v, range), 4.5, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }
      }
    }

    /** The point of a series nearest to the hovered time, or null when it is too far away. */
    pick(series) {
      if (series.points.length === 0) {
        return null;
      }
      const p = series.points[nearest(series.points, this.hoverT)];
      const spacing = median(series.points.slice(1).map((q, i) => q.t - series.points[i].t));
      return Math.abs(p.t - this.hoverT) <= Math.max(spacing * 1.5, (this.data.t1 - this.data.t0) / 200) ? p : null;
    }

    pointer(event) {
      const rect = this.canvas.getBoundingClientRect();
      const px = event.clientX - rect.left;
      const { left, width } = this.box;
      if (px < left - 4 || px > left + width + 4) {
        this.leave();
        return;
      }
      const t = this.data.t0 + ((px - left) / (width || 1)) * (this.data.t1 - this.data.t0);
      this.hoverT = Math.min(this.data.t1, Math.max(this.data.t0, t));
      const rows = [];
      let anchor = null;
      for (const series of this.data.series) {
        const p = this.pick(series);
        if (p) {
          anchor = anchor || p;
          const value = p.v === null || p.v === undefined ? fmt.dash : fmt.num(p.v, this.options.digits) + this.options.unit;
          rows.push(series.name + ": " + value + (p.note ? " (" + p.note + ")" : ""));
        }
      }
      this.draw();
      if (!anchor) {
        this.tip.hidden = true;
        return;
      }
      const when = new Date(anchor.t).toISOString();
      this.tip.replaceChildren(h("div", { text: when.slice(0, 10) + " " + when.slice(11, 16) + " UTC" }), ...rows.map((r) => h("div", { text: r })));
      this.tip.hidden = false;
      const tipWidth = this.tip.offsetWidth;
      const x = this.x(anchor.t);
      const flip = x + tipWidth + 14 > rect.width;
      this.tip.style.left = Math.max(0, flip ? x - tipWidth - 10 : x + 10) + "px";
      this.tip.style.top = "6px";
    }

    leave() {
      this.hoverT = null;
      this.tip.hidden = true;
      this.draw();
    }
  }

  window.Seeing.Plot = Plot;
  window.Seeing.plotHelpers = { valueTicks, timeTicks, timeLabel, nearest, niceStep };
})();
