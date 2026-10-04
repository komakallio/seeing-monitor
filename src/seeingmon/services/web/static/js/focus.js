"use strict";

/*
 * The focus curve of the Align page: the width of the star in each of the last frames, as a line
 * over time, with the best value of the session as a dashed line.
 *
 * The scale never follows a spike. When you touch the telescope, the vibration smears every star
 * for a moment and the width jumps. The server flags such a value (`spike`), and the curve draws it
 * as a hollow mark that stays on the plot at the top edge when it is off the scale, so it neither
 * joins the line nor stretches the axis. The axis comes from the other values: from a little under
 * the smallest to a little over their 90th percentile, so the one frame in ten that the percentile
 * leaves out cannot flatten the curve either.
 *
 * The live view sends the history of the focus values in parallel lists. `merge` keeps the points
 * that the page holds: a message with `reset` replaces them, and any other message adds its new
 * points. The functions that need no page (`merge`, `valueRange`, `valueTicks`) sit apart, so that
 * a test can run them.
 */
(function () {
  const { fmt } = window.Seeing;

  const MAX_POINTS = 120; // the length of the history that the server keeps
  const GAP_S = 6; // a gap in the frames longer than this breaks the line
  const MIN_SPAN_S = 30; // the time axis is at least this wide
  const HEIGHT_PX = 150;

  /**
   * Add the points of `history` (the `focus.history` of a live-view message) to `held`
   * (`{ session, points }`) and return the new `{ session, points }`. A point is
   * `{ index, t, px, arcsec, n, spike }` with `t` in milliseconds since the epoch.
   */
  function merge(held, history) {
    if (!history || !Array.isArray(history.index)) {
      return held;
    }
    const fresh = history.reset || history.session !== held.session;
    const base = fresh ? [] : held.points;
    const last = base.length ? base[base.length - 1].index : 0;
    const added = [];
    for (let i = 0; i < history.index.length; i += 1) {
      if (history.index[i] > last) {
        added.push({
          index: history.index[i],
          t: history.t_utc_ms[i],
          px: history.fwhm_px[i],
          arcsec: history.fwhm_arcsec[i],
          n: history.n_stars[i],
          spike: Boolean(history.spike[i]),
        });
      }
    }
    return { session: history.session, points: base.concat(added).slice(-MAX_POINTS) };
  }

  /**
   * The value axis for the values that are not spikes, as `{ min, max }`, or `null` without
   * values. It runs from a little under the smallest value to a little over the 90th percentile,
   * and it is at least a fifth of the smallest value wide, so that a steady star does not turn
   * noise into a mountain range.
   */
  function valueRange(values) {
    if (values.length === 0) {
      return null;
    }
    const sorted = values.slice().sort((a, b) => a - b);
    const low = sorted[0];
    const high = sorted[Math.floor(0.9 * (sorted.length - 1))];
    const span = Math.max(high - low, 0.2 * low, 1e-6);
    return { min: Math.max(0, low - 0.15 * span), max: high + 0.25 * span };
  }

  /** Round tick values between `min` and `max`, about `count` of them. */
  function valueTicks(min, max, count) {
    const raw = (max - min) / count;
    const power = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / power;
    const step = (norm < 1.5 ? 1 : norm < 3.5 ? 2 : norm < 7.5 ? 5 : 10) * power;
    const ticks = [];
    for (let v = Math.ceil(min / step - 1e-9) * step; v <= max + step * 1e-9; v += step) {
      ticks.push(Number(v.toFixed(10)));
    }
    return ticks;
  }

  function token(name, fallback) {
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
  }

  /** Draw the curve on `canvas`. `update(points, best)` gives it the points and the best value. */
  function create(canvas) {
    const ctx = canvas.getContext("2d");
    let points = [];
    let best = null;

    function draw() {
      const ratio = window.devicePixelRatio || 1;
      const width = canvas.clientWidth || 300;
      const height = HEIGHT_PX;
      if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
        canvas.width = Math.round(width * ratio);
        canvas.height = Math.round(height * ratio);
      }
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, width, height);
      const muted = token("--muted", "#888");
      ctx.font = "12px " + token("--font", "sans-serif");
      ctx.textBaseline = "middle";
      if (points.length === 0) {
        ctx.fillStyle = muted;
        ctx.textAlign = "center";
        ctx.fillText("No focus measure yet", width / 2, height / 2);
        return;
      }
      const arcsec = points.every((p) => p.arcsec !== null && p.arcsec !== undefined);
      const valueOf = (p) => (arcsec ? p.arcsec : p.px);
      const steady = points.filter((p) => !p.spike).map(valueOf);
      const range = valueRange(steady.length ? steady : points.map(valueOf));
      const box = { left: 40, right: width - 8, top: 8, bottom: height - 8 };
      const y = (v) => box.bottom - ((Math.min(Math.max(v, range.min), range.max) - range.min) / (range.max - range.min)) * (box.bottom - box.top);
      const newest = points[points.length - 1].t;
      const t0 = Math.min(points[0].t, newest - MIN_SPAN_S * 1000);
      const x = (t) => box.left + ((t - t0) / (newest - t0)) * (box.right - box.left);

      // The grid and the value axis.
      ctx.lineWidth = 1;
      ctx.textAlign = "right";
      for (const tick of valueTicks(range.min, range.max, 4)) {
        const ty = Math.round(y(tick)) + 0.5;
        ctx.strokeStyle = token("--grid", "#ddd");
        ctx.beginPath();
        ctx.moveTo(box.left, ty);
        ctx.lineTo(box.right, ty);
        ctx.stroke();
        ctx.fillStyle = muted;
        ctx.fillText(fmt.num(tick, arcsec ? 1 : 2), box.left - 6, ty);
      }

      // The best value of the session.
      if (best !== null && best >= range.min && best <= range.max) {
        ctx.strokeStyle = token("--good", "#2a2");
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(box.left, y(best));
        ctx.lineTo(box.right, y(best));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = token("--good", "#2a2");
        ctx.textAlign = "right";
        ctx.fillText("best", box.right - 2, y(best) - 8);
      }

      // The line through the values that are not spikes, broken at long gaps.
      ctx.strokeStyle = token("--series-1", "#25c");
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.beginPath();
      let pen = false;
      let previous = null;
      for (const p of points) {
        if (p.spike) {
          continue;
        }
        const gap = previous !== null && (p.t - previous) / 1000 > GAP_S;
        if (!pen || gap) {
          ctx.moveTo(x(p.t), y(valueOf(p)));
          pen = true;
        } else {
          ctx.lineTo(x(p.t), y(valueOf(p)));
        }
        previous = p.t;
      }
      ctx.stroke();

      // The spikes and the values above the axis: a hollow mark, at the top edge when off the scale.
      ctx.strokeStyle = token("--warn", "#c80");
      ctx.lineWidth = 1.5;
      for (const p of points) {
        const v = valueOf(p);
        if (p.spike || v > range.max) {
          ctx.beginPath();
          ctx.arc(x(p.t), y(v), 3, 0, Math.PI * 2);
          ctx.stroke();
        }
      }

      // The newest value.
      const last = points[points.length - 1];
      ctx.fillStyle = last.spike ? token("--warn", "#c80") : token("--series-1", "#25c");
      ctx.beginPath();
      ctx.arc(x(last.t), y(valueOf(last)), 4, 0, Math.PI * 2);
      ctx.fill();
    }

    if (typeof ResizeObserver === "function") {
      new ResizeObserver(() => draw()).observe(canvas);
    }
    window.addEventListener("seeing:theme", () => draw());

    return {
      update(next, bestValue) {
        points = next;
        best = bestValue === undefined ? null : bestValue;
        draw();
      },
      /** The seconds that the curve spans, for the label under it. */
      spanS() {
        if (points.length < 2) {
          return 0;
        }
        return (points[points.length - 1].t - points[0].t) / 1000;
      },
    };
  }

  window.Seeing.FocusCurve = { create, merge, valueRange, valueTicks };
})();
