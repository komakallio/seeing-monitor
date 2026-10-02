"use strict";

/*
 * The drawing geometry of the sky overlay on the Align page: a polar coordinate grid around the
 * celestial pole, the circle that Polaris follows around the pole, and the places for the labels.
 * The file touches no page and no canvas, so Node can run it (tests/services/web/js/).
 *
 * The camera is the `camera` of the alignment state (`sky.camera`): a rotation from CIRS, the
 * apparent frame of date, to the camera frame, a plate scale, a parity, and a principal point.
 * `project` uses the formula of `CameraAttitude.project` in Python:
 *
 *     w = R u,   x = x_c + (w_x / w_z) / s,   y = y_c + parity * (w_y / w_z) / s
 *
 * with `s` in radians per pixel. A point with `w_z` of 0.05 or less is not in front. A point at
 * colatitude `rho` and right ascension `alpha` of date is the CIRS unit vector
 * `[sin(rho) cos(alpha), sin(rho) sin(alpha), cos(rho)]`, so the pole is `[0, 0, 1]`. A frame
 * pixel (x, y) covers x - 0.5 to x + 0.5 (the center of the first pixel is at 0).
 *
 * `plan(camera, frame, scale, options)` returns what to draw, in display pixels (CSS pixels, with
 * the origin at the top left of the shown image). `scale` is the display size of one frame pixel.
 *
 * The grid does not bunch up at the pole. Declination rings and right-ascension lines (hours of
 * date) both converge there, so the plan thins them out by rules that depend on the display size:
 *
 * - No line enters the hole, a disc of `hole` display pixels around the pole. The pole marker
 *   lives there.
 * - The meridians come in three nested classes: every 6 hours, every 2 hours (the extra lines),
 *   and every hour (the odd hours). A class starts at the distance from the pole where two
 *   neighbors of that class are `minGap` display pixels apart, and it runs outward from there.
 *   Near the pole only four lines remain, and the finer classes appear farther out.
 * - The rings use the smallest step from `ringSteps` for which neighbors are at least
 *   `ringMinGap` pixels apart (and at most `maxRings` of them show). A ring that would sit
 *   within 1.5 holes of the pole, or within `orbitDrop` pixels of the orbit, is left out.
 * - Everything is clipped to the frame. The plan holds at most `maxLines` polylines and
 *   `maxLabels` labels, and no label overlaps another, leaves the frame, or sits in the hole.
 *
 * The plan is a pure function of its input, so a resize gives a new plan without a new state.
 */
(function () {
  const ARCSEC_PER_RAD = 206264.80624709636;
  const DEG = Math.PI / 180;
  const FRONT_MIN = 0.05;

  const DEFAULTS = Object.freeze({
    hole: 20, // display pixels: no grid line enters this disc around the pole
    minGapFraction: 0.06, // of the display width, between the limits below
    minGapLow: 24,
    minGapHigh: 44,
    ringSteps: [0.1, 0.25, 0.5, 1, 2, 5], // degrees
    ringMinGap: 40,
    maxRings: 12,
    ringHoles: 1.5, // a ring needs a radius of at least this many holes
    orbitDrop: 8, // a ring this close to the orbit is left out
    maxLines: 40,
    maxLabels: 16,
    fontPx: 11,
    arrowInset: 26, // how far the arrow for a pole outside the frame stays from the edge
    samples: 360, // points on a ring or on the orbit
    grid: true,
    colatitudeDeg: null,
    orbitText: "",
    poleText: "pole",
    aimText: "",
    markers: [],
  });

  // The meridians, from the coarsest class to the finest. `stepHours` is the spacing of a class
  // in hours of right ascension, and `hours` are the lines that the class adds.
  const CLASSES = Object.freeze([
    { cls: 6, stepHours: 6, hours: [0, 6, 12, 18], strong: true },
    { cls: 2, stepHours: 2, hours: [2, 4, 8, 10, 14, 16, 20, 22], strong: false },
    { cls: 1, stepHours: 1, hours: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23], strong: false },
  ]);

  // The diagonals around the arrow of a pole outside the frame, the ones that point inward first.
  const INWARD = Object.freeze({
    above: ["se", "sw", "ne", "nw"],
    below: ["ne", "nw", "se", "sw"],
    left: ["se", "ne", "sw", "nw"],
    right: ["sw", "nw", "se", "ne"],
  });

  const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

  // --- Projection -------------------------------------------------------------------------------

  /** The unit vector of CIRS at colatitude `rho` and right ascension `alpha`, in radians. */
  function skyVector(rho, alpha) {
    return [Math.sin(rho) * Math.cos(alpha), Math.sin(rho) * Math.sin(alpha), Math.cos(rho)];
  }

  /** The frame pixel of a CIRS unit vector: `{ x, y, front }`, with NaN when not in front. */
  function project(camera, u) {
    const r = camera.rotation;
    const wz = r[6] * u[0] + r[7] * u[1] + r[8] * u[2];
    if (!(wz > FRONT_MIN)) {
      return { x: NaN, y: NaN, front: false };
    }
    const wx = r[0] * u[0] + r[1] * u[1] + r[2] * u[2];
    const wy = r[3] * u[0] + r[4] * u[1] + r[5] * u[2];
    const s = camera.scale_arcsec_px / ARCSEC_PER_RAD;
    return {
      x: camera.center_x_px + wx / wz / s,
      y: camera.center_y_px + (camera.parity * wy) / wz / s,
      front: true,
    };
  }

  /** The display position of a frame pixel (the pixel covers x - 0.5 to x + 0.5). */
  function toDisplay(point, scale) {
    return [(point.x + 0.5) * scale, (point.y + 0.5) * scale];
  }

  // --- Clipping ---------------------------------------------------------------------------------

  /**
   * Clip the segment from `a` to `b` to the rectangle from (0, 0) to (w, h) (Liang and Barsky).
   * Returns `{ a, b, startClipped, endClipped }` or `null` when nothing of it shows.
   */
  function clipSegment(a, b, w, h) {
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    let t0 = 0;
    let t1 = 1;
    const edges = [
      [-dx, a[0]],
      [dx, w - a[0]],
      [-dy, a[1]],
      [dy, h - a[1]],
    ];
    for (const [p, q] of edges) {
      if (p === 0) {
        if (q < 0) {
          return null;
        }
      } else {
        const t = q / p;
        if (p < 0) {
          t0 = Math.max(t0, t);
        } else {
          t1 = Math.min(t1, t);
        }
      }
    }
    if (t0 > t1) {
      return null;
    }
    return {
      a: t0 > 0 ? [a[0] + t0 * dx, a[1] + t0 * dy] : a,
      b: t1 < 1 ? [a[0] + t1 * dx, a[1] + t1 * dy] : b,
      startClipped: t0 > 0,
      endClipped: t1 < 1,
    };
  }

  /**
   * Clip a polyline to the rectangle. A `null` point breaks the line. The result is a list of
   * polylines (each a list of `[x, y]`), one for each visible run.
   */
  function clipPolyline(points, w, h) {
    const runs = [];
    let current = null;
    for (let i = 0; i + 1 < points.length; i += 1) {
      const a = points[i];
      const b = points[i + 1];
      const clipped = a && b ? clipSegment(a, b, w, h) : null;
      if (!clipped) {
        current = null;
        continue;
      }
      if (current && !clipped.startClipped) {
        current.push(clipped.b);
      } else {
        current = [clipped.a, clipped.b];
        runs.push(current);
      }
      if (clipped.endClipped) {
        current = null;
      }
    }
    return runs;
  }

  /**
   * The part of the ray `p + t d` (with `t` from `tMin` on) that lies inside the rectangle, as
   * `[t0, t1]`, or `null`.
   */
  function clipRay(p, d, tMin, w, h) {
    let t0 = tMin;
    let t1 = Infinity;
    for (const [origin, step, size] of [
      [p[0], d[0], w],
      [p[1], d[1], h],
    ]) {
      if (Math.abs(step) < 1e-12) {
        if (origin < 0 || origin > size) {
          return null;
        }
      } else {
        const first = (0 - origin) / step;
        const second = (size - origin) / step;
        t0 = Math.max(t0, Math.min(first, second));
        t1 = Math.min(t1, Math.max(first, second));
      }
    }
    return t1 - t0 >= 2 ? [t0, t1] : null;
  }

  const inside = (p, w, h) => p[0] >= 0 && p[0] <= w && p[1] >= 0 && p[1] <= h;

  /** The visible runs of the circle at colatitude `rho` (radians) around the pole. */
  function circleRuns(camera, rho, scale, w, h, samples) {
    const points = [];
    for (let i = 0; i < samples; i += 1) {
      const p = project(camera, skyVector(rho, (2 * Math.PI * i) / samples));
      points.push(p.front ? toDisplay(p, scale) : null);
    }
    // Start the walk at a point that is not visible, so that no run is cut at the seam.
    const start = points.findIndex((p) => !p || !inside(p, w, h));
    const walk = start < 0 ? points.slice() : points.slice(start).concat(points.slice(0, start));
    walk.push(walk[0]);
    return clipPolyline(walk, w, h);
  }

  // --- Labels -----------------------------------------------------------------------------------

  /** The width that a label needs, from the count of characters. Fonts differ a little. */
  function labelWidth(text, fontPx) {
    return text.length * fontPx * 0.62;
  }

  const PAD = 3;

  /** The box of a label whose text is anchored at (x, y) with the given alignment. */
  function labelBox(text, align, x, y, fontPx) {
    const width = labelWidth(text, fontPx);
    const left = align === "left" ? x : align === "right" ? x - width : x - width / 2;
    return {
      x0: left - PAD,
      x1: left + width + PAD,
      y0: y - fontPx / 2 - PAD,
      y1: y + fontPx / 2 + PAD,
    };
  }

  function boxDistance(x, y, box) {
    return Math.hypot(Math.max(box.x0 - x, 0, x - box.x1), Math.max(box.y0 - y, 0, y - box.y1));
  }

  function boxesOverlap(a, b, gap) {
    return a.x0 < b.x1 + gap && a.x1 > b.x0 - gap && a.y0 < b.y1 + gap && a.y1 > b.y0 - gap;
  }

  /** Declination of date, as a short label: `89°`, `89.5°`, `89.75°`. */
  function declinationText(declinationDeg) {
    return String(Number(declinationDeg.toFixed(2))) + "°";
  }

  // --- The plan ---------------------------------------------------------------------------------

  function emptyPlan(width, height, scale) {
    return {
      width, height, scale, hole: DEFAULTS.hole, minGap: 0, pole: null, arrow: null,
      lines: [], meridians: [], rings: [], orbit: null, labels: [], ringStepDeg: null, cuts: {},
    };
  }

  /** The distance from the pole at which a class starts: its neighbors are `minGap` apart. */
  function cutRadius(stepHours, minGap, hole) {
    const chord = 2 * Math.sin((stepHours * 15 * DEG) / 2);
    return Math.max(hole, minGap / chord);
  }

  /** The arrow for a pole outside the frame: on the edge of the inset frame, toward the pole. */
  function poleArrow(pole, w, h, inset) {
    const cx = w / 2;
    const cy = h / 2;
    const length = Math.hypot(pole[0] - cx, pole[1] - cy);
    if (length < 1e-9) {
      return null;
    }
    const dx = (pole[0] - cx) / length;
    const dy = (pole[1] - cy) / length;
    const halfW = Math.max(1, w / 2 - inset);
    const halfH = Math.max(1, h / 2 - inset);
    const t = Math.min(
      Math.abs(dx) > 1e-12 ? halfW / Math.abs(dx) : Infinity,
      Math.abs(dy) > 1e-12 ? halfH / Math.abs(dy) : Infinity
    );
    const overW = Math.abs(pole[0] - cx) / cx;
    const overH = Math.abs(pole[1] - cy) / cy;
    const side = overH >= overW ? (pole[1] < cy ? "above" : "below") : pole[0] < cx ? "left" : "right";
    return { x: cx + dx * t, y: cy + dy * t, angle: Math.atan2(dy, dx), side };
  }

  /**
   * Build the drawing plan.
   *
   * `camera` is `sky.camera` of the state, `frame` is `{ width, height }` in frame pixels, and
   * `scale` is the display pixels of one frame pixel. The options are listed in `DEFAULTS`:
   * `colatitudeDeg` (the radius of the orbit, or `null`), the texts of the labels `orbitText`,
   * `poleText`, and `aimText` (an empty text leaves a label out), `markers` (more labeled marks
   * that the page draws, each `{ id, text, x, y, radius }` in display pixels, whose labels the
   * plan places), and `grid` (false leaves the rings, the meridians, and their labels out).
   *
   * The result holds `width`, `height`, `pole` (`{ x, y, inside }` or `null`), `arrow` (for a pole
   * outside the frame), `lines` (the polylines of the grid in drawing order, each
   * `{ kind, strong, points }`), `meridians`, `rings`, `orbit` (`{ radius, polylines }` or
   * `null`), and `labels` (each `{ id, kind, text, x, y, align, box }`).
   */
  function plan(camera, frame, scale, options) {
    const o = Object.assign({}, DEFAULTS, options || {});
    const width = frame.width * scale;
    const height = frame.height * scale;
    const out = emptyPlan(width, height, scale);
    if (!camera || !(scale > 0) || !(width > 0) || !(height > 0)) {
      return out;
    }
    const poleFrame = project(camera, [0, 0, 1]);
    if (!poleFrame.front) {
      return out;
    }
    const pole = toDisplay(poleFrame, scale);
    out.pole = { x: pole[0], y: pole[1], inside: inside(pole, width, height) };
    out.hole = o.hole;
    out.minGap = clamp(o.minGapFraction * width, o.minGapLow, o.minGapHigh);
    if (!out.pole.inside) {
      out.arrow = poleArrow(pole, width, height, o.arrowInset);
    }
    const pxPerDeg = (3600 / camera.scale_arcsec_px) * scale;
    const hasOrbit = typeof o.colatitudeDeg === "number" && o.colatitudeDeg > 0;
    const orbitRho = hasOrbit ? o.colatitudeDeg * DEG : null;
    const orbitRadius = hasOrbit ? o.colatitudeDeg * pxPerDeg : null;

    if (orbitRho !== null) {
      out.orbit = { radius: orbitRadius, polylines: circleRuns(camera, orbitRho, scale, width, height, o.samples) };
    }
    const grid = { rings: [], meridians: [] };
    if (o.grid) {
      grid.meridians = buildMeridians(camera, pole, out, o);
      grid.rings = buildRings(camera, pole, pxPerDeg, orbitRadius, out, o);
    }
    out.meridians = grid.meridians;
    out.rings = grid.rings;
    out.lines = collectLines(out, o);
    out.labels = placeLabels(out, o);
    return out;
  }

  /** The meridians of every class, each from its cut to the frame edge. */
  function buildMeridians(camera, pole, out, o) {
    const meridians = [];
    for (const group of CLASSES) {
      const cut = cutRadius(group.stepHours, out.minGap, o.hole);
      out.cuts[group.cls] = cut;
      for (const hour of group.hours) {
        const probe = project(camera, skyVector(0.5 * DEG, hour * 15 * DEG));
        if (!probe.front) {
          continue;
        }
        const target = toDisplay(probe, out.scale);
        const length = Math.hypot(target[0] - pole[0], target[1] - pole[1]);
        if (length < 1e-9) {
          continue;
        }
        const dir = [(target[0] - pole[0]) / length, (target[1] - pole[1]) / length];
        const span = clipRay(pole, dir, cut, out.width, out.height);
        if (!span) {
          continue;
        }
        const start = [pole[0] + dir[0] * span[0], pole[1] + dir[1] * span[0]];
        const end = [pole[0] + dir[0] * span[1], pole[1] + dir[1] * span[1]];
        meridians.push({
          hour, cls: group.cls, strong: group.strong, cut, dir, startRadius: span[0],
          start, end, points: [start, end],
        });
      }
    }
    return meridians;
  }

  /** The declination rings with the step that suits the display size. */
  function buildRings(camera, pole, pxPerDeg, orbitRadius, out, o) {
    const corners = [[0, 0], [out.width, 0], [0, out.height], [out.width, out.height]];
    const farthest = Math.max(...corners.map((c) => Math.hypot(c[0] - pole[0], c[1] - pole[1])));
    const nearest = out.pole.inside
      ? 0
      : Math.hypot(Math.max(0 - pole[0], 0, pole[0] - out.width), Math.max(0 - pole[1], 0, pole[1] - out.height));
    const minRadius = o.ringHoles * o.hole;
    const radiiFor = (step) => {
      const radii = [];
      for (let k = 1; k * step * pxPerDeg <= farthest; k += 1) {
        const radius = k * step * pxPerDeg;
        if (radius >= minRadius && radius >= nearest) {
          radii.push([k, radius]);
        }
      }
      return radii;
    };
    let step = null;
    let chosen = [];
    for (const candidate of o.ringSteps) {
      if (candidate * pxPerDeg < o.ringMinGap) {
        continue;
      }
      const radii = radiiFor(candidate);
      step = candidate;
      chosen = radii;
      if (radii.length <= o.maxRings) {
        break;
      }
    }
    if (step === null) {
      return [];
    }
    out.ringStepDeg = step;
    const rings = [];
    for (const [k, radius] of chosen.slice(0, o.maxRings)) {
      if (orbitRadius !== null && Math.abs(radius - orbitRadius) < o.orbitDrop) {
        continue;
      }
      const colatitude = k * step;
      const polylines = circleRuns(camera, colatitude * DEG, out.scale, out.width, out.height, o.samples);
      if (polylines.length > 0) {
        rings.push({
          colatitudeDeg: colatitude,
          declinationDeg: 90 - colatitude,
          radius,
          text: declinationText(90 - colatitude),
          polylines,
        });
      }
    }
    return rings;
  }

  /**
   * The polylines of the grid in drawing order, within the budget of `maxLines` (which the orbit
   * shares). The meridians come first in the budget, because they are the lines that thin out
   * toward the pole by rule. The rings take what is left, from the pole outward, so the outer
   * rings, which are cut into pieces by the corners, are the first to go.
   */
  function collectLines(out, o) {
    const budget = o.maxLines - (out.orbit ? out.orbit.polylines.length : 0);
    let meridians = out.meridians;
    for (const group of CLASSES.slice().reverse()) {
      if (meridians.length <= budget) {
        break;
      }
      meridians = meridians.filter((m) => m.cls !== group.cls);
    }
    let room = budget - meridians.length;
    const rings = [];
    for (const ring of out.rings) {
      if (ring.polylines.length > room) {
        break;
      }
      rings.push(ring);
      room -= ring.polylines.length;
    }
    out.meridians = meridians;
    out.rings = rings;
    // Draw the rings first, then the meridians from the finest class to the coarsest.
    const lines = [];
    for (const ring of rings) {
      for (const points of ring.polylines) {
        lines.push({ kind: "ring", strong: false, points });
      }
    }
    for (const group of CLASSES.slice().reverse()) {
      for (const meridian of meridians) {
        if (meridian.cls === group.cls) {
          lines.push({ kind: "meridian", strong: meridian.strong, points: meridian.points });
        }
      }
    }
    return lines;
  }

  /** Place the labels in order of importance. A label that does not fit is dropped. */
  function placeLabels(out, o) {
    const labels = [];
    const poleX = out.pole.x;
    const poleY = out.pole.y;
    const width = out.width;
    const height = out.height;
    const halfH = o.fontPx / 2 + PAD;

    const fits = (box, exemptHole) =>
      box.x0 >= 2 && box.x1 <= width - 2 && box.y0 >= 2 && box.y1 <= height - 2 &&
      (exemptHole || boxDistance(poleX, poleY, box) >= o.hole) &&
      labels.every((other) => !boxesOverlap(box, other.box, 2));

    const add = (label, exemptHole) => {
      if (labels.length >= o.maxLabels || !label.text) {
        return false;
      }
      label.box = labelBox(label.text, label.align, label.x, label.y, o.fontPx);
      if (!fits(label.box, exemptHole)) {
        return false;
      }
      labels.push(label);
      return true;
    };

    // Marks first: each tries the four diagonals around its glyph, in a fixed order.
    const around = (id, kind, text, x, y, radius, order) => {
      const reach = radius + 4;
      const spots = {
        ne: ["left", x + reach, y - reach],
        se: ["left", x + reach, y + reach],
        nw: ["right", x - reach, y - reach],
        sw: ["right", x - reach, y + reach],
      };
      for (const name of order) {
        const [align, ax, ay] = spots[name];
        if (add({ id, kind, text, x: ax, y: ay, align }, true)) {
          return;
        }
      }
    };
    if (out.pole.inside) {
      around("pole", "pole", o.poleText, poleX, poleY, 12, ["ne", "se", "nw", "sw"]);
    } else if (out.arrow) {
      around("pole", "pole", o.poleText, out.arrow.x, out.arrow.y, 14, INWARD[out.arrow.side]);
    }
    if (o.aimText) {
      around("aim", "aim", o.aimText, width / 2, height / 2, 12, ["sw", "nw", "se", "ne"]);
    }
    for (const marker of o.markers) {
      around(marker.id, "marker", marker.text, marker.x, marker.y, marker.radius || 10, ["ne", "se", "nw", "sw"]);
    }

    // The orbit gets a label where it crosses a ray from the pole.
    if (out.orbit && o.orbitText) {
      placeOnRay(out.orbit.polylines, "orbit", "orbit", o.orbitText, add, out.pole, halfH);
    }

    // The 6-hour meridians, then the rings, then the finer meridians.
    const meridianLabel = (meridian) => {
      const text = meridian.hour + "h";
      const end = meridian.end;
      let spot;
      if (end[0] >= width - 1e-6) {
        spot = ["right", width - 5, clamp(end[1], 3 + halfH, height - 3 - halfH)];
      } else if (end[0] <= 1e-6) {
        spot = ["left", 5, clamp(end[1], 3 + halfH, height - 3 - halfH)];
      } else if (end[1] <= 1e-6) {
        const half = labelWidth(text, o.fontPx) / 2 + PAD;
        spot = ["center", clamp(end[0], 3 + half, width - 3 - half), 5 + halfH];
      } else {
        const half = labelWidth(text, o.fontPx) / 2 + PAD;
        spot = ["center", clamp(end[0], 3 + half, width - 3 - half), height - 5 - halfH];
      }
      add({ id: "ra-" + meridian.hour, kind: "meridian", text, x: spot[1], y: spot[2], align: spot[0] }, false);
    };
    for (const meridian of out.meridians) {
      if (meridian.cls === 6) {
        meridianLabel(meridian);
      }
    }
    for (const ring of out.rings) {
      placeOnRay(ring.polylines, "ring-" + ring.colatitudeDeg, "ring", ring.text, add, out.pole, halfH);
    }
    for (const cls of [2, 1]) {
      for (const meridian of out.meridians) {
        if (meridian.cls === cls) {
          meridianLabel(meridian);
        }
      }
    }
    return labels;
  }

  /**
   * Label a ring (or the orbit) where it crosses a ray from the pole that runs along a screen
   * axis: right, up, left, then down. The label sits outside the ring, next to the crossing.
   */
  function placeOnRay(polylines, id, kind, text, add, pole, halfH) {
    const rays = [
      { name: "right", test: (a, b) => crossHorizontal(a, b, pole, 1) },
      { name: "up", test: (a, b) => crossVertical(a, b, pole, -1) },
      { name: "left", test: (a, b) => crossHorizontal(a, b, pole, -1) },
      { name: "down", test: (a, b) => crossVertical(a, b, pole, 1) },
    ];
    for (const ray of rays) {
      let hit = null;
      for (const points of polylines) {
        for (let i = 0; i + 1 < points.length && !hit; i += 1) {
          hit = ray.test(points[i], points[i + 1]);
        }
      }
      if (!hit) {
        continue;
      }
      const reach = 4;
      let label;
      if (ray.name === "right") {
        label = { align: "left", x: hit[0] + reach, y: pole.y };
      } else if (ray.name === "left") {
        label = { align: "right", x: hit[0] - reach, y: pole.y };
      } else if (ray.name === "up") {
        label = { align: "center", x: pole.x, y: hit[1] - reach + 1 - halfH };
      } else {
        label = { align: "center", x: pole.x, y: hit[1] + reach - 1 + halfH };
      }
      if (add({ id, kind, text, x: label.x, y: label.y, align: label.align }, false)) {
        return;
      }
    }
  }

  /** Where the segment crosses the horizontal line through the pole, on one side of the pole. */
  function crossHorizontal(a, b, pole, side) {
    if ((a[1] - pole.y) * (b[1] - pole.y) > 0 || a[1] === b[1]) {
      return null;
    }
    const x = a[0] + ((pole.y - a[1]) / (b[1] - a[1])) * (b[0] - a[0]);
    return (x - pole.x) * side > 0 ? [x, pole.y] : null;
  }

  function crossVertical(a, b, pole, side) {
    if ((a[0] - pole.x) * (b[0] - pole.x) > 0 || a[0] === b[0]) {
      return null;
    }
    const y = a[1] + ((pole.x - a[0]) / (b[0] - a[0])) * (b[1] - a[1]);
    return (y - pole.y) * side > 0 ? [pole.x, y] : null;
  }

  window.Seeing.SkyGrid = {
    DEFAULTS, CLASSES, ARCSEC_PER_RAD, project, skyVector, toDisplay, plan, clipSegment, clipPolyline,
    clipRay, labelWidth, labelBox, cutRadius, declinationText,
  };
})();
