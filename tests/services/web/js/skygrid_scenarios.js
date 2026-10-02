"use strict";

/*
 * The scenarios of the sky overlay geometry (static/js/skygrid.js): the projection against golden
 * numbers from Python, the clipping, and the rules that keep the grid from bunching up at the
 * pole. The function takes `SkyGrid`, a `test(name, fn)` function, an `assert` object, and the
 * fixture (`skygrid_fixture.json`), so that Node (skygrid.test.js) and a browser console can both
 * run it. Every scenario is synchronous and deterministic.
 */
module.exports = function scenarios(SkyGrid, test, assert, fixture) {
  const DEG = Math.PI / 180;
  const FRAME = { width: 4144, height: 2822 };
  const SCALE_ARCSEC = 3.82;
  const COLATITUDE = 0.6265; // degrees, about the colatitude of Polaris in 2026
  const WIDTHS = [320, 360, 470, 1100]; // display widths in CSS pixels
  const HOLE = 20;

  const near = (actual, expected, tolerance, what) => {
    assert.ok(
      Math.abs(actual - expected) <= tolerance,
      (what || "value") + ": expected " + expected + " within " + tolerance + " but got " + actual
    );
  };

  // --- Helpers ------------------------------------------------------------------------------------

  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const normalize = (v) => {
    const length = Math.hypot(v[0], v[1], v[2]);
    return [v[0] / length, v[1] / length, v[2] / length];
  };

  /**
   * A camera whose pole falls on the frame pixel (poleX, poleY). The pole is the CIRS z axis, so
   * the third column of the rotation is the camera-frame direction of the pole. `turnDeg` turns the
   * sky about the pole, which moves the lines of right ascension and nothing else.
   */
  function cameraWithPole(poleX, poleY, options) {
    const o = Object.assign({ turnDeg: 25, parity: 1, scale: SCALE_ARCSEC, frame: FRAME }, options);
    const cx = (o.frame.width - 1) / 2;
    const cy = (o.frame.height - 1) / 2;
    const s = o.scale / SkyGrid.ARCSEC_PER_RAD;
    const c = normalize([(poleX - cx) * s, o.parity * (poleY - cy) * s, 1]);
    const e1 = normalize([c[2], 0, -c[0]]);
    const e2 = cross(c, e1);
    const turn = o.turnDeg * DEG;
    const t1 = [0, 1, 2].map((i) => Math.cos(turn) * e1[i] + Math.sin(turn) * e2[i]);
    const t2 = cross(c, t1);
    return {
      rotation: [t1[0], t2[0], c[0], t1[1], t2[1], c[1], t1[2], t2[2], c[2]],
      scale_arcsec_px: o.scale,
      parity: o.parity,
      center_x_px: cx,
      center_y_px: cy,
    };
  }

  const GEOMETRIES = {
    "the pole at the center": [2071.5, 1410.5],
    "the pole near a corner": [3800, 450],
    "the pole outside the frame": [2071.5, -1100],
  };

  const baseOptions = { colatitudeDeg: COLATITUDE, orbitText: "orbit", aimText: "aim" };
  const planFor = (camera, width, extra) =>
    SkyGrid.plan(camera, FRAME, width / FRAME.width, Object.assign({}, baseOptions, extra));

  function distanceToSegment(p, a, b) {
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const length2 = dx * dx + dy * dy;
    const t = length2 === 0 ? 0 : Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2));
    return Math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy));
  }

  function boxDistance(p, box) {
    return Math.hypot(Math.max(box.x0 - p[0], 0, p[0] - box.x1), Math.max(box.y0 - p[1], 0, p[1] - box.y1));
  }

  const overlap = (a, b) => a.x0 < b.x1 && a.x1 > b.x0 && a.y0 < b.y1 && a.y1 > b.y0;

  /** Every guarantee of the plan, for one plan. `name` says which case failed. */
  function checkPlan(plan, name) {
    if (!plan.pole) {
      assert.equal(plan.lines.length, 0, name + ": no pole, no lines");
      return;
    }
    const w = plan.width;
    const h = plan.height;
    const pole = [plan.pole.x, plan.pole.y];

    // No line enters the hole.
    for (const line of plan.lines) {
      for (let i = 0; i + 1 < line.points.length; i += 1) {
        const d = distanceToSegment(pole, line.points[i], line.points[i + 1]);
        assert.ok(d >= HOLE - 1e-6, name + ": a " + line.kind + " comes within " + d.toFixed(2) + " px of the pole");
      }
    }

    // Everything is clipped to the frame.
    const drawn = plan.lines.map((line) => line.points);
    if (plan.orbit) {
      drawn.push(...plan.orbit.polylines);
    }
    for (const points of drawn) {
      assert.ok(points.length >= 2, name + ": a polyline needs two points");
      for (const p of points) {
        assert.ok(p[0] >= -1e-6 && p[0] <= w + 1e-6 && p[1] >= -1e-6 && p[1] <= h + 1e-6, name + ": a point lies outside the frame");
      }
    }

    // The meridians start at their cut, and neighbors are a minimum gap apart there.
    for (const m of plan.meridians) {
      near(Math.hypot(m.start[0] - pole[0], m.start[1] - pole[1]), m.startRadius, 1e-6, name + ": the start of hour " + m.hour);
      assert.ok(m.startRadius >= m.cut - 1e-6, name + ": hour " + m.hour + " starts before its cut");
      assert.ok(m.cut >= HOLE - 1e-9, name + ": a cut lies inside the hole");
    }
    for (let i = 0; i < plan.meridians.length; i += 1) {
      for (let j = i + 1; j < plan.meridians.length; j += 1) {
        const a = plan.meridians[i];
        const b = plan.meridians[j];
        const r = Math.max(a.cut, b.cut);
        const chord = Math.hypot(r * (a.dir[0] - b.dir[0]), r * (a.dir[1] - b.dir[1]));
        assert.ok(chord >= plan.minGap - 1, name + ": hours " + a.hour + " and " + b.hour + " are " + chord.toFixed(1) + " px apart at their cut, and the gap is " + plan.minGap);
      }
    }
    // Near the pole only the four 6-hour lines remain, and the 2-hour lines join before the hours.
    const reaching = (r) => plan.meridians.filter((m) => m.startRadius <= r).length;
    if (plan.cuts[2] !== undefined) {
      assert.ok(reaching(plan.cuts[2] - 1e-6) <= 4, name + ": more than four lines come near the pole");
      assert.ok(reaching(plan.cuts[1] - 1e-6) <= 12, name + ": more than twelve lines come near the pole");
    }

    // The rings are far enough apart and far enough from the pole and from the orbit.
    const radii = plan.rings.map((ring) => ring.radius).sort((a, b) => a - b);
    for (let i = 0; i + 1 < radii.length; i += 1) {
      assert.ok(radii[i + 1] - radii[i] >= 39, name + ": rings " + radii[i].toFixed(1) + " and " + radii[i + 1].toFixed(1) + " bunch up");
    }
    assert.ok(plan.rings.length <= 12, name + ": too many rings");
    for (const ring of plan.rings) {
      assert.ok(ring.radius >= 1.5 * HOLE - 1e-6, name + ": a ring sits too near the pole");
      if (plan.orbit) {
        assert.ok(Math.abs(ring.radius - plan.orbit.radius) >= 8 - 1e-6, name + ": a ring runs along the orbit");
      }
    }

    // The budgets.
    assert.ok(plan.lines.length + (plan.orbit ? plan.orbit.polylines.length : 0) <= 40, name + ": too many polylines");
    assert.ok(plan.labels.length <= 16, name + ": too many labels");

    // The labels: inside the frame, apart from each other, and out of the hole.
    for (let i = 0; i < plan.labels.length; i += 1) {
      const a = plan.labels[i];
      assert.ok(a.box.x0 >= 0 && a.box.x1 <= w && a.box.y0 >= 0 && a.box.y1 <= h, name + ": the label " + a.text + " leaves the frame");
      if (a.kind === "meridian" || a.kind === "ring" || a.kind === "orbit") {
        assert.ok(boxDistance(pole, a.box) >= HOLE - 1e-6, name + ": the label " + a.text + " sits in the hole");
      }
      for (let j = i + 1; j < plan.labels.length; j += 1) {
        assert.ok(!overlap(a.box, plan.labels[j].box), name + ": the labels " + a.text + " and " + plan.labels[j].text + " overlap");
      }
      // A label of the grid keeps off the dashed circle of the orbit.
      if (plan.orbit && (a.kind === "meridian" || a.kind === "ring")) {
        for (const points of plan.orbit.polylines) {
          for (let k = 0; k + 1 < points.length; k += 1) {
            const hit = SkyGrid.clipSegment(
              [points[k][0] - a.box.x0, points[k][1] - a.box.y0],
              [points[k + 1][0] - a.box.x0, points[k + 1][1] - a.box.y0],
              a.box.x1 - a.box.x0,
              a.box.y1 - a.box.y0
            );
            assert.ok(hit === null, name + ": the label " + a.text + " sits on the orbit");
          }
        }
      }
    }
  }

  // --- Projection ---------------------------------------------------------------------------------

  test("the projection agrees with the golden numbers from Python", () => {
    assert.ok(fixture.cases.length >= 5);
    for (const item of fixture.cases) {
      for (const point of item.points) {
        const got = SkyGrid.project(item.camera, point.u);
        assert.equal(got.front, point.front, item.name + ": front");
        if (point.front) {
          near(got.x, point.x, 0.005, item.name + ": x");
          near(got.y, point.y, 0.005, item.name + ": y");
        } else {
          assert.ok(Number.isNaN(got.x) && Number.isNaN(got.y), item.name + ": a point behind the camera has no pixel");
        }
      }
    }
  });

  test("the pole projects where Python says, and the plan puts it in display pixels", () => {
    for (const item of fixture.cases) {
      const scale = 0.1;
      const plan = SkyGrid.plan(item.camera, item.frame, scale, {});
      if (item.pole === null) {
        assert.equal(plan.pole, null, item.name);
        continue;
      }
      near(plan.pole.x, (item.pole[0] + 0.5) * scale, 0.01, item.name + ": x");
      near(plan.pole.y, (item.pole[1] + 0.5) * scale, 0.01, item.name + ": y");
    }
  });

  test("a camera that is built to put the pole on a pixel does so, for both parities", () => {
    for (const parity of [1, -1]) {
      for (const [x, y] of [[2071.5, 1410.5], [3800, 450], [-200, 3000], [2071.5, -1100]]) {
        const camera = cameraWithPole(x, y, { parity, turnDeg: 71 });
        const got = SkyGrid.project(camera, [0, 0, 1]);
        near(got.x, x, 1e-6, "x");
        near(got.y, y, 1e-6, "y");
      }
    }
  });

  test("a point at a colatitude and a right ascension of date is a unit vector, and the pole is z", () => {
    assert.deepEqual(SkyGrid.skyVector(0, 1.2), [0, 0, 1]);
    const v = SkyGrid.skyVector(0.3, 2.0);
    near(Math.hypot(v[0], v[1], v[2]), 1, 1e-12, "length");
    near(v[0], Math.sin(0.3) * Math.cos(2.0), 1e-12, "x");
    near(v[1], Math.sin(0.3) * Math.sin(2.0), 1e-12, "y");
  });

  test("a point at or below the front limit has no pixel", () => {
    const camera = cameraWithPole(2071.5, 1410.5, { turnDeg: 0 });
    // The camera looks at the pole, so w_z is the z component. 0.05 is the limit, as in Python.
    assert.equal(SkyGrid.project(camera, normalize([1, 0, 0.0499])).front, false);
    assert.equal(SkyGrid.project(camera, normalize([1, 0, 0.0501])).front, true);
  });

  // --- Clipping -----------------------------------------------------------------------------------

  test("a segment is clipped to the rectangle", () => {
    assert.deepEqual(SkyGrid.clipSegment([10, 10], [20, 20], 100, 50), { a: [10, 10], b: [20, 20], startClipped: false, endClipped: false });
    const out = SkyGrid.clipSegment([50, 25], [150, 25], 100, 50);
    assert.deepEqual(out.b, [100, 25]);
    assert.equal(out.endClipped, true);
    assert.equal(out.startClipped, false);
    const through = SkyGrid.clipSegment([-50, 25], [150, 25], 100, 50);
    assert.deepEqual([through.a, through.b], [[0, 25], [100, 25]]);
    assert.equal(SkyGrid.clipSegment([-50, -10], [-10, -50], 100, 50), null);
    assert.equal(SkyGrid.clipSegment([-5, 60], [150, 60], 100, 50), null);
    assert.equal(SkyGrid.clipSegment([-5, -1], [-1, 60], 100, 50), null);
  });

  test("a polyline that leaves the frame and returns gives two runs, and a null point breaks a run", () => {
    const points = [[10, 10], [60, 10], [120, 10], [120, 40], [60, 40], [10, 40]];
    const runs = SkyGrid.clipPolyline(points, 100, 50);
    assert.equal(runs.length, 2);
    assert.deepEqual(runs[0], [[10, 10], [60, 10], [100, 10]]);
    assert.deepEqual(runs[1], [[100, 40], [60, 40], [10, 40]]);
    const broken = SkyGrid.clipPolyline([[10, 10], [20, 10], null, [30, 10], [40, 10]], 100, 50);
    assert.equal(broken.length, 2);
  });

  test("a ray is clipped to the rectangle from its cut outward", () => {
    assert.deepEqual(SkyGrid.clipRay([50, 25], [1, 0], 20, 100, 50), [20, 50]);
    assert.equal(SkyGrid.clipRay([50, 25], [1, 0], 60, 100, 50), null);
    assert.equal(SkyGrid.clipRay([150, 25], [1, 0], 20, 100, 50), null); // away from the frame
    const entering = SkyGrid.clipRay([-30, 25], [1, 0], 20, 100, 50);
    near(entering[0], 30, 1e-9, "enters at the left edge");
    near(entering[1], 130, 1e-9, "leaves at the right edge");
  });

  // --- The plan -----------------------------------------------------------------------------------

  test("there is nothing to draw without a camera, and with the pole behind the camera", () => {
    const none = SkyGrid.plan(null, FRAME, 0.1, {});
    assert.equal(none.pole, null);
    assert.equal(none.lines.length, 0);
    assert.equal(none.labels.length, 0);
    const behind = fixture.cases.find((item) => item.pole === null);
    const plan = SkyGrid.plan(behind.camera, behind.frame, 0.1, baseOptions);
    assert.equal(plan.pole, null);
    assert.equal(plan.lines.length, 0);
    assert.equal(plan.orbit, null);
  });

  test("the same input gives the same plan", () => {
    for (const [name, [x, y]] of Object.entries(GEOMETRIES)) {
      const camera = cameraWithPole(x, y);
      for (const width of WIDTHS) {
        assert.deepEqual(planFor(camera, width), planFor(camera, width), name + " at " + width);
      }
    }
  });

  test("the plan scales with the display and stays the same drawing", () => {
    const camera = cameraWithPole(3800, 450);
    const small = planFor(camera, 360);
    assert.equal(small.width, 360);
    near(small.height, 360 * FRAME.height / FRAME.width, 1e-9, "height");
    near(small.pole.x, (3800 + 0.5) * 360 / FRAME.width, 1e-9, "pole x");
    near(small.pole.y, (450 + 0.5) * 360 / FRAME.width, 1e-9, "pole y");
    assert.equal(small.pole.inside, true);
  });

  test("the pole is inside the frame, or an arrow points at it from the edge", () => {
    const inside = planFor(cameraWithPole(3800, 450), 470);
    assert.equal(inside.pole.inside, true);
    assert.equal(inside.arrow, null);
    const outside = planFor(cameraWithPole(2071.5, -1100), 470, { poleText: "pole 1.4° above" });
    assert.equal(outside.pole.inside, false);
    assert.ok(outside.arrow, "an arrow");
    assert.equal(outside.arrow.side, "above");
    near(outside.arrow.x, outside.width / 2, 1e-6, "the arrow is above the center");
    assert.ok(outside.arrow.y > 0 && outside.arrow.y < 40, "the arrow sits near the top edge, inside the frame");
    near(outside.arrow.angle, -Math.PI / 2, 1e-9, "the arrow points up");
    const label = outside.labels.find((l) => l.id === "pole");
    assert.ok(label, "the arrow has a label");
    assert.equal(label.text, "pole 1.4° above");
    const left = planFor(cameraWithPole(-1500, 1410.5), 470);
    assert.equal(left.arrow.side, "left");
    const right = planFor(cameraWithPole(5800, 1410.5), 470);
    assert.equal(right.arrow.side, "right");
    const below = planFor(cameraWithPole(2071.5, 4300), 470);
    assert.equal(below.arrow.side, "below");
  });

  test("the cut of a class is where its neighbors are a minimum gap apart", () => {
    const camera = cameraWithPole(2071.5, 1410.5);
    const expected = { 320: [20, 46.4, 91.9], 360: [20, 46.4, 91.9], 470: [20, 54.5, 108.0], 1100: [31.1, 85.0, 168.5] };
    for (const width of WIDTHS) {
      const plan = planFor(camera, width);
      const gap = Math.min(44, Math.max(24, 0.06 * width));
      near(plan.minGap, gap, 1e-9, "the gap at " + width);
      [6, 2, 1].forEach((cls, index) => near(plan.cuts[cls], expected[width][index], 0.1, "the cut of class " + cls + " at " + width));
    }
  });

  test("the three classes of meridians are the hours that the rules name", () => {
    const camera = cameraWithPole(2071.5, 1410.5);
    const plan = planFor(camera, 1100);
    const byClass = (cls) => plan.meridians.filter((m) => m.cls === cls).map((m) => m.hour).sort((a, b) => a - b);
    assert.deepEqual(byClass(6), [0, 6, 12, 18]);
    assert.deepEqual(byClass(2), [2, 4, 8, 10, 14, 16, 20, 22]);
    assert.deepEqual(byClass(1), [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]);
    assert.ok(plan.meridians.filter((m) => m.cls === 6).every((m) => m.strong));
    assert.ok(plan.meridians.filter((m) => m.cls !== 6).every((m) => !m.strong));
  });

  test("the finer classes start farther from the pole", () => {
    const plan = planFor(cameraWithPole(2071.5, 1410.5), 1100);
    const start = (hour) => plan.meridians.find((m) => m.hour === hour).startRadius;
    assert.ok(start(0) < start(2) && start(2) < start(1), "6 hours, then 2 hours, then 1 hour");
    near(start(0), plan.cuts[6], 1e-6, "6 hours");
    near(start(2), plan.cuts[2], 1e-6, "2 hours");
    near(start(1), plan.cuts[1], 1e-6, "1 hour");
  });

  test("the ring step grows with the sparsity of the display", () => {
    const camera = cameraWithPole(2071.5, 1410.5);
    const steps = WIDTHS.map((width) => planFor(camera, width).ringStepDeg);
    assert.deepEqual(steps, [1, 0.5, 0.5, 0.25]);
    const wide = planFor(camera, 1100);
    assert.deepEqual(wide.rings.map((r) => r.text).slice(0, 4), ["89.75°", "89.5°", "89.25°", "89°"]);
    near(wide.rings[0].radius, 0.25 * (3600 / SCALE_ARCSEC) * (1100 / FRAME.width), 1e-6, "radius of the first ring");
  });

  test("a declination is written with the digits that it needs", () => {
    assert.equal(SkyGrid.declinationText(89), "89°");
    assert.equal(SkyGrid.declinationText(89.5), "89.5°");
    assert.equal(SkyGrid.declinationText(89.75), "89.75°");
    assert.equal(SkyGrid.declinationText(90 - 0.3), "89.7°");
    assert.equal(SkyGrid.declinationText(88), "88°");
  });

  test("a ring that runs along the orbit is left out", () => {
    const camera = cameraWithPole(2071.5, 1410.5);
    // At 1100 px the 0.5 degree ring has a radius of 125 px, and so does an orbit of 0.5 degrees.
    const dropped = planFor(camera, 1100, { colatitudeDeg: 0.5 });
    assert.ok(!dropped.rings.some((r) => r.colatitudeDeg === 0.5), "the 0.5 degree ring is dropped");
    assert.ok(dropped.rings.some((r) => r.colatitudeDeg === 0.25));
    assert.ok(dropped.rings.some((r) => r.colatitudeDeg === 0.75));
    const kept = planFor(camera, 1100, { colatitudeDeg: null });
    assert.ok(kept.rings.some((r) => r.colatitudeDeg === 0.5), "without an orbit the ring stays");
    assert.equal(kept.orbit, null);
  });

  test("the orbit is a closed polyline inside the frame, or runs that end on the edge", () => {
    const centered = planFor(cameraWithPole(2071.5, 1410.5), 1100);
    assert.equal(centered.orbit.polylines.length, 1);
    const ring = centered.orbit.polylines[0];
    assert.deepEqual(ring[0], ring[ring.length - 1]);
    near(centered.orbit.radius, COLATITUDE * (3600 / SCALE_ARCSEC) * (1100 / FRAME.width), 1e-6, "radius");
    for (const p of ring) {
      near(Math.hypot(p[0] - centered.pole.x, p[1] - centered.pole.y), centered.orbit.radius, 0.5, "a point of the orbit");
    }
    const cut = planFor(cameraWithPole(3800, 450), 1100);
    assert.ok(cut.orbit.polylines.length >= 1);
    for (const points of cut.orbit.polylines) {
      for (const end of [points[0], points[points.length - 1]]) {
        const onEdge = end[0] < 1e-6 || end[1] < 1e-6 || Math.abs(end[0] - cut.width) < 1e-6 || Math.abs(end[1] - cut.height) < 1e-6;
        assert.ok(onEdge, "a run of the orbit ends on a frame edge");
      }
    }
  });

  test("without the grid there are no lines and no labels of the grid, and the orbit stays", () => {
    const plan = planFor(cameraWithPole(2071.5, 1410.5), 470, { grid: false });
    assert.equal(plan.lines.length, 0);
    assert.equal(plan.meridians.length, 0);
    assert.equal(plan.rings.length, 0);
    assert.ok(plan.orbit);
    assert.ok(plan.labels.every((l) => l.kind !== "meridian" && l.kind !== "ring"));
    assert.ok(plan.labels.some((l) => l.id === "pole"));
  });

  test("a smaller budget drops the finest classes whole, and keeps the 6-hour lines", () => {
    const plan = planFor(cameraWithPole(2071.5, 1410.5), 1100, { maxLines: 12 });
    assert.ok(plan.lines.length + plan.orbit.polylines.length <= 12);
    assert.ok(plan.meridians.length > 0);
    assert.ok(plan.meridians.every((m) => m.cls === 6), "only the 6-hour lines remain");
  });

  test("a long ring list keeps the inner rings, and the lines stay within the budget", () => {
    const plan = planFor(cameraWithPole(2071.5, 1410.5), 1100);
    assert.equal(plan.meridians.length, 24, "all the hours fit");
    const innermost = plan.rings.map((r) => r.colatitudeDeg);
    assert.deepEqual(innermost, innermost.slice().sort((a, b) => a - b));
    assert.ok(innermost[0] === 0.25, "the rings start at the pole and run outward");
  });

  // --- The labels ---------------------------------------------------------------------------------

  test("the labels name the hours and the declinations of date, at the frame edge and along a ray", () => {
    const plan = planFor(cameraWithPole(2071.5, 1410.5), 1100);
    const texts = plan.labels.map((l) => l.text);
    for (const hour of ["0h", "6h", "12h", "18h"]) {
      assert.ok(texts.includes(hour), "the label " + hour);
    }
    assert.ok(texts.includes("89.75°") && texts.includes("89.5°"), "ring labels");
    assert.ok(texts.includes("pole") && texts.includes("aim") && texts.includes("orbit"));
    for (const label of plan.labels.filter((l) => l.kind === "meridian")) {
      const hour = Number(label.text.slice(0, -1));
      const line = plan.meridians.find((m) => m.hour === hour);
      assert.ok(line, "the label belongs to a drawn line");
    }
    for (const label of plan.labels.filter((l) => l.kind === "ring")) {
      const ring = plan.rings.find((r) => r.text === label.text);
      assert.ok(ring, "the label belongs to a drawn ring");
    }
    // A line is labeled once.
    const ids = plan.labels.map((l) => l.id);
    assert.equal(new Set(ids).size, ids.length);
  });

  test("a mark that the page draws gets a label that keeps clear of the others", () => {
    const camera = cameraWithPole(2071.5, 1410.5);
    const markers = [
      { id: "polaris", text: "Polaris", x: 400, y: 300, radius: 10 },
      { id: "target", text: "target", x: 420, y: 320, radius: 16 },
    ];
    const plan = planFor(camera, 1100, { markers });
    assert.ok(plan.labels.find((l) => l.id === "polaris"));
    checkPlan(plan, "marks");
  });

  test("a label keeps off the glyph of a mark that the page draws, but its own label sits beside it", () => {
    const camera = cameraWithPole(2071.5, 1410.5);
    const base = planFor(camera, 470);
    const crowded = base.labels.find((l) => l.kind === "ring");
    const x = (crowded.box.x0 + crowded.box.x1) / 2;
    const y = (crowded.box.y0 + crowded.box.y1) / 2;
    const plan = planFor(camera, 470, { markers: [{ id: "target", text: "target", x, y, radius: 16, reserve: 30 }] });
    // The glyph: a ring of 16 pixels and two arms of 30 pixels.
    const glyph = [
      { x0: x - 16, x1: x + 16, y0: y - 16, y1: y + 16 },
      { x0: x - 30, x1: x + 30, y0: y - 1.5, y1: y + 1.5 },
      { x0: x - 1.5, x1: x + 1.5, y0: y - 30, y1: y + 30 },
    ];
    for (const label of plan.labels) {
      if (label.id !== "target") {
        assert.ok(glyph.every((box) => !overlap(label.box, box)), "the label " + label.text + " sits on the mark");
      }
    }
    const own = plan.labels.find((l) => l.id === "target");
    assert.ok(own, "the mark has its label");
    assert.ok(Math.hypot((own.box.x0 + own.box.x1) / 2 - x, (own.box.y0 + own.box.y1) / 2 - y) < 60, "beside its mark");
    checkPlan(plan, "crowded");
  });

  test("the pole and the aim keep their labels when the pole sits on the aim", () => {
    const plan = planFor(cameraWithPole(2071.5, 1410.5), 470);
    const pole = plan.labels.find((l) => l.id === "pole");
    const aim = plan.labels.find((l) => l.id === "aim");
    assert.ok(pole && aim, "both labels");
    assert.ok(!overlap(pole.box, aim.box));
  });

  test("a label that cannot be placed is dropped and never forced", () => {
    // A tiny display leaves no room: nothing may overlap, leave the frame, or sit in the hole.
    const plan = planFor(cameraWithPole(2071.5, 1410.5), 120);
    checkPlan(plan, "tiny");
    const limited = planFor(cameraWithPole(2071.5, 1410.5), 1100, { maxLabels: 5 });
    assert.ok(limited.labels.length <= 5);
    checkPlan(limited, "limited");
  });

  // --- The rules against bunching, over many cameras ----------------------------------------------

  for (const [name, [x, y]] of Object.entries(GEOMETRIES)) {
    for (const width of WIDTHS) {
      test(name + ", shown " + width + " px wide, keeps every guarantee", () => {
        for (const turnDeg of [0, 17, 33, 72]) {
          const camera = cameraWithPole(x, y, { turnDeg });
          const plan = planFor(camera, width, {
            markers: [{ id: "polaris", text: "Polaris", x: width * 0.3, y: width * 0.2, radius: 10 }],
          });
          assert.ok(plan.pole, "the pole is in front");
          checkPlan(plan, name + ", " + width + " px, turn " + turnDeg);
          assert.ok(plan.lines.length > 0 || !plan.pole.inside, "there are lines to draw");
        }
      });
    }
  }

  test("the guarantees hold for poles across and beyond the frame, in both parities", () => {
    for (const parity of [1, -1]) {
      for (const x of [-600, 1000, 2071.5, 3200, 4700]) {
        for (const y of [-900, 400, 1410.5, 2500]) {
          for (const turnDeg of [0, 40]) {
            const camera = cameraWithPole(x, y, { parity, turnDeg });
            for (const width of WIDTHS) {
              checkPlan(planFor(camera, width), "pole (" + x + ", " + y + "), parity " + parity + ", turn " + turnDeg + ", " + width + " px");
            }
          }
        }
      }
    }
  });

  test("near the pole only the four 6-hour lines remain, at every display size", () => {
    const camera = cameraWithPole(2071.5, 1410.5);
    for (const width of WIDTHS) {
      const plan = planFor(camera, width);
      const pole = [plan.pole.x, plan.pole.y];
      const closeLines = plan.lines.filter((line) => {
        if (line.kind !== "meridian") {
          return false; // the rings keep their own spacing, which another scenario checks
        }
        let nearest = Infinity;
        for (let i = 0; i + 1 < line.points.length; i += 1) {
          nearest = Math.min(nearest, distanceToSegment(pole, line.points[i], line.points[i + 1]));
        }
        return nearest < plan.cuts[2] - 1e-6;
      });
      assert.ok(closeLines.length <= 4, "at " + width + " px, " + closeLines.length + " meridians come within the cut of the 2-hour class");
      assert.ok(closeLines.every((line) => line.strong), "and they are the 6-hour lines");
      assert.equal(closeLines.length, 4, "all four 6-hour lines reach the pole region");
    }
  });

  test("the lines get no denser than the gap says, even across the whole frame", () => {
    // Walk a circle around the pole at several radii and count the lines that cross it. The
    // crossings of a class cannot be closer than the minimum gap.
    const camera = cameraWithPole(2071.5, 1410.5);
    for (const width of WIDTHS) {
      const plan = planFor(camera, width);
      for (const meridian of plan.meridians) {
        for (const other of plan.meridians) {
          if (meridian === other) {
            continue;
          }
          const r = Math.max(meridian.startRadius, other.startRadius);
          const a = [r * meridian.dir[0], r * meridian.dir[1]];
          const b = [r * other.dir[0], r * other.dir[1]];
          assert.ok(Math.hypot(a[0] - b[0], a[1] - b[1]) >= plan.minGap - 1, "hours " + meridian.hour + " and " + other.hour + " at " + width);
        }
      }
    }
  });

  test("the text of a label takes the room that its characters need", () => {
    near(SkyGrid.labelWidth("89.75°", 11), 6 * 11 * 0.62, 1e-9, "width");
    const box = SkyGrid.labelBox("0h", "left", 100, 50, 11);
    near(box.x0, 97, 1e-9, "left edge");
    near(box.x1, 103 + 2 * 11 * 0.62, 1e-9, "right edge");
    near(box.y1 - box.y0, 11 + 6, 1e-9, "height");
    const right = SkyGrid.labelBox("0h", "right", 100, 50, 11);
    near(right.x1, 103, 1e-9, "right edge of a right-aligned label");
    const center = SkyGrid.labelBox("0h", "center", 100, 50, 11);
    near((center.x0 + center.x1) / 2, 100, 1e-9, "center");
  });
};
