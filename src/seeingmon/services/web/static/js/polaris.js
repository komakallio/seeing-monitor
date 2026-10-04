"use strict";

/*
 * The Polaris widget of the Now page: the camera frame, the pole, the circle that Polaris follows
 * around the pole, the trail of the latest solutions, and Polaris itself, which moves once a second
 * at the sidereal rate (15.04 degrees an hour), so that a glance shows that time passes and where
 * the star is. The fast stream's ROI follows it, so the widget draws the ROI too.
 *
 * The data are the solved pointing records of the last hours: the pixel of Polaris and, when the
 * record has it, the pixel of the pole. With the pole, the widget draws the orbit and moves Polaris
 * along it. Without the pole (an older record), it draws the trail and extrapolates the last
 * movement in a straight line, which is accurate for several minutes.
 */
(function () {
  // The HTML parser knows the namespace of SVG, so the file needs no URL for it. The text that it
  // parses is a constant.
  const SVG_NS = new DOMParser().parseFromString("<svg></svg>", "text/html").body.firstChild.namespaceURI;
  const SIDEREAL_RAD_PER_S = (2 * Math.PI) / 86164.0905;
  const EXTRAPOLATE_MAX_S = 20 * 60;
  const ROI_ARCMIN = 4.1;

  function el(tag, attrs, text) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [name, value] of Object.entries(attrs || {})) {
      node.setAttribute(name, String(value));
    }
    if (text !== undefined) {
      node.textContent = text;
    }
    return node;
  }

  /**
   * What the widget needs from the solved records (oldest first, each `{ t, x, y, px, py }` with
   * `t` in ms, `px` and `py` the pole or null): the position of the pole, the radius, and the
   * direction of the movement (+1 or -1 in the angle `atan2(dy, dx)` of the page).
   */
  function orbitFrom(points) {
    const withPole = points.filter((p) => p.px !== null && p.py !== null);
    if (withPole.length === 0) {
      return null;
    }
    const last = withPole[withPole.length - 1];
    const radius = Math.hypot(last.x - last.px, last.y - last.py);
    const angle = (p) => Math.atan2(p.y - p.py, p.x - p.px);
    let sense = -1; // counter-clockwise on the screen, the way the sky turns around the pole
    if (withPole.length >= 2) {
      const first = withPole[0];
      let delta = angle(last) - angle(first);
      while (delta > Math.PI) {
        delta -= 2 * Math.PI;
      }
      while (delta < -Math.PI) {
        delta += 2 * Math.PI;
      }
      if (Math.abs(delta) > 1e-4) {
        sense = delta > 0 ? 1 : -1;
      }
    }
    return { cx: last.px, cy: last.py, radius, sense, angle: angle(last), t: last.t };
  }

  /** The position of Polaris at `nowMs`, and whether it rests on a solution that is too old. */
  function positionAt(points, orbit, nowMs) {
    if (points.length === 0) {
      return null;
    }
    const last = points[points.length - 1];
    const ageS = Math.max(0, (nowMs - last.t) / 1000);
    if (orbit) {
      const theta = orbit.angle + orbit.sense * SIDEREAL_RAD_PER_S * ((nowMs - orbit.t) / 1000);
      return { x: orbit.cx + orbit.radius * Math.cos(theta), y: orbit.cy + orbit.radius * Math.sin(theta), ageS };
    }
    if (points.length >= 2) {
      const prev = points[points.length - 2];
      const dt = (last.t - prev.t) / 1000;
      if (dt >= 30) {
        const s = Math.min(ageS, EXTRAPOLATE_MAX_S);
        return { x: last.x + ((last.x - prev.x) / dt) * s, y: last.y + ((last.y - prev.y) / dt) * s, ageS };
      }
    }
    return { x: last.x, y: last.y, ageS };
  }

  /** Build the widget in `container`. `update(data)` gives it records, and `tick(nowMs)` moves Polaris. */
  function create(container) {
    const root = el("svg", { class: "polaris-svg", role: "img", "aria-label": "Polaris on its orbit around the celestial pole", preserveAspectRatio: "xMidYMid meet" });
    const frame = el("rect", { class: "pl-frame" });
    const orbitCircle = el("circle", { class: "pl-orbit" });
    const trail = el("polyline", { class: "pl-trail", fill: "none" });
    const dots = el("g", { class: "pl-dots" });
    const poleMark = el("path", { class: "pl-pole" });
    const poleLabel = el("text", { class: "pl-label" }, "pole");
    const roi = el("rect", { class: "pl-roi" });
    const halo = el("circle", { class: "pl-halo" });
    const star = el("circle", { class: "pl-star" });
    const starLabel = el("text", { class: "pl-label" }, "Polaris");
    root.append(frame, orbitCircle, trail, dots, poleMark, poleLabel, roi, halo, star, starLabel);
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "No pointing solution yet.";
    container.append(root, empty);
    root.style.display = "none";

    let data = null;
    let orbit = null;
    let unit = 1;

    function size() {
      const widthPx = container.clientWidth || 360;
      unit = data ? data.frame.width / widthPx : 1;
      const u = unit;
      for (const node of [frame]) {
        node.setAttribute("x", 0);
        node.setAttribute("y", 0);
        node.setAttribute("width", data.frame.width);
        node.setAttribute("height", data.frame.height);
      }
      star.setAttribute("r", 5 * u);
      halo.setAttribute("r", 11 * u);
      for (const label of [poleLabel, starLabel]) {
        label.setAttribute("font-size", 11 * u);
      }
      const side = Math.max(((ROI_ARCMIN * 60) / data.scale) * 1, 8 * u);
      roi.setAttribute("width", side);
      roi.setAttribute("height", side);
      const s = 7 * u;
      if (orbit) {
        poleMark.setAttribute("d", "M " + (orbit.cx - s) + " " + orbit.cy + " L " + (orbit.cx + s) + " " + orbit.cy + " M " + orbit.cx + " " + (orbit.cy - s) + " L " + orbit.cx + " " + (orbit.cy + s));
        poleLabel.setAttribute("x", orbit.cx + 9 * u);
        poleLabel.setAttribute("y", orbit.cy + 15 * u);
      }
      dots.replaceChildren(
        ...data.points.map((p) => el("circle", { class: "pl-dot", cx: p.x, cy: p.y, r: 2.5 * u }))
      );
      trail.setAttribute("points", data.points.map((p) => p.x + "," + p.y).join(" "));
    }

    function update(next) {
      data = next;
      if (!data || data.points.length === 0) {
        root.style.display = "none";
        empty.style.display = "";
        return;
      }
      empty.style.display = "none";
      root.style.display = "";
      root.setAttribute("viewBox", "0 0 " + data.frame.width + " " + data.frame.height);
      orbit = orbitFrom(data.points);
      orbitCircle.style.display = orbit ? "" : "none";
      poleMark.style.display = orbit ? "" : "none";
      poleLabel.style.display = orbit ? "" : "none";
      if (orbit) {
        orbitCircle.setAttribute("cx", orbit.cx);
        orbitCircle.setAttribute("cy", orbit.cy);
        orbitCircle.setAttribute("r", orbit.radius);
      }
      roi.style.display = data.fast ? "" : "none";
      size();
    }

    function tick(nowMs) {
      if (!data || data.points.length === 0) {
        return null;
      }
      const at = positionAt(data.points, orbit, nowMs);
      if (!at) {
        return null;
      }
      star.setAttribute("cx", at.x);
      star.setAttribute("cy", at.y);
      halo.setAttribute("cx", at.x);
      halo.setAttribute("cy", at.y);
      const side = Number(roi.getAttribute("width")) || 0;
      roi.setAttribute("x", at.x - side / 2);
      roi.setAttribute("y", at.y - side / 2);
      starLabel.setAttribute("x", at.x + 14 * unit);
      starLabel.setAttribute("y", at.y - 10 * unit);
      return at;
    }

    if (typeof ResizeObserver === "function") {
      new ResizeObserver(() => {
        if (data && data.points.length > 0) {
          size();
        }
      }).observe(container);
    }
    return { update, tick, get orbit() { return orbit; } };
  }

  window.Polaris = { create, orbitFrom, positionAt, SIDEREAL_RAD_PER_S };
})();
