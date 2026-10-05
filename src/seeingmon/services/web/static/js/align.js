"use strict";

/*
 * The Align page: the live view of the star field with the sky drawn over it, the cards that say
 * what the numbers mean, the focus value with its curve, a histogram of the pixel values, and a warning for
 * saturation. The page starts and stops the alignment (the commands need the token).
 *
 * The sky overlay has three parts that the person can switch off: the pole and the orbit of
 * Polaris (with the aim cross for a first alignment), the coordinate grid, and the target and
 * Polaris. The geometry comes from skygrid.js, which works in display pixels, and the sentences
 * come from aligntext.js. This file draws what they plan, and puts the text into the page.
 *
 * `LiveLink` (live.js) keeps the connection to the server: a WebSocket with reconnects, and the
 * polling of the newest frame as a fallback. This file draws what the link delivers. The page
 * closes the connection while its tab stays hidden.
 */
(function () {
  const { h, $, fmt, api, ApiError, Status, Token, poller, LiveLink, recall, remember, SkyGrid, AlignText, FocusCurve } = window.Seeing;

  const HIDDEN_CLOSE_MS = 20000;
  const FAMILY = "system-ui, sans-serif";
  const MAX_BACKING_PX = 2048; // the widest canvas that the page makes, in device pixels
  const FONT_PX = 11;
  // The pole counts as aligned when it is this close to the aim, in arcminutes.
  const ALIGNED_ARCMIN = 2;

  // The overlays that the person can switch off. The choice stays in this browser.
  const OVERLAYS = [
    { key: "pole", id: "overlay-pole" },
    { key: "grid", id: "overlay-grid" },
    { key: "target", id: "overlay-target" },
  ];

  const view = {
    canvas: null,
    ctx: null,
    hist: null,
    show: { pole: true, grid: true, target: true },
    lastState: null,
    picture: null,
    bitmapOk: typeof createImageBitmap === "function",
  };
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
    for (const { key, id } of OVERLAYS) {
      const box = $(id);
      view.show[key] = recall("align." + key, "1") !== "0";
      box.checked = view.show[key];
      box.addEventListener("change", () => {
        view.show[key] = box.checked;
        remember("align." + key, box.checked ? "1" : "0");
        paint();
      });
    }
    window.addEventListener("seeing:theme", () => {
      paint();
      drawHistogram(view.lastState);
    });
    if (typeof ResizeObserver === "function") {
      new ResizeObserver(() => drawHistogram(view.lastState)).observe(view.hist.parentElement);
      new ResizeObserver(() => requestAnimationFrame(paint)).observe(view.canvas);
    }
    clearView();
  }

  function clearView() {
    const { canvas, ctx } = view;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
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
    if (view.picture && typeof view.picture.close === "function") {
      view.picture.close();
    }
    view.picture = picture;
    view.lastState = state;
    paint();
    renderState(state);
    updateCover();
  }

  /**
   * Draw the picture and the overlays. The canvas gets the size of the picture on the screen in
   * device pixels (at most `MAX_BACKING_PX` wide), so the lines and the text stay sharp whatever
   * the size of the JPEG that the server sent.
   */
  function paint() {
    const { canvas, ctx, picture } = view;
    if (!picture) {
      return;
    }
    const shown = canvas.getBoundingClientRect().width;
    const width = shown > 0 ? Math.min(MAX_BACKING_PX, Math.max(2, Math.round(shown * (window.devicePixelRatio || 1)))) : picture.width;
    const height = Math.max(2, Math.round((width * picture.height) / picture.width));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(picture, 0, 0, width, height);
    drawOverlays(view.lastState);
  }

  // The marks. Sizes are in CSS pixels, because the overlay draws under a scale that makes one
  // unit one CSS pixel of the shown image, and so the marks keep their size on every screen.

  function ring(ctx, x, y, radius) {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.stroke();
  }

  /** A cross with a gap in the middle. */
  function cross(ctx, x, y, gap, arm) {
    ctx.beginPath();
    ctx.moveTo(x - arm, y);
    ctx.lineTo(x - gap, y);
    ctx.moveTo(x + gap, y);
    ctx.lineTo(x + arm, y);
    ctx.moveTo(x, y - arm);
    ctx.lineTo(x, y - gap);
    ctx.moveTo(x, y + gap);
    ctx.lineTo(x, y + arm);
    ctx.stroke();
  }

  function arrowHead(ctx, x, y, angle, size) {
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x - Math.cos(angle - 0.45) * size, y - Math.sin(angle - 0.45) * size);
    ctx.lineTo(x - Math.cos(angle + 0.45) * size, y - Math.sin(angle + 0.45) * size);
    ctx.closePath();
    ctx.fill();
  }

  /** A line of text with a dark halo under it, so that it reads on stars and on the grid. */
  function drawLabel(ctx, label, color, halo) {
    ctx.textAlign = label.align;
    ctx.lineWidth = 3;
    ctx.strokeStyle = halo;
    ctx.strokeText(label.text, label.x, label.y);
    ctx.fillStyle = color;
    ctx.fillText(label.text, label.x, label.y);
  }

  function polyline(ctx, points) {
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i += 1) {
      ctx.lineTo(points[i][0], points[i][1]);
    }
    ctx.stroke();
  }

  const ORBIT_TOKENS = { good: "--overlay-orbit-good", tight: "--overlay-orbit-warn", bad: "--overlay-orbit-bad", none: "--overlay-orbit-good" };

  function drawOverlays(state) {
    if (!state || !state.frame) {
      return;
    }
    const { canvas, ctx, show } = view;
    const frame = state.frame;
    const sky = state.sky || null;
    const shown = canvas.getBoundingClientRect().width || canvas.width;
    const scale = shown / frame.width_px; // display pixels of one frame pixel
    const k = canvas.width / shown; // canvas pixels of one CSS pixel
    const at = (x, y) => [(x + 0.5) * scale, (y + 0.5) * scale];
    const orbitState = AlignText.orbitState(sky, frame);
    const colors = {
      grid: token("--overlay-grid", "#a9b9d6"),
      halo: token("--overlay-halo", "#000000"),
      pole: token("--overlay-pole", "#6fd3ff"),
      aim: token("--overlay-aim", "#f4f4f4"),
      orbit: token(ORBIT_TOKENS[orbitState], "#4be08f"),
      target: token("--overlay-target", "#4be08f"),
      solved: token("--overlay-solved", "#ff6b5e"),
    };
    const gridAlpha = Math.min(1, Math.max(0.05, parseFloat(token("--overlay-grid-alpha", "0.45")) || 0.45));

    const markers = [];
    const target = show.target && state.target ? at(state.target.x_px, state.target.y_px) : null;
    const solved = show.target && state.solved ? at(state.solved.x_px, state.solved.y_px) : null;
    if (solved) {
      markers.push({ id: "polaris", text: "Polaris", x: solved[0], y: solved[1], radius: 10, reserve: 14 });
    }
    if (target) {
      markers.push({ id: "target", text: "target", x: target[0], y: target[1], radius: 16, reserve: 30 });
    }
    const showAim = show.pole && !state.target;
    const plan = sky
      ? SkyGrid.plan(sky.camera, { width: frame.width_px, height: frame.height_px }, scale, {
          colatitudeDeg: sky.polaris_colatitude_deg === undefined ? null : sky.polaris_colatitude_deg,
          grid: show.grid,
          orbitText: "",
          poleText: show.pole ? "pole" : "",
          poleArrowText: show.pole ? (side) => AlignText.poleArrowText(sky, side) : null,
          aimText: "",
          markers,
          fontPx: FONT_PX,
        })
      : null;
    const labelFor = (id) => (plan ? plan.labels.find((label) => label.id === id) : undefined);

    ctx.save();
    ctx.setTransform(k, 0, 0, canvas.height / (frame.height_px * scale), 0, 0);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.font = "600 " + FONT_PX + "px " + FAMILY;
    ctx.textBaseline = "middle";

    // 1. The coordinate grid.
    if (plan && show.grid) {
      ctx.strokeStyle = colors.grid;
      for (const line of plan.lines) {
        ctx.globalAlpha = line.strong ? Math.min(1, gridAlpha + 0.2) : gridAlpha;
        ctx.lineWidth = line.strong ? 1.4 : 1;
        polyline(ctx, line.points);
      }
      ctx.globalAlpha = 1;
      for (const label of plan.labels) {
        if (label.kind === "meridian" || label.kind === "ring") {
          drawLabel(ctx, label, colors.grid, colors.halo);
        }
      }
    }

    // 2. The orbit of Polaris, as the fixed reticle: a dashed circle at the aim with the radius of the
    // orbit (from the server, so it shows without a solution and follows a configured aim pixel), a
    // small cross at its center, and the aim ring on it, where Polaris belongs for this frame (the
    // detected Polaris plus the aim minus the pole, because alt-az moves translate the image). The
    // reticle turns green and the arrow goes away when the pole is within 2 arcminutes of the aim.
    let aimRing = null;
    let circle = null; // { x_px, y_px, radius_px } in frame pixels
    if (state.reticle) {
      circle = state.reticle;
    } else if (sky && sky.polaris_colatitude_deg !== null && sky.polaris_colatitude_deg !== undefined) {
      const camera = sky.camera; // an older core: the circle around the principal point
      circle = {
        x_px: camera.center_x_px,
        y_px: camera.center_y_px,
        radius_px: Math.tan((sky.polaris_colatitude_deg * Math.PI) / 180) / ((camera.scale_arcsec_px * Math.PI) / 648000),
      };
    }
    let ringFrame = sky && sky.aim_ring ? sky.aim_ring : null;
    // Without a solution of the current frame, `core` places the ring from the last good solution.
    const fromLast = !ringFrame && Boolean(state.aim_ring) && state.aim_ring.source === "last solution";
    if (!ringFrame && state.aim_ring) {
      ringFrame = state.aim_ring;
    }
    if (!ringFrame && circle && sky && state.solved && sky.pole && sky.pole.x_px !== null && sky.pole.x_px !== undefined) {
      ringFrame = {
        x_px: circle.x_px + (state.solved.x_px - sky.pole.x_px),
        y_px: circle.y_px + (state.solved.y_px - sky.pole.y_px),
      };
    }
    const aimDistance = sky && sky.aim && sky.aim.distance_arcmin !== null && sky.aim.distance_arcmin !== undefined
      ? sky.aim.distance_arcmin
      : sky && sky.pole ? sky.pole.distance_arcmin : null;
    const aligned = aimDistance !== null && aimDistance !== undefined && aimDistance < ALIGNED_ARCMIN;
    if (circle && show.pole) {
      const middle = at(circle.x_px, circle.y_px);
      const radius = circle.radius_px * scale;
      ctx.save();
      ctx.globalAlpha = sky ? 1 : 0.5; // dimmed until there is a solution
      ctx.strokeStyle = aligned ? colors.target : colors.aim;
      ctx.lineWidth = aligned ? 2.6 : 2;
      ctx.setLineDash([7, 5]);
      ctx.beginPath();
      ctx.arc(middle[0], middle[1], radius, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      cross(ctx, middle[0], middle[1], 2, 9);
      ctx.restore();
      if (!sky) {
        drawLabel(ctx, { text: fromLast ? "no current solution" : "no solution yet", x: middle[0], y: middle[1] + 24, align: "center" }, colors.aim, colors.halo);
      }
      if (ringFrame) {
        aimRing = at(ringFrame.x_px, ringFrame.y_px);
        const ringColor = fromLast ? token("--overlay-orbit-warn", "#ffc233") : colors.target;
        ctx.strokeStyle = ringColor;
        ctx.fillStyle = ringColor;
        ctx.lineWidth = 2.2;
        if (fromLast) {
          ctx.setLineDash([5, 4]);
        }
        ring(ctx, aimRing[0], aimRing[1], 16);
        ctx.setLineDash([]);
        const ringText = fromLast && state.aim_ring.age_s !== null && state.aim_ring.age_s !== undefined ? "aim, " + fmt.num(state.aim_ring.age_s, 0) + " s old" : "aim";
        // The label sits to the right of the ring, and to the left when it would leave the picture.
        const leftSide = aimRing[0] + 20 + ctx.measureText(ringText).width > frame.width_px * scale - 4;
        drawLabel(ctx, { text: ringText, x: aimRing[0] + (leftSide ? -20 : 20), y: aimRing[1] - 24, align: leftSide ? "right" : "left" }, ringColor, colors.halo);
        if (solved && !aligned) {
          const distance = Math.hypot(solved[0] - aimRing[0], solved[1] - aimRing[1]);
          if (distance > 34) {
            const heading = Math.atan2(aimRing[1] - solved[1], aimRing[0] - solved[0]);
            ctx.strokeStyle = colors.solved;
            ctx.fillStyle = colors.solved;
            ctx.lineWidth = 1.8;
            ctx.beginPath();
            ctx.moveTo(solved[0] + Math.cos(heading) * 13, solved[1] + Math.sin(heading) * 13);
            ctx.lineTo(aimRing[0] - Math.cos(heading) * 20, aimRing[1] - Math.sin(heading) * 20);
            ctx.stroke();
            arrowHead(ctx, aimRing[0] - Math.cos(heading) * 20, aimRing[1] - Math.sin(heading) * 20, heading, 9);
          }
        }
      }
    }

    // 3. The pole: a ring with a cross, or an arrow at the edge when it lies outside the frame.
    if (plan && show.pole && plan.pole) {
      ctx.strokeStyle = colors.pole;
      ctx.fillStyle = colors.pole;
      ctx.lineWidth = 1.8;
      if (plan.pole.inside) {
        ring(ctx, plan.pole.x, plan.pole.y, 8);
        cross(ctx, plan.pole.x, plan.pole.y, 3, 15);
      } else if (plan.arrow) {
        const { x, y, angle } = plan.arrow;
        ctx.beginPath();
        ctx.moveTo(x - Math.cos(angle) * 22, y - Math.sin(angle) * 22);
        ctx.lineTo(x - Math.cos(angle) * 8, y - Math.sin(angle) * 8);
        ctx.stroke();
        arrowHead(ctx, x, y, angle, 12);
      }
      const label = labelFor("pole");
      if (label) {
        drawLabel(ctx, label, colors.pole, colors.halo);
      }
    }

    // 4. The aim: where the pole belongs, the center of the reticle (the frame center unless the
    // configuration names another pixel). Its cross is drawn with the reticle above, and a page
    // that has no reticle yet (an older core) still gets a cross at the middle of the frame.
    if (showAim && !circle) {
      ctx.strokeStyle = colors.aim;
      ctx.lineWidth = 1.6;
      cross(ctx, (frame.width_px * scale) / 2, (frame.height_px * scale) / 2, 2, 9);
    }

    // 5. Polaris, wherever the solver finds it, with or without a target.
    if (solved) {
      ctx.strokeStyle = colors.solved;
      ctx.fillStyle = colors.solved;
      ctx.lineWidth = 1.8;
      ring(ctx, solved[0], solved[1], 10);
      drawLabel(ctx, labelFor("polaris") || { text: "Polaris", x: solved[0] + 14, y: solved[1] + 8, align: "left" }, colors.solved, colors.halo);
    }

    // 6. The target (a cross and a ring), and an arrow from it to Polaris.
    if (target) {
      ctx.strokeStyle = colors.target;
      ctx.fillStyle = colors.target;
      ctx.lineWidth = 1.8;
      cross(ctx, target[0], target[1], 6, 28);
      ring(ctx, target[0], target[1], 16);
      if (solved) {
        const distance = Math.hypot(solved[0] - target[0], solved[1] - target[1]);
        if (distance > 30) {
          const angle = Math.atan2(solved[1] - target[1], solved[0] - target[0]);
          ctx.strokeStyle = colors.solved;
          ctx.fillStyle = colors.solved;
          ctx.beginPath();
          ctx.moveTo(target[0] + Math.cos(angle) * 18, target[1] + Math.sin(angle) * 18);
          ctx.lineTo(solved[0] - Math.cos(angle) * 12, solved[1] - Math.sin(angle) * 12);
          ctx.stroke();
          arrowHead(ctx, solved[0] - Math.cos(angle) * 12, solved[1] - Math.sin(angle) * 12, angle, 8);
        }
      }
      drawLabel(ctx, labelFor("target") || { text: "target", x: target[0] + 20, y: target[1] - 26, align: "left" }, colors.target, colors.halo);
    }

    // 7. The two moves of the mount, as a small compass in the lower left corner: where the camera
    // looks when you raise it (alt +) and when you turn it toward the east (az +). The pole lies on
    // the side of the aim in which the camera has to move, so the arrows read against the picture.
    if (sky && sky.axes && show.pole) {
      const width = frame.width_px * scale;
      const height = frame.height_px * scale;
      const reach = Math.min(26, Math.max(16, 0.07 * width));
      const origin = [14 + reach + 36, height - 14 - reach - 14];
      const arms = [
        { text: "alt +", dx: sky.axes.altitude_dx, dy: sky.axes.altitude_dy },
        { text: "az +", dx: sky.axes.azimuth_dx, dy: sky.axes.azimuth_dy },
      ];
      for (const arm of arms) {
        ctx.strokeStyle = colors.aim; // drawLabel leaves the halo style behind, so set it per arm
        ctx.fillStyle = colors.aim;
        ctx.lineWidth = 2;
        const tip = [origin[0] + arm.dx * reach, origin[1] + arm.dy * reach];
        ctx.beginPath();
        ctx.moveTo(origin[0], origin[1]);
        ctx.lineTo(tip[0], tip[1]);
        ctx.stroke();
        arrowHead(ctx, tip[0], tip[1], Math.atan2(arm.dy, arm.dx), 8);
        const side = arm.dx > 0.35 ? "left" : arm.dx < -0.35 ? "right" : "center";
        drawLabel(ctx, { text: arm.text, x: tip[0] + arm.dx * 6, y: tip[1] + arm.dy * 10, align: side }, colors.aim, colors.halo);
      }
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

  const ORBIT_LEVELS = { good: "good", tight: "warn", bad: "bad" };

  /** What the pole card says about the last good solution while the solver finds none. */
  function lastSolutionSentence(state) {
    const last = state.last_solution;
    if (state.solved || !last || last.age_s === null || last.age_s === undefined) {
      return "";
    }
    return " The aim ring comes from the last good solution, " + fmt.num(last.age_s, 0) + " s old. A move of a degree in altitude and azimuth shifts the right place of the ring by a few pixels at most, so you can still aim with it.";
  }

  /** The pole card: where the pole is, and whether the orbit of Polaris fits in the frame. */
  function renderPole(state) {
    const sky = state.sky || null;
    const sentence = $("pole-sentence");
    const orbit = $("orbit-sentence");
    if (!sky) {
      sentence.textContent = AlignText.poleReason(state) + lastSolutionSentence(state);
      orbit.textContent = "";
      delete orbit.dataset.level;
      view.canvas.setAttribute("aria-label", "Live view of the star field");
      return;
    }
    sentence.textContent = AlignText.poleSentence(sky);
    const hint = $("adjust-note");
    hint.textContent = AlignText.siteNote(sky);
    hint.hidden = hint.textContent === "";
    orbit.textContent = AlignText.orbitSentences(sky, state.frame).join(" ");
    const level = ORBIT_LEVELS[AlignText.orbitState(sky, state.frame)];
    if (level) {
      orbit.dataset.level = level;
    } else {
      delete orbit.dataset.level;
    }
    view.canvas.setAttribute("aria-label", "Live view of the star field. " + sentence.textContent);
  }

  /** The offset card: three states, with no solution, with a solution only, and with a target. */
  function renderOffset(state) {
    const card = AlignText.offsetCard(state);
    setFacts("offset-facts", card.rows);
    const note = $("offset-note");
    note.textContent = card.note;
    note.hidden = card.note === "";
    $("roll-block").hidden = !card.roll;
    if (card.roll) {
      $("roll-note").textContent = card.roll.text;
    }
    const settings = AlignText.targetSettings(state.solved);
    const toml = $("target-toml");
    const text = settings || "No solution yet, so there is nothing to copy.";
    // A new frame changes the last digits. Leave the text alone while the person has it selected,
    // so that a copy with Ctrl+C gets what they see.
    if (toml.textContent !== text && !selectionIn(toml)) {
      toml.textContent = text;
    }
    $("copy-target").disabled = settings === "";
  }

  function renderState(state) {
    const solved = state.solved;
    const frame = state.frame;
    renderPole(state);
    renderOffset(state);
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

  // The points of the focus history that the live view has sent, and the curve that draws them.
  const focusHeld = { session: null, points: [] };
  let focusCurve = null;

  /** The words about the value, so that the state never rests on a color alone. */
  function focusVerdict(focus, arcsec, best) {
    if (focus.spike) {
      return "This value is a spike: the stars widened for a moment, as they do when you touch the telescope. The curve leaves it out of its scale.";
    }
    const value = arcsec ? focus.fwhm_arcsec : focus.fwhm_px;
    if (!best || !value) {
      return "";
    }
    const wider = Math.round((value / best - 1) * 100);
    if (value / best <= 1.05) {
      return "At the best value of this session.";
    }
    return value / best <= 1.2 ? "Close to the best value: " + wider + " % wider." : "Wider than the best value by " + wider + " %.";
  }

  function renderFocus(focus) {
    if (focus && focus.history) {
      const next = FocusCurve.merge(focusHeld, focus.history);
      focusHeld.session = next.session;
      focusHeld.points = next.points;
    }
    const value = $("focus-value");
    const sub = $("focus-sub");
    const note = $("focus-note");
    const arcsec = Boolean(focus) && focus.fwhm_arcsec !== null && focus.fwhm_arcsec !== undefined;
    const best = focus ? (arcsec ? focus.best_fwhm_arcsec : focus.best_fwhm_px) : null;
    if (focusCurve) {
      focusCurve.update(focusHeld.points, best === undefined ? null : best);
      $("focus-from").textContent = focusHeld.points.length > 1 ? "\u2212" + Math.round(focusCurve.spanS()) + " s" : "";
    }
    if (!focus || focus.fwhm_px === null || focus.fwhm_px === undefined) {
      value.textContent = fmt.dash;
      value.classList.remove("spike");
      sub.replaceChildren();
      note.textContent = focusHeld.points.length > 0 ? "The newest frame has too few usable stars for a value. It needs 3." : "No focus measure yet.";
      return;
    }
    value.textContent = arcsec ? fmt.num(focus.fwhm_arcsec, 1) + "\u2033" : fmt.num(focus.fwhm_px, 2) + " px";
    value.classList.toggle("spike", Boolean(focus.spike));
    const stars = focus.n_stars === null || focus.n_stars === undefined ? null : focus.n_stars;
    sub.replaceChildren(
      h("strong", { text: (stars === null ? "?" : stars) + (stars === 1 ? " star" : " stars") }),
      " measured" + (arcsec ? ", " + fmt.num(focus.fwhm_px, 2) + " px" : "") + (best ? ", best " + (arcsec ? fmt.num(best, 1) + "\u2033" : fmt.num(best, 2) + " px") : "")
    );
    const few = stars !== null && stars < 8 ? " Few stars make the value less reliable." : "";
    note.textContent = (focusVerdict(focus, arcsec, best) + " A narrower star is better." + few).trim();
  }

  async function resetFocusBest() {
    const note = $("focus-reset-note");
    if (!Token.has()) {
      note.textContent = "Enter the token first: the commands need it.";
      window.Seeing.openTokenPanel();
      return;
    }
    try {
      await api.post("alignment/focus/reset", {});
      note.textContent = "The best value restarts with the next frame.";
    } catch (error) {
      if (error.status === 429) {
        note.textContent = "Too many requests. Wait " + (error.retryAfter || 60) + " s and try again.";
      } else if (error.status === 401) {
        note.textContent = "The token is not right.";
      } else if (error.status === 503) {
        note.textContent = "Core does not answer, so nothing changed.";
      } else {
        note.textContent = error.message;
      }
    }
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

  // --- The target settings ----------------------------------------------------------------------

  /** Whether the person has selected text inside `node`. */
  function selectionIn(node) {
    const selection = window.getSelection();
    return Boolean(selection && selection.rangeCount > 0 && !selection.isCollapsed && node.contains(selection.anchorNode));
  }

  function selectText(node) {
    const range = document.createRange();
    range.selectNodeContents(node);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
  }

  /**
   * Copy the target settings. `navigator.clipboard` exists only on a secure page (https, or a
   * loopback address), and the page often runs on a LAN address over plain HTTP. There the text
   * stays selected, and the old copy command (which works from a click) tries once more.
   */
  async function copyTarget() {
    const node = $("target-toml");
    const note = $("copy-note");
    const text = node.textContent;
    if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
      try {
        await navigator.clipboard.writeText(text);
        note.textContent = "Copied.";
        return;
      } catch (error) {
        /* The browser refused. Select the text below. */
      }
    }
    selectText(node);
    let copied = false;
    try {
      copied = document.execCommand("copy");
    } catch (error) {
      copied = false;
    }
    note.textContent = copied ? "Copied." : "The text is selected. Press Ctrl+C to copy it.";
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
    $("copy-target").addEventListener("click", () => {
      copyTarget();
    });
    $("focus-reset").addEventListener("click", () => {
      resetFocusBest();
    });
    window.addEventListener("seeing:status", () => {
      const enabled = commandsEnabled();
      $("start").disabled = !enabled;
      $("focus-reset").disabled = !enabled;
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
    focusCurve = FocusCurve.create($("focus-curve"));
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
