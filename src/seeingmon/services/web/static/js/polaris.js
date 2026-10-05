"use strict";

/*
 * The live view of Polaris on the Now page: the ROI of the fast stream (128 x 128 pixels), stretched
 * by the server and magnified by the browser, as a video that flickers with the seeing.
 *
 * The server pushes about 20 frames a second over a WebSocket (`polaris/stream`) as lossless PNG.
 * The view shows the star zoomed in: it crops the central half or quarter of the ROI (the zoom
 * buttons choose), follows the star slowly, so that the dance of the star around its mean stays
 * visible, and magnifies the crop by a whole number with nearest-neighbor scaling, so the pixels of
 * the sensor stay square and sharp. `LiveLink` (live.js) keeps the connection: reconnects, the
 * polling fallback, and the close codes. A frame arrives as a state message and the image. The
 * state names the ROI, the exposure, the star, and the stretch. The view draws each frame as it
 * arrives, and it keeps the last frame under a cover when the fast stream pauses (a survey exposure
 * takes the camera for about 40 s of each cycle), so the cover says what happens instead of the
 * picture freezing without a word.
 *
 * The view stops the link while the tab is hidden, and the server stops the stream to the camera
 * process a few seconds later, so a forgotten tab costs nothing.
 */
(function () {
  const { h, api, Status, Token, LiveLink, fmt, watchAge, recall, remember } = window.Seeing;

  const MAX_SCALE = 16; // the most device pixels that one pixel of the sensor gets on a side
  const STALL_S = 2.5; // a frame older than this means that the video is not live
  const ZOOMS = [1, 2, 4]; // 1 shows the whole ROI, 2 the central half, 4 the central quarter
  const FOLLOW_TAU_S = 4; // how slowly the crop follows the mean position of the star

  function socketUrl() {
    const scheme = window.location.protocol === "https:" ? "wss" : "ws";
    return scheme + "://" + window.location.host + api.url("polaris/stream");
  }

  /** Read the newest frame, for the polling fallback. The state comes with the WebSocket only. */
  async function pollFrame(after) {
    try {
      const response = await api.raw("polaris/frame", { after }, { accept: "image/png" });
      if (response.status === 204) {
        return null;
      }
      return { blob: await response.blob(), state: null, seq: Number(response.headers.get("X-Frame-Seq")) || after };
    } catch (error) {
      if (error && error.status === 401) {
        return { denied: true };
      }
      throw error;
    }
  }

  /**
   * Build the view. `canvas`, `cover`, `tag`, `note`, `when`, and `zoomBox` are the elements of the
   * card. `fastRunning()` says whether the scheduler runs the fast stream now, `reason()` says in
   * words what it does instead, and `expect` is the function that `watchAge` takes for the age of
   * the newest frame.
   */
  function create(options) {
    const { canvas, cover, tag, note, when, zoomBox, fastRunning, reason, expect, onState } = options;
    const ctx = canvas.getContext("2d");
    const bitmapOk = typeof createImageBitmap === "function";
    let picture = null;
    let state = null;
    let frameMs = null; // the time of the newest frame on the server clock
    let linkName = "connecting";
    let rate = 0; // frames a second, smoothed
    let lastArrival = 0;
    let link = null;
    let wanted = true; // false while the page has stopped the link on purpose
    let ready = false;
    let zoom = Number(recall("polaris-zoom", "2"));
    if (!ZOOMS.includes(zoom)) {
      zoom = 2;
    }
    let center = null; // the mean position of the star, in pixels of the picture
    let centerAt = 0;

    async function decode(blob) {
      if (bitmapOk) {
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

    /** Move the center of the crop slowly toward the star, so that its dance stays visible. */
    function follow(frameState) {
      if (!picture) {
        return;
      }
      const star = frameState && frameState.star;
      if (!star || !star.found || star.x === null || star.y === null) {
        return;
      }
      const now = performance.now();
      if (center === null) {
        center = { x: star.x, y: star.y };
      } else {
        const alpha = 1 - Math.exp(-Math.min(1, (now - centerAt) / 1000) / FOLLOW_TAU_S);
        center.x += (star.x - center.x) * alpha;
        center.y += (star.y - center.y) * alpha;
      }
      centerAt = now;
    }

    /** The part of the picture to show, in whole pixels of the picture. */
    function crop() {
      const width = Math.max(8, Math.round(picture.width / zoom));
      const height = Math.max(8, Math.round(picture.height / zoom));
      const at = center || { x: picture.width / 2, y: picture.height / 2 };
      const x = Math.min(picture.width - width, Math.max(0, Math.round(at.x - width / 2)));
      const y = Math.min(picture.height - height, Math.max(0, Math.round(at.y - height / 2)));
      return { x, y, width, height };
    }

    /**
     * Draw the crop magnified by a whole number of device pixels, with no interpolation, so that
     * every pixel of the sensor shows as a square of the same size. The canvas keeps that size and
     * sits in the middle of the frame, and it grows or shrinks by whole steps when the card resizes.
     */
    function paint() {
      if (!picture) {
        return;
      }
      const part = crop();
      const ratio = window.devicePixelRatio || 1;
      const room = canvas.parentElement.clientWidth * ratio;
      const scale = Math.max(1, Math.min(MAX_SCALE, Math.floor(room / part.width)));
      const width = part.width * scale;
      const height = part.height * scale;
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
        canvas.style.width = width / ratio + "px";
        canvas.style.height = height / ratio + "px";
      }
      ctx.imageSmoothingEnabled = false; // a resize of the canvas resets it, so set it for every draw
      ctx.drawImage(picture, part.x, part.y, part.width, part.height, 0, 0, width, height);
    }

    async function show(blob, frameState) {
      const next = await decode(blob);
      if (picture && typeof picture.close === "function") {
        picture.close();
      }
      picture = next;
      const arrival = performance.now();
      if (lastArrival > 0) {
        const instant = 1000 / Math.max(1, arrival - lastArrival);
        rate = rate === 0 ? instant : rate * 0.9 + instant * 0.1;
      }
      lastArrival = arrival;
      if (frameState) {
        state = frameState;
        frameMs = Date.parse(frameState.t_utc);
      } else {
        frameMs = Status.nowMs();
      }
      follow(frameState);
      paint();
      if (onState && frameState) {
        onState(frameState);
      }
    }

    function ageS() {
      return frameMs === null ? null : Math.max(0, (Status.nowMs() - frameMs) / 1000);
    }

    function coverText() {
      switch (linkName) {
        case "denied":
          return "The server asks for the token. Enter it above.";
        case "busy":
          return "The server shows this view to as many viewers as it allows. It tries again.";
        case "unavailable":
          return "The web server cannot reach core.";
        case "connecting":
          return "Connecting";
        case "reconnecting":
          return "Reconnecting to the server";
        default:
          break;
      }
      const age = ageS();
      const why = reason();
      if (age === null) {
        return fastRunning() ? "Waiting for the first frame" : "No fast frames yet" + (why ? ": " + why : "");
      }
      if (age > STALL_S) {
        return fastRunning() ? "No frame for " + fmt.duration(age) : (why || "The fast stream is not running") + ". Last frame " + fmt.age(age) + ".";
      }
      return "";
    }

    function render() {
      const text = coverText();
      cover.hidden = text === "";
      cover.textContent = text;
      const live = text === "";
      tag.textContent = live ? "LIVE" + (rate > 0 ? " · " + Math.round(rate) + " fps" : "") : "NOT LIVE";
      tag.dataset.level = live ? "good" : "warn";
      watchAge(when, frameMs, expect, "frame ");
      if (!state) {
        note.textContent = "A live image of Polaris from the fast stream appears here while the stream runs.";
        return;
      }
      const roi = state.roi;
      const field = roi && state.scale_arcsec_px ? fmt.num((roi.width * state.scale_arcsec_px) / 60, 1) + "′ across" : null;
      const parts = [];
      if (roi) {
        parts.push("ROI " + roi.width + " × " + roi.height + " px" + (field ? " (" + field + ")" : ""));
      }
      if (state.exposure_us) {
        parts.push(fmt.num(state.exposure_us / 1000, 1) + " ms");
      }
      if (state.fast_fps) {
        parts.push("camera " + Math.round(state.fast_fps) + " fps");
      }
      let shown = "";
      if (picture) {
        const part = crop();
        shown = part.width === picture.width ? " The whole ROI is shown." : " Shown: " + part.width + " × " + part.height + " px around the star.";
      }
      const star = state.star;
      const starText = star && star.found
        ? "Star width " + fmt.arcsec(star.fwhm_arcsec, 1) + (star.peak_fraction !== null && star.peak_fraction !== undefined ? ", peak " + fmt.percent(star.peak_fraction, 0) + " of full scale" : "") + "."
        : "No star found in the newest frame.";
      note.textContent = parts.join(", ") + "." + shown + " " + starText;
    }

    function onLink(name) {
      linkName = name;
      render();
    }

    function createLink() {
      return new LiveLink({
        url: socketUrl,
        tokenRequired: () => Boolean(Status.current && Status.current.ui.token_required_for_reads),
        token: () => Token.get(),
        isActive: () => fastRunning(),
        staleMs: () => STALL_S * 1000,
        pollIntervalMs: () => 250,
        pollFrame,
        onFrame: (blob, frameState) => {
          show(blob, frameState).catch(() => undefined);
        },
        onLink,
      });
    }

    function visibility() {
      if (!link) {
        return;
      }
      if (document.hidden) {
        link.stop();
      } else if (wanted && link.stopped) {
        link.start();
      }
    }

    function buildZoom() {
      const titles = { 1: "Show the whole ROI", 2: "Show the central half of the ROI", 4: "Show the central quarter of the ROI" };
      for (const level of ZOOMS) {
        zoomBox.append(
          h("button", {
            type: "button",
            text: level + "×",
            title: titles[level],
            "aria-pressed": level === zoom ? "true" : "false",
            onclick: (event) => {
              zoom = level;
              remember("polaris-zoom", level);
              for (const button of zoomBox.children) {
                button.setAttribute("aria-pressed", button === event.currentTarget ? "true" : "false");
              }
              paint();
              render();
            },
          })
        );
      }
    }

    function init() {
      if (ready) {
        return;
      }
      ready = true;
      buildZoom();
      link = createLink();
      document.addEventListener("visibilitychange", visibility);
      if (typeof ResizeObserver === "function") {
        new ResizeObserver(() => paint()).observe(canvas.parentElement);
      }
    }

    return {
      /** Open the link. A page that shows the video only at times calls `stop` and `start` again. */
      start() {
        init();
        wanted = true;
        if (!document.hidden) {
          link.start();
        }
        render();
      },
      /** Close the link, so that no frame travels while the video is out of sight. */
      stop() {
        wanted = false;
        if (link) {
          link.stop();
        }
      },
      restart() {
        if (link && wanted && !document.hidden) {
          link.restart();
        }
      },
      /** Call it once a second: it notices that the frames stopped, and it keeps the texts current. */
      tick() {
        if (link) {
          link.tick();
        }
        render();
      },
    };
  }

  window.PolarisLive = { create };
})();
