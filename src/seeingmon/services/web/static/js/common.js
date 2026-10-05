"use strict";

/*
 * Shared helpers of the pages: the DOM, the API client, the formatting of values, the token, the
 * red night mode, the frame of a page (header, tabs, footer), and polling.
 *
 * The pages load this file first and use the global `Seeing`. There is no library and no external
 * asset. Every text goes into the page as a text node, never as markup.
 */
(function () {
  const API = "/api/v1";
  const PAGES = [
    { id: "now", href: "./", label: "Now" },
    { id: "history", href: "history.html", label: "History" },
    { id: "images", href: "images.html", label: "Images" },
    { id: "align", href: "align.html", label: "Align" },
    { id: "dark", href: "dark.html", label: "Dark" },
    { id: "flat", href: "flat.html", label: "Flat" },
  ];
  const NIGHT_KEY = "seeingmon.night";
  const TOKEN_KEY = "seeingmon.token";
  const EM_DASH = "—";
  const MINUS = "−";
  const ARCSEC = "″";

  // --- Storage that may be missing or blocked ---------------------------------------------------

  // `name` is "localStorage" or "sessionStorage". Even reading the property can throw when the
  // browser blocks site data, so the lookup sits inside the try block too.
  function readStorage(name, key) {
    try {
      return window[name].getItem(key);
    } catch (error) {
      return null;
    }
  }

  function writeStorage(name, key, value) {
    try {
      if (value === null) {
        window[name].removeItem(key);
      } else {
        window[name].setItem(key, value);
      }
    } catch (error) {
      /* Private windows and blocked site data throw. The page works without storage. */
    }
  }

  // A choice that a page remembers in this browser, such as which overlays to show. It lives in
  // the local storage under the prefix below, and a blocked or missing storage means the default.
  const PREFERENCE_PREFIX = "seeingmon.pref.";

  /** The remembered text for `name`, or `fallback` when there is none. */
  function recall(name, fallback) {
    const value = readStorage("localStorage", PREFERENCE_PREFIX + name);
    return value === null ? fallback : value;
  }

  /** Remember `value` (text) for `name`. A `null` forgets it. */
  function remember(name, value) {
    writeStorage("localStorage", PREFERENCE_PREFIX + name, value === null ? null : String(value));
  }

  // --- The DOM ----------------------------------------------------------------------------------

  /** Make an element. `props` may hold class, text, dataset, on-handlers, and attributes. */
  function h(tag, props, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props || {})) {
      if (value === undefined || value === null || value === false) {
        continue;
      }
      if (key === "class") {
        node.className = value;
      } else if (key === "text") {
        node.textContent = value;
      } else if (key === "dataset") {
        Object.assign(node.dataset, value);
      } else if (key.startsWith("on") && typeof value === "function") {
        node.addEventListener(key.slice(2), value);
      } else {
        node.setAttribute(key, value === true ? "" : String(value));
      }
    }
    for (const child of children.flat()) {
      if (child === null || child === undefined || child === false) {
        continue;
      }
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  function $(id) {
    return document.getElementById(id);
  }

  function clear(node) {
    while (node.firstChild) {
      node.removeChild(node.firstChild);
    }
    return node;
  }

  function emit(name, detail) {
    window.dispatchEvent(new CustomEvent(name, { detail }));
  }

  // --- Formatting -------------------------------------------------------------------------------

  const fmt = {
    dash: EM_DASH,

    /** A number with a fixed count of digits, or a dash for a missing value. */
    num(value, digits) {
      if (value === null || value === undefined || Number.isNaN(value)) {
        return EM_DASH;
      }
      const text = Number(value).toFixed(digits === undefined ? 1 : digits);
      return text.replace("-", MINUS);
    },

    signed(value, digits) {
      if (value === null || value === undefined || Number.isNaN(value)) {
        return EM_DASH;
      }
      const text = Number(value).toFixed(digits === undefined ? 1 : digits);
      return value > 0 ? "+" + text : text.replace("-", MINUS);
    },

    arcsec(value, digits) {
      const text = fmt.num(value, digits === undefined ? 2 : digits);
      return text === EM_DASH ? text : text + ARCSEC;
    },

    percent(fraction, digits) {
      if (fraction === null || fraction === undefined) {
        return EM_DASH;
      }
      return (fraction * 100).toFixed(digits === undefined ? 0 : digits) + " %";
    },

    /** Seconds as a short age: "12 s", "3 min", "2.4 h", "1.5 d". */
    duration(seconds) {
      if (seconds === null || seconds === undefined || Number.isNaN(seconds)) {
        return EM_DASH;
      }
      const s = Math.max(0, seconds);
      if (s < 90) {
        return Math.round(s) + " s";
      }
      if (s < 5400) {
        return Math.round(s / 60) + " min";
      }
      if (s < 129600) {
        return (s / 3600).toFixed(1) + " h";
      }
      return (s / 86400).toFixed(1) + " d";
    },

    age(seconds) {
      const text = fmt.duration(seconds);
      return text === EM_DASH ? text : text + " ago";
    },

    /** "03:00:00" from an ISO time, in UTC. */
    clock(iso) {
      return iso ? iso.slice(11, 19) : EM_DASH;
    },

    /** "2026-10-01 03:00" from an ISO time, in UTC. */
    stamp(iso) {
      return iso ? iso.slice(0, 10) + " " + iso.slice(11, 16) : EM_DASH;
    },

    bytes(count) {
      if (count === null || count === undefined) {
        return EM_DASH;
      }
      if (count < 1024) {
        return count + " B";
      }
      if (count < 1024 * 1024) {
        return (count / 1024).toFixed(0) + " kB";
      }
      return (count / 1024 / 1024).toFixed(1) + " MB";
    },

    /** A value of the status words as a level for the pills: good, warn, or bad. */
    level(word) {
      if (word === "healthy" || word === "ok" || word === "info") {
        return "good";
      }
      if (word === "degraded" || word === "warning") {
        return "warn";
      }
      return "bad";
    },
  };

  // --- Words for the codes of the API -----------------------------------------------------------

  const FLAG_HELP = {
    cloud: "Clouds crossed the star.",
    twilight: "The Sun was less than 18 degrees below the horizon, so sky light can skew the data.",
    vibration: "The spectrum shows vibration lines, so the image motion can read high.",
    saturated: "The star saturated, which biases the centroid and the width.",
    partial: "The window ended early.",
    degraded: "The system dropped more than 5 % of the expected frames.",
    time_invalid: "The clock was not synchronized, so the time is not trustworthy.",
    heater_on: "The dew heater was on, so heater plumes can add turbulence.",
    moon: "The Moon was up, so the sky is brighter.",
    dew: "Dew covered the optics.",
    dark_due: "The dark library is due for a new session.",
    saturated_sky: "The sky saturated the survey frame, so it gives no photometry and no clouds.",
    unsolved: "The solver found no solution.",
    few_stars: "The solver matched few stars.",
    moved: "The mount moved since the reference.",
    roll_undefined: "The roll could not be determined.",
    low_space: "Free disk space is low.",
    sink_backlog: "A sink has a backlog of results.",
  };

  const BAD_FLAGS = new Set(["saturated", "time_invalid", "unsolved"]);

  /** A chip element for a flag, with its meaning as a tooltip. */
  function flagChip(flag) {
    return h("span", {
      class: "chip",
      text: flag.replace(/_/g, " "),
      title: FLAG_HELP[flag] || flag,
      dataset: { level: BAD_FLAGS.has(flag) ? "bad" : "warn" },
    });
  }

  /** A sentence for a reason of the health verdict. */
  function explainReason(reason) {
    const [kind, name] = reason.split(":");
    switch (kind) {
      case "store_unreadable":
        return "The store cannot be read.";
      case "no_health_record":
        return "Core has not written a health record yet.";
      case "health_stale":
        return "The newest health record is old, so core may have stopped.";
      case "component_degraded":
        return "The component " + name + " is degraded.";
      case "component_failed":
        return "The component " + name + " has failed.";
      case "system_degraded":
        return "The system reports that it is degraded.";
      case "flag":
        return FLAG_HELP[name] || "The flag " + name + " is set.";
      case "core_unreachable":
        return "The web process cannot reach core.";
      default:
        return reason;
    }
  }

  // --- The token --------------------------------------------------------------------------------

  const Token = {
    memory: "",

    get() {
      return readStorage("sessionStorage", TOKEN_KEY) || Token.memory;
    },

    set(value) {
      const clean = (value || "").trim();
      Token.memory = clean;
      writeStorage("sessionStorage", TOKEN_KEY, clean || null);
      emit("seeing:token", { present: clean !== "" });
    },

    has() {
      return Token.get() !== "";
    },
  };

  // --- The API client ---------------------------------------------------------------------------

  class ApiError extends Error {
    constructor(status, code, message, retryAfter, details) {
      super(message);
      this.name = "ApiError";
      this.status = status;
      this.code = code;
      this.retryAfter = retryAfter;
      this.details = details || null; // one entry for each invalid field of a request
    }
  }

  /**
   * The code, the message, and the details of a failed response. `data` is the decoded body, or
   * `null` when the body was not JSON. An error of the API looks like {"error": {"code",
   * "message", "details"}}. A command that the scheduler rejects (status 409) has the shape of a
   * command answer instead: {"accepted": false, "message", "reason", "state"}. The message says
   * what the person can do ("the scheduler is paused; resume it first"), so the page shows it, and
   * the reason is the code. Any other body gives a message with the status only.
   */
  function failureFrom(status, data) {
    let code = "error";
    let message = "The request failed (" + status + ").";
    let details = null;
    if (data && data.error) {
      code = data.error.code || code;
      message = data.error.message || message;
      details = Array.isArray(data.error.details) ? data.error.details : null;
    } else if (data && data.accepted === false && typeof data.message === "string" && data.message !== "") {
      code = typeof data.reason === "string" && data.reason !== "" ? data.reason : "rejected";
      message = data.message;
    }
    return { code, message, details };
  }

  async function request(method, path, options) {
    const opts = options || {};
    const url = new URL(path.startsWith("/") ? path : API + "/" + path, window.location.origin);
    for (const [key, value] of Object.entries(opts.params || {})) {
      if (value !== undefined && value !== null && value !== "") {
        url.searchParams.set(key, value);
      }
    }
    const headers = { Accept: opts.accept || "application/json" };
    const token = Token.get();
    if (token) {
      headers.Authorization = "Bearer " + token;
    }
    let body;
    if (opts.body !== undefined) {
      headers["Content-Type"] = "application/json";
      body = JSON.stringify(opts.body);
    }
    let response;
    try {
      response = await fetch(url, { method, headers, body, signal: opts.signal, cache: "no-store" });
    } catch (error) {
      if (error && error.name === "AbortError") {
        throw error;
      }
      throw new ApiError(0, "network", "The server does not answer.", null);
    }
    if (response.ok) {
      if (response.status === 204) {
        return opts.raw ? response : null;
      }
      if (opts.raw) {
        return response;
      }
      return response.json();
    }
    let data = null;
    try {
      data = await response.json();
    } catch (error) {
      /* The body was not JSON. The status of the response is all that there is to report. */
    }
    const failure = failureFrom(response.status, data);
    const retry = Number(response.headers.get("Retry-After"));
    if (response.status === 401) {
      emit("seeing:auth-needed", { message: failure.message });
    }
    throw new ApiError(
      response.status,
      failure.code,
      failure.message,
      Number.isFinite(retry) && retry > 0 ? retry : null,
      failure.details
    );
  }

  const api = {
    get: (path, params, options) => request("GET", path, Object.assign({}, options, { params })),
    post: (path, body, options) => request("POST", path, Object.assign({}, options, { body: body === undefined ? {} : body })),
    delete: (path, options) => request("DELETE", path, options),
    raw: (path, params, options) => request("GET", path, Object.assign({}, options, { params, raw: true })),
    url: (path) => (path.startsWith("/") ? path : API + "/" + path),
  };

  // --- Polling ----------------------------------------------------------------------------------

  /**
   * Run `fn` now and then every `interval()` milliseconds while the page is visible. One run at a
   * time, and a hidden page waits. `fn` handles its own errors.
   */
  function poller(fn, interval) {
    let timer = null;
    let running = false;
    let stopped = true;

    function schedule() {
      clearTimeout(timer);
      if (stopped || document.hidden) {
        return;
      }
      timer = setTimeout(tick, interval());
    }

    async function tick() {
      if (running) {
        return;
      }
      running = true;
      try {
        await fn();
      } catch (error) {
        /* fn reports its errors on the page. */
      } finally {
        running = false;
        schedule();
      }
    }

    document.addEventListener("visibilitychange", () => {
      if (!stopped && !document.hidden) {
        clearTimeout(timer);
        tick();
      }
    });

    return {
      start() {
        stopped = false;
        tick();
      },
      stop() {
        stopped = true;
        clearTimeout(timer);
      },
      now() {
        clearTimeout(timer);
        return tick();
      },
    };
  }

  // --- Times that tick --------------------------------------------------------------------------

  /** One timer for the whole page: each job runs once a second, and not while the tab is hidden. */
  const Ticker = {
    jobs: new Set(),
    timer: null,
    add(job) {
      Ticker.jobs.add(job);
      if (Ticker.timer === null) {
        Ticker.timer = setInterval(Ticker.run, 1000);
      }
      return () => Ticker.jobs.delete(job);
    },
    run() {
      if (document.hidden) {
        return;
      }
      for (const job of Ticker.jobs) {
        try {
          job();
        } catch (error) {
          /* A job reports its own failures. */
        }
      }
    },
  };

  function renderAge(node) {
    const t = node.dataset.t === "" || node.dataset.t === undefined ? null : Number(node.dataset.t);
    if (t === null || Number.isNaN(t)) {
      node.textContent = "";
      delete node.dataset.stale;
      return;
    }
    const age = Math.max(0, (Status.nowMs() - t) / 1000);
    const expect = (typeof node.expectS === "function" ? node.expectS() : Number(node.expectS)) || 0;
    node.textContent = (node.dataset.prefix || "") + fmt.age(age);
    const level = expect > 0 && age > 3 * expect ? "bad" : expect > 0 && age > 1.5 * expect ? "warn" : "";
    if (level) {
      node.dataset.stale = level;
    } else {
      delete node.dataset.stale;
    }
  }

  /**
   * Show how old a time is, and keep the age ticking each second, so that a value that stops
   * arriving is easy to see. `expectedS` is how often a new value should come, in seconds, or a
   * function that returns it (0: no limit). The age turns "late" at 1.5 times that and "stale" at
   * 3 times, in words, because the night mode shows only red. `tMs` is the time in milliseconds on
   * the server clock, or null for no value. A node that leaves the page stops ticking.
   */
  function watchAge(node, tMs, expectedS, prefix) {
    node.dataset.t = tMs === null || tMs === undefined ? "" : String(tMs);
    node.expectS = expectedS || 0;
    node.dataset.prefix = prefix || "";
    renderAge(node);
    if (!node.dataset.watched) {
      node.dataset.watched = "1";
      const stop = Ticker.add(() => {
        if (node.isConnected) {
          renderAge(node);
        } else {
          stop();
        }
      });
    }
  }

  // --- The night mode ---------------------------------------------------------------------------

  const Night = {
    on: false,

    init() {
      const query = new URLSearchParams(window.location.search).get("night");
      if (query === "1" || query === "0") {
        writeStorage("localStorage", NIGHT_KEY, query);
      }
      Night.apply(readStorage("localStorage", NIGHT_KEY) === "1");
    },

    apply(on) {
      Night.on = on;
      if (on) {
        document.documentElement.dataset.night = "1";
      } else {
        delete document.documentElement.dataset.night;
      }
      const overlay = $("night-overlay");
      if (overlay) {
        overlay.hidden = !on;
      }
      const button = $("night-button");
      if (button) {
        button.setAttribute("aria-pressed", on ? "true" : "false");
      }
      emit("seeing:theme", { night: on });
    },

    toggle() {
      writeStorage("localStorage", NIGHT_KEY, Night.on ? "0" : "1");
      Night.apply(!Night.on);
    },
  };

  // --- Status of the station --------------------------------------------------------------------

  const Status = {
    current: null,
    receivedAt: 0,

    /**
     * The server time in milliseconds: the time of the newest status plus the time that has passed
     * since it arrived. The browser clock is never used, only the time that has elapsed.
     */
    nowMs() {
      if (!Status.current) {
        return Date.now();
      }
      return Date.parse(Status.current.now) + (performance.now() - Status.receivedAt);
    },

    /** The server time as a Date, from the newest status. The browser clock is never used. */
    now() {
      return Status.current ? new Date(Status.current.now) : new Date();
    },

    async load() {
      const status = await api.get("status");
      Status.current = status;
      Status.receivedAt = performance.now();
      Status.render();
      emit("seeing:status", status);
      return status;
    },

    render() {
      const status = Status.current;
      const pill = $("health-pill");
      const demo = $("demo-badge");
      if (!status) {
        return;
      }
      if (demo) {
        demo.hidden = !status.demo;
      }
      if (pill) {
        const word = status.health.status;
        pill.textContent = word.charAt(0).toUpperCase() + word.slice(1);
        pill.dataset.level = fmt.level(word);
        pill.hidden = false;
      }
      const hint = $("token-hint");
      if (hint) {
        hint.textContent = status.demo
          ? "In the demo, the token is demo."
          : status.ui.token_required_for_reads
            ? "This server asks for the token on every read."
            : "Commands need the token. Reads do not.";
      }
      const use = $("token-demo");
      if (use) {
        use.hidden = !status.demo;
      }
      const version = $("foot-version");
      if (version) {
        version.textContent = "Software " + status.software_version + ", API " + status.api_version;
      }
    },
  };

  // --- The frame of a page ----------------------------------------------------------------------

  function buildTokenPanel() {
    const input = h("input", {
      id: "token-input",
      type: "password",
      autocomplete: "off",
      autocapitalize: "off",
      spellcheck: "false",
      "aria-label": "API token",
      placeholder: "API token",
    });
    const state = h("span", { id: "token-state", class: "hint" });

    function refresh() {
      state.textContent = Token.has()
        ? "A token is set for this tab. It is forgotten when you close the tab."
        : "No token is set.";
    }

    const panel = h(
      "section",
      { id: "token-panel", class: "panel", hidden: true, "aria-label": "API token" },
      h("h2", { text: "API token" }),
      h("p", { id: "token-hint", class: "hint", text: "Commands need the token." }),
      h(
        "div",
        { class: "row" },
        input,
        h("button", {
          class: "primary",
          type: "button",
          text: "Use token",
          onclick: () => {
            Token.set(input.value);
            input.value = "";
            refresh();
            panel.hidden = true;
          },
        }),
        h("button", {
          id: "token-demo",
          type: "button",
          hidden: true,
          text: "Use the demo token",
          onclick: () => {
            Token.set("demo");
            refresh();
            panel.hidden = true;
          },
        }),
        h("button", {
          type: "button",
          text: "Forget",
          onclick: () => {
            Token.set("");
            refresh();
          },
        })
      ),
      state
    );
    refresh();
    window.addEventListener("seeing:token", refresh);
    return panel;
  }

  function openTokenPanel() {
    const panel = $("token-panel");
    if (panel) {
      panel.hidden = false;
      const input = $("token-input");
      if (input) {
        input.focus();
      }
    }
  }

  function buildFrame(pageId) {
    // A tool button has an icon (drawn in CSS) and a label. A narrow screen hides the label, but a
    // screen reader still reads it, and the title shows it as a tooltip.
    const tool = (props, icon, label) =>
      h(
        "button",
        Object.assign({ class: "tool", type: "button" }, props),
        h("span", { class: "icon icon-" + icon, "aria-hidden": "true" }),
        h("span", { class: "tool-label", text: label })
      );
    const tokenButton = tool(
      {
        id: "token-button",
        title: "API token",
        "aria-controls": "token-panel",
        onclick: () => {
          const panel = $("token-panel");
          if (panel.hidden) {
            openTokenPanel();
          } else {
            panel.hidden = true;
          }
        },
      },
      "lock",
      "Token"
    );
    const nightButton = tool(
      {
        id: "night-button",
        "aria-pressed": "false",
        title: "Show the page in red only, to keep your night vision",
        onclick: () => Night.toggle(),
      },
      "moon",
      "Night mode"
    );
    const header = h(
      "header",
      { class: "top" },
      h(
        "div",
        { class: "top-row" },
        h(
          "div",
          { class: "brand" },
          h("span", { class: "brand-name", text: "Seeing monitor" }),
          h("span", { id: "demo-badge", class: "badge badge-demo", text: "Demo", hidden: true }),
          h("span", { id: "health-pill", class: "pill", text: "", hidden: true, role: "status" })
        ),
        h("div", { class: "tools" }, tokenButton, nightButton)
      ),
      h(
        "nav",
        { class: "tabs", "aria-label": "Pages" },
        PAGES.map((page) =>
          h("a", {
            href: page.href,
            text: page.label,
            "aria-current": page.id === pageId ? "page" : null,
          })
        )
      )
    );
    const footer = h(
      "footer",
      { class: "foot" },
      h("span", { id: "foot-version", text: "" }),
      " · ",
      h("a", { href: "api.html", text: "API reference" }),
      " · ",
      h("a", { href: API + "/openapi.json", text: "OpenAPI" }),
      " · All times are UTC."
    );
    const overlay = h("div", { id: "night-overlay", class: "night-overlay", hidden: true, "aria-hidden": "true" });
    document.body.prepend(header);
    const main = document.querySelector("main");
    main.prepend(buildTokenPanel());
    main.after(footer);
    document.body.append(overlay);
  }

  /** Build the frame, set the night mode, and keep the status fresh. Call it once per page. */
  function boot(pageId, options) {
    const opts = options || {};
    buildFrame(pageId);
    Night.init();
    window.addEventListener("seeing:auth-needed", (event) => {
      openTokenPanel();
      const hint = $("token-hint");
      if (hint && event.detail && event.detail.message) {
        hint.textContent = event.detail.message;
      }
    });
    const refresh = opts.statusEvery === false ? null : poller(
      () => Status.load().catch((error) => emit("seeing:status-error", error)),
      () => (Status.current ? Status.current.ui.refresh_s * 1000 : 10000)
    );
    if (refresh) {
      refresh.start();
    }
    return refresh;
  }

  // --- Pictures ---------------------------------------------------------------------------------

  /**
   * Show an image in an `<img>`. A server that asks for the token on reads gets a request with
   * the token, and the page shows the bytes as an object URL, because an `<img>` cannot send a
   * header. Otherwise the element loads the address itself.
   */
  async function showImage(element, url) {
    const status = Status.current;
    if (!status || !status.ui.token_required_for_reads) {
      element.src = url;
      return;
    }
    const response = await api.raw(url);
    const blob = await response.blob();
    if (element.dataset.objectUrl) {
      URL.revokeObjectURL(element.dataset.objectUrl);
    }
    element.dataset.objectUrl = URL.createObjectURL(blob);
    element.src = element.dataset.objectUrl;
  }

  window.Seeing = {
    API, h, $, clear, emit, fmt, api, ApiError, failureFrom, Token, Night, Status, poller, boot, showImage, Ticker, watchAge,
    openTokenPanel, readStorage, writeStorage, recall, remember, flagChip, explainReason, FLAG_HELP,
  };
})();
