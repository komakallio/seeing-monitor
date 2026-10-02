"use strict";

/*
 * The link between the Align page and the live view of the server.
 *
 * A `LiveLink` opens the WebSocket of the live view, pairs each state message with the JPEG that
 * follows it, and hands the pair to `onFrame(blob, state)`. It keeps the link alive:
 *
 * - After a drop, it reconnects with a growing delay (1, 2, 4, 8, 15, then 30 s, each with some
 *   random spread, so that many viewers do not return at the same instant).
 * - After `pollAfterFailures` attempts that never opened, or when the browser has no WebSocket, it
 *   polls the newest frame with `pollFrame()` and keeps trying the WebSocket once a minute. The
 *   polling runs only while the WebSocket is not usable: the open socket ends it, and a poll that
 *   is still in flight then ends without drawing.
 * - Close code 1008 means that the server wants a token (or refused it). The link waits for
 *   `restart()`, which the page calls when the token changes.
 * - Close code 1013 means that the server shows the view to as many viewers as it allows. The
 *   link tries again after 10 to 20 seconds. Such a refusal never starts the polling: the server
 *   is up, and it is full.
 * - `tick()` marks the link stalled when frames stop while the alignment runs.
 *
 * Everything that touches the outside is an option, so the logic can run against a fake socket and
 * a fake clock: `createSocket`, `setTimeout`, `clearTimeout`, `now`, and `random`.
 *
 * The names of the states: connecting, live, waiting (connected, no frame yet), stalled,
 * reconnecting, polling, denied, busy, and unavailable (the server cannot reach core).
 */
(function () {
  const DEFAULTS = {
    backoffMs: [1000, 2000, 4000, 8000, 15000, 30000],
    pollAfterFailures: 3,
    pollRetryWebSocketMs: 60000,
    busyRetryMs: [10000, 20000],
    tokenRequired: () => false,
    token: () => "",
    isActive: () => true,
    staleMs: () => 5000,
    pollIntervalMs: () => 500,
    pollFrame: null,
    onFrame: () => undefined,
    onLink: () => undefined,
    url: () => "",
    createSocket: (url) => new WebSocket(url),
    supportsWebSocket: () => typeof WebSocket === "function",
    setTimeout: (callback, ms) => setTimeout(callback, ms),
    clearTimeout: (id) => clearTimeout(id),
    now: () => Date.now(),
    random: () => Math.random(),
  };

  class LiveLink {
    constructor(options) {
      this.o = Object.assign({}, DEFAULTS, options);
      this.name = "idle";
      this.stopped = true;
      this.socket = null;
      this.opened = false;
      this.failures = 0;
      this.pending = null;
      this.timer = null;
      this.pollTimer = null;
      this.polling = false;
      this.pollRun = 0;
      this.lastFrameAt = 0;
      this.lastSeq = 0;
    }

    setName(name) {
      if (this.name !== name) {
        this.name = name;
        this.o.onLink(name);
      }
    }

    /** True while a frame arrived within the stale time. */
    fresh() {
      return this.lastFrameAt !== 0 && this.o.now() - this.lastFrameAt < this.o.staleMs();
    }

    start() {
      this.stopped = false;
      this.failures = 0;
      this.connect();
    }

    stop() {
      this.stopped = true;
      this.o.clearTimeout(this.timer);
      this.stopPolling();
      const socket = this.socket;
      this.socket = null;
      this.opened = false;
      this.pending = null;
      if (socket) {
        socket.onopen = null;
        socket.onclose = null;
        socket.onmessage = null;
        socket.close(1000);
      }
    }

    /** Start over: the page calls it when the token changes. */
    restart() {
      this.stop();
      this.start();
    }

    connect() {
      this.o.clearTimeout(this.timer);
      if (this.stopped || this.socket) {
        return;
      }
      if (!this.o.supportsWebSocket()) {
        this.startPolling();
        return;
      }
      if (!this.polling) {
        this.setName(this.failures > 0 ? "reconnecting" : "connecting");
      }
      let socket;
      try {
        socket = this.o.createSocket(this.o.url());
      } catch (error) {
        this.closed({ code: 1006 });
        return;
      }
      socket.binaryType = "arraybuffer";
      this.socket = socket;
      this.opened = false;
      this.pending = null;
      // Each handler acts only for the current socket, so that a late event of a socket that the
      // link has dropped cannot change what the link believes about the new one.
      socket.onopen = () => {
        if (this.socket !== socket) {
          return;
        }
        this.opened = true;
        this.failures = 0;
        this.stopPolling();
        if (this.o.tokenRequired()) {
          socket.send(JSON.stringify({ type: "auth", token: this.o.token() }));
        }
        this.setName("waiting");
      };
      socket.onmessage = (event) => {
        if (this.socket === socket) {
          this.message(event.data);
        }
      };
      socket.onerror = () => undefined;
      socket.onclose = (event) => {
        if (this.socket === socket) {
          this.socket = null;
          this.closed(event);
        }
      };
    }

    /** True while the WebSocket is open. Frames then arrive on it, and the polling must stay off. */
    socketUsable() {
      return this.socket !== null && this.opened;
    }

    closed(event) {
      if (this.stopped) {
        return;
      }
      const code = event && event.code;
      if (code === 1008) {
        this.stopPolling();
        this.setName("denied");
        return; // A new token restarts the link.
      }
      if (code === 1013) {
        const [low, high] = this.o.busyRetryMs;
        this.failures = 0; // the server answered, so the path of the WebSocket works
        this.setName("busy");
        this.timer = this.o.setTimeout(() => this.connect(), low + this.o.random() * (high - low));
        return;
      }
      if (!this.opened) {
        this.failures += 1;
      }
      if (this.failures >= this.o.pollAfterFailures && this.o.pollFrame) {
        this.startPolling();
      }
      this.setName(this.polling ? "polling" : "reconnecting");
      this.timer = this.o.setTimeout(() => this.connect(), this.polling ? this.o.pollRetryWebSocketMs : this.delay());
    }

    delay() {
      const steps = this.o.backoffMs;
      const base = steps[Math.min(Math.max(this.failures - 1, 0), steps.length - 1)];
      return base * (0.8 + this.o.random() * 0.4);
    }

    message(data) {
      if (typeof data === "string") {
        let message;
        try {
          message = JSON.parse(data);
        } catch (error) {
          return;
        }
        if (message.type === "state") {
          this.pending = message.state;
        } else if (message.type === "idle") {
          this.pending = null;
          this.setName(this.fresh() ? "live" : "waiting");
        } else if (message.type === "error") {
          this.setName(message.code === "core_unavailable" ? "unavailable" : "waiting");
        }
        return;
      }
      const state = this.pending;
      this.pending = null;
      if (!state) {
        return;
      }
      this.lastFrameAt = this.o.now();
      this.setName("live");
      this.o.onFrame(new Blob([data], { type: "image/jpeg" }), state);
    }

    /** Call it about once a second: it notices that the frames stopped. */
    tick() {
      if (this.stopped || !this.o.isActive()) {
        return;
      }
      if (this.name === "live" && !this.fresh()) {
        this.setName("stalled");
      } else if (this.name === "stalled" && this.fresh()) {
        this.setName("live");
      }
    }

    // --- Polling, the fallback --------------------------------------------------------------

    /** The polling runs only while the WebSocket is not usable: it is closed, or it is still connecting. */
    startPolling() {
      if (this.polling || !this.o.pollFrame || this.socketUsable()) {
        return;
      }
      this.polling = true;
      this.pollRun += 1;
      this.setName("polling");
      this.schedulePoll(0);
    }

    stopPolling() {
      this.polling = false;
      this.pollRun += 1; // a poll that is in flight belongs to the old run, and it ends when it returns
      this.o.clearTimeout(this.pollTimer);
    }

    schedulePoll(delay) {
      this.o.clearTimeout(this.pollTimer);
      const run = this.pollRun;
      this.pollTimer = this.o.setTimeout(() => this.poll(run), delay);
    }

    /** True while the poll of this run may go on: the link runs, and the WebSocket is not open. */
    pollingAllowed(run) {
      if (this.polling && this.socketUsable()) {
        this.stopPolling();
      }
      return this.polling && !this.stopped && run === this.pollRun;
    }

    async poll(run) {
      if (!this.pollingAllowed(run)) {
        return;
      }
      try {
        const result = await this.o.pollFrame(this.lastSeq);
        if (!this.pollingAllowed(run)) {
          return;
        }
        if (result && result.denied) {
          this.stopPolling();
          this.setName("denied");
          return;
        }
        if (result && result.blob) {
          this.lastSeq = result.seq || this.lastSeq;
          this.lastFrameAt = this.o.now();
          this.setName("polling");
          this.o.onFrame(result.blob, result.state);
        }
      } catch (error) {
        /* The next poll tries again. */
      }
      if (this.pollingAllowed(run)) {
        this.schedulePoll(this.o.pollIntervalMs());
      }
    }
  }

  window.Seeing.LiveLink = LiveLink;
})();
