"use strict";

/*
 * The scenarios of the live link (static/js/live.js). They run against a fake socket and a fake
 * clock, so they take no real time. The function takes `LiveLink`, a `test(name, fn)` function,
 * and an `assert` object, so that Node (live_link.test.js) and a browser console can both run it.
 */
module.exports = function scenarios(LiveLink, test, assert) {
  const JPEG = new Uint8Array([0xff, 0xd8, 1, 2, 3]).buffer;

  function state(seq) {
    return JSON.stringify({ type: "state", state: { active: true, frame: { seq } } });
  }

  /** A link over fake sockets and a fake clock. `advance(ms)` runs the timers that fall due. */
  function harness(extra) {
    // The clock starts after zero, because the link reads a last-frame time of zero as "never".
    const h = { names: [], frames: [], polls: [], sockets: [], timers: [], now: 1000, pollResult: null };
    h.options = Object.assign(
      {
        url: () => "ws://example.invalid/stream",
        supportsWebSocket: () => true, // a Node context has no WebSocket, and a browser does
        createSocket: (url) => {
          const socket = { url, sent: [], closed: null, send(data) { this.sent.push(data); }, close(code) { this.closed = code; } };
          h.sockets.push(socket);
          return socket;
        },
        setTimeout: (callback, ms) => {
          const timer = { id: h.timers.length + 1, callback, at: h.now + ms };
          h.timers.push(timer);
          return timer.id;
        },
        clearTimeout: (id) => {
          const timer = h.timers.find((t) => t.id === id);
          if (timer) {
            timer.cancelled = true;
          }
        },
        now: () => h.now,
        random: () => 0.5,
        onLink: (name) => h.names.push(name),
        onFrame: (blob, frameState) => h.frames.push([blob.size, frameState && frameState.frame && frameState.frame.seq]),
        staleMs: () => 5000,
        pollIntervalMs: () => 500,
        pollFrame: async (after) => {
          h.polls.push(after);
          return h.pollResult ? h.pollResult(after) : null;
        },
      },
      extra || {}
    );
    h.advance = async (ms) => {
      const target = h.now + ms;
      for (;;) {
        const due = h.timers.filter((t) => !t.cancelled && !t.done && t.at <= target).sort((a, b) => a.at - b.at)[0];
        if (!due) {
          break;
        }
        h.now = Math.max(h.now, due.at);
        due.done = true;
        due.callback();
        for (let i = 0; i < 4; i += 1) {
          await Promise.resolve();
        }
      }
      h.now = target;
    };
    return h;
  }

  function opened(extra) {
    const h = harness(extra);
    const link = new LiveLink(h.options);
    link.start();
    h.sockets[0].onopen();
    return { h, link, socket: h.sockets[0] };
  }

  test("a link connects, pairs each state with the JPEG after it, and goes live", () => {
    const { h, socket } = opened();
    assert.equal(socket.binaryType, "arraybuffer");
    socket.onmessage({ data: state(1) });
    socket.onmessage({ data: JPEG });
    assert.deepEqual(h.names, ["connecting", "waiting", "live"]);
    assert.deepEqual(h.frames, [[5, 1]]);
    assert.deepEqual(socket.sent, []);
  });

  test("a binary message without a state before it, and text that is not JSON, are ignored", () => {
    const { h, link, socket } = opened();
    socket.onmessage({ data: JPEG });
    socket.onmessage({ data: "{not json" });
    assert.equal(h.frames.length, 0);
    assert.equal(link.name, "waiting");
  });

  test("a server that asks for the token gets it as the first message", () => {
    const { socket } = opened({ tokenRequired: () => true, token: () => "abc" });
    assert.deepEqual(socket.sent, ['{"type":"auth","token":"abc"}']);
  });

  test("a dropped connection is tried again after about a second", async () => {
    const { h, link, socket } = opened();
    socket.onclose({ code: 1006 });
    assert.equal(link.name, "reconnecting");
    await h.advance(900);
    assert.equal(h.sockets.length, 1);
    await h.advance(300);
    assert.equal(h.sockets.length, 2);
  });

  test("the delay grows when attempts never open, and three of them start the polling", async () => {
    const h = harness();
    const link = new LiveLink(h.options);
    link.start();
    const names = [];
    const delays = [];
    for (let attempt = 0; attempt < 3; attempt += 1) {
      const before = h.timers.length;
      h.sockets[h.sockets.length - 1].onclose({ code: 1006 });
      names.push(link.name);
      const timer = h.timers[before];
      delays.push(Math.round(timer && !timer.cancelled ? timer.at - h.now : -1));
      await h.advance(70000);
    }
    assert.deepEqual(names, ["reconnecting", "reconnecting", "polling"]);
    assert.deepEqual(delays.slice(0, 2), [1000, 2000]);
    assert.ok(h.polls.length > 0);
    assert.equal(link.polling, true);
  });

  test("while it polls, it tries the WebSocket once a minute, and an open socket ends the polling", async () => {
    const h = harness();
    const link = new LiveLink(h.options);
    link.start();
    for (let attempt = 0; attempt < 3; attempt += 1) {
      h.sockets[h.sockets.length - 1].onclose({ code: 1006 });
      await h.advance(2500);
    }
    assert.equal(link.name, "polling");
    const before = h.sockets.length;
    await h.advance(61000);
    assert.equal(h.sockets.length - before, 1);
    assert.equal(link.name, "polling"); // the attempt does not hide the polling
    h.sockets[h.sockets.length - 1].onopen();
    const polls = h.polls.length;
    await h.advance(5000);
    assert.equal(h.polls.length, polls);
    assert.equal(link.name, "waiting");
    assert.equal(link.polling, false);
  });

  test("close code 1008 waits for a restart, and a restart connects again", async () => {
    const { h, link, socket } = opened();
    socket.onclose({ code: 1008 });
    assert.equal(link.name, "denied");
    await h.advance(120000);
    assert.equal(h.sockets.length, 1);
    link.restart();
    assert.equal(h.sockets.length, 2);
  });

  test("close code 1013 means busy, and the link tries again after 10 to 20 seconds", async () => {
    const h = harness();
    const link = new LiveLink(h.options);
    link.start();
    h.sockets[0].onclose({ code: 1013 });
    assert.equal(link.name, "busy");
    await h.advance(9000);
    assert.equal(h.sockets.length, 1);
    await h.advance(7000);
    assert.equal(h.sockets.length, 2);
  });

  test("a browser without WebSocket polls from the start, in order of sequence number", async () => {
    const h = harness({ supportsWebSocket: () => false });
    h.pollResult = (after) => ({ blob: new Blob([JPEG]), state: { frame: { seq: after + 1 } }, seq: after + 1 });
    const link = new LiveLink(h.options);
    link.start();
    await h.advance(1600);
    assert.equal(h.sockets.length, 0);
    assert.equal(link.name, "polling");
    assert.deepEqual(h.polls, [0, 1, 2, 3]);
    assert.deepEqual(h.frames.map((f) => f[1]), [1, 2, 3, 4]);
  });

  test("polling that gets a 401 ends in denied", async () => {
    const h = harness({ supportsWebSocket: () => false });
    h.pollResult = () => ({ denied: true });
    const link = new LiveLink(h.options);
    link.start();
    await h.advance(2000);
    assert.equal(link.name, "denied");
    assert.equal(link.polling, false);
  });

  test("a poll that throws is tried again", async () => {
    const h = harness({ supportsWebSocket: () => false });
    let calls = 0;
    h.options.pollFrame = async () => {
      calls += 1;
      throw new Error("network");
    };
    const link = new LiveLink(h.options);
    link.start();
    await h.advance(1600);
    assert.ok(calls >= 3);
    assert.equal(link.polling, true);
  });

  test("frames that stop while the alignment runs make the link stalled, and a frame revives it", () => {
    const { h, link, socket } = opened({ isActive: () => true });
    socket.onmessage({ data: state(1) });
    socket.onmessage({ data: JPEG });
    h.now += 6000;
    link.tick();
    assert.equal(link.name, "stalled");
    socket.onmessage({ data: state(2) });
    socket.onmessage({ data: JPEG });
    link.tick();
    assert.equal(link.name, "live");
  });

  test("a link whose alignment is not running is never stalled", () => {
    const { h, link, socket } = opened({ isActive: () => false });
    socket.onmessage({ data: state(1) });
    socket.onmessage({ data: JPEG });
    h.now += 60000;
    link.tick();
    assert.equal(link.name, "live");
  });

  test("an idle message means waiting once the last frame is old, and an error message names the cause", () => {
    const { h, link, socket } = opened();
    socket.onmessage({ data: state(1) });
    socket.onmessage({ data: JPEG });
    socket.onmessage({ data: JSON.stringify({ type: "idle" }) });
    assert.equal(link.name, "live"); // the last frame is still fresh
    h.now += 6000;
    socket.onmessage({ data: JSON.stringify({ type: "idle" }) });
    assert.equal(link.name, "waiting");
    socket.onmessage({ data: JSON.stringify({ type: "error", code: "core_unavailable" }) });
    assert.equal(link.name, "unavailable");
    socket.onmessage({ data: JSON.stringify({ type: "error", code: "core_error" }) });
    assert.equal(link.name, "waiting");
  });

  test("stop closes the socket with 1000 and ignores what the socket does afterwards", async () => {
    const { h, link, socket } = opened();
    link.stop();
    assert.equal(socket.closed, 1000);
    assert.equal(socket.onclose, null);
    await h.advance(120000);
    assert.equal(h.sockets.length, 1);
  });

  test("a socket that cannot be created counts as a failed attempt", async () => {
    const h = harness({
      createSocket: () => {
        throw new Error("blocked");
      },
    });
    const link = new LiveLink(h.options);
    link.start();
    assert.equal(link.name, "reconnecting");
    await h.advance(1200);
    await h.advance(2400);
    assert.equal(link.polling, true);
  });

  test("a second start does not open a second socket", () => {
    const h = harness();
    const link = new LiveLink(h.options);
    link.start();
    link.connect();
    assert.equal(h.sockets.length, 1);
  });

  // --- The polling runs only while the WebSocket is not usable ------------------------------------

  /** Let the continuations of the promises that a test has just settled run. */
  async function settle() {
    for (let i = 0; i < 6; i += 1) {
      await Promise.resolve();
    }
  }

  /** Fail three attempts, so that the link polls. Returns once the first polls have gone out. */
  async function pollingLink(h) {
    const link = new LiveLink(h.options);
    link.start();
    for (let attempt = 0; attempt < 3; attempt += 1) {
      h.sockets[h.sockets.length - 1].onclose({ code: 1006 });
      await h.advance(2500);
    }
    assert.equal(link.name, "polling");
    return link;
  }

  test("the retry of the WebSocket does not stop the polling until the socket is open, and then it does", async () => {
    const h = harness();
    const link = await pollingLink(h);
    await h.advance(60000); // the minute is over: the retry is connecting, and not open
    const retry = h.sockets[h.sockets.length - 1];
    assert.equal(link.socketUsable(), false);
    const connecting = h.polls.length;
    await h.advance(2000);
    assert.ok(h.polls.length > connecting, "polls while the retry connects");
    retry.onopen();
    assert.equal(link.socketUsable(), true);
    assert.equal(link.polling, false);
    const polls = h.polls.length;
    await h.advance(180000);
    assert.equal(h.polls.length, polls, "no poll while the socket is open");
    assert.equal(link.name, "waiting");
  });

  test("a poll that is in flight when the socket opens ends without drawing and without another poll", async () => {
    const h = harness();
    const returns = [];
    h.options.pollFrame = (after) => {
      h.polls.push(after);
      return new Promise((resolve) => returns.push(resolve));
    };
    const link = await pollingLink(h);
    assert.equal(h.polls.length, 1); // one poll waits for its answer
    await h.advance(60000);
    h.sockets[h.sockets.length - 1].onopen();
    returns[0]({ blob: new Blob([JPEG]), state: { frame: { seq: 9 } }, seq: 9 });
    await h.advance(5000);
    assert.equal(h.polls.length, 1);
    assert.deepEqual(h.frames, []);
    assert.equal(link.polling, false);
    assert.equal(link.name, "waiting");
  });

  test("a poll that is in flight across a stop and a start leaves one chain of polls", async () => {
    const h = harness({ supportsWebSocket: () => false });
    const returns = [];
    h.options.pollFrame = (after) => {
      h.polls.push(after);
      return new Promise((resolve) => returns.push(resolve));
    };
    const link = new LiveLink(h.options);
    link.start();
    await h.advance(100);
    assert.equal(h.polls.length, 1);
    link.stop();
    link.start();
    await h.advance(100);
    assert.equal(h.polls.length, 2); // the new run polls at once, and the old poll still waits
    returns.splice(0).forEach((resolve) => resolve(null));
    await settle();
    await h.advance(600);
    assert.equal(h.polls.length, 3, "only the new run goes on");
    returns.splice(0).forEach((resolve) => resolve(null));
    await settle();
    await h.advance(600);
    assert.equal(h.polls.length, 4);
  });

  test("a connection that opened and then dropped is retried, and it never starts the polling", async () => {
    const h = harness();
    const link = new LiveLink(h.options);
    link.start();
    for (let round = 0; round < 8; round += 1) {
      const socket = h.sockets[h.sockets.length - 1];
      socket.onopen();
      assert.equal(link.polling, false);
      socket.onclose({ code: 1006 });
      assert.equal(link.name, "reconnecting");
      await h.advance(1500);
    }
    assert.equal(h.sockets.length, 9);
    assert.equal(h.polls.length, 0);
    assert.equal(link.polling, false);
  });

  test("a refusal as busy never starts the polling, however often it comes", async () => {
    const h = harness();
    const link = new LiveLink(h.options);
    link.start();
    for (let round = 0; round < 6; round += 1) {
      h.sockets[h.sockets.length - 1].onclose({ code: 1013 });
      assert.equal(link.name, "busy");
      await h.advance(21000);
    }
    assert.equal(h.sockets.length, 7);
    assert.equal(h.polls.length, 0);
    assert.equal(link.polling, false);
  });

  test("a busy answer shows that the WebSocket path works, so earlier failures do not add up to polling", async () => {
    const h = harness();
    const link = new LiveLink(h.options);
    link.start();
    h.sockets[0].onclose({ code: 1006 });
    await h.advance(1500);
    h.sockets[1].onclose({ code: 1006 });
    await h.advance(2500);
    h.sockets[2].onclose({ code: 1013 });
    await h.advance(21000);
    h.sockets[3].onclose({ code: 1006 });
    assert.equal(link.name, "reconnecting");
    assert.equal(link.polling, false);
  });

  test("a late event of a socket that the link dropped changes nothing", async () => {
    const h = harness();
    const link = new LiveLink(h.options);
    link.start();
    const lateOpen = h.sockets[0].onopen;
    const lateMessage = h.sockets[0].onmessage;
    const lateClose = h.sockets[0].onclose;
    link.restart(); // drops the first socket, and connects a second one
    assert.equal(h.sockets.length, 2);
    lateOpen();
    lateMessage({ data: state(1) });
    lateMessage({ data: JPEG });
    lateClose({ code: 1006 });
    assert.equal(link.opened, false);
    assert.equal(link.socketUsable(), false);
    assert.equal(link.name, "connecting");
    assert.deepEqual(h.frames, []);
    assert.equal(h.sockets.length, 2);
    // The second socket still counts its own failures, so three of them start the polling.
    for (let attempt = 0; attempt < 3; attempt += 1) {
      h.sockets[h.sockets.length - 1].onclose({ code: 1006 });
      await h.advance(2500);
    }
    assert.equal(link.polling, true);
  });

  test("startPolling and a poll that is due both refuse to run while the socket is open", async () => {
    const { h, link } = opened();
    link.startPolling();
    assert.equal(link.polling, false);
    link.polling = true; // a wrong flag, as a bug elsewhere could leave it
    await link.poll(link.pollRun);
    assert.equal(link.polling, false, "the poll notices the open socket and stops");
    await h.advance(5000);
    assert.equal(h.polls.length, 0);
  });
};
