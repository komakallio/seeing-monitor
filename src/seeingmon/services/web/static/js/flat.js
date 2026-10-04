"use strict";

/*
 * The Flat page: the flat in use, a form that starts a flat session, the progress of the session,
 * the review of the new flat with the buttons that decide about it, and the library of flats. The
 * owner covers the front of the guide scope with an even light, presses Take flat, waits, looks at
 * the new flat, and presses Use this flat or Discard. The station pauses at the end of a session,
 * so that nothing records data while the light covers the camera. Taking a flat resumes a paused
 * scheduler, because the owner asks for the session now. Stop ends a session.
 *
 * `flattext.js` holds the logic that needs no page (the words, the phases, the verdicts on a flat,
 * and the check of the form). This file reads the API and puts the results into the page. The page
 * reads `GET /flat` every 2 seconds while a session is queued or running, and every 10 seconds
 * otherwise, and a hidden page waits.
 */
(function () {
  const { h, $, clear, fmt, api, Status, Token, poller, showImage } = window.Seeing;
  const { FlatText } = window.Seeing;

  const state = {
    library: null,
    failed: null, // the error of the last read, or null
    sending: false,
    readCount: 0, // numbers the reads, so that an older answer never replaces a newer one
    applied: 0,
    previousTask: null, // the state of the task at the last read, to notice that it ended
    shownImage: null, // the version whose preview the page shows
    reviewing: null, // the version that the review shows
    filled: false, // whether the fields of the form hold their defaults
    armed: null, // the deletion that waits for a second press: { key, timer }
  };
  const ARM_MS = 5000;
  let loop = null;

  function schedulerState() {
    const status = Status.current;
    if (!status || !status.core.reachable || !status.scheduler) {
      return status ? null : undefined;
    }
    return status.scheduler.state;
  }

  function commandsEnabled() {
    const status = Status.current;
    return !status || status.ui.commands_enabled;
  }

  // --- Rendering --------------------------------------------------------------------------------

  function showError(error) {
    const banner = $("flat-error");
    banner.hidden = !error;
    banner.dataset.level = "bad";
    banner.textContent = error ? error.message : "";
  }

  function chip(word, level) {
    return h("span", { class: "chip", text: word, dataset: { level } });
  }

  function renderUse(library) {
    const line = FlatText.libraryLine(library);
    const pill = $("use-pill");
    pill.textContent = line.word;
    pill.dataset.level = line.level;
    $("use-text").textContent = line.text;
    $("temperature-line").textContent = FlatText.temperatureLine(library);
    const blocked = Boolean(library.blocker);
    $("blocker").hidden = !blocked;
    $("blocker-text").textContent = blocked ? FlatText.sentence(library.blocker) : "";
  }

  function renderPhases(task) {
    const list = clear($("phases"));
    const show = FlatText.isActive(task) || task.state === "ok";
    list.hidden = !show;
    if (!show) {
      return;
    }
    for (const phase of FlatText.phases(task)) {
      list.append(
        h(
          "li",
          { class: "phase", dataset: { state: phase.state }, "aria-current": phase.state === "active" ? "step" : null },
          h("span", { class: "mark", "aria-hidden": "true" }),
          h("span", { class: "phase-text" }, h("span", { class: "phase-name", text: phase.label }), phase.detail ? h("span", { class: "phase-detail", text: phase.detail }) : null)
        )
      );
    }
  }

  function renderLevel(task) {
    const gauge = FlatText.levelGauge(task);
    $("level-block").hidden = gauge === null;
    if (gauge === null) {
      return;
    }
    const word = $("level-word");
    word.textContent = gauge.word;
    word.dataset.level = gauge.level;
    $("level-text").textContent = gauge.text;
    const fill = $("level-fill");
    fill.style.width = gauge.fill + "%";
    fill.dataset.level = gauge.level;
    $("level-mark").style.left = "calc(" + gauge.mark + "% - 1px)";
  }

  function fillList(listId, items) {
    const list = clear($(listId));
    for (const text of items) {
      list.append(h("li", { text }));
    }
  }

  function renderResult(library) {
    const result = FlatText.result(library);
    const banner = $("result");
    banner.hidden = result === null;
    if (result === null) {
      return;
    }
    banner.dataset.level = result.level;
    $("result-title").textContent = result.title;
    $("result-text").textContent = result.text;
  }

  function renderTask(library) {
    const task = library.task;
    const active = FlatText.isActive(task);
    $("progress-empty").hidden = task.state !== "idle";
    renderPhases(task);
    renderLevel(task);
    $("task-message").textContent = active ? FlatText.progressMessage(task) : "";
    const notes = FlatText.notes(task);
    $("session-notes").hidden = notes.length === 0;
    fillList("session-notes-list", notes);
    renderResult(library);
  }

  function verdictRows(review) {
    const list = clear($("verdicts"));
    for (const item of review.items) {
      list.append(h("dt", { text: item.label }), h("dd", {}, h("span", { text: item.value }), " ", chip(item.word, item.level), h("span", { class: "note why", text: item.why })));
    }
  }

  function renderReview(library) {
    const flat = FlatText.pendingFlat(library);
    $("review-panel").hidden = flat === null;
    if (flat === null) {
      state.shownImage = null;
      return;
    }
    if (state.reviewing !== flat.version) {
      state.reviewing = flat.version; // a new flat to look at: the note of the last one is old news
      $("review-note").textContent = "";
    }
    const review = FlatText.verdicts(flat);
    const word = $("review-word");
    word.textContent = review.word;
    word.dataset.level = review.level;
    $("review-title").textContent = flat.version + ", made " + FlatText.ageText(flat.age_days) + (flat.second_set ? " from two sets" : " from one set");
    $("review-text").textContent = review.text;
    verdictRows(review);
    const notes = FlatText.flatNotes(flat);
    $("flat-notes").hidden = notes.length === 0;
    fillList("flat-notes-list", notes);
    const second = FlatText.canTakeSecondSet(library);
    $("second-set").hidden = !second;
    $("second-why").textContent = second ? FlatText.secondSetText(library) : "";
    if (flat.has_image && flat.image_url && state.shownImage !== flat.version) {
      state.shownImage = flat.version;
      showImage($("flat-image"), flat.image_url).catch(() => {
        state.shownImage = null;
        $("review-note").textContent = "The preview of the flat is not available.";
      });
    }
    $("flat-image").hidden = !flat.has_image;
  }

  /** A cell of the table. A phone shows the label in front of the value, and it skips an empty cell. */
  function cell(label, content, title, empty) {
    // A dataset turns a null into the text "null", so the mark of an empty cell is there or not.
    const dataset = empty ? { label, empty: "1" } : { label };
    return h("td", { role: "cell", title, dataset }, content);
  }

  const ACTION_LABELS = { use: "Use this flat", again: "Use again", discard: "Discard", delete: "Delete" };
  const DESTRUCTIVE = ["discard", "delete"];

  function isArmed(kind, version) {
    return Boolean(state.armed) && state.armed.key === kind + ":" + version;
  }

  /** The label of an action: a deletion that waits for its second press says so. */
  function labelOf(kind, version) {
    return isArmed(kind, version) ? "Press again to " + kind : ACTION_LABELS[kind];
  }

  function actionButton(kind, version) {
    const danger = DESTRUCTIVE.includes(kind);
    return h("button", { type: "button", class: danger ? "danger" : null, text: labelOf(kind, version), dataset: { action: kind, version }, onclick: () => ask(kind, version) });
  }

  /** The labels that depend on a waiting deletion: the table, and the Discard button of the review. */
  function refreshLabels() {
    if (state.library) {
      renderTable(state.library);
      const flat = FlatText.pendingFlat(state.library);
      $("flat-discard").textContent = flat ? labelOf("discard", flat.version) : ACTION_LABELS.discard;
    }
  }

  function disarm() {
    if (state.armed) {
      clearTimeout(state.armed.timer);
      state.armed = null;
    }
  }

  /**
   * Ask for a decision. A deletion cannot be undone, so the first press only changes the button
   * (and the note) and a second press within a few seconds deletes. Using a flat needs one press.
   */
  function ask(kind, version) {
    if (!DESTRUCTIVE.includes(kind)) {
      return decide(kind, version);
    }
    if (isArmed(kind, version)) {
      disarm();
      refreshLabels();
      return decide(kind, version);
    }
    disarm();
    const timer = setTimeout(() => {
      state.armed = null;
      refreshLabels();
    }, ARM_MS);
    state.armed = { key: kind + ":" + version, timer };
    $("command-note").textContent = "Press the button again within 5 seconds to " + kind + " the flat. A flat that you " + kind + " cannot come back.";
    refreshLabels();
    return undefined;
  }

  function renderTable(library) {
    const rows = FlatText.rows(library);
    $("flats-empty").hidden = rows.length > 0;
    $("flats-wrap").hidden = rows.length === 0;
    const body = clear($("flats-body"));
    for (const row of rows) {
      const actions = row.actions.map((kind) => actionButton(kind, row.version));
      body.append(
        h(
          "tr",
          { role: "row" },
          cell("Made", row.made, row.version),
          cell("State", chip(row.state.word, row.state.level)),
          cell("Corners", row.corners),
          cell("Dust", row.shadows),
          cell("Noise", row.noise),
          cell("Frames", row.sets),
          cell("Actions", h("span", { class: "row" }, actions), undefined, actions.length === 0)
        )
      );
    }
    lockTable();
  }

  /** The buttons of the table wait while a command is on its way or a session holds the library. */
  function lockTable() {
    const library = state.library;
    const locked = state.sending || !library || FlatText.isActive(library.task) || !commandsEnabled();
    for (const button of $("flats-body").querySelectorAll("button")) {
      button.disabled = locked;
    }
  }

  /** The buttons and the hints, which depend on the library, the status of the server, and the token. */
  function renderControls() {
    const library = state.library;
    const task = library ? library.task : null;
    const active = FlatText.isActive(task);
    const paused = schedulerState() === "paused";
    const enabled = commandsEnabled();
    const down = Boolean(state.failed) && state.failed.status === 503;
    const blocked = Boolean(library) && Boolean(library.blocker);
    $("flat-start").disabled = active || state.sending || !library || !enabled || down || blocked;
    $("flat-start").title = enabled ? "" : "The server has no token hash, so it refuses every command.";
    if (!enabled && $("command-note").textContent === "") {
      $("command-note").textContent = "The server has no API token configured, so it refuses every command.";
    }
    $("flat-stop").hidden = !active;
    $("flat-stop").disabled = state.sending;
    const ended = Boolean(task) && !active && task.state !== "idle";
    $("resume-notice").hidden = !(paused && ended);
    if (paused && ended && library) {
      const notice = FlatText.resumeNotice(library);
      $("resume-title").textContent = notice.title;
      $("resume-text").textContent = notice.text;
    }
    $("flat-resume").hidden = !paused || ended; // the notice below has its own button
    $("flat-resume").disabled = state.sending;
    $("start-hint").textContent = down ? "The server cannot reach core, so it cannot start a session now." : FlatText.startHint(schedulerState(), task);
    for (const id of ["flat-use", "flat-discard", "flat-second"]) {
      $(id).disabled = state.sending || active || !enabled || !library;
    }
    if (library) {
      if (!state.filled) {
        $("f-frames").value = String(FlatText.defaults.frames);
        $("f-target").value = String(FlatText.defaults.targetPercent);
        state.filled = true;
      }
      $("flat-note").textContent = active ? "A session is under way. This page reads it every 2 s." : "This page reads the library every 10 s.";
      const pending = FlatText.pendingFlat(library);
      $("flat-discard").textContent = pending ? labelOf("discard", pending.version) : ACTION_LABELS.discard;
    }
    lockTable();
  }

  function render(library) {
    renderUse(library);
    renderTask(library);
    renderReview(library);
    renderTable(library);
    renderControls();
  }

  // --- Reading ----------------------------------------------------------------------------------

  async function read() {
    let library;
    const mine = (state.readCount += 1);
    try {
      library = await api.get("flat");
    } catch (error) {
      state.failed = error;
      if (error.status === 503) {
        showError({ message: "The server cannot reach core, so it cannot show the flat library. Trying again." });
      } else if (error.status !== 401) {
        showError(error);
      }
      renderControls();
      return;
    }
    if (mine < state.applied) {
      return; // a newer answer arrived first
    }
    state.applied = mine;
    state.failed = null;
    showError(null);
    const before = state.previousTask;
    state.library = library;
    state.previousTask = library.task.state;
    if (before !== null && before !== library.task.state && library.task.state !== "queued") {
      $("command-note").textContent = ""; // the answer to the last command is old news now
    }
    render(library);
    if (before !== null && FlatText.isActive({ state: before }) && !FlatText.isActive(library.task)) {
      // The session ended, and the scheduler may have paused. Ask for its state now, not in 10 s.
      Status.load().catch(() => undefined);
    }
  }

  // --- Commands ---------------------------------------------------------------------------------

  const FIELD_IDS = { frames: ["f-frames", "e-frames"], target: ["f-target", "e-target"] };

  function showFieldErrors(errors) {
    let first = null;
    for (const [key, [inputId, errorId]] of Object.entries(FIELD_IDS)) {
      const message = errors[key] || "";
      $(errorId).hidden = message === "";
      $(errorId).textContent = message;
      $(inputId).setAttribute("aria-invalid", message === "" ? "false" : "true");
      if (message !== "" && first === null) {
        first = inputId;
      }
    }
    if (first !== null) {
      $("advanced").open = true;
      $(first).focus();
    }
  }

  /** Send a command. The commands need the token, and the answer says what changed. */
  async function command(method, path, body, busyText) {
    const note = $("command-note");
    if (!Token.has()) {
      note.textContent = "Enter the token first: the commands need it.";
      window.Seeing.openTokenPanel();
      return null;
    }
    state.sending = true;
    renderControls();
    note.textContent = busyText;
    try {
      const answer = method === "delete" ? await api.delete(path) : await api.post(path, body);
      note.textContent = FlatText.sentence(answer.message) + (answer.accepted === false ? " (not accepted)" : "");
      return answer;
    } catch (error) {
      if (error.status === 409 || error.status === 404) {
        note.textContent = FlatText.sentence(error.message);
      } else if (error.status === 429) {
        note.textContent = "Too many requests. Wait " + (error.retryAfter || 60) + " s and try again.";
      } else if (error.status === 401) {
        note.textContent = "The token is not right.";
      } else if (error.status === 403) {
        note.textContent = error.message;
      } else if (error.status === 503) {
        note.textContent = "Core does not answer, so nothing changed.";
      } else if (error.status === 422) {
        const found = FlatText.serverErrors(error);
        const fields = Object.keys(found).filter((key) => key in FIELD_IDS);
        note.textContent = fields.length > 0 ? "The server did not accept the values." : FlatText.sentence(error.message);
        showFieldErrors(found);
      } else {
        note.textContent = error.message;
      }
      return null;
    } finally {
      state.sending = false;
      await Promise.all([read(), Status.load().catch(() => undefined)]);
      renderControls();
    }
  }

  function formValues() {
    return { frames: $("f-frames").value, target: $("f-target").value, pauseAfter: $("f-pause").checked };
  }

  /**
   * Start a session. A paused scheduler holds a session back, so the page resumes it after the
   * session is queued: the owner asked for the flat now, and the session pauses the station again
   * when it ends.
   */
  async function start(setNumber) {
    const checked = FlatText.validate(formValues());
    showFieldErrors(checked.errors);
    if (!checked.ok) {
      $("command-note").textContent = "Fix the values first.";
      return;
    }
    const body = Object.assign({}, checked.body, { set_number: setNumber });
    const wasPaused = schedulerState() === "paused";
    const answer = await command("post", "flat/session", body, "Starting the session.");
    if (answer && answer.accepted && wasPaused) {
      const resumed = await command("post", "mode", { mode: "auto" }, "Resuming the scheduler so that the session can start.");
      $("command-note").textContent = resumed
        ? "The session is queued. The scheduler was paused, so the page resumed it. It pauses again when the session ends."
        : "The session is queued, but the scheduler is still paused. Press Resume to start it.";
    }
  }

  async function startFirst(event) {
    event.preventDefault();
    await start(1);
  }

  function stop() {
    return command("post", "flat/session/stop", {}, "Stopping the session.");
  }

  function resume() {
    return command("post", "mode", { mode: "auto" }, "Resuming the scheduler.");
  }

  /** Use or discard a flat. The review and the table share this. */
  async function decide(kind, version) {
    disarm();
    const using = kind === "use" || kind === "again";
    if (using) {
      await command("post", "flat/" + version + "/activate", {}, "Making the flat the one in use.");
    } else {
      await command("delete", "flat/" + version, undefined, "Deleting the flat.");
    }
  }

  function buildControls() {
    $("flat-form").addEventListener("submit", startFirst);
    $("flat-stop").addEventListener("click", stop);
    $("flat-resume").addEventListener("click", resume);
    $("notice-resume").addEventListener("click", resume);
    $("flat-second").addEventListener("click", () => start(2));
    $("flat-use").addEventListener("click", () => {
      const flat = state.library && FlatText.pendingFlat(state.library);
      return flat ? decide("use", flat.version) : undefined;
    });
    $("flat-discard").addEventListener("click", () => {
      const flat = state.library && FlatText.pendingFlat(state.library);
      return flat ? ask("discard", flat.version) : undefined;
    });
    window.addEventListener("seeing:status", renderControls);
    window.addEventListener("seeing:token", () => loop.now());
  }

  window.addEventListener("DOMContentLoaded", () => {
    window.Seeing.boot("flat");
    buildControls();
    loop = poller(read, () => FlatText.pollInterval(state.library ? state.library.task : null));
    loop.start();
  });
})();
