"use strict";

/*
 * The Images page: a grid of the newest preview images, a viewer for one image with the links to
 * download the JPEG and the FITS frame, and a button for older images. The viewer is an element
 * of the page and not a dialog element, because a dialog sits above the red night filter and
 * would show its colors.
 */
(function () {
  const { h, $, fmt, api, Status, showImage } = window.Seeing;
  const PAGE_SIZE = 24;

  const state = { items: [], cursor: null, open: -1, loading: false, opener: null };

  function label(item) {
    return fmt.stamp(item.t_utc) + " UTC, " + item.kind;
  }

  function showError(error) {
    const banner = $("images-error");
    banner.hidden = !error;
    banner.dataset.level = "bad";
    banner.textContent = error ? (error.status === 401 ? "The server asks for the token. Enter it above." : error.message) : "";
  }

  // --- The grid ---------------------------------------------------------------------------------

  function thumb(item, index) {
    const image = h("img", { alt: label(item), loading: "lazy", width: 480, height: 327 });
    showImage(image, item.preview_url).catch(() => undefined);
    return h(
      "button",
      {
        class: "thumb",
        type: "button",
        "aria-label": "Open the image of " + label(item),
        onclick: (event) => openViewer(index, event.currentTarget),
      },
      image,
      h("span", { class: "cap" }, h("span", { text: fmt.clock(item.t_utc) }), h("span", { text: item.kind + (item.has_fits ? " + FITS" : "") }))
    );
  }

  function render() {
    const grid = $("grid");
    const already = grid.children.length;
    for (let index = already; index < state.items.length; index += 1) {
      grid.append(thumb(state.items[index], index));
    }
    $("images-empty").hidden = state.items.length > 0;
    $("older").hidden = !state.cursor;
    $("images-note").textContent = state.items.length + " images, newest first, times in UTC";
  }

  async function loadMore() {
    if (state.loading) {
      return;
    }
    state.loading = true;
    showError(null);
    try {
      if (!Status.current) {
        await Status.load();
      }
      const page = await api.get("images", { limit: PAGE_SIZE, cursor: state.cursor });
      state.items.push(...page.items);
      state.cursor = page.next_cursor;
      render();
    } catch (error) {
      showError(error);
      $("images-note").textContent = "";
    } finally {
      state.loading = false;
    }
  }

  // --- The viewer -------------------------------------------------------------------------------

  async function download(url, name) {
    try {
      const response = await api.raw(url);
      const blob = await response.blob();
      const link = h("a", { href: URL.createObjectURL(blob), download: name });
      document.body.append(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(link.href), 10000);
    } catch (error) {
      showError(error);
    }
  }

  function downloadLink(url, name, text) {
    const status = Status.current;
    if (status && status.ui.token_required_for_reads) {
      return h("button", { type: "button", text, onclick: () => download(url, name) });
    }
    return h("a", { class: "button", href: url, download: name, text });
  }

  function closeViewer() {
    const box = $("viewer");
    if (box) {
      box.remove();
    }
    document.removeEventListener("keydown", onKey);
    const opener = state.opener;
    state.open = -1;
    if (opener && opener.isConnected) {
      opener.focus();
    }
  }

  function onKey(event) {
    if (event.key === "Escape") {
      closeViewer();
    } else if (event.key === "ArrowLeft") {
      step(-1);
    } else if (event.key === "ArrowRight") {
      step(1);
    }
  }

  async function step(delta) {
    const next = state.open + delta;
    if (next >= state.items.length && state.cursor) {
      await loadMore();
    }
    if (next >= 0 && next < state.items.length) {
      openViewer(next, state.opener);
    }
  }

  function openViewer(index, opener) {
    const item = state.items[index];
    state.open = index;
    state.opener = opener;
    const existing = $("viewer");
    if (existing) {
      existing.remove();
    }
    const image = h("img", { alt: "The image of " + label(item) });
    showImage(image, item.preview_url).catch((error) => showError(error));
    const name = item.id + ".jpg";
    const links = [downloadLink(item.preview_url, name, "Download JPEG")];
    if (item.has_fits) {
      links.push(downloadLink(item.fits_url, item.id.replace(/^[a-z0-9_]+-/, "") + ".fits", "Download FITS (" + fmt.bytes(item.fits_bytes) + ")"));
    }
    const closeButton = h("button", { type: "button", text: "Close", onclick: closeViewer });
    const box = h(
      "div",
      { id: "viewer", class: "lightbox", role: "dialog", "aria-modal": "true", "aria-label": "Image viewer" },
      h("div", { class: "stage" }, image),
      h(
        "div",
        { class: "bar" },
        h("span", { text: label(item) + ", " + fmt.bytes(item.size_bytes) + ", image " + (index + 1) + " of " + state.items.length + (state.cursor ? "+" : "") }),
        h(
          "div",
          { class: "row" },
          h("button", { type: "button", text: "Newer", disabled: index === 0, onclick: () => step(-1) }),
          h("button", { type: "button", text: "Older", disabled: index === state.items.length - 1 && !state.cursor, onclick: () => step(1) }),
          ...links,
          closeButton
        )
      )
    );
    box.addEventListener("click", (event) => {
      if (event.target === box || event.target.classList.contains("stage")) {
        closeViewer();
      }
    });
    document.body.append(box);
    document.removeEventListener("keydown", onKey);
    document.addEventListener("keydown", onKey);
    closeButton.focus();
  }

  window.addEventListener("DOMContentLoaded", () => {
    window.Seeing.boot("images");
    $("older").addEventListener("click", () => loadMore());
    window.addEventListener("seeing:token", () => {
      $("grid").replaceChildren();
      state.items = [];
      state.cursor = null;
      loadMore();
    });
    loadMore();
  });

})();
