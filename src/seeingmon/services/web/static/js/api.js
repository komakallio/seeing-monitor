"use strict";

/*
 * The API reference page: it reads the OpenAPI document that the server builds from its routes and
 * lists the endpoints by tag, with their parameters and responses. The page needs no library, so
 * it works on a network with no access to the Internet.
 */
(function () {
  const { h, $, api } = window.Seeing;
  const TAG_ORDER = ["status", "records", "images", "commands", "alignment", "reference"];

  /** Text with `code` and **bold** spans, as elements. */
  function inline(text) {
    const nodes = [];
    const pattern = /(`[^`]+`|\*\*[^*]+\*\*)/g;
    let last = 0;
    for (const match of text.matchAll(pattern)) {
      if (match.index > last) {
        nodes.push(text.slice(last, match.index));
      }
      const piece = match[0];
      nodes.push(piece.startsWith("`") ? h("code", { text: piece.slice(1, -1) }) : h("strong", { text: piece.slice(2, -2) }));
      last = match.index + piece.length;
    }
    if (last < text.length) {
      nodes.push(text.slice(last));
    }
    return nodes;
  }

  function paragraphs(text) {
    return (text || "")
      .split(/\n\s*\n/)
      .map((chunk) => chunk.replace(/\s*\n\s*/g, " ").trim())
      .filter(Boolean)
      .map((chunk) => h("p", {}, inline(chunk)));
  }

  function typeOf(schema) {
    if (!schema) {
      return "";
    }
    if (schema.$ref) {
      return schema.$ref.split("/").pop();
    }
    if (schema.anyOf) {
      return schema.anyOf.map(typeOf).filter((t) => t !== "null").join(" or ");
    }
    if (schema.enum) {
      return schema.enum.join(" | ");
    }
    if (schema.type === "array") {
      return "list of " + typeOf(schema.items);
    }
    return schema.type || "";
  }

  function parameterTable(parameters) {
    if (!parameters || parameters.length === 0) {
      return null;
    }
    const rows = parameters.map((p) =>
      h(
        "tr",
        {},
        h("td", {}, h("code", { text: p.name })),
        h("td", { text: p.in + (p.required ? ", required" : "") }),
        h("td", { text: typeOf(p.schema) }),
        h("td", {}, inline(p.description || ""))
      )
    );
    return h(
      "div",
      { class: "table-wrap" },
      h("table", { class: "params" }, h("thead", {}, h("tr", {}, h("th", { text: "Parameter" }), h("th", { text: "Where" }), h("th", { text: "Type" }), h("th", { text: "Meaning" }))), h("tbody", {}, rows))
    );
  }

  function endpoint(method, path, operation) {
    const body = operation.requestBody;
    const responses = Object.entries(operation.responses || {}).map(([code, response]) =>
      h("li", {}, h("code", { text: code }), " ", response.description || "")
    );
    const secured = Boolean(operation.security && operation.security.length);
    return h(
      "details",
      { class: "endpoint" },
      h(
        "summary",
        {},
        h("span", { class: "method", text: method.toUpperCase() }),
        h("span", { class: "path", text: path }),
        h("span", { class: "note", text: operation.summary || "" }),
        secured ? h("span", { class: "chip", text: "token" }) : null
      ),
      ...paragraphs(operation.description),
      parameterTable(operation.parameters),
      body ? h("p", { class: "note" }, "Request body: ", h("code", { text: typeOf(((body.content || {})["application/json"] || {}).schema) || "JSON" })) : null,
      h("ul", { class: "note" }, responses)
    );
  }

  async function main() {
    window.Seeing.boot("api");
    let document_;
    try {
      document_ = await api.get("openapi.json");
    } catch (error) {
      const banner = $("api-error");
      banner.hidden = false;
      banner.dataset.level = "bad";
      banner.textContent = error.message;
      return;
    }
    const intro = $("intro");
    intro.append(h("h2", { text: document_.info.title + " " + document_.info.version }), ...paragraphs(document_.info.description));
    const groups = new Map();
    for (const [path, item] of Object.entries(document_.paths)) {
      for (const [method, operation] of Object.entries(item)) {
        if (!operation || typeof operation !== "object" || !operation.responses) {
          continue;
        }
        const tag = (operation.tags && operation.tags[0]) || "other";
        if (!groups.has(tag)) {
          groups.set(tag, []);
        }
        groups.get(tag).push(endpoint(method, path, operation));
      }
    }
    const rank = (tag) => (TAG_ORDER.includes(tag) ? TAG_ORDER.indexOf(tag) : 99);
    const tags = [...groups.keys()].sort((a, b) => rank(a) - rank(b));
    const descriptions = Object.fromEntries((document_.tags || []).map((t) => [t.name, t.description]));
    const box = $("endpoints");
    for (const tag of tags) {
      box.append(h("section", { class: "panel" }, h("h2", { text: tag }), h("p", { class: "note", text: descriptions[tag] || "" }), ...groups.get(tag)));
    }
  }

  window.addEventListener("DOMContentLoaded", main);
})();
