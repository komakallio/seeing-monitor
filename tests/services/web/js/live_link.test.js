"use strict";

// Runs the scenarios of the live link with the test runner of Node (node --test). The link is a
// browser script that sets `window.Seeing.LiveLink`, so the test loads it into a context that has
// the few globals it uses.

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const scenarios = require("./live_link_scenarios.js");

const source = fs.readFileSync(
  path.join(__dirname, "..", "..", "..", "..", "src", "seeingmon", "services", "web", "static", "js", "live.js"),
  "utf8"
);
const context = { window: { Seeing: {} }, setTimeout, clearTimeout, Blob, Date, Math, JSON, Promise, Error };
vm.runInNewContext(source, context);

scenarios(context.window.Seeing.LiveLink, test, assert);
