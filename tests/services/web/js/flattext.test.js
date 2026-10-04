"use strict";

// Runs the scenarios of the logic of the Flat page (static/js/flattext.js) with the test runner of
// Node (node --test). The script reads `window.Seeing.fmt`, which common.js builds, and it touches
// the page only inside functions that the scenarios do not call, so a context with an empty
// `window` is enough.

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const scenarios = require("./flattext_scenarios.js");

const folder = path.join(__dirname, "..", "..", "..", "..", "src", "seeingmon", "services", "web", "static", "js");
const context = { window: {} };
vm.createContext(context);
for (const name of ["common.js", "flattext.js"]) {
  vm.runInContext(fs.readFileSync(path.join(folder, name), "utf8"), context, { filename: name });
}

// The scripts run in another realm, so their arrays have another prototype, and the strict deep
// comparison of Node would tell equal arrays apart. Compare plain copies instead.
const clone = (value) => JSON.parse(JSON.stringify(value));
const checks = {
  equal: assert.equal,
  notEqual: assert.notEqual,
  ok: assert.ok,
  deepEqual: (actual, expected) => assert.deepEqual(clone(actual), clone(expected)),
};

scenarios(context.window.Seeing.FlatText, test, checks);
