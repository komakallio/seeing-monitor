"use strict";

// Runs the scenarios of the pure helpers of the UI with the test runner of Node (node --test).
// The two scripts set `window.Seeing`, and they touch the page only inside functions that the
// scenarios do not call, so a context with an empty `window` is enough.

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const scenarios = require("./helpers_scenarios.js");

const folder = path.join(__dirname, "..", "..", "..", "..", "src", "seeingmon", "services", "web", "static", "js");
const context = { window: {} };
vm.createContext(context);
for (const name of ["common.js", "plot.js"]) {
  vm.runInContext(fs.readFileSync(path.join(folder, name), "utf8"), context, { filename: name });
}

// The scripts run in another realm, so their arrays have another prototype, and the strict deep
// comparison of Node would tell equal arrays apart. Compare plain copies instead.
const clone = (value) => JSON.parse(JSON.stringify(value));
const checks = {
  equal: assert.equal,
  ok: assert.ok,
  deepEqual: (actual, expected) => assert.deepEqual(clone(actual), clone(expected)),
};

scenarios(context.window.Seeing, test, checks);
