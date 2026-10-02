"use strict";

// Runs the scenarios of the sky overlay geometry (static/js/skygrid.js) with the test runner of
// Node (node --test). The script is a browser script that sets `window.Seeing.SkyGrid`, so the
// test loads it into a context with an empty `window`. The golden numbers come from the file
// `skygrid_fixture.json`, which a Python test keeps equal to what `CameraAttitude.project` gives.

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const scenarios = require("./skygrid_scenarios.js");

const source = fs.readFileSync(
  path.join(__dirname, "..", "..", "..", "..", "src", "seeingmon", "services", "web", "static", "js", "skygrid.js"),
  "utf8"
);
const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, "skygrid_fixture.json"), "utf8"));
const context = { window: { Seeing: {} } };
vm.runInNewContext(source, context, { filename: "skygrid.js" });

// The script runs in another realm, so its arrays have another prototype, and the strict deep
// comparison of Node would tell equal arrays apart. Compare plain copies instead.
const clone = (value) => JSON.parse(JSON.stringify(value));
const checks = {
  equal: assert.equal,
  ok: assert.ok,
  deepEqual: (actual, expected) => assert.deepEqual(clone(actual), clone(expected)),
};

scenarios(context.window.Seeing.SkyGrid, test, checks, fixture);
