"use strict";

/*
 * The scenarios of the pure helpers of the UI: the formatting of values and the reading of a failed
 * response (static/js/common.js), and the tick and search functions of the plotter
 * (static/js/plot.js). The function takes the global
 * `Seeing` object that the two scripts build, a `test(name, fn)` function, and an `assert` object,
 * so that Node (helpers.test.js) and a browser console can both run it.
 */
module.exports = function scenarios(Seeing, test, assert) {
  const fmt = Seeing.fmt;
  const helpers = Seeing.plotHelpers;
  const DASH = "—";
  const MINUS = "−";
  const ARCSEC = "″";

  test("a number has a fixed count of digits, a real minus sign, and a dash when it is missing", () => {
    assert.equal(fmt.num(1.2345, 2), "1.23");
    assert.equal(fmt.num(1.5, 0), "2");
    assert.equal(fmt.num(-1.5, 1), MINUS + "1.5");
    assert.equal(fmt.num(0, 2), "0.00");
    assert.equal(fmt.num(null, 2), DASH);
    assert.equal(fmt.num(undefined, 2), DASH);
    assert.equal(fmt.num(NaN, 2), DASH);
    assert.equal(fmt.num(3), "3.0");
  });

  test("a signed number shows its plus sign, and a zero shows none", () => {
    assert.equal(fmt.signed(2, 1), "+2.0");
    assert.equal(fmt.signed(-2, 1), MINUS + "2.0");
    assert.equal(fmt.signed(0, 1), "0.0");
    assert.equal(fmt.signed(null, 1), DASH);
  });

  test("arcseconds carry their mark, and a missing value stays a dash", () => {
    assert.equal(fmt.arcsec(1.234), "1.23" + ARCSEC);
    assert.equal(fmt.arcsec(1.234, 1), "1.2" + ARCSEC);
    assert.equal(fmt.arcsec(null), DASH);
  });

  test("a fraction becomes a percentage", () => {
    assert.equal(fmt.percent(0.256), "26 %");
    assert.equal(fmt.percent(0.2564, 1), "25.6 %");
    assert.equal(fmt.percent(null), DASH);
  });

  test("a duration picks the unit that reads best", () => {
    assert.equal(fmt.duration(0), "0 s");
    assert.equal(fmt.duration(12), "12 s");
    assert.equal(fmt.duration(89), "89 s");
    assert.equal(fmt.duration(90), "2 min");
    assert.equal(fmt.duration(3600), "60 min");
    assert.equal(fmt.duration(5400), "1.5 h");
    assert.equal(fmt.duration(86400), "24.0 h");
    assert.equal(fmt.duration(200000), "2.3 d");
    assert.equal(fmt.duration(-5), "0 s");
    assert.equal(fmt.duration(null), DASH);
  });

  test("an age says ago, and a missing age stays a dash", () => {
    assert.equal(fmt.age(10), "10 s ago");
    assert.equal(fmt.age(null), DASH);
  });

  test("times are cut from the ISO text and stay in UTC", () => {
    assert.equal(fmt.clock("2026-10-01T03:04:05.000000Z"), "03:04:05");
    assert.equal(fmt.stamp("2026-10-01T03:04:05.000000Z"), "2026-10-01 03:04");
    assert.equal(fmt.clock(null), DASH);
    assert.equal(fmt.stamp(undefined), DASH);
  });

  test("a size uses bytes, kilobytes, or megabytes", () => {
    assert.equal(fmt.bytes(500), "500 B");
    assert.equal(fmt.bytes(2048), "2 kB");
    assert.equal(fmt.bytes(5 * 1024 * 1024), "5.0 MB");
    assert.equal(fmt.bytes(null), DASH);
  });

  test("the words of the API map to the three levels", () => {
    for (const word of ["healthy", "ok", "info"]) {
      assert.equal(fmt.level(word), "good");
    }
    for (const word of ["degraded", "warning"]) {
      assert.equal(fmt.level(word), "warn");
    }
    for (const word of ["failed", "error", "something else"]) {
      assert.equal(fmt.level(word), "bad");
    }
  });

  test("a reason of the health verdict becomes a sentence, and an unknown one stays as it is", () => {
    assert.equal(Seeing.explainReason("store_unreadable"), "The store cannot be read.");
    assert.equal(Seeing.explainReason("component_failed:camera"), "The component camera has failed.");
    assert.equal(Seeing.explainReason("component_degraded:store"), "The component store is degraded.");
    assert.equal(Seeing.explainReason("flag:low_space"), "Free disk space is low.");
    assert.equal(Seeing.explainReason("flag:unheard_of"), "The flag unheard_of is set.");
    assert.equal(Seeing.explainReason("brand_new_reason"), "brand_new_reason");
  });

  test("a failed response gives the message of the API error, or of a command that the scheduler rejected", () => {
    const failureFrom = Seeing.failureFrom;
    const apiError = {
      error: { code: "invalid_command", message: "The exposure is out of range.", details: [{ field: "exposure_s" }] },
    };
    assert.deepEqual(failureFrom(422, apiError), {
      code: "invalid_command",
      message: "The exposure is out of range.",
      details: [{ field: "exposure_s" }],
    });
    // The scheduler rejects a command with status 409 and the shape of a command answer.
    const rejected = {
      accepted: false,
      message: "the scheduler is paused; resume it first",
      reason: "paused",
      state: "paused",
      task_id: null,
    };
    assert.deepEqual(failureFrom(409, rejected), {
      code: "paused",
      message: "the scheduler is paused; resume it first",
      details: null,
    });
    // A rejection without a reason still has a code.
    assert.equal(failureFrom(409, { accepted: false, message: "not now" }).code, "rejected");
    // A body of another shape leaves the message with the status only.
    assert.equal(failureFrom(409, { accepted: false, message: "" }).message, "The request failed (409).");
    assert.equal(failureFrom(409, { accepted: true, message: "queued" }).message, "The request failed (409).");
    assert.equal(failureFrom(500, null).message, "The request failed (500).");
    assert.equal(failureFrom(502, "bad gateway").message, "The request failed (502).");
    assert.equal(failureFrom(502, { error: {} }).message, "The request failed (502).");
    assert.equal(failureFrom(502, { error: {} }).code, "error");
  });

  test("value ticks are round numbers inside the range", () => {
    const zero = helpers.valueTicks(0, 10, 4);
    assert.deepEqual(zero.ticks, [0, 2, 4, 6, 8, 10]);
    assert.equal(zero.step, 2);
    const seeing = helpers.valueTicks(1.0, 3.0, 4);
    assert.deepEqual(seeing.ticks, [1, 1.5, 2, 2.5, 3]);
    const small = helpers.valueTicks(0.1, 0.4, 3);
    assert.ok(small.ticks.every((t) => t >= 0.1 - 1e-9 && t <= 0.4 + 1e-9));
    const flat = helpers.valueTicks(5, 5, 4);
    assert.ok(flat.ticks.length >= 1);
  });

  test("time ticks sit on whole hours of UTC", () => {
    const t0 = Date.UTC(2026, 8, 30, 3, 0);
    const t1 = Date.UTC(2026, 9, 1, 3, 0);
    const day = helpers.timeTicks(t0, t1, 4);
    assert.equal(day.step, 6 * 3600e3);
    assert.deepEqual(
      day.ticks.map((t) => new Date(t).toISOString().slice(0, 16)),
      ["2026-09-30T06:00", "2026-09-30T12:00", "2026-09-30T18:00", "2026-10-01T00:00"]
    );
    const hour = helpers.timeTicks(t0, t0 + 3 * 3600e3, 6);
    assert.equal(hour.step, 1800e3);
    const month = helpers.timeTicks(t0, t0 + 30 * 86400e3, 6);
    assert.equal(month.step, 7 * 86400e3);
  });

  test("a time label shows the hour, and the date at midnight and for long steps", () => {
    const noon = Date.UTC(2026, 9, 1, 12, 0);
    const midnight = Date.UTC(2026, 9, 1, 0, 0);
    assert.equal(helpers.timeLabel(noon, 3600e3), "12:00");
    assert.equal(helpers.timeLabel(midnight, 3600e3), "10-01");
    assert.equal(helpers.timeLabel(noon, 86400e3), "10-01");
  });

  test("the nearest point is found by bisection, and a tie goes to the earlier point", () => {
    const points = [10, 20, 30, 50].map((t) => ({ t }));
    assert.equal(helpers.nearest(points, 0), 0);
    assert.equal(helpers.nearest(points, 14), 0);
    assert.equal(helpers.nearest(points, 16), 1);
    assert.equal(helpers.nearest(points, 25), 1);
    assert.equal(helpers.nearest(points, 41), 3);
    assert.equal(helpers.nearest(points, 1000), 3);
    assert.equal(helpers.nearest([{ t: 5 }], 99), 0);
  });

  test("the step of a tick is 1, 2, or 5 times a power of ten", () => {
    for (const span of [0.7, 3, 10, 24, 99, 1234]) {
      const step = helpers.niceStep(span, 4);
      const mantissa = step / Math.pow(10, Math.floor(Math.log10(step) + 1e-9));
      assert.ok([1, 2, 5, 10].some((m) => Math.abs(mantissa - m) < 1e-6), String(step));
    }
  });

  test("the flag chips name their meaning and mark the bad ones", () => {
    assert.ok(Seeing.FLAG_HELP.cloud.length > 0);
    assert.ok(Seeing.FLAG_HELP.saturated.length > 0);
  });
};
