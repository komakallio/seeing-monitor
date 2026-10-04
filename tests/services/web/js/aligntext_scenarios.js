"use strict";

/*
 * The scenarios of the words of the Align page (static/js/aligntext.js): the sentences of the pole
 * card, the three states of the offset card, and the lines of target settings. The function takes
 * `AlignText`, a `test(name, fn)` function, and an `assert` object, so that Node
 * (aligntext.test.js) and a browser console can both run it.
 */
module.exports = function scenarios(AlignText, test, assert) {
  const MINUS = "−";
  const FRAME = { width_px: 4144, height_px: 2822 };

  /** A sky view with a pole 21 arcminutes from the center, 14 right and 16 down. */
  function sky(pole, orbit, extra) {
    return Object.assign(
      {
        camera: { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], scale_arcsec_px: 3.82, parity: 1, center_x_px: 2071.5, center_y_px: 1410.5 },
        pole: Object.assign(
          { in_front: true, x_px: 2291.5, y_px: 1662.5, inside_frame: true, dx_px: 220, dy_px: 252, distance_px: 334.5, distance_arcmin: 21.3, roll_deg: -41 },
          pole
        ),
        polaris_colatitude_deg: 0.62,
        orbit: Object.assign({ fits: true, margin_px: 835, margin_arcmin: 53.2 }, orbit),
      },
      extra
    );
  }

  // --- The pole card ------------------------------------------------------------------------------

  test("the pole sentence says the move in altitude and azimuth, and falls back to the image directions", () => {
    assert.equal(AlignText.poleSentence(sky()), "The pole is 21′ from the center: 14′ right and 16′ down.");
    assert.equal(
      AlignText.poleSentence(sky({ dx_px: -220, dy_px: -252 })),
      "The pole is 21′ from the center: 14′ left and 16′ up."
    );
    // With the site, positive altitude means raise and positive azimuth means east.
    assert.equal(
      AlignText.poleSentence(sky({}, null, { altitude_arcmin: 21, azimuth_arcmin: 8 })),
      "Raise the camera by 21′ in altitude and turn it 8′ toward the east in azimuth."
    );
    assert.equal(
      AlignText.poleSentence(sky({}, null, { altitude_arcmin: -21, azimuth_arcmin: -8 })),
      "Lower the camera by 21′ in altitude and turn it 8′ toward the west in azimuth."
    );
    // The image directions run from the aim when the state names one.
    assert.equal(
      AlignText.poleSentence(sky({}, null, { aim: { x_px: 2071.5, y_px: 1410.5, dx_px: 220, dy_px: 252, distance_arcmin: 21.3 } })),
      "The pole is 21′ from the aim: 14′ right and 16′ down."
    );
  });

  test("a component that rounds to nothing stays out of the sentence", () => {
    assert.equal(
      AlignText.poleSentence(sky({ dx_px: 50, dy_px: 0, distance_arcmin: 3.2 })),
      "The pole is 3′ from the center: 3′ right."
    );
    assert.equal(
      AlignText.poleSentence(sky({ dx_px: 0, dy_px: -157, distance_arcmin: 10 })),
      "The pole is 10′ from the center: 10′ up."
    );
  });

  test("a pole within two arcminutes of the aim is aligned", () => {
    assert.equal(AlignText.poleSentence(sky({ dx_px: 3, dy_px: -4, distance_arcmin: 0.4 })), "Aligned: the pole is within 2′ of the center.");
    assert.equal(
      AlignText.poleSentence(sky({}, null, { aim: { x_px: 2071.5, y_px: 1410.5, dx_px: 20, dy_px: -10, distance_arcmin: 1.9 }, altitude_arcmin: 1.5, azimuth_arcmin: 1.0 })),
      "Aligned: the pole is within 2′ of the aim."
    );
  });

  test("a far pole is described in degrees, and a pole outside the frame says so", () => {
    assert.equal(
      AlignText.poleSentence(sky({ dx_px: 3300, dy_px: -1000, distance_arcmin: 220, inside_frame: false })),
      "The pole is 3.7° from the center: 3.5° right and 1.1° up. It lies outside the frame."
    );
    assert.equal(
      AlignText.poleSentence(sky({ dx_px: 0, dy_px: 0, distance_arcmin: 0.2, inside_frame: false })),
      "Aligned: the pole is within 2′ of the center. It lies outside the frame."
    );
  });

  test("a pole behind the camera, and a view that is missing, have their own words", () => {
    assert.equal(
      AlignText.poleSentence(sky({ in_front: false, x_px: null, y_px: null, dx_px: null, dy_px: null, distance_px: null, distance_arcmin: 7200 })),
      "The pole is behind the camera, so the camera points away from it."
    );
    assert.equal(AlignText.poleSentence(null), null);
    assert.equal(AlignText.poleSentence(sky({ distance_arcmin: null })), "The position of the pole is not known.");
  });

  test("the orbit is good, tight, or bad by its margin, and none without a colatitude", () => {
    assert.equal(AlignText.orbitState(sky(), FRAME), "good");
    assert.equal(AlignText.orbitState(sky({}, { margin_px: 141.2 }), FRAME), "good"); // 5 % of 2822 is 141.1
    assert.equal(AlignText.orbitState(sky({}, { margin_px: 141.0 }), FRAME), "tight");
    assert.equal(AlignText.orbitState(sky({}, { margin_px: 3 }), FRAME), "tight");
    assert.equal(AlignText.orbitState(sky({}, { fits: false, margin_px: -40 }), FRAME), "bad");
    assert.equal(AlignText.orbitState(sky({}, null, { orbit: null, polaris_colatitude_deg: null }), FRAME), "none");
    assert.equal(AlignText.orbitState(null, FRAME), "none");
  });

  test("the orbit sentences say whether the circle fits, and what to do when it does not", () => {
    assert.deepEqual(AlignText.orbitSentences(sky(), FRAME), [
      "Polaris moves on a circle of 0.62° around the pole. The circle fits in the frame with 53′ to spare.",
    ]);
    assert.deepEqual(AlignText.orbitSentences(sky({}, { margin_px: 78, margin_arcmin: 5 }), FRAME), [
      "Polaris moves on a circle of 0.62° around the pole. The circle fits in the frame, but with only 5′ to spare. Move the field so that the pole goes toward the center.",
    ]);
    assert.deepEqual(AlignText.orbitSentences(sky({}, { fits: false, margin_px: -142, margin_arcmin: -9.04 }), FRAME), [
      "Polaris moves on a circle of 0.62° around the pole. The circle leaves the frame by 9′. Move the field so that the pole goes toward the center.",
    ]);
  });

  test("a margin under half an arcminute is a margin of less than one arcminute", () => {
    assert.deepEqual(AlignText.orbitSentences(sky({}, { margin_px: 3, margin_arcmin: 0.2 }), FRAME), [
      "Polaris moves on a circle of 0.62° around the pole. The circle fits in the frame, but with less than 1′ to spare. Move the field so that the pole goes toward the center.",
    ]);
  });

  test("a large margin is written in degrees, and a missing arcminute margin comes from the pixels", () => {
    assert.ok(AlignText.orbitSentences(sky({}, { margin_px: 1000, margin_arcmin: 64 }), FRAME)[0].endsWith("with 1.1° to spare."));
    const fromPixels = AlignText.orbitSentences(sky({}, { margin_px: 836, margin_arcmin: null }), FRAME)[0];
    assert.ok(fromPixels.endsWith("with 53′ to spare."), fromPixels);
  });

  test("without a colatitude the page says that the orbit is not known", () => {
    assert.deepEqual(AlignText.orbitSentences(sky({}, null, { orbit: null, polaris_colatitude_deg: null }), FRAME), [
      "The circle that Polaris follows around the pole is not known for this frame.",
    ]);
  });

  test("the canvas labels for the orbit and for the arrow are short", () => {
    assert.equal(AlignText.orbitLabel("good"), "orbit: fits");
    assert.equal(AlignText.orbitLabel("tight"), "orbit: tight");
    assert.equal(AlignText.orbitLabel("bad"), "orbit: out");
    assert.equal(AlignText.orbitLabel("none"), "");
    assert.equal(AlignText.poleArrowText(sky({ distance_arcmin: 85 }), "above"), "pole 1.4° above");
    assert.equal(AlignText.poleArrowText(sky({ distance_arcmin: 40 }), "right"), "pole 40′ right");
    assert.equal(AlignText.poleArrowText(null, "left"), "pole");
  });

  test("the pole card gives the reason when there is no solution", () => {
    assert.equal(AlignText.poleReason({ quality: { sky: "no solve has finished yet" } }), "The pole needs a solution of the star field. No solve has finished yet.");
    assert.equal(AlignText.poleReason({ quality: {} }), "The pole needs a solution of the star field.");
    assert.equal(AlignText.COORDINATES, "Coordinates are of date.");
  });

  // --- The offset card ----------------------------------------------------------------------------

  const SOLVED = { x_px: 2075.34, y_px: 1400.18, roll_deg: 12.34, n_matched: 40, rms_arcsec: 0.8, age_s: 0.4 };
  const TARGET = { x_px: 2072, y_px: 1411, roll_deg: 10 };
  const OFFSET = { dx_px: 3, dy_px: -11, distance_px: 11.4, dx_arcsec: 11.5, dy_arcsec: -42, distance_arcsec: 43.5, roll_deg: 2 };

  test("without a solution the offset card says so, and gives the reason", () => {
    const card = AlignText.offsetCard({ solved: null, target: null, offset: null, quality: { solved: "no solve has finished yet" } });
    assert.equal(card.state, "none");
    assert.deepEqual(card.rows, [["Offset", "no solution yet"]]);
    assert.equal(card.note, "No solve has finished yet.");
    assert.equal(card.roll, null);
    const withTarget = AlignText.offsetCard({ solved: null, target: TARGET, offset: null, quality: {} });
    assert.equal(withTarget.state, "none");
    assert.equal(withTarget.note, "");
    assert.equal(AlignText.offsetCard(null).state, "none");
  });

  test("with a solution and no target the card says that no target is set, and gives Polaris", () => {
    const card = AlignText.offsetCard({ solved: SOLVED, target: null, offset: null, quality: {} });
    assert.equal(card.state, "untargeted");
    assert.deepEqual(card.rows, [
      ["Target", "No target is set"],
      ["Polaris", "x 2075.3, y 1400.2 px"],
    ]);
    assert.ok(card.note.includes("[alignment]"));
    assert.ok(card.note.includes("local configuration"));
    assert.ok(card.note.includes("optional"));
    assert.deepEqual(card.roll, {
      text: "Camera roll: 12.3° from image up toward image left. The altitude and azimuth adjustments cannot correct it.",
    });
  });

  test("a solution with no roll says that the roll is not defined", () => {
    const card = AlignText.offsetCard({ solved: Object.assign({}, SOLVED, { roll_deg: null }), target: null, offset: null, quality: {} });
    assert.equal(card.rows.length, 2);
    assert.equal(
      card.roll.text,
      "Camera roll: not defined while the pole sits at the center of the frame. The altitude and azimuth adjustments cannot correct it."
    );
  });

  test("with a solution and a target the card gives the offset and the roll line", () => {
    const card = AlignText.offsetCard({ solved: SOLVED, target: TARGET, offset: OFFSET, quality: {} });
    assert.equal(card.state, "targeted");
    assert.deepEqual(card.rows, [
      ["Horizontal (x)", "+3.0 px, +11.5″"],
      ["Vertical (y)", MINUS + "11.0 px, " + MINUS + "42.0″"],
      ["Distance", "11.4 px, 43.5″"],
    ]);
    assert.equal(card.note, "");
    assert.deepEqual(card.roll, {
      text: "Camera roll: 12.3° from image up toward image left. The altitude and azimuth adjustments cannot correct it.",
    });
  });

  test("the roll is information only: it never asks for a correction, whatever the roll offset is", () => {
    for (const rollOffset of [0.05, 2, -170, null]) {
      const card = AlignText.offsetCard({ solved: SOLVED, target: TARGET, offset: Object.assign({}, OFFSET, { roll_deg: rollOffset }), quality: {} });
      assert.equal(card.rows.length, 3);
      assert.ok(card.roll.text.includes("cannot correct it"));
      assert.ok(!/rotate/i.test(card.roll.text));
    }
  });

  // --- The target settings ------------------------------------------------------------------------

  test("the target settings are the TOML lines of the current solution", () => {
    assert.equal(
      AlignText.targetSettings(SOLVED),
      "[alignment]\ntarget_x_px = 2075.34\ntarget_y_px = 1400.18\ntarget_roll_deg = 12.340\n"
    );
  });

  test("the target settings leave the roll out when it is not defined, and use a plain minus", () => {
    assert.equal(
      AlignText.targetSettings({ x_px: 10.25, y_px: -3.5, roll_deg: null }),
      "[alignment]\ntarget_x_px = 10.25\ntarget_y_px = -3.50\n"
    );
    assert.ok(AlignText.targetSettings({ x_px: 1, y_px: 2, roll_deg: -170.25 }).includes("target_roll_deg = -170.250"));
    assert.equal(AlignText.targetSettings(null), "");
  });
};
