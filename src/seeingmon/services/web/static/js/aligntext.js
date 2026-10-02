"use strict";

/*
 * The words of the Align page: the sentences of the pole card and the rows of the offset card,
 * built from the alignment state. The functions are pure (they read numbers and return text), so
 * Node can run them (tests/services/web/js/). `align.js` puts the text into the page.
 *
 * Directions are image directions, as the person sees them on the screen: the pole is "right" when
 * it lies to the right of the center of the image, and "down" when it lies below it. They are not
 * compass directions. Coordinates are of date, and the page says so.
 */
(function () {
  const { fmt } = window.Seeing;

  const PRIME = "′";
  const DEGREE = "°";
  // The orbit is comfortable when its margin is at least this share of the shorter side.
  const TIGHT_FRACTION = 0.05;

  /** Arcminutes as whole numbers, such as 9′ and 21′. Anything under half an arcminute is 0′. */
  function arcminutes(value) {
    return String(Math.round(Math.abs(value))) + PRIME;
  }

  function degrees(arcmin, digits) {
    return fmt.num(Math.abs(arcmin) / 60, digits === undefined ? 1 : digits) + DEGREE;
  }

  /** An angle given in arcminutes, in degrees when `inDegrees` is true. */
  function angle(arcmin, inDegrees) {
    return inDegrees ? degrees(arcmin) : arcminutes(arcmin);
  }

  const COORDINATES = "Coordinates are of date.";

  // --- The pole card ----------------------------------------------------------------------------

  /** The offset of the pole from the center as a sentence, or `null` without a sky view. */
  function poleSentence(sky) {
    if (!sky) {
      return null;
    }
    const pole = sky.pole;
    if (!pole.in_front) {
      return "The pole is behind the camera, so the camera points away from it.";
    }
    const distance = pole.distance_arcmin;
    if (distance === null || distance === undefined || pole.dx_px === null || pole.dy_px === null) {
      return "The position of the pole is not known.";
    }
    const outside = pole.inside_frame ? "" : " It lies outside the frame.";
    if (distance < 1) {
      return "The pole is within 1′ of the center." + outside;
    }
    const inDegrees = distance >= 60;
    const scale = sky.camera.scale_arcsec_px;
    const parts = [];
    const across = (pole.dx_px * scale) / 60;
    const down = (pole.dy_px * scale) / 60;
    if (Math.abs(across) >= 0.05) {
      parts.push(angle(across, inDegrees) + (across > 0 ? " right" : " left"));
    }
    if (Math.abs(down) >= 0.05) {
      parts.push(angle(down, inDegrees) + (down > 0 ? " down" : " up"));
    }
    return "The pole is " + angle(distance, inDegrees) + " from the center: " + parts.join(" and ") + "." + outside;
  }

  /** `none` (no orbit known), `good`, `tight` (fits with a thin margin), or `bad` (leaves the frame). */
  function orbitState(sky, frame) {
    if (!sky || !sky.orbit) {
      return "none";
    }
    if (!sky.orbit.fits) {
      return "bad";
    }
    const shorter = frame ? Math.min(frame.width_px, frame.height_px) : 0;
    return sky.orbit.margin_px >= TIGHT_FRACTION * shorter ? "good" : "tight";
  }

  /** The sentences about the circle that Polaris follows, and whether it fits in the frame. */
  function orbitSentences(sky, frame) {
    const state = orbitState(sky, frame);
    if (state === "none" || sky.polaris_colatitude_deg === null || sky.polaris_colatitude_deg === undefined) {
      return ["The circle that Polaris follows around the pole is not known for this frame."];
    }
    const orbit = sky.orbit;
    const lead = "Polaris moves on a circle of " + fmt.num(sky.polaris_colatitude_deg, 2) + DEGREE + " around the pole.";
    const marginArcmin =
      orbit.margin_arcmin === null || orbit.margin_arcmin === undefined
        ? (orbit.margin_px * sky.camera.scale_arcsec_px) / 60
        : orbit.margin_arcmin;
    const size = Math.abs(marginArcmin);
    const amount = size < 0.5 ? "less than 1" + PRIME : angle(size, size >= 60);
    const move = " Move the field so that the pole goes toward the center.";
    if (state === "good") {
      return [lead + " The circle fits in the frame with " + amount + " to spare."];
    }
    if (state === "tight") {
      return [lead + " The circle fits in the frame, but with " + (size < 0.5 ? "" : "only ") + amount + " to spare." + move];
    }
    return [lead + " The circle leaves the frame by " + amount + "." + move];
  }

  /** The short word that the canvas shows next to the orbit. */
  function orbitLabel(state) {
    return { good: "orbit: fits", tight: "orbit: tight", bad: "orbit: out", none: "" }[state] || "";
  }

  /** The label of the arrow for a pole outside the frame, such as `pole 1.4° above`. */
  function poleArrowText(sky, side) {
    const distance = sky && sky.pole ? sky.pole.distance_arcmin : null;
    if (distance === null || distance === undefined) {
      return "pole";
    }
    return "pole " + angle(distance, distance >= 60) + " " + side;
  }

  /** What the pole card says when there is no sky view, with the reason from `quality`. */
  function poleReason(state) {
    const reason = state && state.quality ? state.quality.sky : "";
    return "The pole needs a solution of the star field." + (reason ? " " + reason.charAt(0).toUpperCase() + reason.slice(1) + "." : "");
  }

  // --- The offset card --------------------------------------------------------------------------

  const TARGET_NOTE =
    "The target comes from [alignment] in the local configuration. When the camera sits where you want it, copy the target settings below into that file.";

  /**
   * The rows and the notes of the offset card. The card has three states:
   *
   * - `none`: no solution yet, so there is nothing to compare.
   * - `untargeted`: a solution without a target. The card gives the position and the roll of
   *   Polaris, and says where the target comes from.
   * - `targeted`: a solution and a target. The card gives the offset, and the roll gauge.
   *
   * The result is `{ state, rows, note, roll }`, where `roll` is `null` or
   * `{ value, text }` for the gauge.
   */
  function offsetCard(state) {
    const solved = state && state.solved;
    if (!solved) {
      const reason = state && state.quality ? state.quality.solved : "";
      return {
        state: "none",
        rows: [["Offset", "no solution yet"]],
        note: reason ? reason.charAt(0).toUpperCase() + reason.slice(1) + "." : "",
        roll: null,
      };
    }
    const offset = state.offset;
    if (!state.target || !offset) {
      return {
        state: "untargeted",
        rows: [
          ["Target", "No target is set"],
          ["Polaris", "x " + fmt.num(solved.x_px, 1) + ", y " + fmt.num(solved.y_px, 1) + " px"],
          ["Roll", solved.roll_deg === null || solved.roll_deg === undefined ? fmt.dash : fmt.num(solved.roll_deg, 2) + DEGREE],
        ],
        note: TARGET_NOTE,
        roll: null,
      };
    }
    const roll = offset.roll_deg || 0;
    return {
      state: "targeted",
      rows: [
        ["Horizontal (x)", fmt.signed(offset.dx_px, 1) + " px, " + fmt.signed(offset.dx_arcsec, 1) + "″"],
        ["Vertical (y)", fmt.signed(offset.dy_px, 1) + " px, " + fmt.signed(offset.dy_arcsec, 1) + "″"],
        ["Distance", fmt.num(offset.distance_px, 1) + " px, " + fmt.num(offset.distance_arcsec, 1) + "″"],
        ["Roll", offset.roll_deg === null || offset.roll_deg === undefined ? fmt.dash : fmt.signed(offset.roll_deg, 2) + DEGREE],
      ],
      note: "",
      roll: {
        value: roll,
        text:
          Math.abs(roll) < 0.1
            ? "The roll is within 0.1 degrees of the target."
            : "Rotate the camera by " + fmt.signed(-roll, 2) + " degrees to match the target roll.",
      },
    };
  }

  /**
   * The TOML lines for the `[alignment]` table that match the current solution, or an empty text
   * without one. The roll stays out when it is not defined. TOML wants a plain minus sign.
   */
  function targetSettings(solved) {
    if (!solved) {
      return "";
    }
    const lines = ["[alignment]", "target_x_px = " + solved.x_px.toFixed(2), "target_y_px = " + solved.y_px.toFixed(2)];
    if (solved.roll_deg !== null && solved.roll_deg !== undefined) {
      lines.push("target_roll_deg = " + solved.roll_deg.toFixed(3));
    }
    return lines.join("\n") + "\n";
  }

  window.Seeing.AlignText = {
    COORDINATES, TARGET_NOTE, TIGHT_FRACTION, poleSentence, orbitState, orbitSentences, orbitLabel,
    poleArrowText, poleReason, offsetCard, targetSettings, arcminutes, degrees, angle,
  };
})();
