"""The survey path: catalog, detection, plate solving, the pointing fit, and the sky quality.

The survey path turns long exposures into survey records. A star detector finds the stars,
a solver adapter (see `seeingmon.solvers`) gives a first solution, and a least-squares fit
refines it against catalog stars moved to the time of the frame (`seeingmon.survey.apparent`).
The sky quality step (`seeingmon.survey.quality`) then measures the stars (photometry and the
zero point), the sky (against the dark library and the flat field), the transparency, and the
clouds. `seeingmon.survey.analyzer.SurveyPipelineAnalyzer` ties the steps together behind the
`SurveyAnalyzer` interface, and `seeingmon.survey.tracker.PointingTracker` serves the Polaris
position to the scheduler.

Calibration lives in `seeingmon.survey.dark` (the dark library, the dark model, and
`dark_due`), and `seeingmon dark` records a set (`seeingmon.survey.dark_session`).
`seeingmon.survey.star_epoch` keeps the nightly star summary, and `seeingmon.survey.sqm_fit`
fits the SQM-LE offset at commissioning. `seeingmon flat make` (`seeingmon.survey.flat_make`)
builds the master flat from frames of a lit panel, `seeingmon flat build`
(`seeingmon.survey.flat_sky`) builds one from the night sky, and `seeingmon.survey.flat_report`
measures what a flat looks like.

Import the submodules you need. The package imports nothing at start-up, so the command-line
entry point stays fast.
"""

from __future__ import annotations
