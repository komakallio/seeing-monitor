"""The survey path: catalog, detection, plate solving, the pointing fit, and the tracker.

The survey path turns long exposures into pointing records. A star detector finds the stars,
a solver adapter (see `seeingmon.solvers`) gives a first solution, and a least-squares fit
refines it against catalog stars moved to the time of the frame (`seeingmon.survey.apparent`).
`seeingmon.survey.analyzer.SurveyPipelineAnalyzer` ties the steps together behind the
`SurveyAnalyzer` interface, and `seeingmon.survey.tracker.PointingTracker` serves the Polaris
position to the scheduler.

Import the submodules you need. The package imports nothing at start-up, so the command-line
entry point stays fast.
"""

from __future__ import annotations
