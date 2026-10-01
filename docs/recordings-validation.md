# Validation on the recorded videos

This page records what the fast path does with your two recorded 10 ms videos, and what those videos can't show. The production analyzer replayed both files with its default configuration, and `tests/fastpath/test_recordings.py` enforces the bounds that this page explains. The tests skip when the recordings are not configured. The page names no file, path, or serial number.

**The r0 and seeing values on this page are not calibrated.** The videos use bin2 (3.82 arcsec per pixel), 8 bits, and a 10 ms exposure. The design uses bin1, 2 ms, and 16 bits. No independent seeing measurement (a differential image motion monitor, for example) exists for those nights. Read the values as a check of order of magnitude and of consistency, not as a measurement.

## Summary

| Question | Result |
|---|---|
| Does the pipeline run on real frames? | Yes. The star is found in all 35,252 frames, and no frame carries an edge, hot-pixel, or no-star flag. Eight frames (0.03%) reach the saturation level. |
| Do drops and windows add up? | Yes. The 300 s capture loses 3 frames and the 60 s capture loses none. Every frame and every lost frame lands in exactly one window. |
| Is the plate scale right? | Within 4%. The star drifts at 0.157 to 0.159 arcsec/s, and the sidereal motion of Polaris gives 0.163 arcsec/s. |
| Is the motion stationary? | No, as real seeing isn't. The variance of 10 s blocks scatters by 13% (29% with one burst block), where a stationary process with the same spectrum scatters by 7%. The 60 s windows hold: their r0 values spread by 10%. |
| Do the spectra look like turbulence? | Yes between 1 and 30 Hz. A floor that doesn't scale with the flux sits above 20 Hz. |
| Do the two estimators agree? | Not at the default wind of 10 m/s: the structure-function estimate is 17 to 34% above the variance estimate. On average they agree within 4% at an assumed wind of 2 m/s. |
| Does the bin2 gain swing of 0.53 to 1.48 show? | No. The star has a core of 2.3 pixels, and the gain stays within about 0.9 to 1.1. |
| Does the 8-bit noise model hold? | No. It over-subtracts the centroid noise, and r0 reads about 3.5% high. |
| What is r0? | 8 to 12 cm at the default assumptions, which is 0.8 to 1.2 arcsec of seeing. Not calibrated. |

## The captures

| | 60 s capture | 300 s capture |
|---|---|---|
| Frames | 5,874 | 29,378 |
| Duration | 60.0 s | 300.2 s |
| Lost frames | 0 | 3 |
| Closed windows | 1 full, 1 partial (2 frames) | 5 full, 1 partial (19 frames) |
| Frame rate | 97.86 fps | 97.85 fps |
| Median flux | 50,400 e⁻ | 50,500 e⁻ |
| Mean peak pixel | 101 of 255 counts | 105 of 255 counts |

Both captures have 320 × 240 pixels in 8 bits, and the analyzer reads them as bin2 at gain 100 with a 10 ms exposure. The options of the replay driver give those three values, because the validation never reads a sidecar.

### Timing and drops

The frame times are the arrival times at the host, not the exposure times. The interval between frames has a mean of 10.219 ms and a standard deviation of 0.26 ms (2.5%), and 99.7% of the intervals lie within 1 ms of the median. A few intervals fall far outside that: 5.2 ms and 15.2 ms in the 60 s capture, and 0.14 ms in the 300 s capture. Frames therefore arrive in small bursts now and then.

The 300 s capture has two gaps: 33.8 ms (2 frames lost) within the first 30 ms of the capture, and 17.2 ms (1 frame lost) at 9.2 s. The window assembler counts the same 3 frames as the replay driver. The first window reports `n_dropped` of 3 and a valid fraction of 0.999, so it doesn't carry `degraded`. The test recomputes the count from the timestamps and compares it with the driver's count and with the sum over the windows.

### The star

The second-moment width of the star is 1.41 to 1.48 pixels (sigma), which is 12.5 to 13.3 arcsec FWHM (3.3 to 3.5 pixels). The brightest pixel holds 16% of the flux, which a Gaussian core of 2.3 pixels FWHM (9 arcsec) gives, so the star has a core of about 2.3 pixels and wings that widen the second moment. Diffraction and 1 arcsec seeing give a core of about 2.3 arcsec for a focused 50 mm aperture, so the star is out of focus or aberrated by a factor of about 4. The design suggests a defocus of about 3 pixels for bin2, and this star is somewhat sharper.

## Stationarity

The variance of the detrended centroid in thirty 10 s blocks of the long capture has a mean of 0.139 arcsec² (0.37 arcsec rms per axis). To judge the scatter, several hundred stationary Gaussian series with the measured spectrum (surrogate data) serve as the reference.

| Quantity | Recording | Stationary reference |
|---|---|---|
| Scatter of 10 s block variances | 13% (29% with the burst block) | 7% |
| Scatter of 60 s window variances | 16% | 7% |
| Trend of the block variance | 6% per 100 s (13% with the burst block) | 1.5% per 100 s (standard deviation) |

The turbulence strength changes within the capture, as real seeing does, and a window of 60 s still holds one value: the spread of r0 between the five windows is 10%, and the sampling error of one window is about 4%. One 10 s block (270 to 280 s) has 2.4 times the median variance, because of the vibration burst described below. The 60 s capture has six blocks that scatter by 10%.

## Spectra

The spectrum of each 60 s window comes from the Welch estimate with 2 s segments, and it has 20 logarithmic bins between 0.5 and 45 Hz. The average of the two axes over the five windows is:

| Frequency (Hz) | 1 | 3 | 6.5 | 10 | 21 | 25 to 45 |
|---|---|---|---|---|---|---|
| Power (arcsec²/Hz) | 0.022 | 0.009 | 0.0045 | 0.0026 | 0.0011 | 0.0009 to 0.0010, flat |

- **The windows agree.** The log-ratio of each window to the mean over the windows has an rms of 0.23, which includes the burst line (0.71 in its bin).
- **Shape.** A single frozen layer with L0 = 20 m, a 10 ms exposure, and the folding of power above the Nyquist limit fits the shape within +51% and -22% over the whole band at 10 m/s. At 5 m/s the fit is within 25% up to 30 Hz, and the model falls short above that. The data have 27 to 51% more power below 4 Hz than the 10 m/s shape (the shapes agree between 5 and 12 Hz by construction).
- **Floor above 20 Hz.** The power is flat at 0.0009 to 0.0010 arcsec²/Hz between 25 and 45 Hz. The 8-bit section shows that this floor is not pixel or photon noise.
- **Aliasing.** The records carry the note that power above 49 Hz folds into the spectrum. At 10 ms and 98 fps the exposure averaging attenuates that power only partly, so the floor can hold power from a fast layer.
- **The axes differ.** The x variance is 1.31 to 1.57 times the y variance in every window. Isotropic turbulence gives equal variances for a round aperture at zero exposure. With a 10 ms exposure, a wind along the y axis makes the x variance larger than the y variance, by a factor that the model puts at 1.22 for 10 m/s and 1.48 for 30 m/s. The power above 36 Hz is 1.5 to 1.9 times higher along x. A layer that moves along y at 20 to 30 m/s explains both, and the slow structure in the time domain doesn't fit that layer alone.

## The two estimators

The variance estimate (`r0_cm`) and the structure-function estimate (`r0_structure_cm`) differ for these data. The table gives the mean of the five full windows of the long capture and the single window of the short capture, in centimeters.

| Assumed wind (m/s) | Exposure factor on the variance | Variance, 300 s | Structure, 300 s | Variance, 60 s | Structure, 60 s |
|---|---|---|---|---|---|
| 1 | 1.005 | 10.6 | 9.2 | 14.0 | 11.6 |
| 2 | 1.020 | 10.6 | 11.1 | 14.0 | 13.9 |
| 5 | 1.098 | 10.2 | 12.0 | 13.5 | 15.1 |
| 10 (default) | 1.263 | 9.4 | 11.8 | 12.4 | 14.7 |
| 20 | 1.560 | 8.3 | 10.8 | 11.0 | 13.6 |

At the default wind of 10 m/s, the structure-function estimate is 17 to 34% above the variance estimate in all six windows. At 2 m/s the means agree within 4%, and single windows within 12%.

The disagreement measures how well the temporal model fits. The structure function uses lags of 0.04 to 0.12 s and divides by 1 minus the model correlation at each lag, so it responds to power between 3 and 25 Hz. The variance counts all power. The spectrum has 27 to 51% more power below 4 Hz than a 10 m/s single layer has, so the variance estimate is larger. That excess can come from a slower layer, a larger outer scale, or motion of the mount, and the data can't separate these. If the excess is not atmospheric, the structure-function estimate is nearer to the truth. The agreement at 2 m/s doesn't show that the wind is 2 m/s.

## Exposure bias at 10 ms

The correction for exposure averaging needs a wind speed, and the data can't supply one (the frame rate is too low). The variance estimate of r0 falls from 10.6 cm at 2 m/s to 8.3 cm at 20 m/s, which is 22%. The default of 10 m/s costs 11% of r0 against 2 m/s. A 2 ms exposure would shrink the factors to 1.000, 1.001, 1.005, 1.020, and 1.068 at the five winds above, so the commissioning default makes the choice of wind nearly irrelevant.

The window records store the assumed wind and the factor, so a reader can undo the correction.

## Centroid gain in bin2

The research notes predict that an in-focus bin2 star has a centroid gain between 0.53 and 1.48 as it drifts across a pixel, with a period of about 22 s. The star in these captures crosses a pixel every 24 s along x (0.0406 px/s) and every 108 s along y.

The swing is absent. The notes give a bias with an amplitude of 0.06 to 0.076 px for that swing, and such a bias would bunch the sub-pixel positions of the centroid: the first harmonic of their density would be 0.4 to 0.5. It is 0.10 in x and 0.08 in y for the 300 s capture, which fits a bias with an amplitude of 0.016 px and a gain between 0.90 and 1.10. The variance of 3 s blocks differs by a factor of 1.2 to 1.35 between sub-pixel phases, and by 1.75 for one axis in which the burst falls into a single phase bin. The variance factor of 0.44 to 1.8 in the notes would give a factor of 4. The notes give a gain variation of about 1.8% for a blur of 3 pixels, and the swing of the in-focus star is 47%. This star has a core of 2.3 pixels and a variation of 10%, between the two, so the swing doesn't apply to it. The estimator's centroid-gain factor for bin2 is 0.991, and the data can't confirm it (see the last section).

## 8-bit quantization

The 8-bit frames hold the top 8 bits of the 14-bit ADC, so one count is 82 e⁻. The profile gives a read noise of 6.5 e⁻, and the analyzer adds the quantization noise of one count (q²/12, or 0.09 count² per pixel).

- **Background pixels.** The border pixels sit at the edge between two counts: 26% read 1 and 74% read 2. Their variance in time is 0.19 count² (0.44 count), twice the model.
- **Pixels near the star.** At 6 to 8 pixels from the star the halo lifts the level away from the edge, and the variance in time is 0.001 to 0.005 count². Quantization noise depends on where the signal sits relative to the count edges, so one number can't describe it.
- **Centroid noise.** The model gives 0.027 px per axis (0.0104 arcsec²), which is 5 to 10% of the motion variance. The variance of the second difference of the centroid doesn't depend on the flux: it changes by 4% between the lowest and the highest tenth of the flux (39,000 and 68,000 e⁻), and the model predicts a change of 27%. A fit with a part that scales as 1/F² gives a noise-like part of 0.0025 ± 0.0015 arcsec², against 0.0156 modeled. That is a white noise of about 0.011 px.

The model therefore over-subtracts noise for these recordings, and r0 reads about 3.5% high. The flat floor above 20 Hz is real motion or an instrument effect, not noise. Neither effect matters for 16-bit frames of a bright star, where the modeled noise is well below 1% of the variance.

## The vibration burst

A line at 17.5 Hz appears in one 10 s block (270 to 280 s) of the long capture: 40 times the neighboring power along x and 4 times along y. The amplitude is about 0.3 arcsec along x and 0.08 arcsec along y. The detector flags the fifth window with `vibration` and lists the line at 17.52 Hz, and no other window has the flag. The burst raises the variance of that window by about 14%, so r0 reads 8% low for it. The source is unknown. Nothing in the data tells whether it is a gust, a touch, or a camera or fan resonance.

## What looks wrong

1. **The estimators disagree at the default wind.** The single frozen layer at 10 m/s doesn't match the temporal structure of the data (27 to 51% too little power below 4 Hz). The cross-check depends on the wind assumption as much as the variance estimate does, so it can't confirm the wind.
2. **The two axes differ.** The x variance is 31 to 57% above the y variance. A single layer explains this only with a fast wind along y, and the slow structure contradicts a single fast layer. The turbulence probably has several layers, and the estimator assumes one.
3. **The 8-bit noise model over-subtracts,** by up to 7% of the variance for these recordings.
4. **The frame times jitter.** Arrival intervals range from 0.14 to 15 ms around 10.2 ms. The analyzer treats the stream as uniformly sampled and uses the times only to count lost frames, so seeing is not affected. Don't read the times as exposure times to better than one frame.
5. **The star is defocused to a core of 2.3 pixels,** which these captures need and the final design avoids. The aperture truncation and the centroid gain were not tested on a focused star.
6. **The first drop** falls in the first 30 ms of the capture, which suggests an unsettled start.

## What this validation can't check

- **The absolute scale of r0.** The plate scale is checked to 4% by the sidereal drift, and the formula is checked on the simulator, but no independent seeing value exists for these nights.
- **The wind and the layers.** At 98 fps the spectrum can't separate a slow layer, a fast layer, and mount motion.
- **The true centroid gain.** It needs a known injected motion, such as a mount step of known size.
- **The flags for edge, hot pixels, and a missing star.** None fires on these captures. Synthetic tests cover the code paths.
- **The zenith correction.** The replay supplies no zenith angle, so every value is for the line of sight.
- **Other exposures, gains, and 16-bit frames.** The captures have one setting.
- **The sidecar.** The analyzer ran on options, because the validation doesn't read the sidecars.

## Repeat the validation

Set `recordings_dir` in the `[replay]` table of `local/config.toml`, and run:

```bash
python -m pytest tests/fastpath/test_recordings.py
```

A replay of both captures takes 20 to 60 s, which depends on the disk.
