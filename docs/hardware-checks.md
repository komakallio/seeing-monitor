# Hardware checks

The hardware-facing code (the camera driver, the GPIO lines, the SQM-LE reader, and the power-cycle hook) passes its tests on fakes. The checks on this page run the same code against a real camera, board, or meter. They confirm what the fakes only model.

The checks live in `tests/hardware/` and carry the `hardware` marker. They skip unless you opt in, so a normal test run never touches a device. To try the whole system on the real camera, and not only the driver, run `seeingmon dev --driver asi --asi-library PATH`, which starts `acquire`, `core`, and `web` in real time (see the development run in `docs/architecture.md`). For a real night with the real catalog and a plate solver, add `--real-sky --data-dir <folder>` and follow [First light on the dev machine](runbook.md#first-light-on-the-dev-machine).

## Run the checks

Pass `--hardware`, or set `SEEINGMON_HARDWARE=1`:

```bash
python -m pytest tests/hardware --hardware -s -rs
```

The `-s` option shows the report that each check prints, and `-rs` lists each skip with its reason. Select one device with `-k`, for example `-k camera`, `-k gpio`, `-k sqm`, or `-k power`.

A check that lacks its device or its configuration skips and says why. A check that finds a device that misbehaves fails. Without `--hardware`, every check skips.

The checks read the same local configuration as the system: `local/config.toml` and the `SEEINGMON_*` variables. Real values stay there, and the checks print no address, serial number, or path.

## Before you run

- **Camera.** Install the vendor library, and tell the driver where it is. Set `library_path` under `[services.acquire.driver_options]` in `local/config.toml`, or set `SEEINGMON_ASI__LIBRARY_PATH`. Connect the camera to a USB 3 port, and install the camera's udev rule, so that your user can open the device.
- **GPIO.** Install `libgpiod` (`apt install libgpiod2` on Debian 12 or `libgpiod3` on Debian 13). Your user needs access to `/dev/gpiochip*`.
- **SQM-LE over TCP.** Set `host` under `[sqm]`.
- **SQM-LE from InfluxDB.** Set `source = "influx"` under `[sqm]` and fill the table `[sqm.influx]` (see [Read the SQM-LE from InfluxDB](runbook.md#read-the-sqm-le-from-influxdb)). Set the environment variable that `token_env` or `password_env` names, in the shell that runs the checks.
- **Power cycle.** Choose a route under `[power]`, and set any environment variable that the route names with `${NAME}`.

## The checks

| Check | Needs | What it does | A pass looks like |
|---|---|---|---|
| Enumerate and open the camera | The library and a camera | Counts the cameras, opens the first, and reads its description | One camera or more, a model name, the SDK version, and 8288 × 5644 pixels for the reference camera |
| Capabilities match the profile | The same | Compares the reported gain range, exposure range, binning, pixel formats, sensor size, and temperature sensor with the profile | No difference. The report also prints the offset range, which the profile leaves out until you know it. |
| ROI round trip | The same | Applies a 128 × 128 ROI, reads frames, and moves the ROI to four positions, including odd ones and the corners | Every frame carries the ROI that the camera applied. The report shows where the camera put the ROI for each request, which tells you if the camera aligns the start position. |
| Stream of 100 frames | The same | Streams the fast mode, then checks the sequence, the arrival and UTC times, the median frame period, and the drop counter | Increasing times, a median period within a factor of 2 of the profile's prediction, at most 5% dropped frames, and a drop counter that matches the frames' `dropped_before` |
| Temperature | The same | Reads the sensor temperature and checks that frames carry it | A value between -30 and 70 °C. A first read that returns 0 for the first 250 ms does not leak through. |
| Stale controls | The same | Makes the USB bandwidth, the flip, the offset, and the high-speed mode stale (what another program can leave), configures the fast stream, and reads the controls back. It streams 40 frames, and it puts the camera back as it was. | Bandwidth 100, flip 0, the offset of the camera, normal speed, all in manual mode, a stream above 70% of the profile's rate, and no setting left changed |
| High-speed flag | The same | Runs the sequence that showed the camera's late latch (n128, h128, h8_128, h128, n128, n8_128, n128: the normal or the high-speed flag, RAW8 where the label has an 8, 128 × 128 pixels, 40 frames each), and takes the median frame period of each step. It puts the camera back as it was. | The steps of one regime agree within 5%, and the high-speed steps run faster than the normal ones by about the ratio that the profile predicts (1.25 on the reference camera). See "The camera takes up the high-speed flag late". |
| Recovery step 1 | The same | Streams, restarts capture, and reads on | The first frame after the step has the `RECOVERED` flag, and the sequence continues |
| Recovery step 3 (USB reset) | The same, plus `SEEINGMON_HARDWARE_USB_RESET=1` | Resets the USB device, waits for the camera, and reads on | The camera reappears within the timeout, and the stream continues. The reset interrupts the camera, so the check needs the extra opt-in. |
| GPIO loopback | `SEEINGMON_HARDWARE_GPIO_OUT` and `SEEINGMON_HARDWARE_GPIO_IN` (each `chip:line`, such as `gpiochip0:17`) and a jumper wire between the two lines | Switches the output five times and reads the input | The input follows the output in both directions |
| SQM-LE | `host` under `[sqm]`, and a `source` that is `tcp` | Sends the four requests (`ix`, `rx`, `ux`, `cx`) and parses the answers | A magnitude between 5 and 25 mag/arcsec², the protocol, model, and feature numbers, and a temperature |
| SQM-LE from InfluxDB | `source = "influx"` and the table `[sqm.influx]` | Reads the newest point of the field once, as `seeingmon hardware sqm` does | A magnitude between 5 and 25 mag/arcsec², a temperature between -50 and 70 °C when `temperature_field` is set, and a point no older than `max_age_s`. The report names no endpoint, bucket, database, measurement, field, or tag. |
| Power-cycle dry run | A route under `[power]` | Expands the route from the environment and reports what it would run | The outcome is a dry run, and nothing runs |

## Windows

A development machine with Windows runs the camera checks too. The library path never goes into a tracked file, so set it for the run.

1. Extract the SDK archive for Windows (V1.41 or later) to a folder outside the repository. The folder `windows-sdk` holds `ASICamera2.dll`, `ASICamera2.h`, and the license.
2. Point the driver at the library. In PowerShell, run `$env:SEEINGMON_ASI__LIBRARY_PATH = '<path>\ASICamera2.dll'` in the shell that runs the checks. The `library_path` option under `[services.acquire.driver_options]` in `local/config.toml` does the same.
3. Close other camera software, such as SharpCap. One process at a time can open the camera.
4. Run `python -m pytest tests/hardware --hardware -s -k camera`.

Two things differ from Linux:

- The SDK needs a call of the camera count before it describes a camera. A call of `ASIGetCameraProperty` first fails with `INVALID_INDEX`. The driver calls `ASIGetNumOfConnectedCameras` first, so the checks are not affected, but a script of your own must do the same.
- The USB reset step (recovery step 3) is Linux-only, because it resets the device with a `usbfs` ioctl or through sysfs. The check of that step skips on Windows.

## Measure the frame rates

The timing model of the profile (a frame overhead plus a row time) comes from published numbers, and the USB bandwidth control alone changes the rate by a factor of two. `seeingmon camera rates` measures the real camera in a table, one factor at a time around the fast stream of the profile, and it fits the frame overhead and the row time. Run it on the Raspberry Pi too, because the performance gate and the soak test need the table of the Pi.

```bash
seeingmon camera rates --json local/camera-rates.json
```

Close other camera programs first. The command reads the driver options from `[services.acquire.driver_options]`, so it needs the same configuration as the checks above (the path of the library). A run takes about two and a half minutes. These options change it:

- `--frames` and `--settle` set the frames that each row measures (default 150) and the frames that it reads and drops first (default 10). A snapshot row takes at most 10 exposures and drops none, because one exposure takes about half a second.
- `--gain` sets the gain of every row (default 120). The gain does not change the rate.
- `--groups` runs a subset of `exposure,roi,format,bandwidth,speed,bin2,snapshot`. The baseline always runs.
- `--json` also writes the table as JSON. The file holds the conditions (the camera model, the SDK version, the sensor temperature, the platform), the rows, and the fits. It carries no host name, serial number, or path, and it belongs in `local/`.

The baseline is the fast stream of the profile: bin1, 128 × 128 pixels, 2 ms, RAW16, bandwidth 100, normal speed. The groups change one factor of it, except for `speed`, `bin2`, and `snapshot`, which add the high-speed mode, the second readout mode, and single exposures:

| Group | Rows |
|---|---|
| `exposure` | 0.1, 0.5, 1, 5, 10, and 20 ms |
| `roi` | 32, 64, 256, and 512 pixels square |
| `format` | RAW8 |
| `bandwidth` | 40 to 90 percent in steps of 10 |
| `speed` | the high-speed mode, at each ROI size |
| `bin2` | bin2 at 0.5 ms (so the readout sets the period), the same at 2 ms, the high-speed mode, and the 320 × 240, 10 ms, RAW8 stream of a recording |
| `snapshot` | Single exposures of the survey readout mode (bin2) at 1 ms, in the full width of the frame at 64, 256, and 1024 rows and for the full frame (see [Measure single exposures](#measure-single-exposures)) |

A high-speed row runs in the high-speed regime only with a driver that makes the camera take up the flag (see "The camera takes up the high-speed flag late"). A table that a driver made before that fix holds high-speed rows in the normal regime, and its high-speed fit equals its normal fit.

Each row prints the measured rate, the rate of the model, the median, the standard deviation (jitter), and the maximum of the frame periods, the dropped frames, and the ADC depth of the readout mode. A rate far below the model on a bandwidth row shows the bandwidth limit, and a rate that follows the exposure shows an exposure limit. After the rows, the command prints `frame_overhead_ms` and `row_time_us` fitted to the rows at bandwidth 100, beside the values of the profile. Put the fitted values in the profile when they differ by more than a few percent, and keep a comment with the conditions.

The command saves every writable control and the geometry of the camera before the first row, and it puts back what the rows changed, even when a row fails or you press Ctrl+C, because other programs such as SharpCap share the camera. It closes the camera at the end. A row that fails prints `FAILED` and the reason, and the table goes on. The exit code is 1 when a row failed or a control could not be restored.

## Measure single exposures

The scheduler takes its brightness frame and its survey frames as single exposures (a start, a status poll, and a read). It waits twice the time that the driver expects, plus 0.5 s (`read_timeout_margin_s`). The driver takes the expected time from the snapshot model of the readout mode in the profile: an overhead (`snapshot_overhead_s`, in seconds) and a row time (`snapshot_row_time_us`, in microseconds for each ROI row), which come on top of the exposure. The video model (`frame_overhead_ms` and `row_time_us`) cannot give that time, because the camera reads the whole frame out before the SDK returns a single exposure. A model that is too short makes the read time out, turns the `camera` component to `degraded`, and starts the recovery ladder.

The values in the profile come from one run on a Raspberry Pi 4 on October 3, 2026, with an idle camera. A 1 ms exposure took 0.293 to 0.296 s for the 20 arcminute watch ROI (312 × 314 pixels in bin2) and 0.480 to 0.532 s for the full frame (4144 × 2822 pixels), and a 2 s exposure of the full frame took 2.52 s. Three runs agree within 0.02 s. A straight line through the two points gives 0.27 s plus 75 µs for each row, and the profile holds that for bin2. Bin1 is unmeasured: the profile states no snapshot values for it, and the software takes its video row time and an overhead of at least 0.3 s.

Refit the model on the final hardware, and again after you change the host, the USB port, or the SDK:

1. Stop the services that use the camera (`sudo systemctl stop seeingmon.target` on the Pi), and close other camera programs.
2. Run the `snapshot` group. It takes about half a minute.

   ```bash
   seeingmon camera rates --groups snapshot --json local/camera-snapshots.json
   ```

   The group measures the survey readout mode, which is the mode of the brightness frame, the survey frames, and the dark sessions. Each row takes single exposures of 1 ms in the full width of the frame, at 64, 256, and 1024 rows and at the full height. It takes 10 exposures, and before each one it configures the camera, as the scheduler does before every exposure of a survey step. It times each exposure from the call that starts it to the returned frame, and it drops none, so `max ms` shows a slow first exposure after a configure, which the median hides. The first survey exposure after a start timed out on the Pi 4.
3. Read the rows. `med ms` is the median time of an exposure, `max ms` is the slowest one, and `model` is the rate that the profile predicts. A median above the model shows a model that is too short.
4. Read the fit under the table. It prints `snapshot_overhead_s` and `snapshot_row_time_us` with the values of the profile beside them, and the largest error of the line against the medians.
5. If a median lies above the model, or the fitted values differ from the profile by more than about 10%, put the fitted values in the entry of the survey mode in `profiles/<id>.toml`. Add a comment with the host, the date, the SDK version, and the conditions. Round the values up: a model that is too short brings the timeouts back, and a model that is a little long only lengthens the wait for a frame that never comes.

The rows use the full width of the frame, so the pixels of a row stay the same and the times fall on a line against the height. A narrow ROI, such as the watch ROI, transfers fewer pixels for each row than the line says, so a refit errs on the long side for it, which is the safe side. The model also sets the time that `acquire` stamps on a snapshot (the arrival time minus the period, plus half the exposure), so the time of a survey frame is off by the error of the model.

If a row fails with `CameraTimeoutError` and `the exposure did not finish within`, the camera is slower than twice the model plus 0.5 s, and the scheduler would time out there too. Raise the values in the profile until the row passes, and run the group again. Until the profile holds the new values, raise `read_timeout_margin_s` under `[scheduler.loop]` and `[services.acquire]` in `local/config.toml`. The setting stays available for a slow camera or host.

## The camera takes up the high-speed flag late

The ASI294MM does not apply the high-speed flag when you set it. The camera runs in a regime, normal or high-speed (another frame rate and another ADC depth), and it takes the value of the `HighSpeedMode` control into the regime in two cases only: at the first `ASISetROIFormat` after `ASIInitCamera`, and when `ASISetROIFormat` changes the image format (RAW8 to RAW16, or back). A change of the flag alone, or of the ROI size, leaves the regime as it is. The SDK reports no error, and it has no call that reports the regime. A new process starts in the normal regime. The measurements come from the camera on a Windows machine with SDK V1.41 on October 2, 2026. The behavior of the SDK for Linux is not measured.

The first frame-rate table showed it. The driver then set the flag alone, so the high-speed rows ran in the normal regime, and their fit equaled the normal fit. The sequence below runs bin1, 128 × 128 pixels, RAW16 unless the label has an 8, 2 ms, gain 120, and bandwidth 100, in a fresh process (n is the normal flag, h is the high-speed flag, and 8 is RAW8):

| Driver | n128 | h128 | h8_128 | h128 | n128 | n8_128 | n128 |
|---|---|---|---|---|---|---|---|
| Sets the flag alone (fps) | 82.1 | 82.1 | 102.9 | 102.9 | 102.9 | 82.1 | 82.1 |
| Sets the other image format first (fps) | 82.1 | 102.9 | 102.9 | 102.9 | 82.1 | 82.1 | 82.1 |

The second and the fifth step change the flag alone, and the first driver did not follow them. Other runs agree. In a fresh process, `h64` runs fast (128.2 fps), and a following `n64` stays fast. After `h8_128`, a normal RAW16 `n64` returns to the normal regime (102.3 fps), because the format changes back.

The rate is not the only symptom. `ActiveStream.adc_bits` and the `adc_bits` of each frame reported the requested regime (10 bits) while the camera stayed in the old one (12 bits), and a consumer that scales a frame by the reported depth is wrong by a factor of 4 then.

The driver now sets the other image format before it sets the requested one, when the flag differs from the one that the camera took up last, or when it cannot know. It cannot know after an open, after a recovery step that reopens the camera, and after `restore_settings`. The driver sets the controls first, so that the format change takes up the flag that the stream asks for. The fake SDK models the latch (`FakeAsiSdk.high_speed_regime`), and `tests/drivers/test_asi_driver.py` checks that what the driver reports is what the fake camera runs.

In bin1 at bandwidth 100, the frame period of the two regimes fits a line in the ROI height (the frame period is the overhead plus the rows times the row time). The normal regime gives 7.37 ms + 37.6 µs per row, and the high-speed regime gives 5.88 ms + 30.0 µs per row. The profile held 6.50 ms + 37.6 µs and 5.00 ms + 30.1 µs when the table was first measured (the numbers came from a fit of ZWO's published frame rates), so its overheads were about 0.87 ms low. The profile now holds the measured numbers, and the fake SDK and the simulator use them too. In bin2 the flag changes no frame rate that the table can see (417 fps against 419 fps for 64 × 64 pixels at 0.5 ms), so the late latch has no visible effect there, and the two fits agree: 1.22 ms + 18.5 µs for the normal flag and 1.21 ms + 18.6 µs for the high-speed flag. One bin2 row does not follow the line of the others. The 64 × 64 ROI at 2 ms ran at 465 fps (2.13 ms a frame, which is the exposure plus 0.13 ms), and the line from the rows at 0.5 ms gives 417 fps (2.40 ms). The model (the frame period is the larger of the exposure and the readout time) cannot give that, and the table has no other bin2 row at an exposure above 0.5 ms and below 10 ms to show where the period changes.

**Open check.** The fixed driver ran the sequence once on the camera (the second row). The check "High-speed flag" in the table above automates it, and that check has not run on a camera yet. Run it with the other camera checks, and run `seeingmon camera rates` once more. A run on the Raspberry Pi shows whether the SDK for Linux latches the flag in the same way.

## Compare the binding with the SDK header

The binding copies the enumerations, the two structures that the SDK fills, and the argument types of the functions that it calls from the vendor header. A new SDK release can change any of them, and a wrong structure layout reads garbage without an error. `tests/hardware/test_asi_header.py` parses `ASICamera2.h` and compares it with the binding:

- the value of every enumerator that the binding lists (the error codes, the image types, the controls, and the exposure states),
- the fields of `ASI_CAMERA_INFO` and `ASI_CONTROL_CAPS`: the order, the types, the array lengths, and the size and offsets in the layout of the platform, with `long` as the `long` of the platform,
- the return type and the argument types of every function that the binding declares.

The test needs no camera and no library. It needs a copy of the header, which the repository never holds. Set the path and run the test on the platform that will use the library, such as the Raspberry Pi with the header of the Linux archive (`include/ASICamera2.h`):

```bash
SEEINGMON_ASI__HEADER_PATH=<path-to>/ASICamera2.h python -m pytest tests/hardware/test_asi_header.py -s
```

Without the variable, the four checks against the header skip. A difference fails the test, and the message names the enumerator, the field, or the function. The header can also hold more than the binding lists, such as the controls of newer cameras. The test prints those with `-s` and does not fail. On a platform where `long` and `int` have the same size, such as Windows, `ctypes` makes the two types one, so the check cannot tell them apart there. The Windows archive and the Linux and macOS archive of V1.41 carry the same header file, and the binding matches it.

## Try the SQM-LE settings

`seeingmon hardware sqm` reads the SQM-LE one time from the source that `[sqm]` names, and it prints one line with the magnitude, the temperature, and the age of the reading. It ignores `enabled`, so you can run it before you turn the reader on. It prints no address and no name from the configuration, and it exits with 1 and one line when a setting is missing or the read fails. The check "SQM-LE from InfluxDB" above runs the same read under `pytest`. See [Check the settings](runbook.md#check-the-settings) for the output, and for the environment variable of the token.

```bash
seeingmon hardware sqm
```

For the `tcp` source, the line reports an age of 0 s, because the unit answers with its current reading.

## Send the SQM-LE sample to the maintainer

This section concerns the `tcp` source. The reader follows the protocol from documentation, and no real unit has confirmed it (blocker B5). To close the blocker, save the raw responses:

```bash
SEEINGMON_HARDWARE_SQM_DUMP=local/sqm-sample.txt python -m pytest tests/hardware --hardware -k sqm -s
```

The file lists each request and its response. A response can hold the unit's serial number, so keep the file in `local/`, which Git ignores, and remove the serial number before you share the file.

## Check that the heater stays off after a stop (blocker B3)

`seeingmon heater-off` drives the heater outputs off and releases the lines. The kernel then returns each line to its default, usually an input, so only the HAT decides whether the heater stays off from there. No automated check can see the HAT, so run this check by hand on a bench when you choose one. You need the heater supply on and a way to see the heater, such as a meter on the driver input or a lamp as a stand-in load.

1. Enable the heater in `local/config.toml` with the pin map of the HAT, and start `seeingmon core` in a terminal under conditions that call for heat, such as a `fixed` ambient sensor with a high humidity. Wait until the heater switches on.
2. Kill the process without a clean stop: `kill -9 <pid>`. The kernel releases the lines, and no software runs. Watch the heater for a minute.
3. Run `seeingmon heater-off`. It requests the lines off, releases them again, and prints `seeingmon heater-off: the heater outputs are off` with the names of the outputs. Watch the heater for another minute.

A pass is a heater that stays off in both steps. A heater that comes on or floats means that the HAT has no pull resistor that keeps it off, and no failsafe of its own. Add a pull-down at the driver input (a pull-up for an active-low driver), or choose a HAT with a failsafe.

## What the checks do not cover

- **The dew heater on a real HAT.** The HAT is undecided. The loopback check confirms only the GPIO layer, and the previous check is manual. When you choose a HAT, add its pin map and sensors to `[heater]`, and run the heater with the dew shield on a bench first.
- **A real power cycle.** The dry run proves that the route is configured. A real cycle cuts the power of the Pi, so run the route's own command or request by hand once, with the Pi attached, before you rely on it.
- **Long-term behavior.** A soak test of days finds the stalls that the recovery ladder exists for. It belongs to commissioning.
- **The time latency.** A light pulse from a GPIO pin measures the delay between the end of a frame and its arrival stamp. Commissioning records the value in `time_error_ms` and in the acquire timing settings.
