# Hardware checks

The hardware-facing code (the camera driver, the GPIO lines, the SQM-LE reader, and the power-cycle hook) passes its tests on fakes. The checks on this page run the same code against a real camera, board, or meter. They confirm what the fakes only model.

The checks live in `tests/hardware/` and carry the `hardware` marker. They skip unless you opt in, so a normal test run never touches a device. To try the whole system on the real camera, and not only the driver, run `seeingmon dev --driver asi --asi-library PATH`, which starts `acquire`, `core`, and `web` in real time (see the development run in `docs/architecture.md`).

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
- **SQM-LE.** Set `host` under `[sqm]`.
- **Power cycle.** Choose a route under `[power]`, and set any environment variable that the route names with `${NAME}`.

## The checks

| Check | Needs | What it does | A pass looks like |
|---|---|---|---|
| Enumerate and open the camera | The library and a camera | Counts the cameras, opens the first, and reads its description | One camera or more, a model name, the SDK version, and 8288 × 5644 pixels for the reference camera |
| Capabilities match the profile | The same | Compares the reported gain range, exposure range, binning, pixel formats, sensor size, and temperature sensor with the profile | No difference. The report also prints the offset range, which the profile leaves out until you know it. |
| ROI round trip | The same | Applies a 128 × 128 ROI, reads frames, and moves the ROI to four positions, including odd ones and the corners | Every frame carries the ROI that the camera applied. The report shows where the camera put the ROI for each request, which tells you if the camera aligns the start position. |
| Stream of 100 frames | The same | Streams the fast mode, then checks the sequence, the arrival and UTC times, the median frame period, and the drop counter | Increasing times, a median period within a factor of 2 of the profile's prediction, at most 5% dropped frames, and a drop counter that matches the frames' `dropped_before` |
| Temperature | The same | Reads the sensor temperature and checks that frames carry it | A value between -30 and 70 °C. A first read that returns 0 for the first 250 ms does not leak through. |
| Recovery step 1 | The same | Streams, restarts capture, and reads on | The first frame after the step has the `RECOVERED` flag, and the sequence continues |
| Recovery step 3 (USB reset) | The same, plus `SEEINGMON_HARDWARE_USB_RESET=1` | Resets the USB device, waits for the camera, and reads on | The camera reappears within the timeout, and the stream continues. The reset interrupts the camera, so the check needs the extra opt-in. |
| GPIO loopback | `SEEINGMON_HARDWARE_GPIO_OUT` and `SEEINGMON_HARDWARE_GPIO_IN` (each `chip:line`, such as `gpiochip0:17`) and a jumper wire between the two lines | Switches the output five times and reads the input | The input follows the output in both directions |
| SQM-LE | `host` under `[sqm]` | Sends the four requests (`ix`, `rx`, `ux`, `cx`) and parses the answers | A magnitude between 5 and 25 mag/arcsec², the protocol, model, and feature numbers, and a temperature |
| Power-cycle dry run | A route under `[power]` | Expands the route from the environment and reports what it would run | The outcome is a dry run, and nothing runs |

## Measure the frame rates

The timing model of the profile (a frame overhead plus a row time) comes from published numbers, and the USB bandwidth control alone changes the rate by a factor of two. `seeingmon camera rates` measures the real camera in a table, one factor at a time around the fast stream of the profile, and it fits the frame overhead and the row time. Run it on the Raspberry Pi too, because the performance gate and the soak test need the table of the Pi.

```bash
seeingmon camera rates --json local/camera-rates.json
```

Close other camera programs first. The command reads the driver options from `[services.acquire.driver_options]`, so it needs the same configuration as the checks above (the path of the library). A run takes about two minutes. These options change it:

- `--frames` and `--settle` set the frames that each row measures (default 150) and the frames that it reads and drops first (default 10).
- `--gain` sets the gain of every row (default 120). The gain does not change the rate.
- `--groups` runs a subset of `exposure,roi,format,bandwidth,speed,bin2`. The baseline always runs.
- `--json` also writes the table as JSON. The file holds the conditions (the camera model, the SDK version, the sensor temperature, the platform), the rows, and the fits. It carries no host name, serial number, or path, and it belongs in `local/`.

The baseline is the fast stream of the profile: bin1, 128 × 128 pixels, 2 ms, RAW16, bandwidth 100, normal speed. The groups change one factor of it, except for `speed` and `bin2`, which add the high-speed mode and the second readout mode:

| Group | Rows |
|---|---|
| `exposure` | 0.1, 0.5, 1, 5, 10, and 20 ms |
| `roi` | 32, 64, 256, and 512 pixels square |
| `format` | RAW8 |
| `bandwidth` | 40 to 90 percent in steps of 10 |
| `speed` | the high-speed mode, at each ROI size |
| `bin2` | bin2 at 0.5 ms (so the readout sets the period), the same at 2 ms, the high-speed mode, and the 320 × 240, 10 ms, RAW8 stream of a recording |

Each row prints the measured rate, the rate of the model, the median, the standard deviation (jitter), and the maximum of the frame periods, the dropped frames, and the ADC depth of the readout mode. A rate far below the model on a bandwidth row shows the bandwidth limit, and a rate that follows the exposure shows an exposure limit. After the rows, the command prints `frame_overhead_ms` and `row_time_us` fitted to the rows at bandwidth 100, beside the values of the profile. Put the fitted values in the profile when they differ by more than a few percent, and keep a comment with the conditions.

The command saves every writable control and the geometry of the camera before the first row, and it puts back what the rows changed, even when a row fails or you press Ctrl+C, because other programs such as SharpCap share the camera. It closes the camera at the end. A row that fails prints `FAILED` and the reason, and the table goes on. The exit code is 1 when a row failed or a control could not be restored.

## Send the SQM-LE sample to the maintainer

The reader follows the protocol from documentation, and no real unit has confirmed it (blocker B5). To close the blocker, save the raw responses:

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
