# Research notes

These notes record the sources and calculations behind [architecture.md](architecture.md). The research ran on October 1, 2026. Each fact carries a confidence tag:

- **V**: vendor or primary source (vendor page, manual, SDK header, or a project's own source).
- **S**: secondary source (forum post, issue tracker, third-party project).
- **D**: derived by calculation from the inputs shown.
- **U**: not published or not verified.

## Raspberry Pi and tooling

| Fact | Value | Tag | Source |
|---|---|---|---|
| Pi 5 | BCM2712, 2.4 GHz quad-core Cortex-A76, 1 to 16 GB RAM, two USB 3.0 ports with simultaneous 5 Gbps operation, RTC with external battery, microSD SDR104 | V | https://www.raspberrypi.com/products/raspberry-pi-5/ |
| Pi 4 Model B | BCM2711, 1 to 8 GB RAM, two USB 3 and two USB 2 ports. The page lists no RTC. | V | https://www.raspberrypi.com/products/raspberry-pi-4-model-b/ |
| Pi 5 USB power | 5 V 5 A supply, or 5 V 3 A with a 600 mA limit for peripherals | V | https://www.raspberrypi.com/documentation/computers/raspberry-pi.html |
| Raspberry Pi OS | Debian 13 (Trixie), kernel 6.18, image dated September 15, 2026. Legacy Debian 12 images (kernel 6.12) are also offered. | V | https://www.raspberrypi.com/software/operating-systems/ |
| Python on Debian | `python3` 3.13.5 in Debian 13. Debian 12 ships 3.11. | V | https://packages.debian.org/trixie/python3 |
| CI | GitHub-hosted Linux arm64 runners are free in public repositories and generally available since August 7, 2025 (`ubuntu-24.04-arm`, `ubuntu-22.04-arm`). | V | https://github.blog/changelog/2025-08-07-arm64-hosted-runners-for-public-repositories-are-now-generally-available/ |
| Pi prices, April 2026 | Retail (PiShop US), from Raspberry Pi's price thread: Pi 4 at 1 GB $35, 2 GB $55, 3 GB $87.35, 4 GB $100, and 8 GB $165. Pi 5 at 1 GB $45, 2 GB $65, 4 GB $110, 8 GB $175, and 16 GB $305. Another report lists the Pi 4 2 GB as unchanged at $45 and the Pi 4 3 GB as $83.75. Raspberry Pi blames memory costs and says it will reverse the increases when they ease. | S | https://forums.raspberrypi.com/viewtopic.php?t=397523, https://www.omgubuntu.co.uk/2026/04/raspberry-pi-4-3gb-announced |

## Camera and optics

### Sources

| ID | Document | URL |
|---|---|---|
| Z1 | ZWO ASI294 manual V2.2 (February 2022) | https://i.zwoastro.com/zwo-website/manuals/ASI294_Manual_EN_V2.2.pdf |
| Z2 | ZWO ASI294 Pro series product page | https://www.zwoastro.com/product/asi294/ |
| Z3 | ZWO ASI294MM and ASI294MC (non-Pro) product page | https://www.zwoastro.com/product/asi294mm-mc/ |
| Z4 | ZWO gain chart, ASI294MM Pro bin2 | https://i.zwoastro.com/wp-content/uploads/2022/06/62bed57657e00ab3540cbe1839ae4d1a-1.jpg |
| Z5 | ZWO gain chart, ASI294MM Pro bin1 | https://i.zwoastro.com/wp-content/uploads/2022/06/ed1153debb44365edf095ecd61b7d8ef.jpg |
| Z11 | `ASICamera2.h` (SDK 1.41, INDI mirror) | https://raw.githubusercontent.com/indilib/indi-3rdparty/master/libasi/ASICamera2.h |
| Z12 | ASICamera2 SDK manual, revision 2.8 (2018, mirror) | https://condorarraytelescope.org/static/pages/pdf/manuals/ZWO-0005.pdf |
| Z13 | ZWO product SDK page (SDK 1.41, January 12, 2026) | https://www.zwoastro.com/software/product-sdk/ |
| T1 | ToupTek store page, GS PAPO guide scope | https://www.touptekastro.com/products/gs-papo-guide-scope |
| T2 | ToupTek main-site page, GS series | https://www.touptek-astro.com/accessories/Guide-Scope/ |
| T3 | GS-250 spot diagram (design data dated July 18, 2025) | https://cdn.shopify.com/s/files/1/0833/1532/7288/files/GS_detail_03.jpg |
| S1 | SharpCap forum, ZWO's SDK update for the ASI294MM Pro (read modes) | https://forums.sharpcap.co.uk/viewtopic.php?t=3426 |
| S4 | SDK 1.16.3 demo output and debug log for an ASI294MM Pro on a Pi (line period 37.60 µs) | https://gist.github.com/vector-kerr/8628c3be7fa463f6b61f3f539deb08ed |
| S11 | Cloudy Nights thread on the ToupTek GS series | https://www.cloudynights.com/forums/topic/988535-touptek-gs-series-guidescope/ |

### Models

| Model | Facts | Tag | Source |
|---|---|---|---|
| ASI294MM Pro | Cooled mono camera, Sony IMX492, two-stage regulated cooler, DDR3 buffer, USB 3.0, external 11 to 15 V supply needed even to enumerate | V | Z1, Z2 |
| ASI294MM (non-Pro) | Uncooled mono camera, IMX492, ST4 port. The store listed it at USD 999 on October 1, 2026. Its spec table lists 8288 × 5644, 12-bit, 2.3 µm, 5.7 fps, 14.4 ke⁻ full well. | V | Z1 p3, Z3 |
| ASI294MC and ASI294MC Pro | Color, Sony IMX294, 4144 × 2822 at 4.63 µm native. Do not use IMX294 data for the mono cameras. | V | Z1, Z2 |

### Read modes of the IMX492 cameras

The binning factor in `ASISetROIFormat` selects the mode. ZWO documents bin1 and bin2. Bin3 and bin4 come from a ZWO message that the SharpCap forum quotes (S1).

| SDK bin | Output | Pixel | Formation | ADC | Tag |
|---|---|---|---|---|---|
| 1 | 8288 × 5644 | 2.315 µm (ZWO writes 2.3) | Native readout ("unlocked bin1") | 12 bit (10 bit in high-speed mode) | V |
| 2 | 4144 × 2822 | 4.63 µm | On-sensor 2 × 2 sum. Full well rises 4.6 times (D). | 14 bit (12 bit in high-speed mode) | V; mechanism S |
| 3 | About 2760 × 1881 | 6.945 µm | Software 3 × 3 on native data | 12 bit | S |
| 4 | 2072 × 1411 | 9.26 µm | On-sensor 2 × 2 plus software 2 × 2 | 14 bit | S |

ZWO does not document a power-on default mode (U). The SDK reports 8288 × 5644 as the maximum size (S). RAW16 carries the ADC value in the high bits: the low 4 bits are zero at 12 bit (ZWO staff answer on the ASI1600MM page, https://astronomy-imaging-camera.com/product/asi1600mm/), and values are multiples of 4 at 14 bit (S, https://forums.sharpcap.co.uk/viewtopic.php?t=3613). The RAW8 scaling rule is undocumented.

### Gain, full well, and read noise (ZWO charts for the Pro; read from the images, about 2% on full well)

| Mode | Gain | Full well | e⁻/ADU | Read noise |
|---|---|---|---|---|
| Bin1 | 0 | 14,417 e⁻ | 3.5 | 2.65 e⁻ |
| Bin1 | 108 (unity) | about 4.2 ke⁻ | about 1.0 | about 1.8 e⁻ |
| Bin1 | 270 (maximum analog gain) | 0.63 ke⁻ | 0.17 | 1.38 e⁻ |
| Bin2 | 0 | 66,387 e⁻ | 4.05 | 8.0 e⁻ |
| Bin2 | 119 (below the HCG step) | 16.5 ke⁻ | 1.03 | 6.2 e⁻ |
| Bin2 | 120 (HCG on) | 15 ke⁻ | 0.88 | 1.85 e⁻ |
| Bin2 | 300 | 1.7 ke⁻ | 0.11 | 1.3 e⁻ |

Full well divided by e⁻/ADU at gain 0 gives 16,392 ADU for bin2 and 4,119 ADU for bin1, which match the 14-bit and 12-bit full scales (D). ZWO recommends gain 120 and offset 30 (S1). The SDK gain range printed for an ASI294MC Pro is 0 to 570 (S, https://pypi.org/project/camera-zwo-asi/). Exposure range: 32 µs to 2000 s (V, Z1).

### Frame rates and row timing

ZWO's USB 3.0 frame rates (Z1, V). The tables do not state RAW8 or RAW16, and ZWO publishes no USB 2.0 rates.

| ROI | Bin2 14-bit | Bin2 12-bit | Bin1 12-bit | Bin1 10-bit |
|---|---|---|---|---|
| Full frame | 16.3 | 19 | 4.6 | 5.7 |
| 1920 × 1080 | 41 | 47.9 | 21.2 | 26.6 |
| 640 × 480 | 86 | 100.5 | 40.8 | 51.1 |
| 320 × 240 | 153.4 | 179.3 | 64.6 | 80.9 |

The points fit a straight line, frame time equals overhead plus line time times rows, within 1 to 2% (D). The 37.6 µs bin1 line time matches the 37.60 µs line period in an SDK debug log (S4).

| Mode | Line time | Overhead | Skew over the full height |
|---|---|---|---|
| Bin1 12-bit | 37.6 µs | 6.5 ms | 212 ms |
| Bin1 10-bit | 30.1 µs | 5.0 ms | 170 ms |
| Bin2 14-bit | 21.3 µs | 1.4 ms | 60 ms |
| Bin2 12-bit | 18.2 µs | 1.2 ms | 51 ms |

These are the vendor's numbers, which gave the first estimate. The real camera at USB bandwidth 100 measures a larger overhead in the small ROIs of the fast streams (`seeingmon camera rates`, October 2026): bin1 normal 7.37 ms and 37.6 µs, bin1 high speed 5.88 ms and 30.0 µs, and bin2 1.22 ms and 18.5 µs for both flags. The profile holds the measured numbers (see `docs/hardware-checks.md`). The vendor's bin2 14-bit line time of 21.3 µs probably belongs to full-width rows, where the USB link limits the rate (4144 pixels of 2 bytes in 21.3 µs is 390 MB/s), and the line model of the profile does not cover that case.

ROI rules (V, Z11, Z12): width and height count pixels after binning, width is a multiple of 8, height is a multiple of 2, and `ASISetROIFormat` recenters the ROI, so call `ASISetStartPos` afterwards. No minimum ROI and no start-position alignment is documented. The shutter is rolling (V, Z1), and ZWO publishes no row time. The SDK has no frame timestamp for this camera (V, Z11).

### Reference camera: the non-Pro ASI294MM

| Item | Non-Pro ASI294MM | Pro, for contrast | Source |
|---|---|---|---|
| Cooling | None: no thermoelectric cooler and no fan | Two-stage regulated, 35 to 40 °C below ambient at 30 °C ambient | Z1 p3, Z3 |
| Power | USB only. The manual gives a maximum draw of 1.85 W (0.37 A at 5 V). | External 11 to 15 V for the cooler, and the camera needs it even to enumerate | Z1 p12, Z17 |
| DDR3 buffer | None, by implication. The manual lists 256 MB for the Pro only. | 256 MB (manuals) or 512 MB (store table) | Z1 p13, Z3 |
| Interface | USB 3.0 Type-B port (also USB 2.0) and an ST4 port. No hub port. | USB 3.0, a USB 2.0 hub port, and a DC jack | Z1 p11 |
| Back focus, thread | 6.5 mm without the 11 mm extender, M42 × 0.75 | Same | Z1 p13 |
| Mass | 140 g | 410 g | Z1 p5 |
| Operating temperature | Maximum 40 °C in manual V2.2. −5 to 45 °C in V1.7 and V1.2. | Same table | Z1 |

The non-Pro page shows the same two modes, the same gain charts (the labels match), the HCG step at gain 120, and the same frame-rate tables as the Pro, so the mode, gain, noise, and frame-rate numbers above apply as published (D). Assumed from the Pro and unverified for the non-Pro: the SDK bin1 to bin4 mapping, the high-speed mode, the gain range 0 to 570, the binning rules, and the 16-bit scaling. With no buffer, a slow or contended USB link can stall readout, so check `ASIGetDroppedFrames` in video mode (D).

- **Sensor temperature.** `ASI_TEMPERATURE` returns tenths of a degree and is read-only (V). ZWO does not document it for this model, but the owner's SharpCap settings files show a sensor temperature for the ASI294MM (18.3 and 17.6 °C), so this camera reports one. Other uncooled ZWO cameras report it in 0.1 °C steps (S). Check `ASIGetControlCaps` at start-up. The first read after opening returns 0 for about 250 ms. Uncooled bodies run about 4 °C above ambient (S).
- **Dark current** (ZWO chart; camera, mode, and gain not stated; points read off the image, about 5% uncertainty):

| Sensor temperature | 30 °C | 25 °C | 20 °C | 10 °C | 0 °C | −10 °C | −20 °C |
|---|---|---|---|---|---|---|---|
| e⁻/s/pixel | 0.70 | 0.36 | 0.20 | 0.065 | 0.019 | 0.0066 | 0.0022 (printed) |

The doubling temperature is about 5.8 °C between 0 and 30 °C (D). The real camera, measured between 20 and 30 °C with four dark sets, doubles every 4.9 to 5.3 °C, and a bin2 pixel collects 0.47 e⁻/s at 20.2 °C. The QHY294M Pro (same sensor) lists 0.002 e⁻/s at −20 °C (S), and Buil measured 0.0010 e⁻/s at −15 °C on a Pro (S). At a sensor temperature of ambient + 4 °C, a 30 s exposure collects about 0.3 e⁻ at −10 °C ambient, 3.1 e⁻ at 10 °C, and 9.7 e⁻ at 20 °C (D). ZWO documents no optical-black or overscan readout, and the SDK returns only the ROI inside 8288 × 5644.
- **USB power on the Pi.** The Pi 4 supplies up to 1.2 A in total to USB devices (documentation; the datasheet says about 1.1 A). The Pi 5 supplies 1.6 A with a 5 V 5 A USB-PD supply and 600 mA with a 3 A supply, unless `usb_max_current_enable=1` lifts the limit. The camera's 0.37 A leaves 0.23 A on a Pi 5 with a 3 A supply. ZWO's quick guide for uncooled cameras says to connect the camera directly, without a hub or extender, when the preview stalls, and sets Turbo USB to 80 to 90%. ZWO gives no guidance on powered hubs for uncooled cameras.
- **Open items.** ZWO publishes no non-Pro SDK printout, dark-current measurement, or optical-black behavior. A sum versus average question remains for the sensor's bin2 mode: ZWO's diagram shows a sum, and Buil's measurement suggests an average (a photon-transfer test settles it). Buil measured about 84% peak QE for the whole camera against ZWO's estimate of 90%.

Sources: Z16 https://astronomy-imaging-camera.com/manuals/QuickGuide.pdf, Z17 https://www.zwoastro.com/product-faqs/, Z20 https://i.zwoastro.com/wp-content/uploads/2022/06/e3724437d9817d7d0fa07841d8712a2a-2.jpg, Buil https://buil.astrosurf.com/asi294mm.html, QHY294M Pro https://www.qhyccd.com/astronomical-camera-qhy294/, uncooled self-heating https://www.cloudynights.com/forums/topic/834753-asi294mc-non-cooled-why-not/, Raspberry Pi power documentation https://raw.githubusercontent.com/raspberrypi/documentation/master/documentation/asciidoc/computers/raspberry-pi/power-supplies.adoc, Pi 5 USB-PD white paper https://pip-assets.raspberrypi.com/categories/685-app-notes-guides-whitepapers/documents/RP-009856-WP-1-USB%20Power%20delivery%20on%20Raspberry%20Pi%205.pdf.

### Optics

| Item | Value | Tag | Source |
|---|---|---|---|
| Model | GS-250AC (Crayford-style focuser) or GS-250AR (rack and pinion). Both use the same optics. | V | T1 |
| Aperture, focal length, ratio | 50 mm, 250 mm, f/5 | V | T1, T2 |
| Design | Three-element planar apochromatic (PAPO) triplet with an integrated field flattener, one ED element | V | T1 |
| Interface | 1.25 inch drawtube, external M42 × 0.75 thread | V | T1, T2 |
| Image circle | Designed around a "1-inch flat, well-corrected image circle" and sensors up to the IMX533 size (diagonal 15.968 mm). No value in millimetres. | V | T1 |
| Design data | Spot diagram fields 0, 2, 3, 4, and 8.002 mm image height. RMS radius 3.26 to 4.23 µm, geometric radius 6.8 to 8.7 µm, evaluated at 486, 588, and 656 nm only. | V | T3 |
| Coverage | A centered circle of radius 8.0 mm covers 72.9% of the IMX492 area (D) | D | calculation |
| Measured illumination | Two test flats on October 3, 2026 with a phone screen held against the lens (bin2, gain 120; the screen was dimmed and bare). The second test took 24 frames at 14.5 ms in each of two orientations, with the phone turned 180° between them and the bias level taken from the dark library (133.9 counts at 29.5 °C). The flat falls by 0.4% at 0.5° from the middle, 2.3% at 1.0°, 4.05% at 1.5°, 6.5% at 2.0°, and 9.3% at 2.5°, the same in both orientations to 0.1% and the same in the first test, so the sensor is lit to its corners. The whole frame has an rms of 2.5% (0.027 mag). The tilt that stays when the screen turns is 0.56% across the width and −0.34% across the height. The screen's own gradient is 0.64% and 0.68% and turns with it, which is why the first test, with one orientation, read a tilt of 1.1%. One dust shadow is 3.0% deep (71 px across, at x 2489, y 1994), one is 1.6% (65 px), one is 1.4% (58 px), and fainter rings stay under 1%. Two edge artifacts of 1.3 to 1.4% sit at the left edge. The frames show no flicker (0.1% between frames) and no banding (0.17%) at 14.5 ms, and a flat needs no bias frames. The test frames and the flat are outside the repository, and the dust and the camera orientation may change before the installation. | D |  |
| Not specified | Image circle in millimetres, data beyond 8.0 mm, back focus, tube length, transmission, spectral range, glass types, measured image quality | U | T1, T2 |

The vendor data and one forum user (S11, post 47) describe a corrected circle of about 16 mm with softer corners. The project owner tested the scope on the sensor and reports good quality to the edges, so the reference profile uses the full sensor as the usable image circle.

### Derived numbers

Plate scale is 206264.8 × pixel / focal length. Field of view is 2 × atan(N × pixel / (2 × focal length)).

| Mode | Pixel | Scale | Field of view | Diagonal |
|---|---|---|---|---|
| Bin1 | 2.315 µm | 1.910 arcsec/px | 4.395 × 2.994 degrees | 5.316 degrees |
| Bin2 | 4.63 µm | 3.820 arcsec/px | 4.395 × 2.994 degrees | 5.316 degrees |
| Bin3 | 6.945 µm | 5.730 arcsec/px | 4.391 × 2.992 degrees | 5.312 degrees |
| Bin4 | 9.26 µm | 7.640 arcsec/px | 4.395 × 2.992 degrees | 5.315 degrees |

The Airy FWHM for D = 50 mm is 2.34 arcsec at 550 nm and 2.97 arcsec at 700 nm. That is 1.22 and 1.56 pixels in bin1, and 0.61 and 0.78 pixels in bin2. The active sensor area is 19.187 × 13.066 mm (23.213 mm diagonal).

### Conflicts and open items

- ZWO's pages disagree on the DDR3 buffer (256 MB in manuals, 512 MB in one table), the cooler current, the sensor size (19.1 × 13.00 mm and 19.2 × 13 mm), and the binning mechanism (software in the manual, a sensor mode on the product page).
- Not published: raw SDK output for this camera (`MaxWidth`, `BitDepth`, `SupportedBins`), the RAW8 scaling rule, the image type behind the frame-rate tables, USB 2.0 rates, minimum ROI, and an official row time.
- ToupTek publishes no back focus. A forum post relays a support reply that the GS-250 reaches focus with 55 mm of back focus from the M42 thread (S11, post 28).

### Larger scopes: GS-300 and GS-350

ToupTek's store data (read October 1, 2026) gives these specifications. Sources: T1, T2, and the store JSON at https://www.touptekastro.com/products/gs-papo-guide-scope.js, with drawings and spot diagrams on the same shop (T3, T4, T5).

| | GS-250 | GS-300 | GS-350 |
|---|---|---|---|
| Aperture, focal length, f-ratio | 50 mm, 250 mm, f/5 | 50 mm, 300 mm, f/6 | 58 mm, 350 mm, f/6 |
| Price (USD, AC and AR alike) | 199 | 229 | 259 |
| OTA mass | 0.82 kg | 0.94 kg | 1.21 kg |
| Main tube, diameter (drawing) | 200 mm, 62 mm | 250 mm, 62 mm | 300 mm, 72 mm |
| Design field in the spot diagram | Last field 8.002 mm | 8.002 mm | 8.003 mm |
| On-axis RMS radius (design) | 3.26 µm, 2.69 arcsec | 2.95 µm, 2.03 arcsec | 3.29 µm, 1.94 arcsec |

All three use a three-element PAPO triplet with one ED element, a 1.25 inch drawtube, and a rear M42 × 0.75 thread. ToupTek's "supports sensors up to the IMX533 size" statement is series-level, so the design circle stays 16 mm, and it covers 72.9% of the IMX492 area for every model. ToupTek publishes no payload or wind data. A forum teardown counted four lens pieces in a GS-300 (secondary, unverified).

Derived values for the IMX492 and the seeing estimator (D; the Polaris photon budget is a guess good to ±30%, and the ratios between scopes are exact):

| | GS-250 | GS-300 | GS-350 |
|---|---|---|---|
| Plate scale, bin1 and bin2 (arcsec/pixel) | 1.910, 3.820 | 1.592, 3.183 | 1.364, 2.729 |
| Field of view (degrees), area (square degrees) | 4.395 × 2.994, 13.2 | 3.663 × 2.495, 9.1 | 3.140 × 2.139, 6.7 |
| Area of the sensor inside the 16 mm circle (square degrees) | 9.6 | 6.7 | 4.9 |
| Gaia stars on the sensor inside the circle, G < 11, 12, 13 | 233, 510, 1,085 | 162, 354, 754 | 119, 260, 554 |
| Airy FWHM at 0.6 µm (arcsec; pixels in bin1, bin2) | 2.55; 1.34, 0.67 | 2.55; 1.60, 0.80 | 2.20; 1.61, 0.81 |
| Pixel over λN at 0.65 µm (the centroid-phase criterion is below 1), bin1, bin2 | 0.71, 1.43 | 0.59, 1.19 | 0.59, 1.18 |
| Optical transfer function at 1 cycle per pixel, bin2, 0.65 µm | 0.186 | 0.073 | 0.070 |
| In-focus bin2 centroid gain range, polychromatic ideal Airy | 0.53 to 1.48 | 0.73 to 1.28 | 0.73 to 1.27 |
| One-axis image motion at r0 = 5, 10, 15 cm (arcsec) | 0.850, 0.477, 0.340 | same | 0.830, 0.466, 0.332 |
| Outer-scale variance ratio at L0 = 10, 20, 50 m | 0.740, 0.793, 0.848 | same | 0.726, 0.783, 0.840 |
| Exposure bias, single layer, 10 ms at 10 m/s (variance ratio) | 0.835 | same | 0.858 |
| Scintillation rms at 2 ms, airmass 1.155 (theory) | 0.30 to 0.45 | same | 0.31 to 0.46 |
| Polaris, electrons per ms | 8,650 | 8,650 | 11,600 |
| Bin1 peak-pixel fraction; time to 70% of full well | 0.386; 3.0 ms | 0.293; 4.0 ms | 0.290; 3.0 ms |
| Detection depth change, regime range | 0 | −0.2 to +0.2 mag | −0.04 to +0.53 mag |
| Spectrum corner at 5 m/s (Hz) | 20 to 50 | same | 17 to 43 |
| Trail in 5 s, bin1: Polaris; far edge of the circle (pixels) | 0.46; 1.7 | 0.55; 1.8 | 0.64; 1.9 |

At f/6 the in-focus bin2 gain swing falls from ±47% to ±27%, which is better but not good enough. A uniform blur of d pixels leaves a residual swing of about ±1.8% at d = 3 for f/5 and ±1.9% at d = 2 for f/6 (focus offset N × d × 4.63 µm, 69 µm at f/5). A Strehl ratio near 0.6, as the vendor's spot radius suggests (a guess), would cut the f/5 swing to about ±29%. A long-pass filter at 0.77 µm would make bin2 bias-free at f/6 but passes about 12% of the photons. Bin1 needs no defocus at either focal ratio. The GS-350 collects 35% more light and gains no peak-pixel headroom, because the star spreads over the same number of pixels.

Wind and mount: the vendor mass, tube length, and side-profile ratios are 1 : 1.15 : 1.48, 1 : 1.25 : 1.5, and 1 : 1.25 : 1.74. A rod model on an unchanged mount (a guess) gives a wind-limited shake variance of 2.4 times for the GS-300 and 6.8 times for the GS-350. Angular shake does not depend on focal length, so pixel jitter scales with it. Harlan and Walker (1965) found that mount stiffness decides the usable wind speed.

Two-star geometry at bin1 and 30 arcmin separation: the stars sit 942, 1,131, and 1,319 pixels apart, and the ROI is 973, 1,162, and 1,350 pixels wide by up to 93 rows (100 fps). The pair's axis must stay within 3.8, 3.1, and 2.7 degrees of the sensor rows, and rotation takes it out of that range every 30, 25, and 21 minutes. Rolling-shutter skew between the two stars leaves a common-mode residual of 2 sin(π f Δt) for vibration at frequency f, so the axis must stay within 0.2 to 0.6 degrees of the rows unless the row delay is corrected. In bin2 the ROI needs 483 rows and the pair can take almost any orientation.

Verdict, from the seeing-theory comparison: keep the GS-250. The GS-300 is neutral to slightly negative (the atmospheric numbers do not change, the field shrinks by 31%, and the shake guess is 2.4 times). The GS-350 is negative for the seeing estimator (the field halves, scintillation rises 2 to 5%, mass rises 48%, and the shake guess is 6.8 times), with a marginal gain for faint-star photometry. The verdict changes if a recorded in-focus bin2 histogram is already flat, if the mount is stiff, or if stars fainter than G = 13 matter.

## Camera access options

### ZWO ASI SDK (V1.41, January 12, 2026)

| Topic | Finding | Tag | Source |
|---|---|---|---|
| License | MIT (Expat), copyright ZWO Company 2015, in the copy that INDI and Debian vendor. Debian files libasi under non-free because the binaries ship without source. ZWO's own V1.41 archives, for Windows and for Linux and macOS, each carry the MIT license text with the notice "Copyright (c) 2015, ZWO Company" (checked on October 2, 2026, after this research ran). | V | https://raw.githubusercontent.com/indilib/indi-3rdparty/master/libasi/license.txt, https://sources.debian.org/src/libasi/1.27%2B20221218230335-2/debian/copyright/, Z13 |
| Platforms | Linux armv8 and x64, Windows x64 (needs ZWO driver V3.28), macOS. Several projects report Pi 3, 4, and 5 working. | V, S | https://github.com/indilib/indi-3rdparty/tree/master/libasi, https://www.zwoastro.com/software/camera-driver/ |
| Frame timestamps | None. Only GPS camera variants return time (`ASI_GPS_DATA`). | V | Z11 |
| Video mode | `ASIGetVideoData` waits `wait_ms` (vendor advice: twice the exposure plus 500 ms). The internal buffer is small, and a late read discards frames. | V | Z11 |
| Dropped frames | `ASIGetDroppedFrames` counts from capture start and resets when capture stops. Whether corrupt frames count is unknown. | V; U | Z11 |
| ROI | `ASISetROIFormat` needs capture stopped. `ASISetStartPos` can move the ROI while streaming. A third-party library saw `ASISetROIFormat` succeed mid-stream with silent geometry changes, and saw `ASIStopVideoCapture` fail to cancel a blocked read. | V; S | https://github.com/DegorasProjectTeam/DegorasASI |
| Bandwidth | `ASI_BANDWIDTHOVERLOAD` sets the transfer percentage. INDI advises 40 on ARM when frames break. | V | https://raw.githubusercontent.com/indilib/indi-3rdparty/master/indi-asi/README.md |
| USB buffer | Linux defaults to 16 MB of usbfs memory. INDI's udev rule writes 1024, and other guides write 200. | V, S | https://raw.githubusercontent.com/indilib/indi-3rdparty/master/libasi/99-asi.rules |
| Stalls | An INDI issue (April 2026) reports ZWO cameras on Pi boards that stop delivering after hours or days, and only a power cycle recovers them. | S | https://github.com/indilib/indi/issues/2361 |
| USB power switching | `uhubctl` cannot switch single ports on the Pi 4B or Pi 5. Pi 4B ports are ganged per hub. On the Pi 5, all four ports are ganged and VBUS drops only when all are off. | V | https://github.com/mvp/uhubctl |
| Python bindings | `zwoasi` (MIT, ctypes, release 0.2.0 in January 2024, last commit September 2025) wraps video, dropped frames, and ROI. Plain `ctypes` also works, and `ctypes` releases the GIL during a foreign call. | V | https://github.com/python-zwoasi/python-zwoasi |

### INDI

- **Packaging.** Debian 12 and 13 ship `indi-asi` 2.2+20221225 (contrib) with `libasi` 1.27 (non-free) and INDI 1.9.9. Upstream is at 2.2.4.2 (August 2026). The ASI driver contains a January 2026 fix for streams that get stuck, which the Debian build lacks.
- **Streaming.** The stream manager sets the camera exposure to 95% of the frame period, copies each frame into a queue, and crops in software (`CCD_STREAM_FRAME`). Hardware ROI uses `CCD_FRAME`. The driver never calls `ASIGetDroppedFrames`. No published measurement covers a 128 × 128 ROI at 90 fps.
- **Timestamps.** Protocol timestamps have one-second resolution. The SER recorder stamps frames with the host clock when it dequeues them, unless a driver supplies a time.
- **License.** INDI is LGPL-2.1. `pyindi-client` is GPL-3.0 or later. `indipyclient` is MIT and pure Python.

### ASCOM Alpaca

- Alpaca is REST over HTTP with JSON bodies and UDP discovery on port 32227. The device classes include no video device (https://ascom-standards.org/alpyca/alpacaclasses.html).
- A camera exposure is `StartExposure`, then `ImageReady` polls, then `ImageArray`. `LastExposureStartTime` is a FITS UTC string with no stated accuracy (https://ascom-standards.org/newdocs/camera.html).
- The binary `ImageBytes` transfer cut a 6000 × 4000 download from 15.1 s to 1.1 s on a wireless link (https://raw.githubusercontent.com/ASCOMInitiative/ASCOMRemote/main/Documentation/AlpacaImageBytes.pdf).
- Servers for ZWO cameras: ZWO's Windows COM driver exposed through ASCOM Remote (GPL-3.0), AlpacaBridge (AGPL-3.0 or later with an SDK exception, Debian 13 arm64), and `astrocam` (MIT, talks to the camera without the SDK and adds a non-standard video action). `alpyca` (MIT) is the Python client.

### Recommended recovery ladder (inferred)

1. A read times out: stop the reader, restart capture, and count a gap.
2. Repeated timeouts: stop capture, close and reopen the camera, and reapply ROI, mode, and controls.
3. The camera is removed or still dead: reset the USB device through sysfs or `USBDEVFS_RESET`.
4. Power-cycle the hub (all ports together on a Pi), or cut the camera's supply through a relay.
5. Restart the worker process that holds the SDK. Do this earlier if the process hangs.
6. Reboot last.

## Plate solvers and catalogs

Tags: **P** primary source, **S** secondary, **I** inferred (test on hardware).

### Modes, sampling, and trails

- Bin2 frames are 4144 × 2822 at 3.82 arcsec/pixel (23 MB). Native frames are 8288 × 5644 at 1.91 arcsec/pixel (94 MB). All three solvers accept both scales (`--scale-low` and `--scale-high`, `-fov`, `fov_estimate`) (P).
- Cedar Detect needs under 10 ms per megapixel, about 0.5 s for a native frame on a Pi 4 (P). ASTAP or `image2xy` need an estimated 5 to 35 s (I). Software binning sums four 12-bit reads and loses the 14-bit ADC (I).
- Airy FWHM is 0.6 pixel in bin2 and 1.2 pixels in bin1 (I). Cedar Detect may mistake bin2 stars for hot pixels, and ASTAP's hot-pixel cut (HFD under 0.8 px) overlaps star HFD (P, I). An ePSF fit reaches 0.02 px on bright stars, while center-of-gravity centroids carry pixel-phase bias.
- A fixed mount trails stars at 15.04 arcsec/s × sin(distance from the pole). At 2.7 degrees and 30 s that is 21 arcsec (5.6 px in bin2). Cedar Detect rejects trails longer than about 1 px (P).

### Solver comparison for the cap-only case

| | astrometry.net | ASTAP CLI | cedar-solve with Cedar Detect |
|---|---|---|---|
| License | GPL v3 or later (the readme says v2) | MPL-2.0 | Apache-2.0. Detector: FSL-1.1-MIT. |
| Pi install | `apt`, arm64, version 0.97 | aarch64 zip, 312 kB | Pins (numpy below 2, Pillow below 9) clash with Python 3.13. The detector needs a Rust build. |
| Detector | Your own star list, or `image2xy` | Internal only | Cedar Detect (8-bit) or any centroids |
| Cap database | Stock indexes 4108 to 4113: 187 MB. Custom `build-astrometry-index -E`: 3 to 6 MB (I). | Copy 28 area files (declination 72 and above): about 25 MB of D50 (I) | Gaia subset in `hip_main` layout: 4 to 15 MB, 0.5 to 6 min build (I) |
| Pi 4 and Pi 5 solve (I) | 0.4 to 2 s and 0.2 to 1 s with a star list | 1.5 to 5 s and 0.6 to 2 s including detection | 25 to 250 ms and 10 to 100 ms, solve only |
| Output | TAN and SIP, matched stars | CD matrix and SIP | Rotation, field of view, one radial term, no SIP |
| Status | 0.97 (December 2024), commit August 2026 | v2026.09.19 | Commits August 2026, PyPI 0.5.1 (June 2024) |

Published anchors: a Pi 5 solves in under 1 s with astrometry.net and 12 ms with Cedar (S). On a Pi 4, a 20 MP frame took 11 s in astrometry.net and 15 s in ASTAP (S). ASTAP blind solves at a 90-degree offset took 4.8 to 23.8 s (P). Skipped: olive-solve, StellarSolver, Siril, twirl, Watney. Watch: tetra3rs (MIT, aarch64 wheels, alpha).

### Behavior near +90 degrees

astrometry.net uses unit vectors and per-quad tangent planes (P). ASTAP has explicit pole code, fixed on December 30, 2023 (P). tetra3 skips proper motion within 2.9 degrees of the pole, and its RA and roll outputs are ill-conditioned there (P, I). A WCS with `CRVAL2` of exactly +90 degrees is a special case (P). The design stores attitude as a rotation matrix, reports the pole's pixel position, and propagates proper motion in three dimensions.

### Thin in-house solver

A predict, match, and least-squares refinement step is defensible for pointing tracking when a proven solver provides recovery (I). It is not defensible as the only path for the alignment helper, because hot pixels, clouds, a saturated Polaris, and trails need hardened code (I).

### Gaia DR3 cap subset

Live Gaia archive queries on October 1, 2026 (P):

| Radius | G < 11 | G < 12 | G < 13 | G < 14 |
|---|---|---|---|---|
| 10 degrees | 7,540 | 16,738 | 35,458 | 72,539 |
| 15 degrees | 17,097 | 38,233 | 82,055 | 169,196 |

At 44 bytes per row, G < 13 and G < 14 take 1.6 and 3.2 MB at 10 degrees, and 3.6 and 7.4 MB at 15 degrees. The catalog file that phase 2 writes takes 50 bytes a row, so the 15 degree cap to G = 13 takes 4.1 MB (82,065 stars in the real build of October 2, 2026). A frame holds about 320, 700, and 1,490 catalog stars at G < 11, 12, and 13 (I). Use an asynchronous ADQL job (anonymous limit 3 million rows). A synchronous query silently truncated at 16,385 of 35,458 rows (P).

### Photometry

- Gaia's bright limit is about G = 3, and G precision is 0.3 mmag below G = 13 (P). Polaris is absent from Gaia: the brightest source within 5 arcmin is Polaris B (G = 8.63). Tycho-2 has Polaris (VT 2.038, no proper motion) and is 99% complete to V = 11.0 (P).
- G − V = −0.02704 + 0.01424 (BP−RP) − 0.2156 (BP−RP)² + 0.01426 (BP−RP)³, scatter 0.030 mag, valid for −0.5 < BP−RP < 5.0 (P, Gaia DR3 documentation). Tycho-2: V = VT − 0.090 (BT − VT) (P). APASS DR10 covers about V = 7.5 to 16.5 (S).
- Sky brightness: SB = ZP − 2.5 log10 (sky rate per arcsec²), and per pixel add 2.5 log10 of the pixel solid angle (2.910 mag for bin2) (I). Fit a BP−RP color term, because the camera band is not Gaia G.
- An SQM band (TSL237 with a Hoya CM-500 filter, with a near-infrared leak) differs from V by 0 to 0.25 mag depending on the sky spectrum, and no universal conversion exists. The SQM also sums light over about 20 degrees. Fit one offset against a co-located SQM and test it against moon and airglow (P, I).

### Sources

astrometry.net: https://astrometry.net/doc/readme.html, https://astrometry.net/doc/build-index.html, https://packages.debian.org/trixie/astrometry.net. ASTAP: https://www.hnsky.org/astap.htm, https://github.com/han-k59/astap. cedar-solve and Cedar Detect: https://github.com/smroid/cedar-solve, https://github.com/smroid/cedar-detect. tetra3: https://github.com/esa/tetra3. SEP: https://sep.readthedocs.io/en/stable/api/sep.extract.html. Gaia DR3 photometric relations: https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photSystem/cu5pho_ssec_photRelations.html. Gaia TAP: https://gea.esac.esa.int/tap-server/tap/sync. Tycho-2: https://cdsarc.cds.unistra.fr/ftp/I/259/ReadMe. Timing anchors: https://astrokeith.com/blogs/latest-blogs/plate-solvers-compared.html.

## Seeing theory

Tags: **V1** verified in the primary text, **V2** verified in secondary text only, **D** derived by calculation (the calculations first reproduced published numbers), **R** recalled and not verified.

### Image-motion variance and seeing

For a circular aperture of diameter D under Kolmogorov turbulence, the one-axis variance is σ² = K λ² r0⁻⁵ᐟ³ D⁻¹ᐟ³ (rad²), and the two-axis variance is twice that.

| Item | Value | Tag | Source |
|---|---|---|---|
| Centroid (G-tilt) coefficient | K = 0.170 | V1 | Martin 1987, PASP 99, 1360, Eq. 7. Kellerer and Tokovinin 2007, A&A 461, 775, Eq. 6. |
| Zernike tilt coefficient | K = 0.182 | V1 (as the DIMM limit 0.364), D | Tokovinin 2002, PASP 114, 1156, Eqs. 7 and 8 |
| Seeing and wavelength | FWHM = 0.98 λ/r0. r0 scales as λ^(6/5), seeing as λ^(−1/5), and λ² r0⁻⁵ᐟ³ is achromatic. | V1 | Tokovinin 2002, Eq. 5 |
| Outer scale, FWHM | (ε_vK/ε0)² = 1 − 2.183 (r0/L0)^0.356, valid for L0/r0 above 20 | V1 | Tokovinin 2002, Eq. 19. Martinez et al. 2010, A&A 516, A90. |
| Outer scale, tilt variance | G-tilt: 1 − 1.525 (D/L0)^(1/3). Z-tilt: about 1 − 1.42 (D/L0)^(1/3). | V2, D | Aristidi et al. 2019, MNRAS 486, 915, Eq. 27 |
| Exact ratio, D = 50 mm, G-tilt | 0.740, 0.793, 0.848 for L0 = 10, 20, 50 m | D | integration |

Ignoring the outer scale makes seeing too low by R^(3/5): 15%, 12%, and 9% for L0 of 10, 20, and 50 m. A single aperture feels the outer scale and a DIMM does not (Martin 1987, Tokovinin 2002, V1).

Worked example, D = 50 mm, λ = 0.65 µm, r0 (500 nm) = 0.10 m (D): r0 (650 nm) = 0.137 m. One-axis rms is 0.477 arcsec (G) or 0.494 arcsec (Z). Seeing is 1.011 arcsec at 500 nm and 0.959 arcsec at 650 nm. A closed form for D = 50 mm is ε500 = 2.36 σx^1.2 (Z-tilt, arcsec).

One-axis rms motion at D = 50 mm (G-tilt): r0 of 5, 10, and 15 cm gives 0.85, 0.48, and 0.34 arcsec, which is 0.45, 0.25, and 0.18 px in bin1 and 0.22, 0.125, and 0.09 px in bin2.

### Exposure time

The controlling parameter is ξ = vT/D (Martin 1987, V1). For T = 10 ms and D = 50 mm, ξ is 1, 2, and 4 at 5, 10, and 20 m/s. The single-layer variance ratio is 0.93, 0.83, and 0.72 (G-tilt), so seeing reads 4, 10, and 18% low (D). Martin's layered model gives seeing 2 to 27% low at 10 ms, 0.3 to 7% low at 2 ms, and 0.1 to 2% low at 1 ms (D, reproducing Martin's Table IV). Martin recommends scaling the exposure with d/T. Extrapolation to zero exposure costs about 1.9 times the precision for a single aperture, so shortening the exposure is cheaper (D).

### Sampling and centroid window

For a box-integrated pixel and a plain centroid, the phase bias depends on the image's optical transfer function at the sampling frequency. A short-exposure image through a clear aperture has no spatial frequencies above D/λ, so a pixel smaller than λ/D gives no centroid phase bias (D). Bin1 (1.91 arcsec) meets this for λ of 0.46 µm or more. Bin2 (3.82 arcsec) does not. In focus, the Airy bias amplitude in bin2 is 0.06 to 0.076 px, the centroid gain swings between 0.53 and 1.48, and the variance factor ranges from 0.44 to 1.8 with a 22 s period as Polaris drifts across pixels. A blur of 3 pixels (11 arcsec, 69 µm of focus offset at f/5) leaves about ±1.8% gain variation. With the vendor's design blur (a Strehl ratio near 0.6, a guess), the f/5 in-focus swing is about ±29% instead of ±47%. At bin1, plain-centroid gain is 0.976 for an 11 × 11 window and 0.983 for 15 × 15 (variance 3 to 5% low). The toy Gaussian simulation in the calculations below agrees with an analytic series (D). Photon plus read noise stay at or below 0.02 px for Polaris and must be subtracted at 1 ms.

### Absolute versus differential motion

Absolute motion carries mount vibration, wind shake, and drift (Tokovinin 2002, Kellerer and Tokovinin 2007, Aristidi et al. 2019, Xin et al. 2020; V1). A linear detrend over 60 s removes 0.3 to 1% of the variance for L0 of 20 to 100 m and 9% for Kolmogorov. A 1 Hz high-pass removes 11 to 16% at L0 of 20 to 50 m (D). A second star in the field rejects common-mode motion. A 5 to 15 arcmin pair suppresses layers below 30 to 100 m, while a pair 30 to 60 arcmin apart keeps at least 50% of the signal from layers at 10 m (D). The second star needs V of about 6 or brighter.

Fixed-telescope Polaris monitors exist: Harlan and Walker 1965 (PASP 77, 246: a 16.5 cm refractor on a concrete-block mount, star trails), Cromwell et al. 1987 (a 30 cm telescope with 17 ms exposures, known through Martin 1987), the ARIES 25 cm polar trail telescope, and Ma et al. 2016 (SPIE 9906). No modern CMOS product was found.

### Scintillation

The index is σI² = (⟨I²⟩ − ⟨I⟩²)/⟨I⟩². Young's law is σ² = 10e-6 C_Y² D^(−4/3) t⁻¹ (cos z)⁻³ exp(−2h/H) with C_Y of 1.3 to 1.67 (Osborn et al. 2015, MNRAS 452, 1707, Eq. 7; V1). For D = 50 mm, near the Fresnel radius, Young's law overshoots. With a wave-optics correction, the rms is about 0.10 to 0.20 at the zenith, 0.15 to 0.25 at Polaris's airmass, and 0.3 at airmass 2 (uncertain by a factor of about 1.5) (D). At 10 ms the exposure is 5 to 8 times the knee (0.62 D/v), so the variance falls as 1/T. The Poisson floor is 1/N, about 1.1e-5 per 10 ms, which is negligible.

### Spectrum and statistics

For a single frozen layer, the tilt spectrum has slope −2/3 up to a corner at 0.2 to 0.5 v/D, then −11/3 (centroid) or −17/3 (Z-tilt). A finite outer scale flattens it below about v/L0. The corner lies at 20 to 100 Hz for D = 50 mm and v of 5 to 10 m/s, so a 90 fps series can miss it. The coherence time is τ0 = 0.314 r0/V (Kellerer and Tokovinin 2007, Eq. 4; V1). A mount vibration shows as a line above the smooth curve (a 40 Hz example: Morzinski et al. 2010, V1). The relative error of a variance from N independent Gaussian samples is √(2/(N−1)), and seeing carries 0.6 times that (V1, V2). For correlated data, N_eff is T/τ_int with τ_int of 10 to 60 ms, so a 60 s window gives 2 to 4% in variance (D). Non-stationarity dominates in practice.

### Sources

Tokovinin 2002 (https://iopscience.iop.org/article/10.1086/342683). Martin 1987 (https://iopscience.iop.org/article/10.1086/132126). Harlan and Walker 1965 (https://iopscience.iop.org/article/10.1086/128210). Kellerer and Tokovinin 2007 (https://arxiv.org/abs/astro-ph/0610207). Kornilov and Safonov 2011 (https://arxiv.org/abs/1108.5681). Kornilov et al. 2007 (https://arxiv.org/abs/0709.2081). Martinez et al. 2010 (https://arxiv.org/abs/1003.4593). Aristidi et al. 2019 (https://arxiv.org/abs/1904.07093). Osborn et al. 2015 (https://arxiv.org/abs/1506.06921). Dravins et al. 1997 and 1998 (https://iopscience.iop.org/article/10.1086/133872, https://iopscience.iop.org/article/10.1086/316161). Bradshaw 2020 on biased moments of undersampled sources (https://arxiv.org/abs/2012.05528). Xin et al. 2020 (https://arxiv.org/abs/2004.12128). Morzinski et al. 2010 (https://arxiv.org/abs/1007.2691). Not retrieved: Sarazin and Roddier 1990, Young 1967, Conan et al. 1995 and 2000, Sasiela and Shelton 1993, and Tyler 1994.

## Long-term science estimates

| Item | Evidence and arithmetic | Tag |
|---|---|---|
| Known variables in the field | A VizieR TAP query on the VSX catalog (table `B/vsx/vsx`) within 2.7 degrees of RA 37.95°, Dec +89.26° returned 1,079 variables. With a maximum magnitude brighter than 13 it returned 139: 62 rotational, 20 δ Scuti, γ Doradus, or SX Phoenicis, 10 W Ursae Majoris, 5 Algol-type eclipsing, 5 semiregular, 5 irregular, 3 α² Canum Venaticorum, 3 RS Canum Venaticorum, and singles that include a classical Cepheid (Polaris) and an RR Lyrae. With a period over 30 days it returned 14: 9 semiregular, 2 Mira, 2 rotational, and 1 mixed. The 2.7 degree cone covers 22.9 square degrees, and the rectangular field covers about 13 (a ratio of 0.57). | V |
| Solar-system objects | IMCCE SkyBoT cone searches (3 to 4 degree boxes around Polaris) on January 15, February 15, March 15, April 15, May 15, June 15, and October 1, 2026 each returned "No solar system object was found". The celestial pole lies at ecliptic latitude 66.6 degrees (90° minus the 23.44° obliquity), and the field spans about 63 to 70 degrees. | V |
| Aberration, precession, nutation | The annual aberration ellipse has semi-axes 20.5″ and 20.5″ × sin β = 18.8″ (β = 66.6°). Precession moves the pole 20.04″ per year relative to the stars. Nutation adds a 9.2″ swing in obliquity with an 18.6-year period, so the pole's rate varies by up to 9.2″ × 2π/18.6 = 3.1″ per year. In pixels, 20.04″ per year is 10.5 px (bin1) or 5.2 px (bin2), and one month is 0.9 px (bin1). | R (standard values, not re-fetched) |
| Sky clock | A 0.05 px centroid error on 320 stars at an rms distance of 2,800 px (bin1) from the pole gives a rotation error of 1.0 × 10⁻⁶ rad (0.21″) per frame. Divided by Earth's rotation rate (7.29 × 10⁻⁵ rad/s), that is 14 ms per frame and 1.1 ms per night of 160 frames. Mount rotation about the polar axis is degenerate with time: 1″ equals 66 ms. | D |
| Polaris pulsation | Period 3.97 days. The period grows about 4.5 s per year (literature). Per cycle the period grows 0.049 s, so the timing shift after N cycles is 0.5 × 0.049 s × N²: 920 cycles (10 years) give 5.8 hours, and 184 cycles (2 years) give 14 minutes. The amplitude exceeded 0.1 mag before the 1960s, fell below 0.05 mag after 1966, grew again, and fell again in 2017 to 2018 (literature). | D, S |
| Photometric precision | Mag 0 gives about 4.6 × 10⁷ e⁻/s in the unfiltered camera. A V = 8 star gives 2.9 × 10⁵ e⁻ in 10 s (photon noise 0.19%), and V = 10 gives 4.6 × 10⁴ e⁻ (0.47%). Scintillation at 10 s is 0.2 × √(0.01 s/10 s) = 0.63%, which dominates. The total is 0.66 to 0.79% per frame, or 0.15 to 0.18% per hour of 20 frames. Polaris in fast mode at 2 ms: the theory gives 0.3 to 0.45 scintillation rms per frame, and 90 fps gives about 5,400 frames per minute, so the minute-level scatter is 0.4 to 0.6% and the night-level scatter is about 0.03%, well below the transparency calibration. | D |
| Storage | `star_list`: 320 stars × 16 bytes × 160 frames per night × 365 nights is 0.30 GB per year. `star_epoch`: 1,500 stars × 24 bytes × 365 nights is 13 MB per year. | D |

Polaris sources: https://arxiv.org/abs/0804.3593, https://arxiv.org/abs/0810.4371, https://arxiv.org/abs/1610.03813, https://doi.org/10.1093/mnrasl/sly170, https://arxiv.org/abs/2309.03257. VSX in VizieR: https://cdsarc.cds.unistra.fr/viz-bin/cat/B/vsx. VizieR TAP: https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync. SkyBoT: http://vo.imcce.fr/webservices/skybot/.

### Which goals this setup can observe

The table above gives the arithmetic for each goal. This table says whether the reference setup can observe it: the GS-250 on the uncooled ASI294MM in bin2 (3.82 arcsec per pixel), a fixed mount with the pole in the middle of the frame, and the default survey step of one 30 s frame every 180 s. The verdicts mean:

- **Yes**: the statistics allow it, and no effect that we know of blocks it.
- **Limited**: the statistics allow it, and an effect that nobody has measured yet sets the real limit.
- **Check only**: the setup sees the effect but cannot tell it from motion of the mount, so it tests the pipeline and teaches nothing new.
- **No**: the signal lies below what the setup can reach.

A verdict that depends on the mount, the flat field, or the real sky stays an estimate until the first nights (B1). The frame precisions use 30 s exposures, the survey default, where the photometric precision row above uses 10 s. Added on October 2, 2026.

| Goal | Observable here? | Evidence and arithmetic | Tag |
|---|---|---|---|
| Aberration | Limited | The ellipse (20.5″ × 18.8″) is 5.4 × 4.9 px in bin2. The mean position of the field has a statistical error of 0.003 px per frame (0.05 px per star over 320 stars), so the signal is about 2,000 times the noise. A mount that tilts with the seasons adds to the ellipse and looks like it, and nobody has measured that tilt. | D, U |
| Precession and nutation | Check only | Precession (20.04″ a year, 5.2 px) and nutation (9.2″ over 18.6 years, at most 0.8 px a year) look like slow motion of the mount, and the pointing series has no independent reference. The residuals after the apparent-place correction measure the stability of the mount, which the brief asks for. | D |
| Sky clock | Check only for clock steps and mount roll. No for Earth rotation | The roll error is 0.21″ per frame (14 ms) and 1.1 ms per night. A step of 50 ms in the system clock turns the field by 0.75″, which is 3.6 times the frame noise, so the sky catches clock steps. UT1 − UTC drifts by up to about 2 ms a day, and 1.1 ms equals 17 milliarcseconds of mount roll, which no mount holds. | D, R |
| Stellar astrometry | Limited, and it adds nothing new | A star's position has 0.19″ of noise in a frame and 15 mas in a night (160 frames). Five years of about 100 nights a year give a differential proper motion near 1 mas a year on statistics. The distortion, the focus shift with temperature, and the pixel-phase bias of the bin2 centroid (0.06 to 0.076 px peak to peak in focus) are unmeasured, and Gaia gives these proper motions better. Use it as a check of the error budget. | D, U |
| Variable stars | Yes at G < 12 with 0.03 mag or more. Limited at G = 12 to 13 | A 30 s frame has a precision of 0.40, 0.46, 0.58, 0.87, and 1.5% at G = 9, 10, 11, 12, and 13 (photon, sky, read, and scintillation noise, with a sky of 21.0 mag/arcsec² in the camera band). A winter month at 30% clear nights has about 2,000 frames, which gives a periodic signal of 0.01 mag (peak to peak) a signal-to-noise ratio of 17 at G = 12 and 10 at G = 13 on random noise alone. The 1% systematic floor of the zero point and the flat field set the real limit near 0.02 to 0.03 mag. A repeat of the VSX query on October 2 with a cone of 2.7° around the pole returned 976 variables, and 132 reach 13 mag at maximum (magnitudes as VSX lists them, mostly V and Gaia G). The first query, around Polaris, returned 1,079 and 139, and I did not find the cause of the difference. 59 of the 132 reach 12 mag, and 35 of those 59 vary by 0.03 mag or more. 14 of the 35 sit inside the 1.5° circle that stays on the sensor at every roll (see the geometry below). | D |
| Polaris pulsation | Yes for the cycle and the amplitude. Limited for the period change | Fast mode gives 0.4 to 0.6% a minute and 0.03% a night (the photometric precision row above), and the survey transparency calibrates it to about 1% (the systematic floor), against an amplitude below 0.05 mag (5%). The 3.97 d cycle shows in one clear week. The period grows 4.5 s a year, so the timing moves 14 minutes in 2 years and 0.9 hours in 4. With a systematic scatter of 0.7% in the hourly flux and a semi-amplitude of 2.3%, a clear month fixes the phase to about 45 minutes, so the period change needs 3 to 4 years of data. | D |
| Eclipses | Yes | 17 eclipsing or ellipsoidal systems reach 13 mag, and 4 reach 12 mag. Their depths run from 0.09 to 0.99 mag, against 0.9 to 1.5% per frame at G = 12 to 13, so a 3 minute cadence fixes the minima to a few minutes. | D |
| Transits | Limited | A transit of 1% depth on a star of G = 11 to 12 has a signal-to-noise ratio of 9 to 13 on random noise (60 frames in 3 hours at 0.58 to 0.87%). Trends of 0.1 to 0.3% in a night cut that. Only a giant planet around a bright star qualifies, and nobody has searched the field for candidates. | D, U |
| Flares, novae, and transients | Yes for bright events | A dimming or flare of 1% that lasts 3 hours on a star of G = 10 has a signal-to-noise ratio of about 17 (60 frames at 0.46%). `star_list` keeps every unmatched detection for a year, so a nova or a new source is there to find. | D |
| Moving objects | Satellites and aircraft: yes. Asteroids: no. Comets: rare | A satellite crosses the 4.4° field in a few seconds (R) and leaves a streak in a 30 s frame. No code searches for streaks, because the plan leaves that analysis until after commissioning. The field lies at ecliptic latitude 63 to 70°, and SkyBoT found no asteroid or comet on seven dates. A long-period comet can arrive from any direction, so the field can catch one, rarely. | D, V |
| Sky and atmosphere | Yes for brightness, transparency, clear fraction, and events. Limited for a trend of 0.01 mag a year | Each frame gets a zero point against Gaia G (close to the camera band) with a 0.01 mag systematic floor, and a sky brightness good to 0.1 mag in the camera band and 0.2 to 0.3 mag in V. A trend of 0.01 mag a year needs a stable flat field, dark model, and lens transmission (dust, dew) for years, and only the star ensemble and the SQM-LE readings check that. Airglow follows the solar cycle by a few tenths of a magnitude (R). | D, R, U |
| Instrument aging | Yes | The dark library holds sets by temperature, and 0.3 to 0.6% of the pixels are hot at 30 s. A sea-level muon rate of about 1 per cm² per minute (R) gives about one hit in each 30 s frame on the 2.5 cm² sensor, so a week of frames fixes the hit rate to a few percent. | D, R |
| Deep stack of faint diffuse light (IFN) | Limited | The photons suffice by a wide margin. The flat field, the sky gradients, and the halo of Polaris decide. See the next section. | D, U |

## Deep stack of faint diffuse light (IFN)

A fixed camera that stares at one field every clear night can stack its long frames into one very deep image. The field lies on the Polaris Flare, a high-latitude dust cloud that amateurs image as an integrated flux nebula (IFN): dust that the starlight of the whole Galaxy lights, and not one star. This section estimates what the reference setup could see and what a stack would need from the system. Nothing here exists in the code, and no real frame has tested it. The sky values are assumptions until the first nights (B1). Added on October 2, 2026.

### The target

| Item | Value | Tag | Source |
|---|---|---|---|
| IFN | Sandage (1976) described these nebulosities at high Galactic latitude. The brightest parts show on the Palomar Sky Survey prints, near their limit, at about 24.5 to 25 mag/arcsec² in V, and the faint structures lie about 2 mag below that. | S | https://adsabs.harvard.edu/pdf/1976AJ.....81..954S (not re-read), https://www.bbastrodesigns.com/Herschels%20Ghosts.html |
| The Polaris Flare | A cloud of molecular gas and dust around the north celestial pole. A CO survey found it over 80% of 50 deg², and Herschel mapped 10 deg² of it. | V | https://www.osti.gov/biblio/6806251 (Heithausen and Thaddeus 1990), https://arxiv.org/abs/1005.2746 |
| Dust in the field | On October 2, 2026, the IRSA service for the dust map of Schlegel, Finkbeiner, and Davis (1998) gave 5.1 to 5.6 MJy/sr at 100 µm within 2° of the pole (E(B−V) 0.25), with a scatter of 2%. Four patches 2° wide, centered 3° from the pole, gave 3.5 to 4.9 (RA 0 h), 4.0 to 4.6 (6 h), 8.6 to 9.9 (12 h), and 3.5 to 3.9 MJy/sr (18 h). | V | https://irsa.ipac.caltech.edu/applications/DUST/ |
| Optical brightness | Witt et al. (2008) fit five high-latitude clouds with I_B = 2.13 × 10⁻³ (I_100 − 0.65), in MJy/sr. At the pole that gives 25.5 mag/arcsec² (AB, B), which is about 25.0 in V for the flat spectrum of their clouds, and about 24.3 in V in the 9 MJy/sr patch. The relation comes from other clouds, and the far-infrared opacity per magnitude of extinction in the Flare is four times the value that the map assumes. That puts the optical light below the relation by up to 1.5 mag, so I use 25 to 27 mag/arcsec² in V, with 26 as the central guess. | V, D, U | https://arxiv.org/abs/0802.0674, https://arxiv.org/abs/astro-ph/0106507 |
| Contrast with the sky | I assume a camera-band sky of 20.5 to 21.0 mag/arcsec² (the survey's `sky_mag_arcsec2` measures it) and a camera-band brightness of IFN equal to its V brightness. IFN at 25 is then 1.6 to 2.5% of the sky, at 26 it is 0.6 to 1.0%, and at 27 it is 0.25 to 0.4%. | D, U | |

### Geometry of the stack

The sky turns about the pole at 15.04 arcsec per second, and the aligned camera has the pole at the middle of its frame (the Align page draws the orbit of Polaris around it). A stack that rotates each frame to the sky therefore covers a disk of 1.50° radius at every roll angle (7.0 deg², 53% of the 13.2 deg² frame). The coverage falls to half at 2.1° and ends at 2.66°, the half-diagonal, and the area with half coverage or more is 14.1 deg² (D). The 8.0 mm design circle of the optics (1.83°) contains the full-coverage disk, so the disk is also the part of the frame that the vendor's design data cover. A star at 1.5 to 2.66° from the pole is on the sensor for only part of the sidereal day. During a 30 s frame the field turns 0.125°, which smears a diffuse structure by 12″ at 1.5° and 21″ at 2.66°, and that is small against the 1 arcmin scale of IFN.

### What the photons allow

A bin2 pixel covers 14.59 arcsec². With 4.6 × 10⁷ e⁻/s at magnitude 0 (good to ±30%, see the photometric precision row), a sky of 20.5 mag/arcsec² gives 15,200 e⁻ an hour in a pixel (127 in a 30 s frame), and a sky of 21.0 gives 9,600 (80). IFN at 25, 26, and 27 gives 240, 96, and 38 e⁻ an hour. Read noise (1.85 e⁻) is 4% of the variance of a 30 s frame at 21.0, and the dark current at a sensor temperature of 0 °C (0.03 e⁻/s) is 1% of the sky rate, so the sky photons set the noise. The signal-to-noise ratio of the mean IFN signal in an element of 1 arcmin² (a sky of 20.5; multiply by 1.24 for 21.0, and by 5 for an element of 5 arcmin²) is:

| Exposure | IFN 25 | IFN 26 | IFN 27 |
|---|---|---|---|
| 1 h | 30 | 12 | 4.8 |
| 10 h | 95 | 38 | 15 |
| 60 h (about one year, see the next section) | 230 | 93 | 37 |

Photon noise alone would therefore show IFN at 26 within an hour. The real limit is a systematic error: a 5σ detection in an element of 1 arcmin² needs the stack to hold its flat field, sky gradients, and halos to 0.13 to 0.20% of the sky at IFN 26, and to 0.32 to 0.50% at IFN 25 (D). Two outside anchors show that stacks of this kind work. The ASAS-SN survey stacked a median of 58 h a field with 14 cm lenses (R) to a surface-brightness limit of 26.1 mag/arcsec² in g (S, https://arxiv.org/abs/2506.14873). An amateur imaged the Polaris Flare through a 50 mm f/1.8 lens, which collects less than a third of the light of this aperture, with 56 frames of 600 s (9 h 20 min) on a color DSLR (S, https://app.astrobin.com/i/1wmdmc, from the page summary).

### Time on the sky

At a latitude of 60° the Sun is below −18° for 2,175 h a year, on 243 nights. The Moon is also below −3° for 1,144 h of that, and it is below −3° or under 25% lit for 1,206 h (D, with the repository's solar ephemeris and the built-in lunar ephemeris of astropy). Clear weather is the largest unknown. At 30% clear nights (U) the usable time is about 360 h a year, which the default cadence (a 30 s frame every 180 s, 20 frames an hour) turns into about 7,200 frames and 60 h of exposure. A doubled survey cadence doubles the stack and takes the time from fast mode.

### What decides the depth

A pattern that is fixed to the camera turns against the sky at the sidereal rate. A stack of frames that are rotated to the sky keeps the structure that is fixed on the sky and averages the camera-fixed structure over the roll angles of its frames. A camera-fixed pattern with m cycles around the pole keeps the fraction |mean(exp(i m ψ))| of its amplitude, where ψ is the roll of each frame (the local sidereal time times 15° an hour). A mount that tracks the sky cannot do this, because it holds camera-fixed and sky-fixed structure still against each other, and every flat-field error stays in its stack. The table gives the share that survives (D, at a latitude of 60°, with the Sun below −18°, and with the Moon down or under 25% lit except in the first row):

| Frames | m = 1 | m = 2 | m = 3 | m = 4 | m = 5 | m = 6 |
|---|---|---|---|---|---|---|
| One night near the solstice (12.7 h, any Moon) | 0.60 | 0.05 | 0.19 | 0.05 | 0.11 | 0.05 |
| A year, every usable hour | 0.38 | 0.15 | 0.04 | 0.02 | 0.03 | 0.02 |
| A year of random clear nights (30%), 90th percentile of 300 draws | 0.43 | 0.20 | 0.08 | 0.04 | 0.05 | 0.04 |
| A year, weights equalized over 24 sidereal hours | 0.20 | 0.18 | 0.15 | 0.09 | 0.03 | 0.04 |

- **No help at m = 0.** A pattern that does not vary around the pole never averages. The vignetting of the lens is one, because the aligned optical axis points at the pole. The stack cannot tell the radial flat from the azimuthal mean of the IFN about the pole, so it loses that mean and keeps the structure around the pole.
- **A gap in the roll angles.** The Sun stays above −18° on every night that shows local sidereal times of 16 to 20 h (summer nights), so a sixth of the circle never appears. That sets a floor of 0.19 for m = 1 and 0.17 for m = 2. Equalizing the weights lowers m = 1 from 0.38 to 0.20, costs 35% of the effective exposure, and raises m = 3 and 4. A fit of a plane (and a quadric) to the masked sky of each frame, in sensor coordinates, removes the lowest orders at no cost. It also removes the airglow gradient: with the long side horizontal, an airglow layer at 90 km, and an extinction of 0.2 per airmass, the van Rhijn factor changes the airglow by 3% across the short side of the frame at an altitude of 60°, and extinction takes 0.7% of that back (D). The fit removes the field-wide gradient of the IFN as well, so structure larger than about 1° does not survive.
- **Orders 3 to 6 average down.** They keep 0.02 to 0.08 of their amplitude in a year's stack, so dust shadows, pixel response, and dark structure of 1% leave up to 0.08%, under the 0.13 to 0.50% that the detection needs.
- **The halo of Polaris does not average.** Polaris gives 7.2 × 10⁶ e⁻/s. If 10⁻⁷ of that light falls on a pixel 100 px (6′) from the star, it adds 0.7 e⁻/s: a fifth to a quarter of the sky level and 27 times IFN at 26. The halo is centered on a star that is fixed on the sky, so the rotation does not average it. Nobody has measured the wings of this lens (U), and a model of the halo is a prerequisite for any structure near Polaris.
- **Frame selection.** Moonlight, twilight, aurora, and thin cirrus add structure that follows neither the sky nor the camera. Thin cirrus is the worst case, because it scatters light like IFN. The survey record carries the cloud fraction, the transparency, the sky level, and the moon and twilight flags, so the stack takes only frames with the Sun below −18°, the Moon down or under 25% lit, a cloud fraction under 0.1, a transparency of at least 0.95, and a sky level within 10% of the night's median (all thresholds are assumptions, U).
- **A smooth pedestal.** Unresolved stars and the zodiacal light (ecliptic latitude 66°) add a smooth background that the per-frame fit removes, except for its small-scale structure. The star mask needs stars fainter than the G = 13 limit of the cap catalog, and the detected stars supply it.
- **The flat field: a panel flat gives the tilt, and the night sky does the rest.** `[survey] flat_file` takes a measured flat, and `seeingmon flat make` builds one from panel frames, and `seeingmon flat build` builds one from the night sky. On the real test frames of October 3, `flat make` gave the same flat as the bench scripts (an rms difference of 0.018%, 0.25% at most) and the same tilts and vignetting. The owner can take a panel flat easily at the start, and that flat has no tilt problem, so it is the base. The sky then keeps the flat current without a trip to the camera. The rotation makes the night sky a flat source: the mean of the star-masked frames in the sensor frame holds the flat times the mean sky, plus the ground-fixed sky gradient (which is also fixed to the sensor), plus the azimuthal mean about the pole of the structure that is fixed on the sky. A simulation (frames binned 4 × 4, a radial vignetting of 30%, a tilt of 1.5%, 14 dust shadows of 1.5 to 4%, a ground-fixed gradient of 4.5%, 1.5% rms of rotating sky structure, and the roll angles of a year at 60° N) gives (D):
  - One clear night of frames (240 at the default cadence) recovers the radial profile of the vignetting to 0.15% rms, and the shadows and the pixel pattern to 0.2% rms. The system keeps every tenth long frame as FITS, which is 24 frames a night, and those alone give 0.14% and 0.3% (0.16% and 0.24% after two nights). Photon noise is not the limit at 240 frames (0.18% per 15 arcsec pixel). The floor is the ring-like structure that the sky leaves around the pole, and many more nights lower it only to about 0.16%.
  - The tilt is lost. The sky cannot tell a tilt of the flat from its own gradient, and putting the sky's gradient into the flat would cost about 1% rms. The recipe (the radial profile times the fine part, with no plane) leaves the tilt out, so the flat still errs by about 1% rms (3 to 4% at most) near the frame edges, where a flat panel or the stars would give the tilt.
  - With no flat, the simulated error is 7.9% rms and up to 37% at the corners, so the sky flat removes 85% of the error. The real lens vignettes much less (see the measured illumination under Optics: 9.9% in the corners), so a unit flat errs by 2.5% rms (0.027 mag) over the real frame.
  - Polaris needs a large mask, because its halo makes a bright ring at the radius of its orbit in the average (the wings of the lens are unmeasured, U). A twilight flat has the same ambiguity with a larger gradient, so it gains nothing.
  The same flat would lower the 1% systematic floor of the survey photometry. The simulation assumes perfect star masks, a perfect dark subtraction, and an axisymmetric vignetting about the middle of the frame. A panel flat and a sky flat can differ in one more way: the night sky carries near-infrared airglow, and a back-illuminated sensor can show fringes in it, which a white panel does not excite (U, nobody has looked at this sensor). With a panel flat as the base, the sky average divided by it shows any new dust, a change of the vignetting, or fringes.

### What the stack takes from the system

One step per frame on a synthetic 4144 × 2822 frame takes 0.3 s on the loaded dev machine (D, measured): a 4 × 4 binning with a star mask over 3% of the pixels, a plane fit, and a bilinear rotation onto the sky grid of 1036 × 705 px (15 arcsec per pixel). The survey analysis of the same frame takes 2.1 to 3.0 s there, so the step adds about a tenth. IFN structures are minutes of arc wide, so the 15 arcsec pixel loses nothing. The accumulator is a sum and a weight of 2.9 MB each. Twelve roll bins of two sidereal hours, one set for each year, take 70 MB, against about 0.85 GB a year of other growth. A separate stack at full resolution (a sum and a weight of 47 MB each for a year) would reach point sources near G = 21.8 at 5σ (a 2 px aperture, 60 h, D) and would serve as the reference image for difference imaging, which the architecture says needs the pixels.

### Open items

- **Build it or not.** The scope is a masked binning, a plane fit, a rotation, and accumulators by roll bin and year in the survey worker, a command that combines them and writes FITS with a WCS and a weight map, and the frame-selection thresholds. It needs no hardware. Phase 2 does not build it, and it is your decision.
- **Cadence.** A deep season could use more long frames, at the cost of fast-mode time.
- **A filter.** An infrared-cut filter would remove the near-infrared airglow that the unfiltered band includes, and it would cost light in the seeing and photometry. The survey's sky level and zero-point color term show how much of the sky is near-infrared, so decide after the first nights.
- **What the first nights must measure for this goal.** The camera-band sky level, the wings of Polaris, the vignetting and the dust shadows (a night-sky flat from the first clear night), and the stability of the dark structure.

Sources for this section: IRSA dust service https://irsa.ipac.caltech.edu/applications/DUST/, Witt et al. 2008 https://arxiv.org/abs/0802.0674, Heithausen and Thaddeus 1990 https://www.osti.gov/biblio/6806251, Herschel Polaris flare https://arxiv.org/abs/1005.2746, far-infrared opacity in the Polaris Flare https://arxiv.org/abs/astro-ph/0106507, ASAS-SN low surface brightness survey https://arxiv.org/abs/2506.14873, and the AstroBin image https://app.astrobin.com/i/1wmdmc. The calculations ran in scripts outside the repository, from the constants of this file.

## Reference instruments, dew heater, and GPIO

- **Owner's setup (not independently verified).** A fixed SQM-LE points 45 degrees up to the north through a plastic dome, and a handheld SQM-L is also available. The dome transmission is unknown, so it becomes a fitted offset, and handheld readings taken outside the dome can calibrate it.
- **SQM-LE.** Unihedron's product page lists an Ethernet interface, an infrared-blocking filter that limits the response to the visual band, a reported sensor temperature, and a sampling time of 1 to 80 s (V, https://www.unihedron.com/projects/sqm-le/). The page gives no field of view or accuracy, and the manual is a scanned PDF that I could not read, so the field of view and the reading protocol are open items. The SQM band differs from V by up to 0.25 mag depending on the sky spectrum (see the solver and catalog section).
- **GPIO on the Pi 5.** The Pi 5 routes the header pins through the RP1 chip, so `RPi.GPIO`, which reads hardware registers through `/dev/mem`, does not work. Libraries that use the kernel's `/dev/gpiochip` interface (`libgpiod`, `gpiozero` with `lgpio`) work on every Pi model (V, https://pip-assets.raspberrypi.com/categories/685-whitepapers-app-notes/documents/RP-006553-WP/A-history-of-GPIO-usage-on-Raspberry-Pi-devices-and-current-best-practices, and https://forums.raspberrypi.com/viewtopic.php?t=361834).
- **Heater plumes.** A heater near the objective can create convection and add image motion, so the design logs the heater duty on every seeing window and keeps the heat on the dew shield or lens cell at the lowest duty that holds the optics a small margin above the dew point (design reasoning, not a measurement).

## Owner's recordings (format only)

The newest recording folder holds two SharpCap 4.1 captures from the ASI294MM, each a SER file with a settings sidecar. One lasts 60 s (5,874 frames, 430 MiB) and the other 300 s (29,378 frames, 2.1 GiB). The settings in both sidecars are the 11 MP read mode with binning 1 (the SDK's bin2), a 320 × 240 ROI, MONO8, gain 100, a 10 ms exposure, high-speed mode off, Turbo USB 72 (auto), the frame-rate limit at maximum, and a sensor temperature of 18.3 and 17.6 °C. SharpCap measured 97.86 fps.

- **Layout.** The file sizes match the SER layout exactly: a 178-byte header, 76,800 bytes per frame, and an 8-byte timestamp per frame (the PC clock, according to SharpCap).
- **Sidecar.** It uses decimal commas (`10,0000ms`, `97,8567fps`), UTC stamps with a trailing Z, Julian dates, and a local time-zone offset. It also holds the camera serial number, which must stay out of the repository.
- **Frame rate.** 97.86 fps matches the exposure-limited 100 fps that the line-time model predicts for a 240-row bin2 ROI (about 6.5 ms of readout against a 10 ms exposure).
- **Difference from the plan.** The recordings are bin2, 10 ms, and 8-bit, while the planned fast mode is bin1, 2 ms, and a 16-bit container. They therefore show the in-focus bin2 centroid gain swing, the 10 ms exposure bias, and the 8-bit quantization (if RAW8 keeps the top 8 bits of the 14-bit value, each level is about 80 e⁻ at this gain).

## Polaris in a bright sky

This section was added on October 5, 2026, for the visibility design ([visibility.md](visibility.md)), and revised the same day, when the detection moved from the fast path's centroid aperture to a matched filter, and on October 6, when the filters learned the size of the star's image. It estimates how bright a sky still shows Polaris in one fast frame, from the simulator's own photon budget, and it sets the provisional search limit. `seeingmon.drivers.sim.detection` computes the tables, and `tests/drivers/sim/test_detection.py` reproduces them.

### Sources

| Fact | Value | Tag | Source |
|---|---|---|---|
| Measured daylight sky | Nickel and Calderwood measured the V-band sky next to bright stars with a 250 mm telescope at a site in central Europe at 200 m, on 17 cloudless days (July to September 2020 and February to April 2021), with the Sun 10° to 52° high. Their Figure 2 plots the sky against the angle from the Sun. The 8 points from 66° to 96° from the Sun read 3.21, 3.59, 3.68, 4.11, 4.24, 4.30, 4.31, and 4.67 mag/arcsec² (read from the figure to about 0.03 mag), with a median of 4.17. All points span 1.8 to 4.8 mag/arcsec², brighter toward the Sun. The paper names the Sun's elevation and the water vapor as further factors, and its Figure 3 shows the sky brightening with the extinction. | V | https://arxiv.org/abs/2112.12673 (JAAVSO 49, 269, 2021) |
| Daylight model | Schaefer gives the daylight sky as `B = 11700 f(ρ) 10^(−0.4 k X(Z☉)) (1 − 10^(−0.4 k X(Z)))` nL, with the scattering function `f(ρ) = 10^5.36 (1.06 + cos² ρ) + 10^(6.15 − ρ/40) + 6.2 × 10^7 ρ^−2` (Eq. 14, ρ in degrees from the Sun), the extinction coefficient `k`, and the airmass `X` of the Sun's zenith angle and of the sky direction's. 1 nL is 26.33 mag/arcsec². He estimates the brightness of a cloudless sky to about 20% (0.2 mag), and calls 5 × 10⁸ nL (4.6 mag/arcsec²) a typical daytime sky. The scan of his Eq. 15 is hard to read, so the readable form above comes from a program that implements it. | V, S | https://www.uai.it/pianeti/wp-content/uploads/2021/03/ppr_Sch93-1.pdf (Vistas in Astronomy 36, 311, 1993), https://github.com/ad5oo/limiting-magnitude (the ClearDarkSky calculator) |

### The daylight sky in the simulator

Above a Sun elevation of +10°, the simulator's sky near the pole is the measured median, 4.2 mag/arcsec² (`DAYLIGHT_SKY_MAG_ARCSEC2` in `src/seeingmon/drivers/sim/sky.py`). The value holds for every higher Sun. The sunlight that the air scatters adds to the dark sky of the site as a flux, so the daylight sky does not depend on the dark-sky setting, and a darker site keeps its darker night. V is the standard band closest to the camera's response (unfiltered, 400 to 900 nm, effective 600 nm) for which a published daylight value near the pole's angle from the Sun exists, and the model uses it as the camera-band value. Up to 0°, the twilight table is unchanged (6.0 mag/arcsec² at 0°, not cited). From 0° to +10°, the sky brightens along a straight line to the daylight value. The +10° point moved from 4.0 to 4.2 mag/arcsec², which is within the measured scatter. The sky at +3° is 5.46 mag/arcsec².

How the geometry enters (D, from Schaefer's model):

- **The pole's angle from the Sun** is 90° − δ☉: 66.6° at the June solstice, 90° at the equinoxes, and 113.4° at the December solstice, at every site and every hour. It enters through `f(ρ)`. The sky 66.6° from the Sun is 0.24 mag brighter than at 90°, and at 113.4° it is 0.11 mag brighter. The measured points cover 66° to 96°, the summer half of the year. No point covers 96° to 113°.
- **The latitude φ** puts the pole at a zenith angle of 90° − φ (35° and an airmass of 1.22 at the synthetic site at 55° N). It enters through `1 − 10^(−0.4 k X(Z))`: a lower pole looks through more air, which scatters more light. Between 45° N and 65° N, this changes the sky by about 0.24 mag, and by at most 0.14 mag from its value at the synthetic site. The latitude also ties the Sun's elevation to the date: at 55° N, the Sun climbs above 35° only between the spring and the autumn equinox, when the pole is less than 90° from the Sun. A high Sun therefore comes with a slightly brighter sky near the pole.
- **The Sun's elevation** enters through `10^(−0.4 k X(Z☉))`, the share of sunlight that reaches the air above the camera. At the pole of the synthetic site, 90° from the Sun, Schaefer's model gives 5.5, 5.0, 4.8, and 4.6 mag/arcsec² for a Sun at +10°, +20°, +30°, and +58° with k = 0.2, and 5.7, 4.9, 4.6, and 4.3 with k = 0.3. For a Sun at +30°, Schaefer's values 66.6° to 90° from the Sun are 0.2 to 0.6 mag fainter than the measured median.

What the model leaves out:

- The dependence on the Sun's elevation above +10°. Schaefer's model darkens the sky near the pole by 0.9 to 1.3 mag from a high Sun to a Sun at +10°. The flat model is therefore too bright for a low Sun, and it makes the estimate below pessimistic between +10° and about +20°.
- The date and the haze. The measured points scatter by 1.5 mag (3.2 to 4.7 mag/arcsec²) for these reasons.
- The color. The daylight sky is bluer than Polaris, and the camera band reaches 900 nm, so the camera sees the sky fainter, relative to Polaris, than V does. The model uses the V value unchanged, which errs toward a bright sky.
- The site's height, snow on the ground, and clouds. In the simulator, clouds dim the stars and leave the sky as it is.
- A measured sky between 0° and +10°. The centroid aperture's crossing of 10 below (+8.9°) falls on the straight line there. The matched filters have no crossing.
- The size of the real image. The simulator's image is as sharp as the 50 mm aperture allows. The owner's recordings show an image 13 times larger in area, and the focus changes it. "The size of the image" below gives the SNR against it.

### The SNR of Polaris in one fast frame

Whether Polaris shows in a frame depends on the star, the sky, and the image: the star's electrons `F` in the frame, the variance `v` of one pixel of sky, and the area `A = 1 / ΣP²` that the image spreads over, where `P` are the shares of the star in its pixels. When the sky dominates the noise, no weighting of the pixels does better than `F / sqrt(v A)`. The detection that comes closest weights each pixel by the image of the star, a matched filter. An aperture also sums the noise of every empty pixel inside it, and a filter narrower than the image misses light, so what they lose is a property of the method, not of the sky.

Inputs (D): the reference profile `asi294mm-gs250` in bin1, normal readout, gain 0, through the simulator's photon budget. A magnitude-0 star gives 4.6 × 10⁷ e⁻/s, so Polaris (V = 2.02) gives 7.16 × 10⁶ e⁻/s, and the simulator applies no extinction. A pixel covers 3.648 arcsec², the read noise is 2.65 e⁻, the gain is 3.5 e⁻/ADU, and the ADC clips at 14,332 e⁻ (the full well). The dark current is 0.18 e⁻/s at 19 °C. The dark sky is 20.5 mag/arcsec².

- **Exposure.** At most 2 ms (`[scheduler.fast] exposure_us`), shortened so that the sky and the dark sit at no more than 0.3 of the full well (`[scheduler.fast] target_background_fraction`), and never below 32 µs (the profile's shortest exposure). The background therefore stays at 4,300 e⁻ in any daylight sky, and the star's electrons fall in proportion to the sky's brightness, so in daylight the SNR falls by a factor of 2.5 for each magnitude of a brighter sky.
- **The image.** The simulator's image of Polaris at an `r0` of 10 cm at the zenith, seen 35° from the zenith. At 600 nm, D/r0 is about 0.45, so the image is the Airy pattern of the 50 mm aperture (1.33 px FWHM in bin1) with a weak halo. Averaged over the star's position within a pixel, it has `A = 5.9 px²` (21.5 arcsec²). The estimate builds this image from the simulator's Gaussian-mixture model of diffraction and seeing. The simulated frames use wave optics, whose image is about 6% larger by the same measure (6.2 px² at 1.2 ms), and the first filter below reaches about 6.3 to 6.4 px² on it against 5.96 on the mixture, so a simulated frame gives 3 to 4% less matched SNR than the estimate in a sky that dominates the noise. The real image is wider: see "The size of the image" below.
- **The best weighting.** The highest SNR that any linear sum of the pixels reaches, with the weights `P / (v + F P)`: `sqrt(Σ (F P)² / (v + F P))`, where `v` is the variance of one pixel: the sky and dark electrons plus the read noise squared plus `e_per_adu² / 12`. It is the physical limit for this star, sky, and image.
- **The matched filters.** The filters of the fast path (`seeingmon.fastpath.matched`): Gaussians of 1, 2, and 4 Airy FWHM (1.33, 2.67, and 5.33 px in bin1, `[fastpath] matched_fwhm_airy_widths`), integrated over each pixel, each on a grid of positions a quarter of a pixel apart. The SNR of a filter is `S / sqrt(v Σw² + Σw² (I − b))`, where `S = Σw (I − b)` and the second term is the star's own photon noise, which the pixels measure. A search frame keeps the filter with the highest SNR, and the missing-star test of measure uses the first. The first filter fits the simulator's image: filters of 1.0 to 1.2 Airy FWHM give the highest daylight SNR, within 0.4% of each other. The estimate averages over 64 positions of the star within a pixel, each 1/16 px from the grid.
- **The centroid aperture.** The fast path's soft-edged aperture, whose centroid gives the seeing: 16.0 px across (12 Airy FWHM), 201 px² of area, holding 97.0% of the star. Its SNR is `F f / sqrt(F f + A v)` for the share `f` inside it and the area `A`. A window reports its median as `star_snr`, and step 5 of the visibility lane measures the bias of the seeing against it. It no longer decides the detection.
- **Scintillation.** The simulator's rms at the pole is 0.46 at 1.2 ms and 0.42 at 2 ms, so the median frame carries 0.91 to 0.92 of the mean flux. A search counts a burst by the median SNR of its frames, so the median frame decides.
- **Left out.** The noise of the background level, which the kernel takes from the trimmed mean of 496 pixels of the ROI border (its central 68%, whose variance is 1.10 times that of a plain mean). It adds `1.1 A_w / 496` to the variance of a filter of effective area `A_w = 1 / Σw²`: 1.1% for the first filter and 14% for the widest, whose SNR therefore reads up to 7% high in a sky that dominates the noise. It adds 45% to the pixel-noise term of the centroid aperture, which lowers the aperture's true SNR in a bright sky to about 0.83 of the values here.

**The physical limit.** In the model's daylight sky, the median frame holds `F` = 7,980 e⁻ of Polaris at 1.23 ms, and a pixel of sky has a variance `v` of 4,310 e⁻². Counting only the sky, the best SNR is `F / sqrt(v · 5.9 px²)` = 50. The star's own photons fall on the same few pixels, and the best weighting reaches 41.5. The fast path's filters reach 41.3, with the first. In a dark sky, the star's photons dominate: there the best weighting gives 110.4, which is 0.96 times the root of the star's electrons, the widest filter, which takes in the halo of the image, gives 109.5, and the first filter 95.2.

| Sun (°) | Sky (mag/arcsec²) | Exposure (ms) | Background (share of full well) | Polaris (e⁻) | SNR, best weighting, median frame | SNR, fast path's matched filters, median frame | SNR, centroid aperture, median frame |
|---|---|---|---|---|---|---|---|
| +10 to +60 | 4.20 | 1.23 | 0.30 | 8,780 | 41.5 | 41.3 | 8.3 |
| +9 | 4.38 | 1.45 | 0.30 | 10,360 | 47.9 | 47.6 | 9.8 |
| +8 | 4.56 | 1.71 | 0.30 | 12,230 | 55.1 | 54.7 | 11.6 |
| +7 | 4.74 | 2.00 | 0.30 | 14,310 | 62.9 | 62.4 | 13.7 |
| +6 | 4.92 | 2.00 | 0.25 | 14,310 | 66.0 | 65.3 | 14.9 |
| +5 | 5.10 | 2.00 | 0.21 | 14,310 | 68.9 | 68.1 | 16.1 |
| +4 | 5.28 | 2.00 | 0.18 | 14,310 | 71.8 | 70.8 | 17.5 |
| +3 | 5.46 | 2.00 | 0.15 | 14,310 | 74.7 | 73.3 | 18.9 |
| +2 | 5.64 | 2.00 | 0.13 | 14,310 | 77.3 | 75.7 | 20.5 |
| +1 | 5.82 | 2.00 | 0.11 | 14,310 | 79.9 | 77.9 | 22.2 |
| 0 | 6.00 | 2.00 | 0.09 | 14,310 | 82.4 | 79.9 | 24.0 |
| −2 | 8.00 | 2.00 | 0.01 | 14,310 | 100.2 | 99.6 | 53.6 |
| −4 | 10.00 | 2.00 | 0.00 | 14,310 | 107.0 | 106.7 | 88.0 |
| −6 | 12.00 | 2.00 | 0.00 | 14,310 | 109.5 | 109.0 | 102.8 |
| −9 | 15.00 | 2.00 | 0.00 | 14,310 | 110.3 | 109.4 | 106.3 |
| −12 | 18.00 | 2.00 | 0.00 | 14,310 | 110.4 | 109.5 | 106.6 |
| −18 | 20.50 | 2.00 | 0.00 | 14,310 | 110.4 | 109.5 | 106.6 |

**No crossing.** The matched SNR of the median frame never falls below 41.3 between −18° and +90°, so it does not cross 10, and in the model Polaris stays detectable frame by frame in full daylight. It falls to 10 only in a sky of 2.53 mag/arcsec² (at 0.26 ms), which is 1.67 mag brighter than the model's daylight and 0.67 mag brighter than the brightest sky that Nickel and Calderwood measured near the pole's angle from the Sun. At that brightest sky (3.2 mag/arcsec², 0.49 ms) it gives 17.9, and at the darkest (4.7 mag/arcsec², 1.94 ms) 60.8. These numbers hold for the simulator's image, which is as sharp as the 50 mm aperture allows. The next paragraph gives them for a wider image.

**The size of the image.** In a sky that dominates the noise, the SNR falls as `1 / sqrt(A)`, so the size of the real image matters as much as the sky. The table widens the simulator's image by a Gaussian blur, as defocus or the color of the optics would, and gives the median daylight frame of the model (the Sun above +10°) and the sky in which the fast path's SNR falls to 10.

| Blur (FWHM, arcsec) | Image area `1 / ΣP²` (arcsec²) | SNR, best weighting | SNR, fast path's matched filters | SNR, first filter alone | SNR, centroid aperture | Sky where the fast path's SNR is 10 (mag/arcsec²) |
|---|---|---|---|---|---|---|
| 0 | 21 | 41.5 | 41.3 | 41.3 | 8.3 | 2.53 |
| 3 | 48 | 30.6 | 30.1 | 28.3 | 8.3 | 2.97 |
| 5 | 92 | 23.1 | 22.8 | 18.2 | 8.3 | 3.29 |
| 7 | 155 | 18.1 | 17.5 | 11.9 | 8.3 | 3.59 |
| 10 | 284 | 13.6 | 13.5 | 6.9 | 8.2 | 3.87 |
| 13 | 453 | 10.8 | 10.4 | 4.4 | 8.0 | 4.16 |

The fast path's three filters stay within 5% of the best weighting up to a blur of 13″. A filter of the Airy FWHM alone, as the first version of the matched filter had, keeps half of the SNR at 10″ and falls below the threshold. No row crosses 10 between −18° and +90° in the model's sky, but a wider image puts the limit in a fainter sky: at 10″ it lies at 3.87 mag/arcsec², within the measured daylight skies. A filter that does not fit the image costs SNR in the same way in the missing-star test, which uses the first filter alone, so a frame there also counts the star as found when the centroid aperture reaches `[fastpath] min_star_snr` (6). A defocused bright star, as in the rapid focus mode, stays found that way.

**The image of the owner's recordings.** In the owner's two 10 ms recordings (bin2, 3.82″ per pixel, about 600 frames of each, the sky from a ring 12 to 24 px from the star), `1 / ΣP²` within 7 px of the star is 19 to 20 px² in bin2, which is 276 to 286 arcsec²: 13 times the simulator's 21.5 arcsec², and the row of the table with a blur of 10″ (D). The image has a narrow core and a broad skirt. In a sky that dominates the noise, the Gaussian filter that fits it best has a FWHM of 7.6″ (3 Airy FWHM) and reaches 0.96 of the best weighting, the fast path's three filters reach 0.95, and the first filter alone 0.86 to 0.87. With that image, Polaris gives about 13.6 in the model's daylight with the best weighting and about 13 with the fast path's filters, and the fast path's SNR falls to 10 in a sky of about 3.9 mag/arcsec². Three of the eight daylight skies that Nickel and Calderwood measured near the pole are brighter than that (3.21, 3.59, and 3.68 mag/arcsec²), so with this image Polaris hides in a bright daylight sky and shows in a median one. Whether the width comes from the focus, from the color of the optics, or from the seeing of that night is for phase 3 to measure. A sharper focus moves the limit toward that of the simulator's image.

**What the centroid aperture loses.** The fast path's aperture holds 97% of the star but sums the variance of 201 pixels, 34 times the 5.9 px² of the image. In daylight it gives 8.3, a fifth of the matched filters' 41.3, and its SNR falls to 10 at +8.9°, where the sky is 4.40 mag/arcsec². That was the crossing of the first version of this estimate, and the search limit of +12° came from it: it was a limit of the method, not of the sky. In a dark sky, where the star's photons dominate, the aperture gives 106.6, against 95.2 for the first filter and 109.5 for the widest. The aperture still gives the centroids of the seeing windows, so a window in a bright sky has noisy centroids.

**The search limit.** Without a crossing, the default of `scheduler.search.max_sun_elevation_deg` is **90°**, which means no limit: the search runs at any height of the Sun, and the probe bursts run only when a person sets a lower limit. A limit would save little, because a burst costs about 3% of the camera's time, and with an image as wide as that of the recordings, the limit depends on the day's sky, not on the Sun. The real sky decides in phase 3, through the measured gate of the brightness frame and the detections themselves.

**False detections.** A burst detects Polaris when the median of its 50 frames reaches an SNR of 10, and each frame reports the highest matched SNR of the three filters within 20 px of the prediction (`[scheduler.search] radius_px`). On a frame without a star, that is the highest of 68,544 positions of noise: for each filter, the reported position lies within 20 px, on one of 16 positions in one of at most 1,428 pixels (the 1,257 pixels of the circle and the grid of the filter around them). The search looks for the brightest place of the filtered image in the circle widened by the half width of the filter, and it reports no star when that place lies beyond 20 px, so a bright star just outside the circle never shows at its edge. Each position is a unit Gaussian under the sky noise, and the star's photon term only lowers the SNR, so the union bound `68,544 Q(t)` caps the chance that a frame reaches `t`. At 10, that is 5.2 × 10⁻¹⁹ per frame. A burst needs at least 25 of its 50 independent frames above 10, which caps its chance at `C(50, 25) p²⁵`, about 10⁻⁴⁴³, and measure needs two such bursts in a row. In 2,000 simulated starless frames of the daylight sky, the best filter of a frame had a median of 3.43 and stayed below 5.7, and in 10% of the frames every filter peaked beyond the radius, so the frame counted as 0, as it does in the scheduler. The peak reached 4.5 in 2.6% and 5 in 0.5% of the frames, below the bound's 23% and 2.0%, so its tail is no heavier than a Gaussian one, and the median of every burst of 50 frames stayed below 3.7. A starless burst therefore sits far below 10, and the missing-star threshold of the kernel (`[fastpath] min_star_snr`, 6) also stays above the noise peaks. A hot pixel cannot imitate the star at the fast exposures: a pixel with a dark current of 1,000 e⁻/s collects 2 e⁻ in 2 ms. The widest filter covers about 65 px², so a step in the sky of 1.2 times the noise of a pixel across it, about 2% of the daylight sky, would read as an SNR of 10. The simulator's sky is flat, so a gradient across the ROI, a dust shadow, or a cloud edge is for phase 3 to check. `tests/fastpath/test_matched.py` checks the tail and the bound.

**The cost.** On the dev desktop, timed side by side with the single filter of the first version (the same machine and frames), the filter of the missing-star test adds about 55% to the kernel of a 128 × 128 frame, about 12 µs on top of the 22 µs of `python -m seeingmon.fastpath.benchmark` without it, and a frame of a search burst with the three filters takes about 3.5 times as long as with one, about 0.2 ms. The search filters only the square around its circle of 20 px, widened by the half width of each filter, never the whole frame. The Pi 4 ran the kernel without the filter in 132 µs, 6.9 times the dev desktop's 19 µs ([performance.md](performance.md), "Results on a Raspberry Pi 4"). On the Pi, the filter therefore adds about 70 µs to each fast frame, which is 0.7% of a core in bin1 at 98 frames per second and about 2% in bin2 64 × 64 at 360 frames per second, and a burst frame costs about 1.3 ms: 11% of a core while a burst of 0.6 s runs, and 0.4% over its interval of 15 s. The `kernel` case of the performance harness times both, as `<mode>.kernel` and `<mode>.search`, so the next run on the Pi measures them.

**A simulated dusk.** The slow end-to-end test `TestPolarisAtDusk` (`tests/services/e2e/test_night.py`) runs `core` with the production analyzers on the simulator: the full reference sensor, the real Polaris, at most 2 ms, and the adaptive exposure, on the evening of April 20, 2026, from a Sun at +11.5°, in the model's daylight sky. The first two bursts find Polaris, at **+11.4°**, with a median matched SNR of 38.6, 6.5% below the 41.3 of the estimate. Most of the difference has known causes: the wave-optics image of the simulated frames costs 3 to 4% (see "The image" above), the short exposure below about 1.2%, and the noise of the background level about 0.5%. The first window takes 1.20 ms against 1.23 ms (the loop counts the camera's offset as sky, which makes the exposure 2.4% short), its background sits at 0.300 of saturation, and its `star_snr` (the centroid aperture) is 8.2 against 8.3. Its `r0` reads 3.4 cm against the injected 10 cm: the centroids come from the wide aperture, and the noise model of the centroid leaves out the sky, so the seeing of a daylight window is biased until step 5 of the visibility lane corrects it. With the first version of this estimate, the same run found Polaris at +8.69°, read an `r0` of 4.8 cm there, and missed the daylight.

### The survey at dusk

This subsection was added on October 6, 2026, for step 6 of the visibility lane. It says how bright a sky still gives a survey frame that solves, and what the adaptive long exposure (`[survey.twilight]`, `seeingmon.scheduler.exposure`) gains in the simulator.

**What the photons allow.** The long survey frame is bin2 at gain 120: 3.82 arcsec per pixel, a full well of 14,417 e⁻, and the simulator's zero point of 19.16 (the G magnitude that gives 1 e⁻/s). A frame of 1 s whose sky sits at the target, 0.3 of the full well (4,325 e⁻ in a pixel, 66 e⁻ of noise), sees a sky of 12.98 mag/arcsec², which the simulator reaches with the Sun about 7° down. The optimal estimator, a fit of the star image, detects a star at 5σ down to G = 12.3 there (543 e⁻). Its noise-equivalent area is 2.6 px for the simulator's undersampled image of 0.73 px FWHM, averaged over the position of the star in the pixel. The first frame of 30 s that does not saturate (0.58 of the full well, with the Sun near −10°) reaches G = 15.7. A frame of 1 s therefore sees 3.4 mag less deep, and it must solve on the brightest stars of the field.

**How far the pipeline falls short.** The detector thresholds each pixel of the filtered image at 5σ and needs 2 pixels above it, which an undersampled star reaches only near an optimal SNR of 20, 1.5 mag above the 5σ limit. In a simulated frame of 1 s at 13.0 mag/arcsec² (the cropped profile of 1,200 by 900 px of `tests/survey/test_sim_end_to_end.py`), 70 catalog stars lie above G = 12.3. The search at full resolution finds 17 of them, and the binned search finds 14. The binned search (`[survey.detect] coarse_bin` 2) runs whenever the tracker gives a trail model, and it has no matched filter, so it needs 2 binned pixels above 5σ, and against a bright sky only the core of a sharp star passes. The expected-star model of the cloud fraction expects 13 stars at an aperture SNR of 20, and the binned search finds 10 of them, so this clear frame reads a cloud fraction of 0.23. The same frame at 30 s in a dark sky reads 0. Phase 3, or a later step, decides whether a bright sky takes the full search.

**The 1 ms frame.** In the 1 ms frame of a survey step (bin2, gain 0, 4.05 e⁻ per count), one count of the ADC of sky is 28% of saturation in a long frame of 1 s at gain 120, about the target. The level of the frame holds the black level, about 120 counts with the simulator's offset. The 1 ms frame alone therefore cannot place a long exposure between 1 and 30 s, and it shows only a sky that is far too bright. The scheduler learns the black level from pairs of frames of one sky, and it skips the long exposure while the 1 ms frame shows more than 4 counts of sky beyond what puts a frame of 1 s at the target (`seeingmon.scheduler.exposure`).

**A simulated dusk.** The slow end-to-end test `TestTheSurveyAtDusk` (`tests/services/e2e/test_night.py`) runs `core` with the production survey pipeline on the simulator: the small sensor (1,280 by 960 bin1 pixels), the cycle of 180 s, the tracker as the only solver, and the evening of January 1, 2026, from 16:00 UTC (the Sun at −2.8°). The simulator's twilight brightens by 1 mag per degree of Sun between −12° and 0°.

| | Adaptive long exposure | Fixed 30 s |
|---|---|---|
| Steps that skip the long exposure | 8, from −2.8° to −5.2° | none |
| Long frames with `saturated_sky` | 1, at −5.6° | every frame down to −9.6° |
| First survey frame that solves | **−6.3°** (1 s, 4 stars, `few_stars`) | **−10.0°** (30 s, 18 stars) |
| First solve with 12 stars or more | **−8.5°** (2.9 s, 12 stars) | −10.0° |

The adaptive exposure gives the first solve 3.7° earlier, about 30 minutes of a January evening at 55° N, and a solve with 12 stars 1.4° earlier. The long frames after the first sit at 0.21 of saturation, below the target of 0.3: the sky darkens by about 1.4 times in a cycle of 180 s, and each exposure scales from the frame before. No frame with `saturated_sky` reports clouds. The twilight frames of 1 to 3 s read cloud fractions of 0.43 to 0.80 under the clear simulated sky, with the loose cloud settings of the end-to-end tests (an expected SNR of 10, and `min_expected` 4), for the reasons above.

## Calculations

| Item | Inputs and result |
|---|---|
| Polaris separation from the pole | Polaris J2000 declination +89° 15′ 50.8″ and RA 2h 31m 49s. The declination rises by 20.04″ × cos(RA) per year, so late 2026 gives +89° 22.9′ and a separation of 0.618° (37 arcmin). |
| Sky motion | The sky turns at 15.041 arcsec per second of time, and a star at angular distance θ from the pole moves at 15.041 × sin θ. Polaris: 0.162 arcsec/s, so 9.7″ in 60 s, 19.5″ in 120 s, and 97″ in 600 s. A star 3.3° from the pole (the far edge of the field) moves 0.86 arcsec/s, a 4.3″ trail in 5 s. |
| Polaris photon count | Polaris is V = 2.02. A flat spectrum over 400 to 900 nm, the quantum-efficiency integral of 0.52, and the 19.6 cm² aperture give 8.7 × 10³ e⁻/ms (the budget is good to ±30% and ignores optics transmission), so 87,000 e⁻ in 10 ms. A hand estimate with a Vega-like spectrum, 75% quantum efficiency, and 85% transmission gave 70,000 to 130,000 e⁻ in 10 ms, which agrees (D). |
| Image motion size | One-axis G-tilt variance 0.170 λ² D⁻¹ᐟ³ r0⁻⁵ᐟ³ (Martin 1987, Eq. 7, verified in the seeing-theory section). For D = 50 mm and r0 (500 nm) of 5, 10, and 15 cm: seeing 2.0, 1.0, and 0.67 arcsec, and one-axis RMS motion 0.85, 0.48, and 0.34 arcsec (0.22, 0.125, and 0.09 pixels in bin2; 0.45, 0.25, and 0.18 pixels in bin1). |
| Centroid pixel phase | A toy, noise-free simulation: box-integrated Gaussian PSF, 11-pixel window, plain centroid. Peak-to-peak centroid bias: 0.065 px at FWHM 0.8 px, 0.018 px at 1.0 px, 0.0002 px at 1.5 px, and under 0.001 px at 2.0 and 2.5 px. The variance of a 0.125 px RMS motion reads 0.93 of the true value at FWHM 0.8 px and 0.98 at 1.0 px. Window truncation, not pixel phase, causes the 0.016 px bias at FWHM 4 px. |
| Fast-frame rates | Frame time is the larger of the exposure and the readout (overhead plus rows times line time). Bin1, 128 rows: 6.5 ms + 128 × 37.6 µs = 11.3 ms, or 88 fps. Bin2, 64 rows: 1.4 ms + 64 × 21.3 µs = 2.8 ms, or 360 fps at short exposures and 100 fps at 10 ms. (These are the vendor figures. The real camera measured 82 fps for bin1 with 128 rows, 103 in high-speed mode, and 417 fps for bin2 with 64 rows at 0.5 ms.) |
| Data sizes | 16-bit frames: 128 × 128 is 32 KB, 64 × 64 is 8 KB, bin2 full frame is 23.4 MB, and bin1 full frame is 93.6 MB. A 40-byte metrics record at 88 fps is 3.5 KB/s, or 304 MB per day. |
| Memory budget (estimate, to be measured in the performance gate) | Peak resident memory in MB: OS 200 (Raspberry Pi OS Lite with idle services), `acquire` 125 (Python and SDK 40, three bin2 frames in the queue 70, fast frames 15), `core` 225 (Python with NumPy, SciPy, and astropy about 180, buffers and queues 45), survey worker 550 (raw frame 23, float32 copy 47, SEP background about 100, imports about 150, detection and photometry arrays 100, resident calibration frames up to 130), `solve-field` 50, `web` 100, tmpfs 130 (three survey frames 70, previews 30, volatile journal 30). Total about 1,380 MB. A 2 GB board has 1.8 to 1.9 GB usable, which leaves about 450 MB for the page cache and spikes. A native bin1 frame (94 MB) makes the worker about 1.1 GB and the total about 1.9 GB, so native frames stay off the Pi. Mitigations: an out-of-memory score that kills the survey worker first, memory-mapped calibration frames, zram instead of swap on the SD card. |
| Storage budget | Rolling tiers: per-frame metrics 2 GB, bursts 2 GB, survey frames 2.4 GB (every tenth frame for 7 days is 1.7 GB, plus one frame per night for 60 nights is 0.7 GB), previews 0.2 GB. That is about 6.6 GB. Growth: results 0.4 GB per year and star lists 0.45 GB per year (phase 2 corrected this from 0.3 GB: a row takes 24 bytes, and a survey step runs every 3 minutes while the sky is dark), so about 0.85 GB per year. A 32 GB card holds the tiers and about 5 years of growth. |
