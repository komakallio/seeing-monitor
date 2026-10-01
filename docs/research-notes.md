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

The non-Pro page shows the same two modes, the same gain charts (the labels match), the HCG step at gain 120, and the same frame-rate tables as the Pro, so the mode, gain, noise, and frame-rate numbers above apply as published (D). Assumed from the Pro and unverified for the non-Pro: the SDK bin1 to bin4 mapping, the high-speed mode, the gain range 0 to 570, the `Temperature` control, the binning rules, and the 16-bit scaling. With no buffer, a slow or contended USB link can stall readout, so check `ASIGetDroppedFrames` in video mode (D).

- **Sensor temperature.** `ASI_TEMPERATURE` returns tenths of a degree and is read-only (V). ZWO does not document it for this model. Other uncooled ZWO cameras report it in 0.1 °C steps (S), so check `ASIGetControlCaps` at start-up. The first read after opening returns 0 for about 250 ms. Uncooled bodies run about 4 °C above ambient (S).
- **Dark current** (ZWO chart; camera, mode, and gain not stated; points read off the image, about 5% uncertainty):

| Sensor temperature | 30 °C | 25 °C | 20 °C | 10 °C | 0 °C | −10 °C | −20 °C |
|---|---|---|---|---|---|---|---|
| e⁻/s/pixel | 0.70 | 0.36 | 0.20 | 0.065 | 0.019 | 0.0066 | 0.0022 (printed) |

The doubling temperature is about 5.8 °C between 0 and 30 °C (D). The QHY294M Pro (same sensor) lists 0.002 e⁻/s at −20 °C (S), and Buil measured 0.0010 e⁻/s at −15 °C on a Pro (S). At a sensor temperature of ambient + 4 °C, a 30 s exposure collects about 0.3 e⁻ at −10 °C ambient, 3.1 e⁻ at 10 °C, and 9.7 e⁻ at 20 °C (D). ZWO documents no optical-black or overscan readout, and the SDK returns only the ROI inside 8288 × 5644.
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
| Not specified | Image circle in millimetres, data beyond 8.0 mm, back focus, tube length, transmission, spectral range, glass types, measured performance | U | T1, T2 |

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
| License | MIT (Expat), copyright ZWO Company 2015, in the copy that INDI and Debian vendor. Debian files libasi under non-free because the binaries ship without source. The file inside ZWO's own archive was not checked. | V; archive U | https://raw.githubusercontent.com/indilib/indi-3rdparty/master/libasi/license.txt, https://sources.debian.org/src/libasi/1.27%2B20221218230335-2/debian/copyright/ |
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

At 44 bytes per row, G < 13 and G < 14 take 1.6 and 3.2 MB at 10 degrees, and 3.6 and 7.4 MB at 15 degrees. A frame holds about 320, 700, and 1,490 catalog stars at G < 11, 12, and 13 (I). Use an asynchronous ADQL job (anonymous limit 3 million rows). A synchronous query silently truncated at 16,385 of 35,458 rows (P).

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

## Calculations

| Item | Inputs and result |
|---|---|
| Polaris separation from the pole | Polaris J2000 declination +89° 15′ 50.8″ and RA 2h 31m 49s. The declination rises by 20.04″ × cos(RA) per year, so late 2026 gives +89° 22.9′ and a separation of 0.618° (37 arcmin). |
| Sky motion | The sky turns at 15.041 arcsec per second of time, and a star at angular distance θ from the pole moves at 15.041 × sin θ. Polaris: 0.162 arcsec/s, so 9.7″ in 60 s, 19.5″ in 120 s, and 97″ in 600 s. A star 3.3° from the pole (the far edge of the field) moves 0.86 arcsec/s, a 4.3″ trail in 5 s. |
| Polaris photon count | Polaris is V = 2.02. A flat spectrum over 400 to 900 nm, the quantum-efficiency integral of 0.52, and the 19.6 cm² aperture give 8.7 × 10³ e⁻/ms (the budget is good to ±30% and ignores optics transmission), so 87,000 e⁻ in 10 ms. A hand estimate with a Vega-like spectrum, 75% quantum efficiency, and 85% transmission gave 70,000 to 130,000 e⁻ in 10 ms, which agrees (D). |
| Image motion size | One-axis G-tilt variance 0.170 λ² D⁻¹ᐟ³ r0⁻⁵ᐟ³ (Martin 1987, Eq. 7, verified in the seeing-theory section). For D = 50 mm and r0 (500 nm) of 5, 10, and 15 cm: seeing 2.0, 1.0, and 0.67 arcsec, and one-axis RMS motion 0.85, 0.48, and 0.34 arcsec (0.22, 0.125, and 0.09 pixels in bin2; 0.45, 0.25, and 0.18 pixels in bin1). |
| Centroid pixel phase | A toy, noise-free simulation: box-integrated Gaussian PSF, 11-pixel window, plain centroid. Peak-to-peak centroid bias: 0.065 px at FWHM 0.8 px, 0.018 px at 1.0 px, 0.0002 px at 1.5 px, and under 0.001 px at 2.0 and 2.5 px. The variance of a 0.125 px RMS motion reads 0.93 of the true value at FWHM 0.8 px and 0.98 at 1.0 px. Window truncation, not pixel phase, causes the 0.016 px bias at FWHM 4 px. |
| Fast-frame rates | Frame time is the larger of the exposure and the readout (overhead plus rows times line time). Bin1, 128 rows: 6.5 ms + 128 × 37.6 µs = 11.3 ms, or 88 fps. Bin2, 64 rows: 1.4 ms + 64 × 21.3 µs = 2.8 ms, or 360 fps at short exposures and 100 fps at 10 ms. |
| Data sizes | 16-bit frames: 128 × 128 is 32 KB, 64 × 64 is 8 KB, bin2 full frame is 23.4 MB, and bin1 full frame is 93.6 MB. A 40-byte metrics record at 88 fps is 3.5 KB/s, or 304 MB per day. |
| Memory budget (estimate, to be measured in the performance gate) | Peak resident memory in MB: OS 200 (Raspberry Pi OS Lite with idle services), `acquire` 125 (Python and SDK 40, three bin2 frames in the queue 70, fast frames 15), `core` 225 (Python with NumPy, SciPy, and astropy about 180, buffers and queues 45), survey worker 550 (raw frame 23, float32 copy 47, SEP background about 100, imports about 150, detection and photometry arrays 100, resident calibration frames up to 130), `solve-field` 50, `web` 100, tmpfs 130 (three survey frames 70, previews 30, volatile journal 30). Total about 1,380 MB. A 2 GB board has 1.8 to 1.9 GB usable, which leaves about 450 MB for the page cache and spikes. A native bin1 frame (94 MB) makes the worker about 1.1 GB and the total about 1.9 GB, so native frames stay off the Pi. Mitigations: an out-of-memory score that kills the survey worker first, memory-mapped calibration frames, zram instead of swap on the SD card. |
| Storage budget | Rolling tiers: per-frame metrics 2 GB, bursts 2 GB, survey frames 2.4 GB (every tenth frame for 7 days is 1.7 GB, plus one frame per night for 60 nights is 0.7 GB), previews 0.2 GB. That is about 6.6 GB. Growth: results 0.4 GB per year and star lists 0.3 GB per year, so about 0.7 GB per year. A 32 GB card holds the tiers and about 5 years of growth. |
