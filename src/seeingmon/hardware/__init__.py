"""Hardware-facing code: the camera SDK binding, the dew heater, the power cycle, and the SQM-LE.

Every part runs against a fake, so the tests need no hardware. Real-hardware code imports its
platform-specific pieces (`ctypes` libraries, `fcntl`, sysfs) when you use it, never when you
import the package, so the package imports on every platform.

- `seeingmon.hardware.asi`: the vendor SDK behind the `AsiApi` protocol, with a `ctypes`
  implementation, a fake, the call watchdog, and the USB reset.
- `seeingmon.hardware.io`: GPIO lines and environment sensors behind small interfaces.
- `seeingmon.hardware.heater`: the dew-heater controller.
- `seeingmon.hardware.power`: the remote power-cycle hook.
- `seeingmon.hardware.sqm`: the SQM-LE reader of the TCP source, the `[sqm]` configuration, and the
  interface of both readers.
- `seeingmon.hardware.sqm_influx`: the SQM-LE reader of the InfluxDB source.
- `seeingmon.hardware.sqm_factory`: builds the SQM-LE reader that `[sqm] source` names, and reads
  one sample to try the settings.
- `seeingmon.hardware.events`: the event type that these components report.
"""

from __future__ import annotations
