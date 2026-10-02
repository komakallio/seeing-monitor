# Runbook

This runbook covers how to install, upgrade, roll back, check, and look after the seeing monitor on a Raspberry Pi. It follows the Deployment section of [architecture.md](architecture.md). The scripts and files that it names live in `deploy/`.

Paths such as `/opt/seeingmon` and `/etc/seeingmon` are examples. The installer has no default for any path, host, or user name: you pass every value. Angle brackets, such as `<pi-host>`, mark a value that you supply.

## What is tested and what is not

The deploy lane could not run anything on a Raspberry Pi (blocker B2), so every step below that needs a real Pi is untested. The table separates what the checks cover from what stays open.

| Piece | What checks it | Untested without a Pi |
|---|---|---|
| `build.sh`, `push.sh`, `install.sh`, `rollback.sh` | `shellcheck` (Windows and Linux x64; Linux arm64 gets `bash -n`), a structural linter (`tools/lint_deploy.py`), and tests that run each script under bash with stub programs. One test makes a real virtual environment and runs real pip against a local wheelhouse. | A run as root on Raspberry Pi OS: `useradd`, `systemctl`, `udevadm`, the real package index, a real SD card |
| Systemd units | The linter parses them and checks the sandbox and the architecture rules. `systemd-analyze verify` printed no warning on a development machine. | Starting them on a Pi: the sandbox and the system call filter with the real libraries, and the memory limits |
| `acquire`, `core`, and `web` processes | Each process sends `READY=1` and `WATCHDOG=1` over the `sd_notify` socket, as the `Type=notify` units expect. Tests read the messages from a stand-in socket. | systemd acting on the messages: the start timeout, the watchdog restart, and the start limit. The installer reports a unit that does not start. |
| udev rule | Syntax (`udevadm verify` passed on a development machine) | A real camera: the group, the mode, and the autosuspend setting |
| USB buffer, journald, chrony fragments | Syntax and rendering | Their effect on a Pi: the `usbfs_memory_mb` write, the volatile journal, time synchronization |
| Polkit rule (`--supervisor-actions`) | Syntax and rendering | That `systemctl reboot` works for the service user |
| `seeingmon heater-off` | Unit tests on fake GPIO lines, including a call that hangs. The `core` unit runs it when it stops. The command exits with 0 when every heater output is off or no heater is configured, and with 1 when an output cannot be switched off, the configuration cannot be read, or the GPIO work passes its 1 s limit. | Real lines: the permission of the service user on `/dev/gpiochip*`, and what a line does after the release (see the heater paragraph of [architecture.md](architecture.md)). A failure writes its reason to the journal, one line for each failed output. Run by hand while `core` runs, the command exits with 1 and reports `the line is busy`. The minus sign in `ExecStopPost=-...` keeps a failure from failing the unit. |
| Recovery ladder, power cycle | The scheduler and the power hook have their own tests | A real stall, a real reboot, and a real power cycle (blocker B4: the route is undecided) |

A section that says "untested" describes the intended behavior, and you confirm it the first time you run it.

## Before you start

You need:

- A Raspberry Pi 4 with 2 GB of RAM, or a Raspberry Pi 5. The design targets the Pi 4.
- Raspberry Pi OS Lite, 64-bit. Debian 13 (Python 3.13) is the target, and Debian 12 (Python 3.11) also works.
- A high-endurance microSD card of 32 GB or more (see [SD card care](#sd-card-care)).
- The ZWO camera on a USB 3 port. A Pi 5 on a 3 A supply or a PoE splitter needs `usb_max_current_enable=1` in `/boot/firmware/config.txt`.
- A development machine with a clone of this repository, `uv`, `ssh`, and `bash`. On Windows, run the scripts in Git Bash.
- An ssh key on the Pi and a login account that can use `sudo`.
- Internet access from the Pi to the Python package index during an install. A Pi without internet needs a folder of wheels (see [Install from a wheelhouse](#install-from-a-wheelhouse)).

## Prepare the Pi

1. Write Raspberry Pi OS Lite (64-bit) to the card. In Raspberry Pi Imager, set a host name, a login user, your ssh public key, and the time zone. Turn password login off.
2. Boot the Pi, log in over ssh, and update it:

   ```bash
   sudo apt-get update
   sudo apt-get full-upgrade
   ```

3. Install the packages that the installer and the services need:

   ```bash
   sudo apt-get install --no-install-recommends python3 python3-venv chrony libusb-1.0-0 libgpiod3
   ```

   On Debian 12, install `libgpiod2` instead of `libgpiod3`. Add `polkitd` if the Pi does not have it and you plan to use `--supervisor-actions`. Add `astrometry.net` for the plate solver of the survey path.

4. Check the Python version. It must be 3.11 or later:

   ```bash
   python3 --version
   ```

5. Turn swap off on the SD card, so that the out-of-memory killer, and not the card, absorbs a memory peak:

   ```bash
   swapon --show
   sudo systemctl disable --now dphys-swapfile      # only if swapon lists a file on the card
   ```

6. Make `/tmp` a tmpfs. The services then write their temporary files to RAM:

   ```bash
   findmnt -n -o FSTYPE /tmp                         # prints tmpfs when it is already one
   sudo systemctl enable tmp.mount                   # otherwise, then reboot
   ```

7. Make the heater default to off at boot, before Linux starts. Add one line to `/boot/firmware/config.txt` for the GPIO pin of the heater output, and use `dh` instead of `dl` for a relay that switches on a low level. This line is untested, and the HAT is undecided (blocker B3):

   ```text
   gpio=<pin>=op,dl
   ```

### Create the data partition

The data directory belongs on its own partition, so that a full data area never fills the root file system. The installer does not partition anything. It warns when the data directory shares the root file system, and it goes on.

The steps depend on how you imaged the card, and the deploy lane did not test them on a Pi. A standard image grows its root partition to fill the card at the first boot. To keep room for a data partition, shrink the root partition on your computer before the first boot, create a second partition in the freed space, and format it:

```bash
sudo mkfs.ext4 -L seeingmon-data /dev/<data-partition>
```

Then mount it at the data directory through `/etc/fstab`. The `nofail` option lets the Pi boot without the partition, and the `core` and `web` units then wait for it and never write to the root file system by mistake:

```text
LABEL=seeingmon-data  /srv/seeingmon-data  ext4  defaults,noatime,nofail  0  2
```

Run `sudo mount -a` and `findmnt /srv/seeingmon-data` to check the mount.

## Prepare your files

Keep these files outside the repository, or under `local/`, which Git ignores. Nothing in them ever goes into a commit.

- **Local configuration.** Copy `config/local.example.toml` to `local/config.toml`. Set at least:
  - `station_id`, and the `[site]` table.
  - `driver = "asi"` in `[services.acquire]`. The default driver is the simulator.
  - `bind_address` in `[web]`, the LAN address of the Pi. The default is the loopback address, so the UI is reachable from the Pi only. To open the UI by a host name, add the name to `allowed_hosts` (see [Reach the web UI through a VPN](#reach-the-web-ui-through-a-vpn)).
  - `catalog_path` and `index_dir` in `[survey]`, the cap catalog and the solver index (see **Cap catalog and solver index** below). `core` does not start without `catalog_path`.
  - The `[power]` route and the `[services.core.escalation]` `reboot_command` (see [The camera recovery ladder](#the-camera-recovery-ladder)).
  - The `[heater]` table, once you know the HAT.

  Do not write a path under `/home` into the file. The units hide `/home` from the services.

  The template has three lines that are not commented out and hold placeholders: `data_dir` in `[paths]`, `recordings_dir` in `[replay]`, and `token_hash` in `[auth]`. Delete the ones that you do not use. A placeholder `token_hash` stops `web` at the start, because it takes precedence over the hash that you give with `--token-hash-file`.
- **API token hash.** Run `seeingmon web hash-token --generate > <hash file>`. The command makes a random token, prints it once on the standard error, and writes the hash to the file. Keep the token in a password manager, and pass the file as `--token-hash-file`. To hash a token of your own, run the command without `--generate`: it reads the token from a hidden prompt or from the standard input.
- **Cap catalog and solver index.** Build them on a machine with a network connection, with `seeingmon catalog build --output <catalog file>`. The command queries the Gaia archive (a job that can queue for many minutes) and VizieR, and it writes the catalog (about 4 MB). When `build-astrometry-index` is installed, it also writes the solver index files to an `index` folder next to the catalog. Copy the catalog file and the folder to a place that the service user can read, such as a folder under the data directory. The Pi never runs the build. A build of the standard 15 degree cap takes a few minutes: it gave 82,065 stars (4.1 MB) and five index files (3.2 MB). Windows has no `build-astrometry-index`, so run the build in WSL or on Linux (`sudo apt install astrometry.net`); the index files work on any machine. The Gaia archive of the European Space Agency sometimes ends a job in the phase `ERROR` with a database lock message (`canceling statement due to lock timeout`). Run the command again, or pass a mirror with the same table names, such as `--gaia-url https://gaia.ari.uni-heidelberg.de/tap`.
- **Vendor SDK.** Download the archive from ZWO, compute its checksum once with `sha256sum`, and keep the checksum with your notes. The repository never holds the SDK.
- **Time sources.** The NTP servers that chrony uses. A Pi 4 has no real-time clock, so it needs at least one source on the LAN or the internet.
- **Environment file (optional).** Lines of `NAME=value` that the services read. Use it for the values that `token_env`, `password_env`, and `${NAME}` in the configuration name, such as a smart-plug token.
- **Connection key (optional).** The key that `acquire`, `core`, and `web` share. The installer makes one at the first install when you give none, and it keeps it.

## Install

Run `deploy/push.sh` on your development machine. Check the plan first, because `--dry-run` prints every command and runs none:

```bash
deploy/push.sh \
  --host <pi-host> --user <login-user> \
  --prefix /opt/seeingmon --service-user seeingmon \
  --data-dir /srv/seeingmon-data --config-dir /etc/seeingmon \
  --time-source <ntp-server> --time-source <second-ntp-server> \
  --local-config local/config.toml \
  --token-hash-file <path to the token hash file> \
  --sdk-archive <path to the SDK archive> --sdk-sha256 <checksum> \
  --installer-arg --supervisor-actions \
  --dry-run
```

Remove `--dry-run` to install. To see what the installer itself would do on the Pi, change nothing there, and read its plan, add `--installer-arg --dry-run`.

| Parameter | Meaning |
|---|---|
| `--host`, `--user` | The Pi and the account that you log in with over ssh. The installer needs root, so the script runs it through `sudo`, unless `--user` is `root`. |
| `--port`, `--identity`, `--ssh-option` | The ssh port, the ssh key file, and an extra ssh option as `NAME=VALUE` (repeat `--ssh-option` for more). Use them when your ssh setup needs more than a host and a user. |
| `--prefix` | Where the releases live. Each release gets its own virtual environment in `releases/`, and `current` points to the one that runs. |
| `--service-user` | The account that runs the three services. The installer creates it as a system account without a login. |
| `--data-dir` | The data directory. Only `core` writes to it. |
| `--config-dir` | Holds `local/config.toml`, the credentials, and the environment file. |
| `--time-source` | A chrony time source. Repeat it. Use `--no-time-config` to leave chrony alone. |
| `--local-config`, `--token-hash-file`, `--connection-key-file`, `--env-file` | Files that the script copies to the Pi, and that the installer installs with owner-only permissions. |
| `--sdk-archive`, `--sdk-sha256` | The vendor SDK and the checksum that it must have. |
| `--installer-arg` | Passes one argument to `install.sh`. Repeat it. See `deploy/install.sh --help` for the rest: `--keep`, `--usbfs-memory-mb`, `--wheelhouse`, `--supervisor-actions`, and more. |
| `--python` | The interpreter that `uv` uses to build the release on your machine. |
| `--dist-dir` | Use a finished build (the wheel and `requirements.txt`) and build nothing. |
| `--keep-stage` | Leave the staging directory on the Pi, so that you can look at it. |

In `push.sh`, `--user` is the account that you log in with. In `install.sh`, `--user` is the service user, and `push.sh` passes your `--service-user` there.

### What the script and the installer do

`build.sh` builds the wheel (`uv build --wheel`) and exports the pinned requirements with hashes from `uv.lock` (`uv export --locked`). The lock must be current: run `uv lock` after you change a dependency. `push.sh` copies both files, the `deploy/` folder, and your files to a private directory on the Pi, runs the installer, and removes the directory, also after a failure.

`install.sh` checks the parameters and the system first, and it changes nothing until every check passes. It then:

1. Creates the service user and joins it to the `gpio` group when the Pi has one.
2. Creates the prefix, the configuration directory (`root`, group of the service user, mode 0750), and the data directory.
3. Installs the release in `releases/<version>-<checksum>/venv` from the hashed requirements (`pip install --require-hashes --only-binary=:all: --no-deps`), then the wheel (`--no-deps`), and checks the result with `pip check`.
4. Checks the SDK checksum and installs the SDK privately, and points the services at its library.
5. Installs the connection key, the token hash, the environment file, and the local configuration, readable by their owners only.
6. Installs the five systemd units, the camera udev rule, the USB buffer setting, the journald setting (the journal stays in RAM), and the chrony sources.
7. Points `current` at the new release in one rename, keeps the previous release for a rollback, and removes older releases beyond `--keep` (two by default).
8. Reloads systemd and udev, applies the settings, and enables and starts the units. It restarts the services only when something changed.

It prints each step, and a summary of what changed and which warnings apply. It is safe to run again: a run with the same inputs changes nothing. It refuses to go on, with a message, when it finds an unexpected state. Examples are a system file that it did not write, an account with a login shell, a `current` that is a real folder, and a prefix that holds other files.

The release name holds the version and a checksum of the wheel and the requirements. A rebuild of the same version with other contents is a new release, so a running release never changes under a service.

### Install from a wheelhouse

When the Pi has no internet, build a folder of wheels for it (the Pi's Python version and `aarch64`) on a machine that has internet, copy the folder to the Pi, and tell the installer where it is:

```bash
deploy/push.sh ... --installer-arg --wheelhouse --installer-arg <folder on the Pi>
```

The installer then runs pip with `--no-index --find-links`. The hashes still apply.

## First start and checks

The installer starts the services. Then check:

```bash
systemctl status seeingmon.target seeingmon-acquire seeingmon-core seeingmon-web
journalctl -u seeingmon-acquire -u seeingmon-core -u seeingmon-web --since "10 minutes ago"
```

Every service must be `active (running)`. Use `<prefix>/bin/rollback.sh` (see [Roll back](#roll-back)) when a service of a new release does not start.

Then confirm, one by one:

- **The camera.** `lsusb -d 03c3:` lists it (03c3 is the USB vendor ID of ZWO). `ls -l /dev/bus/usb/*/*` shows the device node with the group of the service user and the mode `crw-rw----`.
- **The USB buffer.** `cat /sys/module/usbcore/parameters/usbfs_memory_mb` prints the size that you asked for (1000 by default). If it does not, reboot. If it is still wrong, add `usbcore.usbfs_memory_mb=1000` to the single line of `/boot/firmware/cmdline.txt`, and reboot.
- **The sandbox.** `systemd-analyze security seeingmon-core.service` prints an exposure level. The deploy lane measured about 2.5 for `acquire` and `core` and 1.9 for `web` on a development machine.
- **Time.** See [Time sync](#time-sync).
- **Health.** See [Check the health](#check-the-health).

The first seeing record needs one analysis window (60 seconds by default). The scheduler stays in `safe` while the Sun is above -3 degrees at your `[site]`, or while the sky is bright, so a first start by day shows no seeing record until dusk. Records carry `time_invalid` until chrony synchronizes the clock.

## Check the health

`web` serves `GET /api/v1/health` on the address and port of the `[web]` table (the port is 8080 by default):

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://<bind-address>:8080/api/v1/health
```

The endpoint answers 200 for a healthy or degraded system and 503 otherwise. It also answers 503 when the newest health record is older than `health_max_age_s` (180 seconds by default), which means that `core` stopped writing. `GET /api/v1/status` shows the state of every component.

| Result | What it means | What to do |
|---|---|---|
| 200, healthy | Every component works. | Nothing. |
| 200, degraded | A part works with a fault, for example the camera runs a recovery step, a sink lags, or time is not synchronized. | Read the `reasons` and the `components` in the answer, and the log of the named service. |
| 503 | A component failed (for example the camera failed repeatedly), the store cannot be read, `core` has written no health record yet, or its newest record is more than three minutes old because `core` stopped. | Read the `reasons` in the answer, then `systemctl status seeingmon.target` and the log of `core`. |
| No answer | `web` is down, or it binds to another address. | `systemctl status seeingmon-web`, and check `bind_address`. |

An external watchdog on your LAN can poll this endpoint (see [Remote power cycle](#remote-power-cycle)).

## Reach the web UI through a VPN

`web` listens on `bind_address` and on every address of `extra_bind_addresses`, and it never listens on all interfaces. It answers a request only when the `Host` header names an allowed host: the loopback names, every bind address, and the entries of `allowed_hosts`. A WebSocket handshake (the live view of the Align page) that carries an `Origin` header must name an allowed host there too, so a page from another site cannot open it. The server answers a request with another `Host` with 400, and a handshake with another `Origin` with 403.

To reach the UI through a VPN, such as a tailnet:

1. Note the address of the VPN interface of the Pi (IPv4 and IPv6, if both exist), the short host name, and the full name that the VPN gives the Pi.
2. Put them in the `[web]` table of the local configuration (`local/config.toml`, which stays out of the repository):

   ```toml
   [web]
   bind_address = "<LAN address of the Pi>"
   extra_bind_addresses = ["<VPN address, IPv4>", "<VPN address, IPv6>"]
   allowed_hosts = ["<short host name>", "<full VPN name>"]
   ```

3. Run the installer again, so that it copies the file, or restart the unit: `systemctl restart seeingmon-web`. The log of `web` has one `listening on` line for each address.
4. From a client on the VPN, open the full name in a browser or run `curl -s -o /dev/null -w "%{http_code}\n" http://<full VPN name>:8080/api/v1/health`. The answer is 200 or 503. A 400 means that the name is not in `allowed_hosts`, and the log of `web` names each rejected host once.

The addresses and names are deployment values, so the `/api/v1/config` endpoint and the `run` record show them as `<redacted>`. The bind of a VPN address fails while the VPN interface does not exist, and then `web` exits and systemd restarts it. At boot, the VPN can come up later than `web`, and the start limit of the unit can stop the restarts. If that happens, order `seeingmon-web` after the VPN service with a drop-in (`systemctl edit seeingmon-web`), and test a reboot. This is untested on a Pi.

## Where the logs go

The services write to the journal, and journald keeps the journal in RAM (`Storage=volatile`, at most 64 MB), so the logs never wear the SD card. The consequence is that a reboot erases the log of the previous boot. The `event` table of the store survives a reboot, and it holds the warnings and errors that matter, such as recovery steps and retention actions.

```bash
journalctl -u seeingmon-core -f                      # follow one service
journalctl -b -p warning                             # warnings and worse since the boot
journalctl -p crit                                   # a unit that hit its start limit
journalctl --disk-usage                              # how much RAM the journal uses
```

`systemctl status seeingmon-acquire` also shows the one-line status that `acquire` sends to systemd, which is its health summary.

When a unit fails five times in ten minutes, systemd stops restarting it, and `seeingmon-failed@.service` writes a critical message that names the unit and the commands to start it again. Fix the cause, and then run:

```bash
sudo systemctl reset-failed seeingmon-core.service
sudo systemctl start seeingmon-core.service
```

## Time sync

The Pi 4 has no real-time clock, so the clock starts wrong after a boot until chrony synchronizes it. Every record carries `time_invalid` until then, and the absolute time error of a frame includes the error bound of chrony.

```bash
chronyc tracking          # "Leap status: Normal" and a small "System time" offset mean a good lock
chronyc sources -v        # a "^*" marks the source that chrony uses
timedatectl               # "System clock synchronized: yes"
sudo chronyc makestep     # steps the clock at once after you fix a source
```

The installer adds your sources in `/etc/chrony/conf.d/seeingmon.conf`, and it leaves the default sources of Debian in `chrony.conf`. When the Pi has no internet, give it a time source on the LAN. A Pi without any source keeps `time_invalid` for good.

## Upgrade

1. On your development machine, update the repository and make sure that the lock is current (`uv lock`).
2. Run `deploy/push.sh` again with the same parameters. A new build is a new release. The installer installs it next to the old one, switches `current`, and restarts the services. Capture pauses while the services restart.
3. Check the services and the health (see [First start and checks](#first-start-and-checks)).
4. If the new release misbehaves, [roll back](#roll-back).

The installer renders the units, the udev rule, and the other system files from the templates at every run, and it overwrites a file that carries the line `Managed by the seeingmon installer`. To change a setting of a unit, use a drop-in that the installer never touches:

```bash
sudo systemctl edit seeingmon-core.service
```

An SDK update is the same run with a new `--sdk-archive` and `--sdk-sha256`. The OS updates itself for security if you enable `unattended-upgrades`. Application and SDK updates stay manual. Turn automatic reboots off, so that an update never reboots the Pi in the middle of a night.

## Roll back

The installer keeps the previous release. To go back, run on the Pi:

```bash
sudo /opt/seeingmon/bin/rollback.sh --prefix /opt/seeingmon
```

The script points `current` at the previous release and `previous` at the release that you leave, in two renames, and restarts the services. Run it again to go forward. Use `--dry-run` to see what it would do, and `--no-restart` to switch the link and restart later yourself.

A rollback switches the code only. The systemd units, the configuration, and the data stay as they are. To restore the units of an older release, run `push.sh` with the older checkout.

## The camera recovery ladder

A ZWO camera on a Raspberry Pi can stall after hours or days. The system answers with a ladder of six steps. The mildest step that works costs the least data.

| Step | Who | What happens | Your part |
|---|---|---|---|
| 1. Restart capture | The driver in `acquire` | Stops and starts the stream. | None. |
| 2. Reopen | The driver | Closes and opens the camera, and reapplies the settings. | None. |
| 3. USB reset | The driver | Resets the USB device (`USBDEVFS_RESET` on the device node, which the udev rule makes writable for the service group). | None. |
| 4. Restart `acquire` | `core` | Asks `acquire` to exit with code 75. systemd starts a new process after five seconds, so the vendor SDK starts clean. | None. |
| 5. Reboot | `core` | Runs `reboot_command` from `[services.core.escalation]`. | Name the command, and install with `--supervisor-actions`. |
| 6. Power cycle | `core` | Calls the power-cycle hook (see below). | Choose and wire a route. |

Each step gets two attempts. Once the system is `degraded`, the scheduler retries slowly (every 600 seconds). A reboot or a power cycle happens at most once in six hours (`[scheduler.ladder] destructive_interval_s`), and the power hook adds its own limits.

Until you configure them, steps 5 and 6 write an event and change nothing: `escalation.reboot_unavailable` for the reboot, and `power.cycle_unavailable` for a power route of `none`. Once you configure them, each writes its event and then runs your command or request. For the reboot, the service user cannot use `sudo`, because the units set `NoNewPrivileges`. The `--supervisor-actions` option installs a polkit rule that lets the service user, and nobody else, manage the three services (`seeingmon-acquire`, `seeingmon-core`, and `seeingmon-web`) and reboot the Pi. Then name the command in the local configuration:

```toml
[services.core.escalation]
reboot_command = ["/usr/bin/systemctl", "reboot"]
```

When you must act by hand, go from the mildest action to the hardest:

```bash
sudo systemctl restart seeingmon-acquire        # a clean SDK
# unplug and replug the camera
sudo reboot
# cut the power of the Pi
```

An exit of `acquire` with code 70 means that a call into the SDK hung and the watchdog ended the process. Code 71 means that one of its threads died. Code 75 is a restart that `core` asked for. All three end with a new process.

### Remote power cycle

A hard power cycle of the whole Pi is the last step. The camera loses its USB power with the Pi, because the Pi cannot switch a single USB port. The route is undecided (blocker B4), and the architecture lists two options:

- A managed PoE switch: cut and restore the power of the Pi's port.
- A smart plug on the PoE injector.

Both work through the `[power]` table of the local configuration, with either an HTTP request (`route = "http"`) or an argument list that `core` runs without a shell (`route = "command"`). A `${NAME}` in the URL, a header, the body, or an argument expands from the environment, so a secret lives in the environment file, and never in the configuration:

```toml
[power]
route = "http"
state_file = "/var/lib/seeingmon/power-state.json"
dry_run = true                       # check the route first, and set false when it works

[power.http]
method = "POST"
url = "http://<address of the smart plug>/<path of the cycle request>"
headers = { Authorization = "Bearer ${PLUG_TOKEN}" }
```

Put `PLUG_TOKEN=<token>` into the file that you pass with `--env-file`. The state file lives under `/var/lib/seeingmon`, which the `core` unit can write and which survives a reboot, so the limits of the hook (at least 3,600 seconds between attempts and three a day) survive the reboot that a cycle causes. With `dry_run = true`, the hook checks the route and sends nothing. Run the real request or command by hand once, with the Pi attached, before you rely on it.

An external watchdog adds a second line of defense. It runs on another machine of your LAN, polls `/api/v1/health`, and cycles the power when the check fails for several minutes. This sketch is an untested example:

```sh
#!/bin/sh
# Run it every minute from cron. It cycles the power after ten failures in a row.
set -eu
count_file=<a file on this machine>
if curl -fsS --max-time 10 "http://<bind-address>:8080/api/v1/health" >/dev/null; then
  echo 0 > "$count_file"
  exit 0
fi
count=$(( $(cat "$count_file" 2>/dev/null || echo 0) + 1 ))
echo "$count" > "$count_file"
if [ "$count" -ge 10 ]; then
  <the command or request that cycles the power>
  echo 0 > "$count_file"
fi
```

## SD card care

An SD card wears out with writes. The design keeps the write budget under 1 GB a day, and these rules keep it there:

- **Use a high-endurance card** of 32 GB or more. The rolling tiers need about 7 GB, and the results need under 0.5 GB a year.
- **Keep data on its own partition**, mounted with `noatime` (see [Create the data partition](#create-the-data-partition)).
- **Keep logs in RAM.** The installer sets `Storage=volatile`. If `/var/log/journal` exists from an earlier setup, remove it. If `rsyslog` is installed, remove it or point it at RAM, because it writes `/var/log` to the card.
- **Keep temporary files in RAM** (`/tmp` as a tmpfs) and **keep swap off the card**.
- **Check the writes.** The seventh field of the block device statistics counts the sectors written since the boot:

  ```bash
  awk '{printf "%.2f GB written since the boot\n", $7 * 512 / 1e9}' /sys/block/mmcblk0/stat
  ```

  A Pi that runs for a day and shows much more than 1 GB has a writer to find.
- **Know the retention tiers.** A task runs hourly and deletes the oldest files first. It never deletes a file that changed in the last 15 minutes.

  | Tier | Retention |
  |---|---|
  | Results (SQLite) | Forever |
  | Per-frame metrics | 7 days or 2 GB |
  | Star lists | 1 year |
  | Raw bursts | A 2 GB quota for unpinned bursts. A burst with a `PINNED` marker is exempt. |
  | Survey frames | 7 days, then one a night for 60 more days |
  | Previews | 7 days |

  Raw burst capture stops below 1 GB of free space and resumes at 1.5 GB, and the store writes `retention.capture_stopped` and `retention.capture_resumed`. `seeingmon store info <database>` prints the counts and the sink cursors.
- **Keep a copy.** After commissioning, copy the configuration directory and clone the card. The remote sinks hold a second copy of the results.
- **Expect a power cut.** SQLite runs in WAL mode, and the segment writer forces its file to disk every 60 seconds, so a power cut loses at most a minute of frame metrics. After a card error (`dmesg | grep -i mmc`, or a file system that turns read-only), replace the card.

## Commissioning commands

The installer puts a wrapper at `<prefix>/bin/seeingmon`. It runs the command as the service user, in the configuration directory, with the connection key, so the command reads the same configuration as the services:

```bash
sudo /opt/seeingmon/bin/seeingmon <command> --help
```

| Command | What it does | Note |
|---|---|---|
The wrapper does not load `<config-dir>/sdk.env`, which only the `acquire` unit reads. A command that opens the camera (`seeingmon camera rates`, and the `--standalone` forms of `burst`, `sweep`, and `dark`) therefore needs the library path: copy it from `sdk.env` to `library_path` under `[services.acquire.driver_options]` in the local configuration.

| `seeingmon burst` | Records frames to a SER file with a JSON sidecar, and pins the burst. Pinned bursts are exempt from retention, so a burst stays until you remove the `PINNED` file in its folder. | Also `POST /api/v1/commands/burst` (token required). |
| `seeingmon sweep` | Runs a short fast window for each cell of a grid (exposure, gain, ROI, readout mode) and prints saturation, signal-to-noise ratio, frame and drop rates, and estimator noise. | Also `POST /api/v1/commands/sweep` (token required). |
| `seeingmon dark` | Records a dark set with the camera covered, and adds it to the dark library. | It asks the running `core` to record the set, and it shows the progress, including why the latest test frame is not dark while it waits for the cover. `--detach` queues the session and returns. The scheduler pauses afterwards, so uncover the camera and resume the scheduler from the web UI. With `--standalone` it opens the camera itself, so stop the services first: `sudo systemctl stop seeingmon.target`. Start them again afterwards. |
| `seeingmon camera rates` | Measures the frame rates of the connected camera, one factor at a time around the fast stream: the exposure, the ROI size, the pixel format, the USB bandwidth, the high-speed mode, and the second readout mode. It prints the measured and modeled rates with the jitter and the drops, and it fits the frame overhead and the row time of the profile. | It opens the camera itself, so stop the services first. It puts back every control that it changed and closes the camera. `--json PATH` also writes the table to a file. The default `local/camera-rates.json` lies under the configuration directory when you run the wrapper, and the service user cannot write there, so give a path such as `/tmp/camera-rates.json`. |
| `seeingmon replay <source>` | Runs a SER recording through the production fast analysis at the original rate, at the maximum rate, or at a speed factor. `core` reads the file itself, and it needs no camera. | Also `POST /api/v1/commands/replay` (token required). The source is the name of a recording in the `[replay] recordings_dir` folder, or of a burst under `bursts/` of the data directory. The replay writes its own store to `replays/` of the data directory, which retention does not manage. |
| `seeingmon recordings info <path>` | Prints the geometry, the frame count, and the timing of a recording. | Read-only. |
| `seeingmon profile show` | Prints the hardware profile with its derived values. | Read-only. |
| `seeingmon store info <database>` | Prints the counts, the last row IDs, and the sink cursors of a store. | Read-only. The database of the station is `<data-dir>/db/results.sqlite`. |

`burst`, `sweep`, and `replay` queue a task in the running `core`, wait for the result, and print it. `--no-wait` queues the task and returns, and `--standalone` runs the task without `core`. A standalone `burst` or `sweep` opens the camera, so stop the services first, and a standalone `replay` needs no camera. Run `seeingmon <command> --help` for the options.

## Take a dark set from the UI

The analysis subtracts a master dark from every survey frame. The dark current of the sensor roughly doubles for every 5 degrees C (the development camera measures 4.9), so the dark library needs sets at the temperatures that the sensor reaches. The **Dark** page of the web UI records a set without a shell. The recording runs in `core`, so `core` must run.

1. Open the **Dark** page. The status line says `Due`, `Up to date`, or `Empty`, and why, and it shows the sensor temperature now. The chart plots the dark rate of each set against its temperature on a logarithmic scale, with the model as a line and the sensor temperature as a dashed marker. A temperature without a point near it is a gap that a new set closes.
1. Cover the camera, so that no light reaches the sensor. The camera has no lens cap, so use a cloth or a cap of your own. A single cap or a thin cloth is often not enough, because black plastic and thin cloth pass near-infrared light, and the sensor sees it. Put the camera face down on a table or in a closed box, or add a second opaque layer. A good cover reads below 1 count per second above the bias in a 1 s frame at room temperature, which is the dark current of the sensor. A cap and a cloth in daylight left about 5 counts per second on the development camera, and the check kept waiting.
1. Press **Start**. The first time, the page asks for the API token, because every command needs it.
1. Wait. The page lists the phases (bias frames, the wait for the cover, dark frames, and the master dark), marks the active one, and shows its step. The session starts at the next step of the scheduler, after a survey exposure in progress finishes. A paused scheduler or a running alignment holds the session back until you press **Resume** or the alignment ends, and the page says so.
1. When the page says that the session ended, remove the cover and press **Resume**. The page shows the set that the session added: its temperature, its dark rate, and its number of hot pixels.

### The wait for the cover

After the bias frames, the session takes a short test frame every few seconds (`test_exposure_s` and `poll_s` in `[survey.dark]`) and checks whether it is dark. The page shows the latest check in words, such as "The frame is not dark yet: the level is 5.20 e-/s above the bias, over the limit of 1.41 (level 2412 DN). Cover the camera." Two dark test frames in a row (`stable_polls`) count as covered, and the dark frames start.

The wait gives up after `wait_timeout_s` (30 minutes by default). The session then ends as `failed`, and the page says that the camera was not covered. Look for light around the cover, the lens, and the cable, and for light that passes through the cover itself, and press **Start** again. To skip the wait, clear "Wait until a test frame is dark" under **Advanced**. The first frame that is not dark then ends the session as `failed`.

### Why the station pauses afterwards

The scheduler pauses when the session ends, whatever the outcome, because the camera may still be covered, and nothing may record data until you uncover it. **Resume** puts the scheduler in `safe`, which checks the sky and goes on to `auto`. Clear "Pause the scheduler at the end" under **Advanced** only when something else uncovers the camera.

**Cancel** sends the pause command. The session ends as `aborted`, the library stays as it was, and the station stays paused until you press **Resume**.

### Why the temperature matters

The dark rate depends on the sensor temperature, so a set serves only the temperatures near its own. The library counts a set that lies within 3 degrees C of the sensor temperature (`temperature_tolerance_c`) and is younger than about six months (`max_age_days`). When no such set exists, the status line says `Due`. The model of the dark current fits the doubling step once the sets span 4 degrees C or more, and until then it assumes 6 degrees C (`doubling_c` in `[survey.dark]`). A wider span fits it better, so take a set on a cold night and another on a warm one, and look at the chart for the gaps. The `[web.requests]` table caps the exposure that the page may ask for (`max_dark_exposure_s`, 120 s by default).

## Troubleshooting

| Symptom | Likely cause | Check and fix |
|---|---|---|
| The installer says "run the installer as root". | You ran `install.sh` yourself without `sudo`. | Use `push.sh`, or run `sudo bash <staging directory>/deploy/install.sh ...`. |
| "cannot make virtual environments" | `python3-venv` is missing. | `sudo apt-get install python3-venv`. |
| "does not read /etc/chrony/conf.d" | A customized `chrony.conf` lacks `confdir`. | Add the line `confdir /etc/chrony/conf.d`, or pass `--no-time-config`. |
| pip reports a hash mismatch, or no matching distribution. | The requirements file is stale, or a package has no 64-bit ARM wheel for this Python. | Run `uv lock`, build again, and read which package fails. The installer allows binary wheels only. |
| The installer reports `exists and the installer did not write it`. | A file with that name came from somewhere else. | Rename or remove the file, and run again. |
| A unit fails with `status=226/NAMESPACE`. | A path of the unit does not exist, usually the data directory. | `findmnt <data-dir>`, `ls -ld <data-dir>`, and mount the data partition. |
| A unit fails with `status=243/CREDENTIALS`. | A credential file is missing. | `ls -l <config-dir>/credentials`, and run the installer again. |
| A unit fails with `status=203/EXEC`. | `current` is missing, or the release is broken. | `ls -l <prefix>/current`, then run the rollback or install again. |
| The log shows "Operation not permitted" for a system call, or a process ends with a bad system call. | The system call filter blocks a call that a library needs. | Allow it in a drop-in: `sudo systemctl edit seeingmon-core.service`, then add `SystemCallFilter=<call>`. Tell the maintainer. |
| A unit shows `start-limit-hit`. | It failed five times in ten minutes. | Read `journalctl -u <unit>`, fix the cause, then `systemctl reset-failed` and `systemctl start`. |
| `acquire` restarts again and again. | Exit code 70: the SDK hung. Code 71: a thread died. | Read the log. Check the USB cable, the port, and the power. Run `lsusb -d 03c3:`. |
| `core` fails to start, and the log says that `catalog_path` is not set. | `[survey] catalog_path` is empty in the local configuration. | Build the catalog and set the path (see [Prepare your files](#prepare-your-files)), then run the installer again. |
| No camera appears. | The udev rule did not apply, or the SDK path is wrong. | `lsusb -d 03c3:`. `ls -l /dev/bus/usb/*/*` must show the service group. `cat <config-dir>/sdk.env` must name an existing library. |
| Frames drop, or the stream breaks on large frames. | The USB buffer is too small. | `cat /sys/module/usbcore/parameters/usbfs_memory_mb`, and see [First start and checks](#first-start-and-checks). |
| Records carry `time_invalid`. | chrony has no synchronized source. | See [Time sync](#time-sync). |
| The health endpoint answers 503. | A component failed, or `core` wrote no health record for three minutes. | Read the `reasons` in the answer, then `systemctl status seeingmon.target` and the log of `core`. |
| The UI is not reachable from the LAN. | `bind_address` is still the loopback address. | Set the LAN address in `[web]` of the local configuration, and run the installer again. |
| Commands over the API are refused. | `web` has no token hash. | Run `seeingmon web hash-token`, and install the hash with `--token-hash-file`. |
| The data directory warns about the root file system. | No data partition. | See [Create the data partition](#create-the-data-partition). |
| A burst fails because raw capture stopped, and the store wrote `retention.capture_stopped`. | Less than 1 GB of free space. | `df -h <data-dir>`. Free space, or unpin old bursts. |
| `journalctl` shows nothing from before the last boot. | The journal lives in RAM. | Expected. The `event` table keeps the events that matter. |
| The UI answers `400` with `host_not_allowed`. | You opened it by a name that is not in `allowed_hosts`. | Add the name to `allowed_hosts` in `[web]`, or open the UI by the bind address. |
| The heater stays on after a service stops. | `seeingmon heater-off` is missing or failed. | Read the `ExecStopPost` line in `systemctl status seeingmon-core`, and prefer a HAT with its own failsafe. |
| Nothing runs after a reboot. | The units are not enabled. | `systemctl is-enabled seeingmon.target`, and run the installer again. |

## Security notes

- Use ssh keys only. Set `PasswordAuthentication no` for sshd.
- Reads of the API are open on the LAN by default, and commands need the bearer token. Reach the Pi from outside through a VPN (see [Reach the web UI through a VPN](#reach-the-web-ui-through-a-vpn)).
- The connection key, the token hash, the environment file, and the local configuration have the mode 0600 or live in a folder that only root and the service group enter. The installer checks the modes at every run. The units read the credentials through systemd. They hide `/home` from the services and mount the rest of the file system read-only. The exceptions are the data directory and `/var/lib/seeingmon` for `core`, the runtime directory `/run/seeingmon` for each service, and a private `/tmp`.
- The repository never holds a host name, an address, a user name, a key, a token, or the SDK.

## Check the deploy files without a Pi

On a development machine, run the linter and the tests. They cover the scripts, the units, the udev rule, and the templates, and they run on Windows, Linux x64, and Linux arm64:

```bash
python tools/lint_deploy.py
python -m pytest tests/deploy
```

The linter runs `shellcheck` when it finds the binary (the `shellcheck-py` package installs it, except on Linux arm64, where the linter runs `bash -n` and says so). The tests that run the scripts need a Linux system.
