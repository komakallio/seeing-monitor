# Runbook

This runbook covers how to install, upgrade, roll back, check, and look after the seeing monitor on a Raspberry Pi. It follows the Deployment section of [architecture.md](architecture.md). The scripts and files that it names live in `deploy/`. One section, [First light on the dev machine](#first-light-on-the-dev-machine), covers the first run on the real sky without a Pi.

Paths such as `/opt/seeingmon` and `/etc/seeingmon` are examples. The installer has no default for any path, host, or user name: you pass every value. Angle brackets, such as `<pi-host>`, mark a value that you supply.

## What is tested and what is not

The deploy lane could not run anything on a Raspberry Pi. On October 2, 2026 the lead ran `deploy/push.sh` on two Raspberry Pi 5 boards (Debian 13, Python 3.13, one with 8 GB and one with 16 GB of RAM, a ZWO ASI294MM on a USB 3 port), and on October 3 on a Raspberry Pi 4 with 2 GB (Debian 13, Python 3.13, an 8 GB card, the same camera). The next two paragraphs list what those runs confirmed. A data partition, a heater HAT, and a night of running are still untested. The table separates what the checks cover from what stays open.

**Confirmed on a Pi 5.** The install ran to the end, and the three services started and answered `/api/v1/health` with every component `ok` (`Type=notify` and the watchdog work). The udev rule gave the camera device the group of the service user, and `usbfs_memory_mb` was 1000 after a boot. The whole `tests/hardware` suite passed with the real camera on Linux (237 tests, including the USB reset, which the camera survives and streams again). `kill -9` of `acquire`, `core`, or `web` restarted the service after 6 to 7 s (`RestartSec` is 5 s). A hung `acquire` (`SIGSTOP`) was killed by the watchdog after 30 s with `SIGABRT` and restarted 37 s after the hang, and `core` reconnected on its own. Five restarts of one unit within 10 minutes make systemd stop restarting it: the unit stays `failed`, `seeingmon-failed@` writes "hit its start limit and stays stopped" to the journal, and the station stays down until you run `systemctl reset-failed <unit>` and start it (`StartLimitAction` is `none`). After `systemctl reboot`, all three services were active within 45 s, with the flag `time_invalid` until chrony had synchronized the clock, and the flag `low_space` appeared on a card with 400 MB free. The `camera` component reads `degraded` for a while after a restart of `acquire`, until frames flow again.

**Confirmed on a Pi 4 with 2 GB.** The install ran to the end, and the three services started with every component `ok`. The performance harness passed every budget (see [Results on a Raspberry Pi 4](performance.md#results-on-a-raspberry-pi-4)): the processes peak at 895 MB in sum, and the installed system with the real camera used at most 1,053 MiB of the 1,844 MiB, so 2 GB of RAM is enough. The camera streamed at the full rates with no drops, and `solve-field` solved synthetic fields in 0.3 to 0.6 s. The kernel of Raspberry Pi OS leaves the memory cgroup off, so the `MemoryMax` limits were not enforced until the kernel command line had `cgroup_enable=memory cgroup_memory=1` (the installer warns, and [Prepare the Pi](#prepare-the-pi) has the step). The first survey exposure after a start timed out twice on the real camera, and the first recovery step fixed it.

| Piece | What checks it | Untested without a Pi |
|---|---|---|
| `build.sh`, `push.sh`, `install.sh`, `rollback.sh` | `shellcheck` (Windows and Linux x64; Linux arm64 gets `bash -n`), a structural linter (`tools/lint_deploy.py`), and tests that run each script under bash with stub programs. One test makes a real virtual environment and runs real pip against a local wheelhouse. | A run as root on Raspberry Pi OS: `useradd`, `systemctl`, `udevadm`, the real package index, a real SD card |
| Systemd units | The linter parses them and checks the sandbox and the architecture rules. `systemd-analyze verify` printed no warning on a development machine. | Starting them on a Pi: the sandbox and the system call filter with the real libraries, and a unit that hits its memory limit |
| `acquire`, `core`, and `web` processes | Each process sends `READY=1` and `WATCHDOG=1` over the `sd_notify` socket, as the `Type=notify` units expect. Tests read the messages from a stand-in socket. | systemd acting on the messages: the start timeout, the watchdog restart, and the start limit. The installer reports a unit that does not start. |
| udev rule | Syntax (`udevadm verify` passed on a development machine) | A real camera: the group, the mode, and the autosuspend setting |
| USB buffer, journald, chrony fragments | Syntax and rendering | Their effect on a Pi: the `usbfs_memory_mb` write, the volatile journal, time synchronization |
| Polkit rule (`--supervisor-actions`) | Syntax and rendering | That `systemctl reboot` works for the service user |
| `seeingmon heater-off` | Unit tests on fake GPIO lines, including a call that hangs. The `core` unit runs it when it stops. The command exits with 0 when every heater output is off or no heater is configured, and with 1 when an output cannot be switched off, the configuration cannot be read, or the GPIO work passes its 1 s limit. | Real lines: the permission of the service user on `/dev/gpiochip*`, and what a line does after the release (see the heater paragraph of [architecture.md](architecture.md)). A failure writes its reason to the journal, one line for each failed output. Run by hand while `core` runs, the command exits with 1 and reports `the line is busy`. The minus sign in `ExecStopPost=-...` keeps a failure from failing the unit. |
| Recovery ladder, power cycle | The scheduler and the power hook have their own tests | A real stall, a real reboot, and a real power cycle (blocker B4: the route is undecided) |

A section that says "untested" describes the intended behavior, and you confirm it the first time you run it.

## Before you start

You need:

- A Raspberry Pi 4 with 2 GB of RAM, which is the final machine (measured on October 3, 2026), or a Raspberry Pi 5.
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

5. Keep swap off the SD card, so that the out-of-memory killer, and not the card, absorbs a memory peak. Raspberry Pi OS based on Debian 13 swaps to zram (compressed RAM) by default: `swapon --show` then lists `/dev/zram0` and no file on the card, and you need no action. If `swapon --show` lists a file on the card, such as `/var/swap`, turn that swap off:

   ```bash
   swapon --show
   sudo systemctl disable --now dphys-swapfile      # only if swapon lists a file on the card
   ```

6. Turn on the memory cgroup of the kernel. The three units limit their memory with `MemoryMax`, and systemd enforces that limit through the memory controller of the kernel's control groups (cgroup v2). The kernel of Raspberry Pi OS leaves the memory controller off until the kernel command line turns it on. Without it, systemd silently ignores `MemoryMax`, and only the out-of-memory killer of the whole system protects the Pi. Check the controllers:

   ```bash
   cat /sys/fs/cgroup/cgroup.controllers             # must list memory
   ```

   If the list has no `memory`, add two parameters to the single line of the kernel command line. Keep a backup, and keep the file to one line:

   ```bash
   sudo cp /boot/firmware/cmdline.txt /boot/firmware/cmdline.txt.bak
   sudo sed -i '1 s/$/ cgroup_enable=memory cgroup_memory=1/' /boot/firmware/cmdline.txt
   cat /boot/firmware/cmdline.txt                    # one line, with the two parameters at its end
   sudo systemctl reboot
   ```

   An old image that mounts the boot partition at `/boot` keeps the file at `/boot/cmdline.txt`. If the Pi does not boot, put the card in another computer, and copy `cmdline.txt.bak` over `cmdline.txt`.

   After the reboot, `cat /sys/fs/cgroup/cgroup.controllers` must list `memory`. After the install, while the units run, `systemctl show seeingmon-core -p MemoryCurrent` must print a number (see [First start and checks](#first-start-and-checks)). The installer reads the controllers file and warns when `memory` is missing, and it never edits the boot files.

7. Make `/tmp` a tmpfs. The services then write their temporary files to RAM:

   ```bash
   findmnt -n -o FSTYPE /tmp                         # prints tmpfs when it is already one
   sudo systemctl enable tmp.mount                   # otherwise, then reboot
   ```

8. Make the heater default to off at boot, before Linux starts. Add one line to `/boot/firmware/config.txt` for the GPIO pin of the heater output, and use `dh` instead of `dl` for a relay that switches on a low level. This line is untested, and the HAT is undecided (blocker B3):

   ```text
   gpio=<pin>=op,dl
   ```

### Create the data partition

The data directory belongs on its own partition, so that a full data area never fills the root file system. The installer does not partition anything. It warns when the data directory shares the root file system, and it goes on.

The steps depend on how you imaged the card, and the deploy lane did not test them on a Pi. A standard image grows its root partition to fill the card at the first boot. To keep room for a data partition, shrink the root partition on your computer before the first boot, create a second partition in the freed space, and format it:

```bash
sudo mkfs.ext4 -L seeingmon-data /dev/<data-partition>
```

Then mount it at the data directory through `/etc/fstab`. The `nofail` option lets the Pi boot without the partition. The `core` and `web` units require the mount, so they wait for it and fail to start when it does not come up, and they never write to the root file system by mistake. After you fix the mount, start the services again with `sudo systemctl start seeingmon.target`:

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
- **The memory limits.** While the units run, `systemctl show seeingmon-core -p MemoryCurrent` prints a number of bytes. If it prints `MemoryCurrent=[not set]`, the kernel has no memory cgroup, and systemd does not apply `MemoryMax`. Turn the memory cgroup on as [Prepare the Pi](#prepare-the-pi) describes, and reboot.
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

The sketch sends no token, so it fails on every poll when you set `require_token_for_reads`. Add the header `Authorization: Bearer <token>` to the `curl` command in that case, and keep the script readable by its owner only. The endpoint answers 503 for any failed component, including a heater fault, so decide whether such a failure should cycle the power. A lost SQM-LE reading never makes the endpoint answer 503: the `sqm` component stops at `degraded`, because a power cycle of the Pi cannot repair the unit or the program that feeds InfluxDB.

## Read the SQM-LE from InfluxDB

The SQM-LE does not have to be reachable over TCP. When another computer polls it and writes each reading to InfluxDB, `core` can read the readings from the database. Set `source = "influx"` in `[sqm]`, and describe the data in `[sqm.influx]`. Each poll asks InfluxDB for the newest point of the magnitude field, and each new point becomes a `reference` record with the time stamp of the point. The reader supports InfluxDB 1.x (InfluxQL) and 2.x (Flux), and it makes no claim about 3.x. No real server has answered it yet, so treat the first run as a test of your settings. The protocol of the `tcp` source stays unverified (blocker B5).

1. Collect the details: the version of the server, the bucket (version 2) or the database (version 1), the measurement, the field with the magnitude in mag/arcsec², the field with the temperature (optional), and the tags that select the unit when the measurement holds more than one.
2. Make a token that can read that one bucket and nothing else. On InfluxDB 2.x, create a custom API token with read access to the bucket. On 1.x, create a user that has `READ` on the database only. The reader never writes.
3. Put the token in the environment file that you pass to `push.sh` with `--env-file`. The installer installs the file as `<config-dir>/seeingmon.env` (mode 0600, owner root), and the `core` unit reads it as its `EnvironmentFile`:

   ```text
   SQM_INFLUX_TOKEN=<token>
   ```

4. Add the tables to `local/config.toml`. Every value belongs to your installation, so the file stays out of the repository:

   ```toml
   [sqm]
   enabled = true
   source = "influx"
   altitude_deg = 45.0                  # optional: where the unit points
   azimuth_deg = 0.0

   [sqm.influx]
   endpoint = "https://influx.example.org:8086"
   version = 2
   org = "<organization>"
   bucket = "<bucket>"
   token_env = "SQM_INFLUX_TOKEN"
   measurement = "<measurement>"
   field = "<field with the magnitude>"
   temperature_field = "<field with the temperature>"     # optional
   tags = { "<tag name>" = "<tag value>" }                 # optional: selects the unit
   max_age_s = 600.0
   lookback_s = 3600.0
   ```

   For version 1, set `version = 1` and `database` (and `retention_policy`, if you use one) instead of `org` and `bucket`, and give `username` and `password_env` instead of `token_env`. Set `max_age_s` to several times the interval at which the other program writes. `lookback_s` must be at least `max_age_s`, and a larger value lets the reader tell a stale reading from no reading. `verify_tls = false` turns the certificate check off, so use it only on a LAN that you trust. The reader never follows a redirect: put the final address in `endpoint`.
5. Try the settings (see below), then run the installer again so that it copies the file, or restart the unit: `sudo systemctl restart seeingmon-core`.

### Check the settings

`seeingmon hardware sqm` reads the configured source once and prints one line. It ignores `enabled`, so you can run it before you turn the reader on. This line shows made-up values:

```text
magnitude 21.37 mag/arcsec^2, temperature 3.5 C, age 12.3 s (source influx)
```

A failure prints one line on the standard error and exits with 1, for example `seeingmon: error: InfluxDB answered HTTP 401: unauthorized access (check the token or the credentials)`. The command prints no endpoint and no name from your configuration, and no message of the reader holds one: where a server repeats a name, the reader replaces it with `<redacted>`.

The command needs the variable that `token_env` names. On a development machine, set it in the shell and run the command with the same `local/config.toml`. On the Pi, only root reads `seeingmon.env`, so load it for one command. This line is untested on a Pi:

```bash
sudo bash -c 'cd <config-dir> && set -a && . ./seeingmon.env && set +a && <prefix>/current/venv/bin/seeingmon hardware sqm'
```

`seeingmon hardware sqm` also reads the `tcp` source, with `host` in `[sqm]`.

### When the readings stop

The age of a point is the difference between the clock of the Pi and the time stamp that the writer gave it, so keep both clocks synchronized (see [Time sync](#time-sync)). A point older than `max_age_s` makes the poll fail with the cause `Stale`. A failed poll never stops `core`:

- The first failed poll of a streak writes the event `sqm.read_failed` with the `cause` in its detail (`GET /api/v1/events?kind=sqm.read_failed`), and the `sqm` component of `health` reads `degraded`.
- The reader polls again after 5 s, then 10 s, 20 s, and so on up to 300 s (`backoff_initial_s` and `backoff_max_s`). The component reads `failed` from the fifth failed poll in a row, about 75 s after the first, and then `GET /api/v1/health` answers 503.
- The first poll that finds a fresh point writes `sqm.recovered`, returns to the poll interval (60 s), and the component reads `ok`. A point that repeats the last time stamp adds no record and is no failure, so a poll that is faster than the writer costs one small query.

| Cause | What it means | What to do |
|---|---|---|
| `Stale` | The newest point is older than `max_age_s`. | Check that the other program still writes, that both clocks are right, and that `max_age_s` suits the interval of the writer. |
| `NoData` | The last `lookback_s` seconds hold no point of the field. | Check `measurement`, `field`, and `tags`, and that the writer runs. |
| `Unreachable` | The network failed, the server did not answer within `timeout_s`, or it answered 5xx, 408, or 429. | Check that the server runs and that the Pi reaches it. |
| `Unauthorized` | The server answered 401 or 403. | Check the token or the user, and its read access to the bucket or the database. |
| `BadRequest` | The server answered another 4xx, or a redirect, or it reported an error. The message holds the words of the server. | Check the endpoint, the version, and the organization, the bucket, or the database. |
| `Parse` | The reply is not a reading: the field holds no number, or the magnitude lies outside -5 to 30. | Check that `field` names a numeric field, and `temperature_field` too. |

The `run` record shows the endpoint, the token, the password, and the names of the data (organization, bucket, database, retention policy, user name, measurement, fields, and tags) as `<redacted>`.

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
- **Know the retention tiers.** A task runs hourly and deletes the oldest files first. It never deletes a file that changed in the last 15 minutes. The data directory may use 25% of its partition (`quota_fraction` in `[store.retention]`). Past that, the task shrinks the tiers ahead of their age limits and writes `retention.early_delete` events.

  | Tier | Retention |
  |---|---|
  | Results (SQLite) | Forever |
  | Per-frame metrics | 7 days or 2 GB |
  | Star lists | 1 year |
  | Raw bursts | A 2 GB quota for unpinned bursts. A burst with a `PINNED` marker is exempt. |
  | Survey frames (FITS files) | 7 days, then one a night for 60 more days, and at most 4 GB |
  | Previews | 7 days, and as long as the survey frame of the same time stays, and at most 1 GB |

  `core` writes a preview (50 to 150 KB) of each long survey frame. It also writes a FITS file of about 10 MB (bin2) for every tenth long frame and for the first frame after an event: no pointing solution, a pointing that moved, clouds, or a bright sky. A night of 12 hours adds about 0.3 GB, and a week about 2 GB. To write fewer files, raise `keep_every` or `event_min_interval_s` under `[services.core.survey_frames]` in the local configuration, or set `enabled = false` to write none. The two tiers also have size caps, `survey_max_gb` (4 GB) and `previews_max_gb` (1 GB) under `[store.retention]`. Over a cap, retention deletes the oldest files first, keeps the one frame a night for last, and writes a `retention.early_delete` event. The Images page lists the previews of the last 7 days and the preview of the one frame a night that retention keeps for 60 more days, and the FITS files stay in `<data-dir>/survey/`. Open a FITS file with `astropy` (`fits.open(path)[1].data`), with DS9, or unpack it with `funpack`.

  Raw burst capture and the FITS files of the survey frames stop below 1 GB of free space (the previews go on) and resume at 1.5 GB, and the store writes `retention.capture_stopped` and `retention.capture_resumed`. `seeingmon store info <database>` prints the counts and the sink cursors.
- **Keep a copy.** After commissioning, copy the configuration directory and clone the card. The remote sinks hold a second copy of the results.
- **Expect a power cut.** SQLite runs in WAL mode, and the segment writer forces its file to disk every 60 seconds, so a power cut loses at most a minute of frame metrics. After a card error (`dmesg | grep -i mmc`, or a file system that turns read-only), replace the card.

## Commissioning commands

The installer puts a wrapper at `<prefix>/bin/seeingmon`. It runs the command as the service user, in the configuration directory, with the connection key, so the command reads the same configuration as the services:

```bash
sudo /opt/seeingmon/bin/seeingmon <command> --help
```

The wrapper does not load `<config-dir>/sdk.env`, which only the `acquire` unit reads. A command that opens the camera (`seeingmon camera rates`, and the `--standalone` forms of `burst`, `sweep`, and `dark`) therefore needs the library path: copy it from `sdk.env` to `library_path` under `[services.acquire.driver_options]` in the local configuration.

| Command | What it does | Note |
|---|---|---|
| `seeingmon burst` | Records frames to a SER file with a JSON sidecar, and pins the burst. Pinned bursts are exempt from retention, so a burst stays until you remove the `PINNED` file in its folder. | Also `POST /api/v1/commands/burst` (token required). |
| `seeingmon sweep` | Runs a short fast window for each cell of a grid (exposure, gain, ROI, readout mode) and prints saturation, signal-to-noise ratio, frame and drop rates, and estimator noise. | Also `POST /api/v1/commands/sweep` (token required). |
| `seeingmon dark` | Records a dark set with the camera covered, and adds it to the dark library. | It asks the running `core` to record the set, and it shows the progress, including why the latest test frame is not dark while it waits for the cover. `--detach` queues the session and returns. The scheduler pauses afterwards, so uncover the camera and resume the scheduler from the web UI. With `--standalone` it opens the camera itself, so stop the services first: `sudo systemctl stop seeingmon.target`. Start them again afterwards. |
| `seeingmon flat make` | Combines frames of a lit panel into the master flat for `[survey] flat_file`, and prints the vignetting, the tilt, the shadows, and how well the sets agree. | Offline: it needs no camera and no running service. See [Take a flat with a panel](#take-a-flat-with-a-panel). |
| `seeingmon flat build` | Builds a flat from the survey frames of the night sky, adds them to an accumulator file on request, and prints the vignetting, the shadows, and a check of the mask around Polaris. With `--base-flat` it compares the sky with a panel flat instead, and with `--update` it writes the panel flat with the changes that exceed their limits. | Offline: it needs no camera and no running service. It needs a clear night of frames, a dark library, and `[site]`. See [Build a flat from the night sky](#build-a-flat-from-the-night-sky). |
| `seeingmon camera rates` | Measures the frame rates of the connected camera, one factor at a time around the fast stream: the exposure, the ROI size, the pixel format, the USB bandwidth, the high-speed mode, and the second readout mode, and then single exposures of the survey mode (`--groups snapshot`). It prints the measured and modeled rates with the jitter and the drops, and it fits the frame overhead and the row time of the profile, and the snapshot overhead and row time. | It opens the camera itself, so stop the services first. It puts back every control that it changed and closes the camera. `--json PATH` also writes the table to a file. The default `local/camera-rates.json` lies under the configuration directory when you run the wrapper, and the service user cannot write there, so give a path such as `/tmp/camera-rates.json`. |
| `seeingmon hardware sqm` | Reads the SQM-LE once from the source that `[sqm]` names (the unit over TCP, or the readings in InfluxDB), and prints the magnitude, the temperature, and the age of the reading. | Read-only, and it needs no camera. It ignores `enabled`. The wrapper does not load `seeingmon.env`, so a variable that `token_env` names must be in the environment of the command (see [Check the settings](#check-the-settings)). |
| `seeingmon replay <source>` | Runs a SER recording through the production fast analysis at the original rate, at the maximum rate, or at a speed factor. `core` reads the file itself, and it needs no camera. | Also `POST /api/v1/commands/replay` (token required). The source is the name of a recording in the `[replay] recordings_dir` folder, or of a burst under `bursts/` of the data directory. The replay writes its own store to `replays/` of the data directory, which retention does not manage. |
| `seeingmon recordings info <path>` | Prints the geometry, the frame count, and the timing of a recording. | Read-only. |
| `seeingmon pointing set-reference` | Saves the newest good pointing solution of the store as the reference solution, and prints the line to add to `[survey.pointing]`. | It opens the store read-only, so it is safe while `core` runs. Restart `core` afterwards. See [Save the pointing reference](#save-the-pointing-reference). |
| `seeingmon pointing show` | Prints a reference file and, with `--data-dir`, the offset of the newest stored solution from it. | Read-only. |
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

## Take a flat with a panel

A flat field corrects the vignetting of the lens and the shadows of the dust on the sensor window. The survey analysis divides every frame by the flat before it measures the sky, so a flat that you measure makes the sky brightness more accurate. The default is a unit flat, which corrects nothing, and it stays the default until you set `[survey] flat_file`. A panel flat takes about half an hour. Take it before the first clear night, with the camera on the lens as it will run.

### What you need

- **The optical train as it will run.** Keep the same spacing between the camera and the lens, the same focus, and the dew shield if the station has one. Do not turn the camera on the lens afterwards, because the shadows of dust turn with the sensor.
- **A uniform, diffuse light.** The light must cover the whole 50 mm aperture and sit flush against the lens. An EL or LED panel works, and so does a tracing pad. A white cloth over the lens, in front of an evenly lit surface, works too. A phone screen works if you dim it and show a plain white page, but it has a gradient of its own (see [Turn the source](#turn-the-source)).
- **The survey mode.** Set bin2, the survey gain (`[survey.dark] gain`, 120 by default), the survey offset, 16-bit output, no region of interest, and the normal readout (high-speed mode off). The command takes the frame size from the profile (4144 × 2822 pixels in bin2), and it refuses frames of another size.
- **A recorder.** SharpCap or ASICap can record a SER file. A script can also write FITS files: one file for each frame, 16 bits (`BZERO` 32768), uncompressed.
- **A dark set.** You need no bias frames. The command takes the bias level from the dark library, interpolated to the sensor temperature in the FITS headers, so record a dark set first (see [Take a dark set from the UI](#take-a-dark-set-from-the-ui)) at a temperature near the one of the flat.

### Take the frames

1. Set an exposure of 0.1 s or more that gives 30 to 50% of full scale. In a 16-bit file, that is a mean of about 20,000 to 33,000 counts. Dim the light or shorten the exposure to get there, and check that no pixel saturates.
1. Record 24 frames into one SER file, or into one folder of FITS files. Keep the light steady: the command drops a frame whose mean level differs from the median of its set by more than 5%, and a frame outside 20 to 80% of full scale. The command warns when fewer than 15 frames of a set pass.
1. Turn the source by 180 degrees, and record 24 more frames into a second file or folder (see the next section). You can skip this step.

### Turn the source

A light source has a gradient of its own. A phone screen shows up to about 1% across its width, and that gradient lands in the flat as a tilt. One set cannot tell the gradient of the source from the tilt of the optics and the sensor, which is the part that you want. A second set that you record after you turn the source by 180 degrees separates the two: the gradient of the source flips, and the tilt of the optics stays. The command averages the two sets, which cancels a gradient that turned (to first order), and it reports the tilt of each part.

You can skip the second set. The flat then carries the gradient of the source in its tilt, up to about 1% across the frame for a phone screen, and the command warns about it. The night sky cannot correct that tilt later, because it cannot tell a tilt of the flat from a gradient of the sky itself (see [Build a flat from the night sky](#build-a-flat-from-the-night-sky)).

Pass `--source-turned` only when you did turn the source. The pixels cannot show whether you did, so the command takes your word. Without the option, it prints the same numbers under neutral names, and it warns when the two sets differ in tilt by more than 0.3%: the source drifted, or you turned it and did not say so.

### Make the flat

Run the command on any machine that has the project installed. It needs no camera and no running service. Copy the result to the Pi afterwards.

```bash
uv run seeingmon flat make --frames <set A> --frames <set B> --source-turned --out <flat file>.npy
```

`--frames` takes a SER file or a folder of FITS files, and you repeat it for each set. `--out` takes a `.npy` file or a FITS file (`.fits`). The command reads the configuration for the profile and the dark library (`[survey] calibration_dir`, or `calibration/` in the data directory), and `--calibration-dir` and `--local-config` name others. These options change what the command does:

| Option | Meaning |
|---|---|
| `--bias-level N` | Use this bias level, in the counts of the frames, and not the one of the dark library. |
| `--bias <path>` | Check bias frames (a SER file or a folder of FITS files) that you took with the lens covered, and use them when they pass. They fail when the noise of a pixel exceeds 3.5 counts, or the middle is more than 0.5 counts brighter than the corners, which means that light reached the sensor. The command then warns and uses the level of `--bias-level` or of the library. |
| `--temperature-c T` and `--gain N` | Say the sensor temperature and the gain of the frames when the files do not (a SER file never does). The command needs them to look up the bias in the library. |
| `--full-scale N` | Set the count where the ADC saturates. The command finds it from the frames: 65,535 for a 14-bit camera in a 16-bit container (every count a multiple of 4), and 16,383 for native 14-bit counts. |
| `--min-level-percent`, `--max-level-percent`, `--flicker-percent`, `--min-frames` | The tests of a frame and the count that raises the warning. The defaults are 20, 80, 5, and 15. |
| `--bin`, `--high-pass-px`, `--center-x`, `--center-y` | The binning (4), the width of the Gaussian that splits the smooth part from the fine part (40 binned pixels), and the optical center (the middle of the frame) for the report. |

### Read the report

The command prints plain text and no path. A report of two sets of frames looks like this. The numbers follow a real test of the development lens (a 50 mm f/5 guide scope):

```text
Flat from panel frames: 4144 x 2822 pixels, median 1.
Set 1: 24 of 24 frames used. Mean level 30,066 counts, 45.9% of the full scale of 65,535 counts.
  Noise per pixel: 1.22% in one frame, 0.25% in the mean. Saturated pixels: 0.000%.
Set 2: 24 of 24 frames used. ...
Bias: 133.9 native counts (535.6 in the counts of the frames), from the dark library (2 sets of bin2 at gain 120, interpolated to 29.5 C).
Noise of the flat: 0.18% per pixel, 0.04% at 4 x 4 binning.
Tilt of each set after the radial part:
  set 1: -0.08% across the width, +0.34% across the height
  set 2: +1.20% across the width, -1.02% across the height
Tilt of the optics and the sensor (half the sum of sets 1 and 2, which stays when the source turns): +0.56% across the width, -0.34% across the height
Gradient of the light source in set 1 (half the difference, which turns with the source): -0.64% across the width, +0.68% across the height
Quotient of set 1 over set 2:
  smooth part 0.54% rms, fine part 0.09% rms (the noise predicts 0.09% at this binning)
  plane of the quotient: -1.28% across the width, +1.36% across the height
Vignetting at each radius from the center, against the center:
  0.5 degrees: -0.43%
  1.0 degrees: -2.25%
  1.5 degrees: -4.04%
  2.0 degrees: -6.50%
  2.5 degrees: -9.27%
  corners (2.66 degrees): -9.89%
Tilt of the flat after the radial part: +0.56% across the width, -0.34% across the height
Shadows deeper than 1%: 3
  x 2489, y 1994: depth 3.0%, width 71 px
  ...
Edge artifacts deeper than 1% (center within 20 px of an edge): 2
  ...
Wrote flat.npy. Set flat_file in the [survey] table to its path.
```

- **Frames.** Each set lists the frames that passed, the mean level in counts and in percent of full scale, and the noise of a pixel. Frames that dropped are listed with the reason. A level over 80% or under 20%, and a flicker above 5%, drop a frame. A warning follows when fewer than 15 frames pass, when more than 0.1% of the pixels saturate, or when the mean level lies outside 20 to 80%.
- **Bias.** The line says which bias the command used: the dark library, `--bias-level`, or checked bias frames. A bias frame that holds light raises a warning, and the line then names the replacement.
- **Agreement of the sets.** Two sets of a steady source agree to the noise on the fine part. The plane of the quotient is the difference of the two tilts. It shows the gradient of the source only if you turned or moved the source between the sets. If you did not, it shows how far the source drifted.
- **Vignetting.** The flat at five radii from the optical center and in the corners, against the center. The development lens loses about 0.4% of the light 0.5 degrees from the middle, 2.3% at 1 degree, 4% at 1.5, 6.5% at 2, 9.3% at 2.5, and 9.9% in the corners. Another lens gives other numbers.
- **Tilt.** The plane across the frame after the radial part is divided out, as a change across the width and across the height. A positive value means that the flat rises toward the right edge or the bottom edge. The optics and the sensor cause the tilt that stays in the flat, and the gradient of the light source adds to it unless you turned the source.
- **Shadows.** The dips deeper than 1% in the fine part, with the position in sensor pixels, the depth, and the width (the diameter of the circle that has the area of the pixels deeper than half of the depth). A dip whose center lies within 20 pixels of an edge is an edge artifact, and the report lists it apart.

The flat has a median of 1. The command floors a pixel that reads at or below zero at 0.05 and warns, so that the survey path can divide by every pixel.

### Use the flat

1. Copy the file to the Pi, and set `flat_file` in the `[survey]` table of `local/config.toml` to its path.
1. Restart `core`, which loads the flat when it starts. The `sky_quality` records carry `provenance.flat` with the name of the flat (`flat-` and a hash of its pixels), so you can see from which record on the pipeline used it. A unit flat shows `unit`.

### When to take it again

Take a new flat when the camera comes off the lens, when you change the spacing or the focus, and when you clean the sensor window, because each of them moves the shadows of the dust or changes the vignetting. Between panel flats, the night sky reports the small changes of the vignetting and of the dust (see [Compare the sky with your panel flat](#compare-the-sky-with-your-panel-flat)).

## Build a flat from the night sky

The camera is fixed to the ground and points at the pole, so the sky turns about the middle of the frame, 15 arcsec every second. The mean of many frames in the sensor frame, with the stars masked, holds the flat times the mean sky. The structure that is fixed on the sky (faint stars, nebulosity) averages down to its mean about the pole, because the rotation spreads it round the frame. A mount that tracks the sky could not do this. `seeingmon flat build` takes the survey frames that `core` keeps as FITS files and turns them into the flat that `[survey] flat_file` loads. It needs no panel and no camera.

The sky cannot give the tilt. It cannot tell a tilt of the flat from a gradient of the sky itself, so the flat holds none. The tilt of the optics and the sensor stays in your frames, about 1% at the frame edges, until you take a panel flat (see [Take a flat with a panel](#take-a-flat-with-a-panel)). With a panel flat as the base, the command keeps its tilt and takes the changes of the vignetting and the dust from the sky (see [Compare the sky with your panel flat](#compare-the-sky-with-your-panel-flat)).

### When to run it

Run it after the first clear night, and again after later clear nights. You need:

- **Survey frames.** `core` keeps every tenth long frame as a FITS file under `<data-dir>/survey/`, which is about 24 frames in a clear night. The files expire after 7 days, and one frame a night stays for 60 days more, so run the command before the files go. The header of each file carries the cloud fraction and the transparency of its result (`CLOUDFRC` and `TRANSP`). Files from before those cards existed are *unchecked*: pass `--accept-unchecked` to use them anyway.
- **A dark library.** The command subtracts the dark the way the survey pipeline does: the level of the dark model for the sensor temperature and the exposure. When a dark set lies within 3 degrees C of the sensor temperature of a frame and has its exposure, the command also subtracts the per-pixel master dark of that set, so the pattern of the dark does not reach the flat. The report says which dark each frame had. Record a set at the temperature of your nights first (see [Take a dark set from the UI](#take-a-dark-set-from-the-ui)).
- **A site.** The command computes the Sun and the Moon from the time of each frame and from `[site]` in `local/config.toml`, because the header carries no position. Without a site, it cannot rule out twilight or moonlight, so it refuses the frames, unless you pass `--accept-unchecked`. The sky level test still guards against a bright sky then.

### Run it

```bash
uv run seeingmon flat build <data-dir>/survey --out <flat file>.npy --accumulator <accumulator file>.npz
```

The command reads the header of every file, applies the selection below, and prints a line for each frame that it processes. A frame of 4144 × 2822 pixels takes about 1 s on a development machine, and the command holds one frame at a time, which takes a few hundred MB. A Pi takes a few times longer, so run it on the dev machine when you can, and copy the `.npy` file to the Pi.

`--accumulator` keeps the running sums, so that a later run adds only the new frames. The file holds the sum of the masked, normalized frames at 4 × 4 binning, the sum of their squares, a count, and the time and the roll of every frame that went in. A run reads the folder, skips the frames that the file already holds, adds the new ones, saves the file, and builds the flat from the whole file. A frame that expires from the folder stays in the sum. The command writes the file to a temporary name and renames it, so a crash cannot corrupt it. Without the option, one run uses the frames of the folder and keeps nothing.

### Which frames count

An event frame never counts: the `KEPT` card of a frame that `core` kept for no pointing solution, a pointing that moved, clouds, or a bright sky says so. Every other frame counts when all of these hold:

| Test | Default | Option |
|---|---|---|
| The Sun is below this elevation | -18 degrees | `--max-sun-elevation` |
| The Moon is down (below `--moon-min-elevation`, 0 degrees), or it is lit less than this | 25% | `--max-moon-illumination` |
| The cloud fraction is under | 0.1 | `--max-cloud-fraction` |
| The transparency is at least | 0.95 | `--min-transparency` |
| The sky level is within this share of the median level of the chosen frames | 10% | `--sky-tolerance-percent` |
| The exposure is at least | 5 s | `--min-exposure-s` |

The clock must have been synchronized, the frame must be in the survey mode with no region of interest, and the header must carry a sensor temperature. The sky level comes from the pixels (the median above the dark per second), so it needs no header card, and its median covers the frames in the accumulator too. The report counts the rejected frames for each reason.

### What the command does to a frame

It detects the stars (`seeingmon.survey.detect`) and masks each one with a radius that grows with its flux. It also masks the saturated pixels, the hot pixels of the dark library and of `[survey] hot_pixel_file`, the 8 pixels at the frame edge, and a large disk around Polaris. The halo of Polaris makes a bright ring at the radius of its orbit, and nobody has measured the wings of this lens yet, so the disk has a radius of 400 pixels (`--polaris-mask-px`). The command takes the brightest star of the frame, when it outshines the next one by a factor of 3, for Polaris. It divides the frame by its own sky level (a sigma-clipped median of the pixels that stay), and adds the result into the sums.

After the last frame, the mean `M` is the sum over the count. The flat is the radial part of `M` (the azimuthal mean about the middle of the frame, which is the optical center unless you set `--center-x` and `--center-y`) times the fine part (`M` over the radial part and over its smooth rest, a Gaussian of 40 binned pixels, `--high-pass-px`). The smooth rest holds the tilt and the gradients of the sky, and the flat leaves it out. The flat has a median of 1, and the command spreads it over the sensor with bilinear interpolation (`--bin` sets the binning, 4 by default).

### Read the report

```text
Flat from the night sky.
Frames: 31 found in the folder, 4 already in the accumulator, 3 rejected, 24 added now.
  Rejected: 1 frame with a cloud fraction of 0.1 or more, 2 frames with a sky level more than 10% from the median.
Accumulator: 28 frames from 2026-12-09 to 2026-12-10. Roll coverage: 173 degrees.
Dark: the master dark of the set of 2026-12-01 for 24 frames.
Noise: 0.57% per binned pixel (4 x 4) in the mean sky. One pixel of one frame scatters by 11.4%.
Vignetting at each radius from the center, against the center:
  0.5 degrees: -0.38%
  ...
  corners (2.65 degrees): -9.91%
Tilt: not determined. The sky cannot tell a tilt of the flat from a gradient of the sky itself, so the flat holds none. ...
Shadows deeper than 1.5% (the noise raises the search above 1%): 3
  x 2489, y 1994: depth 3.0%, width 71 px
  ...
Edge artifacts deeper than 1.5% (center within 20 px of an edge; the noise raises the search above 1%): none
Polaris orbit: radius 584 px (from the positions of Polaris), bump +0.12% against a limit of 0.3%.
Time: 28.4 s, 1.2 s for each frame added.
Wrote flat.npy. Set flat_file in the [survey] table to its path.
```

The numbers in this example come from a synthetic night. Read the lines like this:

- **Frames and the accumulator.** The report counts the frames of the folder, the ones that the accumulator already holds, the ones that the tests rejected (with the reasons), and the ones that this run added. It then gives the number of frames in the accumulator, their date range, and the roll coverage: 360 degrees minus the largest gap between the roll angles of the frames, and the roll of a frame is the Earth rotation angle at its time.
- **Dark.** The report names the dark sets that the frames used, and the number of frames that had only the level of the dark model.
- **Noise.** The noise of the mean sky in a binned pixel, and the scatter of one pixel of one frame. With 24 frames the noise is about 0.6% at 4 × 4 binning, and it falls with the square root of the number of frames.
- **Vignetting.** The flat at five radii from the optical center and in the corners, against the center. Compare it with the panel flat when you have one: the two should agree within about 0.5%.
- **Shadows.** The dips deeper than 1% in the fine part. When the noise is high (a few frames), the search needs 5 times the local noise instead, and the heading says so. A dip within 20 pixels of an edge is an edge artifact, and the report lists it apart.
- **Polaris orbit.** The command measures the circle that Polaris follows from its positions in the frames (the pole is the center, and the orbit its radius), and falls back on the ephemeris and the optical center when fewer than 8 frames spread over 90 degrees. It compares the mean sky at that radius with a smooth baseline from the rings on both sides. A bump over 0.3% means that the mask around Polaris is too small, and the report warns. Raise `--polaris-mask-px` and build again. The accumulator keeps the masks of the earlier frames, so delete it and run the command on the frames that you still have.
- **Time.** The time of the run and for each frame added.

The command warns when fewer than 20 frames (`--min-frames`) or less than 60 degrees of roll (`--min-roll-deg`) went in, because the rotation has not averaged the structure of the sky then, and the fine structure is noisy. A simulation with the roll angles of a real year at 60 degrees north gave, for one clear night of 24 frames over 190 degrees of roll, a fine part good to 0.31% rms and a radial profile good to 0.14%, and for 240 frames 0.22% and 0.15%. The structure of the sky about the pole sets that floor, and not the photon noise.

### Compare the sky with your panel flat

A panel flat has the tilt, and the sky cannot give it. The sky has what a panel flat cannot keep current: the vignetting and the dust shadows as they are now. `--base-flat` combines the two, and the tilt of the new flat always comes from your panel flat.

```bash
uv run seeingmon flat build --accumulator <accumulator file>.npz --base-flat <panel flat>.npy
```

The command divides the mean sky by the base flat and reports what changed. It writes nothing. Name the accumulator of an earlier run, and the command reads no frame again, or add the folder of the frames to add the new ones first. The base flat is a `.npy` file or a FITS file of the size of the survey mode, as `[survey] flat_file` takes it. This is the report of a synthetic night, for a base flat with 1% of tilt that was taken before the vignetting deepened and before dust landed on the window:

```text
Flat from the night sky, compared with a base flat.
Frames: 0 found in the folder, 0 already in the accumulator, 0 rejected, 0 added now.
Accumulator: 24 frames from 2026-12-09 to 2026-12-10. Roll coverage: 173 degrees.
Noise: 0.57% per binned pixel (4 x 4) in the mean sky. One pixel of one frame scatters by 11.4%.
Base flat: panel.npy. The mean sky is divided by it.
Change of the vignetting against the base flat, at each radius from the center (against the disk within 0.40 degrees of it):
  0.5 degrees: -0.11%
  1.0 degrees: -0.40%
  1.5 degrees: -0.99%
  2.0 degrees: -1.58%
  2.5 degrees: -2.32%
  corners (2.66 degrees): -2.57%
Radial profile: the largest change is -2.3% at 2.5 degrees, over the limit of 1%.
Plane: the mean sky over the base flat has -0.99% across the width, +4.51% across the height (a positive value means that it rises toward the right edge or the bottom edge). That is the gradient of the sky plus any change of the tilt of the flat, and the sky cannot tell them apart, so the tilt comes from the base flat.
New shadows deeper than 1.5% (the noise raises the search above 1%): 1
  x 1500, y 1000: depth 4.5%, width 127 px
Patches brighter than the base flat by more than 1.5% (the noise raises the search above 1%): none
Edge artifacts deeper than 1.5% (center within 20 px of an edge; the noise raises the search above 1%): none
Polaris orbit: radius 584 px (from the positions of Polaris), bump +0.12% against a limit of 0.3%.
An update would apply the radial change and 1 new shadow. It would leave the plane and every smaller change as the base flat has them.
Nothing written. Add --update and --out to write the new flat.
```

Read the lines like this:

- **Vignetting.** The change at each radius is the mean sky over the base flat at that radius, in percent of its mean over the disk at the middle of the frame (the disk within 15% of the way to the corners, which is about 0.4 degrees). The sky does not turn at the middle, so a smaller disk would carry structure of the sky that the rotation does not average, and it would shift every change with it. A change of more than 1% at any of the five radii exceeds the limit (`--radial-limit-percent`). The rings of the sky itself add 0.15 to 0.5% to the profile, so a smaller change is noise.
- **Plane.** The plane of the mean sky over the base flat. It holds the gradient of the sky and any change of the tilt of the flat, and the sky cannot tell them apart. The command reports it, and an update never applies it.
- **New shadows.** The dips deeper than 1% in the fine part of the quotient: dust that landed after the panel flat. When the noise is high, the search needs 5 times the local noise instead, and the heading says so.
- **Bright patches.** The bumps brighter than 1% over a shadow of the base flat: dust that left or moved. A bump over no shadow of the base flat is the residue of a star that the masks missed, and the report counts it as ignored.
- **Edge artifacts.** The dips within 20 pixels of an edge. The report lists them apart, and an update leaves them out.
- **The last line.** It says what an update would apply.

Add `--update` and `--out` to write the new flat. `--out` must not name the base flat, so that your panel flat stays as it is.

```bash
uv run seeingmon flat build --accumulator <accumulator file>.npz --base-flat <panel flat>.npy --update --out <new flat>.npy
```

The new flat is the base flat times a correction, scaled to a median of 1. The correction holds only the changes that exceed their limits:

- the radial change, as a whole, when it exceeds 1% at any of the five radii, and not at all otherwise;
- each new shadow and each bright patch, over its own region with a soft edge.

It never holds the plane, a radial change under its limit, or a dip at the frame edge. Everything else stays as the base flat has it, down to the pixel: the tilt, the pattern of the pixels, and the shadows that did not change. When the ring check fails (the Polaris orbit line of the report warns), the update applies nothing, because the halo of Polaris is in the mean sky: raise `--polaris-mask-px`, and build again.

A light smoothing takes the noise out of a shadow that the update takes from the sky, so the shadow comes out a little shallower. In a simulation of the development lens, the core of a 3% shadow that is 71 pixels wide came out at 2.8%. In a simulation of one clear night of 24 frames, the update took a base flat whose vignetting was off by up to 2.4% and that lacked a shadow of 3% to 0.2% rms from the lens. The flat from the sky alone was off by 0.6% rms, because it lacks the tilt. When many shadows changed, take a new panel flat instead.

### Use the flat

Set `flat_file` in the `[survey]` table of `local/config.toml` to the path of the file on the Pi, and restart `core`. A unit flat stays the default until you do. The `sky_quality` records then carry `provenance.flat` with the name of the flat. Without a panel flat, the tilt of the optics, about 1% at the frame edges, stays in your sky quality values.

## Save the pointing reference

The **Pointing** card shows how far the camera has moved from a reference solution: a pointing solution that you save once, after you align the camera. Until you save one, the large value of the card stays empty, and its note says "no reference solution". With a reference, each `pointing` record carries `offset_arcmin` (the angle between the boresight of the record and the boresight of the reference) and `reference_id`. A record gets the `moved` flag when the offset exceeds `moved_arcmin` (5 arcminutes) or the roll changed by more than `moved_roll_deg` (0.5 degrees), both in `[survey.pointing]`. The reference is not the target of the **Align** page, which stays your setting in `[alignment]`.

### When to run it

Run the command when the camera is aligned: the **Align** page shows Polaris on the aim, you pressed **Stop alignment**, and `core` has solved a few survey frames, so the **Pointing** card shows a solution with hundreds of matched stars. Run it again, with `--force`, after you move the camera on purpose. A reference from before the move makes every later record say that the camera moved.

### Run it

```bash
seeingmon pointing set-reference --data-dir <data folder>
```

On a Pi, run the wrapper of [Commissioning commands](#commissioning-commands): `sudo /opt/seeingmon/bin/seeingmon pointing set-reference`. It reads the data directory from the configuration, so it needs no `--data-dir`.

The command opens the store read-only, so it is safe while `core` runs. It takes the newest `pointing` record that has a solution, at least 100 matched stars (`--min-matched`), a finite residual, and no `time_invalid` flag, and that is at most 60 minutes old (`--max-age-min`). When no record fits, it exits with status 1 and one line that says why.

It writes `pointing-reference.json` to the calibration folder (`calibration_dir` in `[survey]`, or `calibration/` in the data directory). `--out` names another file. When `[survey.pointing] reference_file` names a file, that file is the default instead. The command refuses to replace a file that exists, unless you add `--force`, and it writes the file in one step. The file holds the attitude of the camera in an Earth-fixed frame, so the offset does not depend on the time of day. The command prints a summary of the solution and the lines to add (the numbers are made up):

```text
Saved the pointing reference.
reference ID          reference-20261004T183852Z
solution time         2026-10-04T18:38:52Z (21 min ago)
matched stars         412
residual              0.85 arcsec rms
roll                  25.00 degrees
plate scale           3.820 arcsec/px in bin2
center from the pole  0.400 degrees
file                  <calibration folder>/pointing-reference.json

Add this to local/config.toml, then restart core to load the reference:

[survey.pointing]
reference_file = "<calibration folder>/pointing-reference.json"
```

Check that the solution time is recent and that the center from the pole is small, as it is when the pole sits on the aim at the middle of the frame. The roll is the position angle of the direction to the pole in the image, from image up toward image left.

### Use it

Add the lines to `local/config.toml`, and restart `core`. `core` loads the file once, when it starts. On a Pi, run `sudo systemctl restart seeingmon-core`. On the dev machine, stop `seeingmon dev` with Ctrl+C and start it again. The **Pointing** card shows the offset from the first survey frame that the plate solver solves after the restart. When `reference_file` already names the file that the command wrote, the command says so, and you only restart `core`.

`core` ignores a `reference_file` that names no file, and it logs no error, so the card keeps its note. To check the file that the configuration names, run `seeingmon pointing show`. It prints the same summary as the command, and it says when the file does not exist. With `--data-dir <data folder>`, it also prints the offset of the newest stored solution from the reference.

## First light on the dev machine

`seeingmon dev --driver asi --real-sky --data-dir <data folder>` runs `acquire`, `core`, and `web` on the dev machine against the real sky: your camera in real time, the real star catalog, a real plate solver, and your site. It is the first test of the survey path on real stars, and it needs no Raspberry Pi. It is a development run and not an install: no systemd unit runs, and no sink, heater, SQM-LE reader, or power route takes part.

**What the first light verified, and what is still open.** On October 4, 2026, real star images ran through the system for the first time. The detector found 3,000 stars (its limit) in each 30 s frame. The pointing tracker matched about 750 to 930 of them with a residual of 0.45 pixel, and the sky quality fitted a zero point from 430 to 600 stars. The fast stream ran at 82 frames per second and gave 40 seeing windows. ASTAP found no solution in any survey frame, because the adapter drew its star image at a quarter of the frame size. At half size, ASTAP solved the saved frames in 0.2 to 0.5 s, but no live run has used that yet, so watch the first solve of your run. The Align page found the first pointing at the first light. Read the numbers of a run as a test of the software and not as calibrated measurements. Commissioning (phase 3) chooses the exposures, the gain, and the cadence.

### Before you start

- Connect the camera to a USB 3 port, and close other camera software, because one process opens the camera at a time. Point the launcher at the vendor library with `--asi-library <path>` or with the variable `SEEINGMON_ASI__LIBRARY_PATH` (see [Windows](hardware-checks.md#windows)). The launcher gives the path to `acquire` alone and prints it nowhere.
- Install the dependencies with `uv sync --all-extras`.
- Check that Windows has synchronized its clock. The scheduler and the pointing use the system clock, and the launcher cannot tell on Windows whether it is synchronized, so it trusts it.
- Choose a data folder on a local disk, outside the repository and outside any folder that a cloud service syncs. The run keeps the store (a SQLite database), the dark library, the images, and the logs there.

### Set your site and survey

The launcher reads three tables of `local/config.toml` for `core`: `[site]`, `[survey]`, and the optional `[alignment]`. It reads `[web]` and `[auth]` for `web`, as every dev run does, and it reads nothing else: no `[sinks]`, `[heater]`, `[sqm]`, or `[power]` setting reaches the run. Copy `config/local.example.toml` to `local/config.toml` when you have no file yet, and set at least:

```toml
[site]
latitude_deg = 0.0     # your latitude in degrees, north positive
longitude_deg = 0.0    # your longitude in degrees, east positive
elevation_m = 0.0      # your elevation in meters

[survey]
catalog_path = "<path to the cap catalog file>"
solvers = ["astap"]
astap_command = "<path to the ASTAP command-line program>"
astap_database_dir = "<path to the ASTAP star database folder>"
```

The values stay in the untracked file, or in the variables `SEEINGMON_SITE__LATITUDE_DEG`, `SEEINGMON_SURVEY__CATALOG_PATH`, and so on, which beat the file. They never go into the repository. The launcher checks your tables before it writes a file or starts a child. It refuses to start, and its message names the table and the setting and never shows a value, when:

- `[site]` lacks `latitude_deg`, `longitude_deg`, or `elevation_m`, or it still holds the placeholders of the template (a latitude and a longitude of 0).
- `[survey]` sets no `catalog_path`, or the path does not name a file that reads as a cap catalog.
- `solvers` names a solver other than `astrometry.net` and `astap`, or a table has a key that does not exist or a value that does not fit.

The launcher only warns, and the run goes on, when `solvers` is empty, when it finds no program for a solver in `solvers`, when it finds no index files for astrometry.net, or when `astap_database_dir` names no folder. The run keeps two shortcuts of a dev run: windows of 20 s, and a dark session of 5 frames of each kind (set `[survey.dark]` to change the session). A real camera runs a fast period of seven windows (140 s), so that the fast period and the survey step fill the 3-minute cadence, and a seeing reading is at most about a minute old. The cloud limits stay at the production defaults, and `[survey.cloud]` changes them. The dark library is the folder `calibration` of your data folder unless `[survey]` names a `calibration_dir`, and the optional `[alignment]` table takes the aim and the target of the Align page (see the template).

### Get the catalog and a plate solver

Build the cap catalog as in [Prepare your files](#prepare-your-files), or copy a catalog that you built elsewhere, because the file works on any machine. Check it with `seeingmon catalog info <catalog file>`. The command prints the number of stars (about 82,000 for the standard cap) and the cap (15 degrees around the pole).

Install one plate solver:

- **ASTAP** (Windows, Linux, and macOS). Install the command-line program and a star database for a field of about 4 × 3 degrees (the dev machine used D05). Set `astap_command` to the program and `astap_database_dir` to the folder of the database. ASTAP reads its own database, so it needs no index, and the survey path still uses the cap catalog for everything after the solve: the fit, the matching, and the photometry.
- **astrometry.net** (Linux, and WSL for building the index). Set `index_dir` to the folder with the index files that `seeingmon catalog build` wrote, and `solve_field_command` to the program. The adapter passes Windows paths to the program, so a `wsl solve-field` command does not work from Windows. Use ASTAP there.

The default `solvers` list is `["astrometry.net", "astap"]`. On Windows, set `solvers = ["astap"]`, or each frame spends a run on a program that is not there. The adapters split a command like a shell line, so write a Windows path with forward slashes, and put a path that contains a space in double quotes inside the string, for example `astap_command = '"<folder>/astap_cli.exe"'`. A backslash disappears when the command splits, and the launcher warns about a command that it cannot find.

### Start the run

```bash
uv run seeingmon dev --driver asi --real-sky --data-dir <data folder>
```

Add `--asi-library <path>` when the variable is not set. The launcher checks your tables, starts `acquire`, `core`, and `web`, and prints the banner:

```text
Seeing monitor, real sky (asi driver): real time, full sensor.
Web UI: http://127.0.0.1:8080/
Real: the camera, the system clock, the star catalog, the plate solvers, and the site (the last three come from your local configuration). Nothing about the sky is simulated.
No pointing solution is seeded. The first survey frame goes to the plate solvers, in this order: astap.
The scheduler follows the real Sun at your site (by the clock of this machine). It stays in safe while the Sun is above -3 degrees, so by day it takes no survey frame and records no seeing window. The Align page and a dark session run in safe too.
Of your local configuration, only [site], [survey], [alignment], [web], and [auth] reach the system: no sink, heater, SQM-LE, or power setting does. As in every dev run, the windows are 20 s and a dark session takes 5 frames of each kind.
Real star images ran through the detector, the pointing tracker, and the sky quality at the first light (October 4, 2026). ASTAP has solved real frames offline only, so watch the first solve of your run.
The logs of the children are in the folder logs/20261003T184500Z of your data folder.
Cover the camera by hand for a dark session.
API token for this run (shown once, never stored): <token>
Press Ctrl+C to stop.
```

Apart from the address of the web UI, the banner prints no coordinate, no path, and no host. A line that starts with `Warning:` follows the notes when the launcher finds a problem with a solver. Fix its cause before you rely on the run, because a solver that cannot run finds no pointing solution. The token is for the commands of the Align and Dark pages. The launcher prints none when `[auth]` holds a token hash.

### Point the camera and align it

Open the **Align** page, enter the token when the page asks for it, and press **Start alignment**. The camera streams bin2 frames of 0.5 s at gain 120, and the page shows the newest one. The first quick solve has no pointing to start from, so it detects the stars and runs a plate solver, which takes a few seconds. When it succeeds, the pole card replaces "Waiting for a solution." with a sentence that tells you how to move the camera in altitude and in azimuth (it uses `[site]`), the orbit sentence says whether the circle of Polaris fits in the frame, and the **Solution and frame** card lists the matched stars and the residual. Move the mount until the pole sits on the aim, check the focus bar, and press **Stop alignment**, so that the camera goes back to measuring. In daylight the frames saturate, and the page shows a saturation warning. When the quick solve fails, the offset card gives the reason, such as too few stars for a solver or a solver that found no solution. When the pole sits on the aim and the **Pointing** card shows a solution, save the pointing reference (see [Save the pointing reference](#save-the-pointing-reference)). The text of the page may change until you approve its look (blocker B8).

### Watch the night start

| When | What happens | Where you see it |
|---|---|---|
| At the start | `core` has no pointing, logs which solvers it will try, and starts in `safe`. | `core.log`, and the **System** card (State) |
| The Sun passes -4 degrees (the gate is -3 degrees, and the scheduler resumes a degree lower) | The scheduler enters `auto`, finds no pointing, writes the warning event `scheduler.solve_requested`, and takes a survey step: a 1 ms frame (bin2, gain 0) and a 30 s frame (bin2, gain 120). | **Latest events**, and `core.log` |
| A few seconds after each frame | The analysis finds the stars, runs the solvers in order, and fits the pointing. A full bin2 frame took 2 to 3 s to analyze on the dev machine, before the time of the solver. | The `survey frame` lines of `core.log` |
| After the first solution | The scheduler starts the fast stream with the ROI on Polaris. The first seeing window closes after 20 s. | The **Seeing** card |
| Every 3 minutes | The survey step repeats. The tracker solves each frame from the last solution, so the log shows a solver run only when the tracker loses the field. | `core.log`, and the **Pointing** card |

The warning event `scheduler.solve_requested` at the start is expected, and it comes again after a lost star. It does not mean a fault. The **Pointing** card shows the roll, the matched stars, the residual of the solution, the focus value, the plate scale (3.82 arcsec per pixel in bin2), and the solver. Its large value, the offset from the reference solution, stays empty with the note "no reference solution" until you save a reference (see [Save the pointing reference](#save-the-pointing-reference)). The **Sky brightness** card needs at least 8 measurable stars for the zero point, and its records carry the `dark_due` flag until the library holds a set near the sensor temperature (see [Take a dark set from the UI](#take-a-dark-set-from-the-ui)). Transparency needs 20 clear zero points of history, so it stays empty at first. These are the expected results from the design and the synthetic tests, and the first night shows what a real sky does to them.

### Read the logs

Each child writes its log to the folder `logs/<start time>` of your data folder. The banner names the folder relative to your data folder, and the start time is UTC, such as `20261003T184500Z`. The files are `acquire.log`, `core.log`, and `web.log`. The run logs at the level `info`, and `--log-level warning` shows less. To follow one log in PowerShell:

```powershell
Get-Content "<data folder>\logs\<start time>\core.log" -Wait -Tail 20
```

The time at the start of a line is the local time of this machine. A `survey frame` line names the frame by its UTC time, which ends in `Z`. The first lines of `core.log` after a start with no pointing look like this (the numbers are made up):

```text
2026-10-03 21:45:03,154 INFO seeingmon.services.core.app: no pointing solution yet: the survey frames go to the plate solvers astap, in this order, until one solves
2026-10-03 21:46:11,402 INFO seeingmon.survey: survey frame 2026-10-03T18:45:48Z: only 3 stars for a solver
2026-10-03 21:46:11,403 INFO seeingmon.survey: survey frame 2026-10-03T18:45:48Z: 0.001 s bin2: 3 stars detected, not solved, analysis took 0.9 s
2026-10-03 21:46:52,118 INFO seeingmon.survey: survey frame 2026-10-03T18:45:50Z: solver=astap result=solved stars=312 time_s=0.79 matched=214
2026-10-03 21:46:52,119 INFO seeingmon.survey: survey frame 2026-10-03T18:45:50Z: 30 s bin2: 412 stars detected, solved by astap (214 matched), analysis took 4.1 s
```

Each survey frame gets one line with its outcome, and each run of a solver gets one line before it:

| Field | Meaning |
|---|---|
| `solver=` | The solver that ran: `astrometry.net` or `astap`. |
| `result=` | `solved`: the solver found a field, and the fit confirmed it. `no_solution`: the solver ran and found no field. `rejected`: the solver found a field that the catalog or the fit could not confirm. `error`: the program could not run. |
| `stars=` | The number of stars that went to the solver: the brightest detections without the hot pixels, at most 600 (`[survey.solve] max_stars`). |
| `time_s=` | The time of the solver run, in seconds. |
| `matched=` | The stars that the fit paired with catalog stars. It appears for a solved run. |
| `reason=` | Why a run failed. It appears for every other result. |

A frame that the tracker solves from the last solution has no solver line, and its outcome line says `solved by tracker`. A line holds no coordinate. Retention does not manage the `logs` folder, so delete old run folders by hand. A log can name folders of your machine, so keep it out of the repository.

### Stop the run

Press Ctrl+C in the console. The launcher prints `Stopping ...`, stops `web`, `core`, and `acquire` in that order, which can take up to 25 s for each, and then names the folder of the logs. The data folder and the logs stay. If a child exits on its own, the launcher ends the run, prints the last lines of that child's log, and stops the others.

### When something fails

| Symptom | Likely cause | Check and fix |
|---|---|---|
| The launcher refuses to start and names `[site]` or `[survey]`. | A value is missing or does not fit. | Read the message: it names the table and the setting. Fix the local configuration or the variable. |
| A `Warning:` line says that the machine finds no program for a solver. | The command is not on the PATH, or a backslash or a space broke it. | Give the path with forward slashes, put a path that contains a space in double quotes, and start again. |
| Every solver line says `result=error`. | The program cannot run: it is not installed, it crashes, ASTAP finds no star database, or it hangs. | Read the `reason=` text. Run the command by hand. A hang shows as `did not finish within 25 s`, which is `[survey.solve] timeout_s` (20 s) plus 5 s of grace. |
| Every 30 s frame says `result=no_solution` with a few hundred stars. | The solver does not match the field: the camera does not point at Polaris within the 15 degree cap, or clouds cover the sky. | Open the Align page and look at the frame. |
| A frame says `result=rejected`. | The solver found a field that the catalog or the fit rejects, for example because the catalog is not the cap around the pole or the plate scale differs. | Run `seeingmon catalog info <catalog file>`, and compare the plate scale on the Pointing card with 3.82 arcsec per pixel in bin2. |
| The frames show only "only N stars for a solver". | The detector finds fewer than 4 stars: a cover on the camera, clouds, or a bad focus. The 1 ms frame of each survey step shows this line by design, because it holds only the brightest stars. | Look at the 30 s frame, at the frame on the Align page, and at the focus bar. |
| The state stays `safe`, and no survey frame runs. | The Sun is above -3 degrees at your `[site]`, or the measured sky brightness gate holds the scheduler. | Check the **System** card and the latest events. Check that the values of `[site]` are your own. |
| The launcher ends with `acquire exited` and the end of its log. | The vendor library or the camera is not available. | Check `--asi-library` and `SEEINGMON_ASI__LIBRARY_PATH`, close other camera software, and see [Windows](hardware-checks.md#windows). |
| The **Pointing** card still says "no reference solution" after you saved a reference. | You did not restart the run, or `[survey.pointing] reference_file` names no file, which `core` ignores without an error. | Run `seeingmon pointing show`, fix the path, and start the run again (see [Save the pointing reference](#save-the-pointing-reference)). |

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
| The log says `no frame within` some seconds for a survey or brightness exposure, and the `camera` component reads `degraded` for a while. | A single exposure takes longer than the wait. The scheduler and `acquire` wait twice the time that the driver expects, plus `read_timeout_margin_s` (0.5 s), and the driver takes the expected time from the snapshot model of the profile (`snapshot_overhead_s` and `snapshot_row_time_us` of the survey readout mode: 0.27 s plus 75 µs for each row in bin2, measured on a Pi 4). A slower host, USB port, or SDK exceeds it, and so does a profile for other hardware that states no snapshot values (the model then assumes at least 0.3 s). | Stop the services, and run `seeingmon camera rates --groups snapshot` (see [Measure single exposures](hardware-checks.md#measure-single-exposures)). Put the fitted values in the profile. Until then, raise `read_timeout_margin_s` under `[scheduler.loop]` and `[services.acquire]` of the local configuration, for example to 2.0 as the first Pi 4 run did before the profile had the model. |
| A unit grows past its `MemoryMax`, and nothing stops it. Or the installer warns that the kernel has no memory cgroup. | The kernel of Raspberry Pi OS has the memory controller off, so systemd does not apply `MemoryMax`. A running unit shows `MemoryCurrent=[not set]`. | `cat /sys/fs/cgroup/cgroup.controllers` must list `memory`. If it does not, add `cgroup_enable=memory cgroup_memory=1` to the single line of `/boot/firmware/cmdline.txt`, and reboot (see the memory cgroup step of [Prepare the Pi](#prepare-the-pi)). |
| Records carry `time_invalid`. | chrony has no synchronized source. | See [Time sync](#time-sync). |
| The health endpoint answers 503. | A component failed, or `core` wrote no health record for three minutes. | Read the `reasons` in the answer, then `systemctl status seeingmon.target` and the log of `core`. |
| The UI is not reachable from the LAN. | `bind_address` is still the loopback address. | Set the LAN address in `[web]` of the local configuration, and run the installer again. |
| Commands over the API are refused. | `web` has no token hash. | Run `seeingmon web hash-token`, and install the hash with `--token-hash-file`. |
| The data directory warns about the root file system. | No data partition. | See [Create the data partition](#create-the-data-partition). |
| A burst fails because raw capture stopped, and the store wrote `retention.capture_stopped`. | Less than 1 GB of free space. | `df -h <data-dir>`. Free space, or unpin old bursts by deleting the `PINNED` file in their folders under `<data-dir>/bursts/`. |
| The Images page is empty. | `core` writes the first preview after the first long exposure (30 s) of a survey step, and survey steps run only while the sky is dark. `[services.core.survey_frames]` may be off. | Check the state on the **Now** page, then `ls <data-dir>/previews/*/*/*`, and look for `survey_images.write_failed` events (`GET /api/v1/events?kind=survey_images.write_failed`) and for a full disk (`df -h <data-dir>`). |
| `journalctl` shows nothing from before the last boot. | The journal lives in RAM. | Expected. The `event` table keeps the events that matter. |
| The `sqm` component of `health` is `degraded`, and the event `sqm.read_failed` names a cause. | The SQM-LE reader gets no fresh reading: the point in InfluxDB is stale, the server does not answer or refuses the token, or the unit does not answer over TCP. | See [When the readings stop](#when-the-readings-stop) for each cause, and run `seeingmon hardware sqm` to try the settings. |
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
