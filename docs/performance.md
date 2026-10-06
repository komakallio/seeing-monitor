# Performance

The architecture sets a performance gate for a Raspberry Pi 4 (see [architecture.md](architecture.md), "Processes, data rates, and storage"). This page describes the harness that measures the gate on a development machine, the results of a run, the Pi 4 estimate that follows from them, and the commands that measure it on a Pi 4. A Raspberry Pi 4 with 2 GB ran the harness and the installed system on October 3, 2026, and [Results on a Raspberry Pi 4](#results-on-a-raspberry-pi-4) gives the measured figures. The Pi 4 columns of the older tables stay as the estimate that this run replaced.

## Summary

**Measured on a Pi 4 with 2 GB (October 3, 2026): every budget of that run passes.** The sum of the peaks of all processes is 895 MB against the budget of 1.4 GB. The installed system with the real camera used at most 1,053 MiB of the 1,844 MiB, and a 45-minute simulated run at most 1,155 MiB, so 2 GB of RAM is enough (see [Results on a Raspberry Pi 4](#results-on-a-raspberry-pi-4)). The rows that the visibility lane added on October 6 have no Pi 4 run yet. On the dev machine the current code costs `core` twice what the code of the Pi 4 run cost for a frame, and the search burst fails its budget on the estimate (see the last bullet below). Apart from that bullet and [The cost of measuring all day](#the-cost-of-measuring-all-day), the rest of this summary and the tables below show the estimate from the dev machine that the October 3 run replaced.

On the estimate before the Pi 4 run, the per-frame analysis fits its budget with a wide margin, and so does the receive in `core` since the services lane batched the stream. `acquire` is the one row of the CPU budgets that fails in all six runs. The memory is the open question: the peaks that the whole-system run measures put the estimate across the 1.4 GB budget.

| Budget | Limit | Dev machine | Pi 4 (estimate) | Verdict | Verdict in the six runs |
|---|---|---|---|---|---|
| `acquire`, the production code, bin1 at 98 fps | 10% of a core | 3.4% | 12 to 24% | fail | Fail in six |
| `acquire`, what the fake camera of the tests adds | Not counted | 0.03% | | | |
| Fast path, bin1 at 98 fps | 25% of a core | 0.28% | 1.4 to 3.1% | pass | Pass in six |
| Fast path and receive, bin1 at 98 fps | 25% of a core | 1.2% | 7.5 to 13% | pass | Pass in six |
| Fast path, bin2 at 360 fps | 25% of a core | 0.92% | 4.6 to 10% | pass | Pass in four, marginal in two |
| Fast path and receive, bin2 at 360 fps | 25% of a core | 1.9% | 12 to 21% | pass | Pass in two, marginal in two, fail in two |
| Survey frame, bin2 (derived, not a gate) | 180 s | 0.98 s | 4.9 to 11 s | pass | Pass in six |
| Survey worker, peak memory | 550 MB | 452 MB | 317 to 588 MB | marginal | Marginal in six |
| All processes, peak memory | 1.4 GB budget, 1.6 GB gate | 766 MB | 686 to 1,295 MB | pass | Pass in six |

The table shows the least disturbed of three Linux runs at commit `4afaca4`, after the services lane cut the work of the stream (see [Stream between `acquire` and `core`](#stream-between-acquire-and-core)). The last column gives the verdicts of all six runs: three on Linux, and three on Windows, which is slower in the bin2 fast path and swings by about 30% from run to run, because its CPU time ticks at 15.6 ms. The Pi 4 column is an estimate: read [How the estimate works](#how-the-estimate-works) before you rely on it. The first version of this page showed the figures of commit `0a764af`, before the change: `acquire` at 6.8% on the dev machine (18 to 40% on the estimate), the fast path with the receive at 4.3% in bin1 (19 to 35%, `marginal` or `fail`) and at 14.4% in bin2 (99 to 158%, `fail`).

The `core-sim` case (see [The whole-system case](#the-whole-system-case)) measures `core`, the survey worker, and `web` in the running system. It ran once on Linux and once on Windows at commit `b2e1f93`, which has the same code as `4afaca4`. Another system ran on the machine during both runs (see [The whole system](#the-whole-system)), so these are runs with another system active. The verdicts in the table above stay until a run on a quiet machine replaces them. The table below shows what the new figures give. The Pi 4 column uses the Linux figure.

| Budget | Limit | Table above | Whole system, Linux / Windows | Pi 4 (estimate) | Verdict, Linux / Windows |
|---|---|---|---|---|---|
| Fast path and receive, bin1 at 98 fps, measured in `core` | 25% of a core | 1.2% | 1.9% / 2.5% | 13 to 21% | pass / marginal |
| Survey worker, peak memory | 550 MB | 452 MB | 444 / 414 MB | 311 to 577 MB | marginal / pass |
| All processes, peak memory, `acquire` from the `ipc` case | 1.4 GB budget, 1.6 GB gate | 766 MB | 905 / 775 MB | 783 to 1,476 MB | marginal, pass / pass, pass |
| All processes, sum of the peaks in the whole system, with the simulator in `acquire` | 1.4 GB budget, 1.6 GB gate | | 1,143 / 966 MB | 950 to 1,786 MB | marginal, marginal / marginal, pass |

- **The fast path has room.** One bin1 frame takes 26 µs through `FastPathAnalyzer` on the dev machine, and the kernel takes 19 µs of that. The Pi 4 estimate for the kernel (0.10 to 0.21 ms) agrees with the architecture's 0.2 to 0.4 ms. These runs predate the matched filter of the missing-star test (visibility lane), which adds 11 to 13 µs to the kernel on the dev machine (`<mode>.matched_extra` of the `kernel` case): 0.5 to 1.4% of a Pi 4 core in bin1 at 98 fps. See [The cost of measuring all day](#the-cost-of-measuring-all-day) for it, the search frame, and the weighted centroid.
- **The stream between the processes was the risk, and it has changed.** At commit `0a764af`, the production code of `acquire` spent 0.69 ms of CPU on a frame, and `core` spent 0.41 ms on receiving it. At 360 frames per second, the receive alone cost an estimated 93 to 146% of a Pi 4 core. The costs belonged to each message and each frame, not to the bytes. The services lane then cut the per-message work (commits `0db406c` to `80acdcc`): `core` asks for batches, and `acquire` holds a fast stream for 60 ms and sends its frames as one message, so a batch costs one wake-up. Now `acquire` spends 0.35 ms of CPU on a frame (0.32 to 0.37 ms over the three Linux runs), and `core` spends 0.09 ms on the receive (0.07 to 0.11 ms). The receive in bin2 at 360 frames per second takes 28 µs a frame, 1.0% of a core. The estimate for `acquire` is 12 to 24% of a Pi 4 core, still over its budget of 10%: the capture thread takes 235 of the 348 µs, and the work alone gives 10 to 16%. The receive with the fast path is within the 25% budget: 7.5 to 13% in bin1 and 12 to 21% in bin2 on the estimate, and 13 to 21% in bin1 in the whole system. The camera is not the cause of the `acquire` figure: the fake camera of the tests adds about 3 µs to a call (see [What the camera adds](#what-the-camera-adds)).
- **The survey path has room in time and little in memory.** A bin2 frame takes 1.0 s on the dev machine (1.0 to 1.3 s over the three Linux runs), including 0.31 s for the sky quality step, and the worker peaks at 452 MB in the `survey` case and at 444 MB in the whole system.
- **Two gigabytes of memory was not settled on the estimate, and the Pi 4 run settled it (see [Results on a Raspberry Pi 4](#results-on-a-raspberry-pi-4)).** The estimate with a stand-in of 200 MB for `core` passes both limits. In the whole system, `core` peaks at 307 MB on Linux, and the estimated peak of all processes is 783 to 1,476 MB, which straddles the 1.4 GB budget and stays under the 1.6 GB gate. The range crosses the gate too when `acquire` counts with the simulator inside it (950 to 1,786 MB), and the true figure for `acquire` lies between the two. A run on a Pi 4 decides it (see [Memory](#memory)).
- **Rust for the per-frame metrics is not indicated** (see [The Rust decision](#the-rust-decision)).
- **Measuring all day costs what a clear night costs, a frame of the night costs `core` twice what the Pi 4 measured, and a search burst fails its budget on the estimate** (see [The cost of measuring all day](#the-cost-of-measuring-all-day), October 6, 2026). On the same laptop, the current code costs `core` 2.1 to 2.4 times what the code of the Pi 4 run cost for a frame, which puts a clear night at an estimated 17 to 19% of a Pi 4 core against 25%, where the Pi 4 measured 8%. A frame of daylight costs `core` 0.92 to 1.06 times a frame of the night, so daylight takes 16 to 20%, and the peaks of all processes stay within 8% of the night's, about 0.9 GB. Over whole cycles, `core` uses about 12% of a Pi 4 core by day and 7% under clouds. The peak is the search burst: a search frame costs `core` 2.4 to 5.3 times a frame of measure, so a burst takes an estimated 45 to 50% of a Pi 4 core while it runs (40 to 100% over three runs), against the 25% of the fast path. The simulator's sensor cannot warm in the sun, so the sensor temperature of a sunlit camera, and its effect on the dark library, wait for phase 3.

## What the harness measures

`seeingmon perf run` runs ten cases. Each case runs in a fresh child process, so the peak memory of one case never includes another. The child sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` to 1, so that every figure is per core, which is how the budgets read. A case imports the code that it measures when it runs, and it skips with a reason when a component is not on `main` yet, so the harness works at every commit.

| Case | What it measures | Feeds |
|---|---|---|
| `calibration` | Five fixed workloads: a pure-Python loop, a matrix product, an FFT, a SQLite insert batch, and a thread hand-off. The ratio of a Pi 4 run to a dev-machine run on them replaces the assumed scaling. | The scaling table |
| `kernel` | One call of the fast-path kernel in three modes: bin1 128 × 128 at 16 bits (the planned fast mode), bin2 64 × 64 at 16 bits, and bin2 320 × 240 at 8 bits (the format of the 10 ms recordings). One frame of a search burst in each mode, the three matched filters within 20 px of the ROI center (`<mode>.search`). The kernel with the Gaussian-weighted centroid, which is not the default (`<mode>.kernel_gaussian`), and the kernel without the matched filter of its missing-star test (`<mode>.kernel_without_matched`). Two differences of the medians: what the matched filter adds to a frame of measure (`<mode>.matched_extra`), and what the weighted centroid adds (`<mode>.gaussian_extra`). | The per-frame budgets, the search rows, and the rows of the weighted centroid |
| `fastpath` | The whole per-frame path of `FastPathAnalyzer`: the kernel, the metrics row, the window bookkeeping, the segment append once per second, and the close of a 60 s window (the estimator, the scintillation index, and the spectrum), as a share of one core at the frame rate. | The 25% budget |
| `ipc` | The cost of moving a frame from `acquire` to `core`: the production `AcquireService` in its own process with a camera that costs nothing per frame, and a `RemoteCameraDriver` that reads the frames. The case reads the CPU time per frame of each side, split by thread, and splits it again into the work and the wake-ups. It also runs the bin2 stream at 360 fps, a stream with the fake camera of the tests, and a timing of one `read_frame` call of three cameras, which show what the camera adds. | The 10% budget of `acquire`, and the receive cost in the 25% budgets |
| `survey` | One synthetic bin2 survey frame (4144 × 2822 pixels, 30 s, rendered by the simulator from catalog stars) through `create_survey_analyzer` and the process worker of `make_process_executor`, with the sky quality step and one synthetic dark set: the wall time, the CPU time of the worker, each stage, the cost of the process boundary, the start of the worker, and the peak memory of the worker. | The survey budgets |
| `store` | Sustained result-row inserts, and the cost of the metrics-segment append per frame at 98 fps, in a temporary folder that the case deletes. | The 25% budget |
| `memory` | The resident size of a fresh process after it imports each part of the software: the fast path, the survey path, astropy, and the web stack. | The memory budgets |
| `core-sim` | The whole system on the simulated sky: `acquire` with the simulator, `core`, `web`, and the survey worker, started from the plan of `seeingmon dev` and read from outside for about 6 minutes (see [The whole-system case](#the-whole-system-case)). | The fast path with the receive in `core`, and the memory budgets |
| `day-sim` | The whole system on a simulated day of measuring: noon of midsummer, Polaris visible, the adaptive fast exposure, and the survey steps that skip their long exposure, in the cycle of production, for about 6 minutes (see [The cost of measuring all day](#the-cost-of-measuring-all-day)). | The rows of daylight |
| `cloudy-sim` | The whole system on a simulated cloudy night of searching: an opaque overcast, the search bursts every 15 s, and the survey steps every 100 s, for about 6 minutes (see [The cost of measuring all day](#the-cost-of-measuring-all-day)). | The search rows and the rows of a cloudy night |

The sections below name each figure the way a report does (`<case>: <figure>`).

## Budgets

The Pi 4 budget comes from [architecture.md](architecture.md). The harness adds three rows that the text implies, and it marks a derived row as not a gate.

| Budget | Limit | Where it comes from |
|---|---|---|
| `acquire`, bin1 128 × 128 at 98 fps | 10% of one core | The architecture. The harness counts the code of `acquire` and leaves the camera out, so the figure is a lower bound. |
| Fast path, bin1 128 × 128 at 98 fps | 25% of one core, 2.55 ms per frame | The architecture |
| Fast path and the receive from `acquire`, bin1 | 25% of one core | The same consumer in `core` pays for both. The row reads the CPU time that `core` spends on a frame in the whole system (the `core-sim` case), and it falls back to the sum of the `fastpath` and `ipc` figures when that case did not run. |
| Fast path, bin2 64 × 64 at 360 fps | 25% of one core, 0.69 ms per frame | The architecture |
| Fast path and the receive from `acquire`, bin2 | 25% of one core | The same consumer in `core` pays for both |
| Survey frame, bin2 | 180 s (not a gate) | The survey interval: a frame must end before the next one |
| Survey worker, peak memory | 550 MB | The architecture |
| All processes, peak memory | 1.4 GB (the budget), 1.6 GB (the 2 GB gate) | The architecture: more than 1.6 GB at peak means the 4 GB model |
| Fast path and the receive in daylight, bin1 | 25% of one core | The fast path budget, read in `core` in the `day-sim` case |
| All processes in daylight, peak memory | 1.4 GB, 1.6 GB | The memory budget, with the peaks of the `day-sim` case |
| Search burst and the receive, bin1 128 × 128 at 98 fps | 25% of one core while the burst runs | The fast path budget: a burst reads the frames of the fast stream at its rate. The row reads the cost of a search frame that the `cloudy-sim` case measures in `core`, and without that case the search frame of the `kernel` case with the receive of the `ipc` case. |
| Search burst and the receive, bin2 64 × 64 at 360 fps | 25% of one core while the burst runs | The same, from the `kernel` and `ipc` cases |
| All processes on a cloudy night, peak memory | 1.4 GB, 1.6 GB | The memory budget, with the peaks of the `cloudy-sim` case |
| Fast path with the Gaussian-weighted centroid, bin1 and bin2 | 25% of one core (information, not a gate) | The fast path rows plus what the weighted centroid adds to a frame (`<mode>.gaussian_extra`). The centroid is not the default, and the rows inform the owner's decision on it. |

The harness sums the figures of each row. It reports `pass` when the whole estimated range is within the limit, `fail` when the whole range is above it, and `marginal` when the range straddles it. A report with the label `pi4` and a measured calibration case compares its figures with the limits directly, and the verdict is `pass` or `fail`.

## Run the harness

Run the commands on your development machine, from a clone with the extras installed (`uv sync --all-extras`, see [development.md](development.md)).

```bash
seeingmon perf run --smoke                                  # about 2 minutes: every case works
seeingmon perf run --label dev --json local/perf/dev.json  # the full run: about 20 minutes
seeingmon perf report local/perf/dev.json --budgets         # the tables, the verdicts, and the checks
```

- `--cases NAME,...` runs some cases, and `--list` names them.
- `--smoke` shrinks every case to a tiny workload, and the figures then say nothing about speed. The cases `core-sim`, `day-sim`, and `cloudy-sim` still start their three processes, and each waits for the first search bursts, so each takes about 20 s of sampling and 35 to 50 s in all. CI runs the smoke run as a test (`tests/perf/test_cases.py`). The test checks that every case runs, that its figures are finite and positive, and that the budgets find the figures that they read. It checks nothing about size.
- `--cases core-sim` runs the whole-system case alone, which takes about 6 minutes (see [The whole-system case](#the-whole-system-case)). `--cases day-sim,cloudy-sim` runs the day and the cloudy night, about 6 minutes each (see [The cost of measuring all day](#the-cost-of-measuring-all-day)).
- `--json PATH` saves the report. Use a path under `local/`, which Git ignores. The report holds the architecture, the operating-system family, the Python and library versions, the processor model, and the commit. It holds no host name, user name, or serial number.
- `--label` names the machine class. Use `pi4` on a Raspberry Pi 4.
- `--quiet-wait SECONDS` waits up to that long before each case for the machine to be at most 15% busy. Other work disturbs a timing, and each case records how busy the machine was.
- `seeingmon perf report A.json --baseline B.json` prints the ratio of every figure of A to the same figure of B, and whether the ratio lies in the range that the scaling table assumes. Use it to compare a Pi 4 run with a dev run.
- `seeingmon perf report A.json --details` also prints the details of each figure, such as the number of stars in the survey frame.

A figure is the median of its repeats, and the table also shows the minimum, the 95th percentile, and the maximum. A busy machine moves the median up and the maximum far up, so read the line `machine N% busy` of each case. The minimum is the least disturbed figure.

## Results on the development machine

The tables show two sets of runs on the same laptop, a recent x86-64 machine with performance and efficiency cores. One set ran Linux in a WSL 2 virtual machine (CPython 3.12.14, NumPy 2.5.3, SciPy 1.18.1), and the other ran Windows (CPython 3.13.13, the same libraries). Each set has three runs of the cases `calibration`, `kernel`, `fastpath`, `ipc`, `survey`, `store`, and `memory` at commit `4afaca4`, after the services lane cut the work of the stream. The runs alternated between the two systems, and each case waited up to 30 s for the machine to be at most 15% busy (`--quiet-wait 30`). The first version of this page had three runs of each set at commit `0a764af`, before that change, and the stream section keeps those figures as the "before". The last subsection shows the whole-system case at commit `b2e1f93`. A cell that holds two numbers reads `Linux / Windows`. The Pi runs Linux, so the estimate uses the Linux run.

The laptop did other work during the runs, and the scheduler moves a process between fast and slow cores, so a figure changes from run to run. A disturbance only adds time. The reports of the Windows runs say `machine 10% busy` to `machine 13% busy`. The reports of the Linux runs say `0% busy`, because a virtual machine cannot see the load of its host. The host was about as busy during the Linux runs as during the Windows runs: the processor utility counter of Windows showed a mean of 45 to 52% during the first and 42 to 46% during the second, a figure that scales with the clock rate and reads higher than the time-based figure of a report. The CPU time of Windows ticks at 15.6 ms, so its figures swing by about 30% from run to run, and the Windows columns need the repeats. For these reasons, each table shows the least disturbed run of each set, which is the run with the lowest sum of its key figures divided by the lowest figure of the three. The last tables show all three runs of each set.

### Fast path, per frame

| Mode | Kernel (µs) | Whole push (µs) | Close a 60 s window (ms) | Share of a core |
|---|---|---|---|---|
| bin1 128 × 128, 16 bit, 98 fps | 19.1 / 21.7 | 26.1 / 33.8 | 8.06 / 23.0 | 0.28% / 0.39% |
| bin2 64 × 64, 16 bit, 360 fps | 15.5 / 18.8 | 23.7 / 30.0 | 23.3 / 48.7 | 0.92% / 1.19% |
| bin2 320 × 240, 8 bit, 98 fps | 18.8 / 22.1 | 26.7 / 30.2 | 8.25 / 7.86 | 0.29% / 0.33% |

The push is one call of `FastPathAnalyzer.push`: the kernel, the metrics row, and the window bookkeeping. The share of a core adds the window close and the segment append to it, at the frame rate. The push is about 90% of the share.

### Stream between `acquire` and `core`

The services lane changed the stream between the two processes in the commits `0db406c` to `80acdcc`: `core` asks for batches in the hello of the stream, `acquire` holds a fast stream for up to 60 ms and sends the frames as one message, the receiver reads several frames from one message, and the code on both sides does less for each message. The table shows the figures before (commit `0a764af`) and after (commit `4afaca4`).

| Figure | `acquire`, before | `acquire`, after | `core` receive, before | `core` receive, after |
|---|---|---|---|---|
| CPU per frame, paced at 98 fps (µs) | 691 / 1,076 | 348 / 514 | 405 / 604 | 89 / 173 |
| CPU per frame, bursts of 10 frames (µs) | 238 / 924 | 169 / 241 | 239 / 576 | 100 / 81 |
| Share of a core at 98 fps, work | 1.84% / 8.89% | 1.46% / 2.06% | 2.17% / 5.61% | 0.87% / 0.70% |
| Share of a core at 98 fps, wake-ups | 4.94% / 1.66% | 1.95% / 2.97% | 1.80% / 0.31% | 0% / 1.00% |
| Share of a core at 98 fps, total | 6.78% / 10.5% | 3.41% / 5.03% | 3.97% / 5.92% | 0.87% / 1.70% |

`acquire` runs with a camera that costs nothing per frame: it waits for the frame period and returns a frame that exists already. So the table counts the code of `acquire` and leaves the camera out. The case splits the work from the wake-ups with two runs: a paced stream costs the work plus a wake-up for each frame, and a stream in bursts of 10 frames costs the work plus a tenth of the wake-ups. The Linux virtual machine makes wake-ups expensive, and the Windows figures come from a coarse CPU clock, so the split is one of the least reliable figures on this page. After the change, the split shows that the receive no longer pays a wake-up for each frame on Linux: a paced stream costs `core` 89 µs a frame and a stream in bursts 100 µs, because `acquire` sends the frames of the last 60 ms as one message, and one message and one wake-up serve about six frames. The wake-up share of the receive is therefore zero in the Linux column.

On Linux, `acquire` spends 235 µs per frame in the capture thread, 93 µs in the sender thread, and 21 µs elsewhere, and it peaks at 58 MB. Before the change, the threads took 327, 334, and 30 µs. The capture thread is now 68% of `acquire`. A bin2 stream at 360 fps costs `core` 28 µs per frame to receive (1.0% of a core on Linux, 2.7% on Windows), against 369 µs (13.3% on Linux, 26.1% on Windows) before.

### What the camera adds

The services tests use a fake camera that builds a `Frame` on every read, and the real ASI driver builds one too, so the question is how much a camera adds to the capture thread. The first six rows below come from whole runs. The sender thread is a control, because it runs the same code whatever the camera is, so a difference between the two runs that shows in the sender thread is drift of the laptop, and not the camera.

| Figure (µs per frame) | Linux / Windows |
|---|---|
| Capture thread, zero-cost camera | 235 / 300 |
| Capture thread, fake camera as it is | 268 / 258 |
| Sender thread, zero-cost camera (control) | 93 / 171 |
| Sender thread, fake camera as it is (control) | 93 / 128 |
| `acquire` as a whole, zero-cost camera | 348 / 514 |
| `acquire` as a whole, fake camera as it is | 387 / 468 |
| One `read_frame` call, fake camera | 2.7 / 5.1 |
| One `read_frame` call, zero-cost camera | 0.085 / 0.17 |
| One `read_frame` call, production ASI driver on a stub SDK | 10.0 / 21.1 |

On Linux, the fake camera adds 33 µs to the capture thread in the paced run, and the control did not move (93 and 93 µs), so the difference is not drift. The three rows at the end of the table time one `read_frame` call in a loop with no waiting, so they have none of the noise of a run. The fake camera costs 2.7 µs a call, so the harness does not break the other 30 µs down. They do not count against the production code, because the verdict on `acquire` rests on the zero-cost runs. On Windows, the fake runs came out lower than the zero-cost runs (258 against 300 µs in the capture thread, and 128 against 171 µs in the control), which shows the swing of the 15.6 ms tick. The production ASI driver costs 10 µs a call on Linux and 21 µs on Windows with a stub SDK: it takes a lock, arms the call watchdog, copies the pixels, and builds a `Frame` with its checks, and a real driver adds the vendor library and the USB stack on top. The services lane found that about 10 µs of each frame is the arm and disarm of the `CallWatchdog` around the SDK calls of the ASI driver. The hardware lane owns that driver.

So the camera is a small part of the 235 µs of the capture thread: the stub-SDK read of the production driver is 4% of it. The capture code of `acquire` owns the rest: the gate, the guard, the time stamper, the queue, and the bookkeeping. The verdict on `acquire` rests on the zero-cost runs, which count what the code of `acquire` costs and leave the camera out. A real driver adds 0.10 to 0.21% of a core on the dev machine (10 to 21 µs at 98 fps), so the figure is a lower bound. The sender thread and the receive path of `core` do not depend on the camera, so their verdicts stand.

### Where the stream spends its CPU time

Before the services lane changed the stream, the harness profiled the sender thread and the receive path, so that the lane had a starting point. The profile below is the one of commit `0a764af`: 2,000 frames at 98 fps on Linux (CPython 3.12). It describes the code before the change, and the shares do not hold for the current code. It uses `yappi` with the CPU clock, because `cProfile` of Python 3.12 and later allows one profile for the whole process and cannot tell the threads apart. `acquire` ran with the zero-cost camera, and `core` was a `RemoteCameraDriver` that reads the frames. The profile is one-off, and it is not part of the harness. A profiler adds overhead, so read the shares and not the microseconds. A share includes the functions that the function calls, so the rows of a table overlap.

**Sender thread of `acquire`**

| Function | Calls per frame | Share of the thread |
|---|---|---|
| `StreamSender.has_credit`, which reads the acknowledgements of `core` | 2 | 52% |
| `Wire.recv` | 3 | 44% |
| `Connection.poll` of `multiprocessing`, which builds a selector on every call | 3 | 29% |
| `AcquireService._send_item`, which encodes (6%) and sends the frame | 1 | 36% |
| `FrameQueue.peek`, which waits for the next frame | 1 | 18% |

**Capture thread of `acquire`**

| Function | Calls per frame | Share of the thread |
|---|---|---|
| `AcquireService._on_frame` | 1 | 56% |
| `FrameQueue.put_frame`, which wakes the sender (12%) | 1 | 15% |
| The camera read, which is mostly the wait (14%) | 1 | 15% |
| `TimeStamper.stamp` | 1 | 14% |
| `dataclasses.replace` for the stamped frame, with the checks of `Frame` (4%) | 1 | 11% |

**Stream reader thread of `core`**

| Function | Calls per frame | Share of the thread |
|---|---|---|
| `Wire.recv` | 1 | 70% |
| `Connection.poll` of `multiprocessing` | 1 | 46% |
| `Connection.recv_bytes` | 1 | 19% |
| `Condition.notify_all`, which wakes the consumer | 1 | 15% |
| `decode_message` | 1 | 7% |

**Consumer thread of `core`**, the thread that calls `read_frame`

| Function | Calls per frame | Share of the thread |
|---|---|---|
| `StreamReceiver.recv` | 1 | 57% |
| `decode_frame` | 1 | 34% |
| `Condition.wait` | 1 | 25% |
| `Wire.send`, the acknowledgement of each frame | 1 | 15% |
| `Frame.__init__` | 1 | 7% |

The costs belonged to each message and each frame, not to the bytes. The profile suggested three changes: read the acknowledgements once per batch and build the selector once, acknowledge every few frames instead of every frame, and send several frames in one message. The services lane made changes of that kind (see [Stream between `acquire` and `core`](#stream-between-acquire-and-core)). At commit `0a764af`, the estimate said that the 10% budget of `acquire` needed 1.8 to 4 times less CPU per frame than it took, and the fast path with the receive needed up to 1.4 times less in bin1 and 4 to 6 times less in bin2 at 360 fps. After the change, `acquire` needs 1.2 to 2.4 times less, and the fast path with the receive fits.

### Survey frame

| Figure | Linux / Windows |
|---|---|
| Frame through the analyzer and the worker (s) | 0.979 / 2.04 |
| Stage `detect` (s) | 0.485 / 1.69 |
| Stage `solve`, by the tracker (s) | 0.00667 / 0.0211 |
| Stage `match` (s) | 0.00127 / 0.00319 |
| Stage `quality`, the sky quality step (s) | 0.307 / 0.623 |
| Stage `records` (s) | 0.00073 / 0.00141 |
| Process boundary, the job minus its stages (s) | 0.038 / 0.046 |
| CPU time of the worker for a frame (s) | 0.96 / 1.88 |
| Start of the worker: spawn, imports, catalog (s) | 0.876 / 2.53 |
| Worker memory after the start (MB) | 89.6 / 88.2 |
| Worker peak memory (MB) | 452 / 430 |

The frame has 4144 × 2822 pixels, 1,888 detections, and 1,394 stars that match the catalog, and the zero point uses 904 of them. The stages come from three separate jobs and the total from five frames, so on a noisy run they do not add up to the total. The peak of the worker was 444 MB on Linux before the case ran the sky quality step, so the step adds about 10 MB: the detection step already holds the largest arrays. The `solve` stage is the tracker, because the case gives the pipeline the solution of a previous frame.

**Since these runs.** The detector batches its star mask, its bounding-box counts, and its blend test, and it has the binned search (`[survey.detect] coarse_bin`, see "Coarse search" in [architecture.md](architecture.md)). A real 30 s bin2 frame with 3,000 stars, the case that a synthetic frame with 1,888 detections understates, showed what they save. The star mask takes 0.05 s instead of 0.3 s, and the blend test 0.007 s instead of 0.3 to 4 s, because the old time grew with the size of the largest saturated star. The binned search cuts the `detect` stage of that frame to about a quarter (0.9 to 1.1 s instead of 3.9 to 5.5 s of CPU time on a busy laptop). The table above keeps the figures that were measured before the change. The default is the binned search (`coarse_bin = 2`): against the real catalog it matches the full search to 0.022 pixel and 0.0034 mag.

**The full search at dusk.** A long frame takes the full search when the binned search would miss stars that the cloud fraction expects (see "Coarse search" in [architecture.md](architecture.md), and `seeingmon.survey.completeness`). A frame that takes it runs the binned search first, because the choice needs the noise, the star image, and the share of the light in the fitted image that the binned search measures. The table shows the cost on the full bin2 frame (4144 × 2822 pixels) with the star field of the `survey` case, rendered by the simulator. It ran on the Windows dev machine on October 6, 2026, on one thread, and gives the wall time of the best of 5 runs. No other test run was active. Six idle processes of an earlier run used less than 2% of a core.

| Frame | Binned search (s) | Full search (s) | Detection step of the pipeline (s) | Search that the frame takes |
|---|---|---|---|---|
| Dark, 30 s (the `survey` case) | 0.28 | 0.70 | 0.35 | Binned |
| Dusk, 1 s at 13.0 mag/arcsec² | 0.16 | 0.29 | 0.47 | Full, after the binned |
| Dusk, 4 s at 15.0 mag/arcsec² | 0.19 | 0.36 | 0.61 | Full, after the binned |
| Twilight, 30 s at 17.2 mag/arcsec² | 0.32 | 0.58 | 0.36 | Binned |

The detection step of a frame at dusk costs 0.31 to 0.42 s more. The model of the searches gives each catalog star its own trail and measures the share of the light in the fitted image on the brightest isolated stars (`seeingmon.survey.completeness`, `light_in_core`). On full bin2 frames of a synthetic field with the pole about 600 px from the middle, as Polaris puts it, the choice of the search, the share of the light, and the expected stars of the cloud fraction cost 5 to 19 ms a long frame when the models of its trails exist from an earlier frame, and 12 to 67 ms when the FWHM or the exposure needs new ones, the most on the 30 s frames, whose trails of up to 6.5 pixels need the most models. These figures were measured on October 6, 2026, with the best of 5 warm runs, while a slow test of the other clone kept the machine 30 to 35% busy. **Pi 4 estimate.** The Pi 4 measured 2.68 s for the full search of the dark frame (October 3, 2026), and this machine takes 0.70 s for it, so a Pi 4 takes about 3.9 times as long. A frame at dusk then costs about 1.2 to 1.6 s more on a Pi 4, against a survey cycle of 180 s. The estimate is high, because the Pi 4 measured the detector before it batched its star mask and blend test, which made it faster. The model adds about 20 to 75 ms a frame on a Pi 4, and up to 0.26 s when it builds new models. In the simulated January dusk of `TestTheSurveyAtDusk`, 18 long frames took the full search, from −5.9° to −12.5°, which is about 22 to 29 s of CPU time on a Pi 4 at dusk and as much at dawn. A bright summer night can keep every long frame in the full search: at most 20 frames an hour at the cycle of 180 s, about 30 s of CPU time an hour on a Pi 4, under 1% of a core. Even then, the detection of a frame at dusk takes about 1.8 to 2.4 s on a Pi 4, less than the 2.68 s of every frame when the Pi 4 measured the survey frame, before the binned search existed.

### Alignment live view with a solve

The first light showed a live view that froze and lagged while the quick solve ran. The cause is the GIL: the detector (SEP) holds it for the whole of its background estimate and its extraction, so a solver thread in `core` freezes every other thread of `core` (the case measured 0.36 s for `sep.Background` and 2.45 s for `sep.extract` on a real 30 s frame, and a thread that woke every 0.5 ms got 2 and 5 wakeups). The table shows the effect on the live view of the helper. A producer thread hands a 4144 × 2822 frame to `AlignmentHelper.sink` every 0.5 s, 0.1 s after its capture time, as the scheduler thread does. The detector rows use a real frame, and the other rows use a synthetic frame of the same size. A receiver in another process reads the `alignment` stream as `web` does and timestamps each message. The lag is the time from the capture of the frame to the arrival of its preview.

| Solver | Previews per second | Lag: median / 95th percentile / maximum (s) | Longest wait for a preview (s) |
|---|---|---|---|
| None | 2.00 | 0.28 / 0.47 / 0.48 | 0.67 |
| A thread that sleeps 1 s (the GIL stays free) | 2.00 | 0.23 / 0.32 / 0.47 | 0.71 |
| A thread that sleeps 3 s | 2.00 | 0.28 / 0.41 / 0.49 | 0.75 |
| A thread that holds the GIL for 1 s (before) | 0.05 | 21.6 / 38.8 / 40.7 | 41.7 |
| A thread that holds the GIL for 3 s (before) | 0.03 | 45.4 / 45.4 / 45.4 | 45.4 |
| A worker process that holds its GIL for 1 s (after) | 1.93 | 0.33 / 0.63 / 1.52 | 1.53 |
| A worker process that holds its GIL for 3 s (after) | 2.00 | 0.38 / 0.62 / 0.88 | 0.79 |
| The detector in a thread of `core` (before) | 0.79 | 0.59 / 4.99 / 5.97 | 6.00 |
| The detector in a worker process (after) | 2.00 | 0.44 / 0.56 / 0.74 | 0.81 |

A sleeping solver never held the GIL, so it hid the problem. The rows that hold the GIL use a fake solver that calls `Sleep` through `ctypes.PyDLL`, which keeps the GIL for the whole call, as the detector does. In a thread, such a solver starves the preview: the first preview of the 1 s run arrived 41.7 s after the run began, and the producer, which stands for the scheduler thread, arrived up to 1.0 s late (2.6 s with the 3 s solver). In a worker process the same fake solver costs the live view almost nothing: the producer arrived at most 0.2 s late. A later run of the final code with the sleeping solvers, while other test runs loaded the machine, gave 2.00, 2.00, and 1.85 previews per second for no solver, the 1 s sleep, and the 3 s sleep, with a median lag of 0.5 to 0.7 s. The detector rows ran 90 s on the real frame with a synthetic catalog, so each solve ended unsolved after the detection, which is the cost that matters. In the thread, the producer arrived up to 4 s late, which is the delay of the scheduler thread that reads the camera, and the preview rate fell to 0.79 per second. In the worker, the producer arrived at most 0.1 s late. The development machine ran other jobs during the runs, and the worker runs at the lower priority of the survey worker, so a solve took 4 to 15 s. The live view did not notice. With the first set of quick-solve options (8 sigma, 300 stars), the same run in the worker gave 2.00 previews per second, a median lag of 0.41 s, and a 95th percentile of 0.56 s. The coarse search of the detector came later (see [Alignment helper](architecture.md#alignment-helper)): the detection of the real frame takes 0.7 s of CPU time with it, against 3.0 s with the same options and the full search, and 5.0 s with the survey options.

### Store and calibration

| Figure | Linux / Windows |
|---|---|
| Result row, one transaction each (µs) | 81 / 161 |
| Result row in a batch of 100 (µs) | 81 / 133 |
| Metrics append, fsync every 60 s of frame time (µs per frame) | 0.11 / 0.53 |

| Calibration workload | Linux (ms) | Windows (ms) |
|---|---|---|
| `python_loop` | 69.0 | 119 |
| `numpy_matmul` | 3.51 | 3.05 |
| `numpy_fft` | 3.02 | 2.64 |
| `sqlite_insert` | 16.1 | 21.7 |
| `thread_handoff` | 238 | 43.5 |

The segment append costs under 0.01% of a core at 98 fps, so the store does not matter for the budget. The `thread_handoff` row shows that wake-ups are expensive in the Linux virtual machine: a round of hand-offs takes 238 ms there in the least disturbed run, against 34 to 99 ms on Windows over three runs, while most other workloads run at the same speed or faster on Linux.

### Spread of the runs

| Figure | Linux, run 1 | Linux, run 2 | Linux, run 3 | Windows, run 1 | Windows, run 2 | Windows, run 3 |
|---|---|---|---|---|---|---|
| Kernel, bin1 (µs) | 21.4 | 24.0 | 19.1 | 21.7 | 20.3 | 18.5 |
| Whole push, bin1 (µs) | 24.1 | 34.6 | 26.1 | 33.8 | 68.6 | 78.1 |
| `acquire` CPU per frame, paced (µs) | 323 | 370 | 348 | 514 | 384 | 596 |
| `core` receive CPU per frame, paced (µs) | 73 | 105 | 89 | 173 | 43 | 173 |
| Survey frame (s) | 1.05 | 1.30 | 0.979 | 2.04 | 2.05 | 2.05 |
| Survey worker peak (MB) | 454 | 451 | 452 | 430 | 429 | 430 |
| `python_loop` (ms) | 79.6 | 93.1 | 69.0 | 119 | 105 | 70.4 |
| `thread_handoff` (ms) | 239 | 330 | 238 | 43.5 | 99.0 | 33.5 |
| Machine busy | 0% | 0% | 0% | 13% | 10% | 11% |

The last column of the summary table gives the verdict of each of the six runs. The verdict on `acquire`, the fast path in bin1, and the fast path with the receive in bin1 does not depend on the run. Two verdicts do: the bin2 fast path is `pass` in the three Linux runs and in the first Windows run, and `marginal` in the other two Windows runs, and the bin2 fast path with the receive is `pass` in the first and third Linux runs, `marginal` in the second Linux run and the first Windows run, and `fail` in the other two Windows runs. The Windows figures of the stream swing the most: the receive costs 43 to 173 µs a frame and `acquire` 384 to 596 µs, which is the 15.6 ms tick of the Windows clock.

### The whole system

The `core-sim` case ran once on each system at commit `b2e1f93`, with the full sensor at speed 1 (see [The whole-system case](#the-whole-system-case) for the method). Another system ran on the machine during both runs, so these are runs with another system active. The report of the Linux run says `machine 2% busy`, because a virtual machine does not see the load of its host, and the host was about as busy as during the runs of the other cases (the processor utility counter of Windows showed a mean of 46%). The report of the Windows run says `machine 26% busy`. The case itself used 1.7% of the Linux virtual machine and 2.1% of the Windows machine.

Each run sampled once per second for 303 s, after 8 s (Linux) or 13 s (Windows) for the processes to start. It had 98 s of the fast phase (8,652 frames at 88.3 frames per second on Linux, and 8,658 frames on Windows), 31 s with the scheduler paused, two survey steps, and four survey frames that the worker analyzed.

| Process | Peak (MB) | Resident in the fast phase (MB) | CPU in the fast phase | CPU with the scheduler paused | CPU over the whole cycle |
|---|---|---|---|---|---|
| `acquire`, with the simulator | 297 / 246 | 174 / 140 | 38.9% / 46.2% | 0.61% / 0.45% | 18.3% / 22.8% |
| `core` | 307 / 222 | 261 / 152 | 2.31% / 2.82% | 0.58% / 0.61% | 1.54% / 1.89% |
| Survey worker | 444 / 414 | | | | 4.0 s / 6.9 s for four frames |
| `web`, polled every 5 s | 80 / 85 | 80 / 81 | 0.71% / 0.61% | 0.68% / 0.61% | 0.75% / 0.76% |
| Children of `core` (the resource tracker) | 15 / none | | | | |
| Sum of the peaks | 1,143 / 966 | | | | |

The fast phase excludes the first window and every interval in which the stream switches or a survey frame waits for its analysis. The last column covers the whole cycle of 3 minutes: the fast stream, the survey step, and the gap that waits for the next slot.

- **`acquire` includes the simulator.** It renders every frame, and its CPU time and its memory are not those of the production code. The `ipc` case measures the production code with prebuilt frames: 3.4% of a core at 98 frames per second and a peak of 58 MB (commit `4afaca4`). The budgets read those figures.
- **The cost of a frame in `core`** is the CPU time of the fast phase minus the CPU time with the scheduler paused, divided by the frame rate: 195 µs on Linux (2.31% minus 0.58%, over 88.3 frames per second) and 251 µs on Windows. At 98 frames per second, that is 1.92% of a core on Linux and 2.46% on Windows, and 13 to 21% and 17 to 27% on the Pi 4 estimate. The cost covers the receive, the fast path, and the append of the segment. For comparison, the sum of the `fastpath` and `ipc` figures was 4.3% (440 µs a frame) at commit `0a764af`, and it is 1.15% (117 µs) at commit `4afaca4`. A run of the whole-system case a few commits before `b2e1f93`, before the services lane batched the stream, measured 585 µs a frame (5.7%), so the batching cut the cost of a frame to a third. The whole system costs more than the sum of the two cases, 195 against 117 µs, and the case does not break the difference down. The scheduler of `core` does work around each frame that the `fastpath` and `ipc` cases leave out, such as noting the frame and checking the star.
- **The paused scheduler costs 0.6% of a core.** That is the load that does not belong to the frames: the scheduler loop, the store, the health timer, and the answers to `web` and to the case, which asks for the status once per second.
- **Linux gives the threads.** In the fast phase, `core` uses one thread with 1.5% of a core and one with 0.4%, and each of the others uses 0.1% or less. The threads have no names, because Python of this version sets none, so the case cannot say which is which. The capture thread of `acquire` uses 37.7%, which is the simulator.
- **The survey worker** used 1.0 s of CPU for a frame on Linux and 1.7 s on Windows, which agrees with the 1.4 and 2.0 s of the `survey` case. Its peak is 444 and 414 MB.
- **The alignment worker has its own role on Linux.** The survey worker and the alignment worker (the quick solve of the Align page) both start with the `spawn` method, so their command lines are the same, and nothing outside tells them apart. Each worker names itself, `smon-survey` and `smon-align`, which `ps` and `top` show too, and the case reads the name from `/proc/<pid>/comm`. The case starts no alignment, so its figures hold the survey worker alone. When an alignment runs during a measurement, the alignment worker gets its own peak and CPU time, and its figures stay out of those of the survey worker. A worker that has not named itself 30 s after it starts counts as the survey worker. Windows has no such name, so both workers count as the survey worker there.
- **The resident size differs by system.** `core` holds 261 MB in the fast phase on Linux and 152 MB on Windows, and its peak is 307 and 222 MB. The two systems count resident memory in different ways, so the figures differ by more than the code does. The estimate uses Linux, which is the system of the Pi.
- **`web` takes 80 to 85 MB with one client that polls it.** That is 27 to 32 MB more than the 53 MB of its imports, the lower bound that the earlier estimate used.

## How the estimate works

The harness runs on a development machine, and the budgets are for a Pi 4. A figure from a dev run becomes a Pi 4 estimate when you multiply it by a range for its class of code. The module `seeingmon.perf.scaling` holds every range in one table, with its reason and its sources, and the table below copies it.

| Class | Applies to | Range | Basis |
|---|---|---|---|
| `numpy` | NumPy and SciPy code on arrays that fit in the cache: the kernel and the estimators | 5 to 11 times | Assumption. One core of a modern laptop has about 5 to 7 times the vector throughput of a Pi 4 core and about 4 to 6 times the memory bandwidth. A NumPy call that does little work costs the interpreter ratio. |
| `interpreter` | Python bytecode, SQLite, the bookkeeping of `acquire`, the store | 7 to 11 times | The Geekbench 6 single-core ratio of the two processor classes: about 250 to 300 for a Pi 4 (1.5 to 1.8 GHz) and 2,100 to 2,600 for a recent laptop core ([Pi 4 results](https://browser.geekbench.com/search?q=raspberry+pi+4), [processor chart](https://browser.geekbench.com/processor-benchmarks)). |
| `scheduler` | Thread wake-ups and system calls | 1 to 4 times | Assumption. A wake-up costs the kernel scheduler and the memory system, not the instruction stream, and a virtual machine on a hybrid laptop core makes wake-ups expensive. A bare-metal Pi 4 takes about as long, so the range starts at 1. |
| `memory` | The resident size of a process | 0.7 to 1.3 times | Assumption. The Python and NumPy objects have the same size on both machines, and the resident size counts the shared libraries that the process touched. |

The reference machine is a recent x86-64 laptop or desktop core. A Pi 4 run replaces the table: `seeingmon perf report pi4.json --baseline dev.json` compares each calibration ratio with its range, and you change the table where a ratio falls outside.

### Confidence

| Verdict | Confidence | Why |
|---|---|---|
| Fast path, bin1 | High for `pass` | The figure is 26 µs per frame of NumPy and bookkeeping. The limit holds up to a factor of 89 on the Linux run and 65 on the Windows run (30 to 97 over the six runs), against the 5 to 11 that the estimate assumes. |
| Fast path, bin2 at 360 fps | Medium for `pass` | The limit holds up to a factor of 27 on the Linux run (18 to 28 over the three Linux runs). Two Windows runs give `marginal`, because their limit is a factor of 7 and 10. |
| `acquire`, bin1 | Medium for `fail`, low for the percentages | The verdict is `fail` in all six runs. The work alone gives 10 to 16% on the Linux run, which is at the limit, and the wake-ups add 2 to 8%, so the verdict leans on the wake-up range, which is a guess. The figure leaves the camera out, which a real driver adds (see [What the camera adds](#what-the-camera-adds)). |
| Fast path and receive, bin1 | Medium for `pass` | The sum of the `fastpath` and `ipc` figures gives `pass` in all six runs (6 to 21%). The whole-system figure gives 13 to 21% on Linux, `pass`, and 17 to 27% on Windows, `marginal`, in runs with another system active. |
| Fast path and receive, bin2 at 360 fps | Low | The case runs the stream once, and the verdict depends on the run: `pass` in two, `marginal` in two, and `fail` in two (12 to 61% over the six runs). The Linux runs are the better guide, and the three of them give 12 to 30%. |
| Survey frame | High for `pass` | The estimate is 5 to 23 s in the six runs, against 180 s. |
| Survey worker, peak memory | Low | The range of 0.7 to 1.3 times straddles the limit, so only a Pi 4 run settles it. |
| All processes, peak memory | Low | The sum with the stand-in for `core` passes (686 to 1,295 MB), and the measured peaks of the whole-system run give `marginal` (783 to 1,476 MB). The `acquire` row is a lower bound, and the operating system's share is an assumption (see [Memory](#memory)). |

### The kernel check

The architecture estimates 0.2 to 0.4 ms for the kernel on a 128 × 128 frame on a Pi 4, and it says that the kernel takes about 100 µs on the dev machine. The `seeingmon perf report --budgets` command divides the first figure by the measured time, and it compares the factor that results with the table.

| Run | Kernel (µs) | Factor that the architecture implies | Compared with the table's 5 to 11 | Estimate from the table (ms) |
|---|---|---|---|---|
| Linux, run 1 | 21.4 | 9.3 to 18.7 | Consistent | 0.11 to 0.24 |
| Linux, run 2 | 24.0 | 8.3 to 16.7 | Consistent | 0.12 to 0.26 |
| Linux, run 3 | 19.1 | 10.5 to 21.0 | Consistent | 0.10 to 0.21 |
| Windows, run 1 | 21.7 | 9.2 to 18.4 | Consistent | 0.11 to 0.24 |
| Windows, run 2 | 20.3 | 9.9 to 19.7 | Consistent | 0.10 to 0.22 |
| Windows, run 3 | 18.5 | 10.8 to 21.6 | Consistent | 0.09 to 0.20 |

In every run, the factor that the architecture implies overlaps the range of the table, but only at the top of it: the implied factors start at 8 to 11 and end at 17 to 22, against the 5 to 11 of the table. The kernel takes 18.5 to 24.0 µs over the six runs, so the runs agree within 30%. The estimate from the table (0.09 to 0.26 ms) sits below the architecture's 0.2 to 0.4 ms and meets it at the edge, so the architecture's estimate is the more cautious of the two. The table stays, and a Pi 4 run replaces it.

## Memory

A run on a Pi 4 replaced the estimates of this section (see [Results on a Raspberry Pi 4](#results-on-a-raspberry-pi-4)). The harness adds the peaks of the processes on the dev machine, multiplies the sum by the `memory` range, and adds the share of the operating system. The `core-sim` case measures the peaks of `core`, the survey worker, and `web` in the running system, and the budgets read them in place of the stand-ins that the first version of the page used. The table shows the Linux run at commit `b2e1f93` (a run with another system active, see [The whole system](#the-whole-system)).

| Process | Dev machine peak (MB) | Pi 4 estimate (MB) |
|---|---|---|
| `acquire` with the zero-cost camera, from the `ipc` case | 58 | 41 to 76 |
| `core`, measured in the system | 307 | 215 to 399 |
| Survey worker, measured in the system | 444 | 311 to 577 |
| `web`, measured with one client that polls it | 80 | 56 to 104 |
| Children of `core` (the resource tracker), measured in the system | 15 | 11 to 20 |
| Operating system, an assumption | | 150 to 300 |
| Sum | 905 | 783 to 1,476 |

The estimated sum straddles the 1.4 GB budget, so its verdict is `marginal`, and it stays under the 1.6 GB gate that separates the 2 GB model from the 4 GB model, so it is `pass`. This evidence does not call for 4 GB, and it does not settle the 1.4 GB budget either. The estimate with the stand-in for `core` is lower: 766 MB (686 to 1,295 MB), because it uses 200 MB for `core` (the fast-path process of 147 MB and the store process of 53 MB) and the imports of `web` alone. On Windows the same peaks sum to 775 MB (`core` 222, the worker 414, and `web` 85), and the estimate passes both limits, but the Pi runs Linux.

The `acquire` row is the weak one. The `ipc` case streams 128 × 128 frames, and it never sends a survey frame, so its 58 MB leaves out the passage of the 23 MB survey frames through `acquire`. The whole-system case measures 297 MB for `acquire` on Linux (246 MB on Windows), but that figure includes the simulator, which holds several arrays of 47 MB while it renders a survey frame. A real `acquire` lies between the two. The sum of all the peaks in the whole system, with the simulator in `acquire`, is 1,143 MB on Linux (966 MB on Windows), and its estimate is 950 to 1,786 MB: `marginal` against both limits. Four limits apply:

- The `core` row is measured in a system with one client of `web` and no sink. The sink forwarder, the daily retention, and several browsers add to it. In the fast phase, `core` holds 261 MB, which is 173 MB more than the 88 MB of its imports, and the case does not break the rest down.
- The `web` row counts one client that polls every 5 s. The live views and the preview encoder take more.
- The conditions of the architecture for 2 GB are design rules that the harness does not test: the out-of-memory killer takes the survey worker first, calibration frames stay memory-mapped, and the Pi processes bin2 frames only.
- The survey worker straddles its 550 MB limit by itself, so it is the first figure to read in a Pi 4 run.

The `memory` case reads the resident size of a fresh process after it imports each set of modules. The sets show where the baseline of each process comes from. Multiply them by 0.7 to 1.3 for a Pi 4 (the imports of `core` take an estimated 62 to 115 MB).

| Set of imports | Resident size, Linux / Windows (MB) |
|---|---|
| Python and the memory reader | 19.5 / 22.9 |
| NumPy | 33.9 / 32.0 |
| Frame and record types (pydantic) | 34.5 / 33.4 |
| Fast-path analyzer (SciPy) | 45.7 / 41.9 |
| SQLite store and segment writer | 44.0 / 42.1 |
| Survey analyzer (SEP, pyerfa) | 83.0 / 81.1 |
| `astropy.coordinates` and `astropy.time` | 73.3 / 67.3 |
| FastAPI, uvicorn, Pillow | 54.2 / 53.3 |
| Everything that `core` imports | 88.6 / 87.7 |

## The Rust decision

The architecture keeps Rust (PyO3) as the replacement for the per-frame metrics if the Pi 4 gate fails. On the estimate, the per-frame metrics do not fail the gate: the whole fast path takes 1.4 to 3.1% of a core in bin1, against 25%. Rust would replace the kernel, which is 19 of the 26 µs of a frame, and it would not touch the costs of the stream, where `acquire` still exceeds its budget. Keep Python for the per-frame metrics.

The number that flips the decision is a Pi 4 run in which the fast path (the `fastpath` case, without the receive) takes more than 25% of a core. That is 2.55 ms per frame in bin1 at 98 fps, and 0.69 ms per frame in bin2 at 360 fps. It takes a Pi 4 that runs 30 to 97 times slower than the dev machine in bin1, and 7 to 28 times slower in bin2 (the ranges cover the six runs), against the 5 to 11 times that the estimate assumes. The `calibration` ratios of a Pi 4 run show which side to expect. With the receive, bin2 at 360 fps is the first row to reach the limit: it takes a Pi 4 that runs 5 to 13 times slower than the dev machine, against the 5 to 11 times that the estimate assumes, so this is the row to read first in a Pi 4 run.

## Results on a Raspberry Pi 5

A temporary Pi 5 (Cortex-A76, 4 cores, 8 GB, Debian 13, Python 3.13) ran the harness on October 2, 2026, with the services stopped and the machine 0 to 5% busy (`seeingmon perf run --label pi5 --quiet-wait 120`, about 7 minutes). The Pi 5 is not the final machine. **Do not read the "Pi 4 (estimate)" column of `seeingmon perf report --budgets` for this run.** That column applies the scaling from a fast x86-64 dev machine (`seeingmon.perf.scaling`), and a Pi 5 is not that machine. Raspberry Pi's own Geekbench 6 figures put a Pi 5 about 2.4 times ahead of a Pi 4 for one core and 2.2 times for four cores (the Pi 4 scores 340 and 723, the Pi 5 774 and 1,604; [benchmarking Raspberry Pi 5](https://www.raspberrypi.com/news/benchmarking-raspberry-pi-5/)), so a Pi 4 figure is roughly 2.2 to 2.4 times the Pi 5 figure for CPU-bound work. That is an estimate too, until a Pi 4 runs.

| Figure | Pi 5, measured | Pi 4, about 2.2 to 2.4 times that |
|---|---|---|
| `acquire` without the camera, bin1 at 98 fps | 130 µs a frame, 1.28% of a core | 2.8 to 3.1% (limit 10%) |
| Fast path, bin1 | 49 µs a frame, 0.47% | 1.0 to 1.1% (limit 25%) |
| Fast path and receive in `core`, bin1, measured in the whole system | 1.79% | 3.9 to 4.3% (limit 25%) |
| Fast path and receive, bin2 64 × 64 at 360 fps | 2.58% | 5.7 to 6.2% (limit 25%) |
| Kernel, 128 × 128 | 34 µs | 75 to 82 µs |
| Survey frame, bin2 | 1.42 s (detection 0.85 s, sky quality 0.51 s) | 3.1 to 3.4 s |
| Survey worker, peak memory | 444 MB (limit 550 MB) | the same, within tens of percent |
| All processes in the whole system, sum of the peaks | 1,117 MB, with the simulator in `acquire` | the same (budget 1.4 GB, gate 1.6 GB) |

On this estimate every CPU budget passes with a wide margin, including the `acquire` row that the dev-machine scaling called marginal or failing. The memory rows stay near their budgets, because memory does not scale with the clock. The camera on the Pi streamed bin1 at 82.1 fps with a jitter of 0.01 to 0.03 ms and no drops (`docs/hardware-checks.md`).

## Results on a Raspberry Pi 4

A Raspberry Pi 4 Model B (Cortex-A72, 4 cores, 2 GB of RAM of which 1,844 MiB are usable, Debian 13, kernel 6.18, Python 3.13, zram swap, an 8 GB card, and a ZWO ASI294MM on a USB 3 port) ran the harness on October 3, 2026, with the services stopped, the governor on `performance`, and the machine 0 to 1% busy (`seeingmon perf run --label pi4 --quiet-wait 120`, 8 minutes). A report with the label `pi4` compares its figures with the limits directly. The first run read `vcgencmd get_throttled` as 0x50000 (the board had seen under-voltage and throttling since it powered up), so a second run followed after a reboot, with 0x0 before and after. The two runs agree within 2%, and the table gives the second.

| Budget | Limit | Pi 4, measured | Estimate from the Pi 5 (2.2 to 2.4 times) | Verdict |
|---|---|---|---|---|
| `acquire` without the camera, bin1 at 98 fps | 10% of a core | 3.10% | 2.8 to 3.1% | pass |
| Fast path, bin1 at 98 fps | 25% | 1.84% | 1.0 to 1.1% | pass |
| Fast path and receive in `core`, bin1, in the whole system | 25% | 7.95% | 3.9 to 4.3% | pass |
| Fast path, bin2 64 × 64 at 360 fps | 25% | 6.27% | | pass |
| Fast path and receive, bin2 64 × 64 at 360 fps | 25% | 9.11% | 5.7 to 6.2% | pass |
| Kernel, 128 × 128 | | 132 µs | 75 to 82 µs | |
| Survey frame, bin2 | 180 s | 4.49 s (detection 2.68 s, sky quality 1.50 s) | 3.1 to 3.4 s | pass (not a gate) |
| Survey worker, peak memory | 550 MB | 440 MB | the same | pass |
| All processes, sum of the peaks, with `acquire` from the `ipc` case | 1.4 GB, gate 1.6 GB | 895 MB | the same | pass |

The Pi 4 ran 2.4 to 4.4 times slower than the Pi 5 on the CPU rows (3.1 times on the survey frame), more than the 2.2 to 2.4 times of Raspberry Pi's Geekbench figures, because small NumPy and Python work scales worse than Geekbench does. Every budget still passes by a factor of 2.7 or more. The code has changed since: on the dev machine, a frame of the night now costs `core` about twice as much (see [A frame of the night costs twice what the Pi 4 measured](#a-frame-of-the-night-costs-twice-what-the-pi-4-measured)).

**Memory.** Three measurements bound it:

- **The `core-sim` case.** `core` peaks at 312 MB, the survey worker at 439 MB, `web` at 75 MB, and the resource tracker at 12 MB. With the simulator in `acquire` (290 MB), the peaks sum to 1,128 MB. The budget row reads `acquire` from the `ipc` case (56 MB), and it gives 895 MB.
- **The installed system on the real camera.** It ran for 7 minutes in `auto`, and every survey step ran (the camera saw a room, so the plate solver had no stars). The three services held at most 1,008 MB of resident memory together (`acquire` 275 MB, `core` and the worker 441 MB, and `web` 71 MB). The machine used at most 1,053 MiB, and at least 791 MiB stayed available.
- **A 45-minute run of the whole system on the full sensor** (`seeingmon dev`, with the catalog and the matching sky: 18,477 frames, 15 survey steps, no dropped frame, no fault, and no warning). The processes held at most 992 MB at once, the machine used at most 1,155 MiB of the 1,844 MiB, and at least 689 MiB stayed available. `core` reached 365 MB in the first quarter of the run and held it, so nothing leaks over 45 minutes. The zram swap stayed empty, and the kernel killed no process.

2 GB of RAM is enough. The worst moment used 63% of the memory and left 689 MiB, and every peak sits below the budget of 1.4 GB. The kernel reserves 512 MB of contiguous memory (CMA) on this image, and the figures include it.

**What the run found.**

- The kernel of Raspberry Pi OS leaves the memory cgroup off: `/sys/fs/cgroup/cgroup.controllers` listed no `memory`, so the `MemoryMax` limits of the units were not enforced. The kernel command line needs `cgroup_enable=memory cgroup_memory=1`. The installer warns now, and the runbook has the step. The units also set `MemorySwapMax=0`, so that a unit that passes its limit is killed and restarted and does not swap to zram.
- The first survey exposure after a start timed out twice (`no frame within 0.6 s`) on the real camera. The first recovery step, `restart_capture`, fixed it, and the `camera` component read `degraded` until ten good frames had cleared it, which took 7 minutes at the survey cadence. Later steps never timed out. The wait was too short, not the camera: a 1 ms exposure of the full bin2 frame takes 0.48 to 0.53 s on the Pi 4 (0.29 s for the 20 arcminute watch ROI), and the driver modeled it with the video line, which gives 53 ms. The scheduler then waited twice that plus 0.5 s, which is 0.61 s. The profile now holds a snapshot model (`snapshot_overhead_s = 0.27` and `snapshot_row_time_us = 75` for bin2), the driver reports its period for a single exposure, and the wait for the full frame is 1.5 s. The Pi ran with `read_timeout_margin_s = 2.0` under `[scheduler.loop]` and `[services.acquire]` as a stop-gap. The profile model replaces that stop-gap, and the settings stay available for a slower camera or host (see [Measure single exposures](hardware-checks.md#measure-single-exposures)). The Pi 4 has not run the new model yet.
- The simulator cannot act as the camera on a Pi 4 with the default read margin of 0.5 s, because it renders a full frame inside the read. The scheduler then counts camera faults and stays in `safe`. Set `read_timeout_margin_s = 20.0` under `[services.acquire]` and `[scheduler.loop]`, as the harness and `seeingmon dev` do.

**The camera.** `seeingmon camera rates` on the USB 3 port of the Pi 4 gave the rates of the Windows machine and the Pi 5: 82.1 fps for bin1 128 × 128, 102.9 fps in high-speed mode, and 415.8 fps for bin2 64 × 64 at 0.5 ms, with a jitter of 0.01 to 0.04 ms and no dropped frame in 34 rows. The timing fit matches the profile to 0.0%.

**The plate solver.** `solve-field` 0.97 with the cap index (five files, 3.2 MB) solved six synthetic fields from the real catalog (the pole 0 to 2.5 degrees off the axis, any roll, 150 stars each) in 0.34 to 0.61 s each, with a center error of 0.04 to 0.22 arcsec and the scale within 0.01%. Its peak resident size was 30 MB.

**What the Pi 4 runs do not show.** A real sky (stars through the detector and the solver), a night of running, a data partition on a card of the final size (the 8 GB test card has 2 GB free), the heater HAT, and the fast mode of the real camera under the full scheduler.

## The cost of measuring all day

Since the visibility lane (see [visibility.md](visibility.md)), the system measures seeing whenever Polaris is visible, at any Sun elevation, and it searches for Polaris whenever it does not see it. In the simulator Polaris is visible in full daylight, so on a clear day the system measures from dawn to dusk, and on a cloudy night it searches for hours. Two cases measure what that costs, with the method of the whole-system case (see [The whole-system case](#the-whole-system-case)):

- **`day-sim`, a day of measuring.** The simulated clock starts at noon of midsummer at the synthetic site, with the Sun 58.4 degrees up. The scheduler finds Polaris in two search bursts and measures at the exposure that the bright sky allows, 1.2 ms against 2 ms in the dark. In each survey step the 1 ms frame clips, a watch frame of 32 µs measures the sky, and the step skips its long exposure, because even the shortest long exposure would pass its target.
- **`cloudy-sim`, a cloudy night of searching.** The clock starts on a winter night, and an opaque overcast hides every star for the whole run. The scheduler searches: a burst of 50 frames every 15 s, each frame through the three matched filters. The first survey frame shows the clouds, and the cycle for clouds follows: a search period of 60 s and a survey step every 100 s, with a long exposure that grows from 1 s by 4 times a step. No burst detects Polaris.

Both runs take the cycle of production: fast periods of two analysis windows of 60 s, and a survey step every 180 s. The `core-sim` case keeps the short windows of the launcher. A day at midsummer lasts 16 to 19 hours, and a run cannot take that long, so each run measures about 4 minutes of a cycle that repeats through the day or the night. The share of a core over the cycle covers whole cycles, from the end of one survey step to the end of a later one: from the end of the first step by day, and from the end of the second under clouds, because the first cycle for clouds starts at once after the step that showed the clouds, without its gap. So the search at the start of a run stays out of it.

The figures come from three runs of the harness on the Windows laptop of the tables above, on October 6, 2026. Runs 1 and 2 ran at commit `4adf3c2` with the changes that added the two cases. Run 3 ran on commit `66251aa` with the same changes and two fixes of the method: the share over whole cycles, and search figures that leave out the whole warm-up burst, also when a sample falls inside it. Runs 1 and 2 read the share over the cycle from the start of measure, or from the first sample of the cycle for clouds, to the end, which held one gap fewer than whole cycles, so the share of the day came out about 15% high. Other work loaded the machine: the reports say 43 to 51% busy during the first runs of `core-sim` and `day-sim`, 27% during the first `cloudy-sim`, 25 to 30% during the second runs, and 24 to 30% during run 3. Read each new figure against `core-sim` of the same run, not against the older tables: `core-sim` measured 477, 401, and 396 µs for a frame in `core`, against 251 µs at commit `b2e1f93`, and the next subsection shows that the code made the difference, not the laptop.

### A frame of the night costs twice what the Pi 4 measured

The code that the Pi 4 measured on October 3 (commit `93647df`) and the current code ran `core-sim` back to back on the laptop, in the session of run 3:

| Run of `core-sim` | Code | CPU time of a frame in `core` | Machine busy |
|---|---|---|---|
| 1 | `93647df`, the code of the Pi 4 run | 186 µs | 22% |
| 2 | The current code (run 3 of this section) | 396 µs | 25% |
| 3 | `93647df` | 180 µs | 23% |
| 4 | The current code | 430 µs | 52% |

- **The code doubled what a frame of the night costs `core`.** In the same session, the current code costs 2.1 and 2.4 times the code of October 3. The laptop is not slower than in the older runs: the calibration case runs faster than in the older Windows runs (the Python loop takes 61 to 63 ms, against 70 to 119 ms), and the code of October 3 costs 180 to 186 µs, below the 251 µs of commit `b2e1f93`. The 477 and 401 µs of runs 1 and 2 carry the same rise.
- **The current code puts a clear night at about 17 to 19% of a Pi 4 core.** The Pi 4 measured 7.95% for the code of October 3, which is 811 µs a frame, 4.4 to 4.5 times this laptop. The same factor gives the current code 1.7 to 1.9 ms a frame on a Pi 4. That stays within the 25% budget, with less room than on October 3. The verdicts below scale each new figure from this night.
- **Where the time goes is open.** Since October 3, the fast path gained the sky noise and the SNR of the star (`0c894fd`), the rolling seeing value (`a533e94`), and the matched filter of the missing-star test (`66263a7`, `ae8bc98`). The scheduler gained the adaptive exposure (`993290b`), and `core` the live video of Polaris (`72d6d80`). The `fastpath` case shows about 30 µs of the rise: a bin1 frame takes 56 µs through `FastPathAnalyzer` in run 1, against 26 µs before. Most of the rise, about 200 µs, lies in the work of the scheduler and the services around a frame, which no case times on its own. A run of `core-sim` along the commits since `93647df` finds it.

### Per-frame costs

The visibility lane added two costs to the frames of the fast stream: the matched filter of the missing-star test in every frame of measure, and the three matched filters of a search frame in every frame of a burst. The Gaussian-weighted centroid, which is not the default, would add a third. The `kernel` case times each on its own. A cell reads `run 1 / run 2 / run 3`, and the Pi 4 column scales the range of the runs by the 5 to 11 times of NumPy code.

| Figure, µs per frame | bin1 128 × 128 | bin2 64 × 64 | Pi 4 estimate |
|---|---|---|---|
| Kernel of measure, with the matched filter of its missing-star test | 30.3 / 38.5 / 29.6 | 25.4 / 36.6 / 26.9 | 0.15 to 0.42 ms in bin1 |
| Kernel without that matched filter | 18.1 / 25.2 / 18.8 | 16.3 / 24.0 / 17.1 | |
| What the matched filter adds (`matched_extra`) | 12.2 / 13.3 / 10.8 | 9.1 / 12.6 / 9.8 | 54 to 146 µs in bin1: 0.5 to 1.4% of a core at 98 fps. In bin2: 1.6 to 5.0% at 360 fps |
| A search frame: three matched filters within 20 px (`search`) | 157 / 209 / 158 | 131 / 221 / 133 | 0.79 to 2.3 ms in bin1 |
| Kernel with the Gaussian-weighted centroid | 73.2 / 104 / 74.2 | 80.6 / 125 / 83.9 | |
| What the weighted centroid adds (`gaussian_extra`) | 43.0 / 65.1 / 44.6 | 55.2 / 88.0 / 57.0 | 2.1 to 7.0% of a core in bin1 at 98 fps, 9.9 to 35% in bin2 at 360 fps |

- **The missing-star test is cheap.** It adds 11 to 13 µs to a frame, 40% of the kernel, and the fast path stays far inside its budget: the `fastpath` case gives 0.58% of a dev core in bin1 (2.9 to 6.4% on a Pi 4) and 2.0% in bin2 (10 to 22%).
- **A search frame costs five times a kernel.** Each of the three filters scans the circle of 20 px around the prediction, a box of at least 41 × 41 pixels. The frame is timed alone here, and `core` pays more for it in the system (see [A cloudy night of searching](#a-cloudy-night-of-searching)).
- **The weighted centroid is information.** With it, the fast path in bin2 at 360 fps takes 20 to 44% of a Pi 4 core on the estimate, which straddles the 25% budget (the row `gaussian-bin2`, which gates nothing), and in bin1 5 to 11%. That matches the estimate of the visibility brief (departure 10).

### A day of measuring

| Figure | Day (`day-sim`), run 1 / run 2 / run 3 | Night (`core-sim`), run 1 / run 2 / run 3 |
|---|---|---|
| CPU time of a frame in `core` (µs) | 507 / 368 / 366 | 477 / 401 / 396 |
| The same, as a share of a core at 98 fps | 4.97 / 3.61 / 3.59% | 4.67 / 3.93 / 3.88% |
| `core` in the fast phase | 4.80 / 4.03 / 3.58% | 5.07 / 4.18 / 4.13% |
| `core` with the scheduler paused | 0.66 / 1.01 / 0.57% | 1.16 / 0.89 / 0.89% |
| `core` over the cycle | 4.10 / 3.47 / 2.68% (runs 1 and 2 hold one gap fewer than whole cycles, run 3 whole cycles of production) | 4.40 / 3.02 / 2.86% (the cycle of the launcher) |
| Peak memory of `core` (MB) | 248 / 247 / 249 | 268 / 259 / 259 |
| Peak memory of the survey worker (MB) | 380 / 381 / 381 | 384 / 385 / 384 |
| Peak memory of `web` (MB) | 89 / not read / 90 | 90 / 90 / 90 |
| Peak memory of `acquire`, with the simulator (MB) | 246 / 246 / 245 | 254 / 254 / 253 |
| Fast exposure | 1,196 µs | 2,000 µs |
| Survey frames that the worker analyzed in a step | 1, the 1 ms frame | 2 |

- **A frame of daylight costs what a frame of the night costs.** In the same run, the day's frame costs 0.92 to 1.06 times the night's. The bright sky shortens the exposure and leaves the frame rate at 82 frames per second, which the readout sets, and the per-frame work is the same.
- **A day is no busier than a clear night.** The camera measures for 120 s of each 180 s cycle, as at night. The survey step skips its long exposure, so the camera idles about 55 s of the cycle, and the survey worker analyzes one frame of each step instead of two. The peaks of all processes match those of the night within 8%.
- **Frames that the simulator dropped.** The simulator in `acquire` dropped 35, 6, and 0 of 13,300 frames in the three day runs, on a busy machine. The simulator renders every frame, so this says nothing about the real camera.
- **`web` was not read in the second day run.** The sampler did not find the interpreter of `web` behind its launcher, so the run reports 4 MB, the size of the launcher. The first run reports 89 MB, and the verdicts below use it.

### A cloudy night of searching

| Figure | Run 1 / run 2 / run 3 |
|---|---|
| CPU time of a search frame in `core`, beyond the gaps between the bursts (µs) | 2,525 / 958 / 1,050 |
| The same, as a share of a core at 98 fps: what a burst takes while it runs | 24.7 / 9.38 / 10.3% |
| `core` over the search phase, bursts and gaps | 2.02 / 1.61 / 1.50% |
| `core` with the scheduler paused | 0.99 / 1.15 / 0.99% |
| `core` over the cycle for clouds | 2.12 / 1.76 / 1.67% (run 3 over a whole cycle from the end of the second step) |
| Bursts and search frames in the run | 16 bursts and 800 frames in each run. The search figures count 701, 650, and 651 frames: run 3 leaves the warm-up burst out in full. |
| Survey steps, and the frames that the worker analyzed | 3 steps, 6 frames |
| Peak memory of `core`, the survey worker, and `web` (MB) | 254, 381, 89 / 254, 381, 90 / 254, 382, 90 |
| Peak memory of `acquire`, with the simulator (MB) | 233 / 234 / 234 |

- **A search frame costs `core` 2.4 to 5.3 times a frame of measure** (2.65 times in run 3). The CPU time of `core` per second ticks at 15.6 ms on Windows, so two more runs read the cycle counter of `core` (`QueryProcessCycleTime`) every 0.1 s: 1,829 and 838 µs a search frame. Timed alone, the matched filters take 0.15 to 0.35 ms of it, the median of the frame, which sets the exposure of the next burst, 0.05 to 0.17 ms, and the receive about 0.09 ms, all three slower on a busier machine. The rest is the work around each frame and around the start and the end of the stream of each burst. The scheduler thread does about four fifths of the whole (a count of the cycles of each thread). The work does not depend on the clouds: a burst costs the same in a clear sky.
- **The load of a cloudy night is small.** The bursts take about 3% of the time, so `core` averages 1.7% of a dev core over a whole cycle for clouds in run 3, about 60% of the day's 2.7%. `acquire` streams 50 frames every 15 s in the search periods, about 4% of the frames of a clear night.
- **A burst is the peak, and it fails its budget on the estimate.** While a burst runs, `core` needs 9 to 25% of a dev core. On a Pi 4 that is 45 to 50% in run 3 from the measured night scaled to the current code (40 to 100% over the three runs) and 66 to 272% on the scaling table, above the 25% that the fast path may use. The frames that `core` has not read yet wait in `acquire`, so a slow burst lasts longer, and a frame that `acquire` drops costs the burst a frame, not a window. A Pi 4 run confirms the figure. Two changes are cheap: take the median of a burst frame on a subsample of its pixels, or on a few frames of the burst, and find what the scheduler does around a search frame that it does not do around a frame of measure.

### The verdicts

The harness estimates each row from this machine with the scaling table (`seeingmon perf report --budgets`). On this laptop that table fails the night's row of `core-sim` too (27 to 51% of a Pi 4 core over the three runs), where a Pi 4 measured 7.95% on October 3. So the table also gives each new row from the measured Pi 4 night, scaled to the current code: the ratio of the new figure to `core-sim` of the same run, times the 17 to 19% that the current code takes on a clear night (see [A frame of the night costs twice what the Pi 4 measured](#a-frame-of-the-night-costs-twice-what-the-pi-4-measured)). For memory, it is the sum of the Pi 4 night, 895 MB, which the new peaks match.

| Budget | Limit | Dev machine, run 1 / run 2 / run 3 | Pi 4, scaling table | Pi 4, from the measured night | Verdict |
|---|---|---|---|---|---|
| Fast path and receive in daylight, bin1 at 98 fps, in `core` (`day-core`) | 25% of a core | 4.97 / 3.61 / 3.59% | 25 to 55% | 16 to 20% | pass on the measured night, fail on the table |
| Search burst and receive, bin1 at 98 fps, in `core` (`search-bin1`) | 25% of a core while a burst runs | 24.7 / 9.38 / 10.3% | 66 to 272% | 40 to 100% (run 3: 45 to 50%) | fail |
| Search burst and receive, bin2 64 × 64 at 360 fps, from the `kernel` and `ipc` cases (`search-bin2`) | 25% of a core while a burst runs | 6.09% (run 1) | 33 to 67% | | fail on the table |
| All processes in daylight, peak memory (`day-memory-1.4`, `day-memory-1.6`) | 1.4 GB, gate 1.6 GB | 773 MB (run 1) | 691 to 1,305 MB | about 900 MB | pass |
| All processes on a cloudy night, peak memory (`cloudy-memory-1.4`, `cloudy-memory-1.6`) | 1.4 GB, gate 1.6 GB | 779 MB (run 1) | 695 to 1,312 MB | about 900 MB | pass |
| Fast path with the Gaussian-weighted centroid, bin1 (`gaussian-bin1`) | 25% (not a gate) | 1.00% (run 1) | 5.0 to 11.0% | | pass |
| Fast path with the Gaussian-weighted centroid, bin2 at 360 fps (`gaussian-bin2`) | 25% (not a gate) | 3.98% (run 1) | 20.0 to 43.8% | | marginal |

The shares over whole cycles give the CPU load of a day and of a cloudy night. In run 3, `core` uses 2.7% of a dev core by day and 1.7% under clouds: about 12% and 7% of a Pi 4 core with the factor of 4.4 to 4.5 that the measured night gives (19 to 29% and 12 to 18% on the scaling table). The factor belongs to the work of a frame, so these figures are rough. `acquire` runs its stream by day as at night, 3.1% of a Pi 4 core while it streams (measured), and almost nothing under clouds.

Runs 2 and 3 measured the system cases alone (`--cases calibration,kernel,core-sim,day-sim,cloudy-sim`), so their memory rows and the bin2 search row lack the `ipc` case and read `n/a`.

### The sensor temperature

The simulator cannot measure it. Its sensor reads the ambient temperature plus a constant rise (the options `ambient_c`, 15 °C, and `sensor_rise_c`, 4 °C), so every run read 19.0 °C throughout, and nothing in the simulator warms in the sun. A real camera in a sunlit enclosure runs warmer, and since it measures all day now, the camera and the Pi work through the warmest hours.

The seeing does not depend on it: the real camera's dark current, 0.47 e⁻/s per bin2 pixel at 20 °C and doubling every 5 °C (see [architecture.md](architecture.md)), reaches 7.5 e⁻/s at 40 °C, which is 0.015 e⁻ per bin2 pixel in a fast frame of 2 ms, far below the read noise. The survey does depend on it. The dark model scales the dark current with the temperature, a dark set counts only within 3 °C of the frame (`dark_due`), and the hot pixels of the nearest set mask the frame. A camera that the Sun heated still runs warm at dusk, when the long exposures start again. Phase 3 must measure:

- The sensor temperature against the ambient temperature through a sunny day in the enclosure, from the `sensor_temperature_c` of the `health` records, and how fast it falls after sunset.
- Whether the dark library covers those temperatures: the dark sets that `dark_due` asks for, and how far the fit of the doubling temperature holds above the warmest set.
- The temperature of the Pi in the sunlit enclosure, and whether it throttles (`vcgencmd get_throttled`). The harness cannot make a board stay cool.

## Measure on a Pi 4

The Pi 4 measurement ran on October 3, 2026 (see above). These steps repeat it on another Pi 4, or after a change. You run these commands on the Pi, and you copy one JSON file back. The report holds no host name, user name, or serial number.

1. Install the packages, and clone the repository. Raspberry Pi OS Lite (64-bit) with Python 3.11 or 3.13 works. An editable install lets the report name the commit.

   ```bash
   sudo apt-get update
   sudo apt-get install --no-install-recommends git python3 python3-venv
   git clone https://github.com/komakallio/seeing-monitor.git
   cd seeing-monitor
   python3 -m venv .venv
   .venv/bin/pip install -e ".[survey,web]"
   ```

2. Prepare a clean measurement. Stop the services of the system if you installed them (`sudo systemctl stop seeingmon.target`), close other programs, and make sure that the board has a heatsink or a fan. Set the frequency governor to `performance` so that the clock does not ramp.

   ```bash
   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   vcgencmd get_throttled    # 0x0 before the run
   ```

3. Run the harness. The full run takes about 20 minutes on the dev machine: 2 minutes for the cases and 6 minutes for each of the three runs of the whole system. On a Pi 4 the cases take an estimated 5 to 15 minutes. The `survey` case needs about 1 GB of free memory for its two processes. The whole-system case runs the simulator inside `acquire`, and the simulator may not reach 30 frames per second on a Pi 4. The case then fails with the message "the fast stream never reached a steady state", and so may `day-sim`, and `cloudy-sim` with "the search never ran a burst after its warm-up". Run the other cases in that event (`--cases calibration,kernel,fastpath,ipc,survey,store,memory`), and send the message to the lead.

   ```bash
   .venv/bin/seeingmon perf run --label pi4 --quiet-wait 120 --json local/perf/pi4.json
   ```

   Check `vcgencmd get_throttled` again. The report records its value at the end of the run. A value other than `0x0` means that the board lost power or heat headroom, so the figures are slower than the clock says. Run again with a better supply or cooling.

4. Copy `local/perf/pi4.json` to your development machine, with `scp` or a USB stick, and compare it with your dev run.

   ```bash
   seeingmon perf report local/perf/pi4.json --budgets --baseline local/perf/dev.json
   ```

   The report compares the figures with the limits directly, because the label is `pi4` and the calibration case ran. Send the file to the lead, who updates this page and the scaling table.

## What the harness does not measure

- **USB.** The camera, the SDK, and the USB transfer. The `ipc` case uses a camera that costs nothing per frame, so the cost of the vendor library and of the USB stack of the kernel is missing from the `acquire` figure. The Python of the production ASI driver (10 to 21 µs per frame on a stub SDK) is missing too, and the camera table shows it apart. About 10 µs of it is the arm and disarm of the `CallWatchdog` around each SDK call, which belongs to the hardware lane.
- **The SD card.** The `store` case writes to the disk of the machine. An SD card is slower, and its fsync can take tens of milliseconds. The segment writer calls fsync once per minute of frame time, so the effect is small, and the Pi 4 run shows it.
- **Thermal throttling and the supply.** The harness records `vcgencmd get_throttled` on a Raspberry Pi, and it cannot make a board stay cool. A soak shows the effect.
- **The sensor temperature in sunlight.** The simulator's sensor reads the ambient temperature plus a constant rise, so `day-sim` cannot show how warm a sunlit camera runs, or what that does to the dark library (see [The sensor temperature](#the-sensor-temperature)).
- **A real camera.** Jitter in the frame arrival, drops, and the recovery ladder.
- **A multi-day soak.** Memory growth, file handle leaks, retention, and the long-term behavior of the three processes.
- **The plate solver.** A first solve of a frame needs `solve-field` or ASTAP, which run outside Python. The `survey` case gives the pipeline the solution of a previous frame, as every frame after the first one has. The architecture's solver table estimates the first solve.
- **A busy `web` and the rest of `core`.** The `core-sim` case polls `web` every 5 s, as one open page does, and it configures no sink. Several browsers, the live views of the alignment, the preview encoder, the sink forwarder, and the daily retention are not part of it.

## The whole-system case

The `core-sim` case runs the system and reads it from outside. It starts `acquire` with the simulator, `core`, and `web` from the plan of `seeingmon dev` (`seeingmon.services.dev.build_plan`), on a free port of the loopback interface, with a temporary data folder and none of the settings of the person who runs it. The case changes no code of the system: it has no profiling hook and no extra counter.

**The system.**

- The sensor is the reference sensor (`full`), so the survey frames have the bin2 size of the real camera, 4144 × 2822 pixels, and the survey worker peaks where it peaks on a real night.
- The clock runs at speed 1, in real time.
- The fast stream uses the fast mode of the architecture (bin1, a 128 × 128 region, an exposure of 2 ms), which the readout of the sensor stretches to about 82 frames per second, and the real Polaris. The windows are those of the dev launcher, 20 s long. In production they are 60 s long.
- The scheduler runs a fast stream, then a survey step (a short and a long exposure), then waits for the next slot, which comes every 3 minutes. The run waits for two survey steps and for the results of their four frames. It takes about 6 minutes.
- One client polls `web` every 5 s with three requests (status, the latest seeing, and health), as an open page would.
- The scheduler and `acquire` wait up to 20 s beyond the frame period for a frame. The simulator renders a survey frame inside the read, which takes longer than the default margin of 0.5 s on a slow or busy machine. The scheduler then counts a camera error, and it never completes a survey step: a run on Windows ended after one step in 12 minutes. The longer margin changes no work that the system does.

**What the case reads.**

- **Memory.** The peak resident size of each process, which the operating system keeps: `VmHWM` on Linux and `PeakWorkingSetSize` on Windows. The survey worker is a child of `core`, and the case finds it in the process tree. On Linux, the resource tracker of the `multiprocessing` module is another child, and the figures call it `other`. The case also reads the resident size of each process every second, for the figure in the fast phase.
- **CPU time.** Every second, the CPU time of each process (`/proc/<pid>/stat` on Linux, with a tick of 10 ms, and `GetProcessTimes` on Windows, with a tick of about 15.6 ms), and on Linux the CPU time of each thread (`/proc/<pid>/task/<tid>/stat`). Windows has no such reading here, so it gives the process only. The share of a core in a phase is the CPU time of the phase divided by its length.
- **The state of the scheduler.** The `status` call of `core`, which `web` also makes, says which stream runs and how many frames, windows, and survey frames the scheduler has handled. The samples use it to tell the phases apart.

**The phases.**

- *Fast*: the fast stream runs at 30 frames per second or more, no survey frame waits for its analysis, and the first window is over. The imports and the first allocations of the analysis happen in the first window.
- *Paused*: the run ends with the `Pause` command, which the `web` page has a button for. No frame flows, so what `core` still uses is the load that does not belong to the frames: the scheduler loop, the store, the health timer, and the answers to `web` and to the case. The case reports it apart.
- The cost of a frame in `core` is the difference of the two shares divided by the frame rate. It covers the receive, the fast path, and the append of the segment, and it does not cover the load of the paused scheduler. The case reports it as `core.frame_cost` and as a share of a core at 98 frames per second, the rate of the budget (`core.fastpath_receive_share`). The row "fast path and receive" of the budgets reads that share.

**What includes the simulator.** The simulator renders every frame inside `acquire`, so the CPU time of `acquire` in this case includes it, and so does the peak memory of `acquire`: the simulator holds several arrays of the whole frame while it renders a survey frame, and each array of 32-bit floats takes 47 MB. The budgets keep reading `acquire` from the `ipc` case, which uses prebuilt frames, and the page shows the figures of the whole system apart and marks them. The peaks of `core`, the survey worker, and `web` are real, because none of them runs the simulator.

**Sample interval and run length.** The case samples once per second. A full run takes 6 to 7 minutes of sampling after the processes start. A smoke run takes about 20 s of sampling, because the scheduler searches for Polaris in its first bursts before it measures, and 35 to 50 s with the start of the processes.

**Run it.**

```bash
seeingmon perf run --cases core-sim --label dev --json local/perf/system.json
seeingmon perf report local/perf/system.json --details
```

**The day and the cloudy night.** The cases `day-sim` and `cloudy-sim` run the same system from another plan (`seeingmon.perf.cases.visibility_sim`): a start at noon of midsummer, or a winter night under an opaque overcast, and the cycle of production. The cloudy night adds a search phase, the search period of the cycle with its bursts and the gaps between them, which the samples split into the intervals with the frames of a burst and the gaps (see [A cloudy night of searching](#a-cloudy-night-of-searching)). Slow tests in `tests/perf/test_visibility_sim.py` run both full plans.

The extras `survey` and `web` must be installed. Other work on the machine changes the CPU figures, so run the case when the machine is quiet, and read the line `machine N% busy` and the figure `run.machine_busy`. The figure is the load of all processors during the run, and `system_share_percent` in its detail is the share that the system itself used. On Linux in a virtual machine, the load covers the virtual machine only, and the host can be busy without showing. The `--smoke` mode uses the `small` sensor and a few seconds of sampling. A slow test (`test_the_full_run_reaches_the_steady_state_and_two_survey_steps` in `tests/perf/test_core_sim.py`) runs the full plan, and the nightly workflow of CI runs it with the other slow tests. It checks that the run gets through the phases and that the figures exist, and it checks no size.
