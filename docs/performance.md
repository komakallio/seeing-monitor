# Performance

The architecture sets a performance gate for a Raspberry Pi 4 (see [architecture.md](architecture.md), "Processes, data rates, and storage"). This page describes the harness that measures the gate on a development machine, the results of a run, the Pi 4 estimate that follows from them, and the commands that measure it on a Pi 4. No Pi 4 was available while the harness was built (blocker B2), so every Pi 4 figure on this page is an estimate. The estimate states its assumptions, and a run on a Pi 4 replaces it.

## Summary

On the estimate, the per-frame analysis fits its budget with a wide margin, and the transport between `acquire` and `core` does not.

| Budget | Limit | Dev machine | Pi 4 (estimate) | Verdict | Verdict in the six runs |
|---|---|---|---|---|---|
| `acquire`, the production code, bin1 at 98 fps | 10% of a core | 6.8% | 18 to 40% | fail | Fail in six |
| `acquire`, what the fake camera of the tests adds | Not counted | 0.05% | | | |
| Fast path, bin1 at 98 fps | 25% of a core | 0.35% | 1.8 to 3.8% | pass | Pass in six |
| Fast path and receive, bin1 at 98 fps | 25% of a core | 4.3% | 19 to 35% | marginal | Fail in five, marginal in one |
| Fast path, bin2 at 360 fps | 25% of a core | 1.1% | 5.5 to 12% | pass | Pass in three, marginal in three |
| Fast path and receive, bin2 at 360 fps | 25% of a core | 14.4% | 99 to 158% | fail | Fail in six |
| Survey frame, bin2 (derived, not a gate) | 180 s | 1.5 s | 7.3 to 16 s | pass | Pass in six |
| Survey worker, peak memory | 550 MB | 454 MB | 318 to 590 MB | marginal | Marginal in six |
| All processes, peak memory | 1.4 GB budget, 1.6 GB gate | 762 MB | 683 to 1,290 MB | pass | Pass in six |

The table shows the least disturbed of three Linux runs. The last column gives the verdicts of all six runs: three on Linux, and three on Windows, which is slower in the bin2 fast path. The Pi 4 column is an estimate: read [How the estimate works](#how-the-estimate-works) before you rely on it.

- **The fast path has room.** One bin1 frame takes 32 µs through `FastPathAnalyzer` on the dev machine, and the kernel takes 23 µs of that. The Pi 4 estimate for the kernel (0.11 to 0.25 ms) agrees with the architecture's 0.2 to 0.4 ms.
- **The stream between the processes is the risk.** The production code of `acquire` spends 0.69 ms of CPU on a frame, and `core` spends 0.41 ms on receiving it. Over the three Linux runs, the figures range from 0.69 to 1.0 ms and from 0.41 to 0.60 ms. Both costs come from Python threads that move each frame, with no array computation. The work alone puts `acquire` at 13 to 20% of a Pi 4 core, over its 10% budget, even if wake-ups cost nothing. The camera is not the cause: the fake camera of the tests adds about 5 µs to a frame (see [What the camera adds](#what-the-camera-adds)). At 360 frames per second, the receive alone costs an estimated 93 to 146% of a Pi 4 core. The costs belong to each message and each frame, not to the bytes (see [Where the stream spends its CPU time](#where-the-stream-spends-its-cpu-time)).
- **The survey path has room in time and little in memory.** A bin2 frame takes 1.5 s on the dev machine (1.5 to 3.4 s over the three Linux runs), including 0.5 s for the sky quality step, and the worker peaks at 454 MB.
- **Two gigabytes of memory is enough on this evidence.** The estimated peak of all processes stays under the 1.4 GB budget and the 1.6 GB gate (see [Memory](#memory)).
- **Rust for the per-frame metrics is not indicated** (see [The Rust decision](#the-rust-decision)).

## What the harness measures

`seeingmon perf run` runs eight cases. Each case runs in a fresh child process, so the peak memory of one case never includes another. The child sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` to 1, so that every figure is per core, which is how the budgets read. A case imports the code that it measures when it runs, and it skips with a reason when a component is not on `main` yet, so the harness works at every commit.

| Case | What it measures | Feeds |
|---|---|---|
| `calibration` | Five fixed workloads: a pure-Python loop, a matrix product, an FFT, a SQLite insert batch, and a thread hand-off. The ratio of a Pi 4 run to a dev-machine run on them replaces the assumed scaling. | The scaling table |
| `kernel` | One call of the fast-path kernel in three modes: bin1 128 × 128 at 16 bits (the planned fast mode), bin2 64 × 64 at 16 bits, and bin2 320 × 240 at 8 bits (the format of the 10 ms recordings). | The per-frame budgets |
| `fastpath` | The whole per-frame path of `FastPathAnalyzer`: the kernel, the metrics row, the window bookkeeping, the segment append once per second, and the close of a 60 s window (the estimator, the scintillation index, and the spectrum), as a share of one core at the frame rate. | The 25% budget |
| `ipc` | The cost of moving a frame from `acquire` to `core`: the production `AcquireService` in its own process with a camera that costs nothing per frame, and a `RemoteCameraDriver` that reads the frames. The case reads the CPU time per frame of each side, split by thread, and splits it again into the work and the wake-ups. It also runs the bin2 stream at 360 fps, a stream with the fake camera of the tests, and a timing of one `read_frame` call of three cameras, which show what the camera adds. | The 10% budget of `acquire`, and the receive cost in the 25% budgets |
| `survey` | One synthetic bin2 survey frame (4144 × 2822 pixels, 30 s, rendered by the simulator from catalog stars) through `create_survey_analyzer` and the process worker of `make_process_executor`, with the sky quality step and one synthetic dark set: the wall time, the CPU time of the worker, each stage, the cost of the process boundary, the start of the worker, and the peak memory of the worker. | The survey budgets |
| `store` | Sustained result-row inserts, and the cost of the metrics-segment append per frame at 98 fps, in a temporary folder that the case deletes. | The 25% budget |
| `memory` | The resident size of a fresh process after it imports each part of the software: the fast path, the survey path, astropy, and the web stack. | The memory budgets |
| `core-sim` | A placeholder for the `core` process against `acquire` with the simulator. It skips until `measure_core` exists (see [Enable the core case](#enable-the-core-case)). | The memory budgets |

The sections below name each figure the way a report does (`<case>: <figure>`).

## Budgets

The Pi 4 budget comes from [architecture.md](architecture.md). The harness adds three rows that the text implies, and it marks a derived row as not a gate.

| Budget | Limit | Where it comes from |
|---|---|---|
| `acquire`, bin1 128 × 128 at 98 fps | 10% of one core | The architecture. The harness counts the code of `acquire` and leaves the camera out, so the figure is a lower bound. |
| Fast path, bin1 128 × 128 at 98 fps | 25% of one core, 2.55 ms per frame | The architecture |
| Fast path and the receive from `acquire`, bin1 | 25% of one core | The same consumer in `core` pays for both |
| Fast path, bin2 64 × 64 at 360 fps | 25% of one core, 0.69 ms per frame | The architecture |
| Fast path and the receive from `acquire`, bin2 | 25% of one core | The same consumer in `core` pays for both |
| Survey frame, bin2 | 180 s (not a gate) | The survey interval: a frame must end before the next one |
| Survey worker, peak memory | 550 MB | The architecture |
| All processes, peak memory | 1.4 GB (the budget), 1.6 GB (the 2 GB gate) | The architecture: more than 1.6 GB at peak means the 4 GB model |

The harness sums the figures of each row. It reports `pass` when the whole estimated range is within the limit, `fail` when the whole range is above it, and `marginal` when the range straddles it. A report with the label `pi4` and a measured calibration case compares its figures with the limits directly, and the verdict is `pass` or `fail`.

## Run the harness

Run the commands on your development machine, from a clone with the extras installed (`uv sync --all-extras`, see [development.md](development.md)).

```bash
seeingmon perf run --smoke                                  # under a minute: every case works
seeingmon perf run --label dev --json local/perf/dev.json  # the full run
seeingmon perf report local/perf/dev.json --budgets         # the tables, the verdicts, and the checks
```

- `--cases NAME,...` runs some cases, and `--list` names them.
- `--smoke` shrinks every case to a tiny workload, and the figures then say nothing about speed. CI runs it as a test (`tests/perf/test_cases.py`). The test checks that every case runs, that its figures are finite and positive, and that the budgets find the figures that they read. It checks nothing about size.
- `--json PATH` saves the report. Use a path under `local/`, which Git ignores. The report holds the architecture, the operating-system family, the Python and library versions, the processor model, and the commit. It holds no host name, user name, or serial number.
- `--label` names the machine class. Use `pi4` on a Raspberry Pi 4.
- `--quiet-wait SECONDS` waits up to that long before each case for the machine to be at most 15% busy. Other work disturbs a timing, and each case records how busy the machine was.
- `seeingmon perf report A.json --baseline B.json` prints the ratio of every figure of A to the same figure of B, and whether the ratio lies in the range that the scaling table assumes. Use it to compare a Pi 4 run with a dev run.
- `seeingmon perf report A.json --details` also prints the details of each figure, such as the number of stars in the survey frame.

A figure is the median of its repeats, and the table also shows the minimum, the 95th percentile, and the maximum. A busy machine moves the median up and the maximum far up, so read the line `machine N% busy` of each case. The minimum is the least disturbed figure.

## Results on the development machine

The tables show two sets of runs on the same laptop, a recent x86-64 machine with performance and efficiency cores. One set ran Linux in a WSL 2 virtual machine (CPython 3.12.14, NumPy 2.5.3, SciPy 1.18.1), and the other ran Windows (CPython 3.13.13, the same libraries). Each set has three runs of the whole harness at commit `0a764af`. A cell that holds two numbers reads `Linux / Windows`. The Pi runs Linux, so the estimate uses the Linux run.

The laptop did other work during the runs, and the scheduler moves a process between fast and slow cores, so a figure changes from run to run. A disturbance only adds time. For that reason, each table shows the least disturbed run of each set, which is the run with the lowest sum of its key figures divided by the lowest figure of the three. The last table shows all three Linux runs.

### Fast path, per frame

| Mode | Kernel (µs) | Whole push (µs) | Close a 60 s window (ms) | Share of a core |
|---|---|---|---|---|
| bin1 128 × 128, 16 bit, 98 fps | 22.8 / 18.8 | 32.3 / 118 | 10.4 / 26.1 | 0.35% / 1.23% |
| bin2 64 × 64, 16 bit, 360 fps | 18.7 / 17.0 | 27.9 / 84.8 | 28.7 / 94.5 | 1.08% / 3.27% |
| bin2 320 × 240, 8 bit, 98 fps | 23.5 / 20.7 | 43.3 / 92.8 | 17.8 / 23.4 | 0.47% / 0.98% |

The push is one call of `FastPathAnalyzer.push`: the kernel, the metrics row, and the window bookkeeping. The share of a core adds the window close and the segment append to it, at the frame rate. The push is about 90% of the share. On Windows, the kernel looks small next to the push, probably because the two cases ran at different times, and the speed of the laptop changed between them.

### Stream between `acquire` and `core`

| Figure | `acquire`, the production code | `core` receive |
|---|---|---|
| CPU per frame, paced at 98 fps (µs) | 691 / 1,076 | 405 / 604 |
| CPU per frame, bursts of 10 frames (µs) | 238 / 924 | 239 / 576 |
| Share of a core at 98 fps, work | 1.84% / 8.89% | 2.17% / 5.61% |
| Share of a core at 98 fps, wake-ups | 4.94% / 1.66% | 1.80% / 0.31% |
| Share of a core at 98 fps, total | 6.78% / 10.5% | 3.97% / 5.92% |

`acquire` runs with a camera that costs nothing per frame: it waits for the frame period and returns a frame that exists already. So the table counts the code of `acquire` and leaves the camera out. The case splits the work from the wake-ups with two runs: a paced stream costs the work plus a wake-up for each frame, and a stream in bursts of 10 frames costs the work plus a tenth of the wake-ups. The Linux virtual machine makes wake-ups expensive, and the Windows figures come from a coarse CPU clock, so the split is one of the least reliable figures on this page.

On Linux, `acquire` spends 327 µs per frame in the capture thread, 334 µs in the sender thread, and 30 µs elsewhere, and it peaks at 58 MB. A bin2 stream at 360 fps costs `core` 369 µs per frame to receive (13.3% of a core on Linux, 26.1% on Windows).

### What the camera adds

The services tests use a fake camera that builds a `Frame` on every read, and the real ASI driver builds one too, so the question is how much a camera adds to the capture thread. The first six rows below come from whole runs. The sender thread is a control, because it runs the same code whatever the camera is, so a difference between the two runs that shows in the sender thread is drift of the laptop, and not the camera.

| Figure (µs per frame) | Linux / Windows |
|---|---|
| Capture thread, zero-cost camera | 327 / 428 |
| Capture thread, fake camera as it is | 435 / 696 |
| Sender thread, zero-cost camera (control) | 334 / 560 |
| Sender thread, fake camera as it is (control) | 436 / 698 |
| `acquire` as a whole, zero-cost camera | 691 / 1,076 |
| `acquire` as a whole, fake camera as it is | 915 / 1,440 |
| One `read_frame` call, fake camera | 5.0 / 4.4 |
| One `read_frame` call, zero-cost camera | 0.16 / 0.17 |
| One `read_frame` call, production ASI driver on a stub SDK | 18.7 / 15.5 |

The fake camera runs came after the zero-cost runs in the same case, and the laptop was slower during them: the sender thread cost 102 µs more on Linux, and the capture thread cost 108 µs more. The camera explains 6 µs of that difference. The three rows at the end of the table time one `read_frame` call in a loop with no waiting, so they have none of that noise. The fake camera costs 5 µs a call. The production ASI driver costs 16 to 19 µs a call with a stub SDK: it takes a lock, arms the call watchdog, copies the pixels, and builds a `Frame` with its checks, and a real driver adds the vendor library and the USB stack on top. A one-off run of eight alternating pairs on Linux agreed: the fake camera added a median of 35 µs to the capture thread and 32 µs to the sender thread, which leaves 3 µs for the camera.

So the camera is a few percent of the 327 µs of the capture thread, whichever camera it is, and the capture code of `acquire` owns the rest: the gate, the guard, the time stamper, the queue, and the bookkeeping. The verdict on `acquire` rests on the zero-cost runs, which count what the code of `acquire` costs and leave the camera out. A real driver adds 0.15 to 0.18% of a core on the dev machine (16 to 19 µs at 98 fps), so the figure is a lower bound. The sender thread and the receive path of `core` do not depend on the camera, so their verdicts stand, and the services lane can profile them (see the next section).

### Where the stream spends its CPU time

The services lane needs a starting point for the work on the sender thread and the receive path, so here is a profile of 2,000 frames at 98 fps on Linux (CPython 3.12, commit `0a764af`). It uses `yappi` with the CPU clock, because `cProfile` of Python 3.12 and later allows one profile for the whole process and cannot tell the threads apart. `acquire` ran with the zero-cost camera, and `core` was a `RemoteCameraDriver` that reads the frames. The profile is one-off, and it is not part of the harness. A profiler adds overhead, so read the shares and not the microseconds. A share includes the functions that the function calls, so the rows of a table overlap.

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

The costs belong to each message and each frame, not to the bytes. Three changes would cut them, and they are the first to try. Read the acknowledgements once per batch, and build the selector once. Acknowledge every few frames instead of every frame. Send several frames in one message. On the estimate, the 10% budget of `acquire` needs 1.8 to 4 times less CPU per frame than it takes now. The fast path with the receive needs up to 1.4 times less in bin1 (up to 2.5 times in the slower Linux runs) and 4 to 6 times less in bin2 at 360 fps.

### Survey frame

| Figure | Linux / Windows |
|---|---|
| Frame through the analyzer and the worker (s) | 1.46 / 2.13 |
| Stage `detect` (s) | 0.784 / 2.16 |
| Stage `solve`, by the tracker (s) | 0.0113 / 0.0159 |
| Stage `match` (s) | 0.00187 / 0.00315 |
| Stage `quality`, the sky quality step (s) | 0.459 / 0.651 |
| Stage `records` (s) | 0.00109 / 0.00187 |
| Process boundary, the job minus its stages (s) | 0.077 / 0.049 |
| CPU time of the worker for a frame (s) | 1.44 / 2.03 |
| Start of the worker: spawn, imports, catalog (s) | 1.83 / 2.78 |
| Worker memory after the start (MB) | 89.6 / 86.9 |
| Worker peak memory (MB) | 454 / 429 |

The frame has 4144 × 2822 pixels, 1,888 detections, and 1,394 stars that match the catalog, and the zero point uses 904 of them. The stages come from three separate jobs and the total from five frames, so on a noisy run they do not add up to the total. The peak of the worker was 444 MB on Linux before the case ran the sky quality step, so the step adds about 10 MB: the detection step already holds the largest arrays. The `solve` stage is the tracker, because the case gives the pipeline the solution of a previous frame.

### Store and calibration

| Figure | Linux / Windows |
|---|---|
| Result row, one transaction each (µs) | 100 / 220 |
| Result row in a batch of 100 (µs) | 100 / 209 |
| Metrics append, fsync every 60 s of frame time (µs per frame) | 0.109 / 1.50 |

| Calibration workload | Linux (ms) | Windows (ms) |
|---|---|---|
| `python_loop` | 84.0 | 118 |
| `numpy_matmul` | 3.87 | 3.61 |
| `numpy_fft` | 3.25 | 4.17 |
| `sqlite_insert` | 15.3 | 24.2 |
| `thread_handoff` | 181 | 102 |

The segment append costs under 0.01% of a core at 98 fps, so the store does not matter for the budget. The `thread_handoff` row shows that wake-ups are expensive in the Linux virtual machine: the same hand-offs take 1.8 times longer there than on Windows, while most other workloads run at the same speed or faster on Linux.

### Spread of three runs

| Figure | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| Kernel, bin1 (µs) | 19.4 | 26.3 | 22.8 |
| Whole push, bin1 (µs) | 56.2 | 38.3 | 32.3 |
| `acquire` CPU per frame, paced (µs) | 997 | 958 | 691 |
| `core` receive CPU per frame, paced (µs) | 567 | 603 | 405 |
| Survey frame (s) | 3.36 | 1.91 | 1.46 |
| Survey worker peak (MB) | 453 | 453 | 454 |
| `python_loop` (ms) | 94.6 | 152 | 84.0 |
| `thread_handoff` (ms) | 250 | 351 | 181 |

The last column of the summary table gives the verdict of each of the six runs. Two verdicts depend on the run: the fast path with the receive in bin1 is `marginal` in the least disturbed Linux run and `fail` in the other five, and the bin2 fast path is `pass` in the three Linux runs and `marginal` in the three Windows runs.

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
| Fast path, bin1 | High for `pass` | The figure is 32 µs per frame of NumPy and bookkeeping. The limit holds up to a factor of 70 on the Linux run and 20 on the Windows run, against the 5 to 11 that the estimate assumes. |
| Fast path, bin2 at 360 fps | Medium for `pass` | The limit holds up to a factor of 23 on the Linux run. The Windows run gives `marginal`, because its limit is a factor of 8. |
| `acquire`, bin1 | Medium for `fail`, low for the percentages | The verdict is `fail` in all six runs, and the work alone fails (13 to 20%), so it does not depend on the wake-ups. The wake-up range is a guess, and the figure leaves the camera out, which a real driver adds (see [What the camera adds](#what-the-camera-adds)). |
| Fast path and receive, bin1 | Low | The least disturbed run gives `marginal` and five runs give `fail`, because the receive costs 0.41 to 0.60 ms per frame on Linux, and a busy machine moves the figure. |
| Receive, bin2 at 360 fps | Medium for `fail` | The case runs the stream once. The estimate is about a whole core or more in all six runs, so a different range does not change the verdict. |
| Survey frame | High for `pass` | The estimate is 7 to 37 s in the six runs, against 180 s. |
| Survey worker, peak memory | Low | The range of 0.7 to 1.3 times straddles the limit, so only a Pi 4 run settles it. |
| All processes, peak memory | Medium for `pass` | The `core` row is a stand-in, the `web` row counts imports only, and the operating system's share is an assumption (see [Memory](#memory)). |

### The kernel check

The architecture estimates 0.2 to 0.4 ms for the kernel on a 128 × 128 frame on a Pi 4, and it says that the kernel takes about 100 µs on the dev machine. The `seeingmon perf report --budgets` command divides the first figure by the measured time, and it compares the factor that results with the table.

| Run | Kernel (µs) | Factor that the architecture implies | Compared with the table's 5 to 11 | Estimate from the table (ms) |
|---|---|---|---|---|
| Linux, run 1 | 19.4 | 10.3 to 20.6 | Consistent | 0.10 to 0.21 |
| Linux, run 2 | 26.3 | 7.6 to 15.2 | Consistent | 0.13 to 0.29 |
| Linux, run 3 | 22.8 | 8.8 to 17.5 | Consistent | 0.11 to 0.25 |
| Windows, run 1 | 29.8 | 6.7 to 13.4 | Consistent | 0.15 to 0.33 |
| Windows, run 2 | 71.5 | 2.8 to 5.6 | Consistent | 0.36 to 0.79 |
| Windows, run 3 | 18.8 | 10.6 to 21.3 | Consistent | 0.09 to 0.21 |

In every run, the factor that the architecture implies overlaps the range of the table, although the runs differ by a factor of four in the kernel time. The estimate from the table brackets the architecture's 0.2 to 0.4 ms, so the estimate stays. The slowest Windows run is the only one that ends near the edge of the table: its implied factor starts at 2.8, below the 5 of the table, and the two ranges overlap from 5 to 5.6.

## Memory

The harness adds the peaks of the processes on the dev machine, multiplies the sum by the `memory` range, and adds the share of the operating system.

| Process | Dev machine peak (MB) | Pi 4 estimate (MB) |
|---|---|---|
| `acquire` with the zero-cost camera | 58 | 40 to 75 |
| `core`, a stand-in: the fast-path process (145) plus the store process (52) | 197 | 138 to 256 |
| Survey worker | 454 | 318 to 590 |
| `web`, a lower bound: the imports only | 53 | 37 to 69 |
| Operating system, an assumption | | 150 to 300 |
| Sum | 762 | 683 to 1,290 |

The estimated sum is under the 1.4 GB budget and under the 1.6 GB gate that separates the 2 GB model from the 4 GB model, so this evidence does not call for 4 GB. Four limits apply:

- The `core` row is a stand-in. It adds two whole processes, so it counts the shared imports twice, and it leaves out the parts of `core` that the harness does not run, such as the three survey frames that stay in RAM and the preview encoder. The `core-sim` case replaces it when it exists.
- The `web` row counts what the imports take. A web process that serves connections takes more.
- The architecture's conditions for 2 GB are design rules that the harness does not test: the out-of-memory killer takes the survey worker first, calibration frames stay memory-mapped, and the Pi processes bin2 frames only.
- The survey worker straddles its 550 MB limit by itself, so it is the first figure to read in a Pi 4 run.

The `memory` case reads the resident size of a fresh process after it imports each set of modules. The sets show where the baseline of each process comes from. Multiply them by 0.7 to 1.3 for a Pi 4 (the imports of `core` take an estimated 62 to 115 MB).

| Set of imports | Resident size, Linux / Windows (MB) |
|---|---|
| Python and the memory reader | 19.5 / 22.9 |
| NumPy | 33.9 / 31.9 |
| Frame and record types (pydantic) | 34.4 / 33.4 |
| Fast-path analyzer (SciPy) | 45.9 / 41.7 |
| SQLite store and segment writer | 45.0 / 41.9 |
| Survey analyzer (SEP, pyerfa) | 83.3 / 81.6 |
| `astropy.coordinates` and `astropy.time` | 73.2 / 67.0 |
| FastAPI, uvicorn, Pillow | 53.1 / 53.3 |
| Everything that `core` imports | 88.7 / 87.9 |

## The Rust decision

The architecture keeps Rust (PyO3) as the replacement for the per-frame metrics if the Pi 4 gate fails. On the estimate, the per-frame metrics do not fail the gate: the whole fast path takes 1.8 to 3.8% of a core in bin1, against 25%. Rust would replace the kernel, which is 23 of the 32 µs of a frame, and it would not touch the send and receive costs that fail the gate. Keep Python for the per-frame metrics.

The number that flips the decision is a Pi 4 run in which the fast path (the `fastpath` case, without the receive) takes more than 25% of a core. That is 2.55 ms per frame in bin1 at 98 fps, and 0.69 ms per frame in bin2 at 360 fps. It takes a Pi 4 that runs 20 to 70 times slower than the dev machine in bin1, and 6 to 23 times slower in bin2 (the ranges cover the six runs), against the 5 to 11 times that the estimate assumes. The `calibration` ratios of a Pi 4 run show which side to expect. Even then, bin2 at 360 fps fails on the receive first, so the stream is the first thing to change.

## Measure on a Pi 4

The Pi 4 measurement stays blocked (blocker B2), because no Pi 4 was available while the harness was built. You run these commands on the Pi, and you copy one JSON file back. The report holds no host name, user name, or serial number.

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

3. Run the harness. The full run takes about 2 minutes on the dev machine and an estimated 5 to 15 minutes on a Pi 4. The `survey` case needs about 1 GB of free memory for its two processes.

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

- **USB.** The camera, the SDK, and the USB transfer. The `ipc` case uses a camera that costs nothing per frame, so the cost of the vendor library and of the USB stack of the kernel is missing from the `acquire` figure. The Python of the production ASI driver (16 to 19 µs per frame on a stub SDK) is missing too, and the camera table shows it apart.
- **The SD card.** The `store` case writes to the disk of the machine. An SD card is slower, and its fsync can take tens of milliseconds. The segment writer calls fsync once per minute of frame time, so the effect is small, and the Pi 4 run shows it.
- **Thermal throttling and the supply.** The harness records `vcgencmd get_throttled` on a Raspberry Pi, and it cannot make a board stay cool. A soak shows the effect.
- **A real camera.** Jitter in the frame arrival, drops, and the recovery ladder.
- **A multi-day soak.** Memory growth, file handle leaks, retention, and the long-term behavior of the three processes.
- **The plate solver.** A first solve of a frame needs `solve-field` or ASTAP, which run outside Python. The `survey` case gives the pipeline the solution of a previous frame, as every frame after the first one has. The architecture's solver table estimates the first solve.
- **The rest of `core`.** The scheduler loop, the preview encoder, the sink forwarder, retention, and the web process under load. The `core-sim` case is meant to measure them.

## Enable the core case

The `core-sim` case skips with the reason "the core process is on main, and `measure_core` is not written yet". The entry function `run_core` exists in `seeingmon.services.core.main`. To enable the case, replace the body of `measure_core` in `src/seeingmon/perf/cases/core_sim.py`. The docstring at the top of that file lists the steps and the two figures that the budgets read: `cpu_share` and `peak_rss`. When the case runs, the memory budget uses its peak in place of the stand-in from the `fastpath` and `store` cases.
