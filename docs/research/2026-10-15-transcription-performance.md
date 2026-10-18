# Phonon-2 versus Chatalogue's configured transcription pipeline

Research and local measurements, October 15, 2026. Repository HEAD: `9482348`.

## Recommendation

Keep the current engine for now. Phonon-2 warrants a quality evaluation as an
optional English engine: it showed a useful long-audio decoding improvement and
has a much smaller download. It did not show an order-of-magnitude decoding
improvement over the project's Parakeet pipeline. Recognition accuracy, word
alignment accuracy, diarization quality, and total ingestion speed were not
established by this experiment. No production engine or settings were changed.

## Baseline verified from this checkout

`backend/.env` selects Parakeet TDT v2, sequential execution, batch request 16,
and retention of Parakeet between jobs. The configured Whisper alternative is
`insanely_fast_whisper`, small, FP16, beam 2, with batching enabled. Multilingual
routing is enabled and can select a different model. These are persisted checkout
settings; no live backend process or recent production job provenance was verified.

The baseline invokes the actual `IngestionService._process_transcribe_phase`,
including language detection, model placement, adaptive batching, long-audio
chunking, fallback, output conversion, and checkpoints. A local checkpoint lookup
replaces only NeMo's network/cache resolution. The configured model ID and all
transcription methods remain unchanged. Parakeet was the engine actually used in
the successful baseline results; there was no Whisper fallback.

The harness uses in-memory SQLite and redirects ingestion file paths into a
temporary benchmark directory. No workers, production jobs, or transcript writes
are started. Metadata is synthetic and has no stored language, so the app runs its
Whisper large-v3 language probe. Real videos with known language may skip that work.

## Measurements

Hardware: RTX 5090, 32607 MiB reported VRAM. Local Windows app venv versus Linux
CUDA container under Docker. Engines ran sequentially, not against each other on
the GPU. Other machine activity was not experimentally controlled.

Backend versions: torch 2.11.0+cu128, NeMo 2.7.0, faster-whisper 1.2.1,
Transformers 4.57.1, soundfile 0.13.1. Downloaded Parakeet repository revision:
`ae9ad07059c7c739ffaf932226a8fe64ae2620b0`.

Both excerpts come from the same local recording, beginning at 10:00. The short
excerpt is 120 seconds. The long excerpt is 2880 seconds (48 minutes), selected to
exercise the app's existing always-chunk threshold of twice 1440 seconds.

| Metric | 2-minute excerpt | 48-minute excerpt |
| --- | ---: | ---: |
| Actual app phase, first run | 139.137 s | 256.244 s |
| Actual app phase, subsequent run in same service | 7.345 s | 64.672 s |
| Parakeet inference, subsequent run, NeMo progress timings | ~1.39 s | ~48.46 s across five chunks |
| Phonon first inference, runtime `decode_s` | 1.0627 s | 31.1273 s |
| Phonon subsequent inference, runtime `decode_s` | 1.2563 s | 30.7687 s |
| Phonon paired process, startup + both inputs + output | 40.249 s | 100.771 s |
| Phonon reported `load_s`, once per process | 35.06 s | 33.07 s |
| Parakeet output word timestamps | 365 | 8501 |
| Phonon output word timestamps | 384 | 8706 |

The warm decoder observations imply approximately 1.1x and 1.6x speed differences
on short and long excerpts respectively. NeMo's progress durations and Phonon's
`decode_s` have different instrumentation boundaries; treat those ratios as
approximate observations, not controlled comparative benchmarks. Full app phase
timings include work absent from Phonon's decode metric, so dividing 64.672 by
30.7687 does **not** establish an end-to-end application speedup.

The long app run selected 600-second chunks, 0.35-second overlap, and no model
recycling. It completed without fallback. Actual single-input batch size was 1,
despite a requested batch size of 16. Peak app PyTorch allocated memory was about
2.43 GiB for the short excerpt and 2.80 GiB for the long excerpt. This excludes
CTranslate2 and other GPU allocations; no comparable Phonon peak was measured.

Startup timings are environment-sensitive. A repository validator was running
during part of the long baseline's model restore and was stopped; the long cold
time is unsuitable as a controlled startup comparison. Phonon's load metric
includes graph warmup; do not compare only its inference time to app startup.

All reported outputs passed structural word-timestamp bounds and ordering checks.
This does not prove words are correctly aligned or recognized. Different word
counts are disagreement, not evidence of which engine is better. No corrected
reference transcript was available, so WER was not measured. These are exploratory
tests on one English conversation, not a broad accuracy or performance evaluation.

## Why the release is relevant

[Fermion's release](https://www.fermionresearch.com/research/phonon-2/) describes
Phonon-2 as an English model derived from Parakeet TDT v3, with a 164 MB download.
Its developer-reported average WER is 5.21%, versus 4.96% for full Parakeet v3 and
6.58% for Whisper large-v3-turbo in the supplied table. The social post's Whisper
comparison is not a comparison against Chatalogue's configured Parakeet v2.

The model is much smaller on disk than the downloaded v2 checkpoint
(2,472,222,720 bytes), but its NVIDIA encoder expands to 16-bit weights. Download
size does not imply the same reduction in VRAM.
[Fermion's runtime documentation](https://www.fermionresearch.com/docs/speech/)
provides word timestamps and a local transcription API. The tested GPU path is a
Docker runtime; native Windows GPU integration is not established here. Phonon
provides transcription, not recurring-speaker identity or diarization.

## Artifacts and reproduction

- Runner: `scripts/benchmark_asr.py` (`--help` documents its arguments).
- Input excerpts and JSON reports: `backend/data/asr-benchmark/` (ignored local data).
- Configured checkpoint: `models/parakeet-v2/parakeet-tdt-0.6b-v2.nemo` under that directory.
- Successful app reports: `app-parakeet-120s.json`, `app-parakeet-2880s.json`.
- Phonon reports: `phonon-120s-paired.json`, `phonon-2880s-paired.json`.
- Earlier `app-120s.json` is an offline cache-miss fallback experiment, **not** the
  Parakeet baseline. `phonon-120s.json` is the earlier single-input exploratory run.
- `review-app-120s.txt` and `review-phonon-120s.txt` contain paired short
  transcripts for review against `trespasses-600-720.wav`.
- Docker image: `ghcr.io/fermionresearch/phonon-cuda:1.0.8`, digest
  `sha256:c37e7b0ae99f6a74a2618ecae3fe9cbbc342c1069e02464eadba9c42d19802d5`.

```powershell
backend/.venv/Scripts/python.exe scripts/benchmark_asr.py app backend/data/asr-benchmark/trespasses-600-720.wav --parakeet-checkpoint backend/data/asr-benchmark/models/parakeet-v2/parakeet-tdt-0.6b-v2.nemo --output backend/data/asr-benchmark/app-new-run.json
backend/.venv/Scripts/python.exe scripts/benchmark_asr.py phonon backend/data/asr-benchmark/trespasses-600-720.wav --output backend/data/asr-benchmark/phonon-new-run.json --phonon-cli cuda --phonon-repetitions 2 --engine-audio /audio/trespasses-600-720.wav --command docker run --rm --network none --gpus all -e HF_HUB_OFFLINE=1 -v F:/Projects/PorchtimeIndex/backend/data/asr-benchmark:/audio:ro -v phonon-cache:/home/phonon/.cache ghcr.io/fermionresearch/phonon-cuda:1.0.8
```

Use the `trespasses-600-3480.wav` input for the long case. Choose fresh output
filenames; the runner refuses to overwrite previous results.

## Follow-up decision

The next useful evidence is paired recognition and alignment quality against
human-corrected excerpts, including overlap, names, and interruptions. If Phonon
passes that comparison, consider an optional engine adapter and measure its full
integrated phase. This research does not authorize or implement that integration.
The separate diarization candidates from the initial research remain unbenchmarked.


## Integrated optional-engine implementation measurements

2026-10-15, same RTX 5090 and source excerpts. Actual app transcription phase,
in-memory SQLite and temporary files. Process-local Phonon selection; production
settings, queues and stored transcripts were not used. Language routing remained
enabled, so each synthetic episode still ran its normal Whisper language probe.
One persistent CUDA 1.0.8 container served both runner processes and all chunks.

| Final actual app phase | 120 s input | 2880 s input |
| --- | ---: | ---: |
| First phase in runner process | 9.813 s | 46.257 s |
| Second phase in same runner | 7.305 s | 62.129 s |
| External chunks per phase | 1 | 5 |
| Output words per phase | 384 | 8709 |
| Output segments per phase | 41 | 1045 |
| Structural timestamp issues | 0 | 0 |

The earlier warm Parakeet app observation was 7.345 s / 64.672 s. Final Phonon
warm observations are approximately equal on short audio and about 1.04x faster
on long audio. The first final long phase was faster, but host activity was not
controlled and backend tests ran during this experiment. No consistent large
integrated speedup is established. These results include normalization, language
routing, upload, checkpoint persistence and conversion, rather than decoder time
alone. Startup was outside these phase timings: the container loaded once in
26.2 seconds, including upstream warmup. The health remained BF16, graph steps 16,
six warm buckets; no optimization flags were overridden.

During final short/long experiments uptime increased from 723.0 to 863.9 seconds,
and the request counter increased from 16 to 31 before the last readiness probe.
Across the entire container lifetime it accumulated 12012.6 audio seconds and
126.965 decode seconds. Readiness sends one second of silence; counters include
these probes. Overlap makes uploaded seconds exceed source duration. The final
outputs retain upstream segment boundaries for normal diarization. The slightly
different word count from the earlier single-file run reflects chunk boundaries;
recognition and alignment accuracy remain unmeasured.

Earlier implementation reports (without upstream segment grouping) measured
20.222/7.686 seconds on short audio and 42.662/44.569 seconds on long audio.
Those are retained as exploratory artifacts, not the final implementation result.

Final ignored reports: `backend/data/asr-benchmark/app-phonon-integrated-120s-final.json`
and `app-phonon-integrated-2880s-final.json`. Reproduce with `--app-engine phonon`
and `--phonon-env <ignored key file>` using `scripts/benchmark_asr.py app`; the
runner records health, actual provenance and timestamp checks and refuses
overwrites. CPU/MLX throughput and WER were not measured. No batch-128 throughput
is claimed. Setup and operation: [optional Phonon engine](../phonon-engine.md).


## Native Windows CUDA feasibility — 2026-10-16

Approved integration revision task 7: PASS. The unmodified inference sources
copied from verified CUDA 1.0.8 image digest
`c37e7b0ae99f6a74a2618ecae3fe9cbbc342c1069e02464eadba9c42d19802d5`
ran natively on Windows / RTX 5090 without a container/server or custom kernels.
The probe used a disposable Python 3.11.6 venv with private Transformers 5.17.0,
NumPy 2.4.6, huggingface-hub 1.33.0, soundfile 0.14.0 and existing PyTorch
2.11.0+cu128 made visible through a probe-only .pth file. This differs from the
Linux image's PyTorch 2.9.1+cu128/NumPy 2.3.4. Product installation must install
its own pinned environment; the probe's shared-package mechanism is not shipped.
The backend Transformers remained 4.57.1, unchanged.

First probe: cold constructor 129.466s (upstream model expansion 39.0s, warmup
12.0s; constructor also includes first dependency import). Two decodes of the
same 120-second source: 2.504s / 4.100s, identical text hashes, 383 words and 41
segments each. Follow-up through the committed benchmark command: constructor
42.660s, decodes 3.722s / 1.770s, zero structural timestamp issues; reported
phase wall 50.493s excludes imports before its timer. Neither run establishes
cold startup as cheap or a consistent app speedup. Host activity was uncontrolled.

Runtime reports dense BF16, K16 graphs, five-second buckets, six warmed buckets,
GPU mel and energy-gated long audio. PyTorch reports 1,332,585,984 allocated bytes
and 1,570,766,848 reserved bytes after inference. The first actual CUDA child
PID 56052 and its venv launcher 55228 exited and were absent from process and
NVIDIA compute-process queries. The initial parent exit report checked the
launcher PID, so the actual child was additionally checked explicitly. On this
Windows WDDM host NVIDIA reports per-process memory as N/A: do not infer a
measured byte delta from global VRAM under unrelated host activity. Successful
process exit ends its owned CUDA context; integrated lifecycle tests remain due.

Reproduce from repo root with the probe private interpreter:

```powershell
backend/data/asr-benchmark/phonon-native-probe/venv/Scripts/python.exe scripts/benchmark_asr.py phonon-native backend/data/asr-benchmark/trespasses-600-720.wav --native-source backend/data/asr-benchmark/phonon-native-probe/upstream --native-model backend/data/asr-benchmark/phonon-native-probe/model --output backend/data/asr-benchmark/phonon-native-probe/native-benchmark-new.json
```

Source manifest: ignored `phonon-native-probe/source-manifest.json`; full reports
`native-probe.json`, `native-benchmark.json`; environment snapshot
`environment-freeze.txt`; notices `LICENSE-upstream` (Apache-2.0) and
`MODEL-CARD.md` (CC-BY-4.0). These are research artifacts, not an installed model
runtime. Only inference files needed for the worker will be redistributed with
notices. Do not package the upstream HTTP/WebSocket server.

Pinned source SHA-256: `phonon2_cuda_engine.py`
`931d8aaf0b0ed640bd5f818749c3c9f957b213323fe1a2421493a1497ec0f8e0`.
Model revision resolved from the public hub:
`7e153e4054c0a10db6d47d6fff5c062d4c77c154`.
Verified model SHA-256: `model.fermion`
`4b6bfa3a12cc3c4e0a54f2ab3ec4ca7a842b09e5c7ecfc8e7ca0ac6cc8c11468`;
`config.json` `d0daad3b2a182893844f4abdc11e4f5b7083f7d42c8ad7e8203f71559785a31b`;
`packed_manifest.json` `3c1874501a1efb6ef569eab082aa68d91b93ec6e4d2fb00dbef7851edf2ae510`.

This proves native optimized inference feasibility, not completed app integration,
long-audio app behavior, recognition accuracy, or supported CPU/MLX operation.


### App-owned IPC worker — 2026-10-16

Task 9 real runtime proof used the fully standalone private environment, not the
probe's shared backend packages. Readiness plus two real 120-second requests
used the same actual interpreter PID 42932. Model load/readiness: 55.729s;
requests: 4.745s / 4.691s, 383 valid words each. Runtime counters advanced 1→2→3
(includes one-second silent readiness). BF16, K16 graphs, six warmed buckets
and GPU mel remained active. These are IPC adapter timings under uncontrolled
host activity, not complete app-phase or accuracy measurements.

An anonymous Windows Job Object owns the suspended venv launcher before it runs,
so its real Python descendant inherits ownership. The manager verifies actual
interpreter membership before loading GPU weights and retains a synchronization
handle until unload confirms that interpreter exited. No unrelated process is
stopped. Actual GPU unload passed and all IPC audio files were removed. Abrupt
parent exit is additionally regression-tested with a real venv child/fake model;
no claim of a measured global VRAM byte delta on WDDM is made.

Report: ignored `backend/data/asr-benchmark/owned-worker-probe.json`.
The source result schema is text/segments/words/duration; timed intervals are
validated before return. The pinned decoder bounds tokens by `max_sym * T + 16`
per internal window (rounded to K-step execution), using a finite vocabulary.
Inputs retain the app's 32 MiB/two-hour bounds. The local pipe has no invented
response byte quota; source ownership, fixed protocol/IDs and typed timing
validation reject malformed replies without accepting partial chunk success.
The worker starts no listening HTTP/WebSocket service.


### Managed worker through actual app phase — task 10

An isolated in-memory SQLite app ran `_process_transcribe_phase` twice on the
120s fixture, normalizing and checkpointing temporary copies. Runtime setup/load
was measured separately: 44.599s. Phase timings: 5.780s / 6.824s, both actual
`phonon`, 383 words, no structural timestamp issues, same worker PID 18868 before
and after. The harness closes the worker on exit and redirects Python temp paths.
Language routing was disabled in this process only to isolate English-engine
behavior; language routing/defaults are preserved and tested separately. This
is not full episode time including download, diarization, cloning or other jobs.
Report: ignored `app-phonon-managed-120s-task10.json`.

The existing end-of-job 1.5 GiB global free-memory-drop heuristic would evict an
intentionally retained child and defeat cross-job reuse. It now recognizes a
loaded Phonon worker; unchanged host reserved/allocated checks still trigger
recovery. Regression cases cover retained worker, no worker and actual host
allocator pressure. Language probing through GPU Whisper and actual competing
GPU phases still release Phonon by design and can add reload time; staged mode
alone does not eliminate every reload when audio-language probing is needed.


### Final integrated native long-audio measurement — 2026-10-16

Actual app phase, 2880s (48 minutes) of authorized fixture audio, in-memory
SQLite and temporary copies; language routing disabled only in the benchmark
process. Separate worker startup/readiness: 46.025s. Two complete app phases:
125.762s / 128.842s (22.90x / 22.35x real time). Both produced 8704 words,
1043 upstream-preserving segments and five committed chunks, with no structural
word-clock issues. Same actual PID 22728, request counter 1→11 including initial
silence, BF16/K16/six warmed buckets and actual Phonon provenance throughout.
The worker was stopped by the harness on exit.

This native run is slower than the earlier Docker HTTP app measurements
(46.257s / 62.129s for the same 2880s fixture). Host conditions and dependencies
were not controlled as a cross-platform performance experiment. Full integration
is established; a throughput improvement over that prior implementation is not.
Do not substitute decoder-only timing or promise batch-128/backlog performance.
Neither WER nor listening-based alignment accuracy was measured.
Report: ignored `backend/data/asr-benchmark/app-phonon-managed-2880s-final.json`.
The benchmark now rejects a missing/invalid explicit native cache before loading,
so its offline claim cannot silently initiate an installation.


### Final short-audio and GPU handoff measurement — 2026-10-16

Actual 120s app phases: 6.021s / 7.136s with the same worker PID 56284
(counter 2→3); initial load/readiness separately 46.664s. Both returned 383 words,
41 segments and no structural timing issues. The third phase used the app's
actual `_release_phonon_model` boundary: unload 0.202s, then 49.451s for the next
transcription including reload, with new worker PID 3688. That phase also returned
383 valid words/41 segments. This measures release plus subsequent app reload,
not the inference duration of a competing diarization/Whisper/cloning model.
The full backend suite separately verifies the actual diarization handoff hook.
Defaults and production settings were not changed; controlled English routing was
disabled only in the benchmark process. Reports exclude downloading, queue wait
and diarization. No recognition-accuracy or batch throughput claim follows.
Report: ignored `backend/data/asr-benchmark/app-phonon-managed-120s-final.json`.
Reproduce with `scripts/benchmark_asr.py app <wav> --app-engine phonon
--phonon-runtime backend/runtime/phonon --phonon-handoff --output <fresh-json>`.

### Production episode observations and closeout — 2026-10-18

Owner-provided completed queue screenshots compare the same 3:11:16 episode,
“Gamblers Anonymous,” using each engine:

| Queue metric | Phonon-2 | Parakeet (Coexist) |
| --- | ---: | ---: |
| Transcription | 4:16 (45x realtime) | 3:21 (57x realtime) |
| Diarization | 5:28 | 5:06 |
| Combined processing | 9:44 | 8:28 |
| Wall-clock elapsed | 10:19 | 11:22 |

Parakeet transcription took 55 seconds less (21.5% less time). Phonon completed
sooner by wall clock in this observation, but elapsed time exceeded combined
processing by 35 seconds versus 2:54; the screenshots do not identify that
overhead. These are comparable practical alternatives, not evidence of a Phonon
speed advantage. Warm/cold state and concurrent activity were not controlled;
accuracy was not measured. Diarization is the largest measured processing stage.

Follow-up fixes are committed and pushed: setup wizard selection, download
compatibility, truthful queue provenance, name-repair performance/progress,
stable episode-page metadata snapshots (`2292ce8`), and cold diarization resume
voice-model initialization (`6de54ee`). The latter fails before replacing rows
if voice inference cannot initialize; isolated regression coverage exercises
both successful matching to an existing voice and preservation on load failure.
Final code verification: 114 backend tests passed; Ruff, frontend typecheck,
frontend lint (zero errors, 43 existing warnings), build, SDD validation and
published CI passed. Browser verification confirmed the episode interface and
recognized speaker names. The live-data destructive E2E suite was not run.

The owner-authorized backed-up diarization-only repair of episode 16962,
“Modest Mice,” completed as job 31420 in 4:00.880, reusing saved ASR words.
After consolidation: 3916 segments, 3439 assigned and 477 unassigned (12.2%).
Recognized profiles include Chris Hanna (1599 segments), Jared Higdon (926)
and Flip Johnson (297); other segments retain generic speaker labels.
These counts establish restored matching, not listening-verified attribution
accuracy. The original 4210-row transcript backup remains under
`backend/data/temp/redo_diarization_backup_16962_1792331485036.json`.

Phonon implementation remains complete: the archived plan has all 13 tasks done.
The local app is running the pushed fixes; no active jobs were present at
closeout. Optional runtime caches, benchmark evidence and the repair backup are
retained. Existing defaults and unrelated uncommitted development work are not
part of this closeout. Future work can start from the committed code baseline;
use a separate worktree when it overlaps the existing unrelated dirty files.
