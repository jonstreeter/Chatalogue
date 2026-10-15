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
