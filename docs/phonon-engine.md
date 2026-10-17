# Integrated Phonon-2 engine

Phonon is an optional English transcription engine. Chatalogue installs and owns
its private runtime, loads it when needed and releases it for competing GPU work,
explicit unload or app shutdown. No Docker, server, URL or API key is required.
Existing Auto, Parakeet, Whisper, language routing and defaults are preserved.
Selecting it does not enqueue or reprocess existing media.

## Setup and operation

The verified native platform is Windows x64, Python 3.11 and an NVIDIA CUDA GPU
with hardware BF16 support (tested on RTX 5090). The driver must support the
pinned PyTorch 2.11.0 CUDA 12.8 runtime. Other platforms/hardware show an actionable
unsupported message; use an existing engine there. CPU/MLX are not implemented
in this integrated revision. Native Windows support is locally verified, not an
upstream Windows deployment guarantee.

In Settings, choose **Phonon-2 (English)**, then **Set up and test Phonon**.
The model card shows setup/download/loading progress. Initial setup needs internet
access to official Python wheel sources and Hugging Face: the Torch wheel alone
is about 2.75 GB, plus dependencies and about 178 MB of model files. Allow space
for downloaded wheels, the expanded private environment and temporary setup.
The backend's existing ML environment is unchanged. No paid service is used.

A successful test performs real one-second silence inference and displays actual
model/device information. **Save Changes** persists the engine selection;
setup/test alone does not save it. Installed models load lazily on actual use.
**Load and test Phonon**, **Test Phonon model** and **Unload Phonon** use the same
owned runtime. Busy inference refuses manual unload rather than losing work.
Status checks never download or load a model. Advanced settings retain an optional
positive inference deadline; blank waits for completion. Pause/cancel takes effect
at chunk boundaries unless the request fails or reaches that deadline.

Installation lives below `backend/runtime/phonon`: a versioned private venv,
verified immutable model files and resumable download cache. Completed setup works
offline. Failed setup never publishes a partial installation; retry setup after
restoring connectivity or disk space. Model downloads resume acknowledged byte
ranges and verify hashes. Wheel installation may redownload on retry. App restart
retains the cache but starts with the model unloaded. Logs are under the runtime
root; UI errors omit raw child output. Legacy URL/key settings are accepted as
ignored compatibility inputs and saved secrets are never returned.

Rollback: select your previous engine. Keep the optional cache and Phonon
checkpoints; resuming a Phonon partial requires its original engine or the existing
explicit full-retranscription action.

## Reuse, limits and GPU handoff

The app normalizes to 16 kHz mono PCM16 and processes bounded 600-second windows
with the existing 0.35-second overlap. Temporary WAV input stays below the retained
32 MiB and two-hour per-request bounds. Episodes can exceed both because they are
split first. Requests are serialized; there is no batch-128 throughput mode.
The packaged upstream CUDA 1.0.8 inference source retains BF16, graph steps K16
and six warmed duration buckets. Source hashes and licenses accompany the worker.

Word ownership changes at overlap midpoints; stored timestamps are absolute.
Every successful chunk commits segments and a completion offset, including silence.
Interrupted append tails are discarded on resume; missing committed records and
foreign/unknown engine partials fail visibly. Errors preserve committed work and
never silently mix engines. Upstream segment groups remain available to diarization.
Queue provenance reports the engine that actually transcribed the recording.

The worker stays warm across consecutive transcription jobs. Whisper/Parakeet GPU
loads, language probes, diarization and GPU cloning/reconstruction phases release
it first. Sequential transcription/diarization therefore pays a later reload cost;
staged transcription can reuse it when no competing GPU phase intervenes. Language
routing still sends non-English or mixed audio to Whisper when enabled. A global
purge and app shutdown stop the owned child and release its CUDA context.

## Integrated benchmark reproduction

Set up the optional runtime first. Use authorized test WAVs and fresh output
filenames. App mode requires an explicit verified cache, uses in-memory SQLite
and temporary copies, and calls the actual transcription phase twice. It excludes
queue waiting, downloading and diarization. Cache misses fail without installing.
For controlled English engine measurements, disable routing only in the benchmark
process (never edit the application's saved settings):

```powershell
$env:MULTILINGUAL_ROUTING_ENABLED='false'
backend/.venv/Scripts/python.exe scripts/benchmark_asr.py app SHORT.wav --app-engine phonon --phonon-runtime backend/runtime/phonon --output backend/data/asr-benchmark/short.json
backend/.venv/Scripts/python.exe scripts/benchmark_asr.py app LONG.wav --app-engine phonon --phonon-runtime backend/runtime/phonon --output backend/data/asr-benchmark/long.json
Remove-Item Env:MULTILINGUAL_ROUTING_ENABLED
```

The report separates startup from app phase time, records runtime identity/PID,
request counters, actual provenance and structural word-clock checks. Repeated
runs reuse the same child within each benchmark process. Add `--phonon-handoff`
to include a third app transcription after the real competing-GPU release
boundary; that phase includes reload but excludes competing model inference. See the
[measurements and limitations](research/2026-10-15-transcription-performance.md).
Structural timestamp checks do not establish recognition/alignment accuracy;
that requires a corrected reference and listening. Native cold load is substantial,
and decoder-only ratios are not integrated backlog throughput guarantees.
