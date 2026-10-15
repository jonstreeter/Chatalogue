# Optional Phonon-2 engine

Phonon is an explicitly selected English transcription engine. Existing Auto,
Parakeet, Whisper and language-routing defaults are unchanged. Selecting it does
not reprocess existing media. Phonon supplies word timestamps, not diarization;
the existing diarization phase follows transcription normally.

## NVIDIA CUDA setup

Install Docker with NVIDIA GPU support separately. Chatalogue does not install
Docker, launch containers per episode, or manage runtime lifecycle. Use the
verified image `ghcr.io/fermionresearch/phonon-cuda:1.0.8`, digest
`sha256:c37e7b0ae99f6a74a2618ecae3fe9cbbc342c1069e02464eadba9c42d19802d5`.

Create a private, uncommitted environment file containing
`PHONON_API_KEY=<a generated random key>`. For example, from the repository root:

```powershell
python -c "import secrets; from pathlib import Path; Path('backend/data/phonon.env').write_text('PHONON_API_KEY='+secrets.token_urlsafe(32)+'\n')"
docker run -d --name chatalogue-phonon --gpus all -p 127.0.0.1:8010:8000 --env-file backend/data/phonon.env -v phonon-cache:/home/phonon/.cache ghcr.io/fermionresearch/phonon-cuda:1.0.8 serve phonon-2 --host 0.0.0.0 --port 8000
docker logs chatalogue-phonon
```

Weights download once into `phonon-cache`. Wait for the serving message. The
internal Docker non-loopback bind requires authentication even though the host
port is published only on loopback. Keep the env file private; enter its key in
Settings → Transcription → Phonon-2, with URL `http://127.0.0.1:8010`.
Test Selected Engine checks health/model identity and makes a one-second silent
transcription to prove authentication. It uses unsaved URL/key values and does
not save settings. Save Changes persists the selection and connection settings.

Leave the advanced inference deadline blank to wait for inference, or enter a
positive number of seconds. HTTP connection establishment uses httpx's default
five-second connect timeout. Redirects and environment proxies are refused;
only loopback HTTP(S) URLs without embedded credentials, query or fragment work.

Restart the same container with `docker start chatalogue-phonon`. Stop it with
`docker stop chatalogue-phonon`. Its GPU memory remains allocated between jobs,
including during diarization/cloning. Chatalogue's host model purge cannot unload
it. Account for that independent VRAM ownership when using staged processing.
Rollback: select your previous engine and stop the optional service.

## CPU and Apple silicon

Use a separate virtual environment rather than altering Chatalogue's pinned ML
environment. Upstream supports CPU on Windows, Linux and macOS, and MLX on Apple
silicon. Install `fermion-research` for CPU or `fermion-research[mlx]` for MLX,
then run `fermion serve phonon-2 --host 127.0.0.1 --port 8010 --api-key YOUR_KEY`.
The same app connection settings and word-timestamp adapter apply. This is a
different runtime/device choice; CUDA benchmark numbers do not apply. Native
CPU/MLX performance has not been measured in this implementation session.
See [upstream installation](https://github.com/fermionresearch/phonon#install)
and [server API](https://github.com/fermionresearch/phonon/blob/main/docs/server.md).

## Long audio, errors and resume

The app normalizes once to 16 kHz mono PCM16, then sends bounded 600-second
windows with the existing 0.35-second overlap. Actual multipart Content-Length
must fit the runtime's 32 MiB body limit; each input is below its two-hour limit.
Episodes may exceed both limits because they are split before uploading. Requests
are sequential; the upstream GPU server serializes decoding. BF16, warmed CUDA
graphs (16 steps and six buckets in the verified image) retain upstream defaults.
There is no batch-128 serving option.

Word ownership changes at overlap midpoints, with absolute timestamps in app
checkpoints. Every successful chunk commits segments and a completion offset,
including silent chunks. Interrupted append tails are discarded on resume; a
missing committed record or a partial from another/unknown engine fails visibly.
Use the original engine to resume such a partial, or the existing deliberate
full-retranscription action. Runtime errors retain prior completed chunks and
never silently fall back to another engine. Pause/cancel takes effect at request
boundaries; an in-flight request finishes, fails, or reaches the configured deadline.

401/403 means check the key; connection errors mean check the service and URL;
413 means the runtime rejected upload size; 429/503 means busy/unavailable; decode
errors may require freeing GPU memory. Runtime response bodies and keys are not
included in app errors. Test readiness before queueing a backlog.

## Integrated benchmark reproduction

Use fresh output filenames and ignored credential/input files. This runner uses
in-memory SQLite and scratch files, never production jobs/settings/transcripts:

```powershell
backend/.venv/Scripts/python.exe scripts/benchmark_asr.py app backend/data/asr-benchmark/trespasses-600-720.wav --app-engine phonon --phonon-env backend/data/phonon.env --output backend/data/asr-benchmark/app-phonon-new-short.json
backend/.venv/Scripts/python.exe scripts/benchmark_asr.py app backend/data/asr-benchmark/trespasses-600-3480.wav --app-engine phonon --phonon-env backend/data/phonon.env --output backend/data/asr-benchmark/app-phonon-new-long.json
```

Both runs use the same persistent runtime. The report records runtime health and
request counters before/after, actual app provenance and structural timestamp
checks. See [measurements](research/2026-10-15-transcription-performance.md).
Recognition and alignment accuracy require a corrected reference and listening;
structural timing checks alone do not establish either.
