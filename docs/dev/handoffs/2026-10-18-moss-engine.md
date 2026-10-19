# Handoff: MOSS bounded streaming checkpoints complete

**Date:** 2026-10-19
**Spec:** `docs/dev/specs/2026-10-18-moss-engine-design.md`
**Plan:** `docs/dev/plans/2026-10-18-moss-engine.md`

## Status

Tasks 1–10 complete; plan remains In progress, spec In Review. Task 10 checkpoint
subject: `feat(moss): stream bounded audio with durable chunk checkpoints`.
The previous planner baton is unchanged at
`archive/2026-10/2026-10-18-moss-engine-chunk-planner.md`.

## Next step

Task 11: open `backend/src/services/ingestion/pipeline_diarize.py` native-route
detection and consume the v2 namespace interface below. Owner already authorized
tasks 10–13. No repeated intent approval is needed.

## Verification state

Packed isolated command (backend cwd, DB_PROVIDER=sqlite and DATABASE_URL=sqlite://
set before imports):
`.venv/Scripts/python.exe -m pytest src/test_moss_pipeline.py src/test_moss_chunks.py src/test_moss.py src/test_phonon.py -q`.
Final suite result, scoped Ruff, SDD validation, whitespace and hook results are
recorded in the plan. No production settings, media, database or GPU inference.
Real FFmpeg normalization of a generated 5401-second-plus-one-frame PCM WAV is
covered with instrumented inference: 46 bounded calls, exact source frame and
PCM SHA256 coverage including silence. This is not a real long CUDA speed or
recognition claim. Actual multi-chunk recognition/seam/GPU proof remains task 13.

## Blockers

None for task 10. Task 11 must extend the native identity route before claiming
complete v2 ingestion: current pipeline_diarize detects only legacy moss_result.
The existing identity regressions deliberately write legacy artifacts so their
original matching and lifecycle coverage remains explicit. New v2 transcription
and checkpoint reconstruction are tested independently here.

## Diagnosis worth keeping

`_process_moss_phase` normalizes once to disk, scans bounded PCM for SHA256 and
streams <=120-second canonical WAV requests through one manager. Installed
faster-whisper get_speech_timestamps defaults operate on bounded CPU windows;
local hints are offset to absolute frames. Optional VAD failure records its
exception class (no transcript/error details) and a bounded-cut fallback. VAD
never removes silence. Cancellation checks bracket requests and fingerprint
scanning. Strict per-request validation and canonical 44-byte headers remain.

Raw schema `moss-recording-v2` has engine=moss, moss_source containing size,
mtime_ns, sample_count, sample_rate=16000 and normalized pcm_sha256; chunk_count,
frontier (exclusive contiguous end frame), ceiling, and ordered chunks. Each
chunk contains stable id `chunk-00000000`, start_frame/end_frame, unmodified
moss-result-v1 result, runtime (pinned model_revision/source_revision required),
source (same complete identity), boundary_reason and vad_fallback. Complete raw
also retains route/video metadata, total_duration and convenience segments.

The existing partial metadata path owns the same manifest without chunks, and
existing JSONL path owns complete chunk records. Initial zero ownership is
atomic before first inference. Complete record append/flush/fsync precedes
atomic metadata publication. Resume re-normalizes/fingerprints source, validates
all committed evidence, pins, ordering and frontier, then truncates only the
uncommitted tail. Missing/corrupt committed data, source mismatch, unknown/mixed
engine ownership and gaps fail without deletion. Metadata failure can leave an
uncommitted appended record, which is retranscribed; committed requests skip.
Force explicitly clears partials only; old complete raw survives failed replacement.
Final raw publishes atomically only at complete coverage, then partials clear.

Only MossResourceError triggers bisection of the failed uncommitted interval.
Children remain contiguous; repeated failures terminate at one frame (explicit
split ValueError). Successful resource_split request reduces persisted ceiling
for subsequent input. Failed split plans need not persist: resume begins at
committed frontier with last successful ceiling. General worker/validation
failures never authorize splitting. Successful chunks reuse the same manager.

`_moss_checkpoint_segments` dual-reads legacy moss_result and v2. Legacy format
and complete reuse remain unchanged. V2 checks source stat provenance and all
chunk evidence/pins/coverage; identity-only retry performs no normalization,
VAD, inference or repeated clock correction. Exact Fraction(local clock) plus
Fraction(start_frame,16000) produces adapter float clocks. Native local result,
raw clocks/corrections/text/order/overlaps and words=None remain untouched.
Each v2 adapter segment exposes `chunk_id`, `local_speaker` and `speaker` equal
to `chunk_id + ':' + local_speaker`; convenience raw segments retain both source
components too. These namespace keys must reach existing clean-span extraction
and channel matcher in task 11; local labels alone are never global identities.
No diarization or speaker_identity implementation changed in task 10.

## Next worker prompt

Task 11 only, resume:no. Repack native-route/identity files and tests explicitly.
Use v2 reconstructed namespaced segments above and existing channel-scoped
voice matcher/profile behavior. Prove equal local labels can be distinct voices,
changed labels can match one saved identity, native overlaps remain and only
clean samples enroll. Preserve legacy complete reuse and identity-only retry.
Do not rework bounded streaming or expand into UI/task 13. Keep unrelated dirty
baseline paths untouched; no production reprocessing, dependency changes,
push/deploy or final archive.
