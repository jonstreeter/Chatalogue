# Optional MOSS Engine — Design Spec

**Date:** 2026-10-18
**Status:** In Review
**Owner:** Jon / Codex
**Plan:** `docs/dev/plans/2026-10-18-moss-engine.md`
**Scope:** App-managed MOSS with native speaker turns, segment timing, channel voice matching and bounded, resumable inference chunks for long recordings. Model request limits are not recording limits; existing engines remain available.

---

## Motivation

Jon requested an option like the recently shipped Phonon integration after
evaluating MOSS's joint transcription/diarization, overlap handling and speed.
Cross-recording named voice recognition is explicitly required. Published
benchmarks are motivation, not acceptance evidence for this installation.

Current transcription produces Whisper-style segments; pipeline_diarize.py runs
pyannote clustering, then extracts voice embeddings and matches channel-scoped
Speaker profiles. Its final word-to-timeline mapping chooses one speaker at a
time. Applying that mapper to MOSS would discard the model's explicit attribution
during overlapping speech. The integration must bypass clustering and that
mapper for MOSS while retaining identification and existing persistence behavior.

## Design tenets

1. Add `moss` explicitly; keep auto, Whisper, Parakeet and Phonon behavior and defaults.
2. Own a private native worker/environment/cache as with Phonon. No operator
   server, URL/key configuration, Docker prerequisite or backend ML-pin upgrade.
3. Retain existing voice embedding model, channel-scoped profiles, names,
   thresholds, manual corrections and matching rules. MOSS labels never become
   persistent identities directly. Identity model failure fails the job visibly.
4. Preserve MOSS's speaker label on every segment, including overlapping times.
   Do not infer that label again from a single active-speaker lookup.
5. Report actual capabilities: MOSS has segment times, not documented word times.
   Store `words=null`; never fabricate uniform word clocks or silently run another
   ASR. UI exposes the segment-timing limitation before selection is saved.
6. Reuse existing lifecycle/process-ownership mechanisms where safe. Do not
   introduce a generic engine plugin framework or copy Phonon inference kernels.

## Data flow

```text
Settings / SetupWizard: moss -> existing settings and test-engine endpoints
  -> IngestionService -> managed private MOSS worker -> normalized local audio
  -> validated native {start,end,speaker,text} segments + runtime provenance
  -> atomic raw transcript checkpoint carrying MOSS speaker turns
  -> existing embedding extraction and channel voice-profile matching
  -> explicit MOSS label-to-Speaker mapping -> existing transcript persistence
  -> segment playback/search/export and queue engine badge
```

## Bounded recording batching — owner-directed amendment, 2026-10-19

This amendment is normative and supersedes earlier whole-recording rejection
and whole-input retry language. The owner reported a real job rejected by the
90-minute guard and directed the app to accommodate model limits through chunks.
That direction authorizes this refinement and implementation without another
approval of the same intent. No production recording is reprocessed by verification.

### Request sizing and audio continuity

- Normalize the recording once to mono PCM16 16 kHz on disk. Stream bounded
  frame ranges; do not read an entire long recording into a Python bytes object.
  Separate errors for empty audio, unsupported normalized format and per-request
  bounds; do not claim every validation failure means a recording is too long.
- Initial target ceiling is 120 seconds per request, derived from the successful
  task-8 120-second native CUDA app proof, not a new model limit or speed promise.
  Each actual request also satisfies the worker's 5400-second ceiling and pinned
  131072-token context/completion checks. Do not increase the measured target by
  intuition or expose tuning controls solely for implementation convenience.
- Prefer the latest detected silence gap within the current bounded window using
  the already installed faster-whisper Silero VAD and its installed defaults.
  VAD selects a cut only: it must never remove audio, skip silence, compress time,
  assign speakers, or manufacture an empty transcript. Record a bounded-cut
  fallback if that optional boundary hint is unavailable. No new dependency or
  backend ML upgrade. CPU VAD is preprocessing, not CPU MOSS inference.
- Chunks own contiguous non-overlapping sample intervals covering every original
  normalized frame exactly once. When no usable silence boundary exists, cut at
  the bounded frame limit; record the hard seam. Do not infer word boundaries,
  trim text, deduplicate by a single-speaker midpoint, or discard a native turn.
  Continuous-speech seams require real-audio verification and disclosure; there
  is no claim of lossless ASR recognition at a forced cut.
- A typed CUDA OOM or typed context-budget exhaustion may bisect only the failed,
  uncommitted interval and retry the resulting smaller intervals. Preserve prior
  committed chunks. Use the successful reduced ceiling for the remaining input.
  Stop when the same failure cannot be reduced to positive smaller frame ranges.
  Never classify failures by matching diagnostic strings; malformed generation,
  invalid clocks, worker crashes and unsupported hardware fail visibly without
  hidden splitting, invented retries, CPU or remote fallback.

### Immutable evidence, checkpoints and retry

- Keep moss-result-v1 unchanged for each request. Introduce a versioned
  moss-recording-v2 raw artifact containing normalized total sample dimensions,
  a normalized PCM SHA256, source provenance and ordered chunk records. Each
  record retains its stable sequence ID, exact start/end frames, original local
  native result, runtime/source/model provenance, boundary reason and any
  per-chunk moss-eof-v1 corrections. Validate every local result against the
  actual supplied frame count before committing it.
- Global segment clocks are exact local rational clocks plus start_frame/16000.
  Keep original local clocks/corrections untouched; do not normalize twice or
  silently reinterpret a local end as the recording EOF. Preserve original native
  order and overlap within each chunk, text and words=null. Namespace attribution
  by stable chunk ID plus original local speaker label; retain both components.
- Use existing partial artifact paths with explicit engine=moss and a versioned
  chunk format. Append complete chunk evidence, flush/fsync, then atomically
  publish committed chunk count and contiguous completed frame frontier. Initial
  zero-chunk ownership must be explicit. Reuse the proven Phonon durability
  sequence rather than introducing a generic engine framework.
- On resume, normalize/fingerprint the current source and reject a mismatch.
  Validate all committed records, sample coverage, dimensions, ordering, model
  provenance and frontier. Truncate only an uncommitted tail; reject corrupt or
  missing committed data, gaps and mixed/unknown engine ownership without deletion.
  Partial split plans need not commit failed output; retry starts at the first
  uncommitted sample using the persisted reduced ceiling when available.
- Final raw artifact is atomic and published only when chunks cover the complete
  normalized source. Completed legacy raw transcripts and legacy single-result
  MOSS checkpoints keep their reuse semantics and format; do not invalidate them
  to force batching. Force processing follows explicit user intent, preserves a
  prior completed raw artifact until a replacement succeeds, and clears only
  relevant partial artifacts. Identity retry never retranscribes completed chunks.

### Identity and lifecycle

- Local S01 in two chunks is not evidence of the same person. Preserve namespace
  keys through reconstruction and native diarization, then use existing real
  channel-scoped embeddings, profiles, thresholds and in-job cache enrollment to
  match identities across chunks and recordings. Never persist local labels as
  identity or reset a known person's name merely because a chunk label changes.
- Compute clean embedding spans against all overlapping native turns in absolute
  time. Keep unresolved speakers distinct if usable evidence is absent; never
  enroll mixed voices or force a match to make chunks look consistent. Existing
  manual corrections, save/revisions and cleanup semantics remain.
- Reuse one owned MOSS process across successful chunks. Cancellation is checked
  between requests and retains committed progress. Verified worker exit still
  precedes voice-model GPU loading; busy unload, crash, shutdown and parent-exit
  ownership rules remain. Report chunk progress and completed input duration.

### Required proof and paired review

Unit/integration fixtures must cover recordings over 5400 seconds without passing
an oversized request; exact frame coverage including silence; safe forced seams;
typed OOM/context split and no split for malformed output; atomic chunk failures,
crashes, cancellation/resume, corrupt and mixed partials; per-chunk EOF evidence;
legacy reuse; and real matcher tests where equal local labels denote different
voices and changed labels denote the same saved voice. UI/browser tests must
replace the recording-limit/whole-input-retry notice with accurate batching and
resume behavior. Real CUDA multi-chunk speech checks must exercise at least one
seam, staged reuse, native overlapping attribution, identity and GPU handoff.
The over-90-minute coverage fixture may use instrumented inference; it must be
reported separately from real CUDA proof, with no long-runtime performance claim.

Paired review, 2026-10-19, inline architect/planner/simplification/code-reviewer:
BLOCKER local-label collision resolved by stable namespace plus existing matcher;
BLOCKER seam data loss resolved by exact contiguous ownership and no text/turn
discard; HIGH resume corruption resolved by append/fsync/frontier publication
and normalized source fingerprint; HIGH invented operating sizes resolved by
measured 120-second target and typed, frame-bounded reduction; HIGH legacy damage
resolved by dual-format reuse and atomic replacement. Alternatives reviewed:
keep rejection (does not meet owner outcome), overlapping windows with heuristic
text deduplication (risks losing native turns without word clocks), or contiguous
bounded requests (chosen). Boundary recognition accuracy remains a measured
limitation, never a fabricated guarantee. No necessary planning finding remains
open; implementation and real seam proof are still pending.

Remember to follow the MSW deletion rule for all claims — no exceptions.

## Interface contracts

### Client → API

- Extend transcription_engine with `moss`. Extend existing engine-test request
  with `moss_action=status|load|unload`, default load; retain Phonon inputs.
- Status is read-only. Setup/load is asynchronous and observable using the
  existing status/detail/error/model_loaded/process_running convention.
- Setup/Test never saves engine selection or enqueues production media. Save
  persists only explicit selection. MOSS is multilingual: its explicit route
  must not inherit the English-only engines' non-English reroute to Whisper.
- Controls match Phonon: setup/load/test, status, unload and actionable errors.
  Explain segment timing, automatic bounded chunks and completed-chunk resume in the MOSS panel.
  Unsupported hardware is reported; no silent CPU or remote fallback.

### Service → worker

- Windows NVIDIA CUDA is the initial verification platform. First implementation
  task proves native feasibility before exposing a usable engine in product UI.
- Pin immutable model and helper-code revisions and exact tested requirements
  after that probe. MOSS currently requires Transformers >=5.6,<6; never install
  those dependencies into the backend environment. Review pinned remote code
  before enabling its required trust_remote_code loading; retain license notices.
- Manager-owned temporary WAV paths and request IDs cross serialized JSONL IPC.
  Worker never imports application/database modules or opens a listening port.
- Response includes full transcript, parsed segments, source duration, completion
  reason and actual model revision/device/dtype/runtime timing. Reject malformed
  or unparsed non-whitespace output, nonfinite/out-of-range clocks, empty labels,
  duplicate protocol IDs and incomplete/token-limited generation. Overlap is
  valid: do not impose Phonon's globally increasing end-time validator.
- The sole permitted out-of-range clock correction is the EOF rule below;
  apply it only after complete-generation and structural validation.
- Use documented SDPA or supported optimized attention, not eager long-input
  attention. The output budget must fit the pinned model context after input
  tokens; record termination and fail on exhaustion, not accept a truncated file.

### Tiny EOF overruns (owner-authorized refinement, 2026-10-18)

- Let `D` be the duration of the exact normalized audio supplied to MOSS,
  calculated from decoded sample count / sample rate, not rounded metadata or
  the last generated clock. For each segment whose raw end `e` exceeds `D`,
  accept only `0 < e - D <= 0.01` seconds (inclusive) and normalize its end to
  `D`. In-range clocks remain unchanged. This is a fixed per-endpoint boundary
  allowance; never accumulate it across segments or scale it with input length.
- The 0.01-second value is an application policy derived from the task-1 measured
  `9.01` end for exactly `9.00` seconds, authorized by the owner's request to
  refine tiny boundary overruns. Pinned upstream examples use two decimals, but
  its parser accepts arbitrary decimal precision: this is not an upstream
  guarantee, a measured accuracy bound or a general timestamp-rounding rule.
- Require all raw clocks finite, `0 <= start <= D` and `start <= raw_end` before
  correction. Reject negative clocks, out-of-range starts, ends more than 0.01s
  beyond `D`, and any correction that leaves `normalized_end <= start`. Reject
  the complete result on any invalid segment; do not drop or split segments.
- Compare the original decimal clock tokens against sample-derived duration
  using exact decimal/rational arithmetic, or mathematically equivalent exact
  comparisons. No extra floating-point epsilon may widen the allowance. In
  particular, for `D=9`, `9.010000` is accepted and `9.010001` is rejected.
- Preserve full raw generation, raw endpoint tokens, native speaker labels and
  exact source sample count/rate with the completed raw checkpoint. Record each
  corrected segment index, raw end, normalized end and policy identifier
  `moss-eof-v1` in checkpoint provenance. IPC carries the same evidence; validate
  it at the backend boundary against the manager's actual input, rather than
  trusting a worker-supplied duration or corrected end alone. Do not change DB
  schema or add a user-facing tolerance setting.
- Persistence/playback and identity sample extraction use normalized endpoints.
  Identity retries reuse the completed checkpoint and its policy/provenance
  without cumulative correction or retranscription. Preserve labels, text,
  ordering, native overlaps and `words=null`. EOS, output coverage, context-limit
  and malformed-output checks remain strict; correction cannot rescue incomplete
  generation. Selecting another engine remains rollback.
- Task 1 must rerun the retained synthetic overlap fixture and demonstrate the
  accepted correction, bounded final clocks and native overlaps with raw evidence
  intact before task 2. Parser tests in task 2 and retry tests in task 5 cover
  the exact boundary, rejection and provenance rules. The prior strict failure
  remains historical evidence, not a retroactively passed test.

### Lifecycle / filesystem / persistence

- App-owned environment/cache lives under backend/runtime/moss; temporary IPC
  files use the existing app temp root. Stage installation and atomically publish
  a verified manifest. Serialize setup, preserve valid installs on failure, and
  support offline cached use. Status must not download or initialize CUDA.
- Reuse owned child lifetime/supervision from Phonon where compatible; serialize
  inference and lifecycle operations, reject busy user unload, release competing
  models before MOSS load, and stop MOSS before voice-model GPU loading.
  Verify child exit on shutdown and parent death. No global process-name kills.
- The 5400-second limit bounds an individual worker request, never the total
  recording. The batching amendment below supersedes the initial whole-input
  restriction and defines chunk sizing, identity, retry and checkpoint behavior.
- Cancellation is between model operations, with an owned worker termination
  path for app shutdown. Incomplete generation never replaces a complete raw
  transcript. Resume preserves completed native chunks and completed raw
  transcripts; retry only uncommitted audio or identity/persistence as applicable.
- Add engine/model/source provenance and native speaker labels to the existing
  raw JSON artifact. No new DB tables or columns: TranscriptSegment.words already
  permits null. Keep final identity, revision, cleanup and save paths consistent.
- Completed legacy transcripts retain current reuse semantics. Never reinterpret
  old pyannote RTTM as MOSS turns, mix another engine's partial into MOSS, or lose
  native labels on deserialize/retry. Force retranscription follows existing user
  intent and invalidates only relevant checkpoints. Selecting another engine is
  rollback; existing media, speaker profiles and cached MOSS weights survive.

## Edge cases & failure modes

- Silence may yield zero speech segments only on validated complete inference;
  invalid/undecodable audio fails. Laughter annotations remain original transcript
  text; no promised event ontology, emotion detection or confidence scores.
- Identity extraction uses existing clean sample selection; exclude concurrent
  other-speaker spans where possible so mixed audio does not contaminate profiles.
  When no usable sample exists, preserve an unresolved local speaker rather than
  invent a known match or enroll a mixed voice; disclose this in job results.
- Distinct labels with nested/overlapping times remain distinct through saving,
  reconstruction and export. Existing no-word UI fallback must be browser-tested.
- Model tests, purge, engine switching and staged/sequential queues share GPU
  ownership. A MOSS job must not retain a worker while embedding models load.
- Retry missing identity models, corrupt caches, download interruption, worker
  exit, context exhaustion and long-input OOM without marking the job complete.

## Adversarial review

2026-10-18: inline architect, planner, simplification-expert and code-reviewer
lenses; no subagents spawned. Spec review followed by final paired artifact review.

- BLOCKER architect: copying the existing timeline mapper loses cross-talk labels.
  Resolved: native label-to-Speaker mapping bypasses re-clustering and midpoint
  lookup; integration tests require overlapping/nested turns with distinct labels.
- HIGH architect: replacing diarization could remove recurring identity. Resolved:
  keep embedding/profile model and matching; verify saved named profiles in two
  recordings and failure behavior when the voice model is unavailable.
- HIGH planner: unknown native runtime and long-input memory. Resolved by task 1
  feasibility gate, measured environment pins, and stop before product work on
  failure. No native throughput or 90-minute memory-fit guarantee is made.
- HIGH planner: word clocks and resumability differ from Phonon. Resolved with
  visible segment-only capability and whole-input checkpoint/cancellation terms.
- HIGH code-reviewer: cached RTTM and parser truncation could silently corrupt
  attribution/completeness. Resolved through engine provenance, strict parsing,
  generation termination checks, retry tests and preservation of native labels.
- HIGH simplification: external server, new event schema, forced alignment and a
  generic runtime framework are unnecessary for this option. Excluded; lifecycle
  code extraction is permitted only where required to reuse process ownership.

Final spec+plan planning review passes: necessary design findings have explicit
resolutions and verifiable tasks. Owner approved the original plan and capability
tradeoffs on 2026-10-18 and subsequently authorized this tiny-overrun refinement.
Technical feasibility remains an implementation gate, not a claimed validation.

2026-10-18 refinement review, all four lenses applied inline; no review subagents:
architect compared strict rejection, fixed EOF normalization and broad rounding;
selected only measured EOF normalization. Resolved HIGH trust-boundary finding
by retaining raw tokens and independently checking manager input duration.
Planner resolved HIGH proof gap with task-1 rerun, task-2 exact-boundary tests and
task-5 idempotent checkpoint retry. Simplification rejected configurable tolerance
and general rounding as unnecessary. Code-reviewer resolved HIGH floating-point
and collapsed-span findings with exact comparisons and rejection after correction.
Paired spec/plan review passes; no necessary design finding remains open. Runtime
rerun and product tests remain unperformed under this new rule.
Remember to follow the MSW deletion rule for all claims — no exceptions.

## Out of scope

Defaults change, automatic reprocessing, remote/server deployment, CPU/Mac support
claims, word alignment, DB migration, new
cross-channel identity semantics, structured vocal-event features and speed or
accuracy guarantees. These exclusions do not remove existing engine capabilities.

## Open questions

Original segment-timing, 90-minute and whole-input-resume tradeoffs were approved
for the initial implementation; the owner superseded the latter two on 2026-10-19
by directing automatic batching of recordings to accommodate model limits.
The owner authorized the tiny EOF-overrun refinement above after task 1's strict
failure. Exact runtime/model pins and measured GPU behavior are in the research
report. Task 1 must pass its targeted rerun under this policy before task 2;
another failure stops the build. No further original-plan approval is pending.

Sources: https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize;
https://github.com/OpenMOSS/MOSS-Transcribe-Diarize;
https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/blob/main/pyproject.toml.

### Task8 verified evidence and limits (2026-10-19)

Full isolated browser19PASS and backend253PASS; actual app-phase9s/10s/120s
CUDA/BF16/SDPA decode, native overlap/exact EOF correction, staged child reuse,
verified exit before real GPU embedding, existing channel profiles/named matching
and sequential reload measured. FFmpeg metadata WAV is canonicalized from exact
validated PCM at the IPC boundary; strict worker validation remains unchanged.
Research report records commands, timings, memory and evidence.

Real10s/120s fixtures share source/opening speech. An actual6.04s clean crop
naturally changes native S03 to S01 and matches a persisted real voice profile;
that crop shares the exact earlier profile interval. These demonstrate pipeline
persistence/label independence, not independent recognition accuracy. One valid
clean9s EOF sample was rejected by the existing voice crop boundary check and
remained unresolved without enrollment; a short2.12s seed did not match120s at
unchanged thresholds. No model, crop clock or threshold workaround was added.
Longest actual input120s does not prove90-minute memory fit or speed. Spec stays
In Review; no Shipped/deployment claim.
