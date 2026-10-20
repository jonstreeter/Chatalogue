# Handoff: MOSS canonical source checksum repair

**Date:** 2026-10-19
**Spec:** `docs/dev/specs/2026-10-18-moss-engine-design.md`
**Plan:** `docs/dev/plans/2026-10-18-moss-engine.md`

## Status

All14 tasks verified. Task14 fixes the four source checksums to immutable upstream
LF download bytes; revisions, requirements and all19 model pins stay unchanged.
Plan remains In progress/spec In Review for local closeout. Previous task13 baton
is preserved byte-for-byte at
`archive/2026-10/2026-10-18-moss-engine-batching-complete.md`.

## Next step

Conductor reviews task14 evidence/commit and reports PLAN_COMPLETE. Restart the
backend to import the corrected pins, then retry MOSS Load/Test. Normal verified
standard-root install is ready for reuse; no production job was retried.
No push, deployment, installer update or feature archive is authorized here.

## Verification state

- Isolated backend venv `python -m pytest src/test_moss_runtime.py src/test_moss.py src/test_moss_chunks.py -q`
  with task14 pre-import isolation and `MOSS_TEST_FRESH_SOURCE=1` ->168 PASS,
  zero skips,38.55s,exit0. All four source files downloaded into fresh temporary
  caches; altered/truncated/CRLF responses rejected, verified offline cache reused.
- Scoped Ruff ->PASS. SDD validation and git whitespace ->PASS; required normal
  hooks run at checkpoint.
- Actual normal standard-root `MossInstallation.ensure()` ->PASS,876.271s.
  Existing19 model cache files verified by exact unchanged hashes. All4 new source
  digest paths initially absent; fresh canonical network bytes passed size/SHA.
  Installed helper AST equals original CRLF checkout; source differences newline-only.
- Reused67 exact lock-matching wheels (3,036,101,674 bytes) from read-only task3
  caches after disk-space check; no dependency/model redownload or backend upgrade.
  Offline pip install, pip check, native import/CUDA check and atomic manifest PASS.
- Offline verified ensure with network/download/subprocess APIs forbidden ->PASS.
- Real manager load/readiness/retained9s CUDA decode ->PASS. RTX5090,
  Torch2.10.0+cu128/CUDA12.8, BF16/SDPA;4 turns,3 native overlap pairs,
  strict EOS/context and raw9.01->9 moss-eof-v1 correction. Decode6.283s.
- Owned unload ->PASS, worker66300 absent, launcherexit0, installed/unloaded status.

## Blockers

None. No owned runtime process remains. Nine unrelated baseline paths preserved.
The existing Transformers processor mapping deprecation warning is nonfatal.

## Diagnosis worth keeping

Original source pins came from Windows checkout CRLF bytes. Immutable upstream
revision e3bdd93f8ad58418e04659818efd5fd87a0bc486 downloads LF bytes. Failure was
first source/LICENSE after model downloads, not corrupt weights. Task3's original
checkout-seeded cache masked this fresh-download defect. Exact raw validation
remains strict; no normalization, bypass, model/source revision or requirement change.

New standard install version:
`c19b2f905bd461f27f79731aa9745864128092d26a91f788b54980b5755ca98f`.
Evidence under ignored `backend/runtime/moss/task14/`: exact copied sitecustomize,
proof.py/proof.json/proof.log/proof.exit, tests.log/tests.exit. Production data/.env
blocked by the pre-import audit guard; SQLite/app paths redirected before imports.
Only normal app-managed standard runtime setup/cache writes were allowed.
Original source/evidence, production recordings/settings/DB and checkpoints untouched.
Pip-check log to manifest interval~16s; later quiet intervals were complete SHA
verification, not a native import stall. Full source hashes/timings are in research.

## Next worker prompt

PLAN_COMPLETE after conductor review. Do not rerun closed long/seam/browser proofs.
Leave plan In progress/spec In Review pending later owner closeout direction.
Restart backend before Load/Test so the new pin constants/version take effect.
