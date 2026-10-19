# ADR-0002: App-managed native MOSS worker and exact transcript boundary

**Date:** 2026-10-18
**Status:** Accepted
**Deciders:** Project owner (approved MOSS spec/plan and moss-eof-v1 refinement, 2026-10-18)

## Context

Task 1 proved native Windows RTX 5090 CUDA/BF16/SDPA decoding in a private
Torch 2.10 cu128 / Transformers 5.6 environment. Immutable source revision is
`e3bdd93f8ad58418e04659818efd5fd87a0bc486`; model revision is
`704aa4a9c304e8520be88901e0d1960158ef5b15`. The retained overlap fixture
completed with native labels and a 9.01 end for exactly 9 seconds. Separate
moss-eof-v1 proof passed. Timestamp accuracy is not established: Parakeet
comparisons disagree by approximately one second; the causal explanation for
the boundary overrun remains unresolved. Upstream time markers stay unchanged.

## Decision

Implement explicitly selected MOSS in an app-owned private native CUDA worker;
keep backend ML dependencies and existing defaults unchanged. Use owned JSONL
requests and temporary audio in subsequent lifecycle work, with no server or
Docker requirement. The backend independently validates complete raw generation
against manager-owned sample dimensions and compares all normalized evidence.
Preserve native overlapping labels, raw decimal tokens, ordering and words=null.
Use exact rational sample duration and only moss-eof-v1 end clipping up to 0.01s
inclusive, recording each correction. In-range zero spans follow raw start<=end;
corrected collapsed spans fail. No fabricated word clocks or persistent speaker
identities. The approved spec refines this MOSS-only authority.

The internal result uses schema `moss-result-v1`, exact rational strings for
normalized clocks/duration, raw transcript and decimal tokens, sample count/rate,
completion metadata, segments and corrections. Completion records reason `eos`,
`complete=true`, EOS and terminal token IDs, input/output token counts, output
budget and context limit (131072). Output must terminate with the declared EOS,
use fewer than its output budget and fit the context. The backend recomputes the
entire result and requires structural equality after JSON serialization. Failure
messages never include raw transcript. EOS establishes termination, not recognition
of all audible speech. Whitespace-only complete generation can represent silence.

## Alternatives considered

1. Backend ML upgrades/shared inference — conflicts with isolated optional setup.
2. Remote service or Docker — adds operator lifecycle requirements.
3. Generic engine framework or Phonon clock mapper — unnecessary scope and loses
   native overlap attribution.

## Consequences

Private dependency ownership costs a separate install and process lifecycle.
Segment timing is explicit; identity matching reuses channel-scoped profiles
and avoids enrolling mixed overlap. The initial recording limit of 5400 seconds
is superseded by the owner-directed 2026-10-19 batching amendment in the spec:
5400 seconds is a worker-request ceiling, not an episode ceiling. Chunk labels
are namespaced; existing voice embeddings reconcile persistent identities.
Completed chunks survive retry. No DB schema changes.

## What would change our mind

Revisit if native runtime cannot release its owned process/GPU, pinned dependencies
become unusable, or corrected reference audio demonstrates unacceptable timing.

## Rollback

Select an existing engine and remove the optional MOSS environment after stopping
its owned worker. Existing engines and data formats are unchanged.
