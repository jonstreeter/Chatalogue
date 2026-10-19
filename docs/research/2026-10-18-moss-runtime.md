# MOSS native Windows CUDA feasibility — 2026-10-18

Task 1 is **blocked**, not passed. Native Windows NVIDIA CUDA load and complete
decode work, including native overlapping speaker turns. However, a synthetic
9.00-second cross-talk fixture generated an end clock of 9.01 seconds. The approved
strict source-bound validation rejects it. No tolerance, clamping, fabricated word
clocks, CPU fallback or product implementation was introduced. An owner decision
is required before more attempts or task 2. This is one invalid fixture result,
not evidence of general CUDA incompatibility or recognition accuracy.

## Environment and immutable inputs

- Windows, CPython 3.11.6, uv 0.11.29; NVIDIA RTX 5090, 32607 MiB reported,
  driver 610.88. Torch 2.10.0+cu128 / CUDA 12.8; Transformers 5.6.0.
- Private environment `backend/runtime/moss/venv`; backend `.venv` untouched.
- Helper source: [OpenMOSS repository](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/tree/e3bdd93f8ad58418e04659818efd5fd87a0bc486),
  commit `e3bdd93f8ad58418e04659818efd5fd87a0bc486`.
- Model: [immutable model snapshot](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize/tree/704aa4a9c304e8520be88901e0d1960158ef5b15),
  revision `704aa4a9c304e8520be88901e0d1960158ef5b15`.
  Weight file `model-00000-of-00001.safetensors`: 1,817,113,576 bytes;
  SHA256 `9a0ceb4ab7330357db3ff583dba8d83625d5b733b00e1d55d6970e11b07026c4`,
  matching snapshot download metadata. Both source and model declare Apache-2.0;
  retain upstream LICENSE/notices if later redistributed.
- Reviewed model/config/processor custom Python before `trust_remote_code=True`:
  Whisper encoder, Qwen3 decoder, Torch tensor operations and Transformers APIs;
  no custom kernel installation/execution in those files. Helper audio loading
  accepts paths/URLs, so this probe supplied only local authorized scratch paths.
  Helper `resolve_device` silently falls back to CPU: bypassed with an explicit
  CUDA availability assertion and `cuda:0`. Inference used BF16 and SDPA.
- All runtime files, environments, raw transcripts and generated audio remain
  ignored scratch, never staged. No production DB, settings, jobs or recordings
  changed; no server, Docker or backend ML dependency changes.

## Setup observations

Resumed existing owned uv processes rather than starting duplicate installs.
Torch/torchaudio install PID 10796 started 10:06:31 local; it was still alive
at 10:37:10 and absent at 10:38:38: observed setup interval 30m39s–32m07s.
Helper install PID 47292 started 10:07:01 and had exited by 10:27:20.
These are observed bounds, not exact successful installer durations/exit codes;
the previous worker owned their original sessions. Package metadata and CUDA
imports were verified after both processes disappeared. During extraction,
ATen header counts/CPU/I/O rose continuously, followed by CUDA DLL copies;
the underlying OS bottleneck was not established. No system security changes.
The separate pinned model snapshot download reported 20/20 files in about 7s.
F: had 195 GB free during setup. Do not compare setup time with inference time.

## Complete decode and timing evidence

Source: existing nonproduction research WAV
`backend/data/asr-benchmark/trespasses-600-720.wav` (120 seconds, PCM16 mono
16 kHz). Scratch 10/30/120-second crops preserve original samples. No corrected
reference was used; no accuracy or long-episode throughput claim follows.

| Process | Model+processor load s | Input s | First decode s | Warm repeat s | Tokens | Peak allocated / reserved bytes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 41880 | 2.992310 | 10 | 41.308548 | 4.208857 | 73 | 1872421888 / 1954545664 |
| 41880 | same process | 30 | 15.127879 | 16.323845 | 290 | 1942597632 / 2040528896 |
| 46808 | 1.965834 | 10 | 7.313288 | 4.239444 | 73 | 1872421888 / 1954545664 |
| 46808 | same process | 120 | 57.327404 | 53.470803 | 1016 | 2505355776 / 2625634304 |

Load timing starts after Python/library imports; it does not include process
startup/import time. Decode includes audio preparation, generation and validation.
The first-ever preprocessing cost was much larger; its exact internal cause was
not separately measured. CUDA synchronization brackets timing; peaks reset per
decode. Both repeats produced identical parsed results per fixture. All eight
ordinary-fixture decodes ended at EOS 151645 below the 4096-token budget, with
complete strict text coverage and equality to upstream parsing, finite
source-bounded segment clocks and nonempty native speaker/text fields. No
overlapping intervals occurred in these ordinary-fixture outputs.

## Native overlap evidence and strict gate failure

Synthetic fixture construction is explicit: take source samples 0–6s (native
S01 turn) and 15–22s (native S02 turn); multiply each by 0.5, place the second at
2s, sum and encode PCM16. The resulting file is exactly 9 seconds / 144000
samples, no padding or timestamp adjustment. This is synthetic cross-talk made
from real speech, not a natural-overlap accuracy benchmark. A second fixture
with offset 4s was prepared but never decoded because the first failed.

PID 17852 loaded in 2.054999s and failed the source-bound assertion; exit 1,
18.613026s total process wall time. One exact diagnostic rerun (PID 4676) added
raw-output capture only, retained the same prompt/model/audio/checks, loaded in
2.125531s and failed the same assertion; exit 1, 16.650701s total wall time.
No subsequent inference was attempted.

Raw diagnostic output (108 tokens, EOS 151645, budget 4096):

```text
[0.00][S01] A drastic overhaul than even Utah has.[2.35][2.12][S02] I mean, I was born in 91, so I really don't.[4.75][2.98][S01] We're getting there in every possible way in Salt Lake City.[5.83][5.41][S02] I don't remember not living with a bunch of Mexicans.[9.01]
```

Native intervals are S01 0–2.35, S02 2.12–4.75, S01 2.98–5.83 and S02
5.41–9.01. Cross-speaker overlaps exist; segment ends in this example are increasing;
the final clock exceeds the source by 0.01s and is rejected. Generation was
complete, so this failure is neither a token-budget limit nor an unparsed tail.
Exact failing check, retained in `probe-overlap*.log`:

```python
assert math.isfinite(start) and math.isfinite(end) and 0 <= start <= end <= duration
# AssertionError; duration=9.0, final end=9.01
```

## Process exit and VRAM

Global `nvidia-smi` baseline immediately before first probe: 3902 MiB. First
process observed 6401 MiB during decode, normal-exit marker and shell exit 0,
then 3889 MiB with PID 41880 absent. Second process observed 6971 MiB, normal
exit and shell exit 0; 3897 MiB after completion and PID 46808 later absent.
Synthetic failed processes also exited 1; GPU returned to 3898/3909 MiB.
Unrelated desktop/Python/Ollama processes were left untouched; global usage
varied, so exact global deltas are not attributed solely to this model. Torch
allocated/reserved peaks above are the process-local measurements. This proves
ordinary and failed probe process release, not yet managed app parent-exit IPC.

## Reproduction

### Workflow timing audit requested before the EOF-policy rerun

2026-10-18: owner asked to re-examine the workflow for a cause of the overrun
before resuming. No further generation or EOF correction was performed. A
read-only CPU audit (`backend/runtime/moss/audit_timing.py`, retained scratch)
exercised the pinned processor and loader and saved `timing-audit.json`.
CUDA remained uninitialized during that audit.

- Research WAV and 120-second scratch crop are PCM-identical. Reconstructing
  the synthetic mix from its original sample slices is PCM-identical to the
  retained fixture. Fixture: mono PCM16, 144000 samples at 16000 Hz, exactly
  9 seconds. PCM SHA256:
  `6fa9e0bfb2a432a37680625901b449a429e99c2e4128f407ac69b7474fed7e11`.
- The actual helper audio loader yields exactly 144000 samples, equal to
  original PCM / 32768 sample-for-sample. No resampling length change, extra
  decoded samples, stereo/time-base mismatch or WAV-header discrepancy was found.
- Raw generation already contains the literal `[9.01]`; the upstream parser
  returns 9.01 without changing it. Exact excess is 1/100 second (160 samples),
  not a binary-float comparison artifact. Prompt tensor width and attention
  length both equal 200, so prompt slicing is consistent for this single input.
  EOS 151645 followed 108 generated tokens, well below the probe budget.
- Cached custom processor/model/config hashes equal the pinned snapshot copies;
  helper checkout is clean at its recorded immutable SHA. Source/model revisions
  are independently pinned; no evidence of stale dynamic-code cache was found.
- Native feature extraction pads the mel input to `(1,80,3000)` (30-second
  Whisper window) but supplies only 113 valid merged audio tokens for this
  fixture. At 12.5 tokens/second their nominal span is 9.04 seconds. The decoder
  is not given an exact 9.00-second duration field by this helper. This is
  upstream feature padding/length quantization, not evidence that 30 seconds of
  audio was transcribed. Its causal contribution to `[9.01]` is unproven.
- The synthetic fixture abruptly cuts its second source slice at source 22.00s;
  the ordinary 120s model output placed that utterance's end at 22.35s. That is
  model-derived context, not a corrected reference, but the fixture should be
  described as an overlap/abrupt-boundary stress test rather than a clean
  completed-utterance accuracy example.

**Upstream marker-placement finding:** the pinned model processor config sets
`audio_tokens_per_second=12.5` and `time_marker_every_seconds=5`. Its
`_audio_span_ids` calculates `tokens_per_marker = int(rate * interval)` and
places the kth marker at `k * tokens_per_marker`. Thus fractional token
positions are truncated per interval, producing the following input alignment:

| Marker text (s) | Preceding audio tokens | Nominal token time (s) | Difference (s) |
| ---: | ---: | ---: | ---: |
| 5 | 62 | 4.96 | 0.04 |
| 10 | 124 | 9.92 | 0.08 |
| 30 | 372 | 29.76 | 0.24 |
| 120 | 1488 | 119.04 | 0.96 |

This arithmetic follows directly from pinned code/config and exact rational
comparison. It is an input-marker alignment discrepancy, **not a measured
transcript drift or proof of the cause of the 0.01s prediction**. Training may
have used the same convention. Do not modify marker placement/config or claim
an accuracy improvement without checking upstream intent and controlled evidence.
Do not widen EOF allowance to the token span or marker discrepancy. The approved
0.01s policy remains an application acceptance rule, not a demonstrated fix for
the cause of this observation. Pause its runtime rerun while this finding is
reported to the owner.

Other workflow issues found, not causes established for this overrun:

- Probe `max_new_tokens=4096` was a local, unsubstantiated cap, not the model's
  configured default (5120) or a product bound. It did not truncate the 108-token
  overlap result. Product implementation must derive the output budget from
  available context as the spec requires; do not copy this probe cap. The
  `len(ids) < budget` check also rejects an EOS exactly at the limit; conservative
  probe behavior must not become an unsupported general completeness definition.
- The report previously called these overlap intervals nonmonotonic in end
  ordering. They actually have increasing ends; cross-speaker overlap was proven,
  nested/nonmonotonic ordering was not. Task-2 nested-interval tests remain needed.
- The 32-minute installation delay and the audio-clock overrun are different
  quantities. Two installer processes targeted one private environment and the
  first worker lost its tool sessions on usage exhaustion. Active extraction
  was observed, but its OS bottleneck and installer exit status were not captured.
  Do not blame CUDA, antivirus or network without evidence. Future managed setup
  must serialize installs and retain logs/exit status as task 3 already requires.
- Earlier explanation of the 0.01s error as decimal rounding was too definite.
  Examples use two decimals, but the parser does not enforce that precision.
  The numerical source-duration path is verified; the exact model prediction's
  causal mechanism remains unresolved. No product workaround was written.

### Upstream training/history follow-up

2026-10-18, owner-triggered read-only investigation; no generation, parameter
changes or source patch. Checked upstream repository head
`e3bdd93f8ad58418e04659818efd5fd87a0bc486`, which is still the pinned helper SHA.

- The same `_audio_span_ids` arithmetic is present in the
  [initial public release](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/blob/8ca7350a8e4148b5aff78b465086703b62744ce2/moss_transcribe_diarize/processing_moss_transcribe_diarize.py)
  and current pinned source. File history lists only that release and
  [fine-tuning support](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/commit/53b41928d4c84eee58005c8f0f631bd28b3f163b).
  The latter renamed `_expand_audio_token` to `expand_audio_token` for the
  collator; it did not change marker placement. No upstream correction to this
  arithmetic was found in that file's history.
- The published [fine-tuning entry point](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/blob/e3bdd93f8ad58418e04659818efd5fd87a0bc486/finetune.py)
  loads `MossTranscribeDiarizeProcessor.from_pretrained(model_name_or_path)`.
  Its collator uses that processor for features/placeholder expansion and leaves
  supplied reference transcript targets unchanged. With this checkpoint it
  therefore uses its five-second config and the same marker arithmetic.
  This describes the released fine-tuning recipe, not proof of original
  pretraining implementation, intended calibration or correctness.
- The [technical report v7, section 2](https://arxiv.org/pdf/2601.01554v7)
  describes textual timestamp markers in the input representation, but does not
  specify exact five-second token placement, rounding or endpoint clipping.
  Its synthetic-data discussion describes smoothing mixture boundaries; the
  retained abrupt-cut fixture is not a reproduction of that training recipe.
- [Upstream issue 51](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/issues/51)
  reports much larger timestamp errors on other audio. Its replies contain no
  marker-code diagnosis, patch or validated alternative config. Treat it as
  a separate user report, not a reproduction or explanation of our 0.01s result.
  No matching marker-specific diagnosis was found in the retrieved 62 issue/PR
  entries or file history. Hugging Face file-history pages could not be fetched,
  so original model-history/initial-training intent is not fully established.

Conclusion: our integration/probe did not introduce this marker convention.
Public evidence neither proves it is an intentional trained calibration nor
establishes a safe correction. Changing `time_marker_every_seconds` or rewriting
marker positions would alter model input and could reduce accuracy. Preserve
pinned upstream behavior. The causal explanation for the exact `[9.01]` remains
unproven; do not equate nominal marker offset with transcript drift.

The useful next controlled check is the retained fixture through the unmodified
official generation helper versus the custom probe, capturing raw output before
normalization. Then test the approved EOF rule separately. Reuse the same pinned
environment/audio; do not regenerate a friendlier fixture, widen tolerance,
restart downloads or patch the upstream processor. Long-input app verification
must retain timestamp-quality limitations rather than claiming exact alignment
from the short feasibility samples.

Run from repository root; private Python must remain separate from backend ML.
Setup commands used (source checkout pinned before install):

```powershell
uv venv --python 3.11 backend/runtime/moss/venv
git clone https://github.com/OpenMOSS/MOSS-Transcribe-Diarize backend/runtime/moss/source
git -C backend/runtime/moss/source checkout e3bdd93f8ad58418e04659818efd5fd87a0bc486
uv pip install --python backend/runtime/moss/venv/Scripts/python.exe torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python backend/runtime/moss/venv/Scripts/python.exe -e backend/runtime/moss/source transformers==5.6.0
$env:HF_HOME = (Join-Path (Get-Location) 'backend/runtime/moss/hf')
& backend/runtime/moss/venv/Scripts/python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('OpenMOSS-Team/MOSS-Transcribe-Diarize', revision='704aa4a9c304e8520be88901e0d1960158ef5b15', local_dir='backend/runtime/moss/model')"
& backend/runtime/moss/venv/Scripts/python.exe backend/runtime/moss/probe.py backend/runtime/moss/conversation-10s.wav backend/runtime/moss/conversation-30s.wav
& backend/runtime/moss/venv/Scripts/python.exe backend/runtime/moss/probe.py backend/runtime/moss/conversation-10s.wav backend/runtime/moss/conversation-120s.wav
# Expected strict failure, only rerun with owner authorization:
& backend/runtime/moss/venv/Scripts/python.exe backend/runtime/moss/probe.py backend/runtime/moss/synthetic-overlap-2s.wav
```

The initial install commands resolved transitive packages at execution time;
the exact tested freeze below is authoritative for future reproduction. Install
Torch/torchaudio from cu128, other frozen packages from PyPI, and the helper from
its pinned local checkout instead of fetching an unrelated package version.
Retained scratch evidence: `dependencies.txt`, `probe-cold1.log`,
`probe-cold2.log`, `probe-overlap.log`, `probe-overlap-diagnostic.log`, per-decode
JSON, raw synthetic JSON, `download.log` and `import-check.log` under
`backend/runtime/moss/`. Runnable probe and fixture generator are preserved below
because ignored scratch is not part of the commit.


### Exact tested package freeze

```text
Jinja2==3.1.6
MarkupSafe==3.0.3
PyYAML==6.0.3
Pygments==2.21.0
annotated-doc==0.0.5
annotated-types==0.8.0
anyio==4.15.1
audioread==3.1.0
av==18.1.0
certifi==2026.7.22
cffi==2.1.1
charset-normalizer==3.5.2
click==8.5.0
cloudpickle==3.1.2
colorama==0.4.6
decorator==5.3.1
fastapi==0.145.0
filelock==3.32.3
fsspec==2026.7.0
h11==0.16.0
hf-xet==1.7.0
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.33.0
idna==3.20
joblib==1.6.0
lazy-loader==0.6
librosa==0.11.0
llvmlite==0.50.0
markdown-it-py==4.2.0
mdurl==0.1.2
moss-transcribe-diarize==0.1.0
mpmath==1.3.0
msgpack==1.2.3
narwhals==2.27.1
networkx==3.6.1
numba==0.68.0
numpy==2.4.6
opentelemetry-api==1.45.1
packaging==26.3
platformdirs==4.13.4
pooch==1.9.0
pycparser==3.11
pydantic==2.14.1
pydantic_core==2.50.1
python-multipart==0.0.32
regex==2026.9.29
requests==2.34.2
rich==15.0.0
safetensors==0.8.0
scikit-learn==1.9.1
scipy==1.17.1
shellingham==1.5.4
soundfile==0.14.0
soxr==1.1.0
starlette==1.8.0
sympy==1.14.0
threadpoolctl==3.7.0
tokenizers==0.22.2
torch==2.10.0+cu128
torchaudio==2.10.0+cu128
tqdm==4.70.1
transformers==5.6.0
typer==0.27.3
typing-inspection==0.4.4
typing_extensions==4.16.0
urllib3==2.8.0
uvicorn==0.54.0
```

### Runnable probe

```python
import dataclasses
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import wave

root = Path(__file__).resolve().parent
os.environ['HF_HOME'] = str(root / 'hf')
os.environ['HF_MODULES_CACHE'] = str(root / 'modules')
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from moss_transcribe_diarize.inference_utils import build_transcription_messages, prepare_inputs
from moss_transcribe_diarize import parse_transcript

revision = '704aa4a9c304e8520be88901e0d1960158ef5b15'
assert torch.cuda.is_available(), 'CUDA required; fallback forbidden'
device = torch.device('cuda:0')
torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(str(root / 'model'), trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
processor = AutoProcessor.from_pretrained(str(root / 'model'), trust_remote_code=True, local_files_only=True)
torch.cuda.synchronize()
print(json.dumps({'event':'load','pid':os.getpid(),'seconds':time.perf_counter()-started,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'attention':model.config._attn_implementation}), flush=True)
pattern = re.compile(r'\[([0-9]+(?:\.[0-9]*)?)\]\[(S[0-9]+)\](.*?)\[([0-9]+(?:\.[0-9]*)?)\]', re.S)
for path in sys.argv[1:]:
    with wave.open(path) as wav:
        duration = wav.getnframes()/wav.getframerate()
    for repeat in range(2):
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            inputs = prepare_inputs(processor, build_transcription_messages(path), device=device).to(device)
        prompt = inputs['input_ids'].shape[1]
        budget = min(4096, model.config.text_config.max_position_embeddings-prompt)
        assert budget > 0
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = model.generate(**inputs, max_new_tokens=budget, do_sample=False)
        torch.cuda.synchronize()
        ids = outputs[0,prompt:]
        eos = model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos,int) else eos
        assert ids[-1].item() in eos, 'incomplete generation: no EOS'
        assert len(ids) < budget, 'generation budget exhausted'
        text = processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
        (root / (Path(path).stem + '_raw_' + str(repeat) + '_' + str(os.getpid()) + '.json')).write_text(json.dumps({'duration': duration, 'text': text, 'generated_tokens': len(ids), 'last_token': ids[-1].item(), 'budget': budget}, ensure_ascii=False, indent=2), encoding='utf-8')
        segments = parse_transcript(text)
        matches = list(pattern.finditer(text))
        cursor = 0
        for match in matches:
            assert not text[cursor:match.start()].strip(), 'unparsed output'
            start,end = float(match[1]),float(match[4])
            assert math.isfinite(start) and math.isfinite(end) and 0<=start<=end<=duration
            assert match[2] and match[3].strip()
            cursor=match.end()
        assert not text[cursor:].strip(), 'unparsed tail'
        assert matches and len(matches)==len(segments)
        assert [(float(m[1]),float(m[4]),m[2],m[3].strip()) for m in matches] == [(s.start,s.end,s.speaker,s.text) for s in segments]
        overlaps=[(a,b) for a,s in enumerate(segments) for b,t in enumerate(segments) if a<b and s.speaker!=t.speaker and max(s.start,t.start)<min(s.end,t.end)]
        result={'audio':path,'duration':duration,'repeat':repeat,'seconds':time.perf_counter()-started,'prompt_tokens':prompt,'generated_tokens':len(ids),'last_token':ids[-1].item(),'eos':eos,'budget':budget,'complete':True,'text':text,'segments':[dataclasses.asdict(s) for s in segments],'overlaps':overlaps,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()}
        destination=root/(Path(path).stem+'_'+str(repeat)+'_'+str(os.getpid())+'.json')
        destination.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k not in ('text','segments')}),flush=True)
print(json.dumps({'event':'normal_exit','pid':os.getpid()}),flush=True)
```

### Synthetic fixture generator

```python
from pathlib import Path
import wave
import numpy as np
root=Path(__file__).resolve().parent
with wave.open(str(root/'conversation-120s.wav')) as w:
    params=w.getparams()
    assert params.nchannels==1 and params.sampwidth==2
    samples=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(np.float32)
r=params.framerate
# Different native speaker turns from the untouched research source.
a=samples[0:6*r]
b=samples[15*r:22*r]
for offset in [2,4]:
    mixed=np.zeros((offset+7)*r,dtype=np.float32)
    mixed[:len(a)]+=0.5*a
    mixed[offset*r:offset*r+len(b)]+=0.5*b
    with wave.open(str(root/f'synthetic-overlap-{offset}s.wav'),'wb') as w:
        w.setparams(params)
        w.writeframes(np.clip(mixed,-32768,32767).astype('<i2').tobytes())
```

## Controlled official-helper/probe comparison (2026-10-18)

PASS for this comparison only; task 1 remains in progress. No EOF policy, parser,
normalization, product code, fixture edit or upstream/config change was applied.
The pinned environment/model/source above were reused, offline, on native CUDA
BF16 SDPA. Both paths used the same retained 144000-sample 16 kHz PCM16 fixture,
default prompt and greedy 4096-token budget (experimental control, not a product
default). PCM SHA256: `6fa9e0bfb2a432a37680625901b449a429e99c2e4128f407ac69b7474fed7e11`.

Command from repository root:

```powershell
& backend/runtime/moss/venv/Scripts/python.exe backend/runtime/moss/compare_helper.py *> backend/runtime/moss/helper-comparison.log
$comparisonExit=$LASTEXITCODE
"child_exit=$comparisonExit" | Set-Content backend/runtime/moss/helper-comparison-exit.txt
```

One process loaded one shared model, then serialized probe followed by official
`generate_transcription`. The original probe generation path was reproduced up
to raw output capture, deliberately stopping before its strict parser/validation.
Read-only interception of `model.generate` cloned every supplied input tensor
and the raw return to CPU, passed the original arguments unchanged and returned
the untouched original output. Effective configuration was captured from the
provided configuration (or model configuration) plus generation keyword overrides.
The official helper retained its upstream deep-copy behavior. No upstream file
was edited. Full tensors and outputs remain in ignored `helper-comparison-probe.pt`
and `helper-comparison-official.pt`; IDs, hashes, configurations and raw text are
in `helper-comparison.json`. The runnable comparison source is retained below.

All five tensor keys, shapes, dtypes and values matched exactly: `input_ids`,
`attention_mask`, `input_features`, `audio_feature_lengths`, `audio_chunk_mapping`.
Prompt width, attention length and slicing boundary were all 200 on both paths.
The complete effective generation configurations matched. Full returned IDs,
all 108 generated IDs, and raw decoded text including the EOS special token
matched exactly. Both completed with EOS 151645 before the 4096 budget.
Both retained the same raw terminal `[9.01]` for the exact 9.00s source.
This rules out a helper/probe output difference for this controlled execution;
it does not establish why the model emitted 9.01 or establish general accuracy.

Child PID 36916 loaded in 2.110840s; probe generation 6.081976s (whole path
7.776997s), official generation 5.834155s (whole path 5.841030s). These are
serialized first/second observations, not a performance comparison. The log
contains `normal_exit`; retained shell evidence is `child_exit=0`. Subsequent
`Get-Process -Id 36916 -ErrorAction SilentlyContinue` returned no process, and
`nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader` omitted
that PID. Subsequent global GPU memory was 4452 MiB; no same-run pre-load baseline
was captured, so no numerical VRAM-release delta is claimed. Other processes
were untouched.

Raw comparison evidence (full generated IDs and input tensor hashes):

```json
{
  "probe": {
    "prompt_width": 200,
    "attention_length": 200,
    "generated_ids": [
      58,
      15,
      13,
      15,
      15,
      60,
      42474,
      15,
      16,
      60,
      362,
      59783,
      51674,
      1091,
      1496,
      22389,
      702,
      7873,
      17,
      13,
      18,
      20,
      1457,
      17,
      13,
      16,
      17,
      60,
      42474,
      15,
      17,
      60,
      358,
      3076,
      11,
      358,
      572,
      9223,
      304,
      220,
      24,
      16,
      11,
      773,
      358,
      2167,
      1513,
      944,
      7873,
      19,
      13,
      22,
      20,
      1457,
      17,
      13,
      24,
      23,
      60,
      42474,
      15,
      16,
      60,
      1205,
      2299,
      3709,
      1052,
      304,
      1449,
      3204,
      1616,
      304,
      27040,
      11678,
      4311,
      7873,
      20,
      13,
      23,
      18,
      1457,
      20,
      13,
      19,
      16,
      60,
      42474,
      15,
      17,
      60,
      358,
      1513,
      944,
      6099,
      537,
      5382,
      448,
      264,
      15493,
      315,
      96762,
      7873,
      24,
      13,
      15,
      16,
      60,
      151645
    ],
    "raw_text": "[0.00][S01] A drastic overhaul than even Utah has.[2.35][2.12][S02] I mean, I was born in 91, so I really don't.[4.75][2.98][S01] We're getting there in every possible way in Salt Lake City.[5.83][5.41][S02] I don't remember not living with a bunch of Mexicans.[9.01]<|im_end|>",
    "tensors": {
      "input_ids": {
        "shape": [
          1,
          200
        ],
        "dtype": "torch.int64",
        "sha256": "476f056784aca9f1d315eb6b1e3571f6b4588e37acc29dbd43f9be7b5330db7e"
      },
      "attention_mask": {
        "shape": [
          1,
          200
        ],
        "dtype": "torch.int64",
        "sha256": "4a2ad471a677320287e08c40cb5f743fa45504ce496319ff4369157f7c0723e7"
      },
      "input_features": {
        "shape": [
          1,
          80,
          3000
        ],
        "dtype": "torch.float32",
        "sha256": "5d785df1f008dd9338c6a849739e39ca1aa5c3358869952299cad0973f176a62"
      },
      "audio_feature_lengths": {
        "shape": [
          1
        ],
        "dtype": "torch.int64",
        "sha256": "6d0d22a95af6226f43d672c8a1301c9ae5bb6573ade9e00e0e6344ba3a2b8e6e"
      },
      "audio_chunk_mapping": {
        "shape": [
          1
        ],
        "dtype": "torch.int64",
        "sha256": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc"
      }
    }
  },
  "official": {
    "prompt_width": 200,
    "attention_length": 200,
    "generated_ids": [
      58,
      15,
      13,
      15,
      15,
      60,
      42474,
      15,
      16,
      60,
      362,
      59783,
      51674,
      1091,
      1496,
      22389,
      702,
      7873,
      17,
      13,
      18,
      20,
      1457,
      17,
      13,
      16,
      17,
      60,
      42474,
      15,
      17,
      60,
      358,
      3076,
      11,
      358,
      572,
      9223,
      304,
      220,
      24,
      16,
      11,
      773,
      358,
      2167,
      1513,
      944,
      7873,
      19,
      13,
      22,
      20,
      1457,
      17,
      13,
      24,
      23,
      60,
      42474,
      15,
      16,
      60,
      1205,
      2299,
      3709,
      1052,
      304,
      1449,
      3204,
      1616,
      304,
      27040,
      11678,
      4311,
      7873,
      20,
      13,
      23,
      18,
      1457,
      20,
      13,
      19,
      16,
      60,
      42474,
      15,
      17,
      60,
      358,
      1513,
      944,
      6099,
      537,
      5382,
      448,
      264,
      15493,
      315,
      96762,
      7873,
      24,
      13,
      15,
      16,
      60,
      151645
    ],
    "raw_text": "[0.00][S01] A drastic overhaul than even Utah has.[2.35][2.12][S02] I mean, I was born in 91, so I really don't.[4.75][2.98][S01] We're getting there in every possible way in Salt Lake City.[5.83][5.41][S02] I don't remember not living with a bunch of Mexicans.[9.01]<|im_end|>",
    "tensors": {
      "input_ids": {
        "shape": [
          1,
          200
        ],
        "dtype": "torch.int64",
        "sha256": "476f056784aca9f1d315eb6b1e3571f6b4588e37acc29dbd43f9be7b5330db7e"
      },
      "attention_mask": {
        "shape": [
          1,
          200
        ],
        "dtype": "torch.int64",
        "sha256": "4a2ad471a677320287e08c40cb5f743fa45504ce496319ff4369157f7c0723e7"
      },
      "input_features": {
        "shape": [
          1,
          80,
          3000
        ],
        "dtype": "torch.float32",
        "sha256": "5d785df1f008dd9338c6a849739e39ca1aa5c3358869952299cad0973f176a62"
      },
      "audio_feature_lengths": {
        "shape": [
          1
        ],
        "dtype": "torch.int64",
        "sha256": "6d0d22a95af6226f43d672c8a1301c9ae5bb6573ade9e00e0e6344ba3a2b8e6e"
      },
      "audio_chunk_mapping": {
        "shape": [
          1
        ],
        "dtype": "torch.int64",
        "sha256": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc"
      }
    }
  }
}
```

Runnable comparison source (ignored scratch, retained for reproduction):

```python
"""Controlled raw helper/probe comparison; no parser or EOF policy."""
import copy, hashlib, json, os, time, wave
from pathlib import Path
root=Path(__file__).resolve().parent
os.environ['HF_HOME']=str(root/'hf')
os.environ['HF_MODULES_CACHE']=str(root/'modules')
os.environ['HF_HUB_OFFLINE']='1'
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from moss_transcribe_diarize.inference_utils import build_transcription_messages, prepare_inputs, generate_transcription
assert torch.cuda.is_available()
device=torch.device('cuda:0')
fixture=root/'synthetic-overlap-2s.wav'
with wave.open(str(fixture)) as w:
    frames,rate=w.getnframes(),w.getframerate()
    pcm=w.readframes(frames)
assert frames==144000 and rate==16000
assert hashlib.sha256(pcm).hexdigest()=='6fa9e0bfb2a432a37680625901b449a429e99c2e4128f407ac69b7474fed7e11'
start=time.perf_counter()
model=AutoModelForCausalLM.from_pretrained(str(root/'model'),trust_remote_code=True,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval()
processor=AutoProcessor.from_pretrained(str(root/'model'),trust_remote_code=True,local_files_only=True)
torch.cuda.synchronize()
report={'pid':os.getpid(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'attention':model.config._attn_implementation,'load_seconds':time.perf_counter()-start,'frames':frames,'rate':rate,'pcm_sha256':hashlib.sha256(pcm).hexdigest(),'paths':{}}
print(json.dumps({'event':'loaded',**report}),flush=True)
original=model.generate
captures={}
current=None
def capture(*args,**kwargs):
    tensors={k:v.detach().cpu().clone() for k,v in kwargs.items() if torch.is_tensor(v)}
    config=copy.deepcopy(kwargs.get('generation_config',model.generation_config))
    config.update(**{k:v for k,v in kwargs.items() if k not in tensors and k!='generation_config'})
    started=time.perf_counter()
    output=original(*args,**kwargs)
    torch.cuda.synchronize()
    captures[current]={'tensors':tensors,'output':output.detach().cpu().clone(),'config':config.to_dict(),'seconds':time.perf_counter()-started}
    return output
model.generate=capture
messages=build_transcription_messages(fixture)
for current in ('probe','official'):
    started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    if current=='probe':
        with torch.autocast('cuda',dtype=torch.bfloat16):
            inputs=prepare_inputs(processor,messages,device=device).to(device)
        prompt=inputs['input_ids'].shape[1]
        budget=min(4096,model.config.text_config.max_position_embeddings-prompt)
        assert budget==4096
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            outputs=model.generate(**inputs,max_new_tokens=budget,do_sample=False)
    else:
        helper=generate_transcription(model,processor,messages,max_new_tokens=4096,do_sample=False,device=device,dtype=torch.bfloat16)
        prompt=helper['prompt_len']
    cap=captures[current]
    ids=cap['output'][0,prompt:]
    eos=model.generation_config.eos_token_id
    eos=[eos] if isinstance(eos,int) else eos
    result={'prompt_width':cap['tensors']['input_ids'].shape[1],'attention_length':int(cap['tensors']['attention_mask'][0].sum()),'slice_prompt':prompt,'generated_ids':ids.tolist(),'raw_text':processor.tokenizer.decode(ids,skip_special_tokens=False),'decoded_text':processor.tokenizer.decode(ids,skip_special_tokens=True),'generation_seconds':cap['seconds'],'path_seconds':time.perf_counter()-started,'complete':ids[-1].item() in eos and len(ids)<4096,'last_token':ids[-1].item(),'generated_tokens':len(ids),'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'effective_generation_config':cap['config'],'tensors':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()} for k,v in cap['tensors'].items()}}
    if current=='official': result['helper_return']=helper
    report['paths'][current]=result
    torch.save(cap,root/f'helper-comparison-{current}.pt')
    (root/'helper-comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'event':current,'seconds':result['path_seconds'],'tokens':len(ids),'complete':result['complete']}),flush=True)
a,b=captures['probe'],captures['official']
report['comparison']={'tensor_keys_equal':a['tensors'].keys()==b['tensors'].keys(),'tensor_equal':{k:torch.equal(v,b['tensors'][k]) for k,v in a['tensors'].items()},'effective_config_equal':a['config']==b['config'],'full_output_equal':torch.equal(a['output'],b['output']),'generated_ids_equal':report['paths']['probe']['generated_ids']==report['paths']['official']['generated_ids'],'raw_text_equal':report['paths']['probe']['raw_text']==report['paths']['official']['raw_text']}
(root/'helper-comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
assert all(report['comparison']['tensor_equal'].values()) and all(v for k,v in report['comparison'].items() if k!='tensor_equal')
assert all(p['complete'] for p in report['paths'].values())
print(json.dumps({'event':'normal_exit','pid':os.getpid(),'comparison':report['comparison']}),flush=True)

```
