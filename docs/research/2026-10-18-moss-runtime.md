# MOSS native Windows CUDA feasibility — 2026-10-18

Task 1 is **PASS (1/8 complete)** after the separately authorized moss-eof-v1
proof below. Existing pinned-environment, eight ordinary complete decodes,
cold/warm timing and successful/failed process-release evidence are reused.
The original strict 9.01s/9.00s overlap rejection remains a historical FAIL.
The new proof accepts only its exact 0.01s end overrun under the owner-approved
policy. Product tasks 2–8 remain unstarted; this is runtime feasibility evidence,
not application integration or recognition/timing accuracy validation.

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

## Separate approved EOF proof — task 1 PASS, 2026-10-18

The retained synthetic cross-talk/abrupt-cut fixture is unchanged: 144000 samples
at 16000 Hz, exact duration 9 seconds, PCM SHA256
`6fa9e0bfb2a432a37680625901b449a429e99c2e4128f407ac69b7474fed7e11`.
No model, source, config, backend environment or production data changed.
CUDA/BF16/SDPA and default transcription prompt remain as pinned above.
This targeted run uses model generation_config max_new_tokens=5120 bounded by
available model context, replacing the earlier unsupported local 4096 cap for
this proof. Prompt 200 + generated 108 tokens fits the context; final token
151645 is EOS, far below the budget. Full decoded output is consumed by the
strict segment grammar. This proves completeness for this fixture, not a generic
silence/early-EOS coverage guarantee.

Exact Fraction comparisons implement D=sample_count/sample_rate and independent
end-only clipping when 0 < raw_end-D <= 1/100. Raw starts and all finite decimal
clocks are validated first; any malformed/invalid segment rejects the whole
result. No sorting, dropping, splitting, text/speaker changes or invented word
timestamps occurs. In-range clocks retain their exact values (rational strings
below are diagnostic serialization, not a proposed checkpoint format).

18 runnable checks PASS: inclusive 9.010000, rejected 9.010001; unchanged in-range;
negative/out-of-range/reversed/nonfinite clocks; corrected collapse; multiple
overlapping ends with independent allowance; sample-derived D=1/3 accepted and
rejected fractional boundaries; atomic mixed-result rejection; incomplete raw
generation still rejected; trailing junk/nested brackets/nondecimal/fraction
tokens rejected. Decimal grammar checks were tightened after the real decode;
the retained raw result was revalidated with that final normalizer and is
identical. No additional CUDA inference was run.

Measured load 53.435088s, decode/preparation 29.205355s; these slower cold-run
observations do not replace earlier cold/warm benchmarks or establish a speed
guarantee. Process-local peak allocated 1,874,400,256 bytes, reserved
1,929,379,840 bytes. Python child 26664 emitted normal_exit; private launcher
46912 and child were both absent after shell exit 0. Global GPU memory was
3976 MiB before and 4020 MiB after; 44 MiB difference cannot be attributed to
MOSS amid unrelated desktop/app GPU usage. Process exit plus disappearance of
the ~1.9 GB local model allocation support release; global memory is not an
isolated allocator measurement. No unrelated processes were stopped. Existing
Python 4976 was identified read-only as the app's uvicorn process before launch;
no private MOSS process was present.

Full raw generation and normalized evidence (also retained in ignored
`eof-proof-raw.json`, `eof-proof.json`, `eof-proof.log`, `eof-exit.txt`,
`eof-before-memory.txt`, `eof-after-memory.txt`):

```json
{
  "raw": {
    "pid": 26664,
    "frames": 144000,
    "rate": 16000,
    "pcm_sha256": "6fa9e0bfb2a432a37680625901b449a429e99c2e4128f407ac69b7474fed7e11",
    "raw_text": "[0.00][S01] A drastic overhaul than even Utah has.[2.35][2.12][S02] I mean, I was born in 91, so I really don't.[4.75][2.98][S01] We're getting there in every possible way in Salt Lake City.[5.83][5.41][S02] I don't remember not living with a bunch of Mexicans.[9.01]<|im_end|>",
    "decoded_text": "[0.00][S01] A drastic overhaul than even Utah has.[2.35][2.12][S02] I mean, I was born in 91, so I really don't.[4.75][2.98][S01] We're getting there in every possible way in Salt Lake City.[5.83][5.41][S02] I don't remember not living with a bunch of Mexicans.[9.01]",
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
    "prompt_tokens": 200,
    "budget": 5120,
    "eos": [
      151645
    ],
    "complete": true,
    "load_seconds": 53.435088100028224,
    "decode_seconds": 29.20535549998749,
    "peak_allocated": 1874400256,
    "peak_reserved": 1929379840,
    "torch": "2.10.0+cu128",
    "cuda": "12.8",
    "gpu": "NVIDIA GeForce RTX 5090",
    "attention": "sdpa"
  },
  "normalized": {
    "policy": "moss-eof-v1",
    "frames": 144000,
    "rate": 16000,
    "duration": "9",
    "segments": [
      {
        "raw_start": "0.00",
        "raw_end": "2.35",
        "start": "0",
        "end": "47/20",
        "speaker": "S01",
        "text": "A drastic overhaul than even Utah has.",
        "words": null
      },
      {
        "raw_start": "2.12",
        "raw_end": "4.75",
        "start": "53/25",
        "end": "19/4",
        "speaker": "S02",
        "text": "I mean, I was born in 91, so I really don't.",
        "words": null
      },
      {
        "raw_start": "2.98",
        "raw_end": "5.83",
        "start": "149/50",
        "end": "583/100",
        "speaker": "S01",
        "text": "We're getting there in every possible way in Salt Lake City.",
        "words": null
      },
      {
        "raw_start": "5.41",
        "raw_end": "9.01",
        "start": "541/100",
        "end": "9",
        "speaker": "S02",
        "text": "I don't remember not living with a bunch of Mexicans.",
        "words": null
      }
    ],
    "corrections": [
      {
        "index": 3,
        "raw_end": "9.01",
        "normalized_end": "9",
        "policy": "moss-eof-v1"
      }
    ]
  },
  "overlaps": [
    [
      0,
      1
    ],
    [
      1,
      2
    ],
    [
      2,
      3
    ]
  ]
}
```

All four native segments remain in order with words=null; only index 3 has
raw end 9.01 normalized to 9, policy moss-eof-v1. Native cross-speaker overlap
pairs are (0,1), (1,2), (2,3). The original strict failure is unchanged; this
separate policy acceptance does not repair or explain model timestamps.
Synthetic overlap survival is structural evidence, not natural-overlap accuracy.
The source marker investigation and Parakeet comparison limitations remain.
Backend correction-evidence trust and checkpoint retry validation belong to
tasks 2/5 and are not claimed here.

Reproduce from repository root in PowerShell with retained assets (no installs):

```powershell
& backend/runtime/moss/venv/Scripts/python.exe backend/runtime/moss/eof_check.py
$eofBefore = nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
$eofBefore | Set-Content backend/runtime/moss/eof-before-memory.txt
& backend/runtime/moss/venv/Scripts/python.exe backend/runtime/moss/eof_probe.py *> backend/runtime/moss/eof-proof.log
$eofExit = $LASTEXITCODE
$eofExit | Set-Content backend/runtime/moss/eof-exit.txt
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Set-Content backend/runtime/moss/eof-after-memory.txt
Get-Process -Id 26664,46912 -ErrorAction SilentlyContinue | Select-Object Id
```

PIDs above are this run's observed IDs; use emitted PID and its launcher on a
fresh run. Exact post-decode validation command used after decimal grammar
tightening:

```powershell
& backend/runtime/moss/venv/Scripts/python.exe -c "import sys,json; sys.path.insert(0,'backend/runtime/moss'); from eof_check import normalize; from pathlib import Path; p=Path('backend/runtime/moss'); r=json.loads((p/'eof-proof.json').read_text()); assert normalize(r['raw']['decoded_text'],r['raw']['frames'],r['raw']['rate'],r['raw']['complete'])==r['normalized']; print('PASS retained real output with final decimal-token rejection checks')"
```

Runnable research validator (`backend/runtime/moss/eof_check.py`, ignored):

```python
"""Research-only exact moss-eof-v1 validator and runnable boundary checks."""
from fractions import Fraction
import re

POLICY = 'moss-eof-v1'
PATTERN = re.compile(r'\[([^\[\]]+)\]\[(S[0-9]+)\]([^\[\]]+)\[([^\[\]]+)\]')

def normalize(text, frames, rate, complete=True):
    if not complete:
        raise ValueError('incomplete generation')
    duration = Fraction(frames, rate)
    segments, corrections = [], []
    cursor = 0
    for i, m in enumerate(PATTERN.finditer(text)):
        if text[cursor:m.start()].strip():
            raise ValueError('unparsed output')
        if not all(re.fullmatch(r'[0-9]+(?:\.[0-9]*)?', token) for token in (m[1], m[4])):
            raise ValueError('malformed decimal clock')
        start, end = Fraction(m[1]), Fraction(m[4])
        if not (0 <= start <= duration and start <= end):
            raise ValueError('invalid raw clocks')
        normalized = end
        if end > duration:
            if end-duration > Fraction(1,100):
                raise ValueError('EOF allowance exceeded')
            normalized = duration
            corrections.append({'index':i,'raw_end':m[4],'normalized_end':str(duration),'policy':POLICY})
        if normalized <= start or not m[3].strip():
            raise ValueError('empty span/text')
        segments.append({'raw_start':m[1],'raw_end':m[4],'start':str(start),'end':str(normalized),'speaker':m[2],'text':m[3].strip(),'words':None})
        cursor = m.end()
    if not segments or text[cursor:].strip():
        raise ValueError('missing segments/unparsed tail')
    return {'policy':POLICY,'frames':frames,'rate':rate,'duration':str(duration),'segments':segments,'corrections':corrections}

def checks():
    cases = [
        ('boundary','[0][S0]x[9.010000]',144000,16000,True,True),
        ('epsilon','[0][S0]x[9.010001]',144000,16000,True,False),
        ('inrange','[0.1][S0]x[8.999]',144000,16000,True,True),
        ('negative','[-0.1][S0]x[1]',144000,16000,True,False),
        ('start-outside','[9.001][S0]x[9.01]',144000,16000,True,False),
        ('reversed','[2][S0]x[1]',144000,16000,True,False),
        ('nan','[NaN][S0]x[1]',144000,16000,True,False),
        ('infinity','[0][S0]x[Infinity]',144000,16000,True,False),
        ('collapse','[9][S0]x[9.01]',144000,16000,True,False),
        ('independent-overlap','[0][S0]x[9.01][1][S1]y[9.01]',144000,16000,True,True),
        ('fractional-duration','[0][S0]x[0.343333]',1,3,True,True),
        ('fractional-reject','[0][S0]x[0.343334]',1,3,True,False),
        ('atomic-reject','[0][S0]x[1][1][S1]y[9.010001]',144000,16000,True,False),
        ('incomplete','[0][S0]x[9.01]',144000,16000,False,False),
        ('malformed','[0][S0]x[1]junk',144000,16000,True,False),
        ('nested','[0][S0]x[1][2]',144000,16000,True,False),
        ('fraction-token','[0][S0]x[1/2]',144000,16000,True,False),
        ('nondecimal','[0][S0]x[1e0]',144000,16000,True,False),
    ]
    for name,text,frames,rate,complete,expected in cases:
        try:
            result = normalize(text,frames,rate,complete)
            accepted = True
        except (ValueError,ZeroDivisionError):
            accepted = False
        assert accepted == expected, name
        if name == 'inrange':
            assert result['segments'][0]['end'] == str(Fraction('8.999')) and not result['corrections']
        if name == 'independent-overlap':
            assert len(result['corrections']) == 2
    return [c[0] for c in cases]

if __name__ == '__main__':
    print('PASS', checks())

```

Runnable targeted probe (`backend/runtime/moss/eof_probe.py`, ignored):

```python
"""One retained-fixture decode; preserve raw evidence before exact correction."""
import hashlib, json, os, time, wave
from pathlib import Path
from fractions import Fraction
from eof_check import normalize, checks
root = Path(__file__).resolve().parent
os.environ['HF_HOME'] = str(root/'hf')
os.environ['HF_MODULES_CACHE'] = str(root/'modules')
os.environ['HF_HUB_OFFLINE'] = '1'
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from moss_transcribe_diarize.inference_utils import build_transcription_messages, prepare_inputs
assert torch.cuda.is_available()
fixture = root/'synthetic-overlap-2s.wav'
with wave.open(str(fixture)) as wav:
    frames, rate = wav.getnframes(), wav.getframerate()
    pcm_hash = hashlib.sha256(wav.readframes(frames)).hexdigest()
assert (frames, rate) == (144000,16000)
assert pcm_hash == '6fa9e0bfb2a432a37680625901b449a429e99c2e4128f407ac69b7474fed7e11'
print(json.dumps({'event':'boundary_checks','passed':checks()}),flush=True)
started=time.perf_counter()
model=AutoModelForCausalLM.from_pretrained(str(root/'model'),trust_remote_code=True,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda:0').eval()
processor=AutoProcessor.from_pretrained(str(root/'model'),trust_remote_code=True,local_files_only=True)
torch.cuda.synchronize()
load_seconds=time.perf_counter()-started
started=time.perf_counter()
torch.cuda.reset_peak_memory_stats()
with torch.autocast('cuda',dtype=torch.bfloat16):
    inputs=prepare_inputs(processor,build_transcription_messages(fixture),device=torch.device('cuda:0')).to('cuda:0')
prompt=inputs['input_ids'].shape[1]
budget=min(model.generation_config.max_new_tokens,model.config.text_config.max_position_embeddings-prompt)
assert budget>0
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    output=model.generate(**inputs,max_new_tokens=budget,do_sample=False)
torch.cuda.synchronize()
ids=output[0,prompt:].tolist()
eos=model.generation_config.eos_token_id
eos=[eos] if isinstance(eos,int) else eos
text=processor.tokenizer.decode(ids,skip_special_tokens=True).strip()
raw={'pid':os.getpid(),'frames':frames,'rate':rate,'pcm_sha256':pcm_hash,'raw_text':processor.tokenizer.decode(ids,skip_special_tokens=False),'decoded_text':text,'generated_ids':ids,'prompt_tokens':prompt,'budget':budget,'eos':eos,'complete':bool(ids and ids[-1] in eos),'load_seconds':load_seconds,'decode_seconds':time.perf_counter()-started,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'attention':model.config._attn_implementation}
(root/'eof-proof-raw.json').write_text(json.dumps(raw,indent=2),encoding='utf-8')
assert prompt+len(ids)<=model.config.text_config.max_position_embeddings
normalized=normalize(text,frames,rate,raw['complete'])
segments=normalized['segments']
overlaps=[(a,b) for a,s in enumerate(segments) for b,t in enumerate(segments) if a<b and s['speaker']!=t['speaker'] and max(Fraction(s['start']),Fraction(t['start']))<min(Fraction(s['end']),Fraction(t['end']))]
assert overlaps and normalized['corrections']
result={'raw':raw,'normalized':normalized,'overlaps':overlaps}
(root/'eof-proof.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result),flush=True)
print(json.dumps({'event':'normal_exit','pid':os.getpid()}),flush=True)

```

Task-1 gate conclusion: PASS from the new exact EOF/overlap proof together with
the retained pinned Windows CUDA environment, ordinary complete decodes,
cold/warm timings and exit/release evidence. No full app suites apply to this
research-only phase. SDD validation and git whitespace checks PASS.


## Task 3 — managed installer and worker (2026-10-19)

The managed implementation uses a separate versioned environment under
`backend/runtime/moss`, exact task-1 dependency versions, immutable source/model
file sizes and SHA256s, a hash-checked private wheelhouse, staged publication and
an interprocess installation lock. Source LICENSE/model README notices are retained.
The worker directly reuses `OwnedPhononProcess`; Phonon implementation is unchanged.
Setup subprocesses handshake their actual interpreter PID before running, so
shutdown verifies interpreter exit as well as the Windows venv launcher.
Status does not import ML, download or initialize CUDA. Lifecycle/input tests use
isolated temporary directories and SQLite configured before application imports.

The scoped runtime/parser suite passed **110 tests**; scoped Ruff, SDD validation
and whitespace checks passed. Tests cover separate-process setup serialization,
interrupted/corrupt installs and manifests, previous-version preservation,
cancellable installation-lock waits, owned setup cancellation, Python environment
isolation, worker reuse/death/malformed/wrong/duplicate replies, busy unload,
shutdown during setup/decode and actual Windows parent `os._exit` cleanup. The
parent-exit test runs a real backend-venv interpreter with a fake model worker;
it is process-ownership evidence, not GPU inference evidence. Actual interpreter
PID differs from the venv launcher and membership is checked before loading.

Initial real setup correctly **failed before publishing its manifest**: the
first lock admitted SoundFile 0.14.0's universal wheel, which lacked its Windows
libsndfile DLL. Pip installation and `pip check` passed, but the native
Transformers/SoundFile import gate failed with error 0x7e. The corrected lock pins
`soundfile-0.14.0-py2.py3-none-win_amd64.whl`, SHA256
`299491d3499460fb1b74bb4bd78b57ffc2d243a5fafa7b6ec1b264875c78453e`,
from official PyPI metadata. No system DLL/backend dependency was changed.
The failed stage was cleaned; retry reused 66 individually SHA-matching wheels
and downloaded only the corrected native wheel through the normal installer.

Task-1 research environment/model/source were read only. Verified snapshot bytes
were copied into the separate task-3 download cache; this is a cache-seeded real
installer run, not proof of re-downloading model weights. Hardlinks were
unsupported by this filesystem, so copies were used. The 2,867 MB Torch wheel
was actually downloaded. A narrow read-only check of relevant large pip-cache
bodies found no matching substitute; no pip-cache bytes were changed.
The initial failed run and corrected run retain separate proof/log files under
`backend/runtime/moss/task3/`; the shared setup log records both version paths.

Worker transcription validates canonical PCM16 mono 16 kHz WAV bytes and exact
frame dimensions, rejects input above 5400 seconds before inference, uses the
pinned processor/time markers and BF16 SDPA on asserted NVIDIA CUDA, budgets
from actual remaining 131072-token context, and independently validates the
strict native result at the manager boundary. Runtime provenance is outside
`moss-result-v1`. Readiness is a real complete silence generation probe: it
retains its actual raw output/completion and does not invent an empty transcript
or treat a silence hallucination as a validated transcription.

### Corrected managed run — PASS

The normal staged installer published version
`214ed56931e0623b6d49672627c2336d0db877566235622c1b45a5e1d5f984ff`
after hash checks, private wheel installation, `pip check` and native
Transformers/SoundFile/Torch CUDA imports. Corrected recorded run: 1075.370s;
manifest write was 902.359s after recorded start. These costs exclude the
preceding cache-seeding copies. The initial failed run cost 2009.542s including
its large download, failed native gate and staging cleanup; it remains a failure.
No inference was run with that invalid installation. The active offline pip
process showed measured CPU/I/O increases while installing; monitoring snapshots
are retained, without attributing slow setup to an unmeasured cause.

Real managed worker PID **19972**, Windows venv launcher **48444**, private
interpreter under the published version's `venv/Scripts/python.exe`. Model/source
revisions match task 1. RTX 5090, Torch 2.10.0+cu128, CUDA 12.8, BF16 and SDPA
were reported by the real worker. Worker model-load metric: **3.575s**; real
one-second-silence readiness generation: **13.618s**, 99 input / 23 output tokens,
EOS 151645, remaining-context budget 130973. Its actual output was a hallucinated
apology with a start marker but no terminal clock. This is a complete-generation
smoke probe, **not a valid silence transcription**. The manager exposes its
completion/raw evidence as `runtime.readiness_probe`; future Load/Test API/UI
must describe inference readiness and must not claim validated silence output.

Retained 144000-sample / 16000 Hz overlap fixture decoded in **6.922s**: 200 prompt
and 108 generated tokens, EOS 151645, budget **130872** (actual remaining context),
four native segments and all three overlap pairs (0,1), (1,2), (2,3). Backend
independent validation passed; final raw end `9.01` became exact `9` under
moss-eof-v1 at segment index 3, with full raw text/token IDs and correction
evidence retained. Words remain null. Process-local allocated/reserved bytes
after decode: 1,827,989,504 / 1,952,448,512; peaks: 1,874,400,256 / 1,952,448,512.
No accuracy or complete audible-word-coverage claim follows from EOS.

Unload verified the actual interpreter's tracked process handle signaled exit;
launcher exit code **0**, owned handles cleared, model/process flags false,
runtime metadata empty and temporary audio directory empty. Global GPU usage was
3659 → 6059 → 3647 MiB before load / before unload / after exit; the desktop and
unrelated processes can change these numbers. Actual owned interpreter exit,
rather than global memory alone, is the release evidence.

A separate offline proof then verified the complete installed files and all
**67 cached wheels** in **5.115s**, with downloader and subprocess entry points
set to fail: zero network/subprocess calls and no model load. This proves
verified cached reuse, not a second offline reinstall. Post-pin isolated suite:
**110 PASS** in 25.66s; scoped Ruff PASS. No Phonon shared code was changed.

Reproduce from repository root (SQLite environment is set before imports by the
proof scripts; tests set it explicitly):

```powershell
backend/.venv/Scripts/python.exe backend/runtime/moss/task3/prove-final.py
backend/.venv/Scripts/python.exe backend/runtime/moss/task3/offline-proof.py
$env:DB_PROVIDER='sqlite'; $env:DATABASE_URL='sqlite://'
cd backend
./.venv/Scripts/python.exe -m pytest src/test_moss_runtime.py src/test_moss.py -q
```

Proof scripts deliberately refuse to overwrite their JSON evidence; for a new
measurement use new task-scoped output filenames. Retained ignored artifacts:
`task3/managed-proof-final.json`, `prove-final.py`, `prove-final.log`,
`offline-proof.json`, `offline-proof.py`, `offline-proof.log`,
`installer-monitor-first.json`, `installer-monitor-second.json`,
`managed/setup.log` and `managed/worker.log`, all under `backend/runtime/moss/`.
Final managed-proof SHA256:
`dc13089a006f7d9d8e4316ffae8540b326fac21f90ccb63e51019f68eecc5eee`.
The strict result is `MossRuntime.transcribe(audio_bytes)["result"]`; runtime
provenance is its sibling `runtime`, outside the strict result object.

## Task 8 verification checkpoint — 2026-10-19 (blocked)

Static verification passed: frontend `npx tsc -b`; root backend-venv
`python -m ruff check .`; frontend `npm run lint` (0 errors, 43 existing
warnings). Full backend suite passed: **252 tests**, one pytest warning,
124.31 seconds. TRANSFORMERS_CACHE deprecation and post-exit swigvarlink warning
are recorded; neither is a model-quality result.

Safe backend invocation, from backend/:

```powershell
$env:PYTHONPATH='F:/Projects/PorchtimeIndex/backend/runtime/moss/task8;F:/Projects/PorchtimeIndex/backend'
$env:DB_PROVIDER='sqlite'; $env:DATABASE_URL='sqlite://'
$env:PYTHONDONTWRITEBYTECODE='1'
.venv/Scripts/python.exe -m pytest src -q --basetemp=runtime/moss/task8/pytest-tmp
```

Ignored `task8/sitecustomize.py` disables dotenv before application imports,
sets SQLite before imports, redirects ingestion DATA_DIR/RUNTIME_DIR, src.paths
and env_utils.ENV_PATH into task8/fixture, and denies production backend/data
and backend/.env access via an audit hook. Test assertions and collection remain
unchanged. Bootstrap SHA256:
`6cdddf1da8c65799def6423bb037a4a44f40aedb32e1c5ecff431235d26676ed`.
Backend log SHA256:
`c7abf382e27756700c6d914f1b6a350e82be52f72a566ce988dd703405e93a70`.

Full browser collection selected **19 tests** using installed Chrome and one
worker. It stopped at the first failure: **1 PASS, 1 FAIL, 17 not run**, 10.7s.
Command from frontend/:

```powershell
$env:PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH='C:/Program Files/Google/Chrome/Application/chrome.exe'
npm run test:e2e -- --config F:/Projects/PorchtimeIndex/backend/runtime/moss/task8/playwright.config.cjs
```

Task8 config applies installed Chrome globally, baseURL5173, workers1 and
maxFailures1; it excludes no test. Strict-port owned Vites5173/5175 used a
scratch config with task8 envDir and polling, VITE_API_URL fixture8000. A bind
probe established8000 was free before the task-owned stdlib fixture API started.
Synthetic channel1/video198/speaker26 records handle legacy tests; scan/process
responses are simulated with no jobs, downloader or external operations. The
first audio-download PASS proves fixture-backed browser expectations only,
not actual downloader behavior. No research or production audio was served in
this stopped browser run.

BLOCKER: frontend/e2e/channel-detail.spec.ts:18 uses
`getByRole('link', { name: 'Open' })` and strict mode finds three links: two
`aria-label="Open Chatalogue fixture"` and the exact `Open` link. This existing
channel test/UI mismatch is outside MOSS phase ownership. Proposed minimal
follow-up is test-only `{ name: 'Open', exact: true }`; no fix or workaround was
written. Root belongs to a different subsystem; full verification stops here.

Ignored evidence under backend/runtime/moss/task8/: pytest.log, e2e.log,
fixture-api.jsonl, sitecustomize.py, fixture_api.py, vite.config.mjs,
playwright.config.cjs and e2e-results/channel-detail-Channel-Det-90ab1-s-export-and-delete-buttons-chromium/
(test-failed-1.png and error-context.md). Owned fixture API/Vites were stopped.
Warnings: Node module.register deprecation, NO_COLOR/FORCE_COLOR notice and old
Browserslist data. No dependency update was attempted.

Task8 remains incomplete, 7/8 complete overall. No new real MOSS inference,
voice embeddings, cross-recording names, staged worker reuse, GPU handoff or
sequential reload measurement was performed. Prior task1/3 evidence remains
unchanged; it does not satisfy these remaining app-phase requirements. Do not
claim 120-second app-phase or 90-minute performance/memory coverage. Resume
this phase after explicit ownership/approval of the unrelated verification fix.


## Task8 approved selector repair resume — blocked


- 2026-10-19 task8 resume: owner explicitly approved only test-only exact:true
  Open selectors in frontend/e2e/channel-detail.spec.ts; all four occurrences
  repaired. Scoped `npx eslint e2e/channel-detail.spec.ts` PASS. Backend Python3.11
  fixture bind succeeds after system Python3.14 WinError10013. Initial browser
  attempt failed infrastructure (HTML instead of JSON), not a selector regression.
  Full fixture-backed rerun:19 selected,2 PASS/1 FAIL/16 unrun,39.0s; first channel
  navigation now passes. Next test waits for absent Search tab at
  e2e/pages/ChannelDetailPage.ts:30; snapshot shows Videos/Transcripts/Speakers/
  Chats/Clips. Stop at first genuine failure; no further repair or real inference.
  Task8 remains incomplete. Earlier ignored screenshots were inadvertently removed
  by reused Playwright outputDir; original e2e.log survives, new failure screenshot
  and context retained. Owned API/Vites stopped and no listeners remain on
  8000/5173/5175. WIP checkpoint subject: docs(moss): checkpoint next browser blocker.

Exact browser command (frontend): `npm run test:e2e -- --config F:/Projects/PorchtimeIndex/backend/runtime/moss/task8/playwright.config.cjs`.
Logs: backend/runtime/moss/task8/e2e-resume-1.log (infrastructure), e2e-resume-2.log (genuine Search-tab failure). Fixture API: `backend/.venv/Scripts/python.exe backend/runtime/moss/task8/fixture_api.py`, owned PID60876, observed loopback8000 LISTENING before rerun. Existing scratch Vite config/env retained. Synthetic fixture requests recorded in fixture-api.jsonl; no production backend started. Failed run does not prove the remaining16 browser tests or any actual MOSS app-phase requirement. No accuracy, long-input throughput or90-minute fit claim.
