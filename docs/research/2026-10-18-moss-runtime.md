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
5.41–9.01. Cross-speaker overlaps exist, including nonmonotonic segment-end ordering;
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
