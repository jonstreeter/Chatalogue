# MOSS versus Parakeet timing comparison — 2026-10-18

The engines agree closely on many matched boundaries, but they do not have identical clocks. This comparison cannot establish which engine follows the audible speech more accurately. Task 1 remains in progress (0/8 complete); this research does not apply the approved EOF rule or change any product behavior.

## Inputs, runtime and provenance

The ordinary conversation inputs are exactly identical decoded PCM: PCM16 mono, 16,000 Hz, 1,920,000 frames (120 seconds), SHA256 `836fc11d4a1de01270096e65aebedfbe272596761c7fc9e308da16e26e55322c`. Compared `backend/runtime/moss/conversation-120s.wav` with the audio referenced by retained actual-app result `backend/data/asr-benchmark/app-parakeet-120s.json`: `backend/data/asr-benchmark/trespasses-600-720.wav`. Therefore the retained warm Parakeet inference is reused, without rerunning the 120-second media.

The separate synthetic overlap input is the same retained 144,000-frame (9.000-second) PCM16 mono 16 kHz fixture, SHA256 `6fa9e0bfb2a432a37680625901b449a429e99c2e4128f407ac69b7474fed7e11`. It mixes source 0–6s with source 15–22s beginning at 2s; the second utterance is cut abruptly. MOSS raw labels and clocks, including terminal 9.01s, remain untouched.

MOSS source `e3bdd93f8ad58418e04659818efd5fd87a0bc486`, model `OpenMOSS-Team/MOSS-Transcribe-Diarize` revision `704aa4a9c304e8520be88901e0d1960158ef5b15`, native CUDA BF16 SDPA, private Torch 2.10.0+cu128 / Transformers 5.6.0. Existing raw results: `conversation-120s_0_46808.json` and `synthetic-overlap-2s_raw_0_4676.json` in `backend/runtime/moss/`. See [retained native evidence](2026-10-18-moss-runtime.md).

Parakeet model is `nvidia/parakeet-tdt-0.6b-v2`, restored from existing local `.nemo` checkpoint. Current backend metadata: torch 2.11.0+cu128, nemo_toolkit 2.7.0, transformers 4.57.1, soundfile 0.13.1. These current metadata are not a retroactive environment measurement for the October 15 artifact. Both retained 120s runs report requested/routed/used `parakeet`, CUDA memory and CUDA graph decoder provenance. The fresh 9s helper invokes the actual app transcription phase in in-memory SQLite with temporary ingestion directories, no production jobs, offline caches and process-local engine override. Log reports `Using device: cuda`; source explicitly moves Parakeet to CUDA for that device. Both fresh runs report requested/routed/used `parakeet`, CUDA memory and CUDA decoder provenance, with no Whisper transcription or CPU fallback. The retained 120s run uses Whisper for its English language probe on CUDA, distinct from transcript generation. The fresh 9s route uses video metadata text, without a Whisper language probe.

Fresh runtime evidence (actual app phase wall time; includes routing and app work):

| Input | Run | Phase seconds | Used engine | Peak PyTorch bytes |
| --- | --- | ---: | --- | ---: |
| 9s overlap | cold | 272.232 | parakeet | 2552870400 |
| 9s overlap | warm | 2.087 | parakeet | 2552870400 |

The retained 120s actual-app cold/warm phases took 139.137/7.345s. Existing MOSS 120s raw decodes took 57.327/53.471s. These scopes differ; this is not an equal-scope runtime benchmark. PyTorch memory excludes CTranslate2 and is not whole-device VRAM. Fresh child PID 33104 and its launcher completed with shell exit 0; exit evidence is retained in scratch.

## Reproducible matching method

The final warm run is selected consistently for each Parakeet input.

The runnable standard-library script `backend/runtime/moss/compare_parakeet_timing.py` validates PCM identity, Parakeet word bounds, engine-used provenance and timing arithmetic, and writes `parakeet-timing-comparison-20261018.json`. Lowercase text and remove punctuation (apostrophes included), then split on whitespace; retain numbers and lexical differences. Thus `ninety one` does not match `91`, and `cause` does not match `because`; contractions are not expanded (`we’re`/`were` punctuation removal can collide, an acknowledged limitation). No fuzzy text substitutions, numeric conversions or temporal fitting are used. For each MOSS segment independently, enumerate every contiguous identical normalized token sequence in the entire Parakeet word stream. Exactly one occurrence qualifies for aggregate timing statistics. Multiple occurrences remain ambiguous; there is no global monotonic assignment, especially for overlapping native segments.

Compare MOSS segment start/end against first/last corresponding Parakeet word boundary. Signed delta = MOSS minus Parakeet; positive means MOSS later. No MOSS internal word clock exists, and none is synthesized. Supplemental exact prefix/suffix anchors of at least three tokens are reported separately; the longest globally unique anchor touching that endpoint is used. A missing anchor means no qualifying unique sequence, not proof of omitted audible speech.

## Ordinary 120-second conversation

Coverage: 9/31 segments are unique exact full-phrase matches; 97/392 MOSS normalized tokens covered. Parakeet has 365 word records. Ambiguous segment indices (zero-based): [4, 9, 21, 27]; unmatched: [0, 1, 2, 3, 5, 6, 8, 12, 13, 15, 16, 17, 18, 19, 23, 24, 29, 30].

Across 18 endpoints: median absolute delta 0.110000s; mean absolute delta 0.231667s; signed range -1.110000 to +1.350000s.

All unique full-phrase matches:

| Index | Exact phrase | MOSS start/end | Parakeet start/end | Δstart / Δend seconds |
| ---: | --- | --- | --- | --- |
| 7 | You don't? | 23.83 / 24.42 | 24.08 / 24.40 | -0.25 / +0.02 |
| 10 | That wasn't really a thing. | 31.83 / 32.88 | 31.76 / 32.96 | +0.07 / -0.08 |
| 11 | That's true. Nobody was trying to do it themselves and all that stuff, but. | 33.33 / 36.48 | 33.28 / 36.32 | +0.05 / +0.16 |
| 14 | But that's because they also integrated so well, like nobody had like an accent or anything. | 47.13 / 50.76 | 47.12 / 50.72 | +0.01 / +0.04 |
| 20 | So I think you cut out for a little bit. | 81.27 / 82.95 | 79.92 / 83.12 | +1.35 / -0.17 |
| 22 | Yeah, you came back in on what? | 84.24 / 85.92 | 84.16 / 85.76 | +0.08 / +0.16 |
| 25 | I mean, my, I guess it's just my dad told me all about it. | 94.02 / 96.87 | 94.00 / 97.04 | +0.02 / -0.17 |
| 26 | I was babysat, my babysitter when I was young was a gal named Lucero. | 97.53 / 102.96 | 98.64 / 102.96 | -1.11 / -0.00 |
| 28 | We didn't say it like that. We said Lucero. I had no idea she was. | 103.86 / 107.49 | 104.00 / 107.20 | -0.14 / +0.29 |

All supplemental endpoint anchors (separate from full-phrase statistics):

| Index | Endpoint | Exact normalized anchor | Parakeet boundary | Δ seconds |
| ---: | --- | --- | ---: | ---: |
| 0 | end | drastic overhaul than even utah has although were getting there were getting there in every possible way in salt lake city | 6.40 | -0.46 |
| 1 | start | but i dont know theres | 6.72 | -0.15 |
| 1 | end | theres good mexican restaurants around | 10.08 | +0.03 |
| 2 | end | i went to an iranian restaurant a while back and it was | 13.60 | +0.14 |
| 5 | start | i mean i was born in | 15.04 | +0.08 |
| 5 | end | so i really dont i dont remember not living with a bunch of mexicans | 22.80 | -0.45 |
| 8 | end | and like food network before food network like the idea of like learning about new foods | 30.40 | +0.08 |
| 10 | start | that wasnt really a thing | 31.76 | +0.07 |
| 10 | end | that wasnt really a thing | 32.96 | -0.08 |
| 11 | start | thats true nobody was trying to do it themselves and all that stuff but | 33.28 | +0.05 |
| 11 | end | thats true nobody was trying to do it themselves and all that stuff but | 36.32 | +0.16 |
| 13 | end | when i was young there were mexicans around i didnt realize half of them were mexicans until i got my mission and realized those were mexican last names | 46.80 | +0.12 |
| 14 | start | but thats because they also integrated so well like nobody had like an accent or anything | 47.12 | +0.01 |
| 14 | end | but thats because they also integrated so well like nobody had like an accent or anything | 50.72 | +0.04 |
| 15 | end | maybe they were already all second generation by then or something but | 54.56 | -0.20 |
| 17 | end | acknowledgement that they were that different than any one of us or anything | 66.00 | -0.36 |
| 18 | end | it was only later that i was finally like oh thats a | 69.76 | +0.08 |
| 19 | end | learning mexican names so | 80.16 | -0.03 |
| 20 | start | so i think you cut out for a little bit | 79.92 | +1.35 |
| 20 | end | so i think you cut out for a little bit | 83.12 | -0.17 |
| 22 | start | yeah you came back in on what | 84.16 | +0.08 |
| 22 | end | yeah you came back in on what | 85.76 | +0.16 |
| 23 | start | i was saying what when i found out that my friends names were were | 86.40 | -0.03 |
| 23 | end | were spanish names | 92.08 | +0.05 |
| 25 | start | i mean my i guess its just my dad told me all about it | 94.00 | +0.02 |
| 25 | end | i mean my i guess its just my dad told me all about it | 97.04 | -0.17 |
| 26 | start | i was babysat my babysitter when i was young was a gal named lucero | 98.64 | -1.11 |
| 26 | end | i was babysat my babysitter when i was young was a gal named lucero | 102.96 | -0.00 |
| 28 | start | we didnt say it like that we said lucero i had no idea she was | 104.00 | -0.14 |
| 28 | end | we didnt say it like that we said lucero i had no idea she was | 107.20 | +0.29 |
| 29 | start | you didnt notice | 108.64 | -1.14 |
| 29 | end | like you didnt notice that all the garcias were brown | 111.12 | -0.27 |
| 30 | end | kind of like once in a while say im brownish im brownish or im sort of sort of thing like that but then no a lot | 119.76 | +0.21 |

Unmatched full phrases (no unique statistics assigned):

- 0: A drastic overhaul than even Utah has, although we're getting there, we're getting there in every possible way in Salt Lake City.
- 1: But I don't know, there's, there's good Mexican restaurants around.
- 2: I went to, I went to an Iranian restaurant a while back and it was.
- 3: Oh, really?
- 5: I mean, I was born in ninety one, so I really don't, I don't remember not living with a bunch of Mexicans.
- 6: You know, cause.
- 8: Well, and, and like Food Network, before Food Network, like the idea of like learning about new foods.
- 12: Yeah, no internet.
- 13: When, when I was young, there were Mexicans around. I didn't realize half of them were Mexicans until I got my mission and realized those were Mexican last names.
- 15: And maybe they were already all second generation by then or something, but.
- 16: Uh, like honestly, my Mexican friends when I was little that I went to high school with, I had no, I had no real rec.
- 17: You know, acknowledgement that they were that different than any one of us or anything.
- 18: And, uh, it was only later that I was finally like, oh, that's a.
- 19: Mom, it was a what? You know, I, I did that a whole lot when I was, uh, learning Mexican names. So.
- 23: I was saying what, when I found out that my friends' names were, were Latin, were Spanish names.
- 24: Oh, really?
- 29: You didn't notice, like you didn't notice that all the Garcias were brown?
- 30: I mean, I, I noticed her, I noticed her brown. There were some guys who were kind of like, once in a while say, I'm brownish, I'm brownish, or I'm sort of, sort of thing like that, but then no, a lot.

Ambiguous full phrases (all Parakeet candidate spans, seconds):

- 4: Yeah. — 14.48–14.72; 84.16–84.40
- 9: That's true. — 30.96–31.44; 33.28–33.68
- 21: I did? — 76.40–76.72; 83.44–83.84
- 27: I mean. — 15.04–15.28; 94.00–94.24; 103.28–103.44; 112.00–112.32

## Synthetic overlap — separate result

Warm Parakeet transcript: “A drastic overhaul, and even you can't. I was born in 81. I really don't remember not living with a bunch of Mexicans.” This differs from MOSS's “91” and does not retain the competing S01 phrase. The three exact endpoint anchors below give deltas −0.16, +0.35 and +0.13s; no whole-phrase aggregate can be calculated.

Coverage: 0/4 segments are unique exact full-phrase matches; 0/39 MOSS normalized tokens covered. Parakeet has 23 word records. Ambiguous segment indices (zero-based): []; unmatched: [0, 1, 2, 3].

All unique full-phrase matches:

| Index | Exact phrase | MOSS start/end | Parakeet start/end | Δstart / Δend seconds |
| ---: | --- | --- | --- | --- |

All supplemental endpoint anchors (separate from full-phrase statistics):

| Index | Endpoint | Exact normalized anchor | Parakeet boundary | Δ seconds |
| ---: | --- | --- | ---: | ---: |
| 0 | start | a drastic overhaul | 0.16 | -0.16 |
| 1 | end | i really dont | 4.40 | +0.35 |
| 3 | end | dont remember not living with a bunch of mexicans | 8.88 | +0.13 |

Unmatched full phrases (no unique statistics assigned):

- 0: A drastic overhaul than even Utah has.
- 1: I mean, I was born in 91, so I really don't.
- 2: We're getting there in every possible way in Salt Lake City.
- 3: I don't remember not living with a bunch of Mexicans.

Ambiguous full phrases (all Parakeet candidate spans, seconds):


## Interpretation and limits

Ordinary conversation examples span the recording: the first segment end is 5.94 vs 6.40s (−0.46s, exact suffix anchor), the middle “But that’s because…” phrase is 47.13–50.76 vs 47.12–50.72s (+0.01/+0.04s), and the final exact suffix ends at 119.97 vs 119.76s (+0.21s). Larger full-phrase start differences include “So I think you cut out…” (+1.35s) and “I was babysat…” (−1.11s), while their ends differ −0.17/0.00s. These observations do not establish a consistent accumulating drift model: sparse matched boundaries and lexical disagreements cannot calibrate the entire transcript.

Text differences include initial “A” versus none, repeated “there’s” versus one occurrence, “ninety one” versus “91”, “cause” versus “because”, “Yeah, no internet” versus “after that”, “Latin” versus none, “You didn’t notice” versus “like”, and differing final words. Whole-phrase coverage is consequently limited and selected toward agreement. Native MOSS overlap may capture competing utterances that single-stream Parakeet merges or loses; missing exact overlap phrases cannot be interpreted as simple timing drift or speaker accuracy. No WER, human timing ground truth, speaker accuracy, acceptance threshold or production readiness is claimed. No timestamps were corrected, normalized or substituted into MOSS output.

## Commands and retained evidence

From repository root in PowerShell (choose a new output path for an intentional rerun; the helper refuses overwrite):

```powershell
backend/.venv/Scripts/python.exe scripts/benchmark_asr.py app backend/runtime/moss/synthetic-overlap-2s.wav --output backend/runtime/moss/parakeet-overlap-9s-20261018.json --app-engine parakeet --parakeet-checkpoint backend/data/asr-benchmark/models/parakeet-v2/parakeet-tdt-0.6b-v2.nemo
backend/.venv/Scripts/python.exe backend/runtime/moss/compare_parakeet_timing.py
python scripts/check_sdd_docs.py validate
git diff --check
```

Raw Parakeet output, log and shell exit file are `backend/runtime/moss/parakeet-overlap-9s-20261018.{json,log}` and `parakeet-overlap-9s-20261018-exit.txt`; comparison JSON retains every candidate, lexical phrase and anchor. They remain ignored research scratch. Comparison script is the reproducible arithmetic check; no application suites were run for this research-only change. No production DB/settings/recordings, dependency installations or pinned MOSS assets changed.

Checkpoint SHA256: `d99e39955c9d3d0350d8fb7c75e40c64a2b2eaeb003883d7c941fd2e8747b28c` (read-only `Get-FileHash` of the local Parakeet `.nemo`).

Structural checks: conversation: Parakeet issues=[], MOSS out-of-bounds segment indices=[]; overlap: Parakeet issues=[], MOSS out-of-bounds segment indices=[3]. The overlap raw terminal 9.01s is retained as the prior strict-bound failure; this comparison does not resolve it.

## Complete comparison source

Save this as `backend/runtime/moss/compare_parakeet_timing.py` to reproduce the arithmetic using retained inputs above. The script writes only its new scratch comparison JSON.

```python
import json, re, wave, hashlib, statistics, importlib.metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def norm(s): return re.sub(r'[^\w\s]', '', s.lower()).split()
def wav(p):
    with wave.open(str(p), 'rb') as w:
        return dict(path=str(p.relative_to(ROOT)), frames=w.getnframes(), rate=w.getframerate(), channels=w.getnchannels(), width=w.getsampwidth(), pcm_sha256=hashlib.sha256(w.readframes(w.getnframes())).hexdigest())
def compare(m, p, duration):
    segments=m.get('segments')
    if segments is None:
        segments=[dict(start=float(a),speaker=s,text=t.strip(),end=float(b)) for a,s,t,b in re.findall(r'\[([\d.]+)\]\[(S\d+)\]\s*(.*?)\[([\d.]+)\]',m['text'])]
    run=p['application']['runs'][-1]
    assert run['provenance']['transcription_engine_used']=='parakeet'
    words=[w for s in run['raw']['segments'] for w in s['words']]
    tokens=[]
    for i,w in enumerate(words):
        assert 0<=w['start']<=w['end']<=duration
        for token in norm(w['word']): tokens.append((token,i))
    flat=[t for t,i in tokens]
    rows=[]
    for si,s in enumerate(segments):
        ts=norm(s['text']); hits=[j for j in range(len(flat)-len(ts)+1) if flat[j:j+len(ts)]==ts]
        row=dict(index=si,**s,tokens=len(ts),matches=[],anchors={})
        for side in ('start','end'):
            for size in range(len(ts),2,-1):
                needle=ts[:size] if side=='start' else ts[-size:]
                positions=[j for j in range(len(flat)-size+1) if flat[j:j+size]==needle]
                if len(positions)==1:
                    j=positions[0]; wi=tokens[j if side=='start' else j+size-1][1]
                    row['anchors'][side]=dict(text=' '.join(needle),tokens=size,parakeet=words[wi][side],delta=round(s[side]-words[wi][side],6))
                    break
        for j in hits:
            a,b=words[tokens[j][1]]['start'],words[tokens[j+len(ts)-1][1]]['end']
            row['matches'].append(dict(parakeet_start=a,parakeet_end=b,start_delta=round(s['start']-a,6),end_delta=round(s['end']-b,6),word_start=tokens[j][1],word_end=tokens[j+len(ts)-1][1]))
        rows.append(row)
    matched=[r for r in rows if len(r['matches'])==1]
    ds=[r['matches'][0][k] for r in matched for k in ('start_delta','end_delta')]
    assert all(abs(r['matches'][0][side+'_delta']-(r[side]-r['matches'][0]['parakeet_'+side]))<1e-6 for r in matched for side in ('start','end'))
    # Only segment endpoints are compared; internal matching substrings lack MOSS word clocks.
    return dict(duration=duration,parakeet_words=len(words),parakeet_structural_issues=run['timestamp_review']['issues'],moss_out_of_bounds=[i for i,s in enumerate(segments) if not 0<=s['start']<=s['end']<=duration],moss_tokens=sum(r['tokens'] for r in rows),segments=len(rows),unique_full_matches=len(matched),matched_tokens=sum(r['tokens'] for r in matched),ambiguous=[r['index'] for r in rows if len(r['matches'])>1],unmatched=[r['index'] for r in rows if not r['matches']],endpoint_stats=dict(n=len(ds),mean_absolute=statistics.mean(map(abs,ds)) if ds else None,median_absolute=statistics.median(map(abs,ds)) if ds else None,signed_range=[min(ds),max(ds)] if ds else None),rows=rows)

if __name__=='__main__':
    audio=[wav(ROOT/p) for p in ['backend/data/asr-benchmark/trespasses-600-720.wav','backend/runtime/moss/conversation-120s.wav','backend/runtime/moss/synthetic-overlap-2s.wav']]
    assert {k:v for k,v in audio[0].items() if k!='path'}=={k:v for k,v in audio[1].items() if k!='path'}
    result=dict(audio=audio,packages={n:importlib.metadata.version(n) for n in ['torch','nemo_toolkit','transformers','soundfile']},conversation=compare(read(HERE/'conversation-120s_0_46808.json'),read(ROOT/'backend/data/asr-benchmark/app-parakeet-120s.json'),120),overlap=compare(read(HERE/'synthetic-overlap-2s_raw_0_4676.json'),read(HERE/'parakeet-overlap-9s-20261018.json'),9))
    out=HERE/'parakeet-timing-comparison-20261018.json'
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    for key in ['conversation','overlap']:
        print(key,json.dumps({k:v for k,v in result[key].items() if k!='rows'}))
        for r in result[key]['rows']: print(r['index'],r['text'],r['matches'])
```
