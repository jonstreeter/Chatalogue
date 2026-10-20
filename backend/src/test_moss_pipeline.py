"""MOSS bounded recording checkpoint regressions, using temporary files and SQLite."""
import copy
import io
import json
import wave
from types import SimpleNamespace

import pytest

from src.test_phonon import isolated_service  # noqa: F401
from src.services.ingestion.moss import MossValidationError, parse_generation
from src.services.ingestion.moss_runtime import MODEL_REVISION, SOURCE_REVISION

PROVENANCE = dict(model="MOSS", model_revision=MODEL_REVISION, source_revision=SOURCE_REVISION)


@pytest.fixture
def harness(isolated_service, tmp_path, monkeypatch):  # noqa: F811
    service = isolated_service
    audio = tmp_path / 'audio.wav'
    with wave.open(str(audio), 'wb') as stream:
        stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        stream.writeframes(b'\0\0' * 144001)
    video = SimpleNamespace(id=13, title='native', duration=999)
    completion = dict(complete=True, reason='eos', eos_token_id=151645,
                      terminal_token_id=151645, input_tokens=100, output_tokens=20,
                      max_new_tokens=1000, context_limit=131072)
    native = parse_generation('[0.10][S0]outer[9.0100][2][S1]inner[3]', 144001, 16000, completion)
    calls = []
    def transcribe(data):
        with wave.open(io.BytesIO(data)) as stream:
            calls.append((stream.getnframes(), stream.getframerate()))
        return dict(result=copy.deepcopy(native), runtime=PROVENANCE.copy())
    manager = SimpleNamespace(transcribe=transcribe)
    monkeypatch.setattr(service, '_get_moss_runtime', lambda: manager)
    monkeypatch.setattr(service, '_moss_speech_hints', lambda *_: [])
    monkeypatch.setattr(service, '_update_transcription_stage_progress', lambda *_, **__: None)
    monkeypatch.setattr(service, '_resolve_transcription_route', lambda *_: dict(engine='moss', requested_engine='moss'))
    for method in ('_check_job_not_paused', '_record_job_stage_start', '_update_job_status_detail',
                   '_update_job_progress', '_upsert_job_payload_fields'):
        monkeypatch.setattr(service, method, lambda *_: None)
    def normalize(command, **_):
        import shutil
        shutil.copyfile(audio, command[-1])
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr('subprocess.run', normalize)
    return service, video, audio, manager, calls, native


def run(harness, force=False):
    service, video, audio, *_ = harness
    return service._process_moss_phase(video, audio, None, dict(requested_engine='moss'), force)


def test_ffmpeg_metadata_wav_uses_canonical_pcm_ipc(harness, tmp_path):
    import subprocess
    from src.services.ingestion.moss_worker import validate_wav

    service, _, audio, manager, _, _ = harness
    with wave.open(str(audio)) as stream:
        original_pcm = stream.readframes(stream.getnframes())
    metadata_audio = tmp_path / 'metadata.wav'
    # Popen remains real while this harness replaces pipeline subprocess.run.
    process = subprocess.Popen([
        service._get_ffmpeg_cmd(), '-v', 'error', '-y', '-i', str(audio),
        '-c:a', 'pcm_s16le', '-metadata', 'title=MOSS IPC regression', str(metadata_audio),
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, error = process.communicate()
    assert process.returncode == 0, error.decode(errors='replace')
    metadata_bytes = metadata_audio.read_bytes()
    assert metadata_bytes[36:40] == b'LIST'
    audio.write_bytes(metadata_bytes)
    original_transcribe = manager.transcribe
    accepted = []

    def transcribe(data):
        pcm, count, rate = validate_wav(data)
        assert pcm == original_pcm
        assert (count, rate) == (144001, 16000)
        assert len(data) == 44 + len(original_pcm)
        accepted.append(True)
        return original_transcribe(data)

    manager.transcribe = transcribe
    _, duration = run(harness)
    assert accepted == [True]
    assert duration == 144001 / 16000


def test_native_roundtrip_identity_retry(harness):
    service, video, audio, _, calls, native = harness
    segments, duration = run(harness)
    raw = audio.parent / 'native_transcript_raw.json'
    original = raw.read_bytes()
    for _ in range(2):
        reused, reused_duration, engine = service._load_raw_transcript_checkpoint(video, audio)
        assert engine == 'moss' and reused_duration == duration == 144001 / 16000
        assert [(s.speaker, s.start, s.end, s.text, s.words) for s in reused] == [
            (s.speaker, s.start, s.end, s.text, s.words) for s in segments]
        assert raw.read_bytes() == original
    reused, _ = service._process_transcribe_phase(video, audio)
    assert reused[0].speaker == 'chunk-00000000:S0' and calls == [(144001, 16000)]
    assert json.loads(original)['chunks'][0]['result'] == native
    assert native['corrections'][0]['raw_end'] == '9.0100'


def test_atomic_failure_preserves_complete(harness, monkeypatch):
    run(harness)
    raw = harness[2].parent / 'native_transcript_raw.json'
    original = raw.read_bytes()
    def fail(*_):
        raise OSError('publish failure')
    monkeypatch.setattr('os.replace', fail)
    with pytest.raises(OSError, match='publish failure'):
        run(harness, force=True)
    assert raw.read_bytes() == original
    assert not any(p.is_dir() for p in raw.parent.iterdir())


@pytest.mark.parametrize('mutation', ['dimensions', 'completion', 'raw', 'context'])
def test_invalid_reply_never_publishes(harness, mutation):
    _, _, audio, manager, _, native = harness
    bad = copy.deepcopy(native)
    if mutation == 'dimensions':
        bad['sample_count'] += 1
    elif mutation == 'completion':
        bad['completion']['complete'] = False
    elif mutation == 'context':
        bad['completion']['input_tokens'] = 131072
    else:
        bad['raw_text'] += '[broken'
    manager.transcribe = lambda _: dict(result=bad, runtime=PROVENANCE.copy())
    with pytest.raises(MossValidationError):
        run(harness)
    assert not (audio.parent / 'native_transcript_raw.json').exists()


def test_failure_cancel_retry_full_input(harness, monkeypatch):
    service, _, audio, manager, calls, _ = harness
    actual = manager.transcribe
    def fail(_):
        raise RuntimeError('worker interrupted')
    manager.transcribe = fail
    with pytest.raises(RuntimeError, match='worker interrupted'):
        run(harness)
    manager.transcribe = actual
    def paused(data):
        result = actual(data)
        monkeypatch.setattr(service, '_check_job_not_paused', lambda *_: (_ for _ in ()).throw(RuntimeError('paused after inference')))
        return result
    manager.transcribe = paused
    with pytest.raises(RuntimeError, match='paused'):
        run(harness)
    assert not (audio.parent / 'native_transcript_raw.json').exists()
    manager.transcribe = actual
    monkeypatch.setattr(service, '_check_job_not_paused', lambda *_: None)
    run(harness)
    assert calls == [(144001, 16000), (144001, 16000)]


@pytest.mark.parametrize('records', [False, True])
def test_mixed_partials_and_force(harness, records):
    service, video, _, _, calls, _ = harness
    path = service._get_temp_transcript_jsonl_path(video.id) if records else service._get_temp_transcript_path(video.id)
    path.write_text('{"engine":"phonon"}')
    with pytest.raises(MossValidationError, match='another or unknown engine'):
        run(harness)
    assert not calls
    run(harness, force=True)
    assert not path.exists() and len(calls) == 1




def test_corrupt_cache_rejected(harness):
    service, video, audio, _, calls, _ = harness
    run(harness)
    raw = audio.parent / 'native_transcript_raw.json'
    data = json.loads(raw.read_text())
    data['chunks'][0]['result']['segments'][0]['end'] = '10'
    raw.write_text(json.dumps(data))
    with pytest.raises(MossValidationError):
        service._process_transcribe_phase(video, audio)
    assert len(calls) == 1


def test_legacy_reuse_before_partials(harness):
    service, video, audio, _, calls, _ = harness
    raw = audio.parent / 'native_transcript_raw.json'
    raw.write_text(json.dumps(dict(transcription_engine_used='phonon', total_duration=9,
        segments=[dict(start=0, end=1, text='legacy', words=[dict(start=0, end=1, word='legacy')])])))
    service._get_temp_transcript_path(video.id).write_text('{"engine":"phonon"}')
    segments, duration = service._process_transcribe_phase(video, audio)
    assert segments[0].text == 'legacy' and duration == 9 and not calls


def test_changed_source_refused(harness):
    service, video, audio, _, calls, _ = harness
    run(harness)
    audio.write_bytes(audio.read_bytes() + b'changed')
    with pytest.raises(MossValidationError, match='source changed'):
        service._load_raw_transcript_checkpoint(video, audio)
    assert len(calls) == 1


def test_force_payload_replaces_native_and_ignores_partial(harness):
    from sqlmodel import Session
    from src.db.database import Job
    from src.services.ingestion import runtime
    service, video, audio, _, calls, _ = harness
    run(harness)
    service._get_temp_transcript_path(video.id).write_text('{"engine":"whisper"}')
    with Session(runtime.engine) as session:
        session.add(Job(id=13, video_id=13, job_type='process', status='queued',
                        payload_json='{"force_retranscription":true}'))
        session.commit()
    service._process_transcribe_phase(video, audio, job_id=13)
    assert calls == [(144001, 16000), (144001, 16000)]
    assert not service._get_temp_transcript_path(video.id).exists()


def test_complete_silence_reuses_without_word_coverage_retry(harness):
    service, video, audio, manager, calls, native = harness
    silence = parse_generation('  ', native['sample_count'], native['sample_rate'], native['completion'])
    def transcribe(_):
        calls.append('silence')
        return dict(result=silence, runtime=PROVENANCE.copy())
    manager.transcribe = transcribe
    assert run(harness)[0] == []
    assert service._process_transcribe_phase(video, audio)[0] == []
    assert calls == ['silence']


def test_competing_models_release_before_inference(harness, monkeypatch):
    service, _, _, manager, _, _ = harness
    order = []
    for name in ('phonon', 'whisper', 'parakeet', 'diarization'):
        method = f'_release_{name}_model' if name != 'diarization' else '_release_diarization_models'
        monkeypatch.setattr(service, method, lambda *_, label=name, **__: order.append(label))
    actual = manager.transcribe
    def infer(audio):
        assert order == ['phonon', 'whisper', 'parakeet', 'diarization']
        order.append('inference')
        return actual(audio)
    manager.transcribe = infer
    run(harness)
    assert order[-1] == 'inference'


def legacy_run(harness):
    _, duration = run(harness)
    service, _, audio, *_ = harness
    path = audio.parent / 'native_transcript_raw.json'
    data = json.loads(path.read_text())
    source = audio.stat()
    legacy = dict(transcription_engine_used='moss', moss_result=data['chunks'][0]['result'],
                  moss_runtime=PROVENANCE.copy(), moss_source=dict(size=source.st_size, mtime_ns=source.st_mtime_ns))
    path.write_text(json.dumps(legacy))
    return service._moss_checkpoint_segments(legacy, audio)


@pytest.fixture
def identity_harness(harness, monkeypatch):
    import numpy as np
    from sqlmodel import Session
    from src.db.database import Channel, Video
    from src.services.ingestion import runtime

    service, _, audio, *_ = harness
    with Session(runtime.engine) as session:
        session.add(Channel(id=1, url='fixture', name='Fixture'))
        session.add(Video(id=13, youtube_id='fixture13', channel_id=1, title='native'))
        session.commit()
        video = session.get(Video, 13)
        session.expunge(video)
    harness = (service, video, *harness[2:])
    service.device = 'cpu'
    service.embedding_model = object()
    service.embedding_inference = object()
    monkeypatch.setattr(service, '_release_diarization_models', lambda *_, **__: None)
    monkeypatch.setattr(service, '_load_models', lambda *_, **__: pytest.fail('fixture must not load models'))
    crops = []
    def embed(_, segment):
        crops.append((segment.start, segment.end))
        vector = np.zeros(16)
        vector[0 if segment.start < 2 or segment.start >= 3 else 1] = 1
        return vector
    monkeypatch.setattr(service, 'get_speaker_embedding', embed)
    monkeypatch.setattr(service, '_load_audio_for_pyannote', lambda _: object())
    monkeypatch.setattr(service, '_should_release_parakeet_before_diarize', lambda **_: False)
    monkeypatch.setattr(service, '_apply_entity_repair_to_segments', lambda *_, **__: {'changed': False})
    monkeypatch.setattr(service, '_apply_formatting_cleanup_to_segments', lambda *_, **__: {'changed': False})
    monkeypatch.setattr(service, '_run_diarization_with_adaptive_batch', lambda *_, **__: pytest.fail('native must not cluster'))
    monkeypatch.setattr(service, '_consolidate_transcript_segments', lambda *_: pytest.fail('native must retain turns'))
    return harness, crops


def test_native_identity_two_recordings_and_channel_isolation(identity_harness):
    import pickle
    from sqlmodel import Session, select
    from src.db.database import Channel, Speaker, SpeakerEmbedding, TranscriptSegment, Video
    from src.services.ingestion import runtime

    harness, crops = identity_harness
    service, video, audio, manager, _, native = harness
    # Both voices have a clean region while their native turns still overlap.
    native = parse_generation('[0][S0]outer[4][2][S1]inner[3][5][S1]guest[7]',
                              native['sample_count'], native['sample_rate'], native['completion'])
    manager.transcribe = lambda _: dict(result=native, runtime=PROVENANCE.copy())
    segments, _ = legacy_run(harness)
    # Fixture embedding keyed to clean span, deliberately independent of label.
    import numpy as np
    def embed(_, segment):
        crops.append((segment.start, segment.end))
        return np.eye(16)[0 if segment.start < 2 or segment.start == 3 else 1]
    service.get_speaker_embedding = embed
    with Session(runtime.engine) as session:
        session.add(Channel(id=2, name='Other', url='other'))
        other = Speaker(channel_id=2, name='Wrong channel', embedding_blob=pickle.dumps(np.eye(16)[0]))
        session.add(other)
        session.commit()
        session.add(SpeakerEmbedding(speaker_id=other.id, embedding_blob=other.embedding_blob))
        session.commit()
    for recording in range(2):
        if recording:
            with Session(runtime.engine) as session:
                video = Video(id=14, youtube_id='fixture14', channel_id=1, title='second')
                session.add(video)
                session.commit()
                session.refresh(video)
                session.expunge(video)
            data = json.loads((audio.parent / 'native_transcript_raw.json').read_text())
            raw = data['moss_result']['raw_text'].replace('S0', 'S8').replace('S1', 'S9')
            data['moss_result'] = parse_generation(raw, native['sample_count'], native['sample_rate'], native['completion'])
            (audio.parent / 'second_transcript_raw.json').write_text(json.dumps(data))
        service._get_temp_diarization_path(video.id).write_text('legacy RTTM must be ignored')
        service._process_diarize_phase(video, audio, segments)
        assert not service._get_temp_diarization_path(video.id).exists()
        with Session(runtime.engine) as session:
            rows = session.exec(select(TranscriptSegment).where(TranscriptSegment.video_id == video.id).order_by(TranscriptSegment.id)).all()
            names = [session.get(Speaker, row.speaker_id).name for row in rows]
            assert [(r.start_time, r.end_time, r.text, r.words) for r in rows] == [
                (0, 4, 'outer', None), (2, 3, 'inner', None), (5, 7, 'guest', None)]
            assert rows[0].speaker_id != rows[1].speaker_id == rows[2].speaker_id
            assert names == (['Speaker 1', 'Speaker 2', 'Speaker 2'] if not recording else ['Host', 'Guest', 'Guest'])
            if not recording:
                session.get(Speaker, rows[0].speaker_id).name = 'Host'
                session.get(Speaker, rows[1].speaker_id).name = 'Guest'
                session.commit()
        assert f'{video.title}.srt' in [p.name for p in audio.parent.iterdir()]
    # Existing selection prefers >=2 seconds, so the shorter clean tail is
    # available after subtraction but is not selected for embedding.
    assert set(crops) == {(0, 2), (5, 7)}
    from pyannote.core import Annotation, Segment
    turns = Annotation()
    turns[Segment(0, 4), 0] = 'outer'
    turns[Segment(2, 3), 1] = 'inner'
    assert [(s.start, s.end) for s in service._native_clean_speaker_segments(turns, 'outer')] == [(0, 2), (3, 4)]


def test_native_no_clean_sample_retains_distinct_unenrolled_locals(identity_harness, monkeypatch):
    from sqlmodel import Session, select
    from src.db.database import SpeakerEmbedding, TranscriptSegment
    from src.services.ingestion import runtime
    harness, crops = identity_harness
    service, video, audio, manager, _, native = harness
    native = parse_generation('[0][S0]one[4][0][S1]two[4]', native['sample_count'], native['sample_rate'], native['completion'])
    manager.transcribe = lambda _: dict(result=native, runtime=PROVENANCE.copy())
    segments, _ = legacy_run(harness)
    warnings = []
    monkeypatch.setattr(service, '_upsert_job_payload_fields', lambda _, fields: warnings.append(fields))
    service._process_diarize_phase(video, audio, segments)
    with Session(runtime.engine) as session:
        rows = session.exec(select(TranscriptSegment)).all()
        assert len(rows) == 2 and rows[0].speaker_id != rows[1].speaker_id
        assert all(row.speaker_id and row.matched_profile_id is None for row in rows)
        assert session.exec(select(SpeakerEmbedding)).all() == []
    assert not crops and any('moss_identity_warning' in item for item in warnings)


def test_missing_identity_model_preserves_complete_native_checkpoint(identity_harness, monkeypatch):
    from sqlmodel import Session, select
    from src.db.database import TranscriptSegment, Video
    from src.services.ingestion import runtime
    harness, _ = identity_harness
    service, video, audio, *_ = harness
    segments, _ = legacy_run(harness)
    raw = audio.parent / 'native_transcript_raw.json'
    original = raw.read_bytes()
    service.embedding_model = service.embedding_inference = None
    order = []
    monkeypatch.setattr(service, '_release_moss_model', lambda *_, **__: order.append('exit'))
    def load(_, *, embedding_only):
        assert embedding_only and order == ['exit']
        order.append('load')
    monkeypatch.setattr(service, '_load_models', load)
    with pytest.raises(RuntimeError, match='voice model is unavailable'):
        service._process_diarize_phase(video, audio, segments)
    assert raw.read_bytes() == original and order == ['exit', 'load']
    with Session(runtime.engine) as session:
        assert session.exec(select(TranscriptSegment)).all() == []
        assert session.get(Video, video.id).status != 'completed'

def sized_recording(harness, seconds=250):
    audio, manager, native = harness[2], harness[3], harness[5]
    with wave.open(str(audio), 'wb') as stream:
        stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        for index in range(seconds):
            stream.writeframesraw((index % 128).to_bytes(2, 'little') * 16000)
    calls = []
    def infer(data):
        with wave.open(io.BytesIO(data)) as stream:
            count = stream.getnframes()
            pcm = stream.readframes(count)
        calls.append((count, pcm[:2]))
        return dict(result=parse_generation('[0][S0]voice[1]', count, 16000, native['completion']),
                    runtime=PROVENANCE.copy())
    manager.transcribe = infer
    return calls


def test_long_real_normalized_wav_has_bounded_exact_coverage(harness, monkeypatch):
    import hashlib
    import subprocess
    service, _, audio, manager, _, native = harness
    # Real >90 minute PCM WAV; FFmpeg normalization runs for real, inference is instrumented.
    with wave.open(str(audio), 'wb') as stream:
        stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        for _ in range(5401):
            stream.writeframesraw(b'\0\0' * 16000)
        stream.writeframesraw(b'\1\0')
    def normalize(command, **_):
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, error = process.communicate()
        assert process.returncode == 0, error
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess, 'run', normalize)
    calls, digest = [], hashlib.sha256()
    def infer(data):
        assert len(data) <= 44 + 120 * 16000 * 2
        with wave.open(io.BytesIO(data)) as stream:
            count = stream.getnframes()
            digest.update(stream.readframes(count))
        calls.append(count)
        return dict(result=parse_generation(' ', count, 16000, native['completion']), runtime=PROVENANCE.copy())
    manager.transcribe = infer
    assert run(harness) == ([], (5401 * 16000 + 1) / 16000)
    raw = json.loads((audio.parent / 'native_transcript_raw.json').read_text())
    chunks = raw['chunks']
    assert sum(calls) == raw['frontier'] == 5401 * 16000 + 1
    assert max(calls) == 120 * 16000 and len(calls) == 46
    assert raw['moss_source']['pcm_sha256'] == digest.hexdigest()
    assert [c['start_frame'] for c in chunks] == [0] + [c['end_frame'] for c in chunks[:-1]]
    assert all(c['result']['raw_text'] == ' ' for c in chunks)


@pytest.mark.parametrize('failure', ['crash', 'cancel'])
def test_later_failure_resume_skips_committed(harness, monkeypatch, failure):
    service, video, audio, manager, *_ = harness
    calls = sized_recording(harness)
    actual = manager.transcribe
    def interrupted(data):
        if len(calls) == 1:
            if failure == 'cancel':
                monkeypatch.setattr(service, '_check_job_not_paused', lambda *_: (_ for _ in ()).throw(RuntimeError('cancel')))
                return actual(data)
            raise RuntimeError('crash')
        return actual(data)
    manager.transcribe = interrupted
    with pytest.raises(RuntimeError, match=failure):
        run(harness)
    state = json.loads(service._get_temp_transcript_path(video.id).read_text())
    assert state['chunk_count'] == 1 and state['frontier'] == 120 * 16000
    monkeypatch.setattr(service, '_check_job_not_paused', lambda *_: None)
    manager.transcribe = actual
    run(harness)
    assert sum(count for count, _ in calls) == (370 if failure == 'cancel' else 250) * 16000
    raw = json.loads((audio.parent / 'native_transcript_raw.json').read_text())
    assert len(raw['chunks']) == 3
    assert raw['chunks'][1]['start_frame'] == 120 * 16000


@pytest.mark.parametrize('error_type', ['oom', 'context'])
def test_typed_resource_splits_only_failed_interval_and_learns(harness, monkeypatch, error_type):
    from src.services.ingestion.moss_runtime import MossCudaOOMError, MossContextBudgetError
    service, video, audio, manager, *_ = harness
    calls = sized_recording(harness, 370)
    actual = manager.transcribe
    attempts = []
    def infer(data):
        with wave.open(io.BytesIO(data)) as stream:
            size = stream.getnframes()
            first = stream.readframes(1)
        attempts.append((size, first))
        if len(attempts) == 2:
            raise (MossCudaOOMError if error_type == 'oom' else MossContextBudgetError)('resource')
        if len(calls) == 3:
            raise RuntimeError('later crash')
        return actual(data)
    manager.transcribe = infer
    with pytest.raises(RuntimeError, match='later crash'):
        run(harness)
    state = json.loads(service._get_temp_transcript_path(video.id).read_text())
    assert state['ceiling'] == 60 * 16000 and state['frontier'] == 240 * 16000
    assert [size // 16000 for size, _ in attempts[:4]] == [120, 120, 60, 60]
    manager.transcribe = actual
    run(harness)
    assert [count // 16000 for count, _ in calls] == [120, 60, 60, 60, 60, 10]
    raw = json.loads((audio.parent / 'native_transcript_raw.json').read_text())
    assert [c['boundary_reason'] for c in raw['chunks']][1:3] == ['resource_split'] * 2


def test_metadata_failure_tail_recovered(harness, monkeypatch):
    service, video, _, manager, *_ = harness
    calls = sized_recording(harness)
    publish = service._moss_atomic_json
    def fail(path, data):
        if data.get('chunk_count') == 2:
            raise OSError('metadata')
        return publish(path, data)
    monkeypatch.setattr(service, '_moss_atomic_json', fail)
    with pytest.raises(OSError, match='metadata'):
        run(harness)
    assert json.loads(service._get_temp_transcript_path(video.id).read_text())['chunk_count'] == 1
    monkeypatch.setattr(service, '_moss_atomic_json', publish)
    run(harness)
    assert [count // 16000 for count, _ in calls] == [120, 120, 120, 10]


@pytest.mark.parametrize('damage', ['missing', 'corrupt', 'gap', 'model', 'frontier', 'source', 'mixed'])
def test_invalid_committed_partial_never_truncated_or_skipped(harness, damage):
    service, video, audio, manager, *_ = harness
    calls = sized_recording(harness)
    actual = manager.transcribe
    def fail(data):
        if calls:
            raise RuntimeError('stop')
        return actual(data)
    manager.transcribe = fail
    with pytest.raises(RuntimeError):
        run(harness)
    meta = service._get_temp_transcript_path(video.id)
    records = service._get_temp_transcript_jsonl_path(video.id)
    state = json.loads(meta.read_text())
    chunk = json.loads(records.read_text())
    if damage == 'missing':
        records.unlink()
    elif damage == 'corrupt':
        records.write_bytes(b'bad\n')
    elif damage == 'gap':
        chunk['start_frame'] = 1
    elif damage == 'model':
        chunk['runtime']['model_revision'] = 'wrong'
    elif damage == 'frontier':
        state['frontier'] += 1
    elif damage == 'source':
        state['moss_source']['pcm_sha256'] = '0' * 64
    else:
        state['engine'] = 'phonon'
    if damage in ('gap', 'model'):
        records.write_text(json.dumps(chunk) + '\n')
    meta.write_text(json.dumps(state))
    before = records.read_bytes() if records.exists() else None
    manager.transcribe = actual
    with pytest.raises(MossValidationError):
        run(harness)
    assert len(calls) == 1
    assert (records.read_bytes() if records.exists() else None) == before


def test_vad_defaults_local_to_absolute_and_fallback(harness, monkeypatch):
    import faster_whisper.vad
    service, _, audio, *_ = harness
    seen = []
    def hints(samples):
        seen.append(len(samples))
        return [dict(start=0, end=len(samples))]
    monkeypatch.setattr(faster_whisper.vad, 'get_speech_timestamps', hints)
    from src.services.ingestion.pipeline_transcribe import TranscribePhaseMixin
    assert TranscribePhaseMixin._moss_speech_hints(b'\0\0' * 7, 11) == [dict(start=11, end=18)]
    calls = sized_recording(harness)
    monkeypatch.setattr(service, '_moss_speech_hints', lambda *_: (_ for _ in ()).throw(ImportError('optional')))
    run(harness)
    raw = json.loads((audio.parent / 'native_transcript_raw.json').read_text())
    assert len(calls) == 3 and raw['chunks'][0]['boundary_reason'] == 'vad_unavailable'
    assert raw['chunks'][0]['vad_fallback'] == 'ImportError' and seen == [7]


def test_vad_leading_256_frames_never_becomes_separate_request(harness, monkeypatch):
    import hashlib
    import faster_whisper.vad
    from src.services.ingestion.pipeline_transcribe import TranscribePhaseMixin

    service, _, audio, manager, _, native = harness
    sized_recording(harness)
    monkeypatch.setattr(faster_whisper.vad, 'get_speech_timestamps',
                        lambda samples: [dict(start=256, end=len(samples))])
    monkeypatch.setattr(service, '_moss_speech_hints', TranscribePhaseMixin._moss_speech_hints)
    received, counts = hashlib.sha256(), []
    with wave.open(str(audio)) as stream:
        original = hashlib.sha256(stream.readframes(stream.getnframes())).hexdigest()
    def infer(data):
        with wave.open(io.BytesIO(data)) as stream:
            count = stream.getnframes()
            received.update(stream.readframes(count))
        counts.append(count)
        return dict(result=parse_generation(' ', count, 16000, native['completion']),
                    runtime=PROVENANCE.copy())
    manager.transcribe = infer
    run(harness)
    assert counts == [120 * 16000, 120 * 16000, 10 * 16000]
    assert received.hexdigest() == original

def test_local_eof_corrections_and_exact_offsets(harness):
    from fractions import Fraction
    service, _, audio, manager, _, native = harness
    sized_recording(harness, 240)
    with wave.open(str(audio), 'rb') as stream:
        pcm = stream.readframes(stream.getnframes())
    with wave.open(str(audio), 'wb') as stream:
        stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        stream.writeframes(pcm + b'\0\0')
    def infer(data):
        with wave.open(io.BytesIO(data)) as stream:
            count = stream.getnframes()
        text = '[0.10][S0]outer[120.0100][2][S1]inner[3]' if count > 1 else ' '
        return dict(result=parse_generation(text, count, 16000, native['completion']), runtime=PROVENANCE.copy())
    manager.transcribe = infer
    segments, duration = run(harness)
    assert duration == float(Fraction(3840001, 16000))
    assert [(s.start, s.end) for s in segments] == [(0.1, 120), (2, 3), (120.1, 240), (122, 123)]
    assert segments[0].speaker != segments[2].speaker
    assert segments[0].local_speaker == segments[2].local_speaker == 'S0'
    raw = json.loads((audio.parent / 'native_transcript_raw.json').read_text())
    assert raw['chunks'][0]['result'] == raw['chunks'][1]['result']
    assert raw['chunks'][1]['result']['corrections'][0]['raw_end'] == '120.0100'
    assert raw['chunks'][1]['result']['corrections'][0]['normalized_end'] == '120'


def test_final_atomic_failure_retains_old_raw_and_complete_partial(harness, monkeypatch):
    service, video, audio, _, calls, _ = harness
    run(harness)
    raw = audio.parent / 'native_transcript_raw.json'
    original = raw.read_bytes()
    publish = service._moss_atomic_json
    def fail(path, data):
        if path == raw:
            raise OSError('final publish')
        return publish(path, data)
    monkeypatch.setattr(service, '_moss_atomic_json', fail)
    with pytest.raises(OSError, match='final publish'):
        run(harness, force=True)
    assert raw.read_bytes() == original
    assert json.loads(service._get_temp_transcript_path(video.id).read_text())['frontier'] == 144001
    monkeypatch.setattr(service, '_moss_atomic_json', publish)
    run(harness)
    assert len(calls) == 2  # Fully committed partial publishes without inference.


def test_one_frame_resource_failure_terminates_without_invented_retry(harness):
    from src.services.ingestion.moss_runtime import MossCudaOOMError
    _, _, audio, manager, *_ = harness
    with wave.open(str(audio), 'wb') as stream:
        stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        stream.writeframes(b'\0\0')
    attempts = []
    def infer(_):
        attempts.append(True)
        raise MossCudaOOMError('actual typed failure')
    manager.transcribe = infer
    with pytest.raises(ValueError, match='cannot be split'):
        run(harness)
    assert attempts == [True]

@pytest.mark.parametrize('checkpoint_run', [legacy_run, run])
def test_identity_unresolved_both_formats(identity_harness, monkeypatch, checkpoint_run):
    from sqlmodel import Session, select
    from src.db.database import SpeakerEmbedding, TranscriptSegment
    from src.services.ingestion import runtime
    harness, crops = identity_harness
    service, video, audio, manager, _, native = harness
    result = parse_generation('[0][S01]one[4][0][S08]two[4]', native['sample_count'], 16000, native['completion'])
    manager.transcribe = lambda _: dict(result=result, runtime=PROVENANCE.copy())
    segments, _ = checkpoint_run(harness)
    raw = audio.parent / 'native_transcript_raw.json'
    original = raw.read_bytes()
    service.embedding_model = service.embedding_inference = None
    order = []
    monkeypatch.setattr(service, '_release_moss_model', lambda *_, **__: order.append('exit'))
    def load(_, *, embedding_only):
        assert embedding_only and order == ['exit']
    monkeypatch.setattr(service, '_load_models', load)
    with pytest.raises(RuntimeError, match='voice model is unavailable'):
        service._process_diarize_phase(video, audio, segments)
    assert raw.read_bytes() == original
    service.embedding_model = service.embedding_inference = object()
    for method in ('_moss_speech_hints', '_get_moss_runtime'):
        monkeypatch.setattr(service, method, lambda *_: pytest.fail('must reuse raw'))
    monkeypatch.setattr('subprocess.run', lambda *_, **__: pytest.fail('must not normalize'))
    reused, _, _ = service._load_raw_transcript_checkpoint(video, audio)
    service._process_diarize_phase(video, audio, reused)
    with Session(runtime.engine) as session:
        rows = session.exec(select(TranscriptSegment)).all()
        assert len(rows) == 2 and rows[0].speaker_id != rows[1].speaker_id
        assert all(row.matched_profile_id is None for row in rows)
        assert session.exec(select(SpeakerEmbedding)).all() == []
    assert not crops and raw.read_bytes() == original


def test_chunk_identity_enrollment_swapped_labels_and_retry(identity_harness, monkeypatch):
    import numpy as np
    import pickle
    from sqlmodel import Session, select
    from src.db.database import Channel, Speaker, SpeakerEmbedding, TranscriptSegment, Video
    from src.services.ingestion import runtime
    harness, crops = identity_harness
    service, video, audio, manager, _, native = harness
    sized_recording(harness)
    texts = iter(['[0][S01]host[4][2][S08]nested[3][5][S08]guest[7]',
                  '[0][S01]guest again[4][5][S08]host again[7]', '[0][S08]host EOF[10.0100]'])
    def infer(data):
        with wave.open(io.BytesIO(data)) as stream:
            count = stream.getnframes()
        return dict(result=parse_generation(next(texts), count, 16000, native['completion']), runtime=PROVENANCE.copy())
    manager.transcribe = infer
    segments, _ = run(harness)
    assert [s.speaker for s in segments] == ['chunk-00000000:S01', 'chunk-00000000:S08',
        'chunk-00000000:S08', 'chunk-00000001:S01', 'chunk-00000001:S08', 'chunk-00000002:S08']
    def embed(_, segment):
        crops.append((segment.start, segment.end))
        return np.eye(16)[1 if segment.start in (5, 120) else 0]
    monkeypatch.setattr(service, 'get_speaker_embedding', embed)
    with Session(runtime.engine) as session:
        session.add(Channel(id=2, name='Other', url='other'))
        other = Speaker(channel_id=2, name='Wrong channel', embedding_blob=pickle.dumps(np.eye(16)[0]))
        session.add(other)
        session.commit()
        session.add(SpeakerEmbedding(speaker_id=other.id, embedding_blob=other.embedding_blob))
        session.commit()
    raw = audio.parent / 'native_transcript_raw.json'
    original = raw.read_bytes()
    for method in ('_moss_speech_hints', '_get_moss_runtime'):
        monkeypatch.setattr(service, method, lambda *_: pytest.fail('must reuse chunks'))
    monkeypatch.setattr('subprocess.run', lambda *_, **__: pytest.fail('must not normalize'))
    expected = [(0, 4, 'host'), (2, 3, 'nested'), (5, 7, 'guest'), (120, 124, 'guest again'),
                (125, 127, 'host again'), (240, 250, 'host EOF')]
    for recording in range(2):
        if recording:
            with Session(runtime.engine) as session:
                video = Video(id=14, youtube_id='fixture14', channel_id=1, title='second')
                session.add(video)
                session.commit()
                session.refresh(video)
                session.expunge(video)
            data = json.loads(original)
            for chunk in data['chunks']:
                result = chunk['result']
                text = result['raw_text'].replace('S01', 'TEMP').replace('S08', 'S01').replace('TEMP', 'S08')
                chunk['result'] = parse_generation(text, result['sample_count'], 16000, result['completion'])
            (audio.parent / 'second_transcript_raw.json').write_text(json.dumps(data))
        reused, _, engine = service._load_raw_transcript_checkpoint(video, audio)
        assert engine == 'moss'
        service._get_temp_diarization_path(video.id).write_text('stale RTTM ignored')
        service._process_diarize_phase(video, audio, reused)
        with Session(runtime.engine) as session:
            rows = session.exec(select(TranscriptSegment).where(TranscriptSegment.video_id == video.id)
                                .order_by(TranscriptSegment.id)).all()
            assert [(r.start_time, r.end_time, r.text) for r in rows] == expected
            assert all(r.words is None for r in rows)
            assert rows[0].speaker_id == rows[4].speaker_id == rows[5].speaker_id
            assert rows[1].speaker_id == rows[2].speaker_id == rows[3].speaker_id
            assert rows[0].speaker_id != rows[1].speaker_id
            assert len(session.exec(select(Speaker).where(Speaker.channel_id == 1)).all()) == 2
            if not recording:
                session.get(Speaker, rows[0].speaker_id).name = 'Host'
                session.get(Speaker, rows[1].speaker_id).name = 'Guest'
                session.commit()
            else:
                assert [session.get(Speaker, r.speaker_id).name for r in rows] == ['Host', 'Guest', 'Guest', 'Guest', 'Host', 'Host']
            profiles = session.exec(select(SpeakerEmbedding).where(SpeakerEmbedding.source_video_id == video.id)).all()
            assert profiles and all(not (p.sample_start_time < 3 and p.sample_end_time > 2) for p in profiles)
        assert (audio.parent / f'{video.title}.srt').exists()
        assert not service._get_temp_diarization_path(video.id).exists()
    assert set(crops) == {(0, 2), (5, 7), (120, 124), (125, 127), (240, 250)}
    assert raw.read_bytes() == original
    assert json.loads(original)['chunks'][-1]['result']['corrections'][0]['raw_end'] == '10.0100'
