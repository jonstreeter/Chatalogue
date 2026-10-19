"""MOSS whole-input checkpoint regressions, using temporary files and SQLite."""
import copy
import io
import json
import wave
from types import SimpleNamespace

import pytest

from src.test_phonon import isolated_service  # noqa: F401
from src.services.ingestion.moss import MossValidationError, parse_generation


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
        return dict(result=copy.deepcopy(native), runtime=dict(model='MOSS', model_revision='pinned', source_revision='pinned'))
    manager = SimpleNamespace(transcribe=transcribe)
    monkeypatch.setattr(service, '_get_moss_runtime', lambda: manager)
    monkeypatch.setattr(service, '_resolve_transcription_route', lambda *_: dict(engine='moss', requested_engine='moss'))
    for method in ('_check_job_not_paused', '_record_job_stage_start', '_update_job_status_detail',
                   '_update_job_progress', '_upsert_job_payload_fields'):
        monkeypatch.setattr(service, method, lambda *_: None)
    def normalize(command, **_):
        from pathlib import Path
        Path(command[-1]).write_bytes(audio.read_bytes())
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr('subprocess.run', normalize)
    return service, video, audio, manager, calls, native


def run(harness, force=False):
    service, video, audio, *_ = harness
    return service._process_moss_phase(video, audio, None, dict(requested_engine='moss'), force)


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
    assert reused[0].speaker == 'S0' and calls == [(144001, 16000)]
    assert json.loads(original)['moss_result'] == native
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
    assert sorted(p.name for p in raw.parent.iterdir()) == ['audio.wav', raw.name]


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
    manager.transcribe = lambda _: dict(result=bad, runtime={})
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
    checks = []
    def pause(*_):
        checks.append(1)
        if len(checks) == 3:
            raise RuntimeError('paused after inference')
    monkeypatch.setattr(service, '_check_job_not_paused', pause)
    with pytest.raises(RuntimeError, match='paused'):
        run(harness)
    assert not (audio.parent / 'native_transcript_raw.json').exists()
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


def test_limit_before_inference(harness, monkeypatch):
    fake = SimpleNamespace(getnframes=lambda: 5400 * 16000 + 1, getframerate=lambda: 16000,
                           getnchannels=lambda: 1, getsampwidth=lambda: 2)
    class Reader:
        def __enter__(self):
            return fake
        def __exit__(self, *_):
            pass
    monkeypatch.setattr(wave, 'open', lambda *_: Reader())
    with pytest.raises(MossValidationError, match='90 minutes'):
        run(harness)
    assert harness[4] == []


def test_corrupt_cache_rejected(harness):
    service, video, audio, _, calls, _ = harness
    run(harness)
    raw = audio.parent / 'native_transcript_raw.json'
    data = json.loads(raw.read_text())
    data['moss_result']['segments'][0]['end'] = '10'
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
        return dict(result=silence, runtime={})
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
    manager.transcribe = lambda _: dict(result=native, runtime={})
    segments, _ = run(harness)
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
    manager.transcribe = lambda _: dict(result=native, runtime={})
    segments, _ = run(harness)
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
    segments, _ = run(harness)
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
