"""Exercise app download paths without network, jobs or production storage."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault('DB_PROVIDER', 'sqlite')
os.environ.setdefault('DATABASE_URL', 'sqlite://')


@pytest.mark.parametrize('fallback', [False, True])
def test_downloads_enable_youtube_challenge_runtime(tmp_path, monkeypatch, fallback):
    from src.services.ingestion import IngestionService
    from src.services.ingestion import media_files

    monkeypatch.delenv('YTDLP_COOKIES_FILE', raising=False)
    monkeypatch.delenv('YTDLP_COOKIES_FROM_BROWSER', raising=False)
    monkeypatch.setattr(media_files, 'AUDIO_DIR', tmp_path / 'legacy')
    captured = []

    class Downloader:
        def __init__(self, options):
            self.options = options
            captured.append(options)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def download(self, urls):
            assert urls == ['https://www.youtube.com/watch?v=test-video']
            path = Path(self.options['outtmpl'].replace('%(ext)s', 'webm'))
            path.write_bytes(b'downloaded fixture')
            for hook in self.options['progress_hooks']:
                hook({'status': 'finished', 'filename': str(path)})

    monkeypatch.setattr(media_files.yt_dlp, 'YoutubeDL', Downloader)
    service = IngestionService.__new__(IngestionService)
    path = tmp_path / 'episode.m4a'
    service.get_audio_path = lambda video: path
    video = SimpleNamespace(id=-1, youtube_id='test-video', source_url=None, media_source_type='youtube')
    if fallback:
        path.write_bytes(b'corrupt fixture')
        probes = iter([SimpleNamespace(returncode=1, stderr=b'corrupt'), SimpleNamespace(returncode=0)])
        monkeypatch.setattr(media_files.subprocess, 'run', lambda *args, **kwargs: next(probes))
        service._get_ffmpeg_cmd = lambda: 'ffmpeg'
        service._update_job_status_detail = lambda *args: None
        result = service._validate_and_retry_audio(video, path)
    else:
        result = service.download_audio(video)
    assert result.is_file()
    assert result.read_bytes() == b'downloaded fixture'
    assert captured[0]['js_runtimes'] == {'deno': {}, 'node': {}}
    assert 'cookiefile' not in captured[0]
    assert 'cookiesfrombrowser' not in captured[0]
