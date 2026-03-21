from sqlmodel import SQLModel, create_engine, Field, Relationship
from sqlalchemy import inspect, text, event
from typing import Optional, List, Any
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from collections import deque
import threading
import time as perf_time
import pickle
import os

from .embedded_postgres import ensure_embedded_postgres, build_embedded_postgres_url

# Get the directory where this file is located (backend/src/db)
DB_DIR = Path(__file__).parent
# Go up two levels to backend/data
DATA_DIR = DB_DIR.parent.parent / "data"
sqlite_file_name = DATA_DIR / "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"


def _resolve_database_provider() -> str:
    provider = (os.getenv("DB_PROVIDER") or "postgres").strip().lower()
    if provider in {"postgresql", "postgres"}:
        return "postgres"
    if provider == "sqlite":
        return "sqlite"
    return "postgres"


def _resolve_database_url(provider: str) -> str:
    explicit = (os.getenv("DATABASE_URL") or "").strip()
    if explicit:
        return explicit

    if provider == "postgres":
        ensure_embedded_postgres()
        return build_embedded_postgres_url()

    return sqlite_url


DB_PROVIDER = _resolve_database_provider()
DATABASE_URL = _resolve_database_url(DB_PROVIDER)
IS_POSTGRES = DATABASE_URL.startswith("postgresql")
if IS_POSTGRES:
    parsed = urlparse(DATABASE_URL)
    marker_db_name = (parsed.path or "/chatalogue").lstrip("/") or "chatalogue"
    marker_safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in marker_db_name)
    MIGRATION_MARKER = DATA_DIR / f".sqlite_to_postgres_migrated_{marker_safe}"
else:
    MIGRATION_MARKER = DATA_DIR / ".sqlite_to_postgres_migrated_sqlite"

# Only echo SQL in verbose mode
verbose = os.getenv("VERBOSE_LOGGING", "false").lower() == "true"
engine_kwargs: dict[str, Any] = {
    "echo": verbose,
    "pool_pre_ping": True,
}
if not IS_POSTGRES:
    engine_kwargs["connect_args"] = {"timeout": 30, "check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_kwargs)


class _QueryMetrics:
    def __init__(self, slow_threshold_ms: float = 250.0, recent_window: int = 500):
        self._lock = threading.Lock()
        self.slow_threshold_ms = slow_threshold_ms
        self.recent_window = max(50, int(recent_window))
        self.started_at = datetime.now()
        self.total_queries = 0
        self.total_time_ms = 0.0
        self.slow_queries = 0
        self.error_queries = 0
        self.recent_ms: deque[float] = deque(maxlen=self.recent_window)

    def record_query(self, duration_ms: float) -> None:
        with self._lock:
            self.total_queries += 1
            self.total_time_ms += duration_ms
            if duration_ms >= self.slow_threshold_ms:
                self.slow_queries += 1
            self.recent_ms.append(duration_ms)

    def record_error(self) -> None:
        with self._lock:
            self.error_queries += 1

    @staticmethod
    def _percentile(sorted_values: list[float], p: float) -> float:
        if not sorted_values:
            return 0.0
        if len(sorted_values) == 1:
            return sorted_values[0]
        idx = (len(sorted_values) - 1) * p
        lo = int(idx)
        hi = min(lo + 1, len(sorted_values) - 1)
        frac = idx - lo
        return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            recent = list(self.recent_ms)
            total_queries = self.total_queries
            total_time_ms = self.total_time_ms
            slow_queries = self.slow_queries
            error_queries = self.error_queries
            started_at = self.started_at
            slow_threshold_ms = self.slow_threshold_ms

        recent_sorted = sorted(recent)
        avg_ms = (total_time_ms / total_queries) if total_queries > 0 else 0.0
        recent_avg_ms = (sum(recent) / len(recent)) if recent else 0.0
        return {
            "started_at": started_at.isoformat(),
            "total_queries": total_queries,
            "total_time_ms": round(total_time_ms, 2),
            "avg_ms": round(avg_ms, 2),
            "slow_queries": slow_queries,
            "slow_threshold_ms": round(slow_threshold_ms, 2),
            "error_queries": error_queries,
            "recent_count": len(recent),
            "recent_avg_ms": round(recent_avg_ms, 2),
            "recent_p95_ms": round(self._percentile(recent_sorted, 0.95), 2),
            "recent_p99_ms": round(self._percentile(recent_sorted, 0.99), 2),
        }


_query_metrics = _QueryMetrics(
    slow_threshold_ms=float(os.getenv("DB_SLOW_QUERY_MS", "250")),
    recent_window=int(os.getenv("DB_QUERY_RECENT_WINDOW", "500")),
)


@event.listens_for(engine, "before_cursor_execute")
def _before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    context._chatalogue_query_started = perf_time.perf_counter()


@event.listens_for(engine, "after_cursor_execute")
def _after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    started = getattr(context, "_chatalogue_query_started", None)
    if started is None:
        return
    duration_ms = (perf_time.perf_counter() - started) * 1000.0
    _query_metrics.record_query(duration_ms)


@event.listens_for(engine, "handle_error")
def _handle_error(context):
    _query_metrics.record_error()


def _redact_database_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        if not parsed.scheme.startswith("postgresql"):
            return url
        username = parsed.username or ""
        host = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
        auth = f"{username}:***@" if username else ""
        safe_netloc = f"{auth}{host}{port}"
        return parsed._replace(netloc=safe_netloc).geturl()
    except Exception:
        return url


def get_db_metrics_snapshot() -> dict[str, Any]:
    pool = engine.pool
    pool_stats: dict[str, Any] = {}
    for key, method_name in [
        ("size", "size"),
        ("checked_out", "checkedout"),
        ("checked_in", "checkedin"),
        ("overflow", "overflow"),
    ]:
        method = getattr(pool, method_name, None)
        if callable(method):
            try:
                pool_stats[key] = int(method())
            except Exception:
                pool_stats[key] = None
        else:
            pool_stats[key] = None

    return {
        "provider": DB_PROVIDER,
        "database_url": _redact_database_url(DATABASE_URL),
        "is_postgres": IS_POSTGRES,
        "pool": pool_stats,
        "query_metrics": _query_metrics.snapshot(),
    }

class Channel(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    url: str = Field(index=True, unique=True)
    name: str
    source_type: str = Field(default="youtube")
    icon_url: Optional[str] = None
    header_image_url: Optional[str] = None
    last_updated: Optional[datetime] = None
    status: str = Field(default="active")
    actively_monitored: bool = Field(default=False)
    sync_status_detail: Optional[str] = None
    sync_progress: int = Field(default=0)
    sync_total_items: int = Field(default=0)
    sync_completed_items: int = Field(default=0)
    
    videos: List["Video"] = Relationship(back_populates="channel")
    speakers: List["Speaker"] = Relationship(back_populates="channel")

class Speaker(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    channel_id: int = Field(foreign_key="channel.id")
    name: str = Field(default="Unknown Speaker")
    embedding_blob: bytes # Serialized numpy array
    thumbnail_path: Optional[str] = None
    is_extra: bool = Field(default=False)  # True = minor/background speaker
    created_at: datetime = Field(default_factory=datetime.now)

    channel: Optional[Channel] = Relationship(back_populates="speakers")
    segments: List["TranscriptSegment"] = Relationship(back_populates="speaker")
    embeddings: List["SpeakerEmbedding"] = Relationship(back_populates="speaker")

    @property
    def embedding(self):
        return pickle.loads(self.embedding_blob)

    @embedding.setter
    def embedding(self, value):
        self.embedding_blob = pickle.dumps(value)

class SpeakerEmbedding(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    speaker_id: int = Field(foreign_key="speaker.id")
    embedding_blob: bytes  # Serialized numpy array
    source_video_id: Optional[int] = Field(default=None, foreign_key="video.id")
    sample_start_time: Optional[float] = None
    sample_end_time: Optional[float] = None
    sample_text: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    speaker: Optional["Speaker"] = Relationship(back_populates="embeddings")

    @property
    def embedding(self):
        return pickle.loads(self.embedding_blob)

    @embedding.setter
    def embedding(self, value):
        self.embedding_blob = pickle.dumps(value)

class Video(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    youtube_id: str = Field(index=True, unique=True)
    channel_id: Optional[int] = Field(default=None, foreign_key="channel.id")
    title: str
    media_source_type: str = Field(default="youtube")
    source_url: Optional[str] = None
    media_kind: Optional[str] = None
    manual_media_path: Optional[str] = None
    published_at: Optional[datetime] = None  # May be null if not available from flat extraction
    description: Optional[str] = None  # Video description for search
    thumbnail_url: Optional[str] = None
    duration: Optional[int] = None
    processed: bool = Field(default=False)
    muted: bool = Field(default=False)  # If true, skip transcription
    access_restricted: bool = Field(default=False)
    access_restriction_reason: Optional[str] = None
    status: str = Field(default="pending") # pending, downloaded, transcribed, completed, failed
    humor_context_summary: Optional[str] = None
    humor_context_model: Optional[str] = None
    humor_context_generated_at: Optional[datetime] = None
    youtube_ai_summary: Optional[str] = None
    youtube_ai_chapters_json: Optional[str] = None
    youtube_ai_description_text: Optional[str] = None
    youtube_ai_model: Optional[str] = None
    youtube_ai_generated_at: Optional[datetime] = None
    transcript_source: Optional[str] = None
    transcript_language: Optional[str] = None
    transcript_is_placeholder: bool = Field(default=False)

    channel: Optional[Channel] = Relationship(back_populates="videos")
    segments: List["TranscriptSegment"] = Relationship(back_populates="video")
    jobs: List["Job"] = Relationship(back_populates="video")
    clips: List["Clip"] = Relationship(back_populates="video")
    funny_moments: List["FunnyMoment"] = Relationship(back_populates="video")
    description_revisions: List["VideoDescriptionRevision"] = Relationship(back_populates="video")

class JobBase(SQLModel):
    video_id: int = Field(foreign_key="video.id")
    job_type: str
    status: str = Field(default="queued")  # queued, running, paused, completed, failed
    status_detail: Optional[str] = None  # Fine-grained status (e.g. "Converting audio...", "Loading models...")
    payload_json: Optional[str] = None  # Optional job-specific payload (e.g. clip_id/export format/force flags)
    progress: int = Field(default=0)  # 0-100
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

class Job(JobBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    video: Optional[Video] = Relationship(back_populates="jobs")

class TranscriptSegment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    video_id: int = Field(foreign_key="video.id")
    speaker_id: Optional[int] = Field(default=None, foreign_key="speaker.id")
    matched_profile_id: Optional[int] = Field(default=None, foreign_key="speakerembedding.id")
    start_time: float
    end_time: float
    text: str
    words: Optional[str] = None  # JSON string of word list

    video: Optional[Video] = Relationship(back_populates="segments")
    speaker: Optional[Speaker] = Relationship(back_populates="segments")

class TranscriptSegmentRevision(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    segment_id: int = Field(foreign_key="transcriptsegment.id", index=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    old_text: str
    new_text: str
    source: str = Field(default="manual_edit")
    created_at: datetime = Field(default_factory=datetime.now)

class TranscriptSegmentRead(SQLModel):
    id: Optional[int] = None
    video_id: int
    speaker_id: Optional[int] = None
    matched_profile_id: Optional[int] = None
    start_time: float
    end_time: float
    text: str
    words: Optional[str] = None
    speaker: Optional[str] = None

class Clip(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    video_id: int = Field(foreign_key="video.id")
    start_time: float
    end_time: float
    title: str
    aspect_ratio: str = Field(default="source")  # source,16:9,9:16,1:1,4:5
    crop_x: Optional[float] = None  # normalized 0..1
    crop_y: Optional[float] = None
    crop_w: Optional[float] = None
    crop_h: Optional[float] = None
    portrait_split_enabled: bool = Field(default=False)
    portrait_top_crop_x: Optional[float] = None
    portrait_top_crop_y: Optional[float] = None
    portrait_top_crop_w: Optional[float] = None
    portrait_top_crop_h: Optional[float] = None
    portrait_bottom_crop_x: Optional[float] = None
    portrait_bottom_crop_y: Optional[float] = None
    portrait_bottom_crop_w: Optional[float] = None
    portrait_bottom_crop_h: Optional[float] = None
    script_edits_json: Optional[str] = None  # Serialized text-edit model (e.g. kept_ranges)
    fade_in_sec: float = Field(default=0.0)
    fade_out_sec: float = Field(default=0.0)
    burn_captions: bool = Field(default=False)
    caption_speaker_labels: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.now)

    video: Optional[Video] = Relationship(back_populates="clips")

class ClipExportArtifact(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    clip_id: int = Field(foreign_key="clip.id", index=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    artifact_type: str = Field(default="video")  # video | captions
    format: str = Field(default="mp4")  # mp4 | srt | vtt
    file_path: str
    file_name: str
    file_size_bytes: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.now)

class FunnyMoment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    start_time: float
    end_time: float
    score: float = Field(default=0.0)
    source: str = Field(default="heuristic")  # transcript, acoustic, hybrid
    snippet: Optional[str] = None
    humor_summary: Optional[str] = None
    humor_confidence: Optional[str] = None  # low, medium, high
    humor_model: Optional[str] = None
    humor_explained_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)

    video: Optional[Video] = Relationship(back_populates="funny_moments")

class FunnyMomentRead(SQLModel):
    id: Optional[int] = None
    video_id: int
    start_time: float
    end_time: float
    score: float
    source: str
    snippet: Optional[str] = None
    humor_summary: Optional[str] = None
    humor_confidence: Optional[str] = None
    humor_model: Optional[str] = None
    humor_explained_at: Optional[datetime] = None
    created_at: datetime

class VideoDescriptionRevision(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    description_text: str
    source: str = Field(default="unknown")  # ingest_original | before_ai_publish | before_restore | restored_from_history
    ai_model: Optional[str] = None
    note: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    video: Optional[Video] = Relationship(back_populates="description_revisions")

class VideoDescriptionRevisionRead(SQLModel):
    id: Optional[int] = None
    video_id: int
    description_text: str
    source: str
    ai_model: Optional[str] = None
    note: Optional[str] = None
    created_at: datetime

class TranscriptSegmentRevisionRead(SQLModel):
    id: int
    segment_id: int
    video_id: int
    old_text: str
    new_text: str
    source: str
    created_at: datetime

class ClipExportArtifactRead(SQLModel):
    id: int
    clip_id: int
    video_id: int
    artifact_type: str
    format: str
    file_path: str
    file_name: str
    file_size_bytes: Optional[int] = None
    created_at: datetime

def _column_migrations() -> list[tuple[str, str, str, str]]:
    return [
        ("transcriptsegment", "words", "TEXT", "TEXT"),
        ("transcriptsegment", "matched_profile_id", "INTEGER", "INTEGER"),
        ("clip", "aspect_ratio", "TEXT", "TEXT"),
        ("clip", "crop_x", "REAL", "DOUBLE PRECISION"),
        ("clip", "crop_y", "REAL", "DOUBLE PRECISION"),
        ("clip", "crop_w", "REAL", "DOUBLE PRECISION"),
        ("clip", "crop_h", "REAL", "DOUBLE PRECISION"),
        ("clip", "portrait_split_enabled", "BOOLEAN", "BOOLEAN"),
        ("clip", "portrait_top_crop_x", "REAL", "DOUBLE PRECISION"),
        ("clip", "portrait_top_crop_y", "REAL", "DOUBLE PRECISION"),
        ("clip", "portrait_top_crop_w", "REAL", "DOUBLE PRECISION"),
        ("clip", "portrait_top_crop_h", "REAL", "DOUBLE PRECISION"),
        ("clip", "portrait_bottom_crop_x", "REAL", "DOUBLE PRECISION"),
        ("clip", "portrait_bottom_crop_y", "REAL", "DOUBLE PRECISION"),
        ("clip", "portrait_bottom_crop_w", "REAL", "DOUBLE PRECISION"),
        ("clip", "portrait_bottom_crop_h", "REAL", "DOUBLE PRECISION"),
        ("clip", "script_edits_json", "TEXT", "TEXT"),
        ("clip", "fade_in_sec", "REAL", "DOUBLE PRECISION"),
        ("clip", "fade_out_sec", "REAL", "DOUBLE PRECISION"),
        ("clip", "burn_captions", "BOOLEAN", "BOOLEAN"),
        ("clip", "caption_speaker_labels", "BOOLEAN", "BOOLEAN"),
        ("channel", "icon_url", "TEXT", "TEXT"),
        ("channel", "header_image_url", "TEXT", "TEXT"),
        ("channel", "source_type", "TEXT", "TEXT"),
        ("channel", "actively_monitored", "BOOLEAN", "BOOLEAN"),
        ("channel", "sync_status_detail", "TEXT", "TEXT"),
        ("channel", "sync_progress", "INTEGER", "INTEGER"),
        ("channel", "sync_total_items", "INTEGER", "INTEGER"),
        ("channel", "sync_completed_items", "INTEGER", "INTEGER"),
        ("video", "media_source_type", "TEXT", "TEXT"),
        ("video", "source_url", "TEXT", "TEXT"),
        ("video", "media_kind", "TEXT", "TEXT"),
        ("video", "manual_media_path", "TEXT", "TEXT"),
        ("video", "humor_context_summary", "TEXT", "TEXT"),
        ("video", "humor_context_model", "TEXT", "TEXT"),
        ("video", "humor_context_generated_at", "TEXT", "TIMESTAMP"),
        ("video", "access_restricted", "BOOLEAN", "BOOLEAN"),
        ("video", "access_restriction_reason", "TEXT", "TEXT"),
        ("video", "youtube_ai_summary", "TEXT", "TEXT"),
        ("video", "youtube_ai_chapters_json", "TEXT", "TEXT"),
        ("video", "youtube_ai_description_text", "TEXT", "TEXT"),
        ("video", "youtube_ai_model", "TEXT", "TEXT"),
        ("video", "youtube_ai_generated_at", "TEXT", "TIMESTAMP"),
        ("video", "transcript_source", "TEXT", "TEXT"),
        ("video", "transcript_language", "TEXT", "TEXT"),
        ("video", "transcript_is_placeholder", "BOOLEAN", "BOOLEAN"),
        ("funnymoment", "humor_summary", "TEXT", "TEXT"),
        ("funnymoment", "humor_confidence", "TEXT", "TEXT"),
        ("funnymoment", "humor_model", "TEXT", "TEXT"),
        ("funnymoment", "humor_explained_at", "TEXT", "TIMESTAMP"),
        ("job", "payload_json", "TEXT", "TEXT"),
        ("speakerembedding", "sample_start_time", "REAL", "DOUBLE PRECISION"),
        ("speakerembedding", "sample_end_time", "REAL", "DOUBLE PRECISION"),
        ("speakerembedding", "sample_text", "TEXT", "TEXT"),
    ]


def _ensure_missing_columns(conn: Any) -> None:
    inspector = inspect(conn)
    table_names = set(inspector.get_table_names())
    for table, column, sqlite_type, postgres_type in _column_migrations():
        if table not in table_names:
            continue
        existing = {c["name"] for c in inspector.get_columns(table)}
        if column in existing:
            continue
        col_type = postgres_type if IS_POSTGRES else sqlite_type
        conn.execute(text(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {col_type}'))
        print(f"Migration: added '{column}' column to '{table}' table")


def _seed_speaker_embeddings(conn: Any) -> None:
    conn.execute(
        text(
            """
            INSERT INTO speakerembedding (speaker_id, embedding_blob, created_at)
            SELECT s.id, s.embedding_blob, s.created_at
            FROM speaker s
            WHERE s.embedding_blob IS NOT NULL
              AND s.id NOT IN (SELECT DISTINCT speaker_id FROM speakerembedding)
            """
        )
    )


def _ensure_indexes(conn: Any) -> None:
    index_statements = [
        "CREATE INDEX IF NOT EXISTS idx_speaker_channel_id ON speaker(channel_id)",
        "CREATE INDEX IF NOT EXISTS idx_speakerembedding_speaker_id ON speakerembedding(speaker_id)",
        "CREATE INDEX IF NOT EXISTS idx_video_channel_id ON video(channel_id)",
        "CREATE INDEX IF NOT EXISTS idx_video_channel_published_id ON video(channel_id, published_at DESC, id DESC)",
        "CREATE INDEX IF NOT EXISTS idx_video_channel_processed_muted ON video(channel_id, processed, muted)",
        "CREATE INDEX IF NOT EXISTS idx_video_channel_status ON video(channel_id, status)",
        "CREATE INDEX IF NOT EXISTS idx_job_status_type_created_id ON job(status, job_type, created_at, id)",
        "CREATE INDEX IF NOT EXISTS idx_job_status_started_id ON job(status, started_at, id)",
        "CREATE INDEX IF NOT EXISTS idx_job_video_type_status ON job(video_id, job_type, status)",
        "CREATE INDEX IF NOT EXISTS idx_transcriptsegment_speaker_id ON transcriptsegment(speaker_id)",
        "CREATE INDEX IF NOT EXISTS idx_transcriptsegment_matched_profile_id ON transcriptsegment(matched_profile_id)",
        "CREATE INDEX IF NOT EXISTS idx_transcriptsegment_video_id ON transcriptsegment(video_id)",
        "CREATE INDEX IF NOT EXISTS idx_transcriptsegment_video_start_id ON transcriptsegment(video_id, start_time, id)",
        "CREATE INDEX IF NOT EXISTS idx_transcriptsegment_video_speaker ON transcriptsegment(video_id, speaker_id)",
        "CREATE INDEX IF NOT EXISTS idx_transcriptsegmentrevision_segment_id ON transcriptsegmentrevision(segment_id)",
        "CREATE INDEX IF NOT EXISTS idx_videodescriptionrevision_video_created ON videodescriptionrevision(video_id, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_clipexportartifact_clip_created ON clipexportartifact(clip_id, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_clipexportartifact_video_created ON clipexportartifact(video_id, created_at)",
    ]
    for sql in index_statements:
        conn.execute(text(sql))

    if IS_POSTGRES:
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            conn.execute(text('CREATE INDEX IF NOT EXISTS idx_transcriptsegment_text_trgm ON transcriptsegment USING gin ("text" gin_trgm_ops)'))
            conn.execute(text('CREATE INDEX IF NOT EXISTS idx_transcriptsegment_text_fts ON transcriptsegment USING gin (to_tsvector(\'simple\', "text"))'))
        except Exception as e:
            print(f"Warning: failed to ensure pg_trgm transcript index: {e}")


def _backfill_clip_defaults(conn: Any) -> None:
    conn.execute(text("UPDATE clip SET aspect_ratio='source' WHERE aspect_ratio IS NULL"))
    conn.execute(text("UPDATE clip SET portrait_split_enabled=FALSE WHERE portrait_split_enabled IS NULL"))
    conn.execute(text("UPDATE clip SET fade_in_sec=0 WHERE fade_in_sec IS NULL"))
    conn.execute(text("UPDATE clip SET fade_out_sec=0 WHERE fade_out_sec IS NULL"))
    conn.execute(text("UPDATE clip SET burn_captions=FALSE WHERE burn_captions IS NULL"))
    conn.execute(text("UPDATE clip SET caption_speaker_labels=TRUE WHERE caption_speaker_labels IS NULL"))


def _backfill_channel_defaults(conn: Any) -> None:
    conn.execute(text("UPDATE channel SET source_type='youtube' WHERE source_type IS NULL OR TRIM(source_type) = ''"))
    conn.execute(text("UPDATE channel SET actively_monitored=FALSE WHERE actively_monitored IS NULL"))
    conn.execute(text("UPDATE channel SET sync_progress=0 WHERE sync_progress IS NULL"))
    conn.execute(text("UPDATE channel SET sync_total_items=0 WHERE sync_total_items IS NULL"))
    conn.execute(text("UPDATE channel SET sync_completed_items=0 WHERE sync_completed_items IS NULL"))


def _backfill_video_defaults(conn: Any) -> None:
    conn.execute(text("UPDATE video SET media_source_type='youtube' WHERE media_source_type IS NULL OR TRIM(media_source_type) = ''"))
    conn.execute(text("UPDATE video SET transcript_is_placeholder=FALSE WHERE transcript_is_placeholder IS NULL"))
    conn.execute(text("UPDATE video SET access_restricted=FALSE WHERE access_restricted IS NULL"))


def _reset_postgres_sequences(conn: Any) -> None:
    if not IS_POSTGRES:
        return
    tables = [
        "channel",
        "speaker",
        "speakerembedding",
        "video",
        "job",
        "transcriptsegment",
        "transcriptsegmentrevision",
        "clip",
        "clipexportartifact",
        "funnymoment",
        "videodescriptionrevision",
    ]
    for table in tables:
        conn.execute(
            text(
                f"""
                SELECT setval(
                    pg_get_serial_sequence('"{table}"', 'id'),
                    COALESCE((SELECT MAX(id) FROM "{table}"), 1),
                    (SELECT COUNT(*) > 0 FROM "{table}")
                )
                """
            )
        )


def _migrate_sqlite_to_postgres_if_needed() -> None:
    if not IS_POSTGRES:
        return

    migrate_flag = (os.getenv("DB_MIGRATE_SQLITE_ON_START") or "true").strip().lower() in {"1", "true", "yes", "on"}
    if not migrate_flag or not sqlite_file_name.exists() or MIGRATION_MARKER.exists():
        return

    import sqlite3

    table_order = [
        "channel",
        "video",
        "speaker",
        "speakerembedding",
        "job",
        "transcriptsegment",
        "transcriptsegmentrevision",
        "clip",
        "clipexportartifact",
        "funnymoment",
        "videodescriptionrevision",
    ]

    with engine.begin() as pg_conn:
        existing_channels = pg_conn.execute(text("SELECT COUNT(*) FROM channel")).scalar_one()
        if existing_channels > 0:
            MIGRATION_MARKER.write_text("postgres already had data", encoding="utf-8")
            return

        inspector = inspect(pg_conn)
        pg_tables = set(inspector.get_table_names())

        src_conn = sqlite3.connect(str(sqlite_file_name))
        src_conn.row_factory = sqlite3.Row
        try:
            src_cursor = src_conn.cursor()
            for table in table_order:
                if table not in pg_tables:
                    continue
                src_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
                if not src_cursor.fetchone():
                    continue

                select_sql = f'SELECT * FROM "{table}"'
                if table == "transcriptsegmentrevision":
                    select_sql = (
                        'SELECT r.* FROM "transcriptsegmentrevision" r '
                        'INNER JOIN "transcriptsegment" s ON s.id = r.segment_id'
                    )
                src_cursor.execute(select_sql)
                desc = src_cursor.description or []
                source_cols = [d[0] for d in desc]
                if not source_cols:
                    continue

                target_cols = {c["name"] for c in inspector.get_columns(table)}
                insert_cols = [c for c in source_cols if c in target_cols]
                if not insert_cols:
                    continue
                bool_cols = {
                    c["name"]
                    for c in inspector.get_columns(table)
                    if c["name"] in insert_cols and c["type"].__class__.__name__.lower() in {"boolean", "bool"}
                }

                col_sql = ", ".join(f'"{c}"' for c in insert_cols)
                val_sql = ", ".join(f":{c}" for c in insert_cols)
                insert_sql = text(f'INSERT INTO "{table}" ({col_sql}) VALUES ({val_sql}) ON CONFLICT DO NOTHING')

                while True:
                    rows = src_cursor.fetchmany(500)
                    if not rows:
                        break
                    payload = []
                    for row in rows:
                        item = {}
                        for c in insert_cols:
                            value = row[c]
                            if c in bool_cols and value is not None:
                                value = bool(value)
                            item[c] = value
                        payload.append(item)
                    pg_conn.execute(insert_sql, payload)

            _reset_postgres_sequences(pg_conn)
        finally:
            src_conn.close()

    MIGRATION_MARKER.write_text(datetime.now().isoformat(), encoding="utf-8")
    print("Migration: copied SQLite data into PostgreSQL.")


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

    _migrate_sqlite_to_postgres_if_needed()

    with engine.begin() as conn:
        if not IS_POSTGRES:
            try:
                conn.execute(text("PRAGMA journal_mode=WAL"))
                conn.execute(text("PRAGMA busy_timeout=30000"))
                conn.execute(text("PRAGMA synchronous=NORMAL"))
            except Exception:
                pass

        _ensure_missing_columns(conn)
        _seed_speaker_embeddings(conn)
        _ensure_indexes(conn)
        _backfill_clip_defaults(conn)
        _backfill_channel_defaults(conn)
        _backfill_video_defaults(conn)
