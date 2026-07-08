"""Ingestion service package.

The public surface is ``IngestionService`` (instantiated once by main.py's
lifespan and reached elsewhere via ``deps.get_ingestion_service()``). The
implementation is being split from the former single-module ``ingestion.py``
into domain modules — see docs/ingestion-service-split.md.

Tests that need to redirect DB access patch ``runtime.engine`` /
``runtime.create_db_and_tables`` (see runtime.py).
"""
from .exceptions import (  # noqa: F401
    JobCancelledException,
    JobDeferredException,
    JobNoticeException,
    JobPausedException,
)
from .runtime import (  # noqa: F401
    AUDIO_DIR,
    BACKEND_DIR,
    DATA_DIR,
    EXPORT_DIR,
    HEARTBEAT_FILE,
    MANUAL_MEDIA_DIR,
    PYTHON_TEMP_DIR,
    RUNTIME_DIR,
    TEMP_DIR,
    configure_python_temp_dir,
    ensure_dirs,
    temporary_disabled_blackhole_proxies,
)
from .service import (  # noqa: F401
    IngestionService,
    TransformersWhisperCompatModel,
)
