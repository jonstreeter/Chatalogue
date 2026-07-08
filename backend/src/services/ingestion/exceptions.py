"""Job-control exceptions shared across the ingestion service modules."""


class JobPausedException(Exception):
    """Raised when a job is paused by the user during processing."""
    pass


class JobCancelledException(Exception):
    """Raised when a job is cancelled by the user during processing."""
    pass


class JobDeferredException(Exception):
    """Raised when a queued job should be deferred and retried later."""
    pass


class JobNoticeException(Exception):
    """Raised when a job should surface a user-facing notice instead of a hard error."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "notice",
        video_status: str = "pending",
        technical_detail: str | None = None,
    ):
        super().__init__(message)
        self.notice_message = str(message or "Notice")
        self.notice_code = str(code or "notice")
        self.video_status = str(video_status or "pending")
        self.technical_detail = str(technical_detail or self.notice_message)
