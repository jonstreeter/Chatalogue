"""
Logging utility for Chatalogue backend.

Provides clean mode (essential output only) vs verbose mode (detailed debug output).
Control via VERBOSE_LOGGING environment variable.
"""
import os
import sys


def _safe_print(message: str, flush: bool) -> None:
    try:
        print(message, flush=flush)
        return
    except UnicodeEncodeError:
        # Windows terminals often default to cp1252. Fall back to replacement
        # output so logging never aborts ingest or queue work.
        try:
            text = str(message)
            encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
            safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
            print(safe, flush=flush)
            return
        except Exception:
            pass
    except (OSError, AttributeError, IOError):
        pass


def log(message: str, flush: bool = True):
    """Always print - essential task progress information.
    
    Use for:
    - Job start/completion
    - Stage transitions (downloading, transcribing, diarizing)
    - Errors and warnings
    """
    _safe_print(message, flush)


def log_verbose(message: str, flush: bool = False):
    """Print only if VERBOSE_LOGGING is enabled.
    
    Use for:
    - Individual transcript segments
    - Speaker analysis details
    - File path information
    - Debug/diagnostic output
    """
    if os.getenv("VERBOSE_LOGGING", "false").lower() == "true":
        _safe_print(message, flush)



def is_verbose() -> bool:
    """Check if verbose logging is enabled."""
    return os.getenv("VERBOSE_LOGGING", "false").lower() == "true"
