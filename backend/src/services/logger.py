"""
Logging utility for Chatalogue backend.

Provides clean mode (essential output only) vs verbose mode (detailed debug output).
Control via VERBOSE_LOGGING environment variable.
"""
import os


def log(message: str, flush: bool = True):
    """Always print - essential task progress information.
    
    Use for:
    - Job start/completion
    - Stage transitions (downloading, transcribing, diarizing)
    - Errors and warnings
    """
    try:
        print(message, flush=flush)
    except (OSError, AttributeError, IOError):
        pass # Silently fail if stdout is unavailable


def log_verbose(message: str, flush: bool = False):
    """Print only if VERBOSE_LOGGING is enabled.
    
    Use for:
    - Individual transcript segments
    - Speaker analysis details
    - File path information
    - Debug/diagnostic output
    """
    if os.getenv("VERBOSE_LOGGING", "false").lower() == "true":
        try:
            print(message, flush=flush)
        except (OSError, AttributeError, IOError):
            pass # Silently fail



def is_verbose() -> bool:
    """Check if verbose logging is enabled."""
    return os.getenv("VERBOSE_LOGGING", "false").lower() == "true"
