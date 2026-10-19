"""Sample-only MOSS request planning; no waveform, application or ML imports.

Speech hints use faster-whisper's sample-index dictionaries, in absolute source
coordinates. They suggest cuts only; silence remains in the owned intervals.
"""
from dataclasses import dataclass

SAMPLE_RATE = 16000
INITIAL_FRAME_CEILING = 120 * SAMPLE_RATE  # Longest measured task-8 CUDA input.
MAX_FRAME_CEILING = 5400 * SAMPLE_RATE


@dataclass(frozen=True)
class Chunk:
    start: int
    end: int
    reason: str


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("MOSS frame dimensions must be integers within bounds")


def next_chunk(total_frames, cursor=0, ceiling=INITIAL_FRAME_CEILING, speech=None):
    """Return the next contiguous interval, or None at EOF (including empty input).

    None hints mean unavailable; [] means VAD found no speech. Hints must cover
    the current window's speech, retain installed VAD defaults, and use absolute
    sample offsets. The caller runs VAD on bounded audio in task 10.
    """
    _integer(total_frames)
    _integer(cursor)
    _integer(ceiling, 1)
    if cursor > total_frames or ceiling > MAX_FRAME_CEILING:
        raise ValueError("MOSS frame dimensions exceed request bounds")
    limit = min(total_frames, cursor + ceiling)
    cut, previous = None, cursor
    if speech is not None:
        last_start = -1
        for interval in speech:
            start, end = interval["start"], interval["end"]
            _integer(start)
            _integer(end, 1)
            if start < last_start or end <= start or end > total_frames:
                raise ValueError("Invalid MOSS speech hints")
            last_start = start
            if start > previous:
                candidate = min(start, limit)
                if previous < candidate and cursor < candidate:
                    cut = candidate
            previous = max(previous, end)
        if previous < limit:
            cut = limit
    if cursor == total_frames:
        return None
    if limit == total_frames:
        return Chunk(cursor, limit, "eof")
    if cut is not None:
        return Chunk(cursor, cut, "vad_gap")
    return Chunk(cursor, limit, "vad_unavailable" if speech is None else "hard_cut")


def split_chunk(chunk):
    """Bisect only an uncommitted failed interval; one frame cannot be reduced.

    Every child is positive and strictly smaller, so repeated splitting reaches
    an explicit terminal failure without a guessed retry count. The caller must
    invoke this only for a typed runtime resource failure.
    """
    _integer(chunk.start)
    _integer(chunk.end, 1)
    length = chunk.end - chunk.start
    if length < 2 or length > MAX_FRAME_CEILING:
        raise ValueError("MOSS interval cannot be split into smaller positive ranges")
    middle = chunk.start + length // 2
    return (Chunk(chunk.start, middle, "resource_split"),
            Chunk(middle, chunk.end, "resource_split"))
