"""Pure bounded-frame coverage, VAD hints and finite reduction proofs."""
import pytest

from src.services.ingestion.moss_chunks import (
    Chunk, INITIAL_FRAME_CEILING, MAX_FRAME_CEILING, next_chunk, split_chunk,
)


@pytest.mark.parametrize("total", [1, 16001, 120 * 16000, 5401 * 16000 + 1])
@pytest.mark.parametrize("speech", [None, [], [{"start": 3, "end": 9}]])
def test_exact_contiguous_coverage(total, speech):
    if speech and total < 9:
        speech = []
    cursor, chunks = 0, []
    while chunk := next_chunk(total, cursor, speech=speech):
        assert chunk.start == cursor
        assert 0 < chunk.end - chunk.start <= INITIAL_FRAME_CEILING
        chunks.append(chunk)
        cursor = chunk.end
    assert cursor == total
    assert sum(c.end - c.start for c in chunks) == total
    assert chunks[-1].end == total  # Includes final fractional second/sample.


def test_latest_gap_and_hard_seam_preserve_silence():
    hints = [{"start": 0, "end": 3}, {"start": 5, "end": 7},
             {"start": 9, "end": 20}]
    assert next_chunk(30, ceiling=10, speech=hints) == Chunk(0, 9, "vad_gap")
    assert next_chunk(30, 9, 10, hints) == Chunk(9, 19, "hard_cut")
    assert next_chunk(30, 19, 10, hints) == Chunk(19, 29, "vad_gap")
    assert next_chunk(30, 29, 10, hints) == Chunk(29, 30, "eof")
    assert next_chunk(30, ceiling=10) == Chunk(0, 10, "vad_unavailable")
    assert next_chunk(30, ceiling=10, speech=[]) == Chunk(0, 10, "vad_gap")


def test_overlapping_hints_do_not_create_false_gap():
    assert next_chunk(30, ceiling=10, speech=[
        {"start": 0, "end": 9}, {"start": 4, "end": 20},
    ]) == Chunk(0, 10, "hard_cut")


@pytest.mark.parametrize("cursor", [0, INITIAL_FRAME_CEILING])
def test_leading_silence_belongs_with_following_speech(cursor):
    limit = cursor + INITIAL_FRAME_CEILING
    hints = [{"start": cursor + 256, "end": limit}]
    assert next_chunk(limit + 1, cursor, speech=hints) == Chunk(cursor, limit, "hard_cut")


def test_leading_gap_does_not_hide_later_internal_or_trailing_gap():
    hints = [{"start": 2, "end": 4}, {"start": 6, "end": 12}]
    assert next_chunk(30, ceiling=10, speech=hints) == Chunk(0, 6, "vad_gap")
    assert next_chunk(30, ceiling=15, speech=hints) == Chunk(0, 15, "vad_gap")


@pytest.mark.parametrize("args", [(-1,), (True,), (1.1,), (2, 3), (2, -1),
                                 (2, 0, 0), (2, 0, MAX_FRAME_CEILING + 1)])
def test_invalid_dimensions(args):
    with pytest.raises(ValueError):
        next_chunk(*args)


@pytest.mark.parametrize("hints", [[{"start": 3, "end": 3}],
                                   [{"start": 0, "end": 31}],
                                   [{"start": 4, "end": 8}, {"start": 2, "end": 3}]])
def test_invalid_hints(hints):
    with pytest.raises(ValueError):
        next_chunk(30, speech=hints)


def test_empty_and_completed_input():
    assert next_chunk(0) is None
    assert next_chunk(3, 3) is None


def test_finite_split_tree_preserves_every_frame():
    pending, leaves = [Chunk(17, 118, "hard_cut")], []
    while pending:
        parent = pending.pop(0)
        if parent.end - parent.start == 1:
            with pytest.raises(ValueError, match="cannot be split"):
                split_chunk(parent)
            leaves.append(parent)
        else:
            first, second = split_chunk(parent)
            assert first.start == parent.start
            assert first.end == second.start
            assert second.end == parent.end
            assert all(0 < c.end - c.start < parent.end - parent.start
                       for c in (first, second))
            pending[0:0] = [first, second]
    assert [(c.start, c.end) for c in leaves] == [(i, i + 1) for i in range(17, 118)]
