"""Pure, JSON-safe MOSS transcript boundary; no application or ML imports.

Exact rational strings preserve sample-derived endpoints across JSON retries.
Call validate_worker_result with the manager's actual decoded dimensions.
"""

from fractions import Fraction
import json
import re

POLICY = "moss-eof-v1"
CONTEXT_LIMIT = 131072
EOS_TOKEN_ID = 151645
MAX_INPUT_SECONDS = 5400
_DECIMAL = re.compile(r"[0-9]+(?:\.[0-9]*)?\Z")
_HEADER = re.compile(r"\[([0-9]+(?:\.[0-9]*)?)\]\[(S[0-9]+)\]")
_BRACKET = re.compile(r"\[([^\[\]]*)\]")
_COMPLETION_KEYS = {
    "complete", "reason", "eos_token_id", "terminal_token_id", "input_tokens",
    "output_tokens", "max_new_tokens", "context_limit",
}


class MossValidationError(ValueError):
    """Displayable error that never includes generated transcript contents."""


def _fail():
    raise MossValidationError("Invalid or incomplete MOSS transcript; retry transcription.")


def _positive_int(value):
    return type(value) is int and value > 0


def _completion(data):
    if type(data) is not dict or data.keys() != _COMPLETION_KEYS:
        _fail()
    if data["complete"] is not True or data["reason"] != "eos":
        _fail()
    if any(not _positive_int(data[key]) for key in (
        "input_tokens", "output_tokens", "max_new_tokens", "context_limit",
    )):
        _fail()
    if (type(data["eos_token_id"]) is not int or data["eos_token_id"] != EOS_TOKEN_ID
            or type(data["terminal_token_id"]) is not int
            or data["terminal_token_id"] != data["eos_token_id"]
            or data["context_limit"] != CONTEXT_LIMIT
            or data["output_tokens"] >= data["max_new_tokens"]
            or data["input_tokens"] + data["max_new_tokens"] > CONTEXT_LIMIT):
        _fail()
    return dict(data)


def parse_generation(raw_text, sample_count, sample_rate, completion):
    """Validate the whole native generation atomically, retaining overlap/order.

    Grammar: [decimal][Sdigits]text[decimal], repeated with optional whitespace.
    Non-clock bracket annotations remain literal text. Bracket protocol fragments,
    nonfinite/numeric malformed clocks and nested brackets are never annotations.
    Completion must come from generation metadata, not guessed from text clocks.
    """
    if (type(raw_text) is not str or not _positive_int(sample_count)
            or not _positive_int(sample_rate)):
        _fail()
    metadata = _completion(completion)
    duration = Fraction(sample_count, sample_rate)
    segments, corrections = [], []
    cursor = 0
    while cursor < len(raw_text):
        if raw_text[cursor].isspace():
            cursor += 1
            continue
        header = _HEADER.match(raw_text, cursor)
        if header is None:
            _fail()
        start_token, speaker = header.groups()
        text_start = cursor = header.end()
        while True:
            bracket_start = raw_text.find("[", cursor)
            # An unmatched closing bracket is also malformed protocol.
            if bracket_start < 0 or "]" in raw_text[cursor:bracket_start]:
                _fail()
            token = _BRACKET.match(raw_text, bracket_start)
            if token is None:
                _fail()
            value = token[1]
            if _DECIMAL.fullmatch(value):
                break
            # Preserve event annotations without allowing clock/label fragments.
            if (not value.strip() or re.match(r"[\s+\-.0-9]", value)
                    or value.strip().casefold() in {"nan", "inf", "infinity"}
                    or re.match(r"S(?:[0-9]|\Z)", value)):
                _fail()
            cursor = token.end()
        text = raw_text[text_start:bracket_start].strip()
        if not text:
            _fail()
        start, end = Fraction(start_token), Fraction(value)
        if not 0 <= start <= duration or start > end:
            _fail()
        normalized = end
        if end > duration:
            if end - duration > Fraction(1, 100) or duration <= start:
                raise MossValidationError(
                    "Invalid or incomplete MOSS transcript; invalid EOF endpoint "
                    f"reason={'beyond_allowance' if end - duration > Fraction(1, 100) else 'collapsed_span'} "
                    f"index={len(segments)} end={end} duration={duration} "
                    f"excess={end - duration}; retry transcription.")
            normalized = duration
            corrections.append({"index": len(segments), "raw_end": value,
                                "normalized_end": str(duration), "policy": POLICY})
        segments.append({"raw_start": start_token, "raw_end": value,
                         "start": str(start), "end": str(normalized),
                         "speaker": speaker, "text": text, "words": None})
        cursor = token.end()
    return {"schema": "moss-result-v1", "policy": POLICY, "raw_text": raw_text,
            "sample_count": sample_count, "sample_rate": sample_rate,
            "duration": str(duration), "completion": metadata,
            "segments": segments, "corrections": corrections}


def validate_worker_result(result, sample_count, sample_rate):
    """Recompute worker evidence against trusted input; return fresh validated data.

    Strict JSON comparison also rejects Python bool/int equality tricks and
    additional/missing fields. Transport/provenance belongs outside this schema.
    """
    if type(result) is not dict:
        _fail()
    try:
        expected = parse_generation(result["raw_text"], sample_count, sample_rate,
                                    result["completion"])
        if json.dumps(result, sort_keys=True, allow_nan=False) != json.dumps(
                expected, sort_keys=True, allow_nan=False):
            _fail()
    except (KeyError, TypeError, ValueError, OverflowError):
        _fail()
    return expected
