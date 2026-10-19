"""Pure parser tests load the module without ingestion package/app side effects."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "moss_boundary", Path(__file__).parent / "services/ingestion/moss.py"
)
moss = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(moss)


@pytest.fixture
def completion():
    return {"complete": True, "reason": "eos", "eos_token_id": 151645,
            "terminal_token_id": 151645, "input_tokens": 200, "output_tokens": 108,
            "max_new_tokens": 5120, "context_limit": 131072}


def parse(text, completion, frames=144000, rate=16000):
    return moss.parse_generation(text, frames, rate, completion)


def test_native_overlap_nested_order_and_annotations(completion):
    raw = "[0][S0]outer [laughter] text[9][2][S1]inner[3][1][S0]again[4]"
    result = parse(raw, completion)
    assert result["raw_text"] == raw
    assert [(s["speaker"], s["end"]) for s in result["segments"]] == [
        ("S0", "9"), ("S1", "3"), ("S0", "4")]
    assert result["segments"][0]["text"] == "outer [laughter] text"
    assert all(s["words"] is None for s in result["segments"])
    assert not result["corrections"]


def test_exact_eof_and_json_evidence(completion):
    raw = "[0.10][S0]one[9.010000][1][S1]two[9.01]"
    result = parse(raw, completion)
    assert result["segments"][0] == {
        "raw_start": "0.10", "raw_end": "9.010000", "start": "1/10",
        "end": "9", "speaker": "S0", "text": "one", "words": None}
    assert result["corrections"] == [
        {"index": 0, "raw_end": "9.010000", "normalized_end": "9", "policy": "moss-eof-v1"},
        {"index": 1, "raw_end": "9.01", "normalized_end": "9", "policy": "moss-eof-v1"}]
    serialized = json.loads(json.dumps(result))
    assert moss.validate_worker_result(serialized, 144000, 16000) == result
    assert moss.validate_worker_result(serialized, 144000, 16000) == result


def test_inrange_zero_and_trailing_silence(completion):
    result = parse("[0][S0]zero[0][0.1][S1]short[8.999]", completion)
    assert result["segments"][0]["end"] == "0"
    assert result["segments"][1]["raw_end"] == "8.999"
    assert result["segments"][1]["end"] == "8999/1000"
    assert result["corrections"] == []
    assert parse(" \n\t", completion)["segments"] == []


@pytest.mark.parametrize("raw", [
    "[0][S0]x[9.010001]", "[-0.1][S0]x[1]", "[9.001][S0]x[9.01]",
    "[2][S0]x[1]", "[NaN][S0]x[1]", "[0][S0]x[Infinity]",
    "[9][S0]x[9.01]", "[0][S0]x[1]junk", "[0][S0]x[1][2]",
    "[0][S0]x[1/2]", "[0][S0]x[1e0]", "[0][S0]x[.5]",
    "[0][S0]x[NaN ]", "[0][S0]x[S1oops][1]",
    "[0][S0]x[[1]]", "[0][S0]x[annotation[1]]", "[0][S0]x]bad[1]",
    "[0][S0][S1]x[1]", "[0][]x[1]", "[0][S]x[1]", "[0][S0] [1]",
    "[0][S0]x[1][1][S1]y[9.010001]", "[0][S0]x[1][0][S0]",
])
def test_whole_generation_rejects_bad_protocol_and_clocks(raw, completion):
    with pytest.raises(moss.MossValidationError, match="Invalid or incomplete") as error:
        parse(raw, completion)
    assert raw not in str(error.value)


def test_sample_fraction_boundary(completion):
    accepted = parse("[0][S0]x[0.343333]", completion, 1, 3)
    assert accepted["duration"] == accepted["segments"][0]["end"] == "1/3"
    with pytest.raises(moss.MossValidationError):
        parse("[0][S0]x[0.343334]", completion, 1, 3)


@pytest.mark.parametrize("frames,rate", [(0, 16000), (-1, 16000), (True, 16000),
                                              (1, 0), (1, 1.0), (None, 16000)])
def test_invalid_dimensions(frames, rate, completion):
    with pytest.raises(moss.MossValidationError):
        parse("", completion, frames, rate)


@pytest.mark.parametrize("changes", [
    {"complete": False}, {"complete": 1}, {"reason": "length"},
    {"reason": "unknown"}, {"terminal_token_id": 10}, {"eos_token_id": True},
    {"output_tokens": 5120}, {"output_tokens": 0}, {"input_tokens": -1},
    {"max_new_tokens": 131000}, {"context_limit": 4096}, {"output_tokens": 1.5},
    {"unknown": True}, {"eos_token_id": 1, "terminal_token_id": 1},
])
def test_completion_cannot_be_rescued_by_silence_or_eof(changes, completion):
    completion.update(changes)
    for raw in ("", "[0][S0]x[9.01]"):
        with pytest.raises(moss.MossValidationError):
            parse(raw, completion)


def test_missing_completion(completion):
    for key in completion:
        partial = {k: v for k, v in completion.items() if k != key}
        with pytest.raises(moss.MossValidationError):
            parse("", partial)


@pytest.mark.parametrize("field,value", [
    ("sample_count", 144001), ("sample_rate", 16001), ("duration", "9.01"),
    ("policy", "other"), ("schema", "other"), ("corrections", []),
    ("segments", []), ("raw_text", "[0][S9]changed[9.01]"),
])
def test_backend_rejects_tampered_result(field, value, completion):
    result = parse("[0][S0]x[9.01]", completion)
    result[field] = value
    with pytest.raises(moss.MossValidationError):
        moss.validate_worker_result(result, 144000, 16000)


@pytest.mark.parametrize("field,value", [
    ("speaker", "S2"), ("start", "1"), ("end", "9.01"),
    ("raw_end", "9"), ("words", []), ("text", "tampered"),
])
def test_backend_checks_every_segment(field, value, completion):
    result = parse("[0][S0]x[9.01]", completion)
    result["segments"][0][field] = value
    with pytest.raises(moss.MossValidationError):
        moss.validate_worker_result(result, 144000, 16000)


def test_backend_checks_correction_metadata_dimensions_and_shape(completion):
    original = parse("[0][S0]x[9.01]", completion)
    variants = []
    for field, value in (("index", True), ("raw_end", "9.010000"),
                         ("normalized_end", "9.01"), ("policy", "other")):
        result = copy.deepcopy(original)
        result["corrections"][0][field] = value
        variants.append(result)
    result = copy.deepcopy(original)
    result["completion"]["reason"] = "length"
    variants.extend([result, {**original, "extra": 1}, {}, None])
    for result in variants:
        with pytest.raises(moss.MossValidationError):
            moss.validate_worker_result(result, 144000, 16000)
    with pytest.raises(moss.MossValidationError):
        moss.validate_worker_result(original, 144001, 16000)
