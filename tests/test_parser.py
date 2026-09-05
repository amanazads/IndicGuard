"""Tests for JSONL parsing and response/eval loading."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_validator import load_cases
from src.human_eval import load_evaluations
from src.metrics import _load_jsonl


class TestJSONLParsing:
    def test_load_valid_jsonl(self):
        tf = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        tf.write('{"a": 1}\n')
        tf.write('{"b": 2}\n')
        tf.close()
        records = _load_jsonl(tf.name)
        assert len(records) == 2
        assert records[0]["a"] == 1
        os.unlink(tf.name)

    def test_empty_lines_ignored(self):
        tf = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        tf.write('{"a": 1}\n\n\n{"b": 2}\n')
        tf.close()
        records = _load_jsonl(tf.name)
        assert len(records) == 2
        os.unlink(tf.name)

    def test_invalid_json_lines_skipped(self):
        tf = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        tf.write('{"a": 1}\n')
        tf.write('NOT VALID JSON\n')
        tf.write('{"b": 2}\n')
        tf.close()
        # Should not raise, just skip bad lines
        records = _load_jsonl(tf.name)
        assert len(records) == 2
        os.unlink(tf.name)

    def test_nonexistent_file_returns_empty(self):
        records = _load_jsonl("/tmp/nonexistent_indicguard_test.jsonl")
        assert records == []

    def test_unicode_content_preserved(self):
        tf = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
        content = {"text": "यह एक परीक्षण है। मराठी: हे test आहे।"}
        tf.write(json.dumps(content, ensure_ascii=False) + "\n")
        tf.close()
        records = _load_jsonl(tf.name)
        assert len(records) == 1
        assert "परीक्षण" in records[0]["text"]
        os.unlink(tf.name)


class TestCaseParser:
    def test_multi_turn_parsed_correctly(self):
        cases = load_cases("data/adversarial_cases.jsonl")
        multi_turn = [c for c in cases if c.get("turn_count", 1) > 1]
        for case in multi_turn[:5]:
            assert isinstance(case["borrower_turns"], list)
            assert len(case["borrower_turns"]) == case["turn_count"]

    def test_turn_count_matches_turns_list(self):
        cases = load_cases("data/adversarial_cases.jsonl")
        mismatches = []
        for c in cases:
            if c.get("turn_count") != len(c.get("borrower_turns", [])):
                mismatches.append(c.get("id"))
        assert not mismatches, f"Turn count mismatches: {mismatches[:5]}"

    def test_hinglish_cases_loaded(self):
        cases = load_cases("data/adversarial_cases.jsonl")
        hinglish = [c for c in cases if c["language"] == "hinglish"]
        assert len(hinglish) > 0

    def test_devanagari_content_in_hindi_cases(self):
        cases = load_cases("data/adversarial_cases.jsonl")
        hindi = [c for c in cases if c["language"] == "hindi"]
        # At least some Hindi cases should contain Devanagari
        devanagari = [c for c in hindi
                      if any('\u0900' <= ch <= '\u097F' for turn in c["borrower_turns"] for ch in turn)]
        assert len(devanagari) > 0, "Expected Devanagari script in Hindi cases"

    def test_marathi_cases_have_content(self):
        cases = load_cases("data/adversarial_cases.jsonl")
        marathi = [c for c in cases if c["language"] == "marathi"]
        assert len(marathi) >= 20
        for c in marathi:
            for turn in c["borrower_turns"]:
                assert turn.strip(), f"Empty turn in {c['id']}"
