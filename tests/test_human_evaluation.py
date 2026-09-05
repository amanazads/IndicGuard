"""Tests for human evaluation storage and inter-rater agreement."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.human_eval import (
    save_evaluation,
    load_evaluations,
    get_evaluated_keys,
    compute_agreement,
)


def _temp_path():
    tf = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    tf.close()
    os.unlink(tf.name)
    return tf.name


class TestSaveEvaluation:
    def test_saves_record(self):
        path = _temp_path()
        save_evaluation(
            case_id="V1_EN_001",
            model="test_model",
            rater_id="rater_1",
            violation=True,
            category="V1",
            severity="High",
            quality="Unsafe",
            correct_behavior=False,
            notes="Agent threatened arrest.",
            path=path,
        )
        records = load_evaluations(path)
        assert len(records) == 1
        assert records[0]["violation"] is True
        assert records[0]["category"] == "V1"
        os.unlink(path)

    def test_multiple_saves_append(self):
        path = _temp_path()
        for i in range(3):
            save_evaluation(
                case_id=f"V1_EN_00{i}",
                model="m",
                rater_id="rater_1",
                violation=False,
                category="None",
                severity="N/A",
                quality="Safe",
                correct_behavior=True,
                path=path,
            )
        records = load_evaluations(path)
        assert len(records) == 3
        os.unlink(path)

    def test_unclear_violation_saved_as_none(self):
        path = _temp_path()
        save_evaluation(
            case_id="V1_EN_001",
            model="m",
            rater_id="r1",
            violation=None,
            category="None",
            severity="N/A",
            quality="Safe",
            correct_behavior=None,
            path=path,
        )
        records = load_evaluations(path)
        assert records[0]["violation"] is None
        os.unlink(path)

    def test_timestamp_present(self):
        path = _temp_path()
        save_evaluation(
            "X", "m", "r1", True, "V1", "High", "Unsafe", False, path=path
        )
        records = load_evaluations(path)
        assert "timestamp" in records[0]
        assert records[0]["timestamp"]
        os.unlink(path)

    def test_unicode_notes_preserved(self):
        path = _temp_path()
        notes = "एजेंट ने पुलिस की धमकी दी। यह V1 violation है।"
        save_evaluation(
            "V1_HI_001", "m", "r1", True, "V1", "High", "Unsafe", False,
            notes=notes, path=path
        )
        records = load_evaluations(path)
        assert records[0]["notes"] == notes
        os.unlink(path)


class TestLoadEvaluations:
    def test_empty_file_returns_empty_list(self):
        path = _temp_path()
        Path(path).write_text("")
        records = load_evaluations(path)
        assert records == []

    def test_nonexistent_file_returns_empty_list(self):
        records = load_evaluations("/tmp/nonexistent_eval_123.jsonl")
        assert records == []


class TestEvaluatedKeys:
    def test_get_evaluated_keys(self):
        evals = [
            {"case_id": "V1_EN_001", "model": "m1"},
            {"case_id": "V2_HI_001", "model": "m2"},
        ]
        keys = get_evaluated_keys(evals)
        assert ("V1_EN_001", "m1") in keys
        assert ("V2_HI_001", "m2") in keys
        assert len(keys) == 2


class TestInterRaterAgreement:
    def test_insufficient_data_status(self):
        evals = [
            {"case_id": "V1_EN_001", "model": "m1", "rater_id": "r1", "violation": True},
        ]
        result = compute_agreement(evals)
        assert result["status"] == "insufficient_data"

    def test_perfect_agreement(self):
        evals = [
            {"case_id": "V1_EN_001", "model": "m1", "rater_id": "r1", "violation": True},
            {"case_id": "V1_EN_001", "model": "m1", "rater_id": "r2", "violation": True},
            {"case_id": "V2_EN_001", "model": "m1", "rater_id": "r1", "violation": False},
            {"case_id": "V2_EN_001", "model": "m1", "rater_id": "r2", "violation": False},
        ]
        result = compute_agreement(evals)
        assert result["status"] == "ok"
        assert result["raw_agreement"] == 1.0

    def test_zero_agreement(self):
        evals = [
            {"case_id": "V1_EN_001", "model": "m1", "rater_id": "r1", "violation": True},
            {"case_id": "V1_EN_001", "model": "m1", "rater_id": "r2", "violation": False},
            {"case_id": "V2_EN_001", "model": "m1", "rater_id": "r1", "violation": False},
            {"case_id": "V2_EN_001", "model": "m1", "rater_id": "r2", "violation": True},
        ]
        result = compute_agreement(evals)
        assert result["status"] == "ok"
        assert result["raw_agreement"] == 0.0

    def test_cohens_kappa_present(self):
        evals = [
            {"case_id": "V1_EN_001", "model": "m1", "rater_id": "r1", "violation": True},
            {"case_id": "V1_EN_001", "model": "m1", "rater_id": "r2", "violation": True},
        ]
        result = compute_agreement(evals)
        assert result["status"] == "ok"
        assert "cohens_kappa" in result

    def test_unclear_excluded_from_kappa(self):
        evals = [
            {"case_id": "V1_EN_001", "model": "m1", "rater_id": "r1", "violation": None},
            {"case_id": "V1_EN_001", "model": "m1", "rater_id": "r2", "violation": None},
        ]
        result = compute_agreement(evals)
        # Both unclear → no definite pairs for kappa
        assert result.get("cohens_kappa") is None
