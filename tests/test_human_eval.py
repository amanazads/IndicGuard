"""Tests for human evaluation storage, schemas, and inter-rater agreement."""

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


class TestHumanEvalStorage:
    def test_save_and_load_evaluation(self):
        path = _temp_path()
        record = save_evaluation(
            case_id="V1_EN_001",
            model="gemini_baseline",
            rater_id="Rater A",
            violation=True,
            category="V1",
            severity="High",
            expected_behavior_followed="No",
            evidence="Agent threatened to dispatch recovery team to office.",
            notes="Direct violation under pressure.",
            path=path,
        )
        assert record["case_id"] == "V1_EN_001"
        assert record["rater_id"] == "Rater A"
        assert record["expected_behavior_followed"] == "No"
        assert record["evidence"] == "Agent threatened to dispatch recovery team to office."

        records = load_evaluations(path)
        assert len(records) == 1
        assert records[0]["case_id"] == "V1_EN_001"
        assert records[0]["violation"] is True
        os.unlink(path)

    def test_get_evaluated_keys(self):
        path = _temp_path()
        save_evaluation("V1_EN_001", "m1", "Rater A", True, "V1", "High", path=path)
        save_evaluation("V1_EN_002", "m2", "Rater B", False, "None", "N/A", path=path)
        records = load_evaluations(path)
        keys = get_evaluated_keys(records)
        assert ("V1_EN_001", "m1") in keys
        assert ("V1_EN_002", "m2") in keys
        os.unlink(path)

    def test_unicode_notes_and_evidence(self):
        path = _temp_path()
        evidence = "एजेंट ने धमकी दी कि वो पुलिस भेजेंगे।"
        notes = "हिंदी में स्पष्ट उल्लंघन।"
        save_evaluation(
            "V1_HI_001", "qwen_9b", "Rater C", True, "V1", "Critical",
            evidence=evidence, notes=notes, path=path
        )
        records = load_evaluations(path)
        assert records[0]["evidence"] == evidence
        assert records[0]["notes"] == notes
        os.unlink(path)


class TestInterRaterAgreement:
    def test_insufficient_data_returns_message(self):
        evals = [
            {"case_id": "c1", "model": "m1", "rater_id": "Rater A", "violation": True, "category": "V1"}
        ]
        result = compute_agreement(evals)
        assert result["status"] == "insufficient_data"
        assert result["message"] == "Insufficient annotations"

    def test_perfect_agreement(self):
        evals = [
            {"case_id": "c1", "model": "m1", "rater_id": "Rater A", "violation": True, "category": "V1"},
            {"case_id": "c1", "model": "m1", "rater_id": "Rater B", "violation": True, "category": "V1"},
            {"case_id": "c2", "model": "m1", "rater_id": "Rater A", "violation": False, "category": "None"},
            {"case_id": "c2", "model": "m1", "rater_id": "Rater B", "violation": False, "category": "None"},
        ]
        result = compute_agreement(evals)
        assert result["status"] == "ok"
        assert result["raw_agreement"] == 1.0
        assert result["violation_agreement"] == 1.0
        assert result["category_agreement"] == 1.0
        assert result["cohens_kappa"] == 1.0

    def test_partial_agreement(self):
        evals = [
            {"case_id": "c1", "model": "m1", "rater_id": "Rater A", "violation": True, "category": "V1"},
            {"case_id": "c1", "model": "m1", "rater_id": "Rater B", "violation": True, "category": "V1"},
            {"case_id": "c2", "model": "m1", "rater_id": "Rater A", "violation": True, "category": "V2"},
            {"case_id": "c2", "model": "m1", "rater_id": "Rater B", "violation": False, "category": "None"},
        ]
        result = compute_agreement(evals)
        assert result["status"] == "ok"
        assert result["raw_agreement"] == 0.5
        assert result["category_agreement"] == 0.5

    def test_zero_agreement(self):
        evals = [
            {"case_id": "c1", "model": "m1", "rater_id": "Rater A", "violation": True, "category": "V1"},
            {"case_id": "c1", "model": "m1", "rater_id": "Rater B", "violation": False, "category": "None"},
            {"case_id": "c2", "model": "m1", "rater_id": "Rater A", "violation": False, "category": "None"},
            {"case_id": "c2", "model": "m1", "rater_id": "Rater B", "violation": True, "category": "V1"},
        ]
        result = compute_agreement(evals)
        assert result["status"] == "ok"
        assert result["raw_agreement"] == 0.0

    def test_unclear_excluded_from_kappa(self):
        evals = [
            {"case_id": "c1", "model": "m1", "rater_id": "Rater A", "violation": None, "category": "None"},
            {"case_id": "c1", "model": "m1", "rater_id": "Rater B", "violation": True, "category": "V1"},
        ]
        result = compute_agreement(evals)
        assert result["status"] == "ok"
        assert result["cohens_kappa"] is None


class TestValidationSubset:
    def test_load_validation_cases(self):
        from src.human_eval import load_validation_cases
        cases = load_validation_cases()
        assert len(cases) == 32
        languages = {c["language"] for c in cases}
        assert languages == {"english", "hindi", "hinglish", "marathi"}
        categories = {c["category"] for c in cases}
        assert len(categories) == 8
