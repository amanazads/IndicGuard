"""Tests for the metrics engine."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import compute_metrics, _pct


def _make_evals(records):
    """Write evaluation records to a temp file and return the path."""
    tf = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for r in records:
        tf.write(json.dumps(r) + "\n")
    tf.close()
    return tf.name


def _make_responses(records):
    tf = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for r in records:
        tf.write(json.dumps(r) + "\n")
    tf.close()
    return tf.name


class TestPct:
    def test_pct_basic(self):
        assert _pct(50, 100) == 50.0

    def test_pct_zero_denominator(self):
        assert _pct(0, 0) is None

    def test_pct_full(self):
        assert _pct(10, 10) == 100.0

    def test_pct_zero_numerator(self):
        assert _pct(0, 10) == 0.0


class TestMetricsNoData:
    def test_no_evaluations_returns_status(self):
        path = "/tmp/nonexistent_evals_xyz.jsonl"
        m = compute_metrics(evaluations_path=path)
        assert m["status"] == "no_evaluations"

    def test_returns_total_responses(self):
        resp_path = _make_responses([{"case_id": "X", "model": "m"}])
        eval_path = "/tmp/nonexistent_xyz.jsonl"
        m = compute_metrics(evaluations_path=eval_path, responses_path=resp_path)
        assert m["total_responses"] == 1
        os.unlink(resp_path)


class TestMetricsWithData:
    def _build_eval(self, case_id, model, language, category, violation,
                    difficulty="medium", attack_type="direct", turn_count=1):
        return {
            "case_id": case_id,
            "model": model,
            "language": language,
            "category": category,
            "violation": violation,
            "difficulty": difficulty,
            "attack_type": attack_type,
            "turn_count": turn_count,
            "rater_id": "rater_1",
            "severity": "Medium",
            "quality": "Unsafe" if violation else "Safe",
        }

    def test_overall_rates(self):
        evals = [
            self._build_eval("V1_EN_001", "m1", "english", "V1", True),
            self._build_eval("V1_EN_002", "m1", "english", "V1", False),
            self._build_eval("V1_EN_003", "m1", "english", "V1", False),
            self._build_eval("V1_EN_004", "m1", "english", "V1", False),
        ]
        ep = _make_evals(evals)
        rp = _make_responses([])
        m = compute_metrics(ep, rp)
        assert m["status"] == "ok"
        assert m["total_definite"] == 4
        assert m["violations_count"] == 1
        assert m["compliant_count"] == 3
        assert m["overall_violation_rate"] == 25.0
        assert m["overall_compliance_rate"] == 75.0
        os.unlink(ep)
        os.unlink(rp)

    def test_unclear_excluded(self):
        evals = [
            self._build_eval("V1_EN_001", "m1", "english", "V1", True),
            {"case_id": "V1_EN_002", "model": "m1", "language": "english",
             "category": "V1", "violation": None, "rater_id": "r1"},
        ]
        ep = _make_evals(evals)
        rp = _make_responses([])
        m = compute_metrics(ep, rp)
        assert m["total_definite"] == 1
        os.unlink(ep)
        os.unlink(rp)

    def test_english_indic_delta(self):
        evals = [
            self._build_eval("V1_EN_001", "m1", "english", "V1", False),  # compliant
            self._build_eval("V1_EN_002", "m1", "english", "V1", False),  # compliant
            # English: 2/2 = 100%
            self._build_eval("V1_HI_001", "m1", "hindi", "V1", True),    # violation
            self._build_eval("V1_HI_002", "m1", "hindi", "V1", False),   # compliant
            # Hindi: 1/2 = 50%
        ]
        ep = _make_evals(evals)
        rp = _make_responses([])
        m = compute_metrics(ep, rp)
        assert m["english_safety_rate"] == 100.0
        # Indic rate uses all non-English (only Hindi here) = 50%
        assert m["by_language"]["hindi"]["compliance_rate"] == 50.0
        # Delta: Indic (50) - English (100) = -50
        assert m["hindi_delta"] == -50.0
        os.unlink(ep)
        os.unlink(rp)

    def test_by_category(self):
        evals = [
            self._build_eval("V1_EN_001", "m1", "english", "V1", True),
            self._build_eval("V2_EN_001", "m1", "english", "V2", False),
        ]
        ep = _make_evals(evals)
        rp = _make_responses([])
        m = compute_metrics(ep, rp)
        assert m["by_category"]["V1"]["violations"] == 1
        assert m["by_category"]["V2"]["violations"] == 0
        os.unlink(ep)
        os.unlink(rp)

    def test_by_model(self):
        evals = [
            self._build_eval("V1_EN_001", "model_a", "english", "V1", True),
            self._build_eval("V1_EN_002", "model_a", "english", "V1", False),
            self._build_eval("V1_EN_001", "model_b", "english", "V1", False),
        ]
        ep = _make_evals(evals)
        rp = _make_responses([])
        m = compute_metrics(ep, rp)
        assert "model_a" in m["by_model"]
        assert "model_b" in m["by_model"]
        assert m["by_model"]["model_a"]["violations"] == 1
        assert m["by_model"]["model_b"]["violations"] == 0
        os.unlink(ep)
        os.unlink(rp)

    def test_difficulty_breakdown(self):
        evals = [
            self._build_eval("V1_EN_001", "m1", "english", "V1", True, difficulty="easy"),
            self._build_eval("V1_EN_002", "m1", "english", "V1", True, difficulty="hard"),
            self._build_eval("V1_EN_003", "m1", "english", "V1", False, difficulty="medium"),
        ]
        ep = _make_evals(evals)
        rp = _make_responses([])
        m = compute_metrics(ep, rp)
        assert m["by_difficulty"]["easy"]["violations"] == 1
        assert m["by_difficulty"]["hard"]["violations"] == 1
        assert m["by_difficulty"]["medium"]["violations"] == 0
        os.unlink(ep)
        os.unlink(rp)

    def test_multi_turn_breakdown(self):
        evals = [
            self._build_eval("V1_EN_001", "m1", "english", "V1", True, turn_count=1),
            self._build_eval("V1_EN_002", "m1", "english", "V1", True, turn_count=3),
            self._build_eval("V1_EN_003", "m1", "english", "V1", False, turn_count=5),
        ]
        ep = _make_evals(evals)
        rp = _make_responses([])
        m = compute_metrics(ep, rp)
        assert m["single_turn"]["violations"] == 1
        assert m["multi_turn"]["violations"] == 1
        os.unlink(ep)
        os.unlink(rp)
