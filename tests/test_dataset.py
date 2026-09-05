"""Tests for adversarial dataset loading and validation."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_validator import load_cases, validate, VALID_CATEGORIES, VALID_LANGUAGES


DATASET_PATH = "data/adversarial_cases.jsonl"


class TestDatasetLoading:
    def test_dataset_exists(self):
        assert Path(DATASET_PATH).exists(), f"Dataset not found at {DATASET_PATH}"

    def test_dataset_loads_without_error(self):
        cases = load_cases(DATASET_PATH)
        assert isinstance(cases, list)

    def test_minimum_cases(self):
        cases = load_cases(DATASET_PATH)
        assert len(cases) >= 150, f"Expected at least 150 cases, got {len(cases)}"

    def test_target_cases(self):
        cases = load_cases(DATASET_PATH)
        # We aim for 160
        assert len(cases) >= 160, f"Expected 160 cases, got {len(cases)}"

    def test_all_lines_are_valid_json(self):
        errors = []
        with open(DATASET_PATH) as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"Line {i}: {e}")
        assert not errors, f"JSON parse errors: {errors[:5]}"


class TestDatasetIDs:
    def test_all_ids_unique(self):
        cases = load_cases(DATASET_PATH)
        ids = [c.get("id") for c in cases]
        duplicates = [i for i in ids if ids.count(i) > 1]
        assert not duplicates, f"Duplicate IDs found: {list(set(duplicates))}"

    def test_no_missing_ids(self):
        cases = load_cases(DATASET_PATH)
        missing = [i for i, c in enumerate(cases) if not c.get("id")]
        assert not missing, f"Cases with missing IDs at indices: {missing[:5]}"


class TestRequiredFields:
    REQUIRED = [
        "id", "language", "category", "category_name",
        "difficulty", "attack_type", "turn_count",
        "borrower_turns", "expected_behavior", "violation_condition",
    ]

    def test_required_fields_present(self):
        cases = load_cases(DATASET_PATH)
        errors = []
        for case in cases:
            cid = case.get("id", "<unknown>")
            for field in self.REQUIRED:
                if field not in case:
                    errors.append(f"{cid}: missing '{field}'")
        assert not errors, f"Missing fields: {errors[:10]}"

    def test_borrower_turns_are_non_empty_lists(self):
        cases = load_cases(DATASET_PATH)
        errors = []
        for case in cases:
            cid = case.get("id", "<unknown>")
            turns = case.get("borrower_turns", [])
            if not isinstance(turns, list) or len(turns) == 0:
                errors.append(cid)
        assert not errors, f"Cases with empty borrower_turns: {errors[:5]}"

    def test_expected_behavior_non_empty(self):
        cases = load_cases(DATASET_PATH)
        errors = [c.get("id") for c in cases if not c.get("expected_behavior", "").strip()]
        assert not errors, f"Cases with empty expected_behavior: {errors[:5]}"

    def test_violation_condition_non_empty(self):
        cases = load_cases(DATASET_PATH)
        errors = [c.get("id") for c in cases if not c.get("violation_condition", "").strip()]
        assert not errors, f"Cases with empty violation_condition: {errors[:5]}"


class TestDistribution:
    def test_all_languages_present(self):
        cases = load_cases(DATASET_PATH)
        langs = {c.get("language") for c in cases}
        for lang in VALID_LANGUAGES:
            assert lang in langs, f"Language '{lang}' missing from dataset"

    def test_all_categories_present(self):
        cases = load_cases(DATASET_PATH)
        cats = {c.get("category") for c in cases}
        for cat in VALID_CATEGORIES:
            assert cat in cats, f"Category '{cat}' missing from dataset"

    def test_each_language_minimum_cases(self):
        cases = load_cases(DATASET_PATH)
        from collections import Counter
        lang_counts = Counter(c.get("language") for c in cases)
        for lang in VALID_LANGUAGES:
            assert lang_counts.get(lang, 0) >= 10, \
                f"Language '{lang}' has only {lang_counts.get(lang, 0)} cases (expected ≥10)"

    def test_each_category_minimum_cases(self):
        cases = load_cases(DATASET_PATH)
        from collections import Counter
        cat_counts = Counter(c.get("category") for c in cases)
        for cat in VALID_CATEGORIES:
            assert cat_counts.get(cat, 0) >= 10, \
                f"Category '{cat}' has only {cat_counts.get(cat, 0)} cases (expected ≥10)"

    def test_multi_turn_cases_exist(self):
        cases = load_cases(DATASET_PATH)
        multi = [c for c in cases if c.get("turn_count", 1) > 1]
        assert len(multi) >= 40, f"Expected ≥40 multi-turn cases, got {len(multi)}"

    def test_valid_languages(self):
        cases = load_cases(DATASET_PATH)
        invalid = [c.get("id") for c in cases if c.get("language") not in VALID_LANGUAGES]
        assert not invalid, f"Cases with invalid language: {invalid[:5]}"

    def test_valid_categories(self):
        cases = load_cases(DATASET_PATH)
        invalid = [c.get("id") for c in cases if c.get("category") not in VALID_CATEGORIES]
        assert not invalid, f"Cases with invalid category: {invalid[:5]}"


class TestFullValidation:
    def test_validate_passes(self):
        """The validate() function should return True on the real dataset."""
        result = validate(DATASET_PATH, verbose=False)
        assert result is True, "Dataset validation failed"
