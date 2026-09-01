"""Tests for multi-seed aggregation.

The behaviour that matters most here is honesty about uncertainty: a
single-seed number must be *labelled* as such rather than shown with a
fabricated ``± 0.0000``, and two arms whose gap is smaller than the seed
spread must be reported as indistinguishable rather than ranked.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.aggregate import (  # noqa: E402
    SeedStat,
    aggregate_seeds,
    compare_arms,
    flatten_for_csv,
    format_seed_row,
    parse_seeds,
)


# --------------------------------------------------------------------------- #
# Seed parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,expected", [
    ("0,1,2", [0, 1, 2]),
    ("0 1 2", [0, 1, 2]),
    ("0-2", [0, 1, 2]),
    ("3", [3]),
    ("0,2-4", [0, 2, 3, 4]),
])
def test_parse_seeds_accepts_common_forms(text, expected):
    assert parse_seeds(text) == expected


def test_parse_seeds_defaults_when_empty():
    assert parse_seeds(None) == [0]
    assert parse_seeds("", default=(7, 8)) == [7, 8]


# --------------------------------------------------------------------------- #
# SeedStat
# --------------------------------------------------------------------------- #
def test_mean_and_sample_std():
    stat = SeedStat("image_auroc", [0.90, 0.92, 0.94])
    assert stat.mean == pytest.approx(0.92)
    assert stat.std == pytest.approx(0.02)          # ddof=1
    assert stat.n == 3
    assert stat.spread == (0.90, 0.94)


def test_single_seed_reports_zero_std_but_is_labelled():
    stat = SeedStat("image_auroc", [0.9])
    assert stat.std == 0.0
    assert "1 seed" in stat.format(), "single-seed value must be labelled as such"
    assert "±" not in stat.format(), "must not fabricate an uncertainty bar"


def test_multi_seed_formats_with_a_plus_minus():
    text = SeedStat("m", [0.90, 0.92, 0.94]).format(precision=3)
    assert "±" in text and text.startswith("0.920")


def test_nan_values_are_excluded_from_the_statistics():
    stat = SeedStat("m", [0.9, float("nan"), 0.7])
    assert stat.mean == pytest.approx(0.8)
    assert stat.to_dict()["n_seeds"] == 2


def test_all_nan_yields_nan_mean():
    assert math.isnan(SeedStat("m", [float("nan"), float("nan")]).mean)


def test_empty_stat_formats_as_not_available():
    assert SeedStat("m", []).format() == "n/a"


def test_sem_shrinks_with_more_seeds():
    few = SeedStat("m", [0.1, 0.3])
    many = SeedStat("m", [0.1, 0.3, 0.1, 0.3, 0.1, 0.3])
    assert many.sem < few.sem


# --------------------------------------------------------------------------- #
# Aggregation over result dicts
# --------------------------------------------------------------------------- #
def test_aggregate_collects_every_numeric_metric():
    results = [
        {"image_auroc": 0.90, "au_pro": 0.70, "category": "pushpins"},
        {"image_auroc": 0.94, "au_pro": 0.74, "category": "pushpins"},
    ]
    stats = aggregate_seeds(results)

    assert stats["image_auroc"].mean == pytest.approx(0.92)
    assert stats["au_pro"].mean == pytest.approx(0.72)
    assert "category" not in stats, "non-numeric field was coerced"


def test_aggregate_can_be_restricted_to_named_metrics():
    results = [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}]
    stats = aggregate_seeds(results, metrics=["a"])
    assert set(stats) == {"a"}


def test_aggregate_tolerates_a_metric_missing_from_one_seed():
    results = [{"a": 1.0, "b": 2.0}, {"a": 3.0}]
    stats = aggregate_seeds(results)
    assert stats["a"].n == 2
    assert stats["b"].n == 1


def test_aggregate_of_nothing_is_empty():
    assert aggregate_seeds([]) == {}


def test_booleans_are_not_treated_as_metrics():
    stats = aggregate_seeds([{"ok": True, "score": 1.0}, {"ok": False, "score": 2.0}])
    assert "ok" not in stats and "score" in stats


# --------------------------------------------------------------------------- #
# Row formatting
# --------------------------------------------------------------------------- #
def test_format_seed_row_reports_the_seed_count():
    stats = aggregate_seeds([{"m": 0.1}, {"m": 0.3}, {"m": 0.2}])
    row = format_seed_row("arm_a", stats, ["m"])
    assert row["arm"] == "arm_a"
    assert row["seeds"] == "3"
    assert "±" in row["m"]


def test_format_seed_row_marks_a_missing_metric():
    row = format_seed_row("arm_a", aggregate_seeds([{"m": 0.1}]), ["m", "absent"])
    assert row["absent"] == "n/a"


def test_flatten_for_csv_splits_mean_and_std_into_columns():
    stats = aggregate_seeds([{"m": 0.90}, {"m": 0.94}])
    row = flatten_for_csv("arm_a", "desc", stats, ["m"])
    assert row["m_mean"] == pytest.approx(0.92)
    assert row["m_std"] == pytest.approx(0.0283, abs=1e-3)
    assert row["n_seeds"] == 2


def test_flatten_for_csv_fills_missing_metrics_with_nan():
    row = flatten_for_csv("a", "d", aggregate_seeds([{"m": 1.0}]), ["m", "gone"])
    assert math.isnan(row["gone_mean"])


# --------------------------------------------------------------------------- #
# Arm comparison -- refusing to over-claim
# --------------------------------------------------------------------------- #
def test_a_gap_smaller_than_the_noise_is_called_indistinguishable():
    a = SeedStat("m", [0.900, 0.950, 0.850])      # std ~0.05
    b = SeedStat("m", [0.905, 0.955, 0.855])      # same spread, tiny offset
    assert "indistinguishable" in compare_arms("A", a, "B", b)


def test_a_large_gap_is_reported_as_a_win():
    a = SeedStat("m", [0.90, 0.91, 0.92])
    b = SeedStat("m", [0.50, 0.51, 0.52])
    text = compare_arms("A", a, "B", b)
    assert text.startswith("A > B")
    assert "σ" in text


def test_comparison_names_the_better_arm_regardless_of_order():
    a = SeedStat("m", [0.50, 0.51, 0.52])
    b = SeedStat("m", [0.90, 0.91, 0.92])
    assert compare_arms("A", a, "B", b).startswith("B > A")


def test_single_seed_comparison_flags_the_missing_spread():
    a, b = SeedStat("m", [0.9]), SeedStat("m", [0.5])
    text = compare_arms("A", a, "B", b)
    assert "single seed" in text


def test_comparison_handles_empty_input():
    assert "insufficient" in compare_arms("A", SeedStat("m", []), "B", SeedStat("m", [1.0]))
