"""Validation tests for the new aggregation package.

These tests load the Delta Lake at ``outputs/Minuet_in_G_Major_K.1/`` and
the reference CSVs at ``outputs/Minuet_in_G_Major_K.1/reference_*.csv``,
run the corresponding aggregation strategies, and assert the results match
exactly (within floating-point tolerance for confidence columns).

Skipped automatically if the Delta Lake output or reference CSVs are absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = str(_ROOT / "outputs" / "Minuet_in_G_Major_K.1")
_REF_NONE = str(_ROOT / "outputs" / "Minuet_in_G_Major_K.1" / "reference_none.csv")
_REF_MEAN = str(_ROOT / "outputs" / "Minuet_in_G_Major_K.1" / "reference_mean.csv")

_SKIP = not (
    os.path.isdir(os.path.join(_OUTPUT_DIR, "notes", "_delta_log"))
    and os.path.isfile(_REF_NONE)
    and os.path.isfile(_REF_MEAN)
)

pytestmark = pytest.mark.skipif(_SKIP, reason="Delta Lake output or reference CSVs not found")


# The 21 tasks in display order
ALL_TASKS = [
    "cadence", "localkey", "tonkey", "quality", "inversion", "root", "bass",
    "degree1", "degree2", "hrythm", "pcset", "romanNumeral", "section",
    "phrase", "organ_point", "tpc_in_label", "tpc_is_root", "tpc_is_bass",
    "downbeat", "note_degree", "staff",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def delta_data():
    """Load Delta Lake tables once per module."""
    from analysisgnn.storage.delta_reader import (
        load_hyperedges,
        load_metadata,
        load_notes,
        load_probabilities,
    )

    return {
        "probabilities": load_probabilities(_OUTPUT_DIR),
        "notes": load_notes(_OUTPUT_DIR),
        "hyperedges": load_hyperedges(_OUTPUT_DIR),
        "metadata": load_metadata(_OUTPUT_DIR),
    }


@pytest.fixture(scope="module")
def ref_none():
    """Load the reference CSV for 'none' mode."""
    return pd.read_csv(_REF_NONE, keep_default_na=False)


@pytest.fixture(scope="module")
def ref_mean():
    """Load the reference CSV for 'mean' mode."""
    return pd.read_csv(_REF_MEAN, keep_default_na=False)


# ---------------------------------------------------------------------------
# Tests: "none" strategy
# ---------------------------------------------------------------------------


class TestNoneAggregation:
    """Verify the 'none' strategy matches reference_none.csv."""

    def test_none_labels_match(self, delta_data, ref_none):
        from analysisgnn.aggregation import get_strategy

        strategy = get_strategy("none")
        result = strategy.aggregate(
            delta_data["probabilities"],
            delta_data["notes"],
            delta_data["hyperedges"],
            delta_data["metadata"],
            tasks=ALL_TASKS,
        )

        # Compare task label columns
        for task in ALL_TASKS:
            if task not in ref_none.columns:
                continue
            ref_col = ref_none[task].astype(str).fillna("")
            res_col = result[task].astype(str).fillna("")
            assert res_col.tolist() == ref_col.tolist(), (
                f"Task {task!r}: labels differ.\n"
                f"  First diff at row {_first_diff(res_col, ref_col)}"
            )

    def test_none_confidences_match(self, delta_data, ref_none):
        from analysisgnn.aggregation import get_strategy

        strategy = get_strategy("none")
        result = strategy.aggregate(
            delta_data["probabilities"],
            delta_data["notes"],
            delta_data["hyperedges"],
            delta_data["metadata"],
            tasks=ALL_TASKS,
        )

        for task in ALL_TASKS:
            conf_col = f"{task}_confidence"
            if conf_col not in ref_none.columns:
                continue
            ref_vals = ref_none[conf_col].values.astype(float)
            res_vals = result[conf_col].values.astype(float)
            np.testing.assert_allclose(
                res_vals, ref_vals, atol=1e-5, rtol=1e-5,
                err_msg=f"Confidence mismatch for {task!r}",
            )

    def test_none_row_count(self, delta_data, ref_none):
        from analysisgnn.aggregation import get_strategy

        strategy = get_strategy("none")
        result = strategy.aggregate(
            delta_data["probabilities"],
            delta_data["notes"],
            delta_data["hyperedges"],
            delta_data["metadata"],
            tasks=ALL_TASKS,
        )
        assert len(result) == len(ref_none), (
            f"Row count: got {len(result)}, expected {len(ref_none)}"
        )


# ---------------------------------------------------------------------------
# Tests: "mean" strategy
# ---------------------------------------------------------------------------


class TestMeanAggregation:
    """Verify the 'mean' strategy matches reference_mean.csv."""

    def test_mean_labels_match(self, delta_data, ref_mean):
        from analysisgnn.aggregation import get_strategy

        strategy = get_strategy("mean")
        result = strategy.aggregate(
            delta_data["probabilities"],
            delta_data["notes"],
            delta_data["hyperedges"],
            delta_data["metadata"],
            tasks=ALL_TASKS,
        )

        for task in ALL_TASKS:
            if task not in ref_mean.columns:
                continue
            ref_col = ref_mean[task].astype(str).fillna("")
            res_col = result[task].astype(str).fillna("")
            assert res_col.tolist() == ref_col.tolist(), (
                f"Task {task!r}: labels differ.\n"
                f"  First diff at row {_first_diff(res_col, ref_col)}"
            )

    def test_mean_confidences_match(self, delta_data, ref_mean):
        from analysisgnn.aggregation import get_strategy

        strategy = get_strategy("mean")
        result = strategy.aggregate(
            delta_data["probabilities"],
            delta_data["notes"],
            delta_data["hyperedges"],
            delta_data["metadata"],
            tasks=ALL_TASKS,
        )

        for task in ALL_TASKS:
            conf_col = f"{task}_confidence"
            if conf_col not in ref_mean.columns:
                continue
            ref_vals = ref_mean[conf_col].values.astype(float)
            res_vals = result[conf_col].values.astype(float)
            np.testing.assert_allclose(
                res_vals, ref_vals, atol=1e-5, rtol=1e-5,
                err_msg=f"Confidence mismatch for {task!r}",
            )

    def test_mean_row_count(self, delta_data, ref_mean):
        from analysisgnn.aggregation import get_strategy

        strategy = get_strategy("mean")
        result = strategy.aggregate(
            delta_data["probabilities"],
            delta_data["notes"],
            delta_data["hyperedges"],
            delta_data["metadata"],
            tasks=ALL_TASKS,
        )
        assert len(result) == len(ref_mean), (
            f"Row count: got {len(result)}, expected {len(ref_mean)}"
        )


# ---------------------------------------------------------------------------
# Tests: Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    """Basic registry tests."""

    def test_list_strategies(self):
        from analysisgnn.aggregation import list_strategies

        names = list_strategies()
        assert "none" in names
        assert "mean" in names

    def test_unknown_strategy_raises(self):
        from analysisgnn.aggregation import get_strategy

        with pytest.raises(KeyError, match="Unknown aggregation strategy"):
            get_strategy("nonexistent_strategy")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_diff(a, b):
    """Return the index of the first element that differs."""
    a_list = a.tolist() if hasattr(a, "tolist") else list(a)
    b_list = b.tolist() if hasattr(b, "tolist") else list(b)
    for i, (x, y) in enumerate(zip(a_list, b_list)):
        if str(x) != str(y):
            return f"{i} (got={x!r}, expected={y!r})"
    if len(a_list) != len(b_list):
        return f"length mismatch: {len(a_list)} vs {len(b_list)}"
    return "no diff found"
