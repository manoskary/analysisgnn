"""Tests for Delta Lake reader using the pre-existing output at
``outputs/Minuet_in_G_Major_K.1/``.

These tests read the Delta Lake written by Step 1 (test_delta_writer.py).
If the output directory does not exist, all tests are skipped.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = str(REPO_ROOT / "outputs" / "Minuet_in_G_Major_K.1")

# Skip the entire module if the Delta Lake output does not exist
pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(OUTPUT_DIR, "notes", "_delta_log")),
    reason="Delta Lake output not found at outputs/Minuet_in_G_Major_K.1/",
)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


class TestLoadNotes:
    def test_shape_and_columns(self):
        from analysisgnn.storage.delta_reader import load_notes

        df = load_notes(OUTPUT_DIR)
        assert len(df) > 0, "Notes table should not be empty"
        expected_cols = {
            "note_id", "onset_div", "onset_beat", "duration_div",
            "duration_beat", "pitch_midi", "pitch_spelling", "staff",
            "voice", "measure", "ts_beats",
        }
        assert expected_cols.issubset(set(df.columns)), (
            f"Missing columns: {expected_cols - set(df.columns)}"
        )

    def test_note_ids_unique(self):
        from analysisgnn.storage.delta_reader import load_notes

        df = load_notes(OUTPUT_DIR)
        assert df["note_id"].is_unique, "note_id should be unique"


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


class TestLoadEdges:
    def test_full_load(self):
        from analysisgnn.storage.delta_reader import load_edges

        df = load_edges(OUTPUT_DIR)
        assert len(df) > 0
        assert set(df.columns) == {"src", "dst", "edge_type"}

    def test_filter_by_edge_type(self):
        from analysisgnn.storage.delta_reader import load_edges

        df_all = load_edges(OUTPUT_DIR)
        df_onset = load_edges(OUTPUT_DIR, edge_types=["onset"])
        assert len(df_onset) > 0
        assert set(df_onset["edge_type"].unique()) == {"onset"}
        assert len(df_onset) <= len(df_all)

    def test_filter_multiple_types(self):
        from analysisgnn.storage.delta_reader import load_edges

        df = load_edges(OUTPUT_DIR, edge_types=["onset", "consecutive"])
        actual_types = set(df["edge_type"].unique())
        assert actual_types.issubset({"onset", "consecutive"})


# ---------------------------------------------------------------------------
# Probabilities
# ---------------------------------------------------------------------------


class TestLoadProbabilities:
    def test_full_load(self):
        from analysisgnn.storage.delta_reader import load_probabilities

        df = load_probabilities(OUTPUT_DIR)
        assert len(df) > 0
        expected_cols = {
            "note_id", "task", "class_id", "class_label",
            "probability", "is_argmax", "rank",
        }
        assert expected_cols == set(df.columns)

    def test_filter_by_task(self):
        from analysisgnn.storage.delta_reader import load_probabilities

        df = load_probabilities(OUTPUT_DIR, task="romanNumeral")
        assert len(df) > 0
        assert set(df["task"].unique()) == {"romanNumeral"}

    def test_filter_by_top_k(self):
        from analysisgnn.storage.delta_reader import load_probabilities

        df = load_probabilities(OUTPUT_DIR, task="romanNumeral", top_k=3)
        assert len(df) > 0
        assert df["rank"].max() <= 3

    def test_probabilities_sum_to_one(self):
        from analysisgnn.storage.delta_reader import load_probabilities

        df = load_probabilities(OUTPUT_DIR)
        grouped = df.groupby(["note_id", "task"])["probability"].sum()
        assert np.allclose(grouped.values, 1.0, atol=1e-4), (
            f"Probability sums deviate from 1.0: "
            f"min={grouped.min()}, max={grouped.max()}"
        )


# ---------------------------------------------------------------------------
# Hyperedges
# ---------------------------------------------------------------------------


class TestLoadHyperedges:
    def test_full_load(self):
        from analysisgnn.storage.delta_reader import load_hyperedges

        df = load_hyperedges(OUTPUT_DIR)
        assert len(df) > 0
        expected_cols = {"group_id", "note_id", "edge_type", "parent_group_id"}
        assert expected_cols == set(df.columns)

    def test_filter_by_edge_type(self):
        from analysisgnn.storage.delta_reader import load_hyperedges

        df = load_hyperedges(OUTPUT_DIR, edge_type="beat")
        if len(df) > 0:
            assert set(df["edge_type"].unique()) == {"beat"}


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestLoadMetadata:
    def test_expected_keys(self):
        from analysisgnn.storage.delta_reader import load_metadata

        meta = load_metadata(OUTPUT_DIR)
        expected_keys = {
            "score_id", "inference_timestamp", "task_dict",
            "class_vocabularies", "edge_types_included", "num_edges",
            "hyperedge_types", "num_notes", "aggregation_history",
        }
        assert expected_keys.issubset(set(meta.keys())), (
            f"Missing metadata keys: {expected_keys - set(meta.keys())}"
        )

    def test_task_dict_matches_probabilities(self):
        from analysisgnn.storage.delta_reader import load_metadata, load_probabilities

        meta = load_metadata(OUTPUT_DIR)
        probs_df = load_probabilities(OUTPUT_DIR)
        probs_tasks = set(probs_df["task"].unique())
        meta_tasks = set(meta["task_dict"].keys())
        assert probs_tasks == meta_tasks, (
            f"Tasks mismatch: probs={probs_tasks}, meta={meta_tasks}"
        )


# ---------------------------------------------------------------------------
# Listing / discovery
# ---------------------------------------------------------------------------


class TestListGroupTypes:
    def test_returns_expected_types(self):
        from analysisgnn.storage.delta_reader import list_group_types

        types = list_group_types(OUTPUT_DIR)
        # Should contain at least onset; beat and measure may also be present
        assert "onset" in types
        for t in types:
            assert t in {"onset", "beat", "measure"}


class TestListTables:
    def test_returns_core_tables(self):
        from analysisgnn.storage.delta_reader import list_tables

        tables = list_tables(OUTPUT_DIR)
        expected = {"notes", "edges", "probabilities", "hyperedges"}
        assert expected.issubset(set(tables)), (
            f"Missing tables: {expected - set(tables)}"
        )


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


class TestExportCsv:
    def test_export_and_read_back(self):
        from analysisgnn.storage.delta_reader import export_table_to_csv, load_notes

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "notes.csv")
            result_path = export_table_to_csv(OUTPUT_DIR, "notes", csv_path)
            assert result_path == csv_path
            assert os.path.exists(csv_path)

            # Read back and compare
            csv_df = pd.read_csv(csv_path)
            original_df = load_notes(OUTPUT_DIR)
            assert len(csv_df) == len(original_df)
            assert set(csv_df.columns) == set(original_df.columns)


# ---------------------------------------------------------------------------
# Argmax summary
# ---------------------------------------------------------------------------


class TestArgmaxSummary:
    def test_shape_and_columns(self):
        from analysisgnn.storage.delta_reader import argmax_summary, load_notes

        summary = argmax_summary(OUTPUT_DIR)
        notes_df = load_notes(OUTPUT_DIR)

        # One row per note
        assert len(summary) == len(notes_df)
        # Should have note_id column
        assert "note_id" in summary.columns

    def test_task_columns_present(self):
        from analysisgnn.storage.delta_reader import argmax_summary

        tasks = ["romanNumeral", "localkey", "quality", "cadence"]
        summary = argmax_summary(OUTPUT_DIR, tasks=tasks)

        for task in tasks:
            assert task in summary.columns, f"Missing task column: {task}"
            assert f"{task}_confidence" in summary.columns, (
                f"Missing confidence column: {task}_confidence"
            )

    def test_task_filter(self):
        from analysisgnn.storage.delta_reader import argmax_summary

        summary_all = argmax_summary(OUTPUT_DIR)
        summary_subset = argmax_summary(OUTPUT_DIR, tasks=["romanNumeral"])

        # Both should have the same number of rows (one per note)
        assert len(summary_all) == len(summary_subset)
        # But subset should have fewer task columns
        all_task_cols = [c for c in summary_all.columns if c.endswith("_confidence")]
        sub_task_cols = [c for c in summary_subset.columns if c.endswith("_confidence")]
        assert len(sub_task_cols) < len(all_task_cols)
