"""Integration test for Delta Lake writer using real inference on Mozart K.1.

Requires:
    - conda environment ``analysisgnn`` active
    - Checkpoint at artifacts/gradio_checkpoints/uocj8f6y_full_last.ckpt
    - Score at notebooks/Minuet_in_G_Major_K.1.musicxml
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCORE_PATH = str(REPO_ROOT / "notebooks" / "Minuet_in_G_Major_K.1.musicxml")
CHECKPOINT_PATH = str(
    REPO_ROOT / "artifacts" / "gradio_checkpoints" / "uocj8f6y_full_last.ckpt"
)
OUTPUT_DIR = str(REPO_ROOT / "outputs" / "Minuet_in_G_Major_K.1")


@pytest.fixture(scope="module")
def inference_results():
    """Run inference once and return (predictions, intermediates, model)."""
    from analysisgnn.models.analysis import ContinualAnalysisGNN

    model = ContinualAnalysisGNN.load_from_checkpoint(
        CHECKPOINT_PATH, map_location="cpu", strict=False
    )
    model.eval()

    predictions, intermediates = model.predict(
        SCORE_PATH,
        aggregation_spec={"mode": "none"},
        return_intermediates=True,
    )
    return predictions, intermediates, model


@pytest.fixture(scope="module")
def written_delta(inference_results):
    """Write the Delta Lake and return the output directory."""
    predictions, intermediates, model = inference_results

    from analysisgnn.storage.delta_writer import write_analysis_results

    result_dir = write_analysis_results(
        output_dir=OUTPUT_DIR,
        score=intermediates["score"],
        note_array=intermediates["note_array"],
        predictions=predictions,
        data=intermediates["data"],
        task_dict=model.task_dict,
        metadata={
            "score_path": SCORE_PATH,
            "full_checkpoint": CHECKPOINT_PATH,
            "device": "cpu",
        },
    )
    return result_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNotesTable:
    def test_exists(self, written_delta):
        from deltalake import DeltaTable

        dt = DeltaTable(os.path.join(written_delta, "notes"))
        df = dt.to_pandas()
        assert len(df) > 0, "Notes table should not be empty"

    def test_columns(self, written_delta):
        from deltalake import DeltaTable

        df = DeltaTable(os.path.join(written_delta, "notes")).to_pandas()
        expected_cols = {
            "note_id", "onset_div", "onset_beat", "duration_div",
            "duration_beat", "pitch_midi", "pitch_spelling", "staff",
            "voice", "measure", "ts_beats",
        }
        assert expected_cols.issubset(set(df.columns)), (
            f"Missing columns: {expected_cols - set(df.columns)}"
        )

    def test_sorted_by_onset_pitch(self, written_delta):
        from deltalake import DeltaTable

        df = DeltaTable(os.path.join(written_delta, "notes")).to_pandas()
        # Check that onset_div is non-decreasing, and within same onset,
        # pitch_midi is non-decreasing
        for i in range(1, len(df)):
            assert df.iloc[i]["onset_div"] >= df.iloc[i - 1]["onset_div"]
            if df.iloc[i]["onset_div"] == df.iloc[i - 1]["onset_div"]:
                assert df.iloc[i]["pitch_midi"] >= df.iloc[i - 1]["pitch_midi"]

    def test_note_ids_unique(self, written_delta):
        from deltalake import DeltaTable

        df = DeltaTable(os.path.join(written_delta, "notes")).to_pandas()
        assert df["note_id"].is_unique


class TestEdgesTable:
    def test_exists(self, written_delta):
        from deltalake import DeltaTable

        df = DeltaTable(os.path.join(written_delta, "edges")).to_pandas()
        assert len(df) > 0

    def test_columns(self, written_delta):
        from deltalake import DeltaTable

        df = DeltaTable(os.path.join(written_delta, "edges")).to_pandas()
        assert set(df.columns) == {"src", "dst", "edge_type"}

    def test_edge_types_valid(self, written_delta):
        from deltalake import DeltaTable

        df = DeltaTable(os.path.join(written_delta, "edges")).to_pandas()
        valid_types = {"onset", "consecutive", "during", "rest"}
        actual_types = set(df["edge_type"].unique())
        assert actual_types.issubset(valid_types), (
            f"Unexpected edge types: {actual_types - valid_types}"
        )

    def test_src_dst_are_valid_note_ids(self, written_delta):
        from deltalake import DeltaTable

        notes_df = DeltaTable(os.path.join(written_delta, "notes")).to_pandas()
        edges_df = DeltaTable(os.path.join(written_delta, "edges")).to_pandas()
        note_ids = set(notes_df["note_id"])
        assert set(edges_df["src"]).issubset(note_ids)
        assert set(edges_df["dst"]).issubset(note_ids)


class TestProbabilitiesTable:
    def test_exists(self, written_delta):
        from deltalake import DeltaTable

        df = DeltaTable(os.path.join(written_delta, "probabilities")).to_pandas()
        assert len(df) > 0

    def test_columns(self, written_delta):
        from deltalake import DeltaTable

        df = DeltaTable(os.path.join(written_delta, "probabilities")).to_pandas()
        expected_cols = {
            "note_id", "task", "class_id", "class_label",
            "probability", "is_argmax", "rank",
        }
        assert expected_cols == set(df.columns), (
            f"Column mismatch: expected {expected_cols}, got {set(df.columns)}"
        )

    def test_probabilities_sum_to_one(self, written_delta):
        from deltalake import DeltaTable

        df = DeltaTable(os.path.join(written_delta, "probabilities")).to_pandas()
        # For each (note_id, task), probabilities should sum to ~1.0
        grouped = df.groupby(["note_id", "task"])["probability"].sum()
        assert np.allclose(grouped.values, 1.0, atol=1e-4), (
            f"Probability sums deviate from 1.0: min={grouped.min()}, max={grouped.max()}"
        )

    def test_exactly_one_argmax_per_note_task(self, written_delta):
        from deltalake import DeltaTable

        df = DeltaTable(os.path.join(written_delta, "probabilities")).to_pandas()
        argmax_counts = df.groupby(["note_id", "task"])["is_argmax"].sum()
        assert (argmax_counts == 1).all(), "Each (note, task) should have exactly one argmax"

    def test_rank_1_is_argmax(self, written_delta):
        from deltalake import DeltaTable

        df = DeltaTable(os.path.join(written_delta, "probabilities")).to_pandas()
        rank1 = df[df["rank"] == 1]
        assert rank1["is_argmax"].all(), "Rank 1 should always be the argmax"

    def test_note_ids_match_notes_table(self, written_delta):
        from deltalake import DeltaTable

        notes_df = DeltaTable(os.path.join(written_delta, "notes")).to_pandas()
        probs_df = DeltaTable(os.path.join(written_delta, "probabilities")).to_pandas()
        assert set(probs_df["note_id"].unique()) == set(notes_df["note_id"])

    def test_class_labels_not_empty_for_known_tasks(self, written_delta):
        from deltalake import DeltaTable

        df = DeltaTable(os.path.join(written_delta, "probabilities")).to_pandas()
        # For known tasks, class_label should never be NaN
        assert df["class_label"].notna().all(), "class_label should never be NaN"

    def test_cadence_labels_present(self, written_delta):
        from deltalake import DeltaTable

        df = DeltaTable(os.path.join(written_delta, "probabilities")).to_pandas()
        if "cadence" in df["task"].unique():
            cadence_labels = df[df["task"] == "cadence"]["class_label"].unique()
            # Should include at least the empty string (no cadence) and some cadence types
            assert len(cadence_labels) > 0


class TestHyperedgesTable:
    def test_exists(self, written_delta):
        from deltalake import DeltaTable

        df = DeltaTable(os.path.join(written_delta, "hyperedges")).to_pandas()
        assert len(df) > 0

    def test_columns(self, written_delta):
        from deltalake import DeltaTable

        df = DeltaTable(os.path.join(written_delta, "hyperedges")).to_pandas()
        expected_cols = {"group_id", "note_id", "edge_type", "parent_group_id"}
        assert expected_cols == set(df.columns)

    def test_onset_groups_present(self, written_delta):
        from deltalake import DeltaTable

        df = DeltaTable(os.path.join(written_delta, "hyperedges")).to_pandas()
        assert "onset" in df["edge_type"].unique()

    def test_note_ids_match_notes_table(self, written_delta):
        from deltalake import DeltaTable

        notes_df = DeltaTable(os.path.join(written_delta, "notes")).to_pandas()
        he_df = DeltaTable(os.path.join(written_delta, "hyperedges")).to_pandas()
        assert set(he_df["note_id"]).issubset(set(notes_df["note_id"]))


class TestMetadata:
    def test_exists(self, written_delta):
        meta_path = os.path.join(written_delta, "metadata.json")
        assert os.path.exists(meta_path)

    def test_keys(self, written_delta):
        with open(os.path.join(written_delta, "metadata.json")) as f:
            meta = json.load(f)
        expected_keys = {
            "score_id", "inference_timestamp", "task_dict",
            "class_vocabularies", "edge_types_included", "num_edges",
            "hyperedge_types", "num_notes", "aggregation_history",
        }
        assert expected_keys.issubset(set(meta.keys())), (
            f"Missing metadata keys: {expected_keys - set(meta.keys())}"
        )

    def test_score_id(self, written_delta):
        with open(os.path.join(written_delta, "metadata.json")) as f:
            meta = json.load(f)
        assert meta["score_id"] == "Minuet_in_G_Major_K.1"

    def test_class_vocabularies_ordered(self, written_delta):
        """Verify that class_vocabularies[task][i] resolves class_id i correctly."""
        with open(os.path.join(written_delta, "metadata.json")) as f:
            meta = json.load(f)
        from deltalake import DeltaTable

        probs_df = DeltaTable(os.path.join(written_delta, "probabilities")).to_pandas()

        for task, vocab in meta["class_vocabularies"].items():
            task_rows = probs_df[probs_df["task"] == task]
            if task_rows.empty:
                continue
            # For each class_id, the class_label should match vocab[class_id]
            for _, row in task_rows.drop_duplicates("class_id").iterrows():
                cid = int(row["class_id"])
                expected_label = vocab[cid] if cid < len(vocab) else f"class_{cid}"
                assert row["class_label"] == expected_label, (
                    f"Task {task}, class_id {cid}: "
                    f"expected '{expected_label}', got '{row['class_label']}'"
                )

    def test_num_notes_matches(self, written_delta):
        with open(os.path.join(written_delta, "metadata.json")) as f:
            meta = json.load(f)
        from deltalake import DeltaTable

        notes_df = DeltaTable(os.path.join(written_delta, "notes")).to_pandas()
        assert meta["num_notes"] == len(notes_df)
