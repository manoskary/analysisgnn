#!/usr/bin/env python
"""Generate reference CSVs for aggregation validation.

Runs inference on Mozart K.1 in both ``"none"`` and ``"mean"`` aggregation
modes, then saves the resulting DataFrames as reference CSVs under
``outputs/Minuet_in_G_Major_K.1/``.

Usage::

    conda run -n analysisgnn python scripts/generate_reference_csvs.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysisgnn.models.analysis import ContinualAnalysisGNN
from analysisgnn.inference.hybrid_predictor import (
    predictions_to_dataframe,
)
from analysisgnn.utils.chord_representations import format_table_output

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCORE_PATH = str(PROJECT_ROOT / "notebooks" / "Minuet_in_G_Major_K.1.musicxml")
CHECKPOINT_PATH = str(
    PROJECT_ROOT / "artifacts" / "gradio_checkpoints" / "uocj8f6y_full_last.ckpt"
)
OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "Minuet_in_G_Major_K.1")

# The 21 tasks in display order (matches model.hparams.task_dict key order).
ALL_TASKS: List[str] = [
    "cadence",
    "localkey",
    "tonkey",
    "quality",
    "inversion",
    "root",
    "bass",
    "degree1",
    "degree2",
    "hrythm",
    "pcset",
    "romanNumeral",
    "section",
    "phrase",
    "organ_point",
    "tpc_in_label",
    "tpc_is_root",
    "tpc_is_bass",
    "downbeat",
    "note_degree",
    "staff",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading model from {CHECKPOINT_PATH} ...")
    model = ContinualAnalysisGNN.load_from_checkpoint(
        CHECKPOINT_PATH, map_location="cpu", strict=False
    )
    model.eval()

    modes = ["none", "mean"]
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Running inference with aggregation_mode={mode!r} ...")
        print(f"{'='*60}")

        result = model.predict(
            SCORE_PATH,
            aggregation_spec={"mode": mode},
            return_intermediates=True,
        )
        predictions, intermediates = result
        score = intermediates["score"]

        # Re-write the Delta Lake on the "none" pass (raw predictions)
        # so that labels match the current unified vocabulary.
        if mode == "none":
            from analysisgnn.storage.delta_writer import write_analysis_results

            print("(Re-)writing Delta Lake with updated vocabulary ...")
            write_analysis_results(
                output_dir=OUTPUT_DIR,
                score=score,
                note_array=intermediates["note_array"],
                predictions=predictions,
                data=intermediates["data"],
                task_dict=dict(model.hparams.task_dict),
                metadata={
                    "score_path": SCORE_PATH,
                    "full_checkpoint": CHECKPOINT_PATH,
                    "device": "cpu",
                },
            )
            print(f"Delta Lake written to {OUTPUT_DIR}")

        df = predictions_to_dataframe(
            score,
            predictions,
            tasks=ALL_TASKS,
            include_confidence=True,
            include_class_ids=False,
        )
        df = format_table_output(df, ALL_TASKS)

        # Add a row index column at the front
        if "row" not in df.columns:
            df.insert(0, "row", np.arange(len(df)))

        csv_path = os.path.join(OUTPUT_DIR, f"reference_{mode}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}  ({len(df)} rows, {len(df.columns)} columns)")

        # Print a few sample rows for verification
        print(f"\nFirst 3 rows of {mode}:")
        print(df.head(3).to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
