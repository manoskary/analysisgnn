#!/usr/bin/env python3
"""Hybrid Gradio interface for AnalysisGNN.

This app supports two inference modes with separate checkpoints:
- full-piece prediction (base checkpoint)
- masked-conditioned partial re-prediction (masked checkpoint)

The workflow is designed for iterative editing:
1) Run full inference.
2) Edit labels in the table.
3) Mark known rows and optional target rows.
4) Re-predict with hard constraints from known labels.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
import pandas as pd
import partitura as pt
import torch

from analysisgnn.inference.hybrid_predictor import (
    DEFAULT_EDITABLE_TASKS,
    HybridAnalysisPredictor,
    build_mask_inputs_from_table_edits,
    parse_task_csv,
    predictions_to_dataframe,
)


DEFAULT_FULL_CKPT = os.environ.get("ANALYSISGNN_FULL_CKPT", "")
DEFAULT_MASKED_CKPT = os.environ.get("ANALYSISGNN_MASKED_CKPT", "")
DEFAULT_TASKS = ",".join(DEFAULT_EDITABLE_TASKS)

_PREDICTOR_CACHE: Dict[Tuple[str, str, str], HybridAnalysisPredictor] = {}


def _resolve_score_path(score_file: Any) -> str:
    if score_file is None:
        raise ValueError("Please upload a MusicXML file.")
    if isinstance(score_file, (str, os.PathLike)):
        score_path = str(score_file)
    elif isinstance(score_file, dict) and "name" in score_file:
        score_path = str(score_file["name"])
    else:
        score_path = str(getattr(score_file, "name", ""))

    if not score_path or not os.path.exists(score_path):
        raise ValueError("Uploaded score path is invalid.")
    return score_path


def _load_score(score_path: str):
    suffix = Path(score_path).suffix.lower()
    if suffix in {".xml", ".musicxml", ".mxl"}:
        return pt.load_score(score_path)
    return pt.load_score(score_path)


def _get_predictor(full_ckpt: str, masked_ckpt: str, device: str) -> HybridAnalysisPredictor:
    full_ckpt = (full_ckpt or "").strip()
    masked_ckpt = (masked_ckpt or "").strip()
    if not full_ckpt:
        raise ValueError("Base checkpoint path is required.")
    if not os.path.exists(full_ckpt):
        raise ValueError(f"Base checkpoint not found: {full_ckpt}")
    if masked_ckpt and not os.path.exists(masked_ckpt):
        raise ValueError(f"Masked checkpoint not found: {masked_ckpt}")

    key = (full_ckpt, masked_ckpt or full_ckpt, device)
    if key not in _PREDICTOR_CACHE:
        _PREDICTOR_CACHE[key] = HybridAnalysisPredictor(
            full_checkpoint_path=full_ckpt,
            masked_checkpoint_path=(masked_ckpt or full_ckpt),
            device=device,
        )
    return _PREDICTOR_CACHE[key]


def _select_display_columns(df: pd.DataFrame, tasks: List[str]) -> pd.DataFrame:
    base_cols = [c for c in ["row", "note_id", "measure", "onset_beat", "duration_beat", "pitch_spelling", "pitch_midi"] if c in df.columns]
    task_cols = [task for task in tasks if task in df.columns]
    cols = base_cols + task_cols
    return df[cols] if cols else df


def run_full_inference(
    score_file: Any,
    full_ckpt: str,
    masked_ckpt: str,
    device: str,
    tasks_csv: str,
):
    try:
        score_path = _resolve_score_path(score_file)
        score = _load_score(score_path)
        tasks = parse_task_csv(tasks_csv)
        predictor = _get_predictor(full_ckpt, masked_ckpt, device)

        with torch.no_grad():
            predictions, routing = predictor.predict(
                score,
                force_route="full",
                return_route=True,
            )

        full_df = predictions_to_dataframe(
            score=score,
            predictions=predictions,
            tasks=tasks,
            include_confidence=False,
            include_class_ids=False,
        )
        display_df = _select_display_columns(full_df, tasks)

        status = (
            f"Full inference done using route={routing.route} checkpoint={routing.checkpoint_path}. "
            f"Rows={len(display_df)} tasks={','.join(tasks)}"
        )
        return display_df, status
    except Exception as exc:
        return pd.DataFrame(), f"Error: {exc}"


def run_partial_rerender(
    score_file: Any,
    full_ckpt: str,
    masked_ckpt: str,
    device: str,
    tasks_csv: str,
    known_rows_expr: str,
    target_rows_expr: str,
    edited_table: Any,
):
    try:
        score_path = _resolve_score_path(score_file)
        score = _load_score(score_path)
        tasks = parse_task_csv(tasks_csv)
        predictor = _get_predictor(full_ckpt, masked_ckpt, device)

        edited_df = pd.DataFrame(edited_table) if edited_table is not None else pd.DataFrame()
        user_edits, masked_spec, info = build_mask_inputs_from_table_edits(
            edited_df=edited_df,
            masked_tasks=tasks,
            known_rows_expr=known_rows_expr,
            target_rows_expr=target_rows_expr,
        )

        with torch.no_grad():
            predictions, routing = predictor.predict(
                score,
                user_edits=user_edits,
                masked_spec=masked_spec,
                return_route=True,
            )

        out_df = predictions_to_dataframe(
            score=score,
            predictions=predictions,
            tasks=tasks,
            include_confidence=False,
            include_class_ids=False,
        )
        display_df = _select_display_columns(out_df, tasks)

        status = (
            f"Partial inference done using route={routing.route} checkpoint={routing.checkpoint_path}. "
            f"Known rows={info.get('num_known', 0)} target rows={info.get('num_targets', 0)}"
        )
        return display_df, status
    except Exception as exc:
        return pd.DataFrame(), f"Error: {exc}"


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="AnalysisGNN Hybrid Inference") as demo:
        gr.Markdown("""
# AnalysisGNN Hybrid Inference

Two-checkpoint workflow:
- Base model for full-piece prediction.
- Masked model for edit-conditioned partial re-prediction.

Index expressions for row selection are 1-based. Example: `1-8, 12, 20-24`.
""")

        with gr.Row():
            full_ckpt = gr.Textbox(label="Base (Full Inference) Checkpoint", value=DEFAULT_FULL_CKPT)
            masked_ckpt = gr.Textbox(label="Masked (Partial Inference) Checkpoint", value=DEFAULT_MASKED_CKPT)
            device = gr.Dropdown(label="Device", choices=["auto", "cuda", "cpu"], value="auto")

        with gr.Row():
            score_file = gr.File(label="MusicXML Score", file_types=[".xml", ".musicxml", ".mxl"], type="filepath")
            tasks_csv = gr.Textbox(
                label="Tasks (comma-separated)",
                value=DEFAULT_TASKS,
                info="Default editable tasks: romanNumeral, localkey, quality, inversion, degree1, degree2",
            )

        with gr.Row():
            run_full_btn = gr.Button("Run Full Inference", variant="primary")
            rerun_partial_btn = gr.Button("Re-predict From Edits", variant="secondary")

        with gr.Row():
            known_rows_expr = gr.Textbox(
                label="Known Rows (1-based)",
                value="",
                info="Rows treated as known/corrected labels (context).",
            )
            target_rows_expr = gr.Textbox(
                label="Target Rows (1-based, optional)",
                value="",
                info="Rows to re-predict. Empty means all non-known rows.",
            )

        table = gr.Dataframe(
            label="Predictions (editable)",
            interactive=True,
            wrap=True,
        )
        status = gr.Textbox(label="Status", interactive=False)

        run_full_btn.click(
            fn=run_full_inference,
            inputs=[score_file, full_ckpt, masked_ckpt, device, tasks_csv],
            outputs=[table, status],
        )

        rerun_partial_btn.click(
            fn=run_partial_rerender,
            inputs=[
                score_file,
                full_ckpt,
                masked_ckpt,
                device,
                tasks_csv,
                known_rows_expr,
                target_rows_expr,
                table,
            ],
            outputs=[table, status],
        )

    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch()
