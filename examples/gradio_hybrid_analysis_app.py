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
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
import numpy as np
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


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FULL_CKPT = os.environ.get(
    "ANALYSISGNN_FULL_CKPT",
    str(REPO_ROOT / "artifacts" / "gradio_checkpoints" / "uocj8f6y_full_last.ckpt"),
)
DEFAULT_MASKED_CKPT = os.environ.get(
    "ANALYSISGNN_MASKED_CKPT",
    str(REPO_ROOT / "artifacts" / "gradio_checkpoints" / "t7pxcwri_masked_last.ckpt"),
)
DEFAULT_TASKS = ",".join(DEFAULT_EDITABLE_TASKS)
AVAILABLE_TASKS: Dict[str, str] = {
    "cadence": "Cadence Detection",
    "localkey": "Local Key",
    "tonkey": "Tonalized Key",
    "quality": "Chord Quality",
    "root": "Chord Root",
    "bass": "Bass Note",
    "inversion": "Chord Inversion",
    "degree1": "Primary Degree",
    "degree2": "Secondary Degree",
    "romanNumeral": "Roman Numeral Analysis",
    "phrase": "Phrase Segmentation",
    "section": "Section Detection",
    "hrhythm": "Harmonic Rhythm",
    "pcset": "Pitch-Class Set",
    "tpc_in_label": "Non-Chord Tone (NCT)",
    "note_degree": "Note Degree",
}
TASK_ALIASES: Dict[str, str] = {
    "hrythm": "hrhythm",
}

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


def _resolve_selected_tasks(task_labels: List[str], tasks_csv: str) -> List[str]:
    label_to_task = {v: k for k, v in AVAILABLE_TASKS.items()}
    if task_labels:
        tasks = [label_to_task[label] for label in task_labels if label in label_to_task]
    else:
        tasks = parse_task_csv(tasks_csv)
    normalized: List[str] = []
    for task in tasks:
        resolved = TASK_ALIASES.get(task, task)
        if resolved not in normalized:
            normalized.append(resolved)
    return normalized


def _convert_tpc_column_inplace(df: pd.DataFrame) -> None:
    if "tpc_in_label" not in df.columns:
        return
    numeric = pd.to_numeric(df["tpc_in_label"], errors="coerce")
    mapped = np.where(numeric.fillna(1).astype(int) == 0, "NCT", "Chord Tone")
    keep_original_mask = numeric.isna()
    if keep_original_mask.any():
        original = df.loc[keep_original_mask, "tpc_in_label"].astype(str)
        cleaned = original.str.strip()
        mapped = pd.Series(mapped, index=df.index, dtype=object)
        mapped.loc[keep_original_mask] = cleaned
        df["tpc_in_label"] = mapped.values
    else:
        df["tpc_in_label"] = mapped


def _format_table_output(df: pd.DataFrame, tasks: List[str]) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    if "note_id" not in out.columns:
        out.insert(0, "note_id", np.arange(len(out)))
    _convert_tpc_column_inplace(out)

    timing_cols = [
        col for col in ["row", "note_id", "onset_beat", "measure", "duration_beat", "pitch_spelling", "pitch_midi"]
        if col in out.columns
    ]
    prediction_cols = [task for task in tasks if task in out.columns]
    confidence_cols = [col for col in out.columns if col.endswith("_confidence")]

    ordered_cols: List[str] = timing_cols.copy()
    for pred_col in prediction_cols:
        ordered_cols.append(pred_col)
        conf_col = f"{pred_col}_confidence"
        if conf_col in confidence_cols:
            ordered_cols.append(conf_col)
    remaining_cols = [col for col in out.columns if col not in ordered_cols and not col.endswith("_id")]
    out = out[ordered_cols + remaining_cols]
    return out


def _apply_timing_from_predictions(df: pd.DataFrame, predictions: Dict[str, torch.Tensor]) -> pd.DataFrame:
    out = df.copy()
    onset = predictions.get("onset")
    if isinstance(onset, torch.Tensor) and onset.numel() == len(out):
        out["onset_beat"] = onset.detach().cpu().numpy()
    s_measure = predictions.get("s_measure")
    if isinstance(s_measure, torch.Tensor) and s_measure.numel() == len(out):
        out["measure"] = s_measure.detach().cpu().numpy()
    return out


def _build_iterative_spec(
    enable_iterative: bool,
    iterative_steps: int,
    keep_percentile_per_step: float,
    tasks: List[str],
    target_only_update: bool,
    zero_known_start: bool,
) -> Dict[str, Any]:
    return {
        "enabled": bool(enable_iterative),
        "steps": int(max(1, iterative_steps)),
        "keep_percentile_per_step": float(max(0.0, min(100.0, keep_percentile_per_step))),
        "masked_tasks": list(tasks),
        "mode": "cumulative",
        "freeze_confidence": "joint_mean",
        "target_only_update": bool(target_only_update),
        "min_remaining_targets": 0,
        "confidence_temperature": 1.0,
        "zero_known_start": bool(zero_known_start),
    }


def _format_trace(trace: Dict[str, Any], show_trace: bool) -> str:
    if not show_trace:
        return ""
    if not trace:
        return "{}"
    try:
        return json.dumps(trace, indent=2)
    except Exception:
        return str(trace)


def run_full_inference(
    score_file: Any,
    full_ckpt: str,
    masked_ckpt: str,
    device: str,
    task_labels: List[str],
    tasks_csv: str,
    enable_iterative: bool,
    iterative_steps: int,
    keep_percentile_per_step: float,
    show_trace: bool,
):
    try:
        score_path = _resolve_score_path(score_file)
        score = _load_score(score_path)
        tasks = _resolve_selected_tasks(task_labels, tasks_csv)
        predictor = _get_predictor(full_ckpt, masked_ckpt, device)

        iterative_spec = _build_iterative_spec(
            enable_iterative=enable_iterative,
            iterative_steps=iterative_steps,
            keep_percentile_per_step=keep_percentile_per_step,
            tasks=tasks,
            target_only_update=False,
            zero_known_start=True,
        )
        with torch.no_grad():
            output, routing = predictor.predict(
                score,
                force_route="full",
                iterative_spec=iterative_spec,
                return_iterative_trace=bool(enable_iterative),
                return_route=True,
            )
        if enable_iterative:
            predictions, trace = output
        else:
            predictions = output
            trace = {"enabled": False, "steps": []}

        full_df = predictions_to_dataframe(
            score=score,
            predictions=predictions,
            tasks=tasks,
            include_confidence=True,
            include_class_ids=False,
        )
        full_df = _apply_timing_from_predictions(full_df, predictions)
        display_df = _format_table_output(full_df, tasks)

        status = (
            f"Full inference done using route={routing.route} checkpoint={routing.checkpoint_path}. "
            f"Rows={len(display_df)} tasks={','.join(tasks)}"
        )
        return display_df, status, _format_trace(trace, show_trace)
    except Exception as exc:
        return pd.DataFrame(), f"Error: {exc}", ""


def run_partial_rerender(
    score_file: Any,
    full_ckpt: str,
    masked_ckpt: str,
    device: str,
    task_labels: List[str],
    tasks_csv: str,
    known_rows_expr: str,
    target_rows_expr: str,
    edited_table: Any,
    enable_iterative: bool,
    iterative_steps: int,
    keep_percentile_per_step: float,
    target_only_update: bool,
    show_trace: bool,
):
    try:
        score_path = _resolve_score_path(score_file)
        score = _load_score(score_path)
        tasks = _resolve_selected_tasks(task_labels, tasks_csv)
        predictor = _get_predictor(full_ckpt, masked_ckpt, device)

        edited_df = pd.DataFrame(edited_table) if edited_table is not None else pd.DataFrame()
        user_edits, masked_spec, info = build_mask_inputs_from_table_edits(
            edited_df=edited_df,
            masked_tasks=tasks,
            known_rows_expr=known_rows_expr,
            target_rows_expr=target_rows_expr,
        )

        iterative_spec = _build_iterative_spec(
            enable_iterative=enable_iterative,
            iterative_steps=iterative_steps,
            keep_percentile_per_step=keep_percentile_per_step,
            tasks=tasks,
            target_only_update=target_only_update,
            zero_known_start=False,
        )
        with torch.no_grad():
            output, routing = predictor.predict(
                score,
                user_edits=user_edits,
                masked_spec=masked_spec,
                iterative_spec=iterative_spec,
                return_iterative_trace=bool(enable_iterative),
                return_route=True,
            )
        if enable_iterative:
            predictions, trace = output
        else:
            predictions = output
            trace = {"enabled": False, "steps": []}

        out_df = predictions_to_dataframe(
            score=score,
            predictions=predictions,
            tasks=tasks,
            include_confidence=True,
            include_class_ids=False,
        )
        out_df = _apply_timing_from_predictions(out_df, predictions)
        display_df = _format_table_output(out_df, tasks)

        status = (
            f"Partial inference done using route={routing.route} checkpoint={routing.checkpoint_path}. "
            f"Known rows={info.get('num_known', 0)} target rows={info.get('num_targets', 0)}"
        )
        return display_df, status, _format_trace(trace, show_trace)
    except Exception as exc:
        return pd.DataFrame(), f"Error: {exc}", ""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="AnalysisGNN Hybrid Inference") as demo:
        gr.Markdown("""
# AnalysisGNN Hybrid Inference

Three explicit inference paths:
- Base model for full-piece prediction.
- Iterative refinement (no known labels at step 1) for apples-to-apples full-model benchmarking.
- Masked model for edit-conditioned partial re-prediction.

Index expressions for row selection are 1-based. Example: `1-8, 12, 20-24`.
""")

        with gr.Row():
            full_ckpt = gr.Textbox(label="Base (Full Inference) Checkpoint", value=DEFAULT_FULL_CKPT)
            masked_ckpt = gr.Textbox(label="Masked (Partial Inference) Checkpoint", value=DEFAULT_MASKED_CKPT)
            device = gr.Dropdown(label="Device", choices=["auto", "cuda", "cpu"], value="auto")

        with gr.Row():
            score_file = gr.File(label="MusicXML Score", file_types=[".xml", ".musicxml", ".mxl"], type="filepath")

        task_selector = gr.CheckboxGroup(
            choices=list(AVAILABLE_TASKS.values()),
            value=[AVAILABLE_TASKS[t] for t in DEFAULT_EDITABLE_TASKS if t in AVAILABLE_TASKS],
            label="Select Analysis Tasks",
            info="Choose which tasks to run and show in the editable table.",
        )
        tasks_csv = gr.Textbox(
            label="Tasks Override (internal keys CSV, optional)",
            value=DEFAULT_TASKS,
            info="Used only if no task is selected above. Example: romanNumeral,localkey,quality",
        )

        mode_selector = gr.Dropdown(
            label="Inference Mode",
            choices=["Iterative (no known labels)", "Full", "Edit-conditioned"],
            value="Iterative (no known labels)",
            info="Use Iterative for fair full-vs-iter benchmarks; use Edit-conditioned for user-corrected rerenders.",
        )
        run_btn = gr.Button("Run Inference", variant="primary")

        with gr.Row():
            enable_iterative = gr.Checkbox(
                label="Enable Iterative Refinement",
                value=False,
            )
            iterative_steps = gr.Number(
                label="Refinement Steps",
                value=10,
                precision=0,
            )
            keep_percentile_per_step = gr.Number(
                label="Keep Percentile/Step",
                value=10.0,
            )
        with gr.Row():
            target_only_update = gr.Checkbox(
                label="Target-only overwrite (partial mode)",
                value=True,
            )
            show_trace = gr.Checkbox(
                label="Show Iteration Trace",
                value=False,
            )

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
        trace_output = gr.Textbox(label="Iteration Trace", interactive=False, lines=12)

        def run_by_mode(
            mode: str,
            score_file: Any,
            full_ckpt: str,
            masked_ckpt: str,
            device: str,
            task_labels: List[str],
            tasks_csv: str,
            known_rows_expr: str,
            target_rows_expr: str,
            edited_table: Any,
            enable_iterative: bool,
            iterative_steps: int,
            keep_percentile_per_step: float,
            target_only_update: bool,
            show_trace: bool,
        ):
            mode_value = (mode or "").strip()
            if mode_value == "Full":
                return run_full_inference(
                    score_file=score_file,
                    full_ckpt=full_ckpt,
                    masked_ckpt=masked_ckpt,
                    device=device,
                    task_labels=task_labels,
                    tasks_csv=tasks_csv,
                    enable_iterative=False,
                    iterative_steps=iterative_steps,
                    keep_percentile_per_step=keep_percentile_per_step,
                    show_trace=show_trace,
                )
            if mode_value == "Iterative (no known labels)":
                return run_full_inference(
                    score_file=score_file,
                    full_ckpt=full_ckpt,
                    masked_ckpt=masked_ckpt,
                    device=device,
                    task_labels=task_labels,
                    tasks_csv=tasks_csv,
                    enable_iterative=True,
                    iterative_steps=iterative_steps,
                    keep_percentile_per_step=keep_percentile_per_step,
                    show_trace=show_trace,
                )
            return run_partial_rerender(
                score_file=score_file,
                full_ckpt=full_ckpt,
                masked_ckpt=masked_ckpt,
                device=device,
                task_labels=task_labels,
                tasks_csv=tasks_csv,
                known_rows_expr=known_rows_expr,
                target_rows_expr=target_rows_expr,
                edited_table=edited_table,
                enable_iterative=enable_iterative,
                iterative_steps=iterative_steps,
                keep_percentile_per_step=keep_percentile_per_step,
                target_only_update=target_only_update,
                show_trace=show_trace,
            )

        run_btn.click(
            fn=run_by_mode,
            inputs=[
                mode_selector,
                score_file,
                full_ckpt,
                masked_ckpt,
                device,
                task_selector,
                tasks_csv,
                known_rows_expr,
                target_rows_expr,
                table,
                enable_iterative,
                iterative_steps,
                keep_percentile_per_step,
                target_only_update,
                show_trace,
            ],
            outputs=[table, status, trace_output],
        )

    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch()
