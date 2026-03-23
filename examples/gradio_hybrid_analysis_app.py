#!/usr/bin/env python3
"""Hybrid Gradio interface for AnalysisGNN.

Three-module layout:
  Module 1 — Data Source: run inference (tab 1a) or load Delta Lake (tab 1b)
  Module 2 — Analysis Results: aggregation, CSV export, Verovio, Delta Lake save
  Module 3 — Edit-Conditioned Re-Inference (requires live model from Module 1a)

The workflow is designed for iterative editing:
1) Run full inference (or load from Delta Lake).
2) Optionally change aggregation strategy and re-aggregate.
3) Edit labels in the table.
4) Mark known rows and optional target rows.
5) Re-predict with hard constraints from known labels.
"""

from __future__ import annotations

import os
import json
import html as html_lib
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from analysisgnn.utils.chord_representations import format_table_output
from analysisgnn.utils.roman_decode import decode_roman_numeral

from analysisgnn.aggregation import get_strategy, list_strategies
from analysisgnn.storage.delta_writer import write_analysis_results
from analysisgnn.storage.delta_reader import (
    load_notes,
    load_edges,
    load_probabilities,
    load_hyperedges,
    load_metadata,
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
    "tonkey": "Tonicized Key",
    "quality": "Chord Quality",
    "root": "Chord Root",
    "bass": "Bass Note",
    "inversion": "Chord Inversion",
    "degree1": "Primary Degree",
    "degree2": "Secondary Degree",
    "romanNumeral": "Roman Numeral Analysis",
    "phrase": "Phrase Segmentation",
    "section": "Section Detection",
    "tpc_in_label": "Non-Chord Tone (NCT)",
    "note_degree": "Note Degree",
}
TASK_ALIASES: Dict[str, str] = {}

_PREDICTOR_CACHE: Dict[Tuple[str, str, str], HybridAnalysisPredictor] = {}
ASSETS_DIR = REPO_ROOT / "examples" / "assets"
DEFAULT_EDGE_TYPES = ["onset", "consecutive", "during", "rest"]
EDGE_LABELS = {
    "onset": "Onset",
    "consecutive": "Consecutive",
    "during": "During",
    "rest": "Rest",
}


# ---------------------------------------------------------------------------
# Pure helpers (unchanged from previous version)
# ---------------------------------------------------------------------------


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
    supported_tasks = set(AVAILABLE_TASKS.keys())
    if task_labels:
        tasks = [label_to_task[label] for label in task_labels if label in label_to_task]
    else:
        tasks = parse_task_csv(tasks_csv)
    normalized: List[str] = []
    for task in tasks:
        resolved = TASK_ALIASES.get(task, task)
        if resolved not in supported_tasks:
            continue
        if resolved not in normalized:
            normalized.append(resolved)
    if not normalized:
        normalized = [t for t in DEFAULT_EDITABLE_TASKS if t in supported_tasks]
    return normalized


def _apply_timing_from_predictions(df: pd.DataFrame, predictions: Dict[str, torch.Tensor]) -> pd.DataFrame:
    out = df.copy()
    onset = predictions.get("onset_beat")
    if isinstance(onset, torch.Tensor) and onset.numel() == len(out):
        out["onset_beat"] = onset.detach().cpu().numpy()
    measure = predictions.get("measure")
    if isinstance(measure, torch.Tensor) and measure.numel() == len(out):
        out["measure"] = measure.detach().cpu().numpy()
    return out


def _value_or_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _parse_inversion_value(value: Any) -> Any:
    val = _value_or_none(value)
    if val is None:
        return None
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, float):
        return int(val)
    text = str(val).strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except Exception:
        mapping = {
            "root": 0,
            "root position": 0,
            "6": 1,
            "63": 1,
            "first inversion": 1,
            "64": 2,
            "second inversion": 2,
            "65": 1,
            "43": 2,
            "2": 3,
            "42": 3,
            "third inversion": 3,
        }
        return mapping.get(text.lower(), None)


def _build_complete_rn_column(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) == 0:
        return pd.Series(dtype=object)
    out: List[str] = []
    required = ["degree1", "degree2", "inversion", "quality", "localkey"]
    missing = [k for k in required if k not in df.columns]
    if missing:
        return pd.Series([""] * len(df), index=df.index, dtype=object)
    for _, row in df.iterrows():
        d1 = _value_or_none(row.get("degree1"))
        d2 = _value_or_none(row.get("degree2"))
        inv = _parse_inversion_value(row.get("inversion"))
        quality = _value_or_none(row.get("quality"))
        localkey = _value_or_none(row.get("localkey"))
        if d1 is None or inv is None or quality is None or localkey is None:
            out.append("")
            continue
        try:
            rn = decode_roman_numeral(
                degree1=str(d1),
                degree2=str(d2) if d2 is not None else "None",
                inversion=inv,
                quality=str(quality),
                localkey=str(localkey),
                include_key=False,
            )
        except Exception:
            rn = ""
        out.append(rn)
    return pd.Series(out, index=df.index, dtype=object)


def _read_score_xml_text(score_path: str, score: pt.score.Score) -> str:
    suffix = Path(score_path).suffix.lower()
    if suffix in {".xml", ".musicxml"}:
        with open(score_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        pt.save_musicxml(score, tmp_path)
        with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _build_complete_rn_spans(df: pd.DataFrame) -> List[Tuple[int, int, str]]:
    """Build non-redundant Roman Numeral spans over onset_div."""
    if df is None or len(df) == 0:
        return []
    if "onset_div" not in df.columns:
        return []

    work = df.copy()
    if "romanNumeral_full" not in work.columns:
        work["romanNumeral_full"] = _build_complete_rn_column(work)
    if "duration_div" not in work.columns:
        return []

    work["onset_div"] = pd.to_numeric(work["onset_div"], errors="coerce")
    work["duration_div"] = pd.to_numeric(work["duration_div"], errors="coerce").fillna(0)
    work["romanNumeral_full"] = work["romanNumeral_full"].fillna("").astype(str).str.strip()
    work = work.dropna(subset=["onset_div"])
    if len(work) == 0:
        return []

    by_onset = (
        work.sort_values(["onset_div", "duration_div"])
        .groupby("onset_div", sort=True)
    )
    onset_points: List[int] = []
    onset_rn: List[str] = []
    for onset, group in by_onset:
        onset_i = int(onset)
        candidates = [v for v in group["romanNumeral_full"].tolist() if v]
        rn_value = candidates[0] if candidates else ""
        onset_points.append(onset_i)
        onset_rn.append(rn_value)
    if not onset_points:
        return []

    score_end = int(np.max(work["onset_div"].to_numpy() + np.maximum(1, work["duration_div"].to_numpy())))
    spans: List[Tuple[int, int, str]] = []
    current_rn = ""
    current_start: int | None = None
    for onset, rn in zip(onset_points, onset_rn):
        if rn == current_rn:
            continue
        if current_rn and current_start is not None and onset > current_start:
            spans.append((current_start, onset, current_rn))
        current_rn = rn
        current_start = onset if rn else None
    if current_rn and current_start is not None:
        final_end = max(current_start + 1, score_end)
        spans.append((current_start, final_end, current_rn))
    return spans


def _read_score_xml_with_complete_rn(
    score_path: str,
    score: pt.score.Score,
    df: pd.DataFrame,
) -> str:
    """Export MusicXML with RomanNumeral harmony spans inserted."""
    if df is None or len(df) == 0:
        return _read_score_xml_text(score_path, score)

    note_array = _sorted_note_array(score)
    n = min(len(df), len(note_array))
    if n == 0:
        return _read_score_xml_text(score_path, score)

    work = df.iloc[:n].reset_index(drop=True).copy()
    if "onset_div" in note_array.dtype.names:
        work["onset_div"] = note_array["onset_div"][:n]
    if "duration_div" in note_array.dtype.names:
        work["duration_div"] = note_array["duration_div"][:n]
    work["romanNumeral_full"] = _build_complete_rn_column(work)
    spans = _build_complete_rn_spans(work)
    if not spans:
        return _read_score_xml_text(score_path, score)

    try:
        score_for_xml = _load_score(score_path)
    except Exception:
        score_for_xml = score
    parts = list(getattr(score_for_xml, "parts", []) or [])
    if not parts:
        return _read_score_xml_text(score_path, score)

    harmony_classes = tuple(
        cls for cls in (pt.score.Harmony, pt.score.RomanNumeral, pt.score.ChordSymbol) if cls is not None
    )
    for part in parts:
        for cls in harmony_classes:
            try:
                to_remove = list(part.iter_all(cls))
            except Exception:
                to_remove = []
            for obj in to_remove:
                try:
                    part.remove(obj)
                except Exception:
                    pass

    target_part = parts[0]
    for start_div, end_div, rn_text in spans:
        try:
            rn_obj = pt.score.RomanNumeral(text=str(rn_text))
            target_part.add(rn_obj, start=int(start_div), end=int(end_div))
        except Exception:
            continue

    with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        pt.save_musicxml(score_for_xml, tmp_path)
        with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _sorted_note_array(score: pt.score.Score) -> np.ndarray:
    note_array_raw = score.note_array(
        include_time_signature=True,
        include_pitch_spelling=True,
        include_key_signature=True,
        include_staff=True,
        include_metrical_position=True,
    )
    sort_idx = np.argsort(note_array_raw, order=["onset_div", "pitch"])
    return note_array_raw[sort_idx]


# ---------------------------------------------------------------------------
# Edge extraction from intermediates (replaces _extract_graph_edges_from_score)
# ---------------------------------------------------------------------------

_EDGE_KEY_MAP = {
    "onset": ("note", "onset", "note"),
    "consecutive": ("note", "consecutive", "note"),
    "during": ("note", "during", "note"),
    "rest": ("note", "rest", "note"),
}


def _edges_from_pyg_data(data: Any, num_notes: int) -> Dict[str, List[List[int]]]:
    """Extract edge lists from a PyG HeteroData ``edge_index_dict``.

    Returns a dict mapping edge type names to ``[src_list, dst_list]``.
    """
    edge_index_dict = getattr(data, "edge_index_dict", {})
    edges: Dict[str, List[List[int]]] = {}
    for edge_type, key in _EDGE_KEY_MAP.items():
        if key not in edge_index_dict:
            edges[edge_type] = [[], []]
            continue
        edge_index = edge_index_dict[key]
        if isinstance(edge_index, torch.Tensor):
            src = edge_index[0].detach().cpu().numpy()
            dst = edge_index[1].detach().cpu().numpy()
        else:
            src = np.asarray(edge_index[0])
            dst = np.asarray(edge_index[1])
        valid = (src >= 0) & (src < num_notes) & (dst >= 0) & (dst < num_notes)
        src = src[valid].astype(int).tolist()
        dst = dst[valid].astype(int).tolist()
        edges[edge_type] = [src, dst]
    return edges


def _edges_from_delta_lake_df(edges_df: pd.DataFrame, note_id_to_idx: Dict[str, int]) -> Dict[str, List[List[int]]]:
    """Convert a Delta Lake edges DataFrame to the edge-list format used by Verovio.

    ``note_id_to_idx`` maps note_id strings to integer indices.
    """
    edges: Dict[str, List[List[int]]] = {}
    for etype in DEFAULT_EDGE_TYPES:
        sub = edges_df[edges_df["edge_type"] == etype] if len(edges_df) > 0 else edges_df
        src_ids = sub["src"].tolist() if len(sub) > 0 else []
        dst_ids = sub["dst"].tolist() if len(sub) > 0 else []
        src_idx = [note_id_to_idx[s] for s in src_ids if s in note_id_to_idx]
        dst_idx = [note_id_to_idx[d] for d in dst_ids if d in note_id_to_idx]
        # Ensure same length after filtering
        min_len = min(len(src_idx), len(dst_idx))
        edges[etype] = [src_idx[:min_len], dst_idx[:min_len]]
    return edges


# ---------------------------------------------------------------------------
# Graph overlay payload builder (uses pre-computed edges)
# ---------------------------------------------------------------------------


def _build_graph_overlay_payload(
    df: pd.DataFrame,
    note_array: np.ndarray,
    tasks: List[str],
    edge_types: List[str],
    edges_all: Dict[str, List[List[int]]],
) -> Dict[str, Any]:
    """Build the Verovio overlay payload.

    ``edges_all`` must be pre-computed via ``_edges_from_pyg_data`` or
    ``_edges_from_delta_lake_df`` — this function does NOT call
    ``_extract_graph_edges_from_score`` any more.
    """
    n = min(len(df), len(note_array))
    data = df.iloc[:n].reset_index(drop=True).copy()
    rn_full = _build_complete_rn_column(data)
    spans_df = data.copy()
    if "onset_div" in note_array.dtype.names:
        spans_df["onset_div"] = note_array["onset_div"][:n]
    if "duration_div" in note_array.dtype.names:
        spans_df["duration_div"] = note_array["duration_div"][:n]
    spans_df["romanNumeral_full"] = rn_full
    rn_spans = _build_complete_rn_spans(spans_df)

    notes_payload: List[Dict[str, Any]] = []
    for idx in range(n):
        row = data.iloc[idx]
        conf: Dict[str, float] = {}
        task_vals: Dict[str, Any] = {}
        for task in tasks:
            if task in row.index:
                task_vals[task] = _value_or_none(row.get(task))
            conf_col = f"{task}_confidence"
            if conf_col in row.index:
                conf_val = _value_or_none(row.get(conf_col))
                if conf_val is not None:
                    try:
                        conf[task] = float(conf_val)
                    except Exception:
                        pass

        note_id = _value_or_none(row.get("note_id"))
        score_note_id = _value_or_none(note_array["id"][idx]) if "id" in note_array.dtype.names else None
        notes_payload.append(
            {
                "index": idx,
                "row": int(_value_or_none(row.get("row")) if "row" in row.index else idx),
                "note_id": str(score_note_id) if score_note_id is not None else (str(note_id) if note_id is not None else None),
                "table_note_id": str(note_id) if note_id is not None else None,
                "onset_div": int(note_array["onset_div"][idx]) if "onset_div" in note_array.dtype.names else None,
                "onset_beat": float(_value_or_none(row.get("onset_beat")) or 0.0),
                "measure": int(_value_or_none(row.get("measure"))) if _value_or_none(row.get("measure")) is not None else None,
                "duration_beat": float(_value_or_none(row.get("duration_beat")) or 0.0),
                "pitch_midi": int(_value_or_none(row.get("pitch_midi"))) if _value_or_none(row.get("pitch_midi")) is not None else None,
                "pitch_spelling": str(_value_or_none(row.get("pitch_spelling")) or ""),
                "tasks": task_vals,
                "confidence": conf,
                "romanNumeral_full": str(rn_full.iloc[idx]) if idx < len(rn_full) else "",
            }
        )

    visible = [et for et in edge_types if et in DEFAULT_EDGE_TYPES]
    payload: Dict[str, Any] = {
        "notes": notes_payload,
        "edges": {k: edges_all.get(k, [[], []]) for k in DEFAULT_EDGE_TYPES},
        "meta": {
            "selected_tasks": list(tasks),
            "visible_edge_types": visible,
            "edge_warning": "",
            "roman_spans": [
                {
                    "start_onset_div": int(s),
                    "end_onset_div": int(e),
                    "label": str(lbl),
                }
                for s, e, lbl in rn_spans
                if str(lbl).strip()
            ],
        },
    }
    return payload


@lru_cache(maxsize=1)
def _load_visual_assets() -> Tuple[str, str, str]:
    template = (ASSETS_DIR / "verovio_score_graph.html").read_text(encoding="utf-8")
    script = (ASSETS_DIR / "verovio_score_graph.js").read_text(encoding="utf-8")
    style = (ASSETS_DIR / "verovio_score_graph.css").read_text(encoding="utf-8")
    return template, script, style


def _build_verovio_html(payload: Dict[str, Any]) -> str:
    template, script, style = _load_visual_assets()
    payload_json = json.dumps(payload, ensure_ascii=False)
    doc = template.replace("__AGN_CSS__", style)
    doc = doc.replace("__AGN_JS__", script)
    doc = doc.replace("__AGN_PAYLOAD_JSON__", payload_json)
    srcdoc = html_lib.escape(doc, quote=True)
    return (
        "<iframe "
        "style='width:100%;height:980px;border:1px solid #d1d5db;border-radius:10px;background:white;' "
        f"srcdoc=\"{srcdoc}\"></iframe>"
    )


def _build_visual_payload(
    score_path: str,
    score: pt.score.Score,
    df: pd.DataFrame,
    tasks: List[str],
    edge_types: List[str],
    edges_all: Dict[str, List[List[int]]],
) -> Dict[str, Any]:
    note_array = _sorted_note_array(score)
    payload = _build_graph_overlay_payload(
        df=df, note_array=note_array, tasks=tasks, edge_types=edge_types, edges_all=edges_all,
    )
    payload["score_xml"] = _read_score_xml_with_complete_rn(
        score_path=score_path,
        score=score,
        df=df,
    )
    payload["score_format"] = "musicxml"
    return payload


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


# Voter-related aggregation spec builder — commented out (voter not yet
# implemented in analysisgnn/aggregation/).
# def _build_aggregation_spec(aggregation_mode: str, voter_path: str) -> Tuple[Dict[str, Any], str]:
#     mode_raw = str(aggregation_mode or "Mean").strip().lower()
#     mode = "voter" if mode_raw == "voter" else "mean"
#     path = (voter_path or "").strip()
#     if mode == "voter" and not path:
#         return {"mode": "mean"}, "Aggregation mode 'Voter' selected without checkpoint; falling back to mean."
#     spec: Dict[str, Any] = {"mode": mode}
#     if mode == "voter":
#         spec["voter_path"] = path
#     return spec, ""


def _format_trace(trace: Dict[str, Any], show_trace: bool) -> str:
    if not show_trace:
        return ""
    if not trace:
        return "{}"
    try:
        return json.dumps(trace, indent=2)
    except Exception:
        return str(trace)


def _derive_score_id(score_path: str) -> str:
    """Derive a filesystem-safe score ID from a score path."""
    return Path(score_path).stem


def _derive_output_dir(score_path: str) -> str:
    """Derive the Delta Lake output directory from a score path."""
    return str(REPO_ROOT / "outputs" / _derive_score_id(score_path))


# ---------------------------------------------------------------------------
# Module 1a: Run Inference
# ---------------------------------------------------------------------------


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
    """Run inference with aggregation_spec={"mode": "none"} and return_intermediates=True.

    Returns:
        (display_df, status, trace_str, visual_payload,
         raw_predictions_state, intermediates_state, tasks_state,
         score_path_state, edges_state, model_available_flag)
    """
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

        # Always run with no aggregation — aggregation is post-hoc in Module 2
        aggregation_spec = {"mode": "none"}

        with torch.no_grad():
            result = predictor.predict(
                score,
                force_route="full",
                iterative_spec=iterative_spec,
                aggregation_spec=aggregation_spec,
                return_iterative_trace=bool(enable_iterative),
                return_route=True,
                return_intermediates=True,
            )

        # Unpack: model.predict returns (predictions, [trace], intermediates)
        # then HybridAnalysisPredictor wraps as (output, routing)
        output, routing = result
        # output is a tuple because we asked for return_intermediates (and
        # optionally return_iterative_trace).
        if isinstance(output, tuple):
            parts = list(output)
        else:
            parts = [output]

        # The order of extras in model.predict():
        #   predictions, [iterative_trace if requested], [intermediates if requested]
        predictions = parts[0]
        trace = {"enabled": False, "steps": []}
        intermediates = {}
        if enable_iterative:
            if len(parts) >= 3:
                trace = parts[1]
                intermediates = parts[2]
            elif len(parts) == 2:
                trace = parts[1]
        else:
            if len(parts) >= 2:
                intermediates = parts[-1]

        score_obj = intermediates.get("score", score)
        note_array = intermediates.get("note_array", _sorted_note_array(score))
        pyg_data = intermediates.get("data", None)

        # Build display DataFrame
        full_df = predictions_to_dataframe(
            score=score_obj,
            predictions=predictions,
            tasks=tasks,
            include_confidence=True,
            include_class_ids=False,
        )
        full_df = _apply_timing_from_predictions(full_df, predictions)
        display_df = format_table_output(full_df, tasks)

        # Extract edges from intermediates
        num_notes = len(note_array)
        if pyg_data is not None:
            edges_all = _edges_from_pyg_data(pyg_data, num_notes)
        else:
            edges_all = {k: [[], []] for k in DEFAULT_EDGE_TYPES}

        # Build visual payload
        visual_payload = _build_visual_payload(
            score_path=score_path,
            score=score_obj,
            df=display_df,
            tasks=tasks,
            edge_types=[],
            edges_all=edges_all,
        )

        # Write Delta Lake (guard: only if it doesn't already exist)
        output_dir = _derive_output_dir(score_path)
        dl_status = ""
        if not os.path.isdir(os.path.join(output_dir, "notes", "_delta_log")):
            try:
                # Get task_dict from the model
                model = predictor.get_full_model()
                task_dict = dict(model.task_dict)
                write_analysis_results(
                    output_dir=output_dir,
                    score=score_obj,
                    note_array=note_array,
                    predictions=predictions,
                    data=pyg_data,
                    task_dict=task_dict,
                    metadata={
                        "full_checkpoint": full_ckpt,
                        "masked_checkpoint": masked_ckpt,
                        "device": device,
                        "score_path": score_path,
                    },
                )
                dl_status = f" Delta Lake written to {output_dir}."
            except Exception as dl_exc:
                dl_status = f" Delta Lake write failed: {dl_exc}"
        else:
            dl_status = f" Delta Lake already exists at {output_dir} (not overwritten)."

        status = (
            f"Inference done (route={routing.route}). "
            f"Rows={len(display_df)} tasks={','.join(tasks)} aggregation=none.{dl_status}"
        )

        # State objects to pass downstream
        raw_predictions_state = predictions
        intermediates_state = {
            "score": score_obj,
            "note_array": note_array,
            "data": pyg_data,
            "score_path": score_path,
        }

        return (
            display_df,
            status,
            _format_trace(trace, show_trace),
            visual_payload,
            raw_predictions_state,
            intermediates_state,
            tasks,
            score_path,
            edges_all,
            True,  # model_available
        )
    except Exception as exc:
        return (
            pd.DataFrame(),
            f"Error: {exc}",
            "",
            {},
            {},
            {},
            [],
            "",
            {k: [[], []] for k in DEFAULT_EDGE_TYPES},
            False,
        )


# ---------------------------------------------------------------------------
# Module 1b: Load Delta Lake
# ---------------------------------------------------------------------------


def load_from_delta_lake(metadata_file: Any) -> tuple:
    """Load a Delta Lake output dir from a metadata.json file selection.

    Returns the same shape of outputs as ``run_full_inference`` for seamless
    integration with the same gr.State objects.
    """
    try:
        if metadata_file is None:
            raise ValueError("Please select a metadata.json file.")
        if isinstance(metadata_file, (str, os.PathLike)):
            meta_path = str(metadata_file)
        elif isinstance(metadata_file, dict) and "name" in metadata_file:
            meta_path = str(metadata_file["name"])
        else:
            meta_path = str(getattr(metadata_file, "name", ""))

        if not meta_path or not os.path.exists(meta_path):
            raise ValueError("metadata.json path is invalid.")

        output_dir = str(Path(meta_path).parent)

        notes_df = load_notes(output_dir)
        probs_df = load_probabilities(output_dir)
        hyperedges_df = load_hyperedges(output_dir)
        edges_df = load_edges(output_dir)
        metadata = load_metadata(output_dir)

        tasks = list(metadata.get("task_dict", {}).keys())
        score_path = metadata.get("score_path", "")

        # Build note_id -> index mapping for edge conversion
        note_id_to_idx = {nid: i for i, nid in enumerate(notes_df["note_id"])}
        edges_all = _edges_from_delta_lake_df(edges_df, note_id_to_idx)

        # Build "none" aggregation: argmax summary
        strategy = get_strategy("none")
        display_df = strategy.aggregate(probs_df, notes_df, hyperedges_df, metadata, tasks=tasks)

        # Apply format_table_output
        display_df = format_table_output(display_df, tasks)

        status = (
            f"Loaded Delta Lake from {output_dir}. "
            f"Rows={len(display_df)} tasks={','.join(tasks)}."
        )

        # Build a minimal note_array-like structure for Verovio from notes_df
        # (We don't have the original partitura score, so Verovio won't render
        # the score — but the table and data are available.)
        # Store the probs_df + notes_df + hyperedges_df in intermediates for
        # post-hoc aggregation.
        intermediates_state = {
            "score": None,
            "note_array": None,
            "data": None,
            "score_path": score_path,
            "delta_lake_dir": output_dir,
            "notes_df": notes_df,
            "probs_df": probs_df,
            "hyperedges_df": hyperedges_df,
            "metadata": metadata,
        }

        return (
            display_df,
            status,
            "",  # trace
            {},  # visual_payload (no score XML available)
            {},  # raw_predictions (not available from Delta Lake)
            intermediates_state,
            tasks,
            score_path,
            edges_all,
            False,  # model NOT available
        )
    except Exception as exc:
        return (
            pd.DataFrame(),
            f"Error loading Delta Lake: {exc}",
            "",
            {},
            {},
            {},
            [],
            "",
            {k: [[], []] for k in DEFAULT_EDGE_TYPES},
            False,
        )


# ---------------------------------------------------------------------------
# Module 2: Post-hoc Aggregation
# ---------------------------------------------------------------------------


def run_aggregation(
    strategy_name: str,
    raw_predictions_state: Any,
    intermediates_state: Any,
    tasks_state: Any,
    score_path_state: str,
    edges_state: Any,
):
    """Apply an aggregation strategy to raw predictions and update the table.

    Returns: (display_df, status, visual_payload)
    """
    try:
        strategy_name = (strategy_name or "none").strip().lower()
        tasks = tasks_state or []
        intermediates = intermediates_state or {}

        # Determine data source: Delta Lake loaded data or raw predictions
        if "probs_df" in intermediates:
            # Loaded from Delta Lake
            probs_df = intermediates["probs_df"]
            notes_df = intermediates["notes_df"]
            hyperedges_df = intermediates["hyperedges_df"]
            metadata = intermediates.get("metadata", {})
        elif raw_predictions_state and isinstance(raw_predictions_state, dict):
            # From live inference — need to convert raw predictions to long-format
            from analysisgnn.storage.delta_writer import _build_probabilities_table, _build_notes_table, _build_hyperedges_table
            score_obj = intermediates.get("score")
            note_array = intermediates.get("note_array")
            pyg_data = intermediates.get("data")

            if score_obj is None or note_array is None:
                raise ValueError("No raw data available for aggregation.")

            # Build task_dict from predictions
            task_dict = {}
            for task_name, tensor in raw_predictions_state.items():
                if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
                    task_dict[task_name] = tensor.shape[1]

            # Build note_ids
            n = len(note_array)
            if "id" in note_array.dtype.names:
                note_ids = np.array([str(x) for x in note_array["id"]], dtype=object)
            else:
                note_ids = np.array([f"note_{i}" for i in range(n)], dtype=object)

            # Build DataFrames using delta_writer's internal builders
            import pyarrow as pa
            probs_table = _build_probabilities_table(raw_predictions_state, task_dict, note_ids)
            probs_df = probs_table.to_pandas()

            notes_table = _build_notes_table(note_array, score_obj)
            notes_df = notes_table.to_pandas()

            hyperedges_table, _ = _build_hyperedges_table(pyg_data, note_ids)
            hyperedges_df = hyperedges_table.to_pandas()

            metadata = {}
        else:
            raise ValueError("No predictions available. Run inference or load Delta Lake first.")

        # Apply aggregation
        strategy = get_strategy(strategy_name)
        result_df = strategy.aggregate(probs_df, notes_df, hyperedges_df, metadata, tasks=tasks)
        display_df = format_table_output(result_df, tasks)

        # Build visual payload if score is available
        score_obj = intermediates.get("score")
        note_array = intermediates.get("note_array")
        score_path = score_path_state or intermediates.get("score_path", "")
        edges_all = edges_state or {k: [[], []] for k in DEFAULT_EDGE_TYPES}
        visual_payload = {}
        if score_obj is not None and note_array is not None and score_path:
            visual_payload = _build_visual_payload(
                score_path=score_path,
                score=score_obj,
                df=display_df,
                tasks=tasks,
                edge_types=[],
                edges_all=edges_all,
            )

        status = f"Aggregation '{strategy_name}' applied. Rows={len(display_df)}."
        return display_df, status, visual_payload
    except Exception as exc:
        return pd.DataFrame(), f"Aggregation error: {exc}", {}


def export_csv(table_data: Any, score_path_state: str):
    """Export the current table to a CSV file and return it for download."""
    try:
        df = pd.DataFrame(table_data) if table_data is not None else pd.DataFrame()
        if len(df) == 0:
            raise ValueError("No data to export.")

        score_id = _derive_score_id(score_path_state) if score_path_state else "export"
        csv_path = os.path.join(tempfile.gettempdir(), f"{score_id}_analysis.csv")
        df.to_csv(csv_path, index=False)
        return csv_path, f"CSV exported to {csv_path}"
    except Exception as exc:
        return None, f"Export error: {exc}"


def save_delta_lake(
    raw_predictions_state: Any,
    intermediates_state: Any,
    tasks_state: Any,
    score_path_state: str,
):
    """Write/update Delta Lake with current data."""
    try:
        intermediates = intermediates_state or {}
        score_obj = intermediates.get("score")
        note_array = intermediates.get("note_array")
        pyg_data = intermediates.get("data")
        score_path = score_path_state or intermediates.get("score_path", "")

        if score_obj is None or note_array is None or pyg_data is None:
            # If loaded from Delta Lake, data already exists
            dl_dir = intermediates.get("delta_lake_dir", "")
            if dl_dir:
                return f"Data was loaded from Delta Lake at {dl_dir}. No new data to write."
            raise ValueError("No inference data available to save.")

        if not score_path:
            raise ValueError("No score path available.")

        output_dir = _derive_output_dir(score_path)

        # Get task_dict from predictions
        predictions = raw_predictions_state
        if not predictions or not isinstance(predictions, dict):
            raise ValueError("No raw predictions available.")

        task_dict = {}
        for task_name, tensor in predictions.items():
            if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
                task_dict[task_name] = tensor.shape[1]

        write_analysis_results(
            output_dir=output_dir,
            score=score_obj,
            note_array=note_array,
            predictions=predictions,
            data=pyg_data,
            task_dict=task_dict,
            metadata={
                "score_path": score_path,
            },
        )
        return f"Delta Lake saved to {output_dir}."
    except Exception as exc:
        return f"Save error: {exc}"


# ---------------------------------------------------------------------------
# Module 3: Edit-Conditioned Re-Inference
# ---------------------------------------------------------------------------


def run_edit_conditioned(
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
    intermediates_state: Any,
    edges_state: Any,
):
    """Run edit-conditioned masked inference using edits from Module 2 table.

    Returns: (display_df, status, trace_str, visual_payload,
              raw_predictions_state, intermediates_state, tasks_state,
              score_path_state, edges_state, model_available_flag)
    """
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

        # Always run with no aggregation
        aggregation_spec = {"mode": "none"}

        with torch.no_grad():
            result = predictor.predict(
                score,
                user_edits=user_edits,
                masked_spec=masked_spec,
                iterative_spec=iterative_spec,
                aggregation_spec=aggregation_spec,
                return_iterative_trace=bool(enable_iterative),
                return_route=True,
                return_intermediates=True,
            )

        output, routing = result
        if isinstance(output, tuple):
            parts = list(output)
        else:
            parts = [output]

        predictions = parts[0]
        trace = {"enabled": False, "steps": []}
        intermediates = {}
        if enable_iterative:
            if len(parts) >= 3:
                trace = parts[1]
                intermediates = parts[2]
            elif len(parts) == 2:
                trace = parts[1]
        else:
            if len(parts) >= 2:
                intermediates = parts[-1]

        score_obj = intermediates.get("score", score)
        note_array = intermediates.get("note_array", _sorted_note_array(score))
        pyg_data = intermediates.get("data", None)

        out_df = predictions_to_dataframe(
            score=score_obj,
            predictions=predictions,
            tasks=tasks,
            include_confidence=True,
            include_class_ids=False,
        )
        out_df = _apply_timing_from_predictions(out_df, predictions)
        display_df = format_table_output(out_df, tasks)

        # Use edges from intermediates
        num_notes = len(note_array)
        if pyg_data is not None:
            edges_all = _edges_from_pyg_data(pyg_data, num_notes)
        else:
            edges_all = edges_state or {k: [[], []] for k in DEFAULT_EDGE_TYPES}

        visual_payload = _build_visual_payload(
            score_path=score_path,
            score=score_obj,
            df=display_df,
            tasks=tasks,
            edge_types=[],
            edges_all=edges_all,
        )

        status = (
            f"Edit-conditioned inference done (route={routing.route}). "
            f"Known rows={info.get('num_known', 0)} target rows={info.get('num_targets', 0)} "
            f"aggregation=none."
        )

        new_intermediates = {
            "score": score_obj,
            "note_array": note_array,
            "data": pyg_data,
            "score_path": score_path,
        }

        return (
            display_df,
            status,
            _format_trace(trace, show_trace),
            visual_payload,
            predictions,
            new_intermediates,
            tasks,
            score_path,
            edges_all,
            True,
        )
    except Exception as exc:
        return (
            pd.DataFrame(),
            f"Error: {exc}",
            "",
            {},
            {},
            intermediates_state or {},
            [],
            "",
            edges_state or {k: [[], []] for k in DEFAULT_EDGE_TYPES},
            False,
        )


# ---------------------------------------------------------------------------
# Verovio Visual Tab
# ---------------------------------------------------------------------------


def refresh_visual_tab(
    score_file: Any,
    task_labels: List[str],
    tasks_csv: str,
    table_data: Any,
    edge_type_labels: List[str],
    visual_state: Dict[str, Any],
    intermediates_state: Any,
    edges_state: Any,
):
    try:
        selected_edge_types = [k for k, label in EDGE_LABELS.items() if label in (edge_type_labels or [])]
        tasks = _resolve_selected_tasks(task_labels, tasks_csv)
        intermediates = intermediates_state or {}
        score_obj = intermediates.get("score")
        note_array = intermediates.get("note_array")
        score_path = intermediates.get("score_path", "")
        edges_all = edges_state or {k: [[], []] for k in DEFAULT_EDGE_TYPES}

        # Fall back to loading score from file if not in intermediates
        if score_obj is None and score_file is not None:
            score_path = _resolve_score_path(score_file)
            score_obj = _load_score(score_path)
            note_array = _sorted_note_array(score_obj)

        df = pd.DataFrame(table_data) if table_data is not None else pd.DataFrame()

        if score_obj is not None and note_array is not None and len(df) > 0:
            payload = _build_visual_payload(
                score_path=score_path,
                score=score_obj,
                df=df,
                tasks=tasks,
                edge_types=selected_edge_types,
                edges_all=edges_all,
            )
        elif isinstance(visual_state, dict) and visual_state:
            payload = dict(visual_state)
            payload.setdefault("meta", {})
            payload["meta"]["visible_edge_types"] = selected_edge_types
        else:
            raise ValueError("No predictions available yet. Run inference first to populate the visual tab.")

        html_frame = _build_verovio_html(payload)
        note_count = len(payload.get("notes", []))
        status = (
            f"Visual refreshed: notes={note_count}, "
            f"visible edges={','.join(selected_edge_types) if selected_edge_types else 'none'}."
        )
        return html_frame, status, payload
    except Exception as exc:
        fallback = (
            "<div style='padding:12px;border:1px solid #d1d5db;border-radius:10px;background:#fff;'>"
            f"Visual rendering error: {html_lib.escape(str(exc))}"
            "</div>"
        )
        return fallback, f"Visual error: {exc}", visual_state if isinstance(visual_state, dict) else {}


# ---------------------------------------------------------------------------
# UI Builder
# ---------------------------------------------------------------------------


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="AnalysisGNN Hybrid Inference") as demo:
        gr.Markdown("""
# AnalysisGNN Hybrid Inference

**Module 1** — Data source: run inference on a score or load existing Delta Lake results.
**Module 2** — Analysis results: view, aggregate, export, and visualise.
**Module 3** — Edit-conditioned re-inference (requires live model from Module 1a).

Index expressions for row selection are 1-based. Example: `1-8, 12, 20-24`.
""")

        # ---- gr.State objects ----
        visual_payload_state = gr.State({})
        raw_predictions_state = gr.State({})
        intermediates_state = gr.State({})
        tasks_state = gr.State([])
        score_path_state = gr.State("")
        edges_state = gr.State({k: [[], []] for k in DEFAULT_EDGE_TYPES})
        model_available_state = gr.State(False)

        # ==================================================================
        # MODULE 1: DATA SOURCE
        # ==================================================================
        gr.Markdown("---")
        gr.Markdown("## Module 1: Data Source")

        with gr.Tabs():
            # ------ Tab 1a: Analyse Score ------
            with gr.Tab("Analyse Score"):
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
                    info="Choose which tasks to run and show in the editable table and visual tab.",
                )
                tasks_csv = gr.Textbox(
                    label="Tasks Override (internal keys CSV, optional)",
                    value=DEFAULT_TASKS,
                    info="Used only if no task is selected above. Example: romanNumeral,localkey,quality",
                )

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

                run_inference_btn = gr.Button("Run Inference", variant="primary")

            # ------ Tab 1b: Load Delta Lake ------
            with gr.Tab("Load Delta Lake"):
                gr.Markdown("Select the `metadata.json` file from an existing Delta Lake output directory.")
                delta_lake_file = gr.File(
                    label="Select metadata.json",
                    file_types=[".json"],
                    type="filepath",
                )
                load_delta_btn = gr.Button("Load", variant="primary")

        # ==================================================================
        # MODULE 2: ANALYSIS RESULTS
        # ==================================================================
        gr.Markdown("---")
        gr.Markdown("## Module 2: Analysis Results")

        with gr.Row():
            aggregation_dropdown = gr.Dropdown(
                label="Aggregation Strategy",
                choices=[s.capitalize() for s in list_strategies()],
                value="None",
                info="Select an aggregation strategy and click 'Aggregate!' to apply.",
            )
            aggregate_btn = gr.Button("Aggregate!", variant="secondary")
            export_csv_btn = gr.Button("Export CSV", variant="secondary")
            save_delta_btn = gr.Button("Save Delta Lake", variant="secondary")

        csv_download = gr.File(label="Download CSV", interactive=False, visible=True)
        save_delta_status = gr.Textbox(label="Save Status", interactive=False, visible=True)

        with gr.Tabs():
            # ------ Tab: Analysis Results ------
            with gr.Tab("Analysis Results"):
                table = gr.Dataframe(
                    label="Predictions (editable)",
                    interactive=True,
                    wrap=True,
                )
                status = gr.Textbox(label="Status", interactive=False)

            # ------ Tab: Verovio Visual Score ------
            with gr.Tab("Verovio Visual Score"):
                gr.Markdown(
                    "Render the uploaded score with graph overlays. "
                    "Click a note in the score to inspect note-level predictions and complete RN decoding."
                )
                visual_edge_types = gr.CheckboxGroup(
                    label="Visible Edge Types",
                    choices=[EDGE_LABELS[k] for k in DEFAULT_EDGE_TYPES],
                    value=[],
                    info="Edges are hidden by default; select one or more types and refresh.",
                )
                refresh_visual_btn = gr.Button("Refresh Visual", variant="secondary")
                visual_html = gr.HTML(
                    value=(
                        "<div style='padding:12px;border:1px solid #d1d5db;border-radius:10px;background:#fff;'>"
                        "Run inference first, then click 'Refresh Visual'."
                        "</div>"
                    ),
                    label="Verovio Score + Graph",
                )
                visual_status = gr.Textbox(label="Visual Status", interactive=False)

        # ==================================================================
        # MODULE 3: EDIT-CONDITIONED RE-INFERENCE
        # ==================================================================
        gr.Markdown("---")
        gr.Markdown("## Module 3: Edit-Conditioned Re-Inference")
        gr.Markdown("*Requires a live model (run inference in Module 1a first). "
                     "Grayed out when loading from Delta Lake.*")

        with gr.Row():
            target_only_update = gr.Checkbox(
                label="Target-only overwrite (partial mode)",
                value=True,
                interactive=False,
            )

        with gr.Row():
            known_rows_expr = gr.Textbox(
                label="Known Rows (1-based)",
                value="",
                info="Rows treated as known/corrected labels (context).",
                interactive=False,
            )
            target_rows_expr = gr.Textbox(
                label="Target Rows (1-based, optional)",
                value="",
                info="Rows to re-predict. Empty means all non-known rows.",
                interactive=False,
            )

        update_analysis_btn = gr.Button("Update Analysis", variant="stop", interactive=False)

        # ==================================================================
        # DIAGNOSTIC OUTPUT
        # ==================================================================
        gr.Markdown("---")
        gr.Markdown("## Diagnostics")
        show_trace = gr.Checkbox(label="Show Iteration Trace", value=False)
        trace_output = gr.Textbox(label="Iteration Trace", interactive=False, lines=12)

        # ==================================================================
        # EVENT WIRING
        # ==================================================================

        # Helper to enable/disable Module 3 widgets based on model availability
        def _update_module3_interactivity(model_available: bool):
            interactive = bool(model_available)
            return (
                gr.update(interactive=interactive),  # target_only_update
                gr.update(interactive=interactive),  # known_rows_expr
                gr.update(interactive=interactive),  # target_rows_expr
                gr.update(interactive=interactive),  # update_analysis_btn
            )

        # ---- Module 1a: Run Inference ----
        run_inference_btn.click(
            fn=run_full_inference,
            inputs=[
                score_file,
                full_ckpt,
                masked_ckpt,
                device,
                task_selector,
                tasks_csv,
                enable_iterative,
                iterative_steps,
                keep_percentile_per_step,
                show_trace,
            ],
            outputs=[
                table,
                status,
                trace_output,
                visual_payload_state,
                raw_predictions_state,
                intermediates_state,
                tasks_state,
                score_path_state,
                edges_state,
                model_available_state,
            ],
        ).then(
            fn=_update_module3_interactivity,
            inputs=[model_available_state],
            outputs=[target_only_update, known_rows_expr, target_rows_expr, update_analysis_btn],
        )

        # ---- Module 1b: Load Delta Lake ----
        load_delta_btn.click(
            fn=load_from_delta_lake,
            inputs=[delta_lake_file],
            outputs=[
                table,
                status,
                trace_output,
                visual_payload_state,
                raw_predictions_state,
                intermediates_state,
                tasks_state,
                score_path_state,
                edges_state,
                model_available_state,
            ],
        ).then(
            fn=_update_module3_interactivity,
            inputs=[model_available_state],
            outputs=[target_only_update, known_rows_expr, target_rows_expr, update_analysis_btn],
        )

        # ---- Module 2: Aggregate! ----
        aggregate_btn.click(
            fn=run_aggregation,
            inputs=[
                aggregation_dropdown,
                raw_predictions_state,
                intermediates_state,
                tasks_state,
                score_path_state,
                edges_state,
            ],
            outputs=[table, status, visual_payload_state],
        )

        # ---- Module 2: Export CSV ----
        export_csv_btn.click(
            fn=export_csv,
            inputs=[table, score_path_state],
            outputs=[csv_download, status],
        )

        # ---- Module 2: Save Delta Lake ----
        save_delta_btn.click(
            fn=save_delta_lake,
            inputs=[
                raw_predictions_state,
                intermediates_state,
                tasks_state,
                score_path_state,
            ],
            outputs=[save_delta_status],
        )

        # ---- Module 2: Refresh Visual ----
        refresh_visual_btn.click(
            fn=refresh_visual_tab,
            inputs=[
                score_file,
                task_selector,
                tasks_csv,
                table,
                visual_edge_types,
                visual_payload_state,
                intermediates_state,
                edges_state,
            ],
            outputs=[visual_html, visual_status, visual_payload_state],
        )

        # ---- Module 3: Update Analysis (edit-conditioned) ----
        update_analysis_btn.click(
            fn=run_edit_conditioned,
            inputs=[
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
                intermediates_state,
                edges_state,
            ],
            outputs=[
                table,
                status,
                trace_output,
                visual_payload_state,
                raw_predictions_state,
                intermediates_state,
                tasks_state,
                score_path_state,
                edges_state,
                model_available_state,
            ],
        )

    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch()
