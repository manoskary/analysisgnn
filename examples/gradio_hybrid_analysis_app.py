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
import html as html_lib
import tempfile
from functools import lru_cache
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
from analysisgnn.utils.chord_representations import format_table_output
from analysisgnn.utils.roman_decode import decode_roman_numeral


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
    # Keep score-derived timing as the source of truth for display. Some model
    # tensors (e.g., onset/s_measure logits or class ids) are not absolute
    # timeline values and can misalign bar/beat rendering if used directly.
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
    # For formats like .mxl, export score object to temporary MusicXML text.
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


def _extract_graph_edges_from_score(score: pt.score.Score, note_array: np.ndarray) -> Tuple[Dict[str, List[List[int]]], str]:
    try:
        from analysisgnn.descriptors import select_features
        from graphmuse import create_score_graph
    except Exception as exc:
        return {k: [[], []] for k in DEFAULT_EDGE_TYPES}, f"Could not import graph builders ({exc})."

    warning = ""
    try:
        note_features = select_features(note_array, "voice")
    except Exception as exc:
        warning = f"Feature selection for graph overlay failed ({exc}); using zero features."
        note_features = np.zeros((len(note_array), 1), dtype=np.float32)
    try:
        measures = score[-1].measures
    except Exception:
        measures = None

    try:
        graph = create_score_graph(
            note_features,
            note_array,
            measures=measures,
            add_beats=True,
            labels=None,
        )
        edge_index_dict = graph.edge_index_dict
    except Exception as exc:
        return {k: [[], []] for k in DEFAULT_EDGE_TYPES}, f"Graph construction failed ({exc})."

    key_map = {
        "onset": ("note", "onset", "note"),
        "consecutive": ("note", "consecutive", "note"),
        "during": ("note", "during", "note"),
        "rest": ("note", "rest", "note"),
    }
    n = len(note_array)
    edges: Dict[str, List[List[int]]] = {}
    for edge_type, key in key_map.items():
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
        valid = (src >= 0) & (src < n) & (dst >= 0) & (dst < n)
        src = src[valid].astype(int).tolist()
        dst = dst[valid].astype(int).tolist()
        edges[edge_type] = [src, dst]
    return edges, warning


def _build_graph_overlay_payload(
    score: pt.score.Score,
    df: pd.DataFrame,
    tasks: List[str],
    edge_types: List[str],
) -> Dict[str, Any]:
    note_array = _sorted_note_array(score)
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
    edges_all, edge_warning = _extract_graph_edges_from_score(score, note_array[:n])

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
            "edge_warning": edge_warning,
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
) -> Dict[str, Any]:
    payload = _build_graph_overlay_payload(score=score, df=df, tasks=tasks, edge_types=edge_types)
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


def _build_aggregation_spec(aggregation_mode: str, voter_path: str) -> Tuple[Dict[str, Any], str]:
    mode_raw = str(aggregation_mode or "Mean").strip().lower()
    mode = "voter" if mode_raw == "voter" else "mean"
    path = (voter_path or "").strip()
    if mode == "voter" and not path:
        return {"mode": "mean"}, "Aggregation mode 'Voter' selected without checkpoint; falling back to mean."
    spec: Dict[str, Any] = {"mode": mode}
    if mode == "voter":
        spec["voter_path"] = path
    return spec, ""


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
    aggregation_mode: str,
    voter_checkpoint_path: str,
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
        aggregation_spec, aggregation_warning = _build_aggregation_spec(
            aggregation_mode=aggregation_mode,
            voter_path=voter_checkpoint_path,
        )
        with torch.no_grad():
            output, routing = predictor.predict(
                score,
                force_route="full",
                iterative_spec=iterative_spec,
                aggregation_spec=aggregation_spec,
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
        display_df = format_table_output(full_df, tasks)

        status = (
            f"Full inference done using route={routing.route} checkpoint={routing.checkpoint_path}. "
            f"Rows={len(display_df)} tasks={','.join(tasks)} aggregation={aggregation_spec.get('mode', 'mean')}"
        )
        if aggregation_warning:
            status = f"{status} | {aggregation_warning}"
        visual_payload = _build_visual_payload(
            score_path=score_path,
            score=score,
            df=display_df,
            tasks=tasks,
            edge_types=[],
        )
        return display_df, status, _format_trace(trace, show_trace), visual_payload
    except Exception as exc:
        return pd.DataFrame(), f"Error: {exc}", "", {}


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
    aggregation_mode: str,
    voter_checkpoint_path: str,
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
        aggregation_spec, aggregation_warning = _build_aggregation_spec(
            aggregation_mode=aggregation_mode,
            voter_path=voter_checkpoint_path,
        )
        with torch.no_grad():
            output, routing = predictor.predict(
                score,
                user_edits=user_edits,
                masked_spec=masked_spec,
                iterative_spec=iterative_spec,
                aggregation_spec=aggregation_spec,
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
        display_df = format_table_output(out_df, tasks)

        status = (
            f"Partial inference done using route={routing.route} checkpoint={routing.checkpoint_path}. "
            f"Known rows={info.get('num_known', 0)} target rows={info.get('num_targets', 0)} "
            f"aggregation={aggregation_spec.get('mode', 'mean')}"
        )
        if aggregation_warning:
            status = f"{status} | {aggregation_warning}"
        visual_payload = _build_visual_payload(
            score_path=score_path,
            score=score,
            df=display_df,
            tasks=tasks,
            edge_types=[],
        )
        return display_df, status, _format_trace(trace, show_trace), visual_payload
    except Exception as exc:
        return pd.DataFrame(), f"Error: {exc}", "", {}


def refresh_visual_tab(
    score_file: Any,
    task_labels: List[str],
    tasks_csv: str,
    table_data: Any,
    edge_type_labels: List[str],
    visual_state: Dict[str, Any],
):
    try:
        selected_edge_types = [k for k, label in EDGE_LABELS.items() if label in (edge_type_labels or [])]
        tasks = _resolve_selected_tasks(task_labels, tasks_csv)
        score_path = None
        score = None
        payload: Dict[str, Any] = {}

        if score_file is not None:
            score_path = _resolve_score_path(score_file)
            score = _load_score(score_path)

        df = pd.DataFrame(table_data) if table_data is not None else pd.DataFrame()
        if score is not None and len(df) > 0:
            payload = _build_visual_payload(
                score_path=score_path,
                score=score,
                df=df,
                tasks=tasks,
                edge_types=selected_edge_types,
            )
        elif isinstance(visual_state, dict) and visual_state:
            payload = dict(visual_state)
            payload.setdefault("meta", {})
            payload["meta"]["visible_edge_types"] = selected_edge_types
        else:
            raise ValueError("No predictions available yet. Run inference first to populate the visual tab.")

        html_frame = _build_verovio_html(payload)
        note_count = len(payload.get("notes", []))
        edge_warning = ((payload.get("meta") or {}).get("edge_warning") or "").strip()
        status = (
            f"Visual refreshed: notes={note_count}, "
            f"visible edges={','.join(selected_edge_types) if selected_edge_types else 'none'}."
        )
        if edge_warning:
            status = f"{status} Graph warning: {edge_warning}"
        return html_frame, status, payload
    except Exception as exc:
        fallback = (
            "<div style='padding:12px;border:1px solid #d1d5db;border-radius:10px;background:#fff;'>"
            f"Visual rendering error: {html_lib.escape(str(exc))}"
            "</div>"
        )
        return fallback, f"Visual error: {exc}", visual_state if isinstance(visual_state, dict) else {}


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="AnalysisGNN Hybrid Inference") as demo:
        gr.Markdown("""
# AnalysisGNN Hybrid Inference

Three explicit inference paths:
- Base model for full-piece prediction.
- Iterative refinement (no known labels at step 1) for apples-to-apples full-model benchmarking.
- Masked model for edit-conditioned partial re-prediction.

This app also includes a separate **Verovio Visual Score** tab for score + graph overlays.

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
            info="Choose which tasks to run and show in the editable table and visual tab.",
        )
        tasks_csv = gr.Textbox(
            label="Tasks Override (internal keys CSV, optional)",
            value=DEFAULT_TASKS,
            info="Used only if no task is selected above. Example: romanNumeral,localkey,quality",
        )

        visual_payload_state = gr.State({})

        with gr.Tabs():
            with gr.Tab("Inference & Edits"):
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
                    aggregation_mode = gr.Dropdown(
                        label="Aggregation Mode",
                        choices=["Mean", "Voter"],
                        value="Mean",
                    )
                    voter_checkpoint_path = gr.Textbox(
                        label="Voter Checkpoint Path",
                        value="",
                        info="Optional. Required only when Aggregation Mode is Voter.",
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
                refresh_visual_btn = gr.Button("Refresh Visual from Latest Predictions", variant="secondary")
                visual_html = gr.HTML(
                    value=(
                        "<div style='padding:12px;border:1px solid #d1d5db;border-radius:10px;background:#fff;'>"
                        "Run inference first, then click “Refresh Visual from Latest Predictions”."
                        "</div>"
                    ),
                    label="Verovio Score + Graph",
                )
                visual_status = gr.Textbox(label="Visual Status", interactive=False)

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
            aggregation_mode: str,
            voter_checkpoint_path: str,
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
                    aggregation_mode=aggregation_mode,
                    voter_checkpoint_path=voter_checkpoint_path,
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
                    aggregation_mode=aggregation_mode,
                    voter_checkpoint_path=voter_checkpoint_path,
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
                aggregation_mode=aggregation_mode,
                voter_checkpoint_path=voter_checkpoint_path,
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
                aggregation_mode,
                voter_checkpoint_path,
                target_only_update,
                show_trace,
            ],
            outputs=[table, status, trace_output, visual_payload_state],
        )

        refresh_visual_btn.click(
            fn=refresh_visual_tab,
            inputs=[
                score_file,
                task_selector,
                tasks_csv,
                table,
                visual_edge_types,
                visual_payload_state,
            ],
            outputs=[visual_html, visual_status, visual_payload_state],
        )

    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch()
