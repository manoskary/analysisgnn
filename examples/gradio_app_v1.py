#!/usr/bin/env python3
"""Hybrid Gradio interface for AnalysisGNN.

Three-module layout:
  Module 1 — Data Source: run inference (tab 1a) or load Delta Lake (tab 1b)
  Module 2 — Analysis Results: aggregation, CSV download, Verovio, Delta Lake save
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
import re
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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

import flexohr as flx
import flexohr.codecs.analysisgnn  # noqa: F401 — activate codec
import flexohr.harmony.harmony_enums  # noqa: F401
import flexohr.paradigms.pitchspace.scale  # noqa: F401
import flexohr.paradigms.pitchspace.scale_degrees  # noqa: F401

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
DEFAULT_HF_REPO = os.environ.get("ANALYSISGNN_HF_REPO", "").strip()
DEFAULT_HF_REVISION = os.environ.get("ANALYSISGNN_HF_REVISION", "").strip()


def _resolve_optional_default_path(env_key: str, fallback: Path) -> str:
    env_val = os.environ.get(env_key, "").strip()
    if env_val:
        return env_val if os.path.exists(env_val) else ""
    return str(fallback) if fallback.exists() else ""


DEFAULT_VOTER_CKPT = _resolve_optional_default_path(
    "ANALYSISGNN_VOTER_CKPT",
    REPO_ROOT / "artifacts" / "posthoc_voter" / "uocj8f6y_voter.pt",
)
DEFAULT_BEAT_VOTER_CKPT = _resolve_optional_default_path(
    "ANALYSISGNN_BEAT_VOTER_CKPT",
    (
        Path(DEFAULT_VOTER_CKPT)
        if DEFAULT_VOTER_CKPT
        else REPO_ROOT / "artifacts" / "posthoc_voter" / "uocj8f6y_voter.pt"
    ),
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
# Logging helper
# ---------------------------------------------------------------------------


def _log(existing: str, message: str) -> str:
    """Append *message* as a new line to the running log text."""
    if existing:
        return f"{existing}\n{message}"
    return message


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
DEFAULT_BEAT_TASKS = [
    "localkey",
    "degree1",
    "degree2",
    "quality",
    "inversion",
    "root",
    "bass",
]
BEAT_TASK_CHOICES = list(
    dict.fromkeys(
        DEFAULT_BEAT_TASKS
        + [
            "romanNumeral",
            "cadence",
            "phrase",
        ]
    )
)


@lru_cache(maxsize=8)
def _resolve_hf_bundle_cached(repo_id: str, revision: str) -> Dict[str, str]:
    bundle = resolve_hybrid_bundle(
        source=repo_id,
        revision=(revision or None),
    )
    return {
        "full_ckpt": bundle.full_ckpt,
        "masked_ckpt": bundle.masked_ckpt,
        "voter_ckpt": bundle.voter_ckpt,
        "beat_voter_ckpt": bundle.beat_voter_ckpt,
    }


def _resolve_runtime_artifact_paths(
    *,
    full_ckpt: str,
    masked_ckpt: str,
    voter_ckpt: str,
    beat_voter_ckpt: str,
) -> Tuple[Dict[str, str], str]:
    resolved = {
        "full_ckpt": (full_ckpt or "").strip(),
        "masked_ckpt": (masked_ckpt or "").strip(),
        "voter_ckpt": (voter_ckpt or "").strip(),
        "beat_voter_ckpt": (beat_voter_ckpt or "").strip(),
    }
    missing = [k for k, v in resolved.items() if (not v) or (not os.path.exists(v))]
    if not missing:
        return resolved, ""

    repo_id = DEFAULT_HF_REPO
    revision = DEFAULT_HF_REVISION
    if not repo_id:
        return resolved, ""

    try:
        hf_paths = _resolve_hf_bundle_cached(repo_id, revision)
    except HybridBundleResolutionError as exc:
        return resolved, f"HF bundle resolution failed ({repo_id}): {exc}"

    filled: List[str] = []
    unresolved: List[str] = []
    for key in missing:
        candidate = hf_paths.get(key, "")
        if candidate and os.path.exists(candidate):
            resolved[key] = candidate
            filled.append(key)
        else:
            unresolved.append(key)

    notes: List[str] = []
    if filled:
        rev_txt = f"@{revision}" if revision else ""
        notes.append(
            f"Loaded missing artifacts from HF bundle {repo_id}{rev_txt}: {','.join(filled)}"
        )
    if unresolved:
        notes.append(f"Still missing artifacts: {','.join(unresolved)}")
    return resolved, " | ".join(notes)


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


def _get_predictor(
    full_ckpt: str, masked_ckpt: str, device: str
) -> HybridAnalysisPredictor:
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
        tasks = [
            label_to_task[label] for label in task_labels if label in label_to_task
        ]
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


def _apply_timing_from_predictions(
    df: pd.DataFrame, predictions: Dict[str, torch.Tensor]
) -> pd.DataFrame:
    out = df.copy()
    if "note_id" not in out.columns:
        out.insert(0, "note_id", np.arange(len(out)))
    _convert_tpc_column_inplace(out)

    timing_cols = [
        col
        for col in [
            "row",
            "note_id",
            "onset_beat",
            "measure",
            "duration_beat",
            "pitch_spelling",
            "pitch_midi",
        ]
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
    remaining_cols = [
        col
        for col in out.columns
        if col not in ordered_cols and not col.endswith("_id")
    ]
    out = out[ordered_cols + remaining_cols]
    return out


def _apply_timing_from_predictions(
    df: pd.DataFrame, predictions: Dict[str, torch.Tensor]
) -> pd.DataFrame:
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


def _derive_global_key(df: pd.DataFrame) -> str:
    """Derive the global key from tonic chords (``romanNumeral`` is ``"I"``
    or ``"i"``).

    Finds the most frequent localkey pitch class (case-insensitive) among
    tonic chords, then infers major/minor mode from the ``"I"`` vs ``"i"``
    counts via :func:`_infer_key_mode`.

    Returns an SPC-compatible string — uppercase for major (e.g. ``"G"``),
    lowercase for minor (e.g. ``"c"``).

    Raises
    ------
    ValueError
        If the global key cannot be derived (empty DataFrame, missing
        columns, or no tonic chords found).
    """
    if df is None or len(df) == 0:
        raise ValueError("Cannot derive global key: DataFrame is empty.")
    if "romanNumeral" not in df.columns or "localkey" not in df.columns:
        raise ValueError(
            "Cannot derive global key: DataFrame is missing 'romanNumeral' "
            "and/or 'localkey' columns."
        )
    mask = df["romanNumeral"].isin(["I", "i"])
    candidates = df.loc[mask, ["romanNumeral", "localkey"]].copy()
    if len(candidates) == 0:
        raise ValueError(
            "Cannot derive global key: no rows with romanNumeral 'I' or 'i'."
        )
    # Find the most frequent localkey pitch class (case-insensitive)
    candidates["_lk_upper"] = candidates["localkey"].astype(str).str.upper()
    lk_counts = candidates["_lk_upper"].value_counts()
    best_lk_upper = str(lk_counts.index[0])

    # Infer mode from all tonic chords in this key
    best_mask = candidates["_lk_upper"] == best_lk_upper
    mode = _infer_key_mode(candidates.loc[best_mask, "romanNumeral"])
    if mode is None:
        mode = "major"

    best_lk_raw = str(candidates.loc[best_mask, "localkey"].iloc[0])
    lk = _agnn_to_flx_pitch(best_lk_raw)
    return _apply_key_mode(lk, mode)


def _agnn_to_flx_pitch(name: str) -> str:
    """Normalise an AnalysisGNN pitch-class string for FlexOHR.

    AnalysisGNN uses ``-`` for flat (e.g. ``A-``, ``B--``); FlexOHR's
    ``SPC`` expects ``b`` (e.g. ``Ab``, ``Bbb``).
    """
    return name.replace("-", "b")


def _infer_key_mode(roman_numerals: pd.Series) -> Optional[str]:
    """Infer major/minor key mode from tonic Roman numeral counts.

    Counts ``"I"`` (major tonic) vs ``"i"`` (minor tonic) in the given
    series.  Returns ``"major"`` or ``"minor"`` based on which is more
    frequent, or ``None`` if neither is found.
    """
    if roman_numerals is None or len(roman_numerals) == 0:
        return None
    n_major = int((roman_numerals == "I").sum())
    n_minor = int((roman_numerals == "i").sum())
    if n_major == 0 and n_minor == 0:
        return None
    return "minor" if n_minor > n_major else "major"


def _apply_key_mode(key_str: str, mode: str) -> str:
    """Adjust the case of a key string to encode the given mode.

    ``"major"`` -> uppercase first character, ``"minor"`` -> lowercase
    first character.  Accidentals (``#``, ``b``) are left unchanged.
    """
    if not key_str:
        return key_str
    if mode == "minor":
        return key_str[0].lower() + key_str[1:]
    return key_str[0].upper() + key_str[1:]


def _build_localkey_mode_map(df: pd.DataFrame) -> Dict[str, str]:
    """Build a mapping from localkey pitch class to inferred mode.

    Groups rows by localkey pitch class (case-insensitive) and counts
    ``"I"`` vs ``"i"`` in the ``romanNumeral`` column to determine
    whether each local key is major or minor.

    Returns
    -------
    Dict[str, str]
        Keys are uppercased localkey strings (e.g. ``"G"``, ``"C#"``,
        ``"A-"``); values are ``"major"`` or ``"minor"``.
    """
    if df is None or len(df) == 0:
        return {}
    if "localkey" not in df.columns or "romanNumeral" not in df.columns:
        return {}
    result: Dict[str, str] = {}
    work = df[["localkey", "romanNumeral"]].copy()
    work["_lk_upper"] = work["localkey"].astype(str).str.upper()
    for lk_upper, group in work.groupby("_lk_upper"):
        mode = _infer_key_mode(group["romanNumeral"])
        if mode is None:
            # TODO: FlexOHR will be able to infer the mode from the scale
            # degree and global-key mode in this case in the future.
            mode = "major"
        result[str(lk_upper)] = mode
    return result


def _build_complete_rn_column(df: pd.DataFrame, global_key: str) -> pd.Series:
    """Build the Complete RN column using FlexOHR OHR objects.

    Each row's five principal task predictions (degree1, degree2, inversion,
    quality, localkey) are used to construct a FlexOHR OHR, which is then
    rendered via ``.to_format('dcml')``.

    The mode (major/minor) of each local key is **not** taken from the
    case of the ``localkey`` prediction directly.  Instead, it is inferred
    from the ``romanNumeral`` predictions: within all rows that share the
    same localkey pitch class, the counts of ``"I"`` (major tonic) vs
    ``"i"`` (minor tonic) determine the mode.  This is the same principle
    used by :func:`_derive_global_key` for the global key.
    """
    if df is None or len(df) == 0:
        return pd.Series(dtype=object)
    required = ["degree1", "degree2", "inversion", "quality", "localkey"]
    missing = [k for k in required if k not in df.columns]
    if missing:
        return pd.Series([""] * len(df), index=df.index, dtype=object)

    gk = _agnn_to_flx_pitch(global_key)

    # Infer localkey modes from romanNumeral tonic counts (I vs i)
    localkey_modes = _build_localkey_mode_map(df)

    out: List[str] = []
    for _, row in df.iterrows():
        try:
            quality = flx.harmony.harmony_enums.ChordQuality.from_format(
                str(row["quality"]),
                "analysisgnn",
            )
            inv = flx.harmony.harmony_enums.Inversion.from_format(
                str(int(row["inversion"])),
                "analysisgnn",
            )
            # Apply inferred mode to the localkey string so that FlexOHR's
            # infer_collection_type() picks up the correct major/minor.
            lk_raw = str(row["localkey"])
            lk_mode = localkey_modes.get(lk_raw.upper(), "major")
            lk_str = _apply_key_mode(_agnn_to_flx_pitch(lk_raw), lk_mode)
            lk_coll = flx.paradigms.pitchspace.scale.infer_collection_type(lk_str)

            degree2_val = row.get("degree2")
            if pd.notna(degree2_val) and str(degree2_val).strip() not in ("", "None"):
                sd2 = flx.paradigms.pitchspace.scale_degrees.SD.from_string(
                    str(degree2_val),
                    collection_type=lk_coll,
                )
                # TODO: The tonicized key mode cannot be inferred from the
                # AnalysisGNN output alone (degree2 is a bare integer with
                # no case encoding).  FlexOHR will be able to infer the
                # mode from the scale degree and key context in the future.
                # Defaulting to major for now.
                ref_ohr = flx.paradigms.pitchspace.scale.build_key_context(
                    gk,
                    lk_str,
                    tonicized_key=sd2,
                    tonicized_coll=flx.harmony.harmony_enums.CollectionType.major,
                )
                tonic_coll = flx.harmony.harmony_enums.CollectionType.major
            else:
                # Normalise the row's localkey for FlexOHR before delegating
                norm_row = dict(row)
                norm_row["localkey"] = lk_str
                ref_ohr = flx.codecs.analysisgnn.build_key_context_from_row(
                    norm_row,
                    gk,
                )
                tonic_coll = lk_coll

            degree1_sd = flx.paradigms.pitchspace.scale_degrees.SD.from_string(
                str(row["degree1"]),
                collection_type=tonic_coll,
            )
            ohr = flx.OHR.from_(
                quality,
                degree1_sd,
                inversion=inv,
                reference_ohr=ref_ohr,
            )
            out.append(ohr.to_format("dcml"))
        except Exception:
            out.append("")
    return pd.Series(out, index=df.index, dtype=object)


def _inject_note_ids(xml_text: str, score: pt.score.Score) -> str:
    """Inject partitura note IDs into ``<note>`` elements of the original MusicXML.

    Partitura assigns stable IDs (``p0n0``, ``p0n3``, ...) when loading a score.
    The original MusicXML typically has no ``id`` attributes on ``<note>`` elements.
    Verovio preserves ``id`` attributes in SVG output, so the JS overlay can match
    payload notes to SVG note groups by ID.  Without IDs, the fallback sequential
    mapping fails because our sort order (onset_div, pitch) differs from MusicXML
    document order (onset, voice/staff).

    Each partitura note has a ``doc_order`` attribute giving its 0-based index
    among *all* ``<note>`` elements (including rests) in the MusicXML.  We build
    a mapping from that index to the note ID and inject the ID when we encounter
    the corresponding ``<note>`` element.
    """
    parts = list(getattr(score, "parts", []) or [])
    if not parts:
        return xml_text

    # Map: doc_order -> note ID.
    doc_order_to_id: Dict[int, str] = {}
    for part in parts:
        for n in part.notes_tied:
            doc_ord = getattr(n, "doc_order", None)
            if doc_ord is not None:
                doc_order_to_id[doc_ord] = str(n.id)

    if not doc_order_to_id:
        return xml_text

    # Regex: match ``<note`` followed by optional attributes and ``>``, then body,
    # then ``</note>``.  Group 1 = any existing attributes + the closing ``>``.
    # Group 2 = the body between ``>`` and ``</note>``.
    _note_re = re.compile(r"<note(\s[^>]*)?>(.+?)</note>", re.DOTALL)
    doc_idx = 0

    def _replacer(m: re.Match) -> str:
        nonlocal doc_idx
        current_idx = doc_idx
        doc_idx += 1
        existing_attrs = m.group(1) or ""
        body = m.group(2)
        # If the tag already carries an id, leave it alone.
        if "id=" in existing_attrs:
            return m.group(0)
        nid = doc_order_to_id.get(current_idx)
        if nid is None:
            # Rest or unmatched — return unchanged.
            return m.group(0)
        return f'<note id="{nid}"{existing_attrs}>{body}</note>'

    return _note_re.sub(_replacer, xml_text)


def _read_score_xml_text(score_path: str, score: pt.score.Score) -> str:
    suffix = Path(score_path).suffix.lower()
    if suffix in {".xml", ".musicxml"}:
        with open(score_path, "r", encoding="utf-8", errors="ignore") as f:
            xml_text = f.read()
        return _inject_note_ids(xml_text, score)
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


def _build_complete_rn_spans(
    df: pd.DataFrame, global_key: str
) -> List[Tuple[int, int, str]]:
    """Build non-redundant Roman Numeral spans over onset_div."""
    if df is None or len(df) == 0:
        return []
    if "onset_div" not in df.columns:
        return []

    work = df.copy()
    if "romanNumeral_full" not in work.columns:
        work["romanNumeral_full"] = _build_complete_rn_column(work, global_key)
    if "duration_div" not in work.columns:
        return []

    work["onset_div"] = pd.to_numeric(work["onset_div"], errors="coerce")
    work["duration_div"] = pd.to_numeric(work["duration_div"], errors="coerce").fillna(
        0
    )
    work["romanNumeral_full"] = (
        work["romanNumeral_full"].fillna("").astype(str).str.strip()
    )
    work = work.dropna(subset=["onset_div"])
    if len(work) == 0:
        return []

    by_onset = work.sort_values(["onset_div", "duration_div"]).groupby(
        "onset_div", sort=True
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

    score_end = int(
        np.max(
            work["onset_div"].to_numpy()
            + np.maximum(1, work["duration_div"].to_numpy())
        )
    )
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
    """Extract edge lists from a PyG HeteroData ``edge_index_dict``."""
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


def _edges_from_delta_lake_df(
    edges_df: pd.DataFrame, note_id_to_idx: Dict[str, int]
) -> Dict[str, List[List[int]]]:
    """Convert a Delta Lake edges DataFrame to the edge-list format used by Verovio."""
    edges: Dict[str, List[List[int]]] = {}
    for etype in DEFAULT_EDGE_TYPES:
        sub = (
            edges_df[edges_df["edge_type"] == etype] if len(edges_df) > 0 else edges_df
        )
        src_ids = sub["src"].tolist() if len(sub) > 0 else []
        dst_ids = sub["dst"].tolist() if len(sub) > 0 else []
        src_idx = [note_id_to_idx[s] for s in src_ids if s in note_id_to_idx]
        dst_idx = [note_id_to_idx[d] for d in dst_ids if d in note_id_to_idx]
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
    global_key: str = "",
) -> Dict[str, Any]:
    """Build the Verovio overlay payload."""
    n = min(len(df), len(note_array))
    data = df.iloc[:n].reset_index(drop=True).copy()
    if "romanNumeral_full" in data.columns:
        rn_full = data["romanNumeral_full"].fillna("").astype(str)
    elif global_key:
        rn_full = _build_complete_rn_column(data, global_key)
    else:
        rn_full = pd.Series([""] * n, dtype=object)
    spans_df = data.copy()
    if "onset_div" in note_array.dtype.names:
        spans_df["onset_div"] = note_array["onset_div"][:n]
    if "duration_div" in note_array.dtype.names:
        spans_df["duration_div"] = note_array["duration_div"][:n]
    spans_df["romanNumeral_full"] = rn_full
    rn_spans = _build_complete_rn_spans(spans_df, global_key)

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
        score_note_id = (
            _value_or_none(note_array["id"][idx])
            if "id" in note_array.dtype.names
            else None
        )
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
        f'srcdoc="{srcdoc}"></iframe>'
    )


def _build_visual_payload(
    score_path: str,
    score: pt.score.Score,
    df: pd.DataFrame,
    tasks: List[str],
    edge_types: List[str],
    edges_all: Dict[str, List[List[int]]],
    global_key: str = "",
) -> Dict[str, Any]:
    note_array = _sorted_note_array(score)
    payload = _build_graph_overlay_payload(
        df=df,
        note_array=note_array,
        tasks=tasks,
        edge_types=edge_types,
        edges_all=edges_all,
        global_key=global_key,
    )
    payload["score_xml"] = _read_score_xml_text(score_path, score)
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
        "keep_percentile_per_step": float(
            max(0.0, min(100.0, keep_percentile_per_step))
        ),
        "masked_tasks": list(tasks),
        "mode": "cumulative",
        "freeze_confidence": "joint_mean",
        "target_only_update": bool(target_only_update),
        "min_remaining_targets": 0,
        "confidence_temperature": 1.0,
        "zero_known_start": bool(zero_known_start),
    }


def _build_aggregation_spec(
    aggregation_mode: str,
    voter_path: str,
    *,
    beat_tasks: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], str]:
    mode_raw = str(aggregation_mode or "Mean").strip().lower()
    mode_map = {
        "mean": "mean",
        "voter": "voter",
        "voter consistent beat": "voter_consistent_beat",
        "voter_consistent_beat": "voter_consistent_beat",
    }
    mode = mode_map.get(mode_raw, "mean")
    path = (voter_path or "").strip()
    if mode in {"voter", "voter_consistent_beat"} and not path:
        return {"mode": "mean"}, (
            f"Aggregation mode '{aggregation_mode}' selected without checkpoint; falling back to mean."
        )
    if mode in {"voter", "voter_consistent_beat"} and path and not os.path.exists(path):
        return {"mode": "mean"}, (
            f"Voter checkpoint not found at '{path}'; falling back to mean."
        )
    spec: Dict[str, Any] = {"mode": mode}
    if mode in {"voter", "voter_consistent_beat"}:
        spec["voter_path"] = path
    if beat_tasks:
        spec["beat_tasks"] = [t for t in beat_tasks if t]
    return spec, ""


def _is_trace_payload(obj: Any) -> bool:
    return isinstance(obj, dict) and "enabled" in obj and "steps" in obj


def _is_beat_payload(obj: Any) -> bool:
    return (
        isinstance(obj, dict)
        and obj.get("level", "beat") == "beat"
        and "rows" in obj
        and "tasks" in obj
        and "mode" in obj
    )


def _is_measure_payload(obj: Any) -> bool:
    return (
        isinstance(obj, dict)
        and obj.get("level") == "measure"
        and "rows" in obj
        and "mode" in obj
    )


def _parse_predict_output(
    output: Any,
    *,
    enable_iterative: bool,
    enable_beat: bool,
    enable_measure: bool,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    default_trace: Dict[str, Any] = {"enabled": False, "steps": []}
    if isinstance(output, tuple):
        if len(output) == 0:
            return {}, default_trace, None, None
        predictions = output[0]
        trace = default_trace
        beat_payload = None
        measure_payload = None
        for item in output[1:]:
            if _is_trace_payload(item):
                trace = item
            elif _is_beat_payload(item):
                beat_payload = item
            elif _is_measure_payload(item):
                measure_payload = item
        if enable_iterative and trace is default_trace:
            trace = {"enabled": True, "steps": []}
        if enable_beat and beat_payload is None:
            beat_payload = {"level": "beat", "rows": [], "tasks": [], "mode": "mean"}
        if enable_measure and measure_payload is None:
            measure_payload = {"level": "measure", "rows": [], "mode": "summary_v1"}
        return predictions, trace, beat_payload, measure_payload
    return output, default_trace, None, None


def _beat_payload_to_dataframe(
    beat_payload: Optional[Dict[str, Any]],
    beat_tasks: List[str],
) -> pd.DataFrame:
    if not isinstance(beat_payload, dict):
        return pd.DataFrame()
    rows = beat_payload.get("rows", [])
    if not rows:
        return pd.DataFrame()
    payload_tasks = beat_payload.get("tasks", [])
    tasks = (
        [t for t in beat_tasks if t in payload_tasks]
        if beat_tasks
        else list(payload_tasks)
    )
    if not tasks:
        tasks = list(payload_tasks)

    flat_rows: List[Dict[str, Any]] = []
    for row in rows:
        out: Dict[str, Any] = {
            "beat_id": row.get("beat_id"),
            "beat_index": row.get("beat_index"),
            "measure": row.get("measure"),
            "onset_beat": row.get("onset_beat"),
            "note_count": row.get("note_count"),
            "romanNumeral_full": row.get("romanNumeral_full", ""),
        }
        task_map = row.get("tasks", {}) if isinstance(row, dict) else {}
        for task in tasks:
            entry = task_map.get(task, {}) if isinstance(task_map, dict) else {}
            out[task] = entry.get("label")
            out[f"{task}_confidence"] = entry.get("confidence")
            out[f"{task}_conflict_flag"] = entry.get("conflict_flag")
            out[f"{task}_conflict_prob"] = entry.get("conflict_prob")
        out.update(build_beat_chord_symbol_row(row if isinstance(row, dict) else {}))
        flat_rows.append(out)

    beat_df = pd.DataFrame(flat_rows)
    core_cols = [
        "beat_id",
        "beat_index",
        "measure",
        "onset_beat",
        "note_count",
        "romanNumeral_full",
        "chordSymbol_abs",
        "chordSymbol_context",
        "chordSymbol_supported",
        "chordSymbol_ambiguous",
        "chordSymbol_source",
    ]
    ordered_cols: List[str] = [c for c in core_cols if c in beat_df.columns]
    for task in tasks:
        for col in [
            task,
            f"{task}_confidence",
            f"{task}_conflict_flag",
            f"{task}_conflict_prob",
        ]:
            if col in beat_df.columns:
                ordered_cols.append(col)
    remaining = [c for c in beat_df.columns if c not in ordered_cols]
    return beat_df[ordered_cols + remaining]


def _measure_payload_to_dataframe(
    measure_payload: Optional[Dict[str, Any]],
) -> pd.DataFrame:
    if not isinstance(measure_payload, dict):
        return pd.DataFrame()
    rows = measure_payload.get("rows", [])
    if not rows:
        return pd.DataFrame()

    measure_df = pd.DataFrame(rows)
    ordered_cols = [
        "measure",
        "measure_index",
        "note_count",
        "onset_count",
        "measure_start_beat",
        "measure_end_beat",
        "tonal_space_label",
        "tonal_space_confidence",
        "mixedness",
        "transition_flag",
        "bar_localkey",
        "bar_localkey_confidence",
        "tonicization_target",
        "tonicization_confidence",
        "modulation_confidence",
        "cadential_intent",
        "cadential_confidence",
        "harmonic_stability",
        "harmonic_change_density",
        "top2_label",
        "top2_share",
        "romanNumeral_full_mode",
        "no_evidence",
    ]
    ordered_present = [col for col in ordered_cols if col in measure_df.columns]
    remaining = [col for col in measure_df.columns if col not in ordered_present]
    return measure_df[ordered_present + remaining]


def _dataframe_to_csv_file(df: Optional[pd.DataFrame], prefix: str) -> Optional[str]:
    if df is None or len(df) == 0:
        return None
    safe_prefix = "".join(
        ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in prefix
    ).strip("_")
    safe_prefix = safe_prefix or "analysisgnn"
    with tempfile.NamedTemporaryFile(
        suffix=".csv",
        prefix=f"{safe_prefix}_",
        delete=False,
    ) as tmp:
        csv_path = tmp.name
    df.to_csv(csv_path, index=False)
    return csv_path


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


def _write_csv_to_temp(df: pd.DataFrame, score_path: str) -> Optional[str]:
    """Write *df* to a temp CSV file and return the path (or None on error)."""
    if df is None or len(df) == 0:
        return None
    score_id = _derive_score_id(score_path) if score_path else "export"
    csv_path = os.path.join(tempfile.gettempdir(), f"{score_id}_analysis.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Precompute DataFrames once, cache for aggregation
# ---------------------------------------------------------------------------


def _precompute_delta_dfs(
    raw_predictions: Dict[str, Any],
    intermediates: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert raw predictions + intermediates to the long-format DataFrames
    needed by aggregation strategies.  The result is cached in gr.State so
    that switching aggregation strategy is instant.

    Returns a dict with keys: ``probs_df``, ``notes_df``, ``hyperedges_df``, ``metadata``.
    """
    from analysisgnn.storage.delta_writer import (
        _build_probabilities_table,
        _build_notes_table,
        _build_hyperedges_table,
    )

    score_obj = intermediates.get("score")
    note_array = intermediates.get("note_array")
    pyg_data = intermediates.get("data")
    if score_obj is None or note_array is None:
        raise ValueError("Missing score or note_array in intermediates.")

    # Build task_dict from predictions
    task_dict: Dict[str, int] = {}
    for task_name, tensor in raw_predictions.items():
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
            task_dict[task_name] = tensor.shape[1]

    # Build note_ids
    n = len(note_array)
    if "id" in note_array.dtype.names:
        note_ids = np.array([str(x) for x in note_array["id"]], dtype=object)
    else:
        note_ids = np.array([f"note_{i}" for i in range(n)], dtype=object)

    probs_table = _build_probabilities_table(raw_predictions, task_dict, note_ids)
    probs_df = probs_table.to_pandas()

    notes_table = _build_notes_table(note_array, score_obj)
    notes_df = notes_table.to_pandas()

    if pyg_data is not None:
        hyperedges_table, _ = _build_hyperedges_table(pyg_data, note_ids)
        hyperedges_df = hyperedges_table.to_pandas()
    else:
        hyperedges_df = pd.DataFrame(
            columns=["group_id", "note_id", "edge_type", "parent_group_id"]
        )

    return {
        "probs_df": probs_df,
        "notes_df": notes_df,
        "hyperedges_df": hyperedges_df,
        "metadata": {"task_dict": task_dict},
    }


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
    aggregation_mode: str,
    voter_checkpoint_path: str,
    enable_beat: bool,
    beat_aggregation_mode: str,
    beat_voter_checkpoint_path: str,
    beat_tasks: List[str],
    enable_measure: bool,
    show_trace: bool,
    log_text: str,
):
    """Run inference with aggregation_spec={"mode": "none"} and return_intermediates=True.

    Returns:
        (display_df, log_text, visual_payload, csv_path,
         raw_predictions_state, intermediates_state, delta_dfs_state,
         tasks_state, score_path_state, edges_state, model_available_flag,
         global_key)
    """
    try:
        log_text = _log(log_text, "Starting inference...")
        score_path = _resolve_score_path(score_file)
        score = _load_score(score_path)
        tasks = _resolve_selected_tasks(task_labels, tasks_csv)
        resolved_paths, hf_resolution_note = _resolve_runtime_artifact_paths(
            full_ckpt=full_ckpt,
            masked_ckpt=masked_ckpt,
            voter_ckpt=voter_checkpoint_path,
            beat_voter_ckpt=beat_voter_checkpoint_path,
        )
        predictor = _get_predictor(
            resolved_paths["full_ckpt"],
            resolved_paths["masked_ckpt"],
            device,
        )

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
            voter_path=resolved_paths["voter_ckpt"],
        )
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

        # Build display DataFrame
        predictions, trace, _, _ = _parse_predict_output(
            output,
            enable_iterative=bool(enable_iterative),
            enable_beat=False,
            enable_measure=False,
        )

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

        # State objects
        intermediates_state = {
            "score": score_obj,
            "note_array": note_array,
            "data": pyg_data,
            "score_path": score_path,
        }

        # Precompute DataFrames for aggregation (cached)
        delta_dfs = _precompute_delta_dfs(predictions, intermediates_state)

        # Derive global key from predictions
        try:
            global_key = _derive_global_key(display_df)
        except ValueError as gk_exc:
            global_key = ""
            log_text = _log(log_text, f"Global key derivation failed: {gk_exc}")

        # Add Complete RN column
        if global_key:
            display_df["romanNumeral_full"] = _build_complete_rn_column(
                display_df, global_key
            )

        # Build visual payload
        status = (
            f"Full inference done using route={routing.route} checkpoint={routing.checkpoint_path}. "
            f"Rows={len(display_df)} tasks={','.join(tasks)} aggregation={aggregation_spec.get('mode', 'mean')}"
        )
        if hf_resolution_note:
            status = f"{status} | {hf_resolution_note}"
        if aggregation_warning:
            status = f"{status} | {aggregation_warning}"

        beat_df = pd.DataFrame()
        beat_status = "Beat-level aggregation disabled."
        measure_df = pd.DataFrame()
        measure_status = "Measure-level summary disabled."
        if bool(enable_beat):
            selected_beat_tasks = [t for t in (beat_tasks or []) if t]
            if not selected_beat_tasks:
                selected_beat_tasks = list(DEFAULT_BEAT_TASKS)
            beat_agg_spec, beat_agg_warning = _build_aggregation_spec(
                aggregation_mode=beat_aggregation_mode,
                voter_path=resolved_paths["beat_voter_ckpt"],
                beat_tasks=selected_beat_tasks,
            )
            with torch.no_grad():
                beat_output_raw = predictor.predict(
                    score,
                    force_route="full",
                    iterative_spec=iterative_spec,
                    aggregation_spec=beat_agg_spec,
                    return_beat_predictions=True,
                    return_route=False,
                )
            _, _, beat_payload, _ = _parse_predict_output(
                beat_output_raw,
                enable_iterative=False,
                enable_beat=True,
                enable_measure=False,
            )
            beat_df = _beat_payload_to_dataframe(
                beat_payload=beat_payload,
                beat_tasks=selected_beat_tasks,
            )
            beat_status = f"Beat-level table ready: rows={len(beat_df)} mode={beat_agg_spec.get('mode', 'mean')}"
            if hf_resolution_note:
                beat_status = f"{beat_status} | {hf_resolution_note}"
            if beat_agg_warning:
                beat_status = f"{beat_status} | {beat_agg_warning}"
        if bool(enable_measure):
            with torch.no_grad():
                measure_output_raw = predictor.predict(
                    score,
                    force_route="full",
                    iterative_spec=iterative_spec,
                    aggregation_spec=aggregation_spec,
                    measure_spec={"mode": "summary_v1"},
                    return_measure_predictions=True,
                    return_route=False,
                )
            _, _, _, measure_payload = _parse_predict_output(
                measure_output_raw,
                enable_iterative=False,
                enable_beat=False,
                enable_measure=True,
            )
            measure_df = _measure_payload_to_dataframe(measure_payload)
            measure_status = f"Measure-level table ready: rows={len(measure_df)} mode=summary_v1"
            if hf_resolution_note:
                measure_status = f"{measure_status} | {hf_resolution_note}"

        visual_payload = _build_visual_payload(
            score_path=score_path,
            score=score_obj,
            df=display_df,
            tasks=tasks,
            edge_types=[],
            edges_all=edges_all,
            global_key=global_key,
        )
        note_csv = _dataframe_to_csv_file(display_df, "note_predictions")
        beat_csv = _dataframe_to_csv_file(beat_df, "beat_predictions")
        measure_csv = _dataframe_to_csv_file(measure_df, "measure_predictions")
        return (
            display_df,
            status,
            _format_trace(trace, show_trace),
            visual_payload,
            note_csv,
            beat_df,
            beat_status,
            beat_csv,
            measure_df,
            measure_status,
            measure_csv,
        )

        # Write Delta Lake
        output_dir = _derive_output_dir(score_path)
        try:
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
            log_text = _log(log_text, f"Delta Lake written to {output_dir}.")
        except Exception as dl_exc:
            log_text = _log(log_text, f"Delta Lake write failed: {dl_exc}")

        log_text = _log(
            log_text,
            f"Inference done (route={routing.route}). "
            f"Rows={len(display_df)} tasks={','.join(tasks)} aggregation=none.",
        )
        trace_str = _format_trace(trace, show_trace)
        if trace_str:
            log_text = _log(log_text, f"Trace:\n{trace_str}")

        csv_path = _write_csv_to_temp(display_df, score_path)

        return (
            display_df,
            log_text,
            visual_payload,
            csv_path,
            predictions,
            intermediates_state,
            delta_dfs,
            tasks,
            score_path,
            edges_all,
            True,  # model_available
            global_key,
        )
    except Exception as exc:
        log_text = _log(log_text, f"Error: {exc}")
        return (
            pd.DataFrame(),
            log_text,
            {},
            None,
            {},
            {},
            {},
            [],
            "",
            {k: [[], []] for k in DEFAULT_EDGE_TYPES},
            False,
            "",
        )


# ---------------------------------------------------------------------------
# Module 1b: Load Delta Lake
# ---------------------------------------------------------------------------


def load_from_delta_lake(delta_lake_path: Any, log_text: str) -> tuple:
    """Load a Delta Lake output dir from a FileExplorer selection.

    ``delta_lake_path`` is either a string path to metadata.json or a list
    containing one such path (FileExplorer with file_count='single' returns
    a string).
    """
    try:
        # Normalise FileExplorer output
        if isinstance(delta_lake_path, list):
            delta_lake_path = delta_lake_path[0] if delta_lake_path else None
        if delta_lake_path is None:
            raise ValueError("Please select a metadata.json file.")
        meta_path = str(delta_lake_path)
        if not os.path.isabs(meta_path):
            meta_path = str(REPO_ROOT / meta_path)
        if not os.path.exists(meta_path):
            raise ValueError(f"Path does not exist: {meta_path}")

        output_dir = str(Path(meta_path).parent)
        log_text = _log(log_text, f"Loading Delta Lake from {output_dir}...")

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
        display_df = strategy.aggregate(
            probs_df, notes_df, hyperedges_df, metadata, tasks=tasks
        )
        display_df = format_table_output(display_df, tasks)

        log_text = _log(
            log_text,
            f"Loaded Delta Lake from {output_dir}. "
            f"Rows={len(display_df)} tasks={','.join(tasks)}.",
        )

        intermediates_state = {
            "score": None,
            "note_array": None,
            "data": None,
            "score_path": score_path,
            "delta_lake_dir": output_dir,
        }

        delta_dfs = {
            "probs_df": probs_df,
            "notes_df": notes_df,
            "hyperedges_df": hyperedges_df,
            "metadata": metadata,
        }

        # Derive global key from predictions
        try:
            global_key = _derive_global_key(display_df)
        except ValueError as gk_exc:
            global_key = ""
            log_text = _log(log_text, f"Global key derivation failed: {gk_exc}")

        # Add Complete RN column
        if global_key:
            display_df["romanNumeral_full"] = _build_complete_rn_column(
                display_df, global_key
            )

        csv_path = _write_csv_to_temp(display_df, score_path)

        return (
            display_df,
            log_text,
            {},  # visual_payload (no score XML available without inference)
            csv_path,
            {},  # raw_predictions (not available from Delta Lake)
            intermediates_state,
            delta_dfs,
            tasks,
            score_path,
            edges_all,
            False,  # model NOT available
            global_key,
        )
    except Exception as exc:
        log_text = _log(log_text, f"Error loading Delta Lake: {exc}")
        return (
            pd.DataFrame(),
            log_text,
            {},
            None,
            {},
            {},
            {},
            [],
            "",
            {k: [[], []] for k in DEFAULT_EDGE_TYPES},
            False,
            "",
        )


# ---------------------------------------------------------------------------
# Module 2: Post-hoc Aggregation (with caching)
# ---------------------------------------------------------------------------

# Cache: maps strategy name -> display_df so repeat clicks are instant.
_aggregation_cache: Dict[str, pd.DataFrame] = {}


def _clear_aggregation_cache() -> None:
    _aggregation_cache.clear()


def run_aggregation(
    strategy_name: str,
    delta_dfs_state: Any,
    tasks_state: Any,
    score_path_state: str,
    edges_state: Any,
    intermediates_state: Any,
    global_key_text: str,
    log_text: str,
):
    """Apply an aggregation strategy.  Uses a cache so that toggling back and
    forth between strategies is instant.

    Returns: (display_df, log_text, visual_payload, csv_path)
    """
    try:
        strategy_name = (strategy_name or "none").strip().lower()
        tasks = tasks_state or []
        delta_dfs = delta_dfs_state or {}
        intermediates = intermediates_state or {}
        global_key = (global_key_text or "").strip()

        if not delta_dfs or "probs_df" not in delta_dfs:
            raise ValueError(
                "No data available. Run inference or load Delta Lake first."
            )

        # Check cache
        if strategy_name in _aggregation_cache:
            display_df = _aggregation_cache[strategy_name]
            log_text = _log(
                log_text,
                f"Aggregation '{strategy_name}' (cached). Rows={len(display_df)}.",
            )
        else:
            probs_df = delta_dfs["probs_df"]
            notes_df = delta_dfs["notes_df"]
            hyperedges_df = delta_dfs["hyperedges_df"]
            metadata = delta_dfs.get("metadata", {})

            strategy = get_strategy(strategy_name)
            result_df = strategy.aggregate(
                probs_df, notes_df, hyperedges_df, metadata, tasks=tasks
            )
            display_df = format_table_output(result_df, tasks)
            # Add Complete RN column
            if global_key:
                display_df["romanNumeral_full"] = _build_complete_rn_column(
                    display_df, global_key
                )
            _aggregation_cache[strategy_name] = display_df
            log_text = _log(
                log_text,
                f"Aggregation '{strategy_name}' applied. Rows={len(display_df)}.",
            )

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
                global_key=global_key,
            )

        csv_path = _write_csv_to_temp(display_df, score_path)
        return display_df, log_text, visual_payload, csv_path
    except Exception as exc:
        log_text = _log(log_text, f"Aggregation error: {exc}")
        return pd.DataFrame(), log_text, {}, None


def save_delta_lake(
    raw_predictions_state: Any,
    intermediates_state: Any,
    log_text: str,
):
    """Write/update Delta Lake with current data."""
    try:
        intermediates = intermediates_state or {}
        score_obj = intermediates.get("score")
        note_array = intermediates.get("note_array")
        pyg_data = intermediates.get("data")
        score_path = intermediates.get("score_path", "")

        if score_obj is None or note_array is None or pyg_data is None:
            dl_dir = intermediates.get("delta_lake_dir", "")
            if dl_dir:
                return _log(
                    log_text,
                    f"Data was loaded from Delta Lake at {dl_dir}. No new data to write.",
                )
            raise ValueError("No inference data available to save.")

        if not score_path:
            raise ValueError("No score path available.")

        output_dir = _derive_output_dir(score_path)

        predictions = raw_predictions_state
        if not predictions or not isinstance(predictions, dict):
            raise ValueError("No raw predictions available.")

        task_dict: Dict[str, int] = {}
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
            metadata={"score_path": score_path},
        )
        return _log(log_text, f"Delta Lake saved to {output_dir}.")
    except Exception as exc:
        return _log(log_text, f"Save error: {exc}")


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
    aggregation_mode: str,
    voter_checkpoint_path: str,
    enable_beat: bool,
    beat_aggregation_mode: str,
    beat_voter_checkpoint_path: str,
    beat_tasks: List[str],
    enable_measure: bool,
    show_trace: bool,
    intermediates_state: Any,
    edges_state: Any,
    log_text: str,
):
    """Run edit-conditioned masked inference.

    Returns same shape as run_full_inference.
    """
    try:
        log_text = _log(log_text, "Starting edit-conditioned inference...")
        score_path = _resolve_score_path(score_file)
        score = _load_score(score_path)
        tasks = _resolve_selected_tasks(task_labels, tasks_csv)
        resolved_paths, hf_resolution_note = _resolve_runtime_artifact_paths(
            full_ckpt=full_ckpt,
            masked_ckpt=masked_ckpt,
            voter_ckpt=voter_checkpoint_path,
            beat_voter_ckpt=beat_voter_checkpoint_path,
        )
        predictor = _get_predictor(
            resolved_paths["full_ckpt"],
            resolved_paths["masked_ckpt"],
            device,
        )

        edited_df = (
            pd.DataFrame(edited_table) if edited_table is not None else pd.DataFrame()
        )
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
            voter_path=resolved_paths["voter_ckpt"],
        )
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
        predictions, trace, _, _ = _parse_predict_output(
            output,
            enable_iterative=bool(enable_iterative),
            enable_beat=False,
            enable_measure=False,
        )

        out_df = predictions_to_dataframe(
            score=score_obj,
            predictions=predictions,
            tasks=tasks,
            include_confidence=True,
            include_class_ids=False,
        )
        out_df = _apply_timing_from_predictions(out_df, predictions)
        display_df = format_table_output(out_df, tasks)

        num_notes = len(note_array)
        if pyg_data is not None:
            edges_all = _edges_from_pyg_data(pyg_data, num_notes)
        else:
            edges_all = edges_state or {k: [[], []] for k in DEFAULT_EDGE_TYPES}

        new_intermediates = {
            "score": score_obj,
            "note_array": note_array,
            "data": pyg_data,
            "score_path": score_path,
        }

        # Precompute DataFrames for aggregation
        delta_dfs = _precompute_delta_dfs(predictions, new_intermediates)
        _clear_aggregation_cache()

        # Derive global key from predictions
        try:
            global_key = _derive_global_key(display_df)
        except ValueError as gk_exc:
            global_key = ""
            log_text = _log(log_text, f"Global key derivation failed: {gk_exc}")

        # Add Complete RN column
        if global_key:
            display_df["romanNumeral_full"] = _build_complete_rn_column(
                display_df, global_key
            )

        status = (
            f"Partial inference done using route={routing.route} checkpoint={routing.checkpoint_path}. "
            f"Known rows={info.get('num_known', 0)} target rows={info.get('num_targets', 0)} "
            f"aggregation={aggregation_spec.get('mode', 'mean')}"
        )
        if hf_resolution_note:
            status = f"{status} | {hf_resolution_note}"
        if aggregation_warning:
            status = f"{status} | {aggregation_warning}"

        beat_df = pd.DataFrame()
        beat_status = "Beat-level aggregation disabled."
        measure_df = pd.DataFrame()
        measure_status = "Measure-level summary disabled."
        if bool(enable_beat):
            selected_beat_tasks = [t for t in (beat_tasks or []) if t]
            if not selected_beat_tasks:
                selected_beat_tasks = list(DEFAULT_BEAT_TASKS)
            beat_agg_spec, beat_agg_warning = _build_aggregation_spec(
                aggregation_mode=beat_aggregation_mode,
                voter_path=resolved_paths["beat_voter_ckpt"],
                beat_tasks=selected_beat_tasks,
            )
            with torch.no_grad():
                beat_output_raw = predictor.predict(
                    score,
                    user_edits=user_edits,
                    masked_spec=masked_spec,
                    iterative_spec=iterative_spec,
                    aggregation_spec=beat_agg_spec,
                    return_beat_predictions=True,
                    return_route=False,
                )
            _, _, beat_payload, _ = _parse_predict_output(
                beat_output_raw,
                enable_iterative=False,
                enable_beat=True,
                enable_measure=False,
            )
            beat_df = _beat_payload_to_dataframe(
                beat_payload=beat_payload,
                beat_tasks=selected_beat_tasks,
            )
            beat_status = f"Beat-level table ready: rows={len(beat_df)} mode={beat_agg_spec.get('mode', 'mean')}"
            if hf_resolution_note:
                beat_status = f"{beat_status} | {hf_resolution_note}"
            if beat_agg_warning:
                beat_status = f"{beat_status} | {beat_agg_warning}"
        if bool(enable_measure):
            with torch.no_grad():
                measure_output_raw = predictor.predict(
                    score,
                    user_edits=user_edits,
                    masked_spec=masked_spec,
                    iterative_spec=iterative_spec,
                    aggregation_spec=aggregation_spec,
                    measure_spec={"mode": "summary_v1"},
                    return_measure_predictions=True,
                    return_route=False,
                )
            _, _, _, measure_payload = _parse_predict_output(
                measure_output_raw,
                enable_iterative=False,
                enable_beat=False,
                enable_measure=True,
            )
            measure_df = _measure_payload_to_dataframe(measure_payload)
            measure_status = f"Measure-level table ready: rows={len(measure_df)} mode=summary_v1"
            if hf_resolution_note:
                measure_status = f"{measure_status} | {hf_resolution_note}"

        visual_payload = _build_visual_payload(
            score_path=score_path,
            score=score_obj,
            df=display_df,
            tasks=tasks,
            edge_types=[],
            edges_all=edges_all,
            global_key=global_key,
        )

        log_text = _log(
            log_text,
            f"Edit-conditioned inference done (route={routing.route}). "
            f"Known rows={info.get('num_known', 0)} target rows={info.get('num_targets', 0)}.",
        )
        trace_str = _format_trace(trace, show_trace)
        if trace_str:
            log_text = _log(log_text, f"Trace:\n{trace_str}")

        csv_path = _write_csv_to_temp(display_df, score_path)

        return (
            display_df,
            log_text,
            visual_payload,
            csv_path,
            predictions,
            new_intermediates,
            delta_dfs,
            tasks,
            score_path,
            edges_all,
            True,
            global_key,
        )
        note_csv = _dataframe_to_csv_file(display_df, "note_predictions")
        beat_csv = _dataframe_to_csv_file(beat_df, "beat_predictions")
        measure_csv = _dataframe_to_csv_file(measure_df, "measure_predictions")
        return (
            display_df,
            status,
            _format_trace(trace, show_trace),
            visual_payload,
            note_csv,
            beat_df,
            beat_status,
            beat_csv,
            measure_df,
            measure_status,
            measure_csv,
        )
    except Exception as exc:
        return pd.DataFrame(), f"Error: {exc}", "", {}


def refresh_visual_tab(
    verovio_score_file: Any,
    task_labels: List[str],
    tasks_csv: str,
    table_data: Any,
    edge_type_labels: List[str],
    nct_color_labels: List[str],
    global_key_text: str,
    visual_state: Dict[str, Any],
    intermediates_state: Any,
    edges_state: Any,
    log_text: str,
):
    try:
        selected_edge_types = [
            k for k, label in EDGE_LABELS.items() if label in (edge_type_labels or [])
        ]
        nct_color = bool(
            nct_color_labels and "Colour non-chord tones grey" in nct_color_labels
        )
        tasks = _resolve_selected_tasks(task_labels, tasks_csv)
        intermediates = intermediates_state or {}
        edges_all = edges_state or {k: [[], []] for k in DEFAULT_EDGE_TYPES}
        global_key = (global_key_text or "").strip()

        # Resolve score: prefer the Verovio-tab file upload, fall back to intermediates
        score_obj = None
        note_array = None
        score_path = ""
        if verovio_score_file is not None:
            score_path = _resolve_score_path(verovio_score_file)
            score_obj = _load_score(score_path)
            note_array = _sorted_note_array(score_obj)
        if score_obj is None:
            score_obj = intermediates.get("score")
            note_array = intermediates.get("note_array")
            score_path = intermediates.get("score_path", "")

        df = pd.DataFrame(table_data) if table_data is not None else pd.DataFrame()

        if score_obj is not None and note_array is not None and len(df) > 0:
            # Check note-count mismatch between score and predictions
            n_score = len(note_array)
            n_table = len(df)
            if n_score != n_table:
                log_text = _log(
                    log_text,
                    f"Warning: score has {n_score} notes but predictions table has {n_table} rows. "
                    f"Rendering min({n_score}, {n_table}) notes; overlay alignment may be approximate.",
                )
            payload = _build_visual_payload(
                score_path=score_path,
                score=score_obj,
                df=df,
                tasks=tasks,
                edge_types=selected_edge_types,
                edges_all=edges_all,
                global_key=global_key,
            )
            # Apply NCT coloring
            if nct_color:
                n = min(len(note_array), len(df))
                note_colors = _compute_nct_note_colors(df, n)
                payload["note_colors"] = {str(k): v for k, v in note_colors.items()}
        elif isinstance(visual_state, dict) and visual_state:
            payload = dict(visual_state)
            payload.setdefault("meta", {})
            payload["meta"]["visible_edge_types"] = selected_edge_types
        else:
            raise ValueError(
                "No score available. Upload a score in the Verovio tab "
                "(or run inference in Module 1a) and ensure predictions are loaded."
            )

        html_frame = _build_verovio_html(payload)
        note_count = len(payload.get("notes", []))
        log_text = _log(
            log_text,
            f"Visual refreshed: notes={note_count}, "
            f"visible edges={','.join(selected_edge_types) if selected_edge_types else 'none'}"
            f"{', NCT coloring ON' if nct_color else ''}.",
        )
        return html_frame, log_text, payload
    except Exception as exc:
        fallback = (
            "<div style='padding:12px;border:1px solid #d1d5db;border-radius:10px;background:#fff;'>"
            f"Visual rendering error: {html_lib.escape(str(exc))}"
            "</div>"
        )
        log_text = _log(log_text, f"Visual error: {exc}")
        return (
            fallback,
            log_text,
            visual_state if isinstance(visual_state, dict) else {},
        )


# ---------------------------------------------------------------------------
# UI Builder
# ---------------------------------------------------------------------------


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="AnalysisGNN Hybrid Inference") as demo:
        gr.Markdown(
            """
# AnalysisGNN Hybrid Inference

**Module 1** — Data source: run inference on a score or load existing Delta Lake results.
**Module 2** — Analysis results: view, aggregate, export, and visualise.
**Module 3** — Edit-conditioned re-inference (requires live model from Module 1a).
""")

        # ---- gr.State objects ----
        visual_payload_state = gr.State({})
        raw_predictions_state = gr.State({})
        intermediates_state = gr.State({})
        delta_dfs_state = gr.State({})  # precomputed probs/notes/hyperedges DFs
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
                    full_ckpt = gr.Textbox(
                        label="Base (Full Inference) Checkpoint",
                        value=DEFAULT_FULL_CKPT,
                    )
                    masked_ckpt = gr.Textbox(
                        label="Masked (Partial Inference) Checkpoint",
                        value=DEFAULT_MASKED_CKPT,
                    )
                    device = gr.Dropdown(
                        label="Device", choices=["auto", "cuda", "cpu"], value="auto"
                    )

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

                run_inference_btn = gr.Button("Run Inference", variant="primary")

            # ------ Tab 1b: Load Delta Lake ------
            with gr.Tab("Load Delta Lake"):
                gr.Markdown(
                    "Select a `metadata.json` file from an existing Delta Lake output directory."
                )
                delta_lake_explorer = gr.FileExplorer(
                    glob="**/metadata.json",
                    root_dir=str(REPO_ROOT),
                    file_count="single",
                    label="Select metadata.json",
                )
                load_delta_btn = gr.Button("Load", variant="primary")

        # ==================================================================
        # MODULE 2: ANALYSIS RESULTS
        # ==================================================================
        gr.Markdown("---")
        gr.Markdown("## Module 2: Analysis Results")
                with gr.Row():
                    aggregation_mode = gr.Dropdown(
                        label="Aggregation Mode",
                        choices=["Mean", "Voter"],
                        value="Mean",
                    )
                    voter_checkpoint_path = gr.Textbox(
                        label="Voter Checkpoint Path",
                        value=DEFAULT_VOTER_CKPT,
                        info="Optional. Required only when Aggregation Mode is Voter.",
                    )
                with gr.Accordion("Beat-Level Aggregation (Optional)", open=False):
                    with gr.Row():
                        enable_beat = gr.Checkbox(
                            label="Enable Beat-Level Aggregation",
                            value=False,
                        )
                        beat_aggregation_mode = gr.Dropdown(
                            label="Beat Aggregation Mode",
                            choices=["Mean", "Voter", "Voter Consistent Beat"],
                            value="Mean",
                        )
                    with gr.Row():
                        beat_voter_checkpoint_path = gr.Textbox(
                            label="Beat Voter Checkpoint Path",
                            value=DEFAULT_BEAT_VOTER_CKPT,
                            info="Required for Voter or Voter Consistent Beat.",
                        )
                    beat_tasks = gr.CheckboxGroup(
                        label="Beat Tasks",
                        choices=[
                            (AVAILABLE_TASKS.get(t, t), t)
                            for t in BEAT_TASK_CHOICES
                            if t in AVAILABLE_TASKS
                        ],
                        value=[t for t in DEFAULT_BEAT_TASKS if t in AVAILABLE_TASKS],
                        info="Tasks to include in the beat-level table.",
                    )
                with gr.Accordion("Measure-Level Summary (Optional)", open=False):
                    enable_measure = gr.Checkbox(
                        label="Enable Measure-Level Summary",
                        value=False,
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
            global_key_field = gr.Textbox(
                label="Global Key",
                value="",
                info="Auto-derived from predictions (most frequent tonic). Editable.",
                max_lines=1,
                scale=0,
            )
            aggregation_dropdown = gr.Dropdown(
                label="Aggregation Strategy",
                choices=[s.capitalize() for s in list_strategies()],
                value="None",
                info="Select an aggregation strategy and click 'Aggregate!' to apply.",
            )
            aggregate_btn = gr.Button("Aggregate!", variant="secondary")
            save_delta_btn = gr.Button("Save Delta Lake", variant="secondary")
            csv_download = gr.DownloadButton("Download CSV", variant="secondary")

        with gr.Tabs():
            # ------ Tab: Analysis Results ------
            with gr.Tab("Analysis Results"):
                table = gr.Dataframe(
                    label="Predictions (editable)",
                    interactive=True,
                    wrap=True,
                )
                note_csv_download = gr.File(
                    label="Download Predictions CSV",
                    interactive=False,
                )
                status = gr.Textbox(label="Status", interactive=False)
                trace_output = gr.Textbox(
                    label="Iteration Trace", interactive=False, lines=12
                )
                beat_status = gr.Textbox(label="Beat Status", interactive=False)
                beat_table = gr.Dataframe(
                    label="Beat-Level Table",
                    interactive=False,
                    wrap=True,
                )
                beat_csv_download = gr.File(
                    label="Download Beat-Level CSV",
                    interactive=False,
                )
                measure_status = gr.Textbox(label="Measure Status", interactive=False)
                measure_table = gr.Dataframe(
                    label="Measure-Level Table",
                    interactive=False,
                    wrap=True,
                )
                measure_csv_download = gr.File(
                    label="Download Measure-Level CSV",
                    interactive=False,
                )

            # ------ Tab: Verovio Visual Score ------
            with gr.Tab("Verovio Visual Score"):
                gr.Markdown(
                    "Render the uploaded score with graph overlays. "
                    "Click a note in the score to inspect note-level predictions and complete RN decoding. "
                    "Upload a score below (auto-filled from Module 1a) or load an alternative edition."
                )
                verovio_score_file = gr.File(
                    label="Score for Verovio (auto-filled from Module 1a)",
                    file_types=[".xml", ".musicxml", ".mxl"],
                    type="filepath",
                )
                visual_edge_types = gr.CheckboxGroup(
                    label="Visible Edge Types",
                    choices=[EDGE_LABELS[k] for k in DEFAULT_EDGE_TYPES],
                    value=[],
                    info="Edges are hidden by default; select one or more types and refresh.",
                )
                nct_color_group = gr.CheckboxGroup(
                    label="Non-chord tones",
                    choices=["Colour non-chord tones grey"],
                    value=[],
                    info="Chord tones -> black, non-chord tones -> light grey, scaled by confidence.",
                )
                refresh_visual_btn = gr.Button("Refresh Visual", variant="secondary")
                refresh_visual_btn = gr.Button(
                    "Refresh Visual from Latest Predictions", variant="secondary"
                )
                visual_html = gr.HTML(
                    value=(
                        "<div style='padding:12px;border:1px solid #d1d5db;border-radius:10px;background:#fff;'>"
                        "Run inference or load a Delta Lake, upload a score, then click 'Refresh Visual'."
                        "</div>"
                    ),
                    label="Verovio Score + Graph",
                )

        # ==================================================================
        # MODULE 3: EDIT-CONDITIONED RE-INFERENCE
        # ==================================================================
        gr.Markdown("---")
        gr.Markdown("## Module 3: Edit-Conditioned Re-Inference")
        gr.Markdown(
            "*Requires a live model (run inference in Module 1a first). "
            "Grayed out when loading from Delta Lake.*"
        )

        tasks_csv = gr.Textbox(
            label="Tasks Override (internal keys CSV, optional)",
            value=DEFAULT_TASKS,
            info="Used only if no task is selected above. Example: romanNumeral,localkey,quality",
            interactive=False,
        )

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
                info="Rows treated as known/corrected labels (context). Example: 1-8, 12, 20-24",
                interactive=False,
            )
            target_rows_expr = gr.Textbox(
                label="Target Rows (1-based, optional)",
                value="",
                info="Rows to re-predict. Empty means all non-known rows.",
                interactive=False,
            )

        update_analysis_btn = gr.Button(
            "Update Analysis", variant="stop", interactive=False
        )

        # ==================================================================
        # DIAGNOSTICS (single log for everything)
        # ==================================================================
        gr.Markdown("---")
        gr.Markdown("## Log")
        show_trace = gr.Checkbox(label="Show Iteration Trace", value=False)
        log_output = gr.Textbox(label="Log", interactive=False, lines=12, max_lines=40)

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
                gr.update(interactive=interactive),  # tasks_csv
            )

        # Clear aggregation cache on new inference
        def _on_new_data(*args):
            _clear_aggregation_cache()

        # ---- Module 1a: Run Inference ----
        run_inference_btn.click(
            fn=_on_new_data,
            inputs=[],
            outputs=[],
        ).then(
            fn=run_full_inference,
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
            enable_beat: bool,
            beat_aggregation_mode: str,
            beat_voter_checkpoint_path: str,
            beat_tasks: List[str],
            enable_measure: bool,
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
                    enable_beat=enable_beat,
                    beat_aggregation_mode=beat_aggregation_mode,
                    beat_voter_checkpoint_path=beat_voter_checkpoint_path,
                    beat_tasks=beat_tasks,
                    enable_measure=enable_measure,
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
                    enable_beat=enable_beat,
                    beat_aggregation_mode=beat_aggregation_mode,
                    beat_voter_checkpoint_path=beat_voter_checkpoint_path,
                    beat_tasks=beat_tasks,
                    enable_measure=enable_measure,
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
                enable_beat=enable_beat,
                beat_aggregation_mode=beat_aggregation_mode,
                beat_voter_checkpoint_path=beat_voter_checkpoint_path,
                beat_tasks=beat_tasks,
                enable_measure=enable_measure,
                target_only_update=target_only_update,
                show_trace=show_trace,
            )

        run_btn.click(
            fn=run_by_mode,
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
                aggregation_mode,
                voter_checkpoint_path,
                enable_beat,
                beat_aggregation_mode,
                beat_voter_checkpoint_path,
                beat_tasks,
                enable_measure,
                target_only_update,
                show_trace,
                log_output,
            ],
            outputs=[
                table,
                log_output,
                visual_payload_state,
                csv_download,
                raw_predictions_state,
                intermediates_state,
                delta_dfs_state,
                tasks_state,
                score_path_state,
                edges_state,
                model_available_state,
                global_key_field,
            ],
        ).then(
            fn=_update_module3_interactivity,
            inputs=[model_available_state],
            outputs=[
                target_only_update,
                known_rows_expr,
                target_rows_expr,
                update_analysis_btn,
                tasks_csv,
            ],
        )
        # Auto-fill Verovio score file from Module 1a
        score_file.change(
            fn=lambda f: f,
            inputs=[score_file],
            outputs=[verovio_score_file],
        )

        # ---- Module 1b: Load Delta Lake ----
        load_delta_btn.click(
            fn=_on_new_data,
            inputs=[],
            outputs=[],
        ).then(
            fn=load_from_delta_lake,
            inputs=[delta_lake_explorer, log_output],
            outputs=[
                table,
                log_output,
                visual_payload_state,
                csv_download,
                raw_predictions_state,
                intermediates_state,
                delta_dfs_state,
                tasks_state,
                score_path_state,
                edges_state,
                model_available_state,
                global_key_field,
            ],
        ).then(
            fn=_update_module3_interactivity,
            inputs=[model_available_state],
            outputs=[
                target_only_update,
                known_rows_expr,
                target_rows_expr,
                update_analysis_btn,
                tasks_csv,
            ],
        )

        # ---- Module 2: Aggregate! ----
        aggregate_btn.click(
            fn=run_aggregation,
            inputs=[
                aggregation_dropdown,
                delta_dfs_state,
                tasks_state,
                score_path_state,
                edges_state,
                intermediates_state,
                global_key_field,
                log_output,
            ],
            outputs=[table, log_output, visual_payload_state, csv_download],
        )

        # ---- Module 2: Save Delta Lake ----
        save_delta_btn.click(
            fn=save_delta_lake,
            inputs=[
                raw_predictions_state,
                intermediates_state,
                log_output,
            ],
            outputs=[log_output],
        )

        # ---- Module 2: Refresh Visual ----
            outputs=[
                table,
                status,
                trace_output,
                visual_payload_state,
                note_csv_download,
                beat_table,
                beat_status,
                beat_csv_download,
                measure_table,
                measure_status,
                measure_csv_download,
            ],
        )

        refresh_visual_btn.click(
            fn=refresh_visual_tab,
            inputs=[
                verovio_score_file,
                task_selector,
                tasks_csv,
                table,
                visual_edge_types,
                nct_color_group,
                global_key_field,
                visual_payload_state,
                intermediates_state,
                edges_state,
                log_output,
            ],
            outputs=[visual_html, log_output, visual_payload_state],
        )

        # ---- Module 3: Update Analysis (edit-conditioned) ----
        update_analysis_btn.click(
            fn=_on_new_data,
            inputs=[],
            outputs=[],
        ).then(
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
                log_output,
            ],
            outputs=[
                table,
                log_output,
                visual_payload_state,
                csv_download,
                raw_predictions_state,
                intermediates_state,
                delta_dfs_state,
                tasks_state,
                score_path_state,
                edges_state,
                model_available_state,
                global_key_field,
            ],
        )

    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch()
