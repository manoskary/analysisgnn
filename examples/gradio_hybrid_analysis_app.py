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
from analysisgnn.utils.chord_symbols import build_beat_chord_symbol_row
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


def _prepare_prediction_table_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Apply app-only presentation cleanup for the note prediction table."""
    if df is None or len(df) == 0:
        return df
    out = df.copy()

    confidence_cols = [col for col in out.columns if col.endswith("_confidence")]
    for col in confidence_cols:
        numeric = pd.to_numeric(out[col], errors="coerce")
        if numeric.notna().any():
            out[col] = numeric.round(3)

    if "romanNumeral_full" in out.columns:
        cols = [col for col in out.columns if col != "romanNumeral_full"]
        if "pitch_midi" in cols:
            insert_at = cols.index("pitch_midi") + 1
        elif "cadence" in cols:
            insert_at = cols.index("cadence")
        else:
            insert_at = len(cols)
        cols.insert(insert_at, "romanNumeral_full")
        out = out[cols]

    return out

_GLOBAL_KEY_K = 5

_KEY_TASKS = ("localkey", "tonkey")


def _fix_key_mode(df: pd.DataFrame) -> pd.DataFrame:
    """Correct the case of ``localkey`` and ``tonkey`` predictions in-place.

    The model's 50-class key vocabulary encodes mode via case (uppercase =
    major, lowercase = minor), but the softmax argmax does not reliably
    land on the correct case variant.  This function infers the true mode
    from the ``romanNumeral`` column: within each key pitch-class group,
    if the count of minor-tonic chords (``"i"``) is not lower than the
    count of major-tonic chords (``"I"``), all occurrences of that key are
    lowercased (= minor).

    The correction is applied to every key-task column present in *df*
    (``localkey``, ``tonkey``).

    Returns *df* (modified in-place) for chaining convenience.

    .. note::
        This is a workaround for a model deficiency — the localkey/tonkey
        softmax does not reliably separate major from minor classes for the
        same pitch class.  Remove this function once the model outputs have
        been fixed (e.g. by retraining with a loss that penalises mode
        confusion, or by collapsing major/minor into a single pitch-class
        prediction and inferring mode from a separate head).
    """
    if df is None or len(df) == 0:
        return df
    if "romanNumeral" not in df.columns:
        return df

    key_cols = [c for c in _KEY_TASKS if c in df.columns]
    if not key_cols:
        return df

    tonic_mask = df["romanNumeral"].isin(["I", "i"])
    if not tonic_mask.any():
        return df

    for col in key_cols:
        tonic_rows = df.loc[tonic_mask, [col, "romanNumeral"]].copy()
        tonic_rows["_pc"] = tonic_rows[col].astype(str).str.upper()
        minor_mode: dict[str, bool] = {}
        for pc, grp in tonic_rows.groupby("_pc"):
            n_minor = int((grp["romanNumeral"] == "i").sum())
            n_major = int((grp["romanNumeral"] == "I").sum())
            minor_mode[str(pc)] = not (n_minor < n_major)

        vals = df[col].astype(str).values.copy()
        for i, v in enumerate(vals):
            if not v or v == "None":
                continue
            pc = v.upper()
            if pc in minor_mode and minor_mode[pc]:
                vals[i] = v[0].lower() + v[1:]
        df[col] = vals

    return df


def _derive_global_key(df: pd.DataFrame, k: int = _GLOBAL_KEY_K) -> str:
    """Derive the global key from the first *k* tonic chords.

    Takes the first *k* rows (in score order) where ``romanNumeral`` is
    ``"I"`` or ``"i"`` and lets their ``localkey`` values vote.  Grouping
    is case-insensitive (so ``"C"`` and ``"c"`` count toward the same
    pitch class); among the winning pitch class the most frequent cased
    variant determines the mode.

    Using only the first *k* tonics avoids bias from extended middle
    sections whose key may outnumber the main key's tonic chords.

    Returns the raw ``localkey`` string (e.g. ``"G"``, ``"c"``, ``"A-"``).

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
    candidates = df.loc[mask, "localkey"].astype(str).head(k)
    if len(candidates) == 0:
        raise ValueError(
            "Cannot derive global key: no rows with romanNumeral 'I' or 'i'."
        )
    # Group by pitch class (case-insensitive) to find the winning tonic
    lk_upper = candidates.str.upper()
    best_pc = str(lk_upper.value_counts().index[0])
    # Among those, take the most frequent cased variant for mode.
    pc_mask = lk_upper == best_pc
    return str(candidates[pc_mask].value_counts().index[0])


def _format_dcml_with_global_key(dcml: str, global_key: str) -> str:
    """Reformat a FlexOHR DCML string to show the global key as a prefix.

    FlexOHR's ``OHR.to_format('dcml')`` produces ``V7/I/G`` (global key
    last).  This function reformats to ``G: V7/I`` (global key first,
    separated by colon + space).
    """
    parts = dcml.split("/")
    if len(parts) < 2:
        return f"{global_key}: {dcml}"
    # The last segment is the global key — drop it and use the explicit
    # global_key parameter (which preserves the user's chosen mode/case).
    return f"{global_key}: {'/'.join(parts[:-1])}"


def _agnn_to_flx_pitch(name: str) -> str:
    """Normalise an AnalysisGNN pitch-class string for FlexOHR.

    AnalysisGNN uses ``-`` for flat (e.g. ``A-``, ``B--``); FlexOHR's
    ``SPC`` expects ``b`` (e.g. ``Ab``, ``Bbb``).
    """
    return name.replace("-", "b")


def _prepare_df_for_flexohr(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a prediction DataFrame for FlexOHR consumption.

    Normalises string ``"None"`` to NaN in ``degree2`` and converts numeric
    task columns (``degree1``, ``degree2``, ``inversion``) from strings to
    floats, as ``build_ohrs_from_dataframe`` expects.  Also normalises
    pitch-class strings (``localkey``, ``tonkey``) from ``-`` to ``b``.
    """
    work = df.copy()
    # degree2: string "None" -> NaN
    if "degree2" in work.columns:
        work["degree2"] = work["degree2"].replace({"None": np.nan, "": np.nan})
        work["degree2"] = pd.to_numeric(work["degree2"], errors="coerce")
    # degree1, inversion: ensure numeric
    for col in ("degree1", "inversion"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    # Normalise pitch-class columns
    for col in ("localkey", "tonkey"):
        if col in work.columns:
            work[col] = work[col].astype(str).str.replace("-", "b", regex=False)
    return work


def _build_ohrs_check(df: pd.DataFrame, global_key: str) -> None:
    """Validate preconditions for OHR construction.

    Raises
    ------
    ValueError
        If *global_key* is empty, *df* is empty, or required columns are
        missing.
    """
    if not global_key:
        raise ValueError("Cannot build OHRs: global_key is required.")
    if df is None or len(df) == 0:
        raise ValueError("Cannot build OHRs: DataFrame is empty.")
    required = ["degree1", "degree2", "inversion", "quality", "localkey"]
    missing = [k for k in required if k not in df.columns]
    if missing:
        raise ValueError(
            f"Cannot build OHRs: missing columns {missing}."
        )


def _build_ohrs(df: pd.DataFrame, global_key: str) -> list:
    """Build OHRs from a prediction DataFrame using FlexOHR's bulk builder.

    Returns a list of OHRs (one per row).  Validates preconditions first,
    then delegates to ``build_ohrs_from_dataframe``.
    """
    _build_ohrs_check(df, global_key)
    gk = _agnn_to_flx_pitch(global_key)
    work = _prepare_df_for_flexohr(df)
    return flx.codecs.analysisgnn.build_ohrs_from_dataframe(work, gk)


def _build_complete_rn_column(df: pd.DataFrame, global_key: str) -> pd.Series:
    """Build the Complete RN column using FlexOHR OHR objects.

    Each row's principal task predictions (degree1, degree2, inversion,
    quality, localkey) are used to construct a FlexOHR OHR, which is then
    rendered via ``.to_format('dcml')``.

    Structural failures (empty global key, missing columns) raise.
    Per-row failures (e.g. NaN degree values for individual notes) produce
    empty strings — these are data-quality issues, not programming errors.
    """
    if df is None or len(df) == 0:
        return pd.Series(dtype=object)
    # Validate structural preconditions (raises on failure)
    _build_ohrs_check(df, global_key)

    gk = _agnn_to_flx_pitch(global_key)
    work = _prepare_df_for_flexohr(df)

    # Identify rows with valid numeric core columns (NaN = no label)
    core_numeric = ["degree1", "inversion"]
    valid_mask = work[core_numeric].notna().all(axis=1)

    out = pd.Series("", index=df.index, dtype=object)
    valid_df = work.loc[valid_mask]
    if len(valid_df) > 0:
        ohrs = flx.codecs.analysisgnn.build_ohrs_from_dataframe(valid_df, gk)
        for ohr, idx in zip(ohrs, valid_df.index):
            dcml = ohr.to_format("dcml")
            out.at[idx] = _format_dcml_with_global_key(dcml, global_key)
    return out


def _enumerate_rn_candidates(
    display_df: pd.DataFrame,
    probs_df: pd.DataFrame,
    notes_df: pd.DataFrame,
    hyperedges_df: pd.DataFrame,
    global_key: str,
    k: int = 3,
    top_n: int = 3,
) -> Dict[str, List[Dict[str, Any]]]:
    """Enumerate top-N Roman-numeral candidates per beat group.

    Returns a dict mapping ``note_id`` to a list of candidate dicts, each
    containing ``dcml``, ``score``, and ``expected`` (task→value mapping
    for agreement coloring).
    """
    from analysisgnn.aggregation.roman_numeral import (
        _derive_validation_labels,
        enumerate_roman_numerals,
    )
    from analysisgnn.aggregation.scoring import GeometricMeanScorer, ScoringContext

    gk = _agnn_to_flx_pitch(global_key)

    # Build full scoring context from Delta-style DataFrames
    edges_df = pd.DataFrame({"src": pd.Series(dtype=str), "dst": pd.Series(dtype=str), "edge_type": pd.Series(dtype=str)})
    ctx = ScoringContext(
        note_ids=list(notes_df["note_id"]),
        notes=notes_df,
        probabilities=probs_df,
        edges=edges_df,
    )

    # Beat groups: note_id -> group_id
    beat_groups = hyperedges_df[hyperedges_df["edge_type"] == "beat"]
    if beat_groups.empty:
        # Fallback to onset groups
        beat_groups = hyperedges_df[hyperedges_df["edge_type"] == "onset"]
    if beat_groups.empty:
        return {}

    group_note_map = beat_groups.groupby("group_id")["note_id"].apply(list).to_dict()

    # note_id -> pitch_spelling (for tpc_in_label / note_degree derivation)
    note_pitch: Dict[str, str] = {}
    if "pitch_spelling" in notes_df.columns:
        note_pitch = dict(zip(notes_df["note_id"], notes_df["pitch_spelling"]))

    # note_id -> localkey from display_df (for note_degree)
    note_localkey: Dict[str, str] = {}
    if "note_id" in display_df.columns and "localkey" in display_df.columns:
        note_localkey = dict(zip(display_df["note_id"], display_df["localkey"]))

    scorer = GeometricMeanScorer()
    result: Dict[str, List[Dict[str, Any]]] = {}

    for gid, gnotes in group_note_map.items():
        # Filter to notes that exist in the scoring context
        valid_notes = [nid for nid in gnotes if nid in ctx.note_id_set]
        if not valid_notes:
            continue
        try:
            sub = ctx.subcontext(valid_notes)
            candidates, _trace = enumerate_roman_numerals(
                sub, gk, k=k, top_n=top_n, scorer=scorer,
            )
        except Exception:
            continue

        # Build per-candidate expected labels (core from candidate + validation from OHR)
        group_candidates: List[Dict[str, Any]] = []
        for cand in candidates:
            base_expected = dict(cand.candidate)  # core tasks
            try:
                val_labels = _derive_validation_labels(cand.ohr, gk)
                base_expected.update(val_labels)
            except Exception:
                pass
            # Fallback: tonkey = localkey when no tonicization
            if "tonkey" not in base_expected:
                d2 = base_expected.get("degree2", "None")
                if d2 in ("None", "", None):
                    base_expected["tonkey"] = base_expected.get("localkey", "")
            group_candidates.append({
                "dcml": _format_dcml_with_global_key(cand.dcml, global_key),
                "score": round(cand.result.core.score, 4),
                "base_expected": base_expected,
                "_raw_dcml": cand.dcml,
            })

        # Per-note candidates: add note-level expected (tpc_in_label, note_degree)
        for nid in valid_notes:
            note_cands: List[Dict[str, Any]] = []
            for gc in group_candidates:
                expected = dict(gc["base_expected"])
                # Derive note-level expected from candidate OHR + note pitch
                pitch = note_pitch.get(nid, "")
                if pitch:
                    import re as _re

                    m = _re.match(r"^([A-Ga-g][#b]*)(\d+)?$", pitch)
                    if m:
                        pc = m.group(1)
                        # tpc_in_label: check if pitch is in chord components
                        try:
                            cand_obj = [c for c in candidates if c.dcml == gc["_raw_dcml"]][0]
                            resolved = cand_obj.ohr.resolve()
                            chord_names = set()
                            for comp in resolved.components("b", depth=1):
                                if hasattr(comp, "value") and hasattr(comp.value, "name"):
                                    chord_names.add(comp.value.name)
                            expected["tpc_in_label"] = "True" if pc in chord_names else "False"
                        except Exception:
                            pass
                        # note_degree from localkey + pitch
                        lk = note_localkey.get(nid, "")
                        if lk and lk not in ("", "None", "nan"):
                            try:
                                from flexohr.paradigms.pitchspace.pitch import (
                                    SpecificPitchClass as SPC,
                                )
                                from flexohr.paradigms.pitchspace.scale import (
                                    get_scale,
                                    infer_collection_type,
                                )

                                lk_str = lk.replace("-", "b")
                                lk_coll = infer_collection_type(lk_str)
                                lk_root = SPC(lk_str[0].upper() + lk_str[1:])
                                lk_scale = get_scale(lk_coll, lk_root)
                                note_spc = SPC(pc)
                                sic = note_spc - lk_root
                                sd = lk_scale.make_scale_degree(sic)
                                expected["note_degree"] = sd.to_format("analysisgnn")
                            except Exception:
                                pass

                # Copy tpc_in_label -> pitch_spelling for Pitch tile coloring
                if "tpc_in_label" in expected:
                    expected["pitch_spelling"] = expected["tpc_in_label"]
                note_cands.append({
                    "dcml": gc["dcml"],
                    "score": gc["score"],
                    "expected": expected,
                })
            result[nid] = note_cands

    return result


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
    rn_candidates_map: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Build the Verovio overlay payload."""
    n = min(len(df), len(note_array))
    data = df.iloc[:n].reset_index(drop=True).copy()

    # Use top-1 enumerated candidate as the Complete RN when available
    if rn_candidates_map and "note_id" in data.columns:
        rn_vals = []
        for idx in range(n):
            nid = str(data.iloc[idx].get("note_id", ""))
            cands = rn_candidates_map.get(nid, [])
            rn_vals.append(cands[0]["dcml"] if cands else "")
        rn_full = pd.Series(rn_vals, dtype=object)
    elif "romanNumeral_full" in data.columns:
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

    # Compute expected labels for agreement coloring (fallback when no candidates)
    expected_labels: List[Dict[str, str]] = []
    if not rn_candidates_map and global_key:
        prepared = _prepare_df_for_flexohr(data)
        ohrs = _build_ohrs(data, global_key)
        expected_labels = flx.codecs.analysisgnn.derive_expected_labels(
            prepared, ohrs, _agnn_to_flx_pitch(global_key)
        )
        # Copy tpc_in_label -> pitch_spelling for Pitch tile coloring
        for rec in expected_labels:
            if "tpc_in_label" in rec:
                rec["pitch_spelling"] = rec["tpc_in_label"]
    if not expected_labels:
        expected_labels = [{}] * n

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
                "row": int(
                    _value_or_none(row.get("row")) if "row" in row.index else idx
                ),
                "note_id": str(score_note_id)
                if score_note_id is not None
                else (str(note_id) if note_id is not None else None),
                "table_note_id": str(note_id) if note_id is not None else None,
                "onset_div": int(note_array["onset_div"][idx])
                if "onset_div" in note_array.dtype.names
                else None,
                "onset_beat": float(_value_or_none(row.get("onset_beat")) or 0.0),
                "measure": int(_value_or_none(row.get("measure")))
                if _value_or_none(row.get("measure")) is not None
                else None,
                "duration_beat": float(_value_or_none(row.get("duration_beat")) or 0.0),
                "pitch_midi": int(_value_or_none(row.get("pitch_midi")))
                if _value_or_none(row.get("pitch_midi")) is not None
                else None,
                "pitch_spelling": str(_value_or_none(row.get("pitch_spelling")) or ""),
                "tasks": task_vals,
                "confidence": conf,
                "romanNumeral_full": str(rn_full.iloc[idx])
                if idx < len(rn_full)
                else "",
                "rn_expected": expected_labels[idx] if idx < len(expected_labels) else {},
                "rn_candidates": (
                    rn_candidates_map.get(str(note_id), [])
                    if rn_candidates_map and note_id
                    else []
                ),
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
            "global_key": global_key,
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
    rn_candidates_map: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    note_array = _sorted_note_array(score)
    payload = _build_graph_overlay_payload(
        df=df,
        note_array=note_array,
        tasks=tasks,
        edge_types=edge_types,
        edges_all=edges_all,
        global_key=global_key,
        rn_candidates_map=rn_candidates_map,
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
) -> Tuple[
    Dict[str, torch.Tensor],
    Dict[str, Any],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
]:
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
        [task for task in beat_tasks if task in payload_tasks]
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
    ordered_cols: List[str] = [col for col in core_cols if col in beat_df.columns]
    for task in tasks:
        for col in [
            task,
            f"{task}_confidence",
            f"{task}_conflict_flag",
            f"{task}_conflict_prob",
        ]:
            if col in beat_df.columns:
                ordered_cols.append(col)
    remaining = [col for col in beat_df.columns if col not in ordered_cols]
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


# Voter-related aggregation spec builder — commented out (voter not yet
# implemented in analysisgnn/aggregation/).
# def _build_aggregation_spec(aggregation_mode: str, voter_path: str) -> ...: ...


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
        predictor = _get_predictor(full_ckpt, masked_ckpt, device)

        iterative_spec = _build_iterative_spec(
            enable_iterative=enable_iterative,
            iterative_steps=iterative_steps,
            keep_percentile_per_step=keep_percentile_per_step,
            tasks=tasks,
            target_only_update=False,
            zero_known_start=True,
        )

        # Always run with no aggregation
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
        full_df = predictions_to_dataframe(
            score=score_obj,
            predictions=predictions,
            tasks=tasks,
            include_confidence=True,
            include_class_ids=False,
        )
        full_df = _apply_timing_from_predictions(full_df, predictions)
        display_df = format_table_output(full_df, tasks)
        _fix_key_mode(display_df)

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

        # Derive global key from predictions — must always succeed
        global_key = _derive_global_key(display_df)

        # Add Complete RN column
        display_df["romanNumeral_full"] = _build_complete_rn_column(
            display_df, global_key
        )
        display_df = _prepare_prediction_table_for_display(display_df)

        # Build visual payload
        visual_payload = _build_visual_payload(
            score_path=score_path,
            score=score_obj,
            df=display_df,
            tasks=tasks,
            edge_types=[],
            edges_all=edges_all,
            global_key=global_key,
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
        _fix_key_mode(display_df)

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

        # Derive global key from predictions — must always succeed
        global_key = _derive_global_key(display_df)

        # Add Complete RN column
        display_df["romanNumeral_full"] = _build_complete_rn_column(
            display_df, global_key
        )
        display_df = _prepare_prediction_table_for_display(display_df)

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
            _fix_key_mode(display_df)
            # Add Complete RN column
            if not global_key:
                raise ValueError(
                    "Global key is required for aggregation. "
                    "Set it in the Global Key field."
                )
            display_df["romanNumeral_full"] = _build_complete_rn_column(
                display_df, global_key
            )
            display_df = _prepare_prediction_table_for_display(display_df)
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
        predictor = _get_predictor(full_ckpt, masked_ckpt, device)

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
        _fix_key_mode(display_df)

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

        # Derive global key from predictions — must always succeed
        global_key = _derive_global_key(display_df)

        # Add Complete RN column
        display_df["romanNumeral_full"] = _build_complete_rn_column(
            display_df, global_key
        )
        display_df = _prepare_prediction_table_for_display(display_df)

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
    except Exception as exc:
        log_text = _log(log_text, f"Error: {exc}")
        return (
            pd.DataFrame(),
            log_text,
            {},
            None,
            {},
            intermediates_state or {},
            {},
            [],
            "",
            edges_state or {k: [[], []] for k in DEFAULT_EDGE_TYPES},
            False,
            "",
        )


# ---------------------------------------------------------------------------
# Verovio Visual Tab
# ---------------------------------------------------------------------------


def _compute_nct_note_colors(df: pd.DataFrame, n: int) -> Dict[int, str]:
    """Compute per-note colors based on tpc_in_label predictions.

    The colour encodes how confidently a note is classified as a chord tone
    (in label) vs. a non-chord tone (out of label):
    - Confidently in label ("True", high confidence) -> black (#000000)
    - Confidently out of label ("False", high confidence) -> light grey (#d3d3d3)
    - Low confidence in either direction -> middle grey

    The effective "in-label score" is P("True"):
      - If argmax is "True":  score = confidence
      - If argmax is "False": score = 1 - confidence
    Colour = linear interpolation from lightgrey (score=0) to black (score=1).
    """
    if "tpc_in_label" not in df.columns or "tpc_in_label_confidence" not in df.columns:
        return {}
    colors: Dict[int, str] = {}
    for idx in range(min(n, len(df))):
        row = df.iloc[idx]
        label = str(_value_or_none(row.get("tpc_in_label")) or "")
        conf_val = _value_or_none(row.get("tpc_in_label_confidence"))
        if conf_val is None:
            continue
        try:
            conf = float(conf_val)
        except (ValueError, TypeError):
            continue
        # P("True") = score: 1.0 means certainly in-label, 0.0 means certainly out-of-label
        if label == "True":
            score = conf
        else:
            score = 1.0 - conf
        # Interpolate: lightgrey (211,211,211) at score=0 -> black (0,0,0) at score=1
        grey = int(round(211 * (1.0 - score)))
        colors[idx] = f"rgb({grey},{grey},{grey})"
    return colors


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
    delta_dfs_state: Any = None,
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
            # Enumerate RN candidates if Delta Lake data is available
            rn_cands_map = None
            delta_dfs = delta_dfs_state if isinstance(delta_dfs_state, dict) else {}
            if global_key and delta_dfs.get("probs_df") is not None:
                try:
                    rn_cands_map = _enumerate_rn_candidates(
                        display_df=df,
                        probs_df=delta_dfs["probs_df"],
                        notes_df=delta_dfs["notes_df"],
                        hyperedges_df=delta_dfs["hyperedges_df"],
                        global_key=global_key,
                    )
                    log_text = _log(
                        log_text,
                        f"Enumerated RN candidates for {len(rn_cands_map)} notes.",
                    )
                except Exception as enum_exc:
                    log_text = _log(
                        log_text,
                        f"RN enumeration failed: {enum_exc}",
                    )

            payload = _build_visual_payload(
                score_path=score_path,
                score=score_obj,
                df=df,
                tasks=tasks,
                edge_types=selected_edge_types,
                edges_all=edges_all,
                global_key=global_key,
                rn_candidates_map=rn_cands_map,
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
# Global Key Change → Regenerate romanNumeral_full
# ---------------------------------------------------------------------------


def regenerate_rn_column(
    global_key_text: str,
    table_data: Any,
    tasks_state: Any,
    score_path_state: str,
    log_text: str,
):
    """Regenerate the romanNumeral_full column after a global key change.

    Called on global_key_field blur.  Updates the table and CSV download.
    """
    try:
        global_key = (global_key_text or "").strip()
        if not global_key:
            raise ValueError("Global key must not be empty.")
        df = pd.DataFrame(table_data) if table_data is not None else pd.DataFrame()
        if len(df) == 0:
            return df, log_text, None
        tasks = tasks_state or []
        df["romanNumeral_full"] = _build_complete_rn_column(df, global_key)
        df = _prepare_prediction_table_for_display(df)
        csv_path = _write_csv_to_temp(df, score_path_state)
        log_text = _log(log_text, f"Regenerated RN column with global key '{global_key}'.")
        return df, log_text, csv_path
    except Exception as exc:
        log_text = _log(log_text, f"RN regeneration error: {exc}")
        df = pd.DataFrame(table_data) if table_data is not None else pd.DataFrame()
        return df, log_text, None


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
        enable_iterative = gr.State(False)
        iterative_steps = gr.State(10)
        keep_percentile_per_step = gr.State(10.0)

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
                    score_file = gr.File(
                        label="MusicXML Score",
                        file_types=[".xml", ".musicxml", ".mxl"],
                        type="filepath",
                    )

                task_selector = gr.CheckboxGroup(
                    choices=list(AVAILABLE_TASKS.values()),
                    value=list(
                        AVAILABLE_TASKS.values()
                    ),  # All tasks selected by default
                    label="Select Analysis Tasks",
                    info="Choose which tasks to run and show in the editable table and visual tab.",
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

        # ---- Module 2: Global Key Change → Regenerate RN column ----
        global_key_field.blur(
            fn=regenerate_rn_column,
            inputs=[
                global_key_field,
                table,
                tasks_state,
                score_path_state,
                log_output,
            ],
            outputs=[table, log_output, csv_download],
        )

        # ---- Module 2: Refresh Visual ----
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
                delta_dfs_state,
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
