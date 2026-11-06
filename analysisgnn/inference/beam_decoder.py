"""Onset-level beam decoding for RNA tasks.

Two decoder versions are supported:

- ``legacy`` reproduces the original independent-head Cartesian-product beam.
- ``structured_v2`` decodes legal harmonic states over onset sequences and can
  optionally apply a lightweight learned rescoring model to the N-best list.
- ``component_v3`` decodes legal harmonic states directly from the five core
  harmony tasks ``(localkey, degree1, degree2, quality, inversion)`` and keeps
  the auxiliary RN head out of the decoding state.
"""

from __future__ import annotations

import itertools
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from analysisgnn.inference.harmonic_state import (
    COMPONENT_STATE_TASKS,
    HarmonicState,
    get_default_component_harmonic_state_library,
    get_default_harmonic_state_library,
)
from analysisgnn.utils.chord_representations import available_representations
from analysisgnn.utils.roman_decode import decode_roman_numeral


DEFAULT_BEAM_TASKS: Tuple[str, ...] = (
    "romanNumeral",
    "localkey",
    "quality",
    "inversion",
    "degree1",
    "degree2",
)

DEFAULT_COMPONENT_BEAM_TASKS: Tuple[str, ...] = COMPONENT_STATE_TASKS

DEFAULT_TOPK_BY_TASK: Dict[str, int] = {
    "romanNumeral": 8,
    "localkey": 6,
    "quality": 4,
    "inversion": 4,
    "degree1": 4,
    "degree2": 4,
}

DEFAULT_EMISSION_WEIGHTS: Dict[str, float] = {
    "romanNumeral": 1.0,
    "localkey": 0.9,
    "quality": 0.8,
    "inversion": 0.5,
    "degree1": 0.8,
    "degree2": 0.6,
    "root": 0.20,
    "bass": 0.20,
    "tpc_in_label": 0.08,
}

DEFAULT_TRANSITION_WEIGHTS: Dict[str, float] = {
    "localkey_switch": 0.35,
    "inversion_jump": 0.04,
    "degree1_change": 0.08,
    "degree2_change": 0.06,
    "stay_bonus": 0.18,
    "change_penalty": 0.08,
    "key_distance": 0.05,
    "function_switch": 0.03,
    "short_run_penalty": 0.10,
    "boundary_bonus": 0.12,
}

DEFAULT_COMPONENT_TOPK_BY_TASK: Dict[str, int] = {
    "localkey": 6,
    "degree1": 4,
    "degree2": 3,
    "quality": 4,
    "inversion": 4,
}

DEFAULT_COMPONENT_EMISSION_WEIGHTS: Dict[str, float] = {
    "localkey": 1.0,
    "degree1": 1.0,
    "degree2": 0.5,
    "quality": 0.9,
    "inversion": 0.6,
    "root": 0.15,
    "bass": 0.15,
    "tpc_in_label": 0.05,
}


def normalize_beam_spec(spec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = dict(spec or {})
    enabled = bool(cfg.get("enabled", False))
    beam_width = int(cfg.get("beam_width", cfg.get("width", 8)))
    if beam_width < 1:
        beam_width = 1

    level = str(cfg.get("level", "onset")).lower().strip()
    if level != "onset":
        level = "onset"

    version = str(cfg.get("version", "legacy")).lower().strip()
    if version not in {"legacy", "structured_v2", "component_v3"}:
        version = "legacy"

    default_tasks = list(DEFAULT_COMPONENT_BEAM_TASKS) if version == "component_v3" else list(DEFAULT_BEAM_TASKS)
    tasks_cfg = cfg.get("tasks", default_tasks)
    if isinstance(tasks_cfg, str):
        tasks_cfg = [t.strip() for t in tasks_cfg.split(",") if t.strip()]
    tasks = [str(t).strip() for t in tasks_cfg if str(t).strip()]
    if not tasks:
        tasks = list(DEFAULT_BEAM_TASKS)
    dedup: List[str] = []
    seen = set()
    for task in tasks:
        if task in seen:
            continue
        seen.add(task)
        dedup.append(task)
    tasks = dedup

    topk_cfg = cfg.get("topk_by_task", {})
    if not isinstance(topk_cfg, dict):
        topk_cfg = {}
    topk_by_task = dict(DEFAULT_TOPK_BY_TASK)
    if version == "component_v3":
        topk_by_task.update(DEFAULT_COMPONENT_TOPK_BY_TASK)
    for task, value in topk_cfg.items():
        try:
            topk_by_task[str(task)] = max(1, int(value))
        except Exception:
            continue

    emission_cfg = cfg.get("emission_weights", {})
    if not isinstance(emission_cfg, dict):
        emission_cfg = {}
    emission_weights = dict(DEFAULT_EMISSION_WEIGHTS)
    if version == "component_v3":
        emission_weights.update(DEFAULT_COMPONENT_EMISSION_WEIGHTS)
    for task, value in emission_cfg.items():
        try:
            emission_weights[str(task)] = float(value)
        except Exception:
            continue

    transition_cfg = cfg.get("transition_weights", {})
    if not isinstance(transition_cfg, dict):
        transition_cfg = {}
    transition_weights = dict(DEFAULT_TRANSITION_WEIGHTS)
    for key, value in transition_cfg.items():
        try:
            transition_weights[str(key)] = float(value)
        except Exception:
            continue

    consistency_weight = float(cfg.get("consistency_weight", 0.35))
    if consistency_weight < 0:
        consistency_weight = 0.0
    epsilon = float(cfg.get("epsilon", 1e-8))
    if epsilon <= 0:
        epsilon = 1e-8
    local_prune_factor = int(cfg.get("local_prune_factor", 6))
    if local_prune_factor < 1:
        local_prune_factor = 1
    smoothing_off_prob = float(cfg.get("smoothing_off_prob", 1e-3))
    smoothing_off_prob = min(max(smoothing_off_prob, 0.0), 0.25)
    nbest = int(cfg.get("nbest", beam_width))
    nbest = max(1, min(nbest, beam_width))
    rescorer_path = cfg.get("rescorer_path", None)
    rescorer_weight = float(cfg.get("rescorer_weight", 1.0))
    if rescorer_weight < 0:
        rescorer_weight = 0.0

    return {
        "enabled": enabled,
        "level": level,
        "version": version,
        "beam_width": beam_width,
        "tasks": tasks,
        "topk_by_task": topk_by_task,
        "emission_weights": emission_weights,
        "transition_weights": transition_weights,
        "consistency_weight": consistency_weight,
        "epsilon": epsilon,
        "local_prune_factor": local_prune_factor,
        "smoothing_off_prob": smoothing_off_prob,
        "nbest": nbest,
        "rescorer_path": rescorer_path,
        "rescorer_weight": rescorer_weight,
    }


def _class_labels_for_task(task: str, num_classes: int) -> List[Any]:
    rep = available_representations.get(task, None)
    class_list = getattr(rep, "classList", None) if rep is not None else None
    if isinstance(class_list, Sequence) and not isinstance(class_list, (str, bytes)):
        labels = list(class_list)
        if len(labels) >= num_classes:
            return labels[:num_classes]
        return labels + list(range(len(labels), num_classes))
    return list(range(num_classes))


def _label_from_index(task: str, idx: int, class_labels: Dict[str, List[Any]]) -> Any:
    labels = class_labels.get(task, [])
    if 0 <= idx < len(labels):
        return labels[idx]
    return idx


def _parse_degree_digit(value: Any) -> str:
    text = str(value).strip()
    if text in {"", "None"}:
        return "None"
    while text.startswith("#") or text.startswith("-"):
        text = text[1:]
    return text if text else "None"


def _project_complete_rn(complete_rn: str, rn_label: str, legal_set: Optional[set]) -> str:
    comp = (complete_rn or "").strip()
    rn = (rn_label or "").strip()
    if comp and (not legal_set or comp in legal_set):
        return comp
    if rn and (not legal_set or rn in legal_set):
        return rn
    return comp or rn


def _transition_penalty(
    prev_labels: Dict[str, Any],
    curr_labels: Dict[str, Any],
    transition_weights: Dict[str, float],
) -> float:
    penalty = 0.0
    if str(prev_labels.get("localkey", "")) != str(curr_labels.get("localkey", "")):
        penalty += float(transition_weights.get("localkey_switch", 0.0))
    try:
        prev_inv = int(float(prev_labels.get("inversion", 0)))
        curr_inv = int(float(curr_labels.get("inversion", 0)))
        penalty += abs(prev_inv - curr_inv) * float(transition_weights.get("inversion_jump", 0.0))
    except Exception:
        pass
    if _parse_degree_digit(prev_labels.get("degree1", "None")) != _parse_degree_digit(curr_labels.get("degree1", "None")):
        penalty += float(transition_weights.get("degree1_change", 0.0))
    if _parse_degree_digit(prev_labels.get("degree2", "None")) != _parse_degree_digit(curr_labels.get("degree2", "None")):
        penalty += float(transition_weights.get("degree2_change", 0.0))
    return penalty


def _onset_group_inverse(onset_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    onset_ids = onset_ids.to(dtype=torch.long)
    unique_onsets, inverse = torch.unique(onset_ids, sorted=True, return_inverse=True)
    return unique_onsets, inverse


def _onset_mean_probs(task_probs: torch.Tensor, inverse: torch.Tensor, num_onsets: int) -> torch.Tensor:
    out = torch.zeros(
        (num_onsets, task_probs.size(-1)),
        dtype=task_probs.dtype,
        device=task_probs.device,
    )
    counts = torch.zeros((num_onsets,), dtype=task_probs.dtype, device=task_probs.device)
    for onset_idx in range(num_onsets):
        idx = torch.where(inverse == onset_idx)[0]
        if idx.numel() == 0:
            continue
        out[onset_idx] = task_probs[idx].mean(dim=0)
        counts[onset_idx] = float(idx.numel())
    counts = torch.clamp(counts, min=1.0).unsqueeze(-1)
    return out / counts * counts


def _candidate_entropy(probs: torch.Tensor, eps: float) -> float:
    p = torch.clamp(probs, min=eps)
    return float((-(p * torch.log(p)).sum()).item())


def _safe_log(prob: float, eps: float) -> float:
    return float(math.log(max(float(prob), eps)))


def _key_distance(prev_state: HarmonicState, curr_state: HarmonicState) -> float:
    if prev_state.tonic_pc is None or curr_state.tonic_pc is None:
        return 0.0
    diff = abs(int(prev_state.tonic_pc) - int(curr_state.tonic_pc)) % 12
    return float(min(diff, 12 - diff))


def _boundary_signal(onset_probs: Dict[str, torch.Tensor], onset_idx: int) -> float:
    boundary = 0.0
    if "cadence" in onset_probs:
        probs = onset_probs["cadence"][onset_idx]
        if probs.numel() > 1:
            boundary = max(boundary, float(probs[1:].max().item()))
    if "phrase" in onset_probs:
        probs = onset_probs["phrase"][onset_idx]
        if probs.numel() > 1:
            boundary = max(boundary, float(probs[1:].max().item()))
    return boundary


def _structured_transition_penalty(
    prev_state: HarmonicState,
    curr_state: HarmonicState,
    *,
    run_length: int,
    transition_weights: Dict[str, float],
    boundary_signal: float,
) -> Tuple[float, float]:
    if prev_state.state_id == curr_state.state_id:
        return -float(transition_weights.get("stay_bonus", 0.0)), 0.0
    penalty = float(transition_weights.get("change_penalty", 0.0))
    penalty += _key_distance(prev_state, curr_state) * float(transition_weights.get("key_distance", 0.0))
    penalty += abs(int(prev_state.inversion) - int(curr_state.inversion)) * float(
        transition_weights.get("inversion_jump", 0.0)
    )
    if prev_state.functional_class != curr_state.functional_class:
        penalty += float(transition_weights.get("function_switch", 0.0))
    if run_length <= 1:
        penalty += float(transition_weights.get("short_run_penalty", 0.0))
    penalty -= boundary_signal * float(transition_weights.get("boundary_bonus", 0.0))
    return penalty, _key_distance(prev_state, curr_state)


class StructuredBeamRescorer(nn.Module):
    """Small MLP over whole-sequence beam features."""

    feature_names: Tuple[str, ...] = (
        "avg_emission",
        "avg_component_support",
        "avg_root_bass_support",
        "avg_chord_tone_support",
        "avg_boundary_signal",
        "num_changes",
        "mean_run_length",
        "total_key_distance",
        "entropy_mean",
    )

    def __init__(self, hidden_dim: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        input_dim = len(self.feature_names)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def input_dim(self) -> int:
        return len(self.feature_names)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def load_structured_beam_rescorer(
    path: Optional[str],
    *,
    device: torch.device,
) -> Optional[StructuredBeamRescorer]:
    if not path:
        return None
    if not os.path.exists(path):
        return None
    payload = torch.load(path, map_location="cpu")
    hidden_dim = 16
    dropout = 0.1
    state_dict = payload
    if isinstance(payload, dict):
        hidden_dim = int(payload.get("hidden_dim", hidden_dim))
        dropout = float(payload.get("dropout", dropout))
        state_dict = payload.get("state_dict", payload)
    model = StructuredBeamRescorer(hidden_dim=hidden_dim, dropout=dropout)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model.to(device)


def extract_structured_candidate_features(candidate: Dict[str, Any]) -> torch.Tensor:
    total_steps = max(1, int(len(candidate.get("path", []))))
    mean_run_length = float(total_steps / max(1, int(candidate.get("num_changes", 0)) + 1))
    values = [
        float(candidate.get("emission_total", 0.0)) / float(total_steps),
        float(candidate.get("component_support_total", 0.0)) / float(total_steps),
        float(candidate.get("root_bass_support_total", 0.0)) / float(total_steps),
        float(candidate.get("chord_tone_support_total", 0.0)) / float(total_steps),
        float(candidate.get("boundary_signal_total", 0.0)) / float(total_steps),
        float(candidate.get("num_changes", 0.0)),
        mean_run_length,
        float(candidate.get("total_key_distance", 0.0)),
        float(candidate.get("entropy_total", 0.0)) / float(total_steps),
    ]
    return torch.tensor(values, dtype=torch.float32)


def _decode_onset_beam_legacy(
    *,
    note_prob_dict: Dict[str, torch.Tensor],
    onset_ids: torch.Tensor,
    task_num_classes: Dict[str, int],
    legal_rn_set: Optional[set] = None,
    spec: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    cfg = normalize_beam_spec(spec)
    if not cfg["enabled"]:
        return None
    if onset_ids is None or onset_ids.numel() == 0:
        return None

    tasks = [t for t in cfg["tasks"] if t in note_prob_dict and t in task_num_classes]
    required = ["romanNumeral", "localkey", "quality", "inversion", "degree1", "degree2"]
    if not all(t in tasks for t in required):
        return None

    num_notes = int(onset_ids.numel())
    unique_onsets, inverse = _onset_group_inverse(onset_ids)
    num_onsets = int(unique_onsets.numel())
    if num_onsets == 0:
        return None

    onset_probs: Dict[str, torch.Tensor] = {}
    class_labels: Dict[str, List[Any]] = {}
    for task in tasks:
        task_probs = note_prob_dict[task][:num_notes]
        if task_probs.ndim != 2:
            return None
        onset_probs[task] = _onset_mean_probs(task_probs, inverse, num_onsets)
        class_labels[task] = _class_labels_for_task(task, int(task_num_classes[task]))

    epsilon = float(cfg["epsilon"])
    beam_width = int(cfg["beam_width"])
    local_keep = beam_width * int(cfg["local_prune_factor"])
    consistency_weight = float(cfg["consistency_weight"])

    task_topk: Dict[str, int] = {}
    for task in tasks:
        num_classes = int(task_num_classes[task])
        task_topk[task] = max(1, min(int(cfg["topk_by_task"].get(task, 1)), num_classes))

    beams: List[Dict[str, Any]] = []
    trace_steps: List[Dict[str, Any]] = []

    for onset_idx in range(num_onsets):
        per_task_candidates: Dict[str, List[Tuple[int, float]]] = {}
        for task in tasks:
            probs = onset_probs[task][onset_idx]
            topk = task_topk[task]
            values, indices = torch.topk(probs, k=topk, dim=-1)
            cands = []
            for cls_idx, prob in zip(indices.tolist(), values.tolist()):
                cands.append((int(cls_idx), float(math.log(max(prob, epsilon)))))
            per_task_candidates[task] = cands

        unconstrained_best: Optional[Dict[str, Any]] = None
        local_states: List[Dict[str, Any]] = []
        product_iter = itertools.product(
            per_task_candidates["romanNumeral"],
            per_task_candidates["localkey"],
            per_task_candidates["quality"],
            per_task_candidates["inversion"],
            per_task_candidates["degree1"],
            per_task_candidates["degree2"],
        )
        for rn_c, lk_c, q_c, inv_c, d1_c, d2_c in product_iter:
            class_ids = {
                "romanNumeral": rn_c[0],
                "localkey": lk_c[0],
                "quality": q_c[0],
                "inversion": inv_c[0],
                "degree1": d1_c[0],
                "degree2": d2_c[0],
            }
            labels = {
                task: _label_from_index(task, class_ids[task], class_labels)
                for task in class_ids.keys()
            }
            try:
                comp_rn = decode_roman_numeral(
                    degree1=str(labels["degree1"]),
                    degree2=str(labels["degree2"]),
                    inversion=int(float(labels["inversion"])),
                    quality=str(labels["quality"]),
                    localkey=str(labels["localkey"]),
                )
            except Exception:
                comp_rn = ""
            rn_label = str(labels["romanNumeral"])
            projected_rn = _project_complete_rn(comp_rn, rn_label, legal_rn_set)

            emission = (
                float(cfg["emission_weights"].get("romanNumeral", 1.0)) * rn_c[1]
                + float(cfg["emission_weights"].get("localkey", 1.0)) * lk_c[1]
                + float(cfg["emission_weights"].get("quality", 1.0)) * q_c[1]
                + float(cfg["emission_weights"].get("inversion", 1.0)) * inv_c[1]
                + float(cfg["emission_weights"].get("degree1", 1.0)) * d1_c[1]
                + float(cfg["emission_weights"].get("degree2", 1.0)) * d2_c[1]
            )
            consistency_penalty = 0.0
            if comp_rn and rn_label and comp_rn != rn_label:
                consistency_penalty = consistency_weight
            local_score = emission - consistency_penalty

            candidate = {
                "class_ids": class_ids,
                "labels": labels,
                "projected_rn": projected_rn,
                "comp_rn": comp_rn,
                "rn_label": rn_label,
                "local_score": local_score,
            }
            if unconstrained_best is None or local_score > unconstrained_best["local_score"]:
                unconstrained_best = candidate

            if legal_rn_set and projected_rn and projected_rn not in legal_rn_set:
                continue
            local_states.append(candidate)

        if not local_states and unconstrained_best is not None:
            local_states = [unconstrained_best]
        if not local_states:
            return None

        local_states.sort(key=lambda x: x["local_score"], reverse=True)
        local_states = local_states[:local_keep]

        if not beams:
            beams = [
                {
                    "score": state["local_score"],
                    "path": [state],
                }
                for state in local_states[:beam_width]
            ]
        else:
            new_beams: List[Dict[str, Any]] = []
            for prev in beams:
                prev_last = prev["path"][-1]
                for state in local_states:
                    penalty = _transition_penalty(
                        prev_labels=prev_last["labels"],
                        curr_labels=state["labels"],
                        transition_weights=cfg["transition_weights"],
                    )
                    new_beams.append(
                        {
                            "score": float(prev["score"] + state["local_score"] - penalty),
                            "path": prev["path"] + [state],
                        }
                    )
            new_beams.sort(key=lambda x: x["score"], reverse=True)
            beams = new_beams[:beam_width]

        trace_steps.append(
            {
                "onset_index": int(onset_idx),
                "num_candidates": int(len(local_states)),
                "best_score": float(beams[0]["score"]) if beams else float("nan"),
            }
        )

    if not beams:
        return None
    best = beams[0]
    best_path = best["path"]

    onset_class_ids: Dict[str, torch.Tensor] = {}
    for task in required:
        onset_class_ids[task] = torch.tensor(
            [int(step["class_ids"][task]) for step in best_path],
            dtype=torch.long,
            device=onset_ids.device,
        )
    note_class_ids = {task: onset_class_ids[task][inverse] for task in onset_class_ids.keys()}

    note_confidence: Dict[str, torch.Tensor] = {}
    for task in onset_class_ids.keys():
        probs = note_prob_dict[task][:num_notes]
        cls = note_class_ids[task]
        note_confidence[task] = probs[torch.arange(num_notes, device=probs.device), cls]

    return {
        "tasks": list(required),
        "note_class_ids": note_class_ids,
        "onset_class_ids": onset_class_ids,
        "onset_values": unique_onsets,
        "note_inverse": inverse,
        "note_confidence": note_confidence,
        "beam_trace": {
            "enabled": True,
            "version": "legacy",
            "beam_width": beam_width,
            "num_onsets": num_onsets,
            "steps": trace_steps,
            "best_score": float(best["score"]),
        },
    }


def _build_onset_prob_dict(
    note_prob_dict: Dict[str, torch.Tensor],
    inverse: torch.Tensor,
    num_onsets: int,
    num_notes: int,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for task, task_probs in note_prob_dict.items():
        if not isinstance(task_probs, torch.Tensor) or task_probs.ndim != 2:
            continue
        out[task] = _onset_mean_probs(task_probs[:num_notes], inverse, num_onsets)
    return out


def _state_component_support(
    state: HarmonicState,
    onset_probs: Dict[str, torch.Tensor],
    onset_idx: int,
    epsilon: float,
) -> Tuple[float, float, float, float]:
    component_terms: List[float] = []
    root_bass_terms: List[float] = []

    for task in ("quality", "inversion", "degree1", "degree2"):
        cls_idx = state.class_ids.get(task, None)
        if cls_idx is None or task not in onset_probs:
            continue
        prob = float(onset_probs[task][onset_idx, cls_idx].item())
        component_terms.append(_safe_log(prob, epsilon))
    for task in ("root", "bass"):
        cls_idx = state.class_ids.get(task, None)
        if cls_idx is None or task not in onset_probs:
            continue
        prob = float(onset_probs[task][onset_idx, cls_idx].item())
        root_bass_terms.append(_safe_log(prob, epsilon))

    p_chord_tone = 0.0
    if "tpc_in_label" in onset_probs and onset_probs["tpc_in_label"].size(-1) > 1:
        p_chord_tone = float(onset_probs["tpc_in_label"][onset_idx, 1].item())

    component_avg = sum(component_terms) / float(len(component_terms)) if component_terms else 0.0
    root_bass_avg = sum(root_bass_terms) / float(len(root_bass_terms)) if root_bass_terms else 0.0
    return component_avg, root_bass_avg, p_chord_tone, float(len(component_terms))


def _component_state_local_score(
    *,
    state: HarmonicState,
    onset_probs: Dict[str, torch.Tensor],
    onset_idx: int,
    cfg: Dict[str, Any],
    epsilon: float,
) -> Tuple[float, float, float, float]:
    emission = 0.0
    confidence_terms: List[float] = []
    component_support_terms: List[float] = []
    for task in COMPONENT_STATE_TASKS:
        cls_idx = state.class_ids.get(task, None)
        if cls_idx is None or task not in onset_probs:
            continue
        prob = float(onset_probs[task][onset_idx, cls_idx].item())
        confidence_terms.append(prob)
        log_prob = _safe_log(prob, epsilon)
        emission += float(cfg["emission_weights"].get(task, 1.0)) * log_prob
        if task != "localkey":
            component_support_terms.append(log_prob)

    root_bass_avg = 0.0
    root_bass_terms: List[float] = []
    for task in ("root", "bass"):
        cls_idx = state.class_ids.get(task, None)
        if cls_idx is None or task not in onset_probs:
            continue
        prob = float(onset_probs[task][onset_idx, cls_idx].item())
        support = _safe_log(prob, epsilon)
        root_bass_terms.append(support)
        emission += float(cfg["emission_weights"].get(task, 0.0)) * support
    if root_bass_terms:
        root_bass_avg = sum(root_bass_terms) / float(len(root_bass_terms))

    p_chord_tone = 0.0
    if "tpc_in_label" in onset_probs and onset_probs["tpc_in_label"].size(-1) > 1:
        p_chord_tone = float(onset_probs["tpc_in_label"][onset_idx, 1].item())
        emission += float(cfg["emission_weights"].get("tpc_in_label", 0.0)) * p_chord_tone

    confidence = sum(confidence_terms) / float(len(confidence_terms)) if confidence_terms else 0.0
    component_avg = (
        sum(component_support_terms) / float(len(component_support_terms))
        if component_support_terms
        else 0.0
    )
    return float(emission), float(confidence), float(component_avg), float(root_bass_avg)


def _decode_onset_beam_structured_v2(
    *,
    note_prob_dict: Dict[str, torch.Tensor],
    onset_ids: torch.Tensor,
    task_num_classes: Dict[str, int],
    legal_rn_set: Optional[set] = None,
    spec: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    cfg = normalize_beam_spec(spec)
    if not cfg["enabled"]:
        return None
    if onset_ids is None or onset_ids.numel() == 0:
        return None
    if "romanNumeral" not in note_prob_dict or "localkey" not in note_prob_dict:
        return None

    num_notes = int(onset_ids.numel())
    unique_onsets, inverse = _onset_group_inverse(onset_ids)
    num_onsets = int(unique_onsets.numel())
    if num_onsets == 0:
        return None

    library = get_default_harmonic_state_library()
    onset_probs = _build_onset_prob_dict(note_prob_dict, inverse, num_onsets, num_notes)
    epsilon = float(cfg["epsilon"])
    beam_width = int(cfg["beam_width"])
    local_keep = beam_width * int(cfg["local_prune_factor"])
    nbest = int(cfg["nbest"])

    rn_topk = max(1, min(int(cfg["topk_by_task"].get("romanNumeral", 8)), onset_probs["romanNumeral"].size(-1)))
    key_topk = max(1, min(int(cfg["topk_by_task"].get("localkey", 6)), onset_probs["localkey"].size(-1)))
    beams: List[Dict[str, Any]] = []
    trace_steps: List[Dict[str, Any]] = []

    for onset_idx in range(num_onsets):
        rn_values, rn_indices = torch.topk(onset_probs["romanNumeral"][onset_idx], k=rn_topk, dim=-1)
        lk_values, lk_indices = torch.topk(onset_probs["localkey"][onset_idx], k=key_topk, dim=-1)
        candidate_states = library.candidates_from_topk(rn_indices.tolist(), lk_indices.tolist())
        if not candidate_states:
            return None

        local_states: List[Dict[str, Any]] = []
        boundary_signal = _boundary_signal(onset_probs, onset_idx)
        entropy_mean = (
            _candidate_entropy(onset_probs["romanNumeral"][onset_idx], epsilon)
            + _candidate_entropy(onset_probs["localkey"][onset_idx], epsilon)
        ) / 2.0

        for state in candidate_states:
            rn_prob = float(onset_probs["romanNumeral"][onset_idx, state.class_ids["romanNumeral"]].item())
            lk_prob = float(onset_probs["localkey"][onset_idx, state.class_ids["localkey"]].item())
            component_avg, root_bass_avg, p_chord_tone, used_components = _state_component_support(
                state,
                onset_probs,
                onset_idx,
                epsilon,
            )
            emission = (
                float(cfg["emission_weights"].get("romanNumeral", 1.0)) * _safe_log(rn_prob, epsilon)
                + float(cfg["emission_weights"].get("localkey", 0.9)) * _safe_log(lk_prob, epsilon)
            )
            if used_components > 0:
                emission += float(cfg["emission_weights"].get("quality", 0.8)) * component_avg
            if root_bass_avg != 0.0:
                emission += float(cfg["emission_weights"].get("root", 0.2)) * root_bass_avg
            if p_chord_tone > 0.0:
                emission += float(cfg["emission_weights"].get("tpc_in_label", 0.08)) * p_chord_tone
            if legal_rn_set and state.complete_rn and state.complete_rn not in legal_rn_set and state.roman_numeral not in legal_rn_set:
                continue
            confidence = float(
                (rn_prob + lk_prob + (math.exp(component_avg) if used_components > 0 else 0.0)) / float(2 + (1 if used_components > 0 else 0))
            )
            local_states.append(
                {
                    "state": state,
                    "local_score": float(emission),
                    "confidence": confidence,
                    "component_support": component_avg,
                    "root_bass_support": root_bass_avg,
                    "chord_tone_support": p_chord_tone,
                    "boundary_signal": boundary_signal,
                    "entropy": entropy_mean,
                }
            )
        if not local_states:
            return None
        local_states.sort(key=lambda x: x["local_score"], reverse=True)
        local_states = local_states[:local_keep]

        if not beams:
            beams = []
            for cand in local_states[:beam_width]:
                beams.append(
                    {
                        "score": cand["local_score"],
                        "path": [cand],
                        "run_length": 1,
                        "num_changes": 0,
                        "total_key_distance": 0.0,
                        "emission_total": cand["local_score"],
                        "component_support_total": cand["component_support"],
                        "root_bass_support_total": cand["root_bass_support"],
                        "chord_tone_support_total": cand["chord_tone_support"],
                        "boundary_signal_total": cand["boundary_signal"],
                        "entropy_total": cand["entropy"],
                    }
                )
        else:
            new_beams: List[Dict[str, Any]] = []
            for prev in beams:
                prev_state = prev["path"][-1]["state"]
                for cand in local_states:
                    penalty, key_dist = _structured_transition_penalty(
                        prev_state,
                        cand["state"],
                        run_length=int(prev["run_length"]),
                        transition_weights=cfg["transition_weights"],
                        boundary_signal=float(cand["boundary_signal"]),
                    )
                    same_state = prev_state.state_id == cand["state"].state_id
                    run_length = int(prev["run_length"]) + 1 if same_state else 1
                    new_beams.append(
                        {
                            "score": float(prev["score"] + cand["local_score"] - penalty),
                            "path": prev["path"] + [cand],
                            "run_length": run_length,
                            "num_changes": int(prev["num_changes"]) + (0 if same_state else 1),
                            "total_key_distance": float(prev["total_key_distance"] + key_dist),
                            "emission_total": float(prev["emission_total"] + cand["local_score"]),
                            "component_support_total": float(prev["component_support_total"] + cand["component_support"]),
                            "root_bass_support_total": float(prev["root_bass_support_total"] + cand["root_bass_support"]),
                            "chord_tone_support_total": float(prev["chord_tone_support_total"] + cand["chord_tone_support"]),
                            "boundary_signal_total": float(prev["boundary_signal_total"] + cand["boundary_signal"]),
                            "entropy_total": float(prev["entropy_total"] + cand["entropy"]),
                        }
                    )
            new_beams.sort(key=lambda x: x["score"], reverse=True)
            beams = new_beams[:beam_width]

        trace_steps.append(
            {
                "onset_index": int(onset_idx),
                "num_candidates": int(len(local_states)),
                "best_score": float(beams[0]["score"]) if beams else float("nan"),
                "boundary_signal": float(boundary_signal),
            }
        )

    if not beams:
        return None

    rescorer = load_structured_beam_rescorer(cfg.get("rescorer_path"), device=onset_ids.device)
    if rescorer is not None:
        features = torch.stack([extract_structured_candidate_features(b) for b in beams[:nbest]]).to(onset_ids.device)
        with torch.no_grad():
            rescoring = rescorer(features)
        for beam_item, resc in zip(beams[:nbest], rescoring.tolist()):
            beam_item["rescorer_score"] = float(resc)
            beam_item["score"] = float(beam_item["score"] + cfg["rescorer_weight"] * float(resc))
        beams.sort(key=lambda x: x["score"], reverse=True)

    best = beams[0]
    best_path = best["path"]
    output_tasks = list(DEFAULT_BEAM_TASKS)
    onset_class_ids: Dict[str, torch.Tensor] = {}
    onset_confidence = torch.tensor(
        [float(step.get("confidence", 0.0)) for step in best_path],
        dtype=torch.float32,
        device=onset_ids.device,
    )
    for task in output_tasks:
        class_ids: List[int] = []
        for onset_idx, step in enumerate(best_path):
            state = step["state"]
            cls_idx = state.class_ids.get(task, None)
            if cls_idx is None and task in onset_probs:
                cls_idx = int(torch.argmax(onset_probs[task][onset_idx]).item())
            if cls_idx is None:
                cls_idx = 0
            class_ids.append(int(cls_idx))
        onset_class_ids[task] = torch.tensor(class_ids, dtype=torch.long, device=onset_ids.device)

    note_class_ids = {task: onset_class_ids[task][inverse] for task in onset_class_ids.keys()}
    note_confidence: Dict[str, torch.Tensor] = {}
    for task, onset_cls in onset_class_ids.items():
        if task not in note_prob_dict:
            continue
        probs = note_prob_dict[task][:num_notes]
        cls = note_class_ids[task]
        note_confidence[task] = probs[torch.arange(num_notes, device=probs.device), cls]

    nbest_candidates = []
    for beam_item in beams[:nbest]:
        path = beam_item.get("path", [])
        nbest_candidates.append(
            {
                "score": float(beam_item.get("score", 0.0)),
                "rescorer_score": float(beam_item.get("rescorer_score", 0.0)),
                "num_changes": int(beam_item.get("num_changes", 0)),
                "states": [
                    {
                        "localkey": step["state"].localkey,
                        "romanNumeral": step["state"].roman_numeral,
                        "complete_rn": step["state"].complete_rn,
                    }
                    for step in path
                ],
                "class_ids": {
                    task: [
                        int(step["state"].class_ids.get(task, 0))
                        for step in path
                    ]
                    for task in DEFAULT_BEAM_TASKS
                },
                "features": extract_structured_candidate_features(beam_item).tolist(),
            }
        )

    return {
        "tasks": output_tasks,
        "note_class_ids": note_class_ids,
        "onset_class_ids": onset_class_ids,
        "onset_values": unique_onsets,
        "note_inverse": inverse,
        "note_confidence": note_confidence,
        "onset_confidence": onset_confidence,
        "beam_trace": {
            "enabled": True,
            "version": "structured_v2",
            "beam_width": beam_width,
            "num_onsets": num_onsets,
            "steps": trace_steps,
            "best_score": float(best["score"]),
            "nbest": nbest_candidates,
        },
    }


def _decode_onset_beam_component_v3(
    *,
    note_prob_dict: Dict[str, torch.Tensor],
    onset_ids: torch.Tensor,
    task_num_classes: Dict[str, int],
    legal_rn_set: Optional[set] = None,
    spec: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    del legal_rn_set
    cfg = normalize_beam_spec(spec)
    if not cfg["enabled"]:
        return None
    if onset_ids is None or onset_ids.numel() == 0:
        return None
    required_tasks = [task for task in COMPONENT_STATE_TASKS if task in note_prob_dict]
    if len(required_tasks) != len(COMPONENT_STATE_TASKS):
        return None

    num_notes = int(onset_ids.numel())
    unique_onsets, inverse = _onset_group_inverse(onset_ids)
    num_onsets = int(unique_onsets.numel())
    if num_onsets == 0:
        return None

    library = get_default_component_harmonic_state_library()
    onset_probs = _build_onset_prob_dict(note_prob_dict, inverse, num_onsets, num_notes)
    epsilon = float(cfg["epsilon"])
    beam_width = int(cfg["beam_width"])
    local_keep = max(beam_width * int(cfg["local_prune_factor"]), beam_width)
    nbest = int(cfg["nbest"])

    beams: List[Dict[str, Any]] = []
    trace_steps: List[Dict[str, Any]] = []
    onset_margin_values: List[float] = []

    for onset_idx in range(num_onsets):
        class_candidates: Dict[str, List[int]] = {}
        entropy_terms: List[float] = []
        for task in COMPONENT_STATE_TASKS:
            probs = onset_probs[task][onset_idx]
            topk = max(1, min(int(cfg["topk_by_task"].get(task, 1)), probs.size(-1)))
            values, indices = torch.topk(probs, k=topk, dim=-1)
            class_candidates[task] = [int(idx) for idx in indices.tolist()]
            entropy_terms.append(_candidate_entropy(probs, epsilon))

        candidate_states = library.candidates_from_topk(
            class_candidates,
            fallback_match_min=3,
            max_states=max(local_keep * 4, 64),
        )
        if not candidate_states:
            return None

        local_states: List[Dict[str, Any]] = []
        boundary_signal = _boundary_signal(onset_probs, onset_idx)
        entropy_mean = (
            sum(entropy_terms) / float(len(entropy_terms)) if entropy_terms else 0.0
        )
        for state in candidate_states:
            emission, confidence, component_avg, root_bass_avg = _component_state_local_score(
                state=state,
                onset_probs=onset_probs,
                onset_idx=onset_idx,
                cfg=cfg,
                epsilon=epsilon,
            )
            local_states.append(
                {
                    "state": state,
                    "local_score": float(emission),
                    "confidence": float(confidence),
                    "component_support": float(component_avg),
                    "root_bass_support": float(root_bass_avg),
                    "chord_tone_support": float(
                        onset_probs["tpc_in_label"][onset_idx, 1].item()
                    ) if "tpc_in_label" in onset_probs and onset_probs["tpc_in_label"].size(-1) > 1 else 0.0,
                    "boundary_signal": float(boundary_signal),
                    "entropy": float(entropy_mean),
                }
            )

        local_states.sort(key=lambda x: x["local_score"], reverse=True)
        local_states = local_states[:local_keep]
        best_local = float(local_states[0]["local_score"])
        second_local = float(local_states[1]["local_score"]) if len(local_states) > 1 else float(local_states[0]["local_score"])
        onset_margin_values.append(best_local - second_local)

        if not beams:
            beams = []
            for cand in local_states[:beam_width]:
                beams.append(
                    {
                        "score": cand["local_score"],
                        "path": [cand],
                        "run_length": 1,
                        "num_changes": 0,
                        "total_key_distance": 0.0,
                        "emission_total": cand["local_score"],
                        "component_support_total": cand["component_support"],
                        "root_bass_support_total": cand["root_bass_support"],
                        "chord_tone_support_total": cand["chord_tone_support"],
                        "boundary_signal_total": cand["boundary_signal"],
                        "entropy_total": cand["entropy"],
                    }
                )
        else:
            new_beams: List[Dict[str, Any]] = []
            for prev in beams:
                prev_state = prev["path"][-1]["state"]
                for cand in local_states:
                    penalty, key_dist = _structured_transition_penalty(
                        prev_state,
                        cand["state"],
                        run_length=int(prev["run_length"]),
                        transition_weights=cfg["transition_weights"],
                        boundary_signal=float(cand["boundary_signal"]),
                    )
                    same_state = prev_state.state_id == cand["state"].state_id
                    run_length = int(prev["run_length"]) + 1 if same_state else 1
                    new_beams.append(
                        {
                            "score": float(prev["score"] + cand["local_score"] - penalty),
                            "path": prev["path"] + [cand],
                            "run_length": run_length,
                            "num_changes": int(prev["num_changes"]) + (0 if same_state else 1),
                            "total_key_distance": float(prev["total_key_distance"] + key_dist),
                            "emission_total": float(prev["emission_total"] + cand["local_score"]),
                            "component_support_total": float(prev["component_support_total"] + cand["component_support"]),
                            "root_bass_support_total": float(prev["root_bass_support_total"] + cand["root_bass_support"]),
                            "chord_tone_support_total": float(prev["chord_tone_support_total"] + cand["chord_tone_support"]),
                            "boundary_signal_total": float(prev["boundary_signal_total"] + cand["boundary_signal"]),
                            "entropy_total": float(prev["entropy_total"] + cand["entropy"]),
                        }
                    )
            new_beams.sort(key=lambda x: x["score"], reverse=True)
            beams = new_beams[:beam_width]

        trace_steps.append(
            {
                "onset_index": int(onset_idx),
                "num_candidates": int(len(local_states)),
                "best_score": float(beams[0]["score"]) if beams else float("nan"),
                "boundary_signal": float(boundary_signal),
                "margin": float(onset_margin_values[-1]),
            }
        )

    if not beams:
        return None

    rescorer = load_structured_beam_rescorer(cfg.get("rescorer_path"), device=onset_ids.device)
    if rescorer is not None:
        features = torch.stack([extract_structured_candidate_features(b) for b in beams[:nbest]]).to(onset_ids.device)
        with torch.no_grad():
            rescoring = rescorer(features)
        for beam_item, resc in zip(beams[:nbest], rescoring.tolist()):
            beam_item["rescorer_score"] = float(resc)
            beam_item["score"] = float(beam_item["score"] + cfg["rescorer_weight"] * float(resc))
        beams.sort(key=lambda x: x["score"], reverse=True)

    best = beams[0]
    best_path = best["path"]
    output_tasks: List[str] = list(COMPONENT_STATE_TASKS)
    for opt_task in ("root", "bass"):
        if opt_task in note_prob_dict:
            output_tasks.append(opt_task)

    onset_class_ids: Dict[str, torch.Tensor] = {}
    onset_confidence = torch.tensor(
        [float(step.get("confidence", 0.0)) for step in best_path],
        dtype=torch.float32,
        device=onset_ids.device,
    )
    onset_margin = torch.tensor(
        onset_margin_values,
        dtype=torch.float32,
        device=onset_ids.device,
    )
    onset_complete_rn: List[str] = []
    for task in output_tasks:
        class_ids: List[int] = []
        for step in best_path:
            state = step["state"]
            cls_idx = state.class_ids.get(task, None)
            if cls_idx is None and task in onset_probs:
                cls_idx = int(torch.argmax(onset_probs[task][len(class_ids)]).item())
            if cls_idx is None:
                cls_idx = 0
            class_ids.append(int(cls_idx))
        onset_class_ids[task] = torch.tensor(class_ids, dtype=torch.long, device=onset_ids.device)
    for step in best_path:
        onset_complete_rn.append(str(step["state"].complete_rn))

    note_class_ids = {
        task: onset_class_ids[task][inverse]
        for task in onset_class_ids.keys()
    }
    note_confidence: Dict[str, torch.Tensor] = {}
    for task, onset_cls in onset_class_ids.items():
        if task not in note_prob_dict:
            continue
        probs = note_prob_dict[task][:num_notes]
        cls = note_class_ids[task]
        note_confidence[task] = probs[torch.arange(num_notes, device=probs.device), cls]

    nbest_candidates = []
    for beam_item in beams[:nbest]:
        path = beam_item.get("path", [])
        nbest_candidates.append(
            {
                "score": float(beam_item.get("score", 0.0)),
                "rescorer_score": float(beam_item.get("rescorer_score", 0.0)),
                "num_changes": int(beam_item.get("num_changes", 0)),
                "states": [
                    {
                        "localkey": step["state"].localkey,
                        "degree1": step["state"].degree1,
                        "degree2": step["state"].degree2,
                        "quality": step["state"].quality,
                        "inversion": step["state"].inversion,
                        "complete_rn": step["state"].complete_rn,
                    }
                    for step in path
                ],
                "class_ids": {
                    task: [
                        int(step["state"].class_ids.get(task, 0))
                        for step in path
                    ]
                    for task in output_tasks
                },
                "features": extract_structured_candidate_features(beam_item).tolist(),
            }
        )

    return {
        "tasks": output_tasks,
        "note_class_ids": note_class_ids,
        "onset_class_ids": onset_class_ids,
        "onset_values": unique_onsets,
        "note_inverse": inverse,
        "note_confidence": note_confidence,
        "onset_confidence": onset_confidence,
        "onset_margin": onset_margin,
        "onset_complete_rn": onset_complete_rn,
        "beam_trace": {
            "enabled": True,
            "version": "component_v3",
            "beam_width": beam_width,
            "num_onsets": num_onsets,
            "steps": trace_steps,
            "best_score": float(best["score"]),
            "nbest": nbest_candidates,
        },
    }


def decode_onset_beam(
    *,
    note_prob_dict: Dict[str, torch.Tensor],
    onset_ids: torch.Tensor,
    task_num_classes: Dict[str, int],
    legal_rn_set: Optional[set] = None,
    spec: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    cfg = normalize_beam_spec(spec)
    if cfg.get("version") == "component_v3":
        return _decode_onset_beam_component_v3(
            note_prob_dict=note_prob_dict,
            onset_ids=onset_ids,
            task_num_classes=task_num_classes,
            legal_rn_set=legal_rn_set,
            spec=cfg,
        )
    if cfg.get("version") == "structured_v2":
        return _decode_onset_beam_structured_v2(
            note_prob_dict=note_prob_dict,
            onset_ids=onset_ids,
            task_num_classes=task_num_classes,
            legal_rn_set=legal_rn_set,
            spec=cfg,
        )
    return _decode_onset_beam_legacy(
        note_prob_dict=note_prob_dict,
        onset_ids=onset_ids,
        task_num_classes=task_num_classes,
        legal_rn_set=legal_rn_set,
        spec=cfg,
    )


def build_smoothed_note_probs_from_class_ids(
    class_ids: torch.Tensor,
    num_classes: int,
    *,
    off_prob: float = 1e-3,
) -> torch.Tensor:
    if num_classes <= 1:
        return torch.ones((class_ids.numel(), 1), dtype=torch.float32, device=class_ids.device)
    off_prob = float(min(max(off_prob, 0.0), 0.25))
    on_prob = float(1.0 - off_prob)
    off_each = float(off_prob / float(num_classes - 1))
    probs = torch.full(
        (class_ids.numel(), num_classes),
        fill_value=off_each,
        dtype=torch.float32,
        device=class_ids.device,
    )
    probs.scatter_(1, class_ids.view(-1, 1), on_prob)
    probs = probs / torch.clamp(probs.sum(dim=-1, keepdim=True), min=1e-9)
    return probs
