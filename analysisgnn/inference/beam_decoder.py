"""Onset-level constrained beam decoding for RNA tasks."""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

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
}

DEFAULT_TRANSITION_WEIGHTS: Dict[str, float] = {
    "localkey_switch": 0.35,
    "inversion_jump": 0.04,
    "degree1_change": 0.08,
    "degree2_change": 0.06,
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

    tasks_cfg = cfg.get("tasks", list(DEFAULT_BEAM_TASKS))
    if isinstance(tasks_cfg, str):
        tasks_cfg = [t.strip() for t in tasks_cfg.split(",") if t.strip()]
    tasks = [str(t).strip() for t in tasks_cfg if str(t).strip()]
    if not tasks:
        tasks = list(DEFAULT_BEAM_TASKS)
    # Keep deterministic order.
    dedup: List[str] = []
    seen = set()
    for t in tasks:
        if t in seen:
            continue
        seen.add(t)
        dedup.append(t)
    tasks = dedup

    topk_cfg = cfg.get("topk_by_task", {})
    if not isinstance(topk_cfg, dict):
        topk_cfg = {}
    topk_by_task = dict(DEFAULT_TOPK_BY_TASK)
    for task, value in topk_cfg.items():
        try:
            topk_by_task[str(task)] = max(1, int(value))
        except Exception:
            continue

    emission_cfg = cfg.get("emission_weights", {})
    if not isinstance(emission_cfg, dict):
        emission_cfg = {}
    emission_weights = dict(DEFAULT_EMISSION_WEIGHTS)
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

    return {
        "enabled": enabled,
        "level": level,
        "beam_width": beam_width,
        "tasks": tasks,
        "topk_by_task": topk_by_task,
        "emission_weights": emission_weights,
        "transition_weights": transition_weights,
        "consistency_weight": consistency_weight,
        "epsilon": epsilon,
        "local_prune_factor": local_prune_factor,
        "smoothing_off_prob": smoothing_off_prob,
    }


def _class_labels_for_task(task: str, num_classes: int) -> List[Any]:
    rep = available_representations.get(task, None)
    class_list = getattr(rep, "classList", None) if rep is not None else None
    if isinstance(class_list, Sequence) and not isinstance(class_list, (str, bytes)):
        labels = list(class_list)
        if len(labels) >= num_classes:
            return labels[:num_classes]
        # Some checkpoints expose class counts that exceed the representation
        # list by 1-2 sentinel classes; keep known labels and pad tail with ids.
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
    return out / counts * counts  # keep shape and dtype stable


def decode_onset_beam(
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
            "beam_width": beam_width,
            "num_onsets": num_onsets,
            "steps": trace_steps,
            "best_score": float(best["score"]),
        },
    }


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
