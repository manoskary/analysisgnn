#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from tqdm import tqdm
import wandb

from analysisgnn.data.datamodules.analysis import AnalysisDataModule
from analysisgnn.models.analysis import ContinualAnalysisGNN
from analysisgnn.models.posthoc_aggregator import (
    DEFAULT_BEAT_OUTPUT_TASKS,
    DEFAULT_TASKS_BY_LEVEL,
    HARMONIC_BEAT_TASKS,
    PosthocAggregationBundle,
)
from analysisgnn.train.train_analysisgnn import TASK_DICT

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover - optional dependency
    roc_auc_score = None


def _parse_csv(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _resolve_tasks_by_level(args: argparse.Namespace) -> Dict[str, List[str]]:
    requested_levels = _parse_csv(args.levels) or ["onset", "beat", "measure"]
    if bool(getattr(args, "train_consistent_beat_voter", False)) and "beat" not in requested_levels:
        requested_levels.append("beat")
    out: Dict[str, List[str]] = {}
    for level in requested_levels:
        if level not in {"onset", "beat", "measure"}:
            raise ValueError(f"Unknown level '{level}'. Expected onset|beat|measure.")
        override = _parse_csv(getattr(args, f"tasks_{level}", ""))
        out[level] = override if override else list(DEFAULT_TASKS_BY_LEVEL.get(level, []))
    if bool(getattr(args, "train_consistent_beat_voter", False)):
        beat_override = _parse_csv(getattr(args, "beat_output_tasks", ""))
        if beat_override:
            out["beat"] = beat_override
        elif "beat" not in out or not out["beat"]:
            out["beat"] = list(DEFAULT_BEAT_OUTPUT_TASKS)
    return out


def _beat_group_ids(batch, batch_size: int) -> torch.Tensor:
    note_store = batch["note"]
    device = note_store.x.device
    cluster = getattr(note_store, "beat_cluster", None)
    if isinstance(cluster, torch.Tensor) and cluster.numel() >= batch_size:
        return cluster[:batch_size].to(device=device, dtype=torch.long)

    out = torch.full((batch_size,), -1, dtype=torch.long, device=device)
    edge_index_dict = getattr(batch, "edge_index_dict", {})
    key = ("beat", "connects", "note")
    if key not in edge_index_dict:
        return out
    edge = edge_index_dict[key]
    if edge.numel() == 0:
        return out
    edge = edge[:, edge[1] < batch_size]
    if edge.numel() == 0:
        return out
    out[edge[1]] = edge[0].to(dtype=torch.long)
    return out


def _apply_harmonic_filter_per_beat(
    *,
    base_eligible: torch.Tensor,
    chord_tone_mask: Optional[torch.Tensor],
    beat_group_ids: torch.Tensor,
) -> torch.Tensor:
    if chord_tone_mask is None:
        return base_eligible
    eligible = base_eligible.clone()
    valid_mask = eligible & (beat_group_ids >= 0)
    if not torch.any(valid_mask):
        return eligible
    unique_beats = torch.unique(beat_group_ids[valid_mask], sorted=True)
    for beat in unique_beats.tolist():
        beat_mask = valid_mask & (beat_group_ids == int(beat))
        if not torch.any(beat_mask):
            continue
        beat_chord = beat_mask & chord_tone_mask
        if torch.any(beat_chord):
            eligible[beat_mask] = False
            eligible[beat_chord] = True
    return eligible


def _build_beat_targets(
    *,
    labels: torch.Tensor,
    beat_group_ids: torch.Tensor,
    eligible_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = eligible_mask & (beat_group_ids >= 0)
    if not torch.any(valid):
        device = labels.device
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_float = torch.empty(0, dtype=torch.float32, device=device)
        return empty_long, empty_long, empty_long, empty_float

    ce_anchor_indices: List[int] = []
    ce_labels: List[int] = []
    conflict_beats: List[int] = []
    conflict_targets: List[float] = []
    unique_beats = torch.unique(beat_group_ids[valid], sorted=True)
    for beat in unique_beats.tolist():
        idx = torch.where(valid & (beat_group_ids == int(beat)))[0]
        if idx.numel() == 0:
            continue
        beat_labels = labels[idx]
        if beat_labels.numel() == 0:
            continue
        first = int(beat_labels[0].item())
        is_consistent = bool(torch.all(beat_labels == beat_labels[0]).item())
        conflict_beats.append(int(beat))
        conflict_targets.append(0.0 if is_consistent else 1.0)
        if is_consistent:
            ce_anchor_indices.append(int(idx[0].item()))
            ce_labels.append(first)

    device = labels.device
    return (
        torch.tensor(ce_anchor_indices, dtype=torch.long, device=device),
        torch.tensor(ce_labels, dtype=torch.long, device=device),
        torch.tensor(conflict_beats, dtype=torch.long, device=device),
        torch.tensor(conflict_targets, dtype=torch.float32, device=device),
    )


def _binary_f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall <= 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _binary_auroc(probs: List[float], targets: List[int]) -> float:
    if not probs or not targets:
        return float("nan")
    if len(set(int(t) for t in targets)) < 2:
        return float("nan")
    if roc_auc_score is None:
        return float("nan")
    try:
        return float(roc_auc_score(targets, probs))
    except Exception:
        return float("nan")


def _build_datamodule(
    *,
    ckpt_hparams: Dict[str, object],
    args: argparse.Namespace,
    use_cached_embeddings: bool,
) -> AnalysisDataModule:
    main_tasks = _parse_csv(args.main_tasks) or list(ckpt_hparams.get("main_tasks", ["all", "rna"]))
    feature_type = args.feature_type or ckpt_hparams.get("feature_type", "simple")
    use_transpositions = (
        bool(args.use_transpositions)
        if args.use_transpositions is not None
        else bool(ckpt_hparams.get("use_transpositions", False))
    )
    num_layers = int(ckpt_hparams.get("num_layers", 3))
    subgraph_size = int(ckpt_hparams.get("subgraph_size", 500))
    num_workers = int(args.num_workers if args.num_workers is not None else ckpt_hparams.get("num_workers", 5))

    alignment_dir = args.musicbert_alignment_dir
    if alignment_dir is None:
        alignment_dir = ckpt_hparams.get("musicbert_alignment_dir", None)

    cached_embeddings_dir = args.musicbert_cached_embeddings_dir
    if cached_embeddings_dir is None:
        cached_embeddings_dir = ckpt_hparams.get("musicbert_cached_embeddings_dir", None)

    return AnalysisDataModule(
        batch_size=int(args.batch_size),
        num_workers=num_workers,
        subgraph_size=subgraph_size,
        num_neighbors=[5] * max(1, num_layers - 1),
        raw_dir=args.raw_dir if args.raw_dir is not None else ckpt_hparams.get("raw_dir", None),
        force_reload=bool(args.force_reload),
        verbose=bool(args.verbose),
        tasks=list(TASK_DICT.keys()),
        random_split=bool(ckpt_hparams.get("random_split", False)),
        max_samples=args.max_samples,
        main_tasks=main_tasks,
        remove_beats=not bool(ckpt_hparams.get("add_beats", False)),
        remove_measures=not bool(ckpt_hparams.get("add_measures", False)),
        feature_type=str(feature_type),
        augment=use_transpositions,
        training_dataloader_type="combined",
        alignment_dir=alignment_dir if not use_cached_embeddings else None,
        require_alignment=bool(ckpt_hparams.get("use_musicbert", False)) and not use_cached_embeddings,
        musicbert_embedding_cache_dir=cached_embeddings_dir if use_cached_embeddings else None,
        require_cached_embeddings=use_cached_embeddings,
    )


def _iter_subbatches(batch_obj):
    while isinstance(batch_obj, tuple) and len(batch_obj) > 0:
        candidate = None
        for item in batch_obj:
            if hasattr(item, "x_dict"):
                candidate = item
                break
            if isinstance(item, dict):
                candidate = item
                break
        batch_obj = candidate if candidate is not None else batch_obj[0]

    if isinstance(batch_obj, dict):
        for _, sub in batch_obj.items():
            if sub is not None:
                item = sub
                while isinstance(item, tuple) and len(item) > 0:
                    nested = None
                    for val in item:
                        if hasattr(val, "x_dict"):
                            nested = val
                            break
                        if isinstance(val, dict):
                            nested = val
                            break
                    item = nested if nested is not None else item[0]
                if item is not None:
                    yield item
        return
    if batch_obj is not None:
        yield batch_obj


def _extract_per_note_supervision(
    model: ContinualAnalysisGNN,
    batch,
    batch_size: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
    labels_dict = {
        task: batch["note"][task][:batch_size]
        for task in model.task_dict.keys()
        if task in batch["note"].keys()
    }
    if not labels_dict:
        return {}, {}, torch.zeros(batch_size, dtype=torch.bool, device=batch["note"].x.device)

    mask_dict = model.create_mask_dict(labels_dict, batch, batch_size)
    if "valid_label" in batch["note"].keys():
        valid_label_mask = batch["note"]["valid_label"][:batch_size].bool()
    else:
        valid_label_mask = torch.ones(batch_size, dtype=torch.bool, device=batch["note"].x.device)

    labels_out: Dict[str, torch.Tensor] = {}
    task_mask_out: Dict[str, torch.Tensor] = {}
    for task in labels_dict.keys():
        labels_task = labels_dict[task][:batch_size]
        mask_task = mask_dict[task][:batch_size].bool()
        valid_task = model._metric_safe_mask(labels_task, task).bool()
        labels_out[task] = labels_task
        task_mask_out[task] = valid_label_mask & mask_task & valid_task
    return labels_out, task_mask_out, valid_label_mask


def _extract_valid_labels(
    model: ContinualAnalysisGNN,
    batch,
    batch_size: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
    labels_dict = {
        task: batch["note"][task][:batch_size]
        for task in model.task_dict.keys()
        if task in batch["note"].keys()
    }
    if not labels_dict:
        return {}, {}, torch.zeros(batch_size, dtype=torch.bool, device=batch["note"].x.device)
    mask_dict = model.create_mask_dict(labels_dict, batch, batch_size)
    if "valid_label" in batch["note"].keys():
        valid_label_mask = batch["note"]["valid_label"][:batch_size].bool()
    else:
        valid_label_mask = torch.ones(batch_size, dtype=torch.bool, device=batch["note"].x.device)

    labels_valid: Dict[str, torch.Tensor] = {}
    task_mask_valid: Dict[str, torch.Tensor] = {}
    for task in labels_dict.keys():
        labels_task = labels_dict[task][valid_label_mask]
        mask_task = mask_dict[task][valid_label_mask]
        valid_task = model._metric_safe_mask(labels_task, task)
        labels_task = labels_task[valid_task]
        mask_task = mask_task[valid_task]
        labels_valid[task] = labels_task
        task_mask_valid[task] = mask_task
    return labels_valid, task_mask_valid, valid_label_mask


def _compute_subbatch_loss(
    *,
    base_model: ContinualAnalysisGNN,
    bundle: PosthocAggregationBundle,
    batch,
    entropy_lambda: float,
    train_consistent_beat_voter: bool = False,
    conflict_lambda: float = 1.0,
    conflict_threshold: float = 0.5,
) -> Tuple[Optional[torch.Tensor], int, Dict[str, Any]]:
    batch_size = int(batch["note"].batch_size)
    if batch_size <= 0:
        return None, 0, {}

    x_dict = base_model._maybe_encode_x_dict(batch, batch.x_dict)
    labels_per_note, task_valid_per_note, valid_label_mask = _extract_per_note_supervision(
        base_model,
        batch,
        batch_size,
    )
    if not labels_per_note:
        return None, 0, {}

    with torch.no_grad():
        note_probs = base_model._predict_note_probs_once(
            data=batch,
            batch_size=batch_size,
            conditioning=None,
            overrides=None,
            x_dict_override=x_dict,
            neighbor_mask_node=batch.num_sampled_nodes_dict,
            neighbor_mask_edge=batch.num_sampled_edges_dict,
        )

    bundle.reset_entropy_stats()
    if not train_consistent_beat_voter:
        labels_valid, task_mask_valid, valid_label_mask = _extract_valid_labels(
            model=base_model,
            batch=batch,
            batch_size=batch_size,
        )
        if not labels_valid:
            return None, 0, {}

        agg_probs = base_model._aggregate_note_probs(
            note_prob_dict=note_probs,
            data=batch,
            batch_size=batch_size,
            aggregation_mode="voter",
            aggregation_bundle=bundle,
        )
        task_losses: List[torch.Tensor] = []
        for task, labels in labels_valid.items():
            if task not in agg_probs:
                continue
            if labels.numel() == 0:
                continue

            probs = agg_probs[task][:batch_size]
            probs = probs[valid_label_mask]
            task_mask = task_mask_valid[task].bool()
            if task_mask.numel() != probs.size(0):
                min_len = min(int(task_mask.numel()), int(probs.size(0)))
                task_mask = task_mask[:min_len]
                probs = probs[:min_len]
                labels = labels[:min_len]
            probs = probs[task_mask]
            labels = labels[task_mask]
            if labels.numel() == 0:
                continue
            valid_cls = (labels >= 0) & (labels < probs.size(-1))
            if not torch.any(valid_cls):
                continue
            probs = probs[valid_cls]
            labels = labels[valid_cls].long()
            log_probs = torch.log(torch.clamp(probs, min=1e-8))
            task_losses.append(F.nll_loss(log_probs, labels, reduction="mean"))

        if not task_losses:
            return None, 0, {}
        ce_loss = torch.stack(task_losses).mean()
        loss = ce_loss
        aux = {
            "consistent_ce": float(ce_loss.detach().cpu().item()),
            "conflict_bce": float("nan"),
            "consistent_terms": float(len(task_losses)),
            "conflict_terms": 0.0,
            "conflict_tp": 0.0,
            "conflict_fp": 0.0,
            "conflict_fn": 0.0,
            "conflict_probs": [],
            "conflict_targets": [],
        }
        entropy_term = bundle.mean_entropy()
        if entropy_term is not None and entropy_lambda > 0:
            loss = loss + float(entropy_lambda) * (-entropy_term)
        return loss, len(task_losses), aux

    beat_groups = _beat_group_ids(batch, batch_size=batch_size)
    beat_tasks = list(bundle.tasks_by_level.get("beat", []))
    if not beat_tasks:
        return None, 0, {}

    task_ce_losses: List[torch.Tensor] = []
    conflict_losses: List[torch.Tensor] = []
    conflict_probs_all: List[float] = []
    conflict_targets_all: List[int] = []
    conflict_tp = 0
    conflict_fp = 0
    conflict_fn = 0

    tpc_labels = labels_per_note.get("tpc_in_label")
    tpc_valid = task_valid_per_note.get("tpc_in_label")
    for task in beat_tasks:
        if task not in note_probs or task not in labels_per_note:
            continue
        labels = labels_per_note[task][:batch_size]
        base_eligible = task_valid_per_note[task][:batch_size]
        if not torch.any(base_eligible):
            continue

        chord_mask = None
        if task in HARMONIC_BEAT_TASKS and tpc_labels is not None and tpc_valid is not None:
            chord_mask = (tpc_valid[:batch_size] & (tpc_labels[:batch_size] == 1))
        eligible = _apply_harmonic_filter_per_beat(
            base_eligible=base_eligible,
            chord_tone_mask=chord_mask,
            beat_group_ids=beat_groups,
        )
        if not torch.any(eligible & (beat_groups >= 0)):
            continue

        task_probs = note_probs[task][:batch_size]
        agg_probs, _, group_conflict_prob, group_values = bundle.aggregate_task_with_conflict(
            level="beat",
            task=task,
            task_probs=task_probs,
            all_task_probs=note_probs,
            graph=batch,
            group_ids=beat_groups,
            eligible_mask=eligible,
        )

        ce_anchor_idx, ce_labels, conflict_beats, conflict_targets = _build_beat_targets(
            labels=labels,
            beat_group_ids=beat_groups,
            eligible_mask=eligible,
        )
        if ce_anchor_idx.numel() > 0:
            ce_probs = agg_probs[ce_anchor_idx]
            log_probs = torch.log(torch.clamp(ce_probs, min=1e-8))
            task_ce_losses.append(F.nll_loss(log_probs, ce_labels.long(), reduction="mean"))

        if (
            group_conflict_prob is not None
            and group_values is not None
            and conflict_beats.numel() > 0
            and conflict_targets.numel() > 0
        ):
            beat_to_pos = {int(g): i for i, g in enumerate(group_values.tolist())}
            pred_list: List[torch.Tensor] = []
            tgt_list: List[float] = []
            for beat_id, target in zip(conflict_beats.tolist(), conflict_targets.tolist()):
                pos = beat_to_pos.get(int(beat_id))
                if pos is None:
                    continue
                pred_list.append(group_conflict_prob[pos])
                tgt_list.append(float(target))
            if pred_list:
                pred_tensor = torch.stack(pred_list)
                tgt_tensor = torch.tensor(tgt_list, dtype=pred_tensor.dtype, device=pred_tensor.device)
                conflict_losses.append(F.binary_cross_entropy(pred_tensor, tgt_tensor, reduction="mean"))
                pred_bin = (pred_tensor >= float(conflict_threshold)).long()
                tgt_bin = tgt_tensor.long()
                conflict_tp += int(((pred_bin == 1) & (tgt_bin == 1)).sum().item())
                conflict_fp += int(((pred_bin == 1) & (tgt_bin == 0)).sum().item())
                conflict_fn += int(((pred_bin == 0) & (tgt_bin == 1)).sum().item())
                conflict_probs_all.extend(pred_tensor.detach().cpu().tolist())
                conflict_targets_all.extend(tgt_bin.detach().cpu().tolist())

    if not task_ce_losses and not conflict_losses:
        return None, 0, {}

    ce_loss = torch.stack(task_ce_losses).mean() if task_ce_losses else torch.tensor(
        0.0,
        device=batch["note"].x.device,
    )
    conflict_loss = torch.stack(conflict_losses).mean() if conflict_losses else torch.tensor(
        0.0,
        device=batch["note"].x.device,
    )
    loss = ce_loss + float(conflict_lambda) * conflict_loss
    aux = {
        "consistent_ce": float(ce_loss.detach().cpu().item()),
        "conflict_bce": float(conflict_loss.detach().cpu().item()),
        "consistent_terms": float(len(task_ce_losses)),
        "conflict_terms": float(len(conflict_losses)),
        "conflict_tp": float(conflict_tp),
        "conflict_fp": float(conflict_fp),
        "conflict_fn": float(conflict_fn),
        "conflict_probs": conflict_probs_all,
        "conflict_targets": conflict_targets_all,
    }

    entropy_term = bundle.mean_entropy()
    if entropy_term is not None and entropy_lambda > 0:
        loss = loss + float(entropy_lambda) * (-entropy_term)
    task_terms = int(len(task_ce_losses) + len(conflict_losses))
    return loss, task_terms, aux


def _epoch_loop(
    *,
    base_model: ContinualAnalysisGNN,
    bundle: PosthocAggregationBundle,
    loader,
    optimizer: Optional[torch.optim.Optimizer],
    entropy_lambda: float,
    train_consistent_beat_voter: bool,
    conflict_lambda: float,
    conflict_threshold: float,
    train_mode: bool,
    device: torch.device,
) -> Dict[str, float]:
    running_loss = 0.0
    running_batches = 0
    running_tasks = 0
    running_consistent_ce = 0.0
    running_conflict_bce = 0.0
    running_consistent_terms = 0.0
    running_conflict_terms = 0.0
    conflict_tp = 0
    conflict_fp = 0
    conflict_fn = 0
    conflict_probs_all: List[float] = []
    conflict_targets_all: List[int] = []

    if train_mode:
        bundle.train()
    else:
        bundle.eval()

    try:
        total_batches = len(loader)
    except Exception:
        total_batches = None
    iterator = tqdm(iter(loader), total=total_batches, desc="train" if train_mode else "val", leave=False)
    for batch_obj in iterator:
        sub_losses: List[torch.Tensor] = []
        sub_tasks = 0
        for sub_batch in _iter_subbatches(batch_obj):
            if hasattr(sub_batch, "to"):
                sub_batch = sub_batch.to(device)
            loss, task_count, aux = _compute_subbatch_loss(
                base_model=base_model,
                bundle=bundle,
                batch=sub_batch,
                entropy_lambda=entropy_lambda,
                train_consistent_beat_voter=train_consistent_beat_voter,
                conflict_lambda=conflict_lambda,
                conflict_threshold=conflict_threshold,
            )
            if loss is None:
                continue
            sub_losses.append(loss)
            sub_tasks += int(task_count)
            running_consistent_ce += float(aux.get("consistent_ce", 0.0))
            conf_bce = aux.get("conflict_bce", float("nan"))
            if conf_bce is not None and not math.isnan(float(conf_bce)):
                running_conflict_bce += float(conf_bce)
            running_consistent_terms += float(aux.get("consistent_terms", 0.0))
            running_conflict_terms += float(aux.get("conflict_terms", 0.0))
            conflict_tp += int(aux.get("conflict_tp", 0.0))
            conflict_fp += int(aux.get("conflict_fp", 0.0))
            conflict_fn += int(aux.get("conflict_fn", 0.0))
            conflict_probs_all.extend([float(x) for x in aux.get("conflict_probs", [])])
            conflict_targets_all.extend([int(x) for x in aux.get("conflict_targets", [])])

        if not sub_losses:
            continue
        batch_loss = torch.stack(sub_losses).mean()
        if train_mode:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(bundle.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += float(batch_loss.detach().cpu().item())
        running_batches += 1
        running_tasks += sub_tasks
        iterator.set_postfix({"loss": f"{running_loss / max(running_batches, 1):.4f}"})

    return {
        "loss": running_loss / max(running_batches, 1),
        "steps": float(running_batches),
        "task_terms": float(running_tasks),
        "consistent_ce": running_consistent_ce / max(running_batches, 1),
        "conflict_bce": (
            running_conflict_bce / max(running_batches, 1)
            if running_conflict_terms > 0
            else float("nan")
        ),
        "conflict_f1": _binary_f1(conflict_tp, conflict_fp, conflict_fn),
        "conflict_auroc": _binary_auroc(conflict_probs_all, conflict_targets_all),
        "conflict_tp": float(conflict_tp),
        "conflict_fp": float(conflict_fp),
        "conflict_fn": float(conflict_fn),
        "device": str(device),
    }


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train post-hoc weighted aggregation voters.")
    parser.add_argument("--base_checkpoint", type=str, required=True, help="Path to frozen base AnalysisGNN checkpoint.")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save voter artifact (.pt).")
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--raw_dir", type=str, default=None)
    parser.add_argument("--force_reload", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--gpus", type=str, default="0", help="GPU id list (e.g. 0 or 0,1). Use -1 for CPU.")

    parser.add_argument("--main_tasks", type=str, default=None)
    parser.add_argument("--feature_type", type=str, default=None, choices=["cadence", "simple"])
    parser.add_argument("--use_transpositions", dest="use_transpositions", action="store_true")
    parser.add_argument("--no_use_transpositions", dest="use_transpositions", action="store_false")
    parser.set_defaults(use_transpositions=None)

    parser.add_argument("--use_musicbert", action="store_true")
    parser.add_argument("--musicbert_alignment_dir", type=str, default=None)
    parser.add_argument("--musicbert_cached_embeddings_dir", type=str, default=None)

    parser.add_argument("--levels", type=str, default="onset,beat,measure")
    parser.add_argument("--tasks_onset", type=str, default="")
    parser.add_argument("--tasks_beat", type=str, default="")
    parser.add_argument("--tasks_measure", type=str, default="")
    parser.add_argument(
        "--train_consistent_beat_voter",
        action="store_true",
        help="Train consistency-aware beat voter (strict consistent beats + conflict heads).",
    )
    parser.add_argument(
        "--beat_output_tasks",
        type=str,
        default=",".join(DEFAULT_BEAT_OUTPUT_TASKS),
        help="Comma-separated beat tasks for consistent beat voter mode.",
    )
    parser.add_argument(
        "--conflict_lambda",
        type=float,
        default=1.0,
        help="Weight for beat conflict BCE loss (consistent beat voter mode).",
    )
    parser.add_argument(
        "--conflict_threshold",
        type=float,
        default=0.5,
        help="Threshold for conflict classification metrics.",
    )
    parser.add_argument(
        "--consistency_policy",
        type=str,
        default="strict_all_notes",
        help="Metadata tag persisted with the trained voter artifact.",
    )
    parser.add_argument(
        "--harmonic_filter_policy",
        type=str,
        default="chord_tone_only",
        help="Metadata tag persisted with the trained voter artifact.",
    )

    parser.add_argument("--entropy_lambda", type=float, default=0.01, help="Weight for -entropy regularization.")
    parser.add_argument("--early_stop_patience", type=int, default=8)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="AnalysisGNN-MusicBERT")
    parser.add_argument("--wandb_entity", type=str, default="melkisedeath")
    parser.add_argument("--wandb_group", type=str, default="posthoc-voter")
    parser.add_argument("--wandb_name", type=str, default=None)
    return parser


def _resolve_device(gpus: str) -> torch.device:
    if str(gpus).strip() == "-1" or not torch.cuda.is_available():
        return torch.device("cpu")
    gpu_list = [x.strip() for x in str(gpus).split(",") if x.strip()]
    gpu_idx = int(gpu_list[0]) if gpu_list else 0
    return torch.device(f"cuda:{gpu_idx}")


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(0, workers=True)

    if not os.path.exists(args.base_checkpoint):
        raise FileNotFoundError(f"Base checkpoint not found: {args.base_checkpoint}")

    device = _resolve_device(args.gpus)
    ckpt = torch.load(args.base_checkpoint, map_location="cpu")
    ckpt_hparams = ckpt.get("hyper_parameters", {}) if isinstance(ckpt, dict) else {}

    use_cached_embeddings = bool(
        args.musicbert_cached_embeddings_dir
        or ckpt_hparams.get("musicbert_cached_embeddings_dir")
    )

    datamodule = _build_datamodule(
        ckpt_hparams=ckpt_hparams,
        args=args,
        use_cached_embeddings=use_cached_embeddings,
    )
    datamodule.setup()

    base_model = ContinualAnalysisGNN.load_from_checkpoint(
        args.base_checkpoint,
        map_location=device,
        strict=False,
    )
    base_model.to(device)
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False

    tasks_by_level = _resolve_tasks_by_level(args)
    train_consistent_beat_voter = bool(args.train_consistent_beat_voter)
    bundle = PosthocAggregationBundle(
        tasks_by_level=tasks_by_level,
        consistency_policy=args.consistency_policy if train_consistent_beat_voter else "none",
        harmonic_filter_policy=args.harmonic_filter_policy if train_consistent_beat_voter else "all_notes",
        use_conflict_heads=train_consistent_beat_voter,
    ).to(device)
    bundle.enable_entropy_stats(True)

    optimizer = torch.optim.AdamW(bundle.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    wandb_run = None
    if args.use_wandb:
        wandb_name = args.wandb_name or f"posthoc-voter-{Path(args.base_checkpoint).stem}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=wandb_name,
            config={
                "base_checkpoint": args.base_checkpoint,
                "tasks_by_level": tasks_by_level,
                "num_epochs": args.num_epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "entropy_lambda": args.entropy_lambda,
                "train_consistent_beat_voter": train_consistent_beat_voter,
                "conflict_lambda": args.conflict_lambda,
                "conflict_threshold": args.conflict_threshold,
                "consistency_policy": args.consistency_policy,
                "harmonic_filter_policy": args.harmonic_filter_policy,
                "device": str(device),
            },
        )

    best_val = float("inf")
    best_state = copy.deepcopy(bundle.state_dict())
    best_epoch = -1
    bad_epochs = 0

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    for epoch in range(int(args.num_epochs)):
        train_metrics = _epoch_loop(
            base_model=base_model,
            bundle=bundle,
            loader=train_loader,
            optimizer=optimizer,
            entropy_lambda=float(args.entropy_lambda),
            train_consistent_beat_voter=train_consistent_beat_voter,
            conflict_lambda=float(args.conflict_lambda),
            conflict_threshold=float(args.conflict_threshold),
            train_mode=True,
            device=device,
        )
        with torch.no_grad():
            val_metrics = _epoch_loop(
                base_model=base_model,
                bundle=bundle,
                loader=val_loader,
                optimizer=None,
                entropy_lambda=float(args.entropy_lambda),
                train_consistent_beat_voter=train_consistent_beat_voter,
                conflict_lambda=float(args.conflict_lambda),
                conflict_threshold=float(args.conflict_threshold),
                train_mode=False,
                device=device,
            )

        if args.use_wandb and wandb_run is not None:
            log_payload = {
                "epoch": epoch,
                "train/voter_total_loss": train_metrics["loss"],
                "val/voter_total_loss": val_metrics["loss"],
                "train/voter_consistent_ce": train_metrics["consistent_ce"],
                "val/voter_consistent_ce": val_metrics["consistent_ce"],
                "train/voter_conflict_bce": train_metrics["conflict_bce"],
                "val/voter_conflict_bce": val_metrics["conflict_bce"],
                "train/voter_conflict_f1": train_metrics["conflict_f1"],
                "val/voter_conflict_f1": val_metrics["conflict_f1"],
                "train/voter_conflict_auroc": train_metrics["conflict_auroc"],
                "val/voter_conflict_auroc": val_metrics["conflict_auroc"],
                "train/steps": train_metrics["steps"],
                "val/steps": val_metrics["steps"],
            }
            wandb.log(log_payload)

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_consistent_ce={val_metrics['consistent_ce']:.6f} "
            f"val_conflict_f1={val_metrics['conflict_f1']:.4f}"
        )

        monitor_val = float(val_metrics["loss"])
        if train_consistent_beat_voter:
            consistent_val = float(val_metrics.get("consistent_ce", float("inf")))
            if math.isfinite(consistent_val):
                monitor_val = consistent_val

        if monitor_val < best_val:
            best_val = monitor_val
            best_state = copy.deepcopy(bundle.state_dict())
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(args.early_stop_patience):
                print(f"Early stopping at epoch {epoch} (best epoch={best_epoch}, best val={best_val:.6f}).")
                break

    bundle.load_state_dict(best_state, strict=True)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": bundle.state_dict(),
        "metadata": bundle.to_serializable_metadata(),
        "base_checkpoint": os.path.abspath(args.base_checkpoint),
        "task_dict": dict(base_model.task_dict),
        "levels": list(tasks_by_level.keys()),
        "tasks_by_level": tasks_by_level,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "train_consistent_beat_voter": train_consistent_beat_voter,
        "consistency_policy": args.consistency_policy,
        "harmonic_filter_policy": args.harmonic_filter_policy,
    }
    torch.save(payload, output_path)
    print(f"Saved post-hoc voter artifact to: {output_path}")

    if args.use_wandb and wandb_run is not None:
        wandb.summary["best_epoch"] = int(best_epoch)
        wandb.summary["best_val_loss"] = float(best_val)
        wandb.summary["output_path"] = str(output_path)
        artifact = wandb.Artifact("posthoc-voter", type="model")
        artifact.add_file(str(output_path))
        wandb_run.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    main()
