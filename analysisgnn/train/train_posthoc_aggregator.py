#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from tqdm import tqdm
import wandb

from analysisgnn.data.datamodules.analysis import AnalysisDataModule
from analysisgnn.models.analysis import ContinualAnalysisGNN
from analysisgnn.models.posthoc_aggregator import DEFAULT_TASKS_BY_LEVEL, PosthocAggregationBundle
from analysisgnn.train.train_analysisgnn import TASK_DICT


def _parse_csv(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _resolve_tasks_by_level(args: argparse.Namespace) -> Dict[str, List[str]]:
    requested_levels = _parse_csv(args.levels) or ["onset", "beat", "measure"]
    out: Dict[str, List[str]] = {}
    for level in requested_levels:
        if level not in {"onset", "beat", "measure"}:
            raise ValueError(f"Unknown level '{level}'. Expected onset|beat|measure.")
        override = _parse_csv(getattr(args, f"tasks_{level}", ""))
        out[level] = override if override else list(DEFAULT_TASKS_BY_LEVEL.get(level, []))
    return out


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
) -> Tuple[Optional[torch.Tensor], int]:
    batch_size = int(batch["note"].batch_size)
    if batch_size <= 0:
        return None, 0

    x_dict = base_model._maybe_encode_x_dict(batch, batch.x_dict)
    labels_valid, task_mask_valid, valid_label_mask = _extract_valid_labels(base_model, batch, batch_size)
    if not labels_valid:
        return None, 0

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
        return None, 0

    ce_loss = torch.stack(task_losses).mean()
    entropy_term = bundle.mean_entropy()
    if entropy_term is not None and entropy_lambda > 0:
        ce_loss = ce_loss + float(entropy_lambda) * (-entropy_term)
    return ce_loss, len(task_losses)


def _epoch_loop(
    *,
    base_model: ContinualAnalysisGNN,
    bundle: PosthocAggregationBundle,
    loader,
    optimizer: Optional[torch.optim.Optimizer],
    entropy_lambda: float,
    train_mode: bool,
    device: torch.device,
) -> Dict[str, float]:
    running_loss = 0.0
    running_batches = 0
    running_tasks = 0

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
            loss, task_count = _compute_subbatch_loss(
                base_model=base_model,
                bundle=bundle,
                batch=sub_batch,
                entropy_lambda=entropy_lambda,
            )
            if loss is None:
                continue
            sub_losses.append(loss)
            sub_tasks += int(task_count)

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
    bundle = PosthocAggregationBundle(tasks_by_level=tasks_by_level).to(device)
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
                train_mode=False,
                device=device,
            )

        if args.use_wandb and wandb_run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/voter_total_loss": train_metrics["loss"],
                    "val/voter_total_loss": val_metrics["loss"],
                    "train/steps": train_metrics["steps"],
                    "val/steps": val_metrics["steps"],
                }
            )

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
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
