"""Train a lightweight rescorer for structured beam candidates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import torch
import torch.nn.functional as F

from analysisgnn.data.datamodules.analysis import AnalysisDataModule
from analysisgnn.inference.beam_decoder import (
    DEFAULT_BEAM_TASKS,
    StructuredBeamRescorer,
    extract_structured_candidate_features,
)
from analysisgnn.models.analysis import ContinualAnalysisGNN
from analysisgnn.models.musicbert_backbone import MusicBertAdapterConfig
from analysisgnn.models.musicbert_note_encoder import MusicBertNoteEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train structured beam rescorer")
    parser.add_argument("--base_checkpoint", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--main_tasks", type=str, default="rna")
    parser.add_argument("--feature_type", type=str, default="simple")
    parser.add_argument("--use_transpositions", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="AnalysisGNN-MusicBERT")
    parser.add_argument("--wandb_group", type=str, default="structured-beam-rescorer")
    parser.add_argument("--wandb_name", type=str, default="structured-beam-rescorer")
    parser.add_argument("--early_stop_patience", type=int, default=8)
    parser.add_argument("--beam_width", type=int, default=8)
    parser.add_argument("--nbest", type=int, default=8)
    parser.add_argument(
        "--beam_decoder_version",
        type=str,
        default="structured_v2",
        choices=["structured_v2", "component_v3"],
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _unwrap_loader_batch(batch):
    # CombinedLoader yields (payload, batch_idx, dataloader_idx) outside Lightning.
    if isinstance(batch, tuple) and len(batch) == 3 and isinstance(batch[1], int):
        batch = batch[0]
    # Some loaders still wrap the payload in a single-item tuple/list.
    while isinstance(batch, (tuple, list)) and len(batch) == 1:
        batch = batch[0]
    return batch


def _iter_batches(loader) -> Iterator:
    for batch in loader:
        batch = _unwrap_loader_batch(batch)
        if isinstance(batch, dict):
            for value in batch.values():
                if value is None:
                    continue
                value = _unwrap_loader_batch(value)
                yield value
        else:
            yield batch


def _build_note_encoder(config: Dict[str, object]):
    if not config.get("use_musicbert", False):
        return None
    if config.get("musicbert_use_cached_embeddings", False):
        return None
    adapter_cfg = MusicBertAdapterConfig(
        use_lora=bool(config.get("musicbert_use_lora", False)),
        lora_r=int(config.get("musicbert_lora_r", 8)),
        lora_alpha=int(config.get("musicbert_lora_alpha", 16)),
        lora_dropout=float(config.get("musicbert_lora_dropout", 0.1)),
    )
    return MusicBertNoteEncoder(
        pretrained_name=str(config.get("musicbert_model_name", "manoskary/musicbert-large")),
        adapter_cfg=adapter_cfg,
        freeze_backbone=bool(config.get("musicbert_freeze_backbone", True)),
    )


def _load_config_from_checkpoint(path: str) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu")
    hparams = checkpoint.get("hyper_parameters", {})
    if not isinstance(hparams, dict):
        raise ValueError("Checkpoint does not contain a hyperparameter dict.")
    return dict(hparams)


def _make_datamodule(
    config: Dict[str, object],
    batch_size: int,
    augment: bool,
    num_workers_override: Optional[int] = None,
) -> AnalysisDataModule:
    main_tasks = config.get("main_tasks", ["all", "rna"])
    if isinstance(main_tasks, str):
        main_tasks = [t.strip() for t in str(main_tasks).split(",") if t.strip()]
    datamodule = AnalysisDataModule(
        batch_size=batch_size,
        num_workers=(
            int(num_workers_override)
            if num_workers_override is not None
            else int(config.get("num_workers", 0))
        ),
        subgraph_size=int(config.get("subgraph_size", 1000)),
        num_neighbors=[5] * max(1, int(config.get("num_layers", 3)) - 1),
        raw_dir=str(config.get("raw_dir", "data")),
        force_reload=bool(config.get("force_reload", False)),
        verbose=bool(config.get("verbose", False)),
        tasks=list(config.get("task_dict", {}).keys()),
        random_split=bool(config.get("random_split", False)),
        max_samples=config.get("max_samples", None),
        main_tasks=main_tasks,
        remove_beats=not bool(config.get("add_beats", False)),
        remove_measures=not bool(config.get("add_measures", False)),
        feature_type=str(config.get("feature_type", "simple")),
        augment=augment,
        training_dataloader_type=str(config.get("training_dataloader_type", "combined")),
        alignment_dir=config.get("musicbert_alignment_dir") if not config.get("musicbert_use_cached_embeddings", False) else None,
        require_alignment=bool(config.get("use_musicbert", False) and not config.get("musicbert_use_cached_embeddings", False)),
        musicbert_embedding_cache_dir=config.get("musicbert_cached_embeddings_dir") if config.get("musicbert_use_cached_embeddings", False) else None,
        require_cached_embeddings=bool(config.get("musicbert_require_cached_embeddings", False)),
    )
    datamodule.setup()
    return datamodule


def _candidate_target_score(
    candidate: Dict[str, object],
    onset_labels: Dict[str, torch.Tensor],
    stable_mask: torch.Tensor,
    tasks: List[str],
) -> float:
    if not tasks or stable_mask.numel() == 0 or not torch.any(stable_mask):
        return 0.0
    matches = []
    for task in tasks:
        if task not in candidate.get("class_ids", {}) or task not in onset_labels:
            continue
        pred = torch.tensor(candidate["class_ids"][task], dtype=torch.long, device=stable_mask.device)
        gold = onset_labels[task].to(device=stable_mask.device, dtype=torch.long)
        usable = stable_mask & (gold >= 0)
        if usable.numel() == 0 or not torch.any(usable):
            continue
        pred = pred[: gold.numel()]
        matches.append((pred[usable] == gold[usable]).float().mean())
    if not matches:
        return 0.0
    return float(torch.stack(matches).mean().item())


def _run_epoch(
    *,
    model: ContinualAnalysisGNN,
    rescorer: StructuredBeamRescorer,
    loader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    beam_width: int,
    nbest: int,
    beam_decoder_version: str,
) -> float:
    running_loss = 0.0
    count = 0
    is_train = optimizer is not None
    rescorer.train(is_train)
    for batch in _iter_batches(loader):
        batch = batch.to(device)
        batch_size = int(batch["note"].batch_size)
        labels_dict = {
            task: batch["note"][task][:batch_size]
            for task in model.task_dict.keys()
            if task in batch["note"].keys()
        }
        if beam_decoder_version == "component_v3":
            tasks = model._component_refine_tasks_for_available_labels(list(labels_dict.keys()))
        else:
            tasks = model._structured_refine_tasks_for_available_labels(list(labels_dict.keys()))
        if not tasks:
            continue
        x_dict = model._maybe_encode_x_dict(batch, batch.x_dict)
        with torch.no_grad():
            note_probs = model._predict_note_probs_once(
                data=batch,
                batch_size=batch_size,
                conditioning=None,
                overrides=None,
                x_dict_override=x_dict,
            )
            onset_probs = model._aggregate_onset_only_note_probs(
                note_prob_dict=note_probs,
                data=batch,
                batch_size=batch_size,
                aggregation_mode="mean",
                aggregation_bundle=None,
            )
            beam_payload = model._decode_beam_from_note_probs(
                note_prob_dict=onset_probs,
                data=batch,
                batch_size=batch_size,
                enabled_override=True,
                version_override=beam_decoder_version,
            )
        if beam_payload is None:
            continue
        candidates = beam_payload.get("beam_trace", {}).get("nbest", [])
        if not candidates or len(candidates) < 2:
            continue
        onset_labels, stable_mask, _, _, _ = model._aggregate_onset_labels_from_labels(
            labels_dict=labels_dict,
            batch=batch,
            batch_size=batch_size,
            tasks=tasks,
            device=device,
        )
        target_scores = [
            _candidate_target_score(candidate, onset_labels, stable_mask, tasks)
            for candidate in candidates
        ]
        best_idx = int(max(range(len(target_scores)), key=lambda idx: target_scores[idx]))
        features = torch.stack(
            [extract_structured_candidate_features(candidate) for candidate in candidates],
            dim=0,
        ).to(device)
        scores = rescorer(features)
        target = torch.tensor([best_idx], dtype=torch.long, device=device)
        loss = F.cross_entropy(scores.unsqueeze(0), target)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        running_loss += float(loss.detach().cpu().item())
        count += 1
    return running_loss / max(1, count)


def main() -> None:
    args = parse_args()
    config = _load_config_from_checkpoint(args.base_checkpoint)
    config["main_tasks"] = [t.strip() for t in args.main_tasks.split(",") if t.strip()]
    config["feature_type"] = args.feature_type
    note_encoder = _build_note_encoder(config)
    datamodule = _make_datamodule(
        config,
        batch_size=args.batch_size,
        augment=bool(args.use_transpositions),
        num_workers_override=args.num_workers,
    )
    model = ContinualAnalysisGNN.load_from_checkpoint(
        args.base_checkpoint,
        strict=False,
        note_encoder=note_encoder,
    )
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    rescorer = StructuredBeamRescorer().to(device)
    optimizer = torch.optim.AdamW(rescorer.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_state = None
    patience = 0
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    use_wandb = bool(args.use_wandb)
    if use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_name,
            config=vars(args),
        )
    for epoch in range(args.num_epochs):
        train_loss = _run_epoch(
            model=model,
            rescorer=rescorer,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            beam_width=args.beam_width,
            nbest=args.nbest,
            beam_decoder_version=args.beam_decoder_version,
        )
        with torch.no_grad():
            val_loss = _run_epoch(
                model=model,
                rescorer=rescorer,
                loader=val_loader,
                optimizer=None,
                device=device,
                beam_width=args.beam_width,
                nbest=args.nbest,
                beam_decoder_version=args.beam_decoder_version,
            )
        if use_wandb:
            wandb.log({"train/loss": train_loss, "val/loss": val_loss, "epoch": epoch})
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in rescorer.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                break

    if best_state is None:
        raise RuntimeError("Rescorer training produced no valid updates.")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": best_state,
        "hidden_dim": 16,
        "dropout": 0.1,
        "metadata": {
            "feature_names": list(StructuredBeamRescorer.feature_names),
            "beam_width": args.beam_width,
            "nbest": args.nbest,
            "base_checkpoint": args.base_checkpoint,
            "tasks": list(DEFAULT_BEAM_TASKS),
            "val_loss": best_val,
        },
    }
    torch.save(payload, output_path)
    print(f"Saved structured beam rescorer to {output_path}")


if __name__ == "__main__":
    main()
