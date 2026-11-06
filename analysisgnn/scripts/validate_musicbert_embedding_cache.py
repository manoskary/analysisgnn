#!/usr/bin/env python3
"""
Validate cached note-level MusicBERT embeddings against on-the-fly encoding.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from analysisgnn.data.remi_bpe_aligner import load_alignment_npz
from analysisgnn.models.musicbert_backbone import MusicBertAdapterConfig
from analysisgnn.models.musicbert_note_encoder import MusicBertNoteEncoder


@torch.inference_mode()
def _encode_alignment(note_encoder: MusicBertNoteEncoder, alignment_path: Path, device: torch.device) -> np.ndarray:
    alignment = load_alignment_npz(str(alignment_path))
    input_ids = torch.from_numpy(np.asarray(alignment.input_ids, dtype=np.int64)).unsqueeze(0).to(device)
    attention_mask = torch.from_numpy(np.asarray(alignment.attention_mask, dtype=np.int64)).unsqueeze(0).to(device)
    token2note = [torch.from_numpy(np.asarray(alignment.token2note, dtype=np.float32)).to(device)]
    num_notes = [int(alignment.num_notes)]
    note_embeddings, _ = note_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token2note=token2note,
        num_notes=num_notes,
    )
    return note_embeddings[0, : num_notes[0]].detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate cached MusicBERT note embeddings.")
    parser.add_argument("--alignment_dir", type=str, default="artifacts/musicbert_alignments")
    parser.add_argument("--cache_dir", type=str, default="artifacts/musicbert_note_embeddings")
    parser.add_argument("--model_name", type=str, default="manoskary/musicbert-large")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_abs_tol", type=float, default=1e-2)
    parser.add_argument("--mean_abs_tol", type=float, default=1e-3)
    args = parser.parse_args()

    alignment_dir = Path(args.alignment_dir)
    cache_dir = Path(args.cache_dir)
    if not alignment_dir.exists():
        raise ValueError(f"Alignment dir not found: {alignment_dir}")
    if not cache_dir.exists():
        raise ValueError(f"Cache dir not found: {cache_dir}")

    shared = sorted({p.name for p in alignment_dir.glob("*.npz")} & {p.name for p in cache_dir.glob("*.npz")})
    if not shared:
        raise ValueError("No matching .npz files between alignment_dir and cache_dir.")

    random.seed(args.seed)
    if args.num_samples < len(shared):
        shared = random.sample(shared, args.num_samples)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device.index or 0)
        if major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

    note_encoder = MusicBertNoteEncoder(
        pretrained_name=args.model_name,
        adapter_cfg=MusicBertAdapterConfig(use_lora=False),
        freeze_backbone=True,
    ).to(device)
    note_encoder.eval()

    max_abs = 0.0
    mean_abs_values = []
    bad = []
    for name in shared:
        online = _encode_alignment(note_encoder, alignment_dir / name, device)
        with np.load(cache_dir / name) as cache_data:
            if "note_embeddings" not in cache_data:
                bad.append((name, "missing note_embeddings"))
                continue
            cached = np.asarray(cache_data["note_embeddings"], dtype=np.float32)
        online = np.asarray(online, dtype=np.float32)
        if cached.shape != online.shape:
            bad.append((name, f"shape mismatch {cached.shape} vs {online.shape}"))
            continue
        abs_err = np.abs(cached - online)
        mae = float(abs_err.mean()) if abs_err.size else 0.0
        mxe = float(abs_err.max()) if abs_err.size else 0.0
        mean_abs_values.append(mae)
        max_abs = max(max_abs, mxe)
        if mxe > args.max_abs_tol or mae > args.mean_abs_tol:
            bad.append((name, f"max_abs={mxe:.6f}, mean_abs={mae:.6f}"))

    avg_mae = float(np.mean(mean_abs_values)) if mean_abs_values else 0.0
    print(f"checked={len(shared)} bad={len(bad)} avg_mae={avg_mae:.6f} max_abs={max_abs:.6f}")
    if bad:
        print("mismatches:")
        for name, reason in bad[:20]:
            print(f"- {name}: {reason}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
