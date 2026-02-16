#!/usr/bin/env python3
"""
Precompute note-level MusicBERT embeddings from alignment .npz files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from analysisgnn.data.remi_bpe_aligner import load_alignment_npz
from analysisgnn.models.musicbert_backbone import MusicBertAdapterConfig
from analysisgnn.models.musicbert_note_encoder import MusicBertNoteEncoder


def _get_tqdm():
    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover
        return lambda x, **kwargs: x
    return tqdm


def _resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _configure_ampere_fastmath(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    major, _ = torch.cuda.get_device_capability(device.index or 0)
    if major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


@torch.inference_mode()
def _encode_file(note_encoder: MusicBertNoteEncoder, alignment_path: Path, device: torch.device) -> np.ndarray:
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
    parser = argparse.ArgumentParser(description="Precompute note-level MusicBERT embeddings.")
    parser.add_argument(
        "--alignment_dir",
        type=str,
        default="artifacts/musicbert_alignments",
        help="Directory containing alignment .npz files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/musicbert_note_embeddings",
        help="Output directory for cached note embedding .npz files.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="manoskary/musicbert-large",
        help="MusicBERT model name or local path.",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device (e.g. cuda:0, cpu).")
    parser.add_argument(
        "--output_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Stored dtype for cached embeddings.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing cached embeddings.")
    parser.add_argument("--max_files", type=int, default=None, help="Limit number of files for quick tests.")
    args = parser.parse_args()

    alignment_dir = Path(args.alignment_dir)
    output_dir = Path(args.output_dir)
    if not alignment_dir.exists():
        raise ValueError(f"Alignment directory not found: {alignment_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    _configure_ampere_fastmath(device)

    adapter_cfg = MusicBertAdapterConfig(use_lora=False)
    note_encoder = MusicBertNoteEncoder(
        pretrained_name=args.model_name,
        adapter_cfg=adapter_cfg,
        freeze_backbone=True,
    ).to(device)
    note_encoder.eval()

    dtype = np.float16 if args.output_dtype == "float16" else np.float32
    files = sorted(alignment_dir.glob("*.npz"))
    if args.max_files is not None:
        files = files[: args.max_files]

    tqdm = _get_tqdm()
    built = 0
    skipped = 0
    failed = 0
    for alignment_path in tqdm(files, desc="Precomputing MusicBERT note embeddings"):
        out_path = output_dir / alignment_path.name
        if out_path.exists() and not args.force:
            skipped += 1
            continue
        try:
            note_embeddings = _encode_file(note_encoder, alignment_path, device)
            np.savez_compressed(
                out_path,
                note_embeddings=note_embeddings.astype(dtype, copy=False),
                num_notes=np.asarray(note_embeddings.shape[0], dtype=np.int64),
                model_name=np.asarray(args.model_name),
            )
            built += 1
        except Exception as exc:  # pragma: no cover
            failed += 1
            print(f"[cache] Failed: {alignment_path.name} -> {exc}")

    print(f"done | built={built} skipped={skipped} failed={failed} output_dir={output_dir}")


if __name__ == "__main__":
    main()
