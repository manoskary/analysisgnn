#!/usr/bin/env python3
"""
Build MusicBERT token-to-note alignment files for DilemmaData-derived datasets.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Tuple

from analysisgnn.data.musicbert_alignment import build_alignment_from_tsv, save_alignment_npz
from analysisgnn.utils.dcl_tsv_utils import create_spec_file


def _get_tqdm():
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - optional dependency
        return lambda x, **kwargs: x
    return tqdm


def iter_dlc_items(raw_dir: str | None, force_reload: bool, verbose: bool):
    from analysisgnn.data.datasets.dlc import DLCDataset, make_dlc_nickname

    base_dataset = DLCDataset(raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)
    spec_file_path = os.path.join(base_dataset.raw_path, "processing", "DLC", "dlc_pitch_array_specs.csv")
    replace_dtypes = dict(object="string", int64="Int64")
    spec_file, converters = create_spec_file(spec_file_path, **replace_dtypes)

    for tsv_path, collection in zip(base_dataset.scores, base_dataset.collections):
        base_name = os.path.splitext(os.path.basename(tsv_path))[0]
        nickname = make_dlc_nickname(collection, base_name)
        yield tsv_path, nickname, spec_file, converters, ["tpc"]


def iter_rna_items(raw_dir: str | None, force_reload: bool, verbose: bool):
    from analysisgnn.data.datasets.chord import AugmentedNetv100Dataset

    base_dataset = AugmentedNetv100Dataset(raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)
    spec_file_path = os.path.join(base_dataset.raw_path, "processing", "DLC", "dlc_pitch_array_specs.csv")
    spec_file, converters = create_spec_file(spec_file_path)

    for tsv_path in base_dataset.scores:
        base_name = os.path.splitext(os.path.basename(tsv_path))[0]
        yield tsv_path, base_name, spec_file, converters, ["s_note"]


def build_alignments(
    items: Iterable[Tuple[str, str, dict, dict, List[str]]],
    tokenizer_name: str,
    alignment_dir: Path,
    include_transpositions: bool,
    force: bool,
    max_files: int | None,
):
    from miditok import MusicTokenizer

    tokenizer = MusicTokenizer.from_pretrained(tokenizer_name)
    alignment_dir.mkdir(parents=True, exist_ok=True)
    intervals = ["P1"]
    if include_transpositions:
        intervals = ["P1", "m2", "M2", "m3", "M3", "P4", "A4", "P5", "m6", "M6", "m7", "M7"]

    tqdm = _get_tqdm()
    count = 0
    for tsv_path, name, spec_file, converters, drop_na_subset in tqdm(items):
        for interval in intervals:
            out_name = name if interval == "P1" else f"{name}_{interval}"
            out_path = alignment_dir / f"{out_name}.npz"
            if out_path.exists() and not force:
                continue

            try:
                alignment = build_alignment_from_tsv(
                    tsv_path=tsv_path,
                    tokenizer=tokenizer,
                    spec_file=spec_file,
                    converters=converters,
                    drop_na_subset=drop_na_subset,
                    interval=interval,
                )
                save_alignment_npz(alignment, str(out_path))
            except Exception as exc:
                print(f"[align] Failed: {tsv_path} ({interval}) -> {exc}")
                continue

        count += 1
        if max_files is not None and count >= max_files:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MusicBERT alignments for DilemmaData.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dlc",
        choices=["dlc", "rna", "all"],
        help="Dataset to process (dlc, rna, or all).",
    )
    parser.add_argument("--raw_dir", type=str, default=None, help="Override dataset raw directory.")
    parser.add_argument(
        "--alignment_dir",
        type=str,
        default="artifacts/musicbert_alignments",
        help="Output directory for alignment .npz files.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="manoskary/miditok-REMI",
        help="Miditok tokenizer name.",
    )
    parser.add_argument("--include_transpositions", action="store_true", help="Build alignments per transposition.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing .npz files.")
    parser.add_argument("--max_files", type=int, default=None, help="Limit the number of processed files.")
    parser.add_argument("--force_reload", action="store_true", help="Force dataset reload.")
    parser.add_argument("--verbose", action="store_true", help="Verbose dataset loading.")
    args = parser.parse_args()

    alignment_dir = Path(args.alignment_dir)
    items = []
    if args.dataset in {"dlc", "all"}:
        items.extend(list(iter_dlc_items(args.raw_dir, args.force_reload, args.verbose)))
    if args.dataset in {"rna", "all"}:
        items.extend(list(iter_rna_items(args.raw_dir, args.force_reload, args.verbose)))

    build_alignments(
        items=items,
        tokenizer_name=args.tokenizer_name,
        alignment_dir=alignment_dir,
        include_transpositions=args.include_transpositions,
        force=args.force,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
