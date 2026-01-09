#!/usr/bin/env python3
"""
Validate MusicBERT token-to-note alignment files.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from analysisgnn.data.remi_bpe_aligner import load_alignment_npz


def _get_tqdm():
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - optional dependency
        return lambda x, **kwargs: x
    return tqdm


def _load_dataset(dataset: str, raw_dir: Optional[str], include_transpositions: bool, force_reload: bool, verbose: bool):
    if dataset == "dlc":
        from analysisgnn.data.datasets.dlc import DLCGraphDataset

        return DLCGraphDataset(
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transpose=include_transpositions,
        )
    if dataset == "rna":
        from analysisgnn.data.datasets.chord import RNAGraphDataset

        return RNAGraphDataset(
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transpose=include_transpositions,
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def _expected_alignment_name(graph) -> str:
    name = getattr(graph, "name", None)
    if name is None:
        try:
            name = graph["name"]
        except Exception:
            name = None
    if name is None:
        return ""

    interval = getattr(graph, "transposition", None)
    if interval is None:
        interval = getattr(graph, "interval", None)
    if interval is None:
        try:
            interval = graph["transposition"]
        except Exception:
            interval = None
    if interval is None:
        try:
            interval = graph["interval"]
        except Exception:
            interval = None

    if interval and interval != "P1":
        return f"{name}_{interval}"
    return str(name)


def _evaluate_alignment(path: Path, weight_tol: float) -> Dict[str, object]:
    try:
        alignment = load_alignment_npz(str(path))
    except Exception as exc:
        return {"path": str(path), "error": f"load_failed: {exc}"}

    input_ids = np.asarray(alignment.input_ids)
    token2note = np.asarray(alignment.token2note)
    num_tokens = int(input_ids.shape[0])
    num_notes = int(alignment.num_notes)

    if num_tokens == 0 or num_notes == 0:
        return {"path": str(path), "error": "empty_sequence"}
    if token2note.ndim != 2 or token2note.shape[1] != 3:
        return {"path": str(path), "error": "bad_token2note_shape"}

    token_idx = token2note[:, 0].astype(int)
    note_idx = token2note[:, 1].astype(int)
    weights = token2note[:, 2].astype(float)

    invalid_token = (token_idx < 0) | (token_idx >= num_tokens)
    invalid_note = (note_idx < 0) | (note_idx >= num_notes)
    valid_mask = ~(invalid_token | invalid_note)

    token_idx = token_idx[valid_mask]
    note_idx = note_idx[valid_mask]
    weights = weights[valid_mask]

    note_cover = np.zeros(num_notes, dtype=bool)
    if note_idx.size:
        note_cover[note_idx] = True
    note_coverage = float(note_cover.mean()) if num_notes else 0.0

    token_sum = np.zeros(num_tokens, dtype=float)
    for t, w in zip(token_idx, weights):
        token_sum[t] += w
    token_coverage = float((token_sum > 0).mean()) if num_tokens else 0.0

    nonzero = token_sum[token_sum > 0]
    if nonzero.size:
        weight_ok = np.isclose(nonzero, 1.0, atol=weight_tol)
        weight_ok_ratio = float(weight_ok.mean())
    else:
        weight_ok_ratio = 0.0

    return {
        "path": str(path),
        "num_tokens": num_tokens,
        "num_notes": num_notes,
        "note_coverage": note_coverage,
        "token_coverage": token_coverage,
        "weight_ok_ratio": weight_ok_ratio,
        "invalid_edges": int((~valid_mask).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MusicBERT alignment files.")
    parser.add_argument(
        "--alignment_dir",
        type=str,
        default="artifacts/musicbert_alignments",
        help="Directory containing alignment .npz files.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dlc",
        choices=["dlc", "rna", "none"],
        help="Dataset to cross-check note counts (or 'none').",
    )
    parser.add_argument("--raw_dir", type=str, default=None, help="Override dataset raw directory.")
    parser.add_argument("--include_transpositions", action="store_true", help="Expect transposed alignments.")
    parser.add_argument("--force_reload", action="store_true", help="Force dataset reload.")
    parser.add_argument("--verbose", action="store_true", help="Verbose dataset loading.")
    parser.add_argument("--max_files", type=int, default=None, help="Limit number of alignment files to scan.")
    parser.add_argument("--weight_tol", type=float, default=1e-2, help="Tolerance for token weight sums.")
    parser.add_argument("--min_note_coverage", type=float, default=0.8, help="Flag files below this coverage.")
    args = parser.parse_args()

    alignment_dir = Path(args.alignment_dir)
    if not alignment_dir.exists():
        raise SystemExit(f"Alignment dir not found: {alignment_dir}")

    all_paths = sorted(alignment_dir.glob("*.npz"))
    paths = all_paths
    if args.max_files is not None:
        paths = paths[: args.max_files]

    tqdm = _get_tqdm()
    stats = []
    errors = []
    for path in tqdm(paths, desc="Validating"):
        info = _evaluate_alignment(path, args.weight_tol)
        if "error" in info:
            errors.append(info)
        else:
            stats.append(info)

    if not stats:
        print("No valid alignments found.")
        if errors:
            print("Errors:")
            for err in errors[:10]:
                print(f"- {err['path']}: {err['error']}")
        return

    note_cov = [s["note_coverage"] for s in stats]
    token_cov = [s["token_coverage"] for s in stats]
    weight_ok = [s["weight_ok_ratio"] for s in stats]

    print(f"files scanned: {len(stats)} (errors: {len(errors)})")
    print(f"avg note coverage: {sum(note_cov) / len(note_cov):.3f}")
    print(f"avg token coverage: {sum(token_cov) / len(token_cov):.3f}")
    print(f"avg weight-ok ratio: {sum(weight_ok) / len(weight_ok):.3f}")

    low_cov = [s for s in stats if s["note_coverage"] < args.min_note_coverage]
    if low_cov:
        low_cov = sorted(low_cov, key=lambda x: x["note_coverage"])
        print(f"low note coverage (< {args.min_note_coverage}): {len(low_cov)}")
        for s in low_cov[:10]:
            print(f"- {os.path.basename(s['path'])}: {s['note_coverage']:.3f}")

    bad_weight = [s for s in stats if s["weight_ok_ratio"] < 0.95]
    if bad_weight:
        bad_weight = sorted(bad_weight, key=lambda x: x["weight_ok_ratio"])
        print(f"weight sum failures (< 0.95): {len(bad_weight)}")
        for s in bad_weight[:10]:
            print(f"- {os.path.basename(s['path'])}: {s['weight_ok_ratio']:.3f}")

    if errors:
        print("alignment load errors:")
        for err in errors[:10]:
            print(f"- {os.path.basename(err['path'])}: {err['error']}")

    if args.dataset != "none":
        dataset = _load_dataset(
            dataset=args.dataset,
            raw_dir=args.raw_dir,
            include_transpositions=args.include_transpositions,
            force_reload=args.force_reload,
            verbose=args.verbose,
        )
        alignment_map = {p.stem: p for p in all_paths}
        missing = []
        mismatched = []
        for graph in dataset:
            expected = _expected_alignment_name(graph)
            if not expected:
                continue
            path = alignment_map.get(expected)
            if path is None:
                missing.append(expected)
                continue
            try:
                alignment = load_alignment_npz(str(path))
            except Exception:
                mismatched.append((expected, "load_failed"))
                continue
            graph_notes = int(graph["note"].x.shape[0])
            if alignment.num_notes != graph_notes:
                mismatched.append((expected, f"{alignment.num_notes} != {graph_notes}"))

        if missing:
            print(f"missing alignments for dataset graphs: {len(missing)}")
            for name in missing[:10]:
                print(f"- {name}")
        if mismatched:
            print(f"note count mismatches: {len(mismatched)}")
            for name, detail in mismatched[:10]:
                print(f"- {name}: {detail}")


if __name__ == "__main__":
    main()
