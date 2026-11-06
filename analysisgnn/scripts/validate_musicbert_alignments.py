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

from analysisgnn.data.musicbert_alignment import build_alignment_from_tsv
from analysisgnn.data.remi_bpe_aligner import load_alignment_npz
from analysisgnn.utils.dcl_tsv_utils import create_spec_file

INTERVALS = ["P1", "m2", "M2", "m3", "M3", "P4", "A4", "P5", "m6", "M6", "m7", "M7"]


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


def _parse_alignment_name(stem: str) -> Tuple[str, str]:
    for interval in sorted(INTERVALS, key=len, reverse=True):
        suffix = f"_{interval}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], interval
    return stem, "P1"


def _build_tsv_index(dataset: str, raw_dir: Optional[str], force_reload: bool, verbose: bool) -> Dict[str, str]:
    if dataset == "dlc":
        from analysisgnn.data.datasets.dlc import DLCDataset, make_dlc_nickname

        base_dataset = DLCDataset(raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)
        index = {}
        for tsv_path, collection in zip(base_dataset.scores, base_dataset.collections):
            base_name = os.path.splitext(os.path.basename(tsv_path))[0]
            nickname = make_dlc_nickname(collection, base_name)
            index[nickname] = tsv_path
        return index
    if dataset == "rna":
        from analysisgnn.data.datasets.chord import AugmentedNetv100Dataset

        base_dataset = AugmentedNetv100Dataset(raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)
        return {os.path.splitext(os.path.basename(path))[0]: path for path in base_dataset.scores}
    return {}


def _get_alignment_spec(
    dataset: str,
    raw_dir: Optional[str],
    force_reload: bool,
    verbose: bool,
) -> Tuple[dict, dict, Optional[List[str]]]:
    if dataset == "dlc":
        from analysisgnn.data.datasets.dlc import DLCDataset

        base_dataset = DLCDataset(raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)
        spec_file_path = os.path.join(base_dataset.raw_path, "processing", "DLC", "dlc_pitch_array_specs.csv")
        replace_dtypes = dict(object="string", int64="Int64")
        spec_file, converters = create_spec_file(spec_file_path, **replace_dtypes)
        return spec_file, converters, ["tpc"]
    if dataset == "rna":
        from analysisgnn.data.datasets.chord import AugmentedNetv100Dataset

        base_dataset = AugmentedNetv100Dataset(raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)
        spec_file_path = os.path.join(base_dataset.raw_path, "processing", "DLC", "dlc_pitch_array_specs.csv")
        spec_file, converters = create_spec_file(spec_file_path)
        return spec_file, converters, ["s_note"]
    raise ValueError(f"Unsupported dataset: {dataset}")


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


def _compare_edges(a: np.ndarray, b: np.ndarray, weight_tol: float) -> Dict[str, float]:
    if a.size == 0 and b.size == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "weight_mae": 0.0}
    if a.size == 0 or b.size == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "weight_mae": 1.0}

    a_keys = {(int(t), int(n)): float(w) for t, n, w in a}
    b_keys = {(int(t), int(n)): float(w) for t, n, w in b}
    inter = set(a_keys.keys()) & set(b_keys.keys())
    precision = len(inter) / len(a_keys) if a_keys else 0.0
    recall = len(inter) / len(b_keys) if b_keys else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    if inter:
        weight_mae = float(
            np.mean([abs(a_keys[key] - b_keys[key]) for key in inter])
        )
    else:
        weight_mae = 1.0

    if weight_mae < weight_tol:
        weight_mae = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "weight_mae": weight_mae,
    }


def _note_meta_arrays(note_meta: Dict[int, object]) -> Tuple[np.ndarray, np.ndarray]:
    if not note_meta:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    ids = sorted(note_meta.keys())
    pitches = np.asarray([note_meta[idx].pitch for idx in ids], dtype=int)
    onset_ticks = np.asarray([note_meta[idx].onset_tick for idx in ids], dtype=int)
    return pitches, onset_ticks


def _deep_validate(
    paths: List[Path],
    dataset: str,
    raw_dir: Optional[str],
    tokenizer_name: str,
    include_transpositions: bool,
    force_reload: bool,
    verbose: bool,
    max_files: Optional[int],
    seed: int,
    weight_tol: float,
) -> None:
    from miditok import MusicTokenizer

    if dataset == "none":
        print("deep validation skipped: --dataset none")
        return

    rng = np.random.default_rng(seed)
    sample_paths = paths
    if max_files is not None and len(sample_paths) > max_files:
        sample_paths = list(rng.choice(sample_paths, size=max_files, replace=False))

    tsv_index = _build_tsv_index(dataset, raw_dir, force_reload, verbose)
    spec_file, converters, drop_na_subset = _get_alignment_spec(dataset, raw_dir, force_reload, verbose)
    tokenizer = MusicTokenizer.from_pretrained(tokenizer_name)

    mismatched_tokens = []
    mismatched_edges = []
    note_meta_mismatch = []
    missing_tsv = []
    failures = []

    for path in sample_paths:
        stem = path.stem
        base_name, interval = _parse_alignment_name(stem)
        tsv_path = tsv_index.get(base_name)
        if tsv_path is None:
            missing_tsv.append(stem)
            continue
        if interval != "P1" and not include_transpositions:
            continue

        try:
            stored = load_alignment_npz(str(path))
        except Exception as exc:
            failures.append((stem, f"load_failed: {exc}"))
            continue

        try:
            rebuilt = build_alignment_from_tsv(
                tsv_path=tsv_path,
                tokenizer=tokenizer,
                spec_file=spec_file,
                converters=converters,
                drop_na_subset=drop_na_subset,
                interval=interval,
            )
        except Exception as exc:
            failures.append((stem, f"rebuild_failed: {exc}"))
            continue

        stored_ids = np.asarray(stored.input_ids)
        rebuilt_ids = np.asarray(rebuilt.input_ids)
        if stored_ids.shape != rebuilt_ids.shape or not np.array_equal(stored_ids, rebuilt_ids):
            mismatch = abs(len(stored_ids) - len(rebuilt_ids))
            min_len = min(len(stored_ids), len(rebuilt_ids))
            overlap = float(np.mean(stored_ids[:min_len] == rebuilt_ids[:min_len])) if min_len else 0.0
            mismatched_tokens.append((stem, mismatch, overlap))

        edge_cmp = _compare_edges(
            np.asarray(stored.token2note),
            np.asarray(rebuilt.token2note),
            weight_tol=weight_tol,
        )
        if edge_cmp["f1"] < 0.98 or edge_cmp["weight_mae"] > weight_tol:
            mismatched_edges.append((stem, edge_cmp))

        stored_pitch, stored_onset = _note_meta_arrays(stored.note_meta)
        rebuilt_pitch, rebuilt_onset = _note_meta_arrays(rebuilt.note_meta)
        if stored_pitch.shape != rebuilt_pitch.shape:
            note_meta_mismatch.append((stem, "count"))
        else:
            pitch_match = float(np.mean(stored_pitch == rebuilt_pitch)) if stored_pitch.size else 1.0
            onset_match = float(np.mean(stored_onset == rebuilt_onset)) if stored_onset.size else 1.0
            if pitch_match < 0.98 or onset_match < 0.98:
                note_meta_mismatch.append((stem, f"pitch={pitch_match:.3f}, onset={onset_match:.3f}"))

    print(f"deep checked: {len(sample_paths)}")
    if missing_tsv:
        print(f"deep missing tsv: {len(missing_tsv)}")
        for stem in missing_tsv[:10]:
            print(f"- {stem}")
    if failures:
        print(f"deep failures: {len(failures)}")
        for stem, err in failures[:10]:
            print(f"- {stem}: {err}")
    if mismatched_tokens:
        print(f"deep token mismatches: {len(mismatched_tokens)}")
        for stem, length_diff, overlap in mismatched_tokens[:10]:
            print(f"- {stem}: len_diff={length_diff}, prefix_match={overlap:.3f}")
    if mismatched_edges:
        print(f"deep edge mismatches: {len(mismatched_edges)}")
        for stem, stats in mismatched_edges[:10]:
            print(f"- {stem}: f1={stats['f1']:.3f}, weight_mae={stats['weight_mae']:.3f}")
    if note_meta_mismatch:
        print(f"deep note_meta mismatches: {len(note_meta_mismatch)}")
        for stem, detail in note_meta_mismatch[:10]:
            print(f"- {stem}: {detail}")


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
    parser.add_argument("--deep", action="store_true", help="Rebuild a subset of alignments for consistency checks.")
    parser.add_argument(
        "--deep_max_files",
        type=int,
        default=25,
        help="Max alignments to rebuild in deep mode.",
    )
    parser.add_argument("--deep_seed", type=int, default=0, help="Seed for deep sampling.")
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="manoskary/miditok-REMI",
        help="Tokenizer name for deep validation.",
    )
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

    if args.deep:
        _deep_validate(
            paths=paths,
            dataset=args.dataset,
            raw_dir=args.raw_dir,
            tokenizer_name=args.tokenizer_name,
            include_transpositions=args.include_transpositions,
            force_reload=args.force_reload,
            verbose=args.verbose,
            max_files=args.deep_max_files,
            seed=args.deep_seed,
            weight_tol=args.weight_tol,
        )


if __name__ == "__main__":
    main()
