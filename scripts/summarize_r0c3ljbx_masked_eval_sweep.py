#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict

import yaml


EXPECTED_RATIOS = [0.05, 0.15, 0.25, 0.50, 0.75, 0.85, 0.95]
EXPECTED_SEEDS = [0, 1, 2, 3, 4]
CHECKPOINT_PATH = "AnalysisGNN-MusicBERT/r0c3ljbx/checkpoints/last.ckpt"


def _unwrap(conf: dict, key: str, default=None):
    value = conf.get(key, default)
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _format_num(value: float) -> str:
    text = f"{value:.3f}"
    if text.startswith("0"):
        return text[1:]
    if text.startswith("-0"):
        return "-" + text[2:]
    return text


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(max(var, 0.0))


def _masked_only(score: float, known_ratio: float) -> float:
    return (score - known_ratio) / (1.0 - known_ratio)


def load_runs() -> dict[float, dict[int, dict[str, float]]]:
    grouped: dict[float, dict[int, dict[str, float]]] = defaultdict(dict)
    for cfg_path in glob.glob("wandb/run-*/files/config.yaml"):
        run_dir = os.path.dirname(cfg_path)
        summary_path = os.path.join(run_dir, "wandb-summary.json")
        if not os.path.exists(summary_path):
            continue
        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                conf = yaml.safe_load(fh) or {}
        except Exception:
            continue
        if not _unwrap(conf, "do_eval", False):
            continue
        if _unwrap(conf, "do_train", False):
            continue
        if _unwrap(conf, "checkpoint_path") != CHECKPOINT_PATH:
            continue
        if not _unwrap(conf, "masked_prediction_train", False):
            continue

        ratio = float(_unwrap(conf, "known_ratio"))
        seed = int(_unwrap(conf, "seed", 0))
        try:
            with open(summary_path, "r", encoding="utf-8") as fh:
                summary = json.load(fh)
        except Exception:
            continue
        rna = summary.get("test/RN(Onset)_rna_accuracy")
        all_acc = summary.get("test/RN(Onset)_all_accuracy")
        if rna is None or all_acc is None:
            continue
        mtime = os.path.getmtime(summary_path)
        prev = grouped[ratio].get(seed)
        if prev is not None and prev.get("_mtime", -1.0) > mtime:
            continue
        grouped[ratio][seed] = {
            "_mtime": mtime,
            "rna_whole": float(rna),
            "all_whole": float(all_acc),
            "rna_masked": _masked_only(float(rna), ratio),
            "all_masked": _masked_only(float(all_acc), ratio),
        }
    return grouped


def emit(grouped: dict[float, dict[int, dict[str, float]]]) -> int:
    missing = []
    rows = []
    warnings = []
    for ratio in EXPECTED_RATIOS:
        ratio_runs = grouped.get(ratio, {})
        missing_seeds = [seed for seed in EXPECTED_SEEDS if seed not in ratio_runs]
        if missing_seeds:
            missing.append((ratio, missing_seeds))
            continue
        aug_vals = [ratio_runs[seed]["rna_masked"] for seed in EXPECTED_SEEDS]
        dlc_vals = [ratio_runs[seed]["all_masked"] for seed in EXPECTED_SEEDS]
        for val in aug_vals + dlc_vals:
            if val < 0 or val > 1:
                warnings.append((ratio, val))
        rows.append(
            (
                ratio,
                sum(aug_vals) / len(aug_vals),
                _sample_std(aug_vals),
                sum(dlc_vals) / len(dlc_vals),
                _sample_std(dlc_vals),
            )
        )

    if missing:
        print("Missing run cells:")
        for ratio, seeds in missing:
            print(f"  ratio={ratio:.2f} missing seeds={seeds}")
        return 1

    if warnings:
        print("Warnings: derived masked-only values outside [0, 1]:")
        for ratio, value in warnings:
            print(f"  ratio={ratio:.2f} value={value:.6f}")
        print()

    print("\\begin{table}[]")
    print("    \\centering")
    print("    \\begin{tabular}{lcc}")
    print("    \\toprule")
    print("    Known ratio & \\textbf{AugNet} & \\textbf{DLC} \\\\")
    print("    \\midrule")
    for ratio, aug_mean, aug_std, dlc_mean, dlc_std in rows:
        ratio_pct = int(round(ratio * 100))
        aug = f"{_format_num(aug_mean)} $\\\\pm$ {_format_num(aug_std)}"
        dlc = f"{_format_num(dlc_mean)} $\\\\pm$ {_format_num(dlc_std)}"
        print(f"    {ratio_pct}\\% & {aug} & {dlc} \\\\")
    print("    \\bottomrule")
    print("    \\end{tabular}")
    print(
        "    \\caption{Masked model performance on different mask ratios. "
        "The percentage denotes the amount of known labels. AugNet and DLC denote the RN accuracy "
        "on the sets of AugmentedNet and Distant Listening test sets accordingly. "
        "Accuracy on the masked is derived from whole-piece RN(Onset) accuracy as "
        "$ (\\mathrm{RN(Onset)} - \\mathrm{known\\ ratio}) / (1 - \\mathrm{known\\ ratio}) $.}"
    )
    print("    \\label{tab:placeholder}")
    print("\\end{table}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    grouped = load_runs()
    return emit(grouped)


if __name__ == "__main__":
    raise SystemExit(main())
