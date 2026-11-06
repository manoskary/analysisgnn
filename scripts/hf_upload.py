#!/usr/bin/env python3
"""Package and upload AnalysisGNN hybrid inference artifacts to Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FULL_CKPT = REPO_ROOT / "artifacts" / "gradio_checkpoints" / "uocj8f6y_full_last.ckpt"
DEFAULT_MASKED_CKPT = REPO_ROOT / "artifacts" / "gradio_checkpoints" / "t7pxcwri_masked_last.ckpt"
DEFAULT_VOTER_CKPT = REPO_ROOT / "artifacts" / "posthoc_voter" / "uocj8f6y_voter.pt"
DEFAULT_BEAT_VOTER_CKPT = DEFAULT_VOTER_CKPT


def _parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"{label} is not readable: {path}")


def _read_analysisgnn_version(repo_root: Path) -> str:
    init_path = repo_root / "analysisgnn" / "__init__.py"
    if not init_path.exists():
        return "unknown"
    text = init_path.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    return match.group(1) if match else "unknown"


def _load_hparams(ckpt_path: Path) -> Dict[str, object]:
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})
    return hparams if isinstance(hparams, dict) else {}


def _build_external_dependencies(full_hp: Dict[str, object], masked_hp: Dict[str, object]) -> Dict[str, object]:
    deps: Dict[str, object] = {}
    candidates = [full_hp, masked_hp]
    for key in ("musicbert_model_name", "musicbert_tokenizer_name", "use_musicbert"):
        for hp in candidates:
            if key in hp and hp[key] is not None:
                deps[key] = hp[key]
                break
    if "musicbert_model_name" not in deps:
        deps["musicbert_model_name"] = "manoskary/musicbert-large"
    if "musicbert_tokenizer_name" not in deps:
        deps["musicbert_tokenizer_name"] = "manoskary/miditok-REMI"
    return deps


def _copy_artifact(src: Path, dst: Path, staging_root: Path) -> Dict[str, object]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "relative_path": str(dst.relative_to(staging_root)),
        "sha256": _sha256(dst),
        "size_bytes": int(dst.stat().st_size),
    }


def _stage_bundle(
    *,
    staging_root: Path,
    full_ckpt: Path,
    masked_ckpt: Path,
    voter_ckpt: Path,
    beat_voter_ckpt: Path,
) -> Dict[str, object]:
    full_dst = staging_root / "checkpoints" / "full.ckpt"
    masked_dst = staging_root / "checkpoints" / "masked.ckpt"
    voter_dst = staging_root / "voters" / "note_voter.pt"
    beat_voter_dst = staging_root / "voters" / "beat_voter.pt"

    artifacts = {
        "full_ckpt": _copy_artifact(full_ckpt, full_dst, staging_root),
        "masked_ckpt": _copy_artifact(masked_ckpt, masked_dst, staging_root),
        "voter_ckpt": _copy_artifact(voter_ckpt, voter_dst, staging_root),
        "beat_voter_ckpt": _copy_artifact(beat_voter_ckpt, beat_voter_dst, staging_root),
    }
    return artifacts


def _relative_path(key: str, artifacts: Dict[str, Dict[str, object]]) -> str:
    rel = artifacts[key].get("relative_path")
    if not isinstance(rel, str):
        raise ValueError(f"Invalid artifact relative_path for '{key}'")
    return rel


def _write_manifest_and_readme(
    *,
    staging_root: Path,
    repo_id: str,
    artifacts: Dict[str, Dict[str, object]],
    external_dependencies: Dict[str, object],
    analysisgnn_version: str,
) -> Dict[str, object]:
    manifest: Dict[str, object] = {
        "format_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
        "defaults": {
            "full_ckpt": _relative_path("full_ckpt", artifacts),
            "masked_ckpt": _relative_path("masked_ckpt", artifacts),
            "voter_ckpt": _relative_path("voter_ckpt", artifacts),
            "beat_voter_ckpt": _relative_path("beat_voter_ckpt", artifacts),
        },
        "external_dependencies": external_dependencies,
        "compatibility": {
            "analysisgnn_version": analysisgnn_version,
            "runtime_notes": [
                "This bundle excludes MusicBERT backbone/tokenizer weights.",
                "Runtime may download MusicBERT/tokenizer from Hugging Face if not available locally.",
            ],
        },
    }
    with (staging_root / "bundle.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    readme = f"""# AnalysisGNN Hybrid Bundle

This private bundle stores hybrid inference artifacts for AnalysisGNN:

- full-piece checkpoint
- masked checkpoint
- note-level posthoc voter
- beat-level voter

## Repo
`{repo_id}`

## Runtime
Use with the Gradio app by setting:

```bash
export ANALYSISGNN_HF_REPO="{repo_id}"
# optional
export ANALYSISGNN_HF_REVISION="main"
```

Local artifact paths still take priority if they exist.
"""
    (staging_root / "README.md").write_text(readme, encoding="utf-8")
    return manifest


def _upload_to_hf(
    *,
    staging_root: Path,
    repo_id: str,
    private: bool,
    token: Optional[str],
    revision: Optional[str],
    commit_message: str,
) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(staging_root),
        path_in_repo=".",
        revision=revision,
        commit_message=commit_message,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Package and upload AnalysisGNN hybrid artifacts to HF.")
    parser.add_argument("--repo_id", type=str, required=True, help="HF model repo id, e.g. manoskary/analysisgnn-hybrid")
    parser.add_argument("--private", type=_parse_bool, default=True, help="Create/use private HF repo (default: true)")
    parser.add_argument("--full_ckpt", type=str, default=str(DEFAULT_FULL_CKPT))
    parser.add_argument("--masked_ckpt", type=str, default=str(DEFAULT_MASKED_CKPT))
    parser.add_argument("--voter_ckpt", type=str, default=str(DEFAULT_VOTER_CKPT))
    parser.add_argument("--beat_voter_ckpt", type=str, default=str(DEFAULT_BEAT_VOTER_CKPT))
    parser.add_argument("--revision", type=str, default=None, help="Optional HF revision/branch")
    parser.add_argument("--token", type=str, default=None, help="Optional HF token")
    parser.add_argument("--commit_message", type=str, default="Upload AnalysisGNN hybrid bundle")
    parser.add_argument("--dry_run", action="store_true", help="Stage and validate only; do not upload")
    args = parser.parse_args()

    full_ckpt = Path(args.full_ckpt).expanduser().resolve()
    masked_ckpt = Path(args.masked_ckpt).expanduser().resolve()
    voter_ckpt = Path(args.voter_ckpt).expanduser().resolve()
    beat_voter_ckpt = Path(args.beat_voter_ckpt).expanduser().resolve()

    _ensure_file(full_ckpt, "full checkpoint")
    _ensure_file(masked_ckpt, "masked checkpoint")
    _ensure_file(voter_ckpt, "voter checkpoint")
    _ensure_file(beat_voter_ckpt, "beat voter checkpoint")

    full_hparams = _load_hparams(full_ckpt)
    masked_hparams = _load_hparams(masked_ckpt)
    external_dependencies = _build_external_dependencies(full_hparams, masked_hparams)
    analysisgnn_version = _read_analysisgnn_version(REPO_ROOT)

    with tempfile.TemporaryDirectory(prefix="analysisgnn_hf_bundle_") as tmp_dir:
        staging_root = Path(tmp_dir).resolve()
        artifacts = _stage_bundle(
            staging_root=staging_root,
            full_ckpt=full_ckpt,
            masked_ckpt=masked_ckpt,
            voter_ckpt=voter_ckpt,
            beat_voter_ckpt=beat_voter_ckpt,
        )
        manifest = _write_manifest_and_readme(
            staging_root=staging_root,
            repo_id=args.repo_id,
            artifacts=artifacts,
            external_dependencies=external_dependencies,
            analysisgnn_version=analysisgnn_version,
        )

        print("Staged bundle at:", staging_root)
        print("Manifest defaults:", json.dumps(manifest.get("defaults", {}), indent=2))
        if args.dry_run:
            print("Dry run enabled: upload skipped.")
        else:
            _upload_to_hf(
                staging_root=staging_root,
                repo_id=args.repo_id,
                private=bool(args.private),
                token=args.token,
                revision=args.revision,
                commit_message=args.commit_message,
            )
            print(f"Uploaded bundle to HF model repo: {args.repo_id}")

    print("\nApp usage:")
    print(f"export ANALYSISGNN_HF_REPO=\"{args.repo_id}\"")
    if args.revision:
        print(f"export ANALYSISGNN_HF_REVISION=\"{args.revision}\"")
    else:
        print("# export ANALYSISGNN_HF_REVISION=\"main\"  # optional")
    print("# Local paths still take priority when present.")


if __name__ == "__main__":
    main()
