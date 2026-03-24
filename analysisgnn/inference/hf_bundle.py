"""Hugging Face bundle resolution utilities for hybrid AnalysisGNN inference."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional


class HybridBundleResolutionError(RuntimeError):
    """Raised when an HF/local hybrid bundle cannot be resolved."""


@dataclass(frozen=True)
class HybridBundlePaths:
    """Concrete local paths for all hybrid inference artifacts."""

    bundle_root: str
    full_ckpt: str
    masked_ckpt: str
    voter_ckpt: str
    beat_voter_ckpt: str
    manifest: Dict[str, object]


def _is_local_bundle_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "bundle.json"))


def _load_manifest(bundle_root: str) -> Dict[str, object]:
    manifest_path = os.path.join(bundle_root, "bundle.json")
    if not os.path.isfile(manifest_path):
        raise HybridBundleResolutionError(f"bundle.json not found under '{bundle_root}'")
    with open(manifest_path, "r", encoding="utf-8") as f:
        try:
            manifest = json.load(f)
        except json.JSONDecodeError as exc:
            raise HybridBundleResolutionError(f"Invalid bundle.json at '{manifest_path}': {exc}") from exc
    if not isinstance(manifest, dict):
        raise HybridBundleResolutionError(f"bundle.json at '{manifest_path}' must contain a JSON object")
    return manifest


def _artifact_relpath(manifest: Dict[str, object], default_key: str) -> str:
    defaults = manifest.get("defaults", {})
    if not isinstance(defaults, dict):
        raise HybridBundleResolutionError("Manifest field 'defaults' must be an object")
    relpath = defaults.get(default_key, "")
    if not isinstance(relpath, str) or not relpath.strip():
        raise HybridBundleResolutionError(f"Manifest missing defaults['{default_key}']")
    return relpath


def _resolve_local_path(bundle_root: str, relpath: str, default_key: str) -> str:
    abs_path = os.path.join(bundle_root, relpath)
    if not os.path.isfile(abs_path):
        raise HybridBundleResolutionError(
            f"Manifest defaults['{default_key}'] points to missing file: '{relpath}'"
        )
    return abs_path


def _download_bundle_snapshot(
    repo_id: str,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover
        raise HybridBundleResolutionError(
            "huggingface_hub is required to resolve HF bundles; install huggingface_hub."
        ) from exc
    try:
        return snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            cache_dir=cache_dir,
            token=token,
        )
    except Exception as exc:
        raise HybridBundleResolutionError(
            f"Failed to download HF bundle '{repo_id}' (revision={revision!r}): {exc}"
        ) from exc


def resolve_hybrid_bundle(
    source: str,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> HybridBundlePaths:
    """Resolve a local or HF-hosted hybrid bundle to local artifact paths.

    Parameters
    ----------
    source:
        Either a local directory containing ``bundle.json`` or an HF model id.
    revision:
        Optional HF revision/branch/tag (ignored for local bundles).
    cache_dir:
        Optional HF cache directory for downloaded snapshots.
    token:
        Optional HF token. If omitted, huggingface_hub default auth resolution is used.
    """
    src = (source or "").strip()
    if not src:
        raise HybridBundleResolutionError("Bundle source is empty.")

    if _is_local_bundle_dir(src):
        bundle_root = os.path.abspath(src)
    elif os.path.exists(src):
        raise HybridBundleResolutionError(
            f"Local path exists but is not a valid bundle directory (missing bundle.json): '{src}'"
        )
    else:
        bundle_root = _download_bundle_snapshot(
            repo_id=src,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
        )

    manifest = _load_manifest(bundle_root)
    full_rel = _artifact_relpath(manifest, "full_ckpt")
    masked_rel = _artifact_relpath(manifest, "masked_ckpt")
    voter_rel = _artifact_relpath(manifest, "voter_ckpt")
    beat_voter_rel = _artifact_relpath(manifest, "beat_voter_ckpt")

    return HybridBundlePaths(
        bundle_root=bundle_root,
        full_ckpt=_resolve_local_path(bundle_root, full_rel, "full_ckpt"),
        masked_ckpt=_resolve_local_path(bundle_root, masked_rel, "masked_ckpt"),
        voter_ckpt=_resolve_local_path(bundle_root, voter_rel, "voter_ckpt"),
        beat_voter_ckpt=_resolve_local_path(bundle_root, beat_voter_rel, "beat_voter_ckpt"),
        manifest=manifest,
    )
