"""Hybrid inference routing for full vs masked-conditioned prediction.

This module adds a non-invasive, additive inference layer:
- base model checkpoint for full-piece prediction
- masked model checkpoint for partial re-prediction with known labels
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

import partitura as pt

from analysisgnn.models.analysis import ContinualAnalysisGNN
from analysisgnn.utils.chord_representations import available_representations


DEFAULT_EDITABLE_TASKS: Tuple[str, ...] = (
    "romanNumeral",
    "localkey",
    "quality",
    "inversion",
    "degree1",
    "degree2",
)


@dataclass
class RoutingDecision:
    """Routing metadata for hybrid inference."""

    route: str
    checkpoint_path: str



def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a user-facing device string to a torch device."""
    value = (device or "auto").strip().lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda":
        if not torch.cuda.is_available():
            return torch.device("cpu")
        try:
            _ = torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception:
            return torch.device("cpu")
    return torch.device(value)



def parse_index_expression(expr: str, max_len: int) -> List[int]:
    """Parse 1-based index expression like ``1-3, 8, 10`` into 0-based indices."""
    if expr is None:
        return []
    text = str(expr).strip()
    if not text:
        return []

    result: List[int] = []
    for chunk in text.split(","):
        part = chunk.strip()
        if not part:
            continue
        if "-" in part:
            bounds = part.split("-", 1)
            if len(bounds) != 2:
                continue
            try:
                start = int(bounds[0].strip())
                end = int(bounds[1].strip())
            except ValueError:
                continue
            if end < start:
                start, end = end, start
            for idx_1b in range(start, end + 1):
                idx_0b = idx_1b - 1
                if 0 <= idx_0b < max_len:
                    result.append(idx_0b)
            continue

        try:
            idx_1b = int(part)
        except ValueError:
            continue
        idx_0b = idx_1b - 1
        if 0 <= idx_0b < max_len:
            result.append(idx_0b)

    return sorted(set(result))



def parse_task_csv(tasks_csv: str, fallback: Sequence[str] = DEFAULT_EDITABLE_TASKS) -> List[str]:
    """Parse a comma-separated task string."""
    if not tasks_csv or not str(tasks_csv).strip():
        return list(fallback)
    tasks = [x.strip() for x in str(tasks_csv).split(",") if x.strip()]
    dedup: List[str] = []
    for task in tasks:
        if task not in dedup:
            dedup.append(task)
    return dedup



def build_label_overrides_from_dataframe(
    df: pd.DataFrame,
    masked_tasks: Sequence[str],
    known_indices: Sequence[int],
) -> Dict[str, Dict[str, List[Any]]]:
    """Build ``user_edits.label_overrides`` from an editable dataframe."""
    overrides: Dict[str, Dict[str, List[Any]]] = {}
    if df is None or len(df) == 0 or not known_indices:
        return overrides

    known_set = set(int(i) for i in known_indices)
    for task in masked_tasks:
        if task not in df.columns:
            continue
        indices: List[int] = []
        labels: List[Any] = []
        for idx in sorted(known_set):
            if idx < 0 or idx >= len(df.index):
                continue
            value = df.iloc[idx][task]
            if pd.isna(value):
                continue
            label = value
            if isinstance(label, str):
                label = label.strip()
                if not label:
                    continue
                if label.lstrip("-").isdigit():
                    label = int(label)
            elif isinstance(label, (np.integer, int)):
                label = int(label)
            elif isinstance(label, float) and label.is_integer():
                label = int(label)
            indices.append(idx)
            labels.append(label)

        if indices:
            overrides[task] = {
                "indices": indices,
                "labels": labels,
            }

    return overrides



def build_mask_inputs_from_table_edits(
    edited_df: pd.DataFrame,
    masked_tasks: Sequence[str],
    known_rows_expr: str,
    target_rows_expr: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
    """Build ``user_edits`` and ``masked_spec`` payloads from UI edits."""
    if edited_df is None or len(edited_df) == 0:
        return None, None, {"num_known": 0, "num_targets": 0}

    n_rows = len(edited_df)
    known_indices = parse_index_expression(known_rows_expr, n_rows)
    target_indices = parse_index_expression(target_rows_expr, n_rows)

    if not target_indices:
        target_indices = sorted(set(range(n_rows)) - set(known_indices))

    if not known_indices and not target_indices:
        return None, None, {"num_known": 0, "num_targets": 0}

    label_overrides = build_label_overrides_from_dataframe(
        df=edited_df,
        masked_tasks=masked_tasks,
        known_indices=known_indices,
    )

    node_mask = {
        "targets": target_indices,
        "context": known_indices,
        "context_weight": 0.1,
    }

    user_edits: Dict[str, Any] = {
        "node_mask": node_mask,
        "label_overrides": label_overrides,
    }
    masked_spec: Dict[str, Any] = {
        "masked_tasks": list(masked_tasks),
        "constraint_mode": "hard",
        "feedback_mode": "single_pass",
    }
    info = {
        "num_known": len(known_indices),
        "num_targets": len(target_indices),
    }
    return user_edits, masked_spec, info



def score_note_table(score: pt.score.Score) -> pd.DataFrame:
    """Create stable note-wise metadata table aligned with model prediction ordering."""
    note_array = score.note_array(include_pitch_spelling=True)
    note_array = np.sort(note_array, order=["onset_div", "pitch"])

    step = note_array["step"].astype(str)
    alter = note_array["alter"]
    octave = note_array["octave"] - 1
    accidental = np.where(
        alter == 0,
        "",
        np.where(alter > 0, np.char.multiply("#", alter), np.char.multiply("b", -alter)),
    )
    spelling = np.char.add(np.char.add(step, accidental), octave.astype(str))

    out = {
        "row": np.arange(len(note_array), dtype=int),
        "onset_beat": note_array["onset_beat"],
        "duration_beat": note_array["duration_beat"],
        "pitch_spelling": spelling,
        "pitch_midi": note_array["pitch"],
    }

    if "id" in note_array.dtype.names:
        out["note_id"] = note_array["id"]

    try:
        first_part = score.parts[0]
        out["measure"] = first_part.measure_number_map(note_array["onset_div"])
    except Exception:
        pass

    return pd.DataFrame(out)



def _decode_task_predictions(task: str, probs_or_ids: torch.Tensor) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Decode task predictions into label strings + confidence + class ids."""
    tensor = probs_or_ids.detach().cpu()
    if tensor.ndim == 1:
        class_ids = tensor.long().numpy()
        confidence = None
    else:
        probs = torch.softmax(tensor, dim=-1) if tensor.dtype.is_floating_point else tensor.float()
        class_ids = torch.argmax(probs, dim=-1).long().numpy()
        confidence = torch.max(probs, dim=-1).values.numpy()

    decoded: np.ndarray
    if task in available_representations:
        try:
            decoded_obj = available_representations[task].decode(np.asarray(class_ids).reshape(-1, 1))
            decoded = np.asarray(decoded_obj).reshape(-1)
        except Exception:
            decoded = np.asarray(class_ids)
    else:
        decoded = np.asarray(class_ids)

    return decoded, confidence, np.asarray(class_ids)



def predictions_to_dataframe(
    score: pt.score.Score,
    predictions: Dict[str, torch.Tensor],
    tasks: Optional[Sequence[str]] = None,
    include_confidence: bool = True,
    include_class_ids: bool = True,
) -> pd.DataFrame:
    """Convert model predictions into an editable note-wise dataframe."""
    base_df = score_note_table(score)
    selected_tasks = [t for t in (tasks or list(predictions.keys())) if t in predictions]

    n = len(base_df)
    for task in selected_tasks:
        decoded, confidence, class_ids = _decode_task_predictions(task, predictions[task])
        if len(decoded) != n:
            m = min(len(decoded), n)
            padded = np.full((n,), "", dtype=object)
            padded[:m] = decoded[:m]
            decoded = padded
            class_fill = np.full((n,), -1, dtype=int)
            class_fill[:m] = class_ids[:m]
            class_ids = class_fill
            if confidence is not None:
                conf_fill = np.zeros((n,), dtype=float)
                conf_fill[:m] = confidence[:m]
                confidence = conf_fill

        base_df[task] = decoded
        if include_class_ids:
            base_df[f"{task}_id"] = class_ids
        if include_confidence and confidence is not None:
            base_df[f"{task}_confidence"] = confidence

    return base_df


class HybridAnalysisPredictor:
    """Inference router using separate checkpoints for full and masked prediction."""

    def __init__(
        self,
        full_checkpoint_path: str,
        masked_checkpoint_path: Optional[str] = None,
        device: str = "auto",
    ) -> None:
        if not full_checkpoint_path:
            raise ValueError("full_checkpoint_path is required")

        self.full_checkpoint_path = full_checkpoint_path
        self.masked_checkpoint_path = masked_checkpoint_path or full_checkpoint_path
        self.device = resolve_device(device)

        self._full_model: Optional[ContinualAnalysisGNN] = None
        self._masked_model: Optional[ContinualAnalysisGNN] = None

    @staticmethod
    def should_use_masked_model(
        user_edits: Optional[Dict[str, Any]] = None,
        masked_spec: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Return True when conditioning information implies masked routing."""
        if user_edits:
            if user_edits.get("label_overrides"):
                return True
            if user_edits.get("node_mask") is not None:
                return True

        if not masked_spec:
            return False
        if masked_spec.get("known_labels"):
            return True
        if masked_spec.get("node_mask") is not None:
            return True
        if masked_spec.get("masked_tasks"):
            return True
        return False

    def _load_checkpoint(self, checkpoint_path: str) -> ContinualAnalysisGNN:
        model = ContinualAnalysisGNN.load_from_checkpoint(
            checkpoint_path,
            map_location=self.device,
            strict=False,
        )
        model.eval()
        model.to(self.device)
        return model

    def get_full_model(self) -> ContinualAnalysisGNN:
        if self._full_model is None:
            self._full_model = self._load_checkpoint(self.full_checkpoint_path)
        return self._full_model

    def get_masked_model(self) -> ContinualAnalysisGNN:
        if self._masked_model is None:
            self._masked_model = self._load_checkpoint(self.masked_checkpoint_path)
        return self._masked_model

    def predict(
        self,
        score: Any,
        user_edits: Optional[Dict[str, Any]] = None,
        masked_spec: Optional[Dict[str, Any]] = None,
        iterative_spec: Optional[Dict[str, Any]] = None,
        force_route: Optional[str] = None,
        return_edit_info: bool = False,
        return_iterative_trace: bool = False,
        return_route: bool = False,
    ) -> Any:
        """Predict using the full or masked model depending on the request payload."""
        route = force_route
        if route is None:
            route = "masked" if self.should_use_masked_model(user_edits=user_edits, masked_spec=masked_spec) else "full"
        route = route.lower().strip()
        if route not in {"full", "masked"}:
            raise ValueError("force_route must be one of: full, masked")

        if route == "masked":
            model = self.get_masked_model()
            ckpt = self.masked_checkpoint_path
        else:
            model = self.get_full_model()
            ckpt = self.full_checkpoint_path

        output = model.predict(
            score,
            user_edits=user_edits,
            masked_spec=masked_spec,
            iterative_spec=iterative_spec,
            return_edit_info=return_edit_info,
            return_iterative_trace=return_iterative_trace,
        )

        if not return_route:
            return output

        routing = RoutingDecision(route=route, checkpoint_path=ckpt)
        return output, routing
