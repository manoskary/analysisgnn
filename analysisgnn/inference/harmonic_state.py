"""Structured harmonic-state utilities for onset-level harmony decoding.

This module exposes two cached libraries:

- ``get_default_harmonic_state_library`` keeps the original RN-anchored state
  space used by the legacy/structured_v2 paths.
- ``get_default_component_harmonic_state_library`` builds a component-first
  state space keyed by ``(localkey, degree1, degree2, quality, inversion)``.

The component-first library matches the actual harmonic target used by the
project. The auxiliary ``romanNumeral`` head can still be recovered when
possible, but it is not treated as the primary harmonic state variable.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from music21 import key as m21key
from music21 import roman

from analysisgnn.utils.chord_representations import available_representations
from analysisgnn.utils.roman_decode import decode_roman_numeral


COMPONENT_STATE_TASKS: Tuple[str, ...] = (
    "localkey",
    "degree1",
    "degree2",
    "quality",
    "inversion",
)


def _labels_for_task(task: str) -> List[str]:
    rep = available_representations.get(task, None)
    class_list = getattr(rep, "classList", None) if rep is None else getattr(rep, "classList", None)
    if isinstance(class_list, Sequence) and not isinstance(class_list, (str, bytes)):
        return [str(x) for x in class_list]
    return []


def _index_map(labels: Iterable[str]) -> Dict[str, int]:
    return {str(label): idx for idx, label in enumerate(labels)}


def _pitch_name_to_vocab(name: Optional[str]) -> str:
    if not name:
        return ""
    return str(name).strip().replace("b", "-")


def _degree_label_from_rn(rn_obj: Optional[roman.RomanNumeral]) -> str:
    if rn_obj is None:
        return "None"
    scale_degree = getattr(rn_obj, "scaleDegree", None)
    if scale_degree in (None, ""):
        return "None"
    accidental = str(getattr(rn_obj, "frontAlterationString", "") or "")
    accidental = accidental.replace("b", "-")
    return f"{accidental}{int(scale_degree)}"


def _quality_from_rn(rn_obj: roman.RomanNumeral) -> str:
    figure = str(getattr(rn_obj, "figure", "") or "")
    common_name = str(getattr(rn_obj, "commonName", "") or "")
    if figure.startswith("Ger"):
        return "German augmented sixth chord"
    if figure.startswith("Fr"):
        return "French augmented sixth chord"
    if figure.startswith("It"):
        return "Italian augmented sixth chord"
    if common_name in {
        "major triad",
        "minor triad",
        "diminished triad",
        "augmented triad",
        "dominant seventh chord",
        "minor seventh chord",
        "major seventh chord",
        "diminished seventh chord",
        "half-diminished seventh chord",
        "minor-augmented tetrachord",
        "augmented sixth",
        "German augmented sixth chord",
        "French augmented sixth chord",
        "Italian augmented sixth chord",
    }:
        return common_name
    aliases = {
        "major-minor seventh chord": "dominant seventh chord",
        "dominant-11th": "dominant seventh chord",
        "minor-major seventh chord": "minor-augmented tetrachord",
        "minor sixth chord": "minor triad",
        "major sixth chord": "major triad",
        "power chord": "major triad",
    }
    if common_name in aliases:
        return aliases[common_name]
    return "major triad"


def _functional_class(rn_label: str) -> str:
    text = str(rn_label or "").strip()
    if not text or text == "none":
        return "none"
    if text.startswith(("Ger", "Fr", "It", "N", "bII")):
        return "predominant"
    lowered = text.lower()
    if lowered.startswith(("vii", "#vii")) or "cad64" in lowered or lowered.startswith("v"):
        return "dominant"
    if lowered.startswith(("ii", "iv")):
        return "predominant"
    if lowered.startswith(("i", "vi", "iii")):
        return "tonic"
    return "transitional"


def _tonic_pitch_class(localkey: str) -> Optional[int]:
    try:
        return int(m21key.Key(localkey).tonic.pitchClass)
    except Exception:
        return None


@dataclass(frozen=True)
class HarmonicState:
    state_id: int
    localkey: str
    roman_numeral: str
    quality: str
    inversion: int
    degree1: str
    degree2: str
    root: str
    bass: str
    functional_class: str
    complete_rn: str
    component_rn: str
    tonic_pc: Optional[int]
    root_pc: Optional[int]
    bass_pc: Optional[int]
    class_ids: Dict[str, int]


class HarmonicStateLibrary:
    """Legal harmonic-state library shared by structured decoding paths."""

    def __init__(self, states: List[HarmonicState]) -> None:
        self.states = tuple(states)
        self.by_pair: Dict[Tuple[int, int], HarmonicState] = {}
        self.by_roman: Dict[int, List[HarmonicState]] = {}
        self.by_localkey: Dict[int, List[HarmonicState]] = {}
        for state in self.states:
            rn_idx = state.class_ids.get("romanNumeral", None)
            lk_idx = state.class_ids.get("localkey", None)
            if rn_idx is not None and lk_idx is not None:
                self.by_pair[(rn_idx, lk_idx)] = state
                self.by_roman.setdefault(rn_idx, []).append(state)
                self.by_localkey.setdefault(lk_idx, []).append(state)

    def get(self, roman_idx: int, localkey_idx: int) -> Optional[HarmonicState]:
        return self.by_pair.get((int(roman_idx), int(localkey_idx)))

    def candidates_from_topk(
        self,
        roman_indices: Iterable[int],
        localkey_indices: Iterable[int],
    ) -> List[HarmonicState]:
        out: List[HarmonicState] = []
        seen = set()
        for rn_idx in roman_indices:
            for lk_idx in localkey_indices:
                state = self.get(int(rn_idx), int(lk_idx))
                if state is None:
                    continue
                if state.state_id in seen:
                    continue
                seen.add(state.state_id)
                out.append(state)
        return out


class ComponentHarmonicStateLibrary:
    """Legal harmonic states keyed by the five core component tasks."""

    def __init__(self, states: List[HarmonicState]) -> None:
        self.states = tuple(states)
        self.by_components: Dict[Tuple[int, int, int, int, int], HarmonicState] = {}
        self.by_task_class: Dict[str, Dict[int, List[HarmonicState]]] = {
            task: {} for task in COMPONENT_STATE_TASKS
        }
        for state in self.states:
            key_values: List[int] = []
            valid_key = True
            for task in COMPONENT_STATE_TASKS:
                cls_idx = state.class_ids.get(task, None)
                if cls_idx is None:
                    valid_key = False
                    break
                key_values.append(int(cls_idx))
                self.by_task_class[task].setdefault(int(cls_idx), []).append(state)
            if valid_key:
                self.by_components[tuple(key_values)] = state

    def get(
        self,
        localkey_idx: int,
        degree1_idx: int,
        degree2_idx: int,
        quality_idx: int,
        inversion_idx: int,
    ) -> Optional[HarmonicState]:
        key = (
            int(localkey_idx),
            int(degree1_idx),
            int(degree2_idx),
            int(quality_idx),
            int(inversion_idx),
        )
        return self.by_components.get(key)

    def candidates_from_topk(
        self,
        class_id_candidates_by_task: Dict[str, Iterable[int]],
        *,
        fallback_match_min: int = 3,
        max_states: int = 256,
    ) -> List[HarmonicState]:
        per_task: Dict[str, List[int]] = {}
        for task in COMPONENT_STATE_TASKS:
            values = class_id_candidates_by_task.get(task, [])
            deduped: List[int] = []
            seen = set()
            for value in values:
                try:
                    idx = int(value)
                except Exception:
                    continue
                if idx in seen:
                    continue
                seen.add(idx)
                deduped.append(idx)
            if deduped:
                per_task[task] = deduped

        if len(per_task) == len(COMPONENT_STATE_TASKS):
            exact: List[HarmonicState] = []
            seen_state_ids = set()
            candidate_lists = [per_task[task] for task in COMPONENT_STATE_TASKS]
            for values in itertools.product(*candidate_lists):
                state = self.by_components.get(tuple(int(v) for v in values))
                if state is None or state.state_id in seen_state_ids:
                    continue
                seen_state_ids.add(state.state_id)
                exact.append(state)
            if exact:
                return exact

        allowed_sets = {
            task: set(values)
            for task, values in per_task.items()
        }
        scored: List[Tuple[int, int, HarmonicState]] = []
        for state in self.states:
            matches = 0
            for task, allowed in allowed_sets.items():
                cls_idx = state.class_ids.get(task, None)
                if cls_idx is not None and int(cls_idx) in allowed:
                    matches += 1
            if matches < max(1, int(fallback_match_min)):
                continue
            localkey_match = int(
                state.class_ids.get("localkey", -1) in allowed_sets.get("localkey", set())
            )
            scored.append((matches, localkey_match, state))
        scored.sort(key=lambda item: (item[0], item[1], -item[2].state_id), reverse=True)
        out: List[HarmonicState] = []
        seen_state_ids = set()
        for _, _, state in scored:
            if state.state_id in seen_state_ids:
                continue
            seen_state_ids.add(state.state_id)
            out.append(state)
            if len(out) >= int(max_states):
                break
        return out


@lru_cache(maxsize=1)
def get_default_harmonic_state_library() -> HarmonicStateLibrary:
    roman_labels = _labels_for_task("romanNumeral")
    localkey_labels = _labels_for_task("localkey")
    quality_map = _index_map(_labels_for_task("quality"))
    degree1_map = _index_map(_labels_for_task("degree1"))
    degree2_map = _index_map(_labels_for_task("degree2"))
    inversion_map = _index_map(_labels_for_task("inversion"))
    root_map = _index_map(_labels_for_task("root"))
    bass_map = _index_map(_labels_for_task("bass"))
    roman_map = _index_map(roman_labels)
    localkey_map = _index_map(localkey_labels)

    states: List[HarmonicState] = []
    next_state_id = 0
    for rn_label in roman_labels:
        if rn_label in {"", "none", "185"}:
            continue
        for localkey in localkey_labels:
            try:
                rn_obj = roman.RomanNumeral(rn_label, localkey)
            except Exception:
                continue

            quality = _quality_from_rn(rn_obj)
            inversion = int(getattr(rn_obj, "inversion", lambda: 0)() or 0)
            degree1 = _degree_label_from_rn(rn_obj)
            secondary = getattr(rn_obj, "secondaryRomanNumeral", None)
            degree2 = _degree_label_from_rn(secondary) if secondary is not None else "None"
            root = _pitch_name_to_vocab(rn_obj.root().name if rn_obj.root() is not None else "")
            bass = _pitch_name_to_vocab(rn_obj.bass().name if rn_obj.bass() is not None else "")
            try:
                component_rn = decode_roman_numeral(
                    degree1=degree1,
                    degree2=degree2,
                    inversion=inversion,
                    quality=quality,
                    localkey=localkey,
                )
            except Exception:
                component_rn = ""
            complete_rn = rn_label or component_rn

            class_ids: Dict[str, int] = {
                "romanNumeral": roman_map[str(rn_label)],
                "localkey": localkey_map[str(localkey)],
            }
            if quality in quality_map:
                class_ids["quality"] = quality_map[quality]
            if degree1 in degree1_map:
                class_ids["degree1"] = degree1_map[degree1]
            if degree2 in degree2_map:
                class_ids["degree2"] = degree2_map[degree2]
            inv_key = str(inversion)
            if inv_key in inversion_map:
                class_ids["inversion"] = inversion_map[inv_key]
            elif inversion in inversion_map:
                class_ids["inversion"] = inversion_map[inversion]
            if root in root_map:
                class_ids["root"] = root_map[root]
            if bass in bass_map:
                class_ids["bass"] = bass_map[bass]

            root_pc = None
            bass_pc = None
            try:
                root_pc = int(rn_obj.root().pitchClass) if rn_obj.root() is not None else None
            except Exception:
                root_pc = None
            try:
                bass_pc = int(rn_obj.bass().pitchClass) if rn_obj.bass() is not None else None
            except Exception:
                bass_pc = None

            states.append(
                HarmonicState(
                    state_id=next_state_id,
                    localkey=str(localkey),
                    roman_numeral=str(rn_label),
                    quality=quality,
                    inversion=inversion,
                    degree1=degree1,
                    degree2=degree2,
                    root=root,
                    bass=bass,
                    functional_class=_functional_class(rn_label),
                    complete_rn=complete_rn,
                    component_rn=component_rn,
                    tonic_pc=_tonic_pitch_class(localkey),
                    root_pc=root_pc,
                    bass_pc=bass_pc,
                    class_ids=class_ids,
                )
            )
            next_state_id += 1
    return HarmonicStateLibrary(states)


@lru_cache(maxsize=1)
def get_default_component_harmonic_state_library() -> ComponentHarmonicStateLibrary:
    states: List[HarmonicState] = []
    seen_component_keys = set()
    for base_state in get_default_harmonic_state_library().states:
        component_key = (
            str(base_state.localkey),
            str(base_state.degree1),
            str(base_state.degree2),
            str(base_state.quality),
            int(base_state.inversion),
        )
        if component_key in seen_component_keys:
            continue
        if any(task not in base_state.class_ids for task in COMPONENT_STATE_TASKS):
            continue
        seen_component_keys.add(component_key)
        component_rn = str(base_state.component_rn or "").strip()
        complete_rn = str(base_state.complete_rn or component_rn or base_state.roman_numeral).strip()
        states.append(
            HarmonicState(
                state_id=len(states),
                localkey=str(base_state.localkey),
                roman_numeral=str(base_state.roman_numeral or complete_rn),
                quality=str(base_state.quality),
                inversion=int(base_state.inversion),
                degree1=str(base_state.degree1),
                degree2=str(base_state.degree2),
                root=str(base_state.root),
                bass=str(base_state.bass),
                functional_class=str(base_state.functional_class),
                complete_rn=complete_rn,
                component_rn=(component_rn or complete_rn),
                tonic_pc=base_state.tonic_pc,
                root_pc=base_state.root_pc,
                bass_pc=base_state.bass_pc,
                class_ids=dict(base_state.class_ids),
            )
        )
    return ComponentHarmonicStateLibrary(states)
