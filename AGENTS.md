# AGENTS.md — AnalysisGNN Aggregation Research

## Environment Setup

All commands (inference, tests, scripts) **must** be run inside the `analysisgnn`
conda environment:

```bash
conda activate analysisgnn
```

Without this, none of the project's dependencies (torch, graphmuse, partitura, etc.)
will be available.

---

## IMPORTANT: AGENTS.md Writing Rules

**When updating this file, match the conciseness of existing "DONE" sections.**
Steps 1 and 2 are the reference style: ~15-20 lines each, listing what was created
and key API surfaces. Do NOT write verbose prose, do NOT repeat information that is
already documented elsewhere (e.g., in the code, docstrings, or table schemas above),
do NOT add sub-sub-sections. A completed step should be **shorter** than its
pre-implementation description, not longer.

---

## Project Context

**AnalysisGNN** is a multi-task Graph Neural Network for music analysis. Given a MusicXML
score, it produces per-note probability distributions for ~14 analysis tasks (Roman
numeral, local key, chord quality, inversion, scale degrees, cadence, phrase, etc.). The
graph representation uses four edge types from the GraphMuse paradigm:

| Edge type      | Semantics                                         |
|----------------|---------------------------------------------------|
| `onset`        | Notes sounding at the same onset time (vertical)  |
| `consecutive`  | Note ends where another begins (horizontal)       |
| `during`       | Note onset overlaps another note's duration        |
| `rest`         | Gap-bridging when no direct consecutive connection |

**Grouping notes** is a fundamental mechanism in this project. Beyond the four core
edge types, notes are often grouped into higher-level units (beats, measures, chords,
phrases, etc.). We adopt a **hyperedge-as-table** approach for groupings: a single
`hyperedges` table (paralleling the `edges` table) stores all group memberships,
differentiated by an `edge_type` column. Each group has a self-describing string
`group_id` (e.g., `beat_1`, `measure_13`, `chord_change_7`) and an optional
`parent_group_id` for recursive nesting. This design supports arbitrarily deep
hierarchies — e.g., a group of notes corresponding to a V7 harmony may contain a nested
subgroup forming a viio triad and another subgroup of non-chord tones — while remaining
relational, easy to query, and framework-agnostic. Pre-existing metrical groupings
(beat, measure) from GraphMuse are optional and can be regenerated with different
parameters (e.g., different beat granularity or measure grouping).

### Training Data Background

The training data are expert-labelled scores using Roman-numeral analysis (DCML
standard: keys, modulations, phrase boundaries, cadences; RomanText: keys + Roman
numerals only). These annotations represent a **contiguous segmentation of score
time**: each segment carries a single harmonic label, and the segment boundaries
partition the timeline without gaps or overlaps. For GNN training, these segment-level
labels were **decomposed** in two ways: first, each label was split into its atomic
features (one per analysis task: root, quality, inversion, key, etc.); second, these
atomic labels were **propagated** to every note whose onset falls within the annotated
segment. The research goal is essentially the **inverse** of both operations — but
not merely to reconstruct the original segmentation. Human analysts themselves choose
different granularity levels (from beat-by-beat detail to broad harmonic regions), and
the same score admits multiple valid analyses. We aim to support a **continuous range
of aggregation granularities**, from individual notes to large harmonic spans, including
nested and overlapping views. This opens up novel ways to use fine-grained ML outputs
in musicologically meaningful and useful ways, and paves the way for new evaluation
metrics and paradigms beyond simple label-matching accuracy.

---

## Research Goals (Overarching)

The current aggregation (`_resolve_aggregation_runtime()` in
`analysisgnn/models/analysis.py`) applies a fixed pipeline: onset-mean → beat-mean →
measure-mean (or a learned voter model). This collapses per-note information into
group-level consensus too early and too rigidly. The research aims to:

1. **Differentiate contradiction vs. agreement**: Identify regions where nearby notes'
   predictions are mutually supportive vs. contradictory, rather than averaging them
   away.

2. **Multi-level aggregation**: Support static groupings (onset, beat, measure) AND
   dynamic groupings (chord-change-point detection) AND hierarchical/nested groupings
   (clusters of notes with mutually consistent predictions).

3. **Roman-numeral reconstruction**: Given the probability distributions over atomic
   tasks for a group of notes, enumerate the **legal** (non-contradictory)
   Roman-numeral analyses, each with a composite probability. "Legal" means the
   combination of root, quality, inversion, and bass is internally consistent.

4. **Top-k label selection**: For arbitrary note groupings, compute top-k candidate
   Roman-numeral labels and select among them for maximal homogeneity (e.g.,
   maintaining a consistent local key over longer spans).

5. **Leverage graph structure**: Use the four edge types to propagate information,
   detect communities, and cluster notes based on graph connectivity + prediction
   similarity.

6. **Flexible parametrization**: Every part of the aggregation pipeline should be
   independently configurable so that researchers can fix some parts while exploring
   alternatives for others.

---

## Variable Parts of the Aggregation Architecture

These are the independently configurable dimensions of the aggregation pipeline. When
exploring, any subset can be fixed while others vary.

### V1. Context Window / Grouping Strategy

How notes are grouped before aggregation:

| Strategy            | Description                                               |
|---------------------|-----------------------------------------------------------|
| `onset`             | Notes sharing the same onset_div (current default level 1)|
| `beat`              | Notes within the same beat (current default level 2)      |
| `measure`           | Notes within the same measure (current default level 3)   |
| `custom_beat(size)` | Custom beat granularity (e.g., half-beat, two-beat)        |
| `custom_measure(n)` | Custom measure grouping (e.g., 2-bar hypermeter)          |
| `chord_change`      | Dynamic: segment boundaries at detected chord changes     |
| `graph_cluster`     | Dynamic: community detection on the note graph            |
| `consistency_cluster` | Dynamic: group notes with mutually consistent predictions |

### V2. Eligible Note Filter

Which notes within a group participate in aggregation:

- All notes (current default for some tasks)
- Chord tones only (`tpc_in_label == True`, current default for RNA tasks)
- By voice/staff
- By confidence threshold
- By prediction entropy threshold

### V3. Aggregation Operator

How probability distributions within a group are combined:

- Arithmetic mean (current `scatter_mean`)
- Weighted mean (weighted by confidence, by learned voter score, by duration, etc.)
- Geometric mean (product of distributions, renormalized)
- Max-pool (take highest confidence prediction)
- Majority vote (argmax per note, then mode)
- Harmonic mean
- Custom learned aggregator

### V4. Cross-Task Consistency Enforcement

How to resolve contradictions between tasks:

- Independent per-task (current: each task aggregated separately)
- Joint: enumerate legal Roman-numeral combinations from task distributions
- Constrained: filter illegal combinations (e.g., bass impossible given root+inversion)
- Beam search: top-k joint labels, prune by consistency
- ILP/constraint satisfaction: optimize global consistency

### V5. Multi-Level Composition

How aggregation results at different levels are combined:

- Sequential pipeline (current: onset → beat → measure)
- Parallel: aggregate at each level independently, compare
- Hierarchical/nested: finer levels inform coarser ones
- Bottom-up: aggregate upward through the graph
- Top-down: use coarser-level predictions to constrain finer ones

### V6. Temporal Smoothing / Continuity

How to enforce continuity over time:

- None (each group independent)
- Change-point detection (current: partial implementation in `onsetwise_logit_aggregation`)
- HMM/Viterbi over group-level predictions
- Graph-based message passing along consecutive edges
- Penalize key changes (prefer longer consistent key spans)

### V7. Certainty / Confidence Model

How to compute and use confidence values:

- Max probability (current)
- Entropy of distribution
- Margin (difference between top-2 probabilities)
- Calibrated confidence (temperature scaling, Platt scaling)
- Group-level agreement as confidence proxy

---

## Phase 1: Storage Architecture — Delta Lake for Analysis Results

### Overview

Analysis results (raw probability distributions + graph structure) are stored in a
**Delta Lake** per (score, inference settings, checkpoints) combination. This provides:

- **Columnar compression** via Parquet for large probability distributions
- **Built-in versioning** to track aggregation iterations and experiments
- **Single long-format probabilities table** — one row per (note, task, class)
  combination, making it easy to query, filter, and aggregate across tasks without
  dealing with heterogeneous column schemas
- **Resolved human-readable class labels** via `resolve_task_vocabulary()` in
  `analysisgnn/utils/chord_representations.py`, which resolves labels from
  `available_representations`, `CadenceEncoder`, `NoteDegree49`, known binary task
  semantics, and integer-label tasks (see the function for the full resolution chain)
- **Organic growth**: new aggregation and grouping tables are added to the same Delta
  Lake over time

Additionally, **CSV export** is available as a user-triggered convenience from any
Delta Lake table (e.g., for inspection, sharing, or git-diffable summaries).

### Delta Lake Directory Layout

```
outputs/<score_id>/
├── notes/                 # Delta table: one row per note
│   ├── _delta_log/
│   └── *.parquet
├── edges/                 # Delta table: one row per directed edge
│   ├── _delta_log/
│   └── *.parquet
├── hyperedges/            # Delta table: group memberships (all grouping types)
│   ├── _delta_log/
│   └── *.parquet
├── probabilities/         # Delta table: one row per (note, task, class)
│   ├── _delta_log/
│   └── *.parquet
├── agg_<method>_<task>/   # Aggregation result tables (added over time)
│   ├── _delta_log/
│   └── *.parquet
└── metadata.json          # Score ID, checkpoint paths, inference params,
                           #   task_dict, class vocabularies, creation timestamp
```

### Table Schemas

#### `notes` table

One row per note, sorted by `(onset_div, pitch)` to match graph indices.

| Column           | Type    | Description                              |
|------------------|---------|------------------------------------------|
| `note_id`        | string  | Score-level note ID (from partitura)     |
| `onset_div`      | int     | Onset in score divisions                 |
| `onset_beat`     | float   | Onset in beats                           |
| `duration_div`   | int     | Duration in score divisions              |
| `duration_beat`  | float   | Duration in beats                        |
| `pitch_midi`     | int     | MIDI pitch number                        |
| `pitch_spelling` | string  | e.g., "C#4", "Bb3"                       |
| `staff`          | int     | Staff number (0-based)                   |
| `voice`          | int     | Voice number                             |
| `measure`        | int     | Measure number                           |
| `ts_beats`       | float   | Time signature beats value               |

#### `edges` table

One row per directed edge between notes.

| Column      | Type   | Description                                      |
|-------------|--------|--------------------------------------------------|
| `src`       | string | Source `note_id`                                  |
| `dst`       | string | Destination `note_id`                              |
| `edge_type` | string | One of: `onset`, `consecutive`, `during`, `rest`  |

#### `hyperedges` table

One row per (group, member note) pair. All grouping types live in this single table,
differentiated by `edge_type` — paralleling how the `edges` table uses `edge_type`
for the four core edge types. Standard groupings created at inference time include
types `onset`, `beat`, `measure`. Additional types (e.g., `chord_change`,
`consistency_cluster`) are added as needed during aggregation experiments.

| Column            | Type        | Description                                   |
|-------------------|-------------|-----------------------------------------------|
| `group_id`        | string      | Self-describing ID, e.g., `beat_1`, `measure_13` |
| `note_id`         | string      | `note_id` of a member note                    |
| `edge_type`       | string      | Grouping type, e.g., `beat`, `measure`, `chord_change` |
| `parent_group_id` | string or null | Parent group for nesting (null = top-level) |

A note can appear in multiple groups. Groups can nest arbitrarily via
`parent_group_id`. For example, a group `chord_change_7` with 8 member notes may
have a nested subgroup `nct_7_1` whose `parent_group_id` is `chord_change_7`,
containing only the 3 non-chord tones among those 8 notes.

Optional additional columns depending on the grouping type:

| Column          | Type    | Description                                     |
|-----------------|---------|-------------------------------------------------|
| `weight`        | float32 | Membership weight (e.g., soft clustering)        |
| `label`         | string  | Human-readable label for the group (optional)    |
| `onset_start`   | int     | Start onset_div of the group span (optional)     |
| `onset_end`     | int     | End onset_div of the group span (optional)       |

#### `probabilities` table (long format)

One row per (note, task, class) combination. For a score with N notes and T tasks
averaging C classes each, this produces N * T * C rows. The long format makes it
easy to query, filter, and aggregate across tasks without heterogeneous column schemas.

| Column        | Type    | Description                                        |
|---------------|---------|----------------------------------------------------|
| `note_id`     | string  | Note ID (join key to `notes` table)                |
| `task`        | string  | Task name, e.g., `romanNumeral`, `cadence`         |
| `class_id`    | int32   | Integer index in the softmax output (0-based)      |
| `class_label` | string  | Resolved human-readable label (see below)          |
| `probability` | float32 | Softmax probability for this class                 |
| `is_argmax`   | bool    | True for the class with highest probability        |
| `rank`        | int32   | Rank by probability (1 = highest)                  |

Invariants:
- For each `(note_id, task)` group, `probability` sums to 1.0
- Exactly one row per group has `is_argmax = True`
- The row with `rank = 1` always has `is_argmax = True`

**Class label resolution** is handled by `resolve_task_vocabulary()` in
`analysisgnn/utils/chord_representations.py` (the single canonical source; also
re-exported by `analysisgnn/storage/delta_writer.py` for backward compatibility).
The resolution chain:
1. `available_representations` (canonical vocabularies from `chord_representations.py`)
2. Key aliases (e.g., model key `hrythm` maps to `hrhythm` in `available_representations`)
3. `CadenceEncoder.accepted_cadences` (for the `cadence` task)
4. Extra representation classes not in `available_representations` (e.g., `NoteDegree49`)
5. Known binary task semantics — all binary tasks use `"False"` / `"True"` string labels
   (class 0 = `"False"`, class 1 = `"True"`)
6. Known integer-label tasks (e.g., `downbeat` -> `["0", "1", ..., "44"]`)
7. Fallback: `class_0`, `class_1`, ...

The `class_label` column can be empty string (e.g., cadence class 0 = no cadence)
but is never null/NaN.

**Label decoding is unified** across the entire codebase: the Gradio display
(`_decode_task_predictions()` in `hybrid_predictor.py`), the Delta Lake writer
(`delta_writer.py`), and reference CSV generation (`scripts/generate_reference_csvs.py`)
all use `resolve_task_vocabulary()` from `chord_representations.py` as the single source
of truth. This was previously inconsistent — the Gradio path used `available_representations[task].decode()` with raw-integer fallback for unrecognized tasks, while
the Delta Lake path had a richer resolution chain.

#### `agg_<method>_<task>` tables (added over time)

**Note:** The exact schema for aggregation result tables is tentative and will evolve
as we develop specific aggregation methods. The following is a draft based on current
understanding; it will be refined during implementation.

Aggregation result tables store probability distributions after applying a named
aggregation method. They have the same probability columns as the corresponding
`probs_<task>` table, re-computed after aggregation. Additional columns may include
references to the grouping table used, but the specifics depend on the aggregation
paradigm (e.g., whether the method produces per-note distributions, per-group
distributions, or both).

Tentative additional columns:

| Column          | Type   | Description                                   |
|-----------------|--------|-----------------------------------------------|
| `edge_type`     | string | Hyperedge type used for grouping               |
| `group_id`      | int    | Which group this note/row belongs to           |

#### `metadata.json`

```json
{
  "score_path": "path/to/score.musicxml",
  "score_id": "derived_from_filename_or_hash",
  "full_checkpoint": "path/to/full.ckpt",
  "masked_checkpoint": "path/to/masked.ckpt",
  "device": "cuda",
  "inference_timestamp": "2026-03-22T14:30:00Z",
  "task_dict": {"romanNumeral": 185, "localkey": 50, ...},
  "class_vocabularies": {
    "romanNumeral": ["I", "i", "II", "ii", ...],
    "localkey": ["C", "c", "D", "d", ...],
    "quality": ["major_triad", "minor_triad", ...],
    ...
  },
  "edge_types_included": ["onset", "consecutive", "during", "rest"],
  "hyperedge_types": ["onset", "beat", "measure"],
  "aggregation_history": []
}
```

### Graph Reconstruction

The Delta Lake tables contain everything needed to reconstruct the graph in multiple
frameworks:

#### PyTorch Geometric (HeteroData)

```python
import pyarrow.parquet as pq
from torch_geometric.data import HeteroData

notes_df = pq.read_table("outputs/score_id/notes").to_pandas()
edges_df = pq.read_table("outputs/score_id/edges").to_pandas()

id_to_idx = {nid: i for i, nid in enumerate(notes_df["note_id"])}

data = HeteroData()
data["note"].onset_div = torch.tensor(notes_df["onset_div"].values)
data["note"].pitch = torch.tensor(notes_df["pitch_midi"].values)
# ... other note attributes

for etype in ["onset", "consecutive", "during", "rest"]:
    mask = edges_df["edge_type"] == etype
    src = edges_df.loc[mask, "src"].map(id_to_idx).values
    dst = edges_df.loc[mask, "dst"].map(id_to_idx).values
    data["note", etype, "note"].edge_index = torch.tensor([src, dst])
```

#### rustworkx

```python
import rustworkx as rx

graph = rx.PyDiGraph()
n = len(notes_df)
note_indices = graph.add_nodes_from(range(n))

for _, row in edges_df.iterrows():
    graph.add_edge(id_to_idx[row["src"]], id_to_idx[row["dst"]], row["edge_type"])
```

#### Loading grouping tables

```python
# Load a grouping and convert to a dict of group_id -> list of note indices
groups_df = pq.read_table("outputs/score_id/hyperedges").to_pandas()
beat_groups = groups_df[groups_df["edge_type"] == "beat"]
groups = beat_groups.groupby("group_id")["note_id"].apply(list).to_dict()

# Nesting: find children of a parent group
children = groups_df[groups_df["parent_group_id"] == parent_id]
```

### CSV Export (User-Triggered)

Any Delta Lake table can be exported to CSV on demand:

```python
# From the Gradio app or CLI
df = pq.read_table("outputs/score_id/probabilities").to_pandas()
df.to_csv("probabilities.csv", index=False)

# Or with argmax-only summary
notes_df = pq.read_table("outputs/score_id/notes").to_pandas()
probs_df = pq.read_table("outputs/score_id/probabilities").to_pandas()
argmax = probs_df[probs_df["is_argmax"]].pivot(index="note_id", columns="task", values="class_label")
notes_df = notes_df.merge(argmax, on="note_id", how="left")
notes_df.to_csv("notes_summary.csv", index=False)
```

### Dependencies to Add

```
deltalake>=0.22.0
pyarrow>=14.0.0
```

---

## Phase 1: Implementation Plan

### Step 1: Delta Lake Writer Module — DONE

Created `analysisgnn/storage/delta_writer.py`:

- `write_analysis_results(output_dir, score, predictions, data, task_dict, metadata)`
  - Writes `notes/`, `edges/`, `probabilities/`, `hyperedges/` tables + `metadata.json`
  - Resolves class labels via `resolve_task_vocabulary()` which chains through
    `available_representations`, key aliases, `CadenceEncoder`, `NoteDegree49`,
    known binary task semantics, integer-label tasks, and fallback
  - `probabilities` table uses **long format**: one row per (note, task, class),
    with columns `note_id`, `task`, `class_id`, `class_label`, `probability`,
    `is_argmax`, `rank`
  - Populates the `hyperedges` table with default groupings (`onset`, `beat`, `measure`)
    derived from the graph's onset_div, beat cluster, and measure cluster attributes

Also done:
- `predict()` in `analysisgnn/models/analysis.py` gains `return_intermediates=True`
  to return the `score`, `note_array`, and `data` objects alongside predictions
- `_resolve_aggregation_runtime()` gains support for `"none"` mode to skip
  aggregation entirely
- 25 integration tests in `tests/test_delta_writer.py` (real inference on Mozart K.1)
- Dependencies `deltalake>=0.22.0`, `pyarrow>=14.0.0` added to `requirements.txt`

### Step 2: Delta Lake Reader Module + Demo Notebook — DONE

Created `analysisgnn/storage/delta_reader.py`:

- `load_notes(output_dir) -> pd.DataFrame`
- `load_edges(output_dir, edge_types=None) -> pd.DataFrame`
- `load_probabilities(output_dir, task=None, top_k=None) -> pd.DataFrame`
- `load_hyperedges(output_dir, edge_type=None) -> pd.DataFrame`
- `load_metadata(output_dir) -> dict`
- `list_group_types(output_dir) -> List[str]`
- `list_tables(output_dir) -> List[str]` — scans for subdirectories with `_delta_log/`
- `export_table_to_csv(output_dir, table_name, csv_path) -> str`
- `argmax_summary(output_dir, tasks=None) -> pd.DataFrame` — wide-format pivot with
  per-task argmax class label + confidence columns, joined with notes table

Also done:
- 19 tests in `tests/test_delta_reader.py` (reads existing Delta Lake at
  `outputs/Minuet_in_G_Major_K.1/`; skipped via `pytestmark` if output absent)
- Demo notebook `notebooks/delta_lake_demo.ipynb` — full predict → store → load →
  inspect workflow. Write is guarded with an existence check to avoid Delta Log
  version pollution (see Conventions below)

### Step 3: Unified Label Decoding + Reference CSVs + Aggregation Package — DONE

Unified all label decoding behind `resolve_task_vocabulary()` in
`analysisgnn/utils/chord_representations.py` (single canonical source; `delta_writer.py`
re-exports for backward compatibility). `_decode_task_predictions()` in
`hybrid_predictor.py` now uses it exclusively. Binary tasks use `"False"`/`"True"`,
the `romanNumeral` missing-comma bug was fixed (185 entries), and
`format_table_output()` was moved to `chord_representations.py` so both the Gradio app
and `scripts/generate_reference_csvs.py` use the same function.

Created `analysisgnn/aggregation/` package:
- `base.py`, `registry.py`, `mean.py`, `__init__.py`
- `"none"` (passthrough) and `"mean"` (onset → beat → measure) strategies
- `MeanAggregation` validated to match the model's built-in `"mean"` mode exactly

Also: `scripts/generate_reference_csvs.py` produces `reference_none.csv` and
`reference_mean.csv`; 8 tests in `tests/test_aggregation_mean.py`; demo notebook
updated. All 52 tests pass.

### Step 4: Gradio Restructuring + Delta Lake Integration — DONE

Rewrote `examples/gradio_hybrid_analysis_app.py` into a 3-module layout:
- **Module 1** — Data Source: inference (`aggregation=none`, `return_intermediates`)
  with auto Delta Lake write, or load via `gr.FileExplorer`
- **Module 2** — Analysis Results: aggregation, CSV download, editable table, Verovio
- **Module 3** — Edit-Conditioned Re-Inference (grayed out until Module 1a runs)

Key: `_merge_or_create()` in `delta_writer.py` (Delta merge + vacuum);
`_precompute_delta_dfs()` converts predictions to long-format once after inference.

### Step 4a: Verovio Score Upload + Note Coloring — DONE

Verovio tab has its own `gr.File` upload (auto-filled from Module 1a). NCT coloring
via `tpc_in_label` confidence → linear grey interpolation. Note coloring interface:
`payload.note_colors` dict → CSS `--agn-note-color` + `.agn-colored`.

### Step 4b: FlexOHR-Based Complete RN Column — DONE

Replaced hand-rolled `decode_roman_numeral()` with FlexOHR's `OHR.from_()` +
`.to_format('dcml')`. Key additions:
- `_fix_key_mode(df)`: infers major/minor from romanNumeral tonic counts per key
- `_derive_global_key(df, k=5)`: votes from first *k* tonic chords' localkey values
- `_build_complete_rn_column(df, global_key)`: uses `build_ohrs_from_dataframe`
- `global_key_field` text field in Module 2, auto-populated, editable

### Step 5a: FlexOHR Codec Extension + Scoring Framework — DONE

Created `analysisgnn/aggregation/scoring.py` with `ScoringContext` (thin wrapper
over Delta Lake DataFrames, filters on the fly) and pluggable scorers
(`ProductScorer`, `GeometricMeanScorer`, `WeightedTaskScorer`, `SeparateScorer`).
Extended FlexOHR with `ChordQuality.generic_augmented_sixth`, decode-only aliases,
and full 15-quality codec in `codecs/analysisgnn.py`.

### Step 5b: Top-k Roman Numeral Enumeration — DONE

Created `analysisgnn/aggregation/roman_numeral.py`:
`enumerate_roman_numerals(context, global_key, *, scorer, k, top_n)` returns
`(list[RankedCandidate], EnumerationTrace)`. Pipeline: top-k per task → Cartesian
product → inversion legality → OHR construction + validation → DCML dedup → score → rank.
Demo notebook: `notebooks/roman_numeral_enumeration.py`.

### Step 5c: Grouped Note Panel + Agreement Coloring — DONE

Verovio note info panel with grouped tiles and agreement coloring. Top-k DCML
labels as button group; clicking a candidate recolors harmony tiles.
`derive_expected_labels()` in FlexOHR codec derives expected values from OHRs.

### Step 5d: Two-Dropdown Aggregation UI — DONE

- **"Aggregation Groups"** dropdown: `None`, `Onset`, `Beat`, `Measure`.
- **"Aggregation Strategy"** dropdown: `Mean` (+ future scorers).
- `GroupedMeanAggregation(level)` in `mean.py`: single-level mean at onset/beat/measure.
- `_enumerate_rn_candidates` accepts `grouping=` parameter.
- Bug fixes: FlexOHR `build_key_context()` mode inference, NaN-safe OHR builds,
  DCML global-key prefix format, localkey mode map for enumeration.

### Step 5e: Multi-Table Aggregation + Agreement Coloring Fixes — DONE

**Multi-table UI**: Aggregation now produces two outputs — a group-level table and
an updated notes table with a label column. `gr.Radio` (`table_selector`) switches
between tables. Naming: column `{group}_{strategy}_label` (e.g., `measure_mean_label`),
Radio label `{Group_plural} ({Strategy})` (e.g., `Measures (Mean)`). Tables accumulate
across aggregations. Group-level table built by `_build_group_table()` (representative
note per hyperedge group, min onset_beat, FlexOHR label). Labels mapped to notes via
`_map_group_labels_to_notes()`.

**Column rename**: `romanNumeral_full` → `note_label` in the display layer (Gradio app,
JS payload, tests). Internal model/utility code retains `romanNumeral_full`.

**Agreement coloring fixes** (FlexOHR codec + `roman_numeral.py`):
- romanNumeral expected value: `ohr.with_(inversion=0)` for root-position DCML (vocab
  excludes inversion figures)
- Pitch-class notation: `_flx_to_agnn()` converts FlexOHR `b` → AnalysisGNN `-` for
  root, bass, tonkey, localkey expected values
- Aggregation labels shown as tiles in Verovio note panel (`agg_labels` payload field)

**Verovio iframe**: `ResizeObserver` auto-resize, `min-height:980px` initial.

---

## Verovio Integration Architecture

### Overview

Verovio-based score viewer embedded as an `<iframe srcdoc>`. Python builds a payload
dict (`_build_visual_payload()`) containing `score_xml` (original MusicXML, never
partitura re-export), per-note metadata, edges, and `meta.roman_spans`. The JS
renders via Verovio WASM 5.0.0, maps notes by ID, overlays edges/RN labels.

**Do not upgrade to Verovio 6.x** — version 6.0.0 changed SVG styling structure,
breaking the `.page-margin` selector.

### Files

| File | Role |
|------|------|
| `examples/assets/verovio_score_graph.html` | Template: CSS, JS, payload, Verovio CDN |
| `examples/assets/verovio_score_graph.js` | Rendering: Verovio init, note mapping, overlays, panel |
| `examples/assets/verovio_score_graph.css` | Styling: toolbar, panel, note highlighting, edges |

### Important: always use the original MusicXML

Partitura's `save_musicxml()` strips `<accidental>` elements. Always use the
original uploaded file; re-export only as fallback for `.mxl` format.

---

## Conventions

- Always say **note**, never "vertex" or "node" — since groupings are hyperedges (not
  hypernodes), there is no ambiguity
- Tables are named `probabilities` for raw distributions (long format),
  `agg_<method>_<task>` for aggregated results, `hyperedges` for group memberships
  (all grouping types)
- All per-note tables share the same row ordering (by `onset_div, pitch` sort)
  and are joined via `note_id`
- Class label columns use the **resolved human-readable names** from the task vocabulary
  as-is (e.g., `major triad` with space, not `major_triad`)
- Delta Lake versioning tracks the evolution of aggregation experiments within a run
- **Merge-based writes**: `delta_writer.py` uses `_merge_or_create()` which performs
  a standard Delta Lake merge (upsert + delete stale rows) followed by
  `vacuum(retention_hours=0)` to remove orphaned parquet files. First write to a
  non-existent table uses plain `write_deltalake`. Callers do not need existence
  guards — the merge handles both creation and update.

### Notebook Style

These rules are **mandatory** for all `.py` (jupytext) and `.ipynb` notebook files:

1. **Never use `print()` in loops.** Build a DataFrame or dict instead. A `for` loop
   with `print()` inside it is always wrong in a notebook.
2. **Vectorize.** Use pandas/numpy operations, not Python loops over rows or groups.
   When iterating over groups is unavoidable, collect results into a list of dicts
   and convert to a DataFrame in one shot — never print intermediate results.
3. **No `print()` for inspection.** Use `__repr__()` / `_repr_html_()` methods on
   classes, or bare expressions that the notebook renderer displays automatically.
   Use on-the-fly dicts only when a quick one-off inspection is needed.
4. **Never use `df.head()`.** The IDE / notebook renderer handles output truncation.
   Just return the bare DataFrame expression.
5. **Prefer rich display.** Classes that appear in notebook output should have
   `_repr_html_()` returning an HTML table or summary. Dataclasses should have a
   concise `__repr__`.
