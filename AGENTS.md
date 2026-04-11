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
- **Module 1** — Data Source: "Analyse Score" tab (inference with `aggregation=none`,
  `return_intermediates=True`, auto-writes Delta Lake via merge) and "Load Delta Lake"
  tab (`gr.FileExplorer` for `metadata.json` selection)
- **Module 2** — Analysis Results: aggregation dropdown (`list_strategies()`), cached
  post-hoc aggregation via `Aggregate!` button, `gr.DownloadButton` auto-updated CSV,
  `Save Delta Lake` button, editable predictions table, Verovio visual score tab
- **Module 3** — Edit-Conditioned Re-Inference: grayed out until Module 1a runs;
  tasks CSV override lives here

Key changes in `hybrid_predictor.py`:
- `HybridAnalysisPredictor.predict()` now forwards `return_intermediates=True`

Key changes in `delta_writer.py`:
- `_merge_or_create()` replaces `write_deltalake(mode="overwrite")`: standard Delta
  Lake merge (`when_matched_update_all` / `when_not_matched_insert_all` /
  `when_not_matched_by_source_delete`) + `vacuum(retention_hours=0)` after each merge

Also done: voter logic commented out; `_extract_graph_edges_from_score` replaced by
`_edges_from_pyg_data()` reading `intermediates["data"].edge_index_dict`; single log
textbox replaces all status fields; aggregation results cached in-memory per strategy
name; `_precompute_delta_dfs()` converts raw predictions to long-format DataFrames
once after inference. All 52 tests pass.

### Step 4a: Verovio Score Upload + Note Coloring — DONE

The Verovio Visual Score tab now has its own `gr.File` upload, auto-filled from
Module 1a's score file via `score_file.change()`. Users can upload an alternative
edition; if note counts differ, a warning is logged and rendering uses
`min(n_score, n_table)` notes.

**Note coloring interface**: the visual payload accepts `note_colors` (dict of note
index -> CSS color string). The JS applies colors via CSS custom property
`--agn-note-color` + class `.agn-colored`, so the active-note highlight (`.agn-active`
with `!important`) still overrides. Any future coloring scheme (e.g., aggregation
confidence heatmaps) can populate `note_colors` the same way.

**NCT coloring** ("Colour non-chord tones grey" checkbox): uses `tpc_in_label` and
`tpc_in_label_confidence` from the predictions table. Effective in-label score =
`P("True")` (confidence if argmax is "True", else 1-confidence). Color = linear
interpolation from lightgrey `rgb(211,211,211)` at score=0 to black `rgb(0,0,0)` at
score=1.

### Step 4b: FlexOHR-Based Complete RN Column — DONE

Replaced the hand-rolled `decode_roman_numeral()` (from `analysisgnn/utils/roman_decode.py`)
with FlexOHR's `OHR.from_()` + `.to_format('dcml')` for the "Complete RN" column in the
Gradio app. The old `_parse_inversion_value`, `_build_complete_rn_column` (old version),
and the `decode_roman_numeral` import were removed entirely.

Key changes in `examples/gradio_hybrid_analysis_app.py`:
- Imports: `flexohr.codecs.analysisgnn` (codec activation), `OHR`, `ChordQuality`,
  `Inversion`, `CollectionType`, `SD`, `build_key_context`, `infer_collection_type`,
  `build_key_context_from_row`
- `_derive_global_key(df, k=5)`: takes the first *k* tonic chords (romanNumeral
  `"I"` / `"i"`) in score order and lets their `localkey` values vote (case-insensitive
  pitch-class grouping, most frequent cased variant wins); avoids bias from extended
  middle sections whose key may outnumber the main key's tonic chords
- `_build_complete_rn_column(df, global_key)`: per-row FlexOHR OHR construction from
  five principal tasks (degree1, degree2, inversion, quality, localkey), rendered via
  `.to_format('dcml')` — localkey and tonkey mode read directly from prediction case
  (uppercase = major, lowercase = minor); tonicized key mode from `tonkey` when available
- Removed `_infer_key_mode`, `_apply_key_mode`, `_build_localkey_mode_map` — these
  inferred mode from romanNumeral counts instead of trusting the localkey/tonkey case
- `global_key` parameter threaded through `_build_complete_rn_spans`,
  `_build_graph_overlay_payload`, `_build_visual_payload`
- New Gradio `global_key_field` text field in Module 2, auto-populated on inference /
  Delta Lake load, editable by the user; passed as input to `run_aggregation` and
  `refresh_visual_tab`; returned as output from `run_full_inference`,
  `load_from_delta_lake`, and `run_edit_conditioned`

### Step 5: Top-k Roman Numeral Enumeration from Task Distributions

The core mechanism for aggregation: given per-task probability distributions for a
note (or group of notes), enumerate which Roman-numeral labels (OHRs) are possible,
rank them by composite likelihood, and return the top-k candidates.

#### FlexOHR API Available for This Step

The FlexOHR library (dev-installed from `flexohr_project/flexohr/`) provides the
construction, validation, and rendering backend. Key API:

**Construction:**
- `OHR.from_(quality, sd, inversion=inv, reference_ohr=ctx)` — builds a chord OHR
  from a `ChordQuality`, root `SD`, optional `Inversion`, and key-context OHR
- `ChordQuality.from_format(label, "analysisgnn")` — decodes model quality labels
- `Inversion.from_format(str(n), "analysisgnn")` — decodes model inversion integers
- `SD.from_string(label, collection_type=ct)` / `SD.from_int(n, collection_type=ct)`
  — decodes model degree labels (e.g., `"5"`, `"#4"`, `"-7"`)
- `build_key_context(global_key, local_key, tonicized_key=, tonicized_coll=)` — nested
  key-context OHR (global -> local [-> tonicized])
- `build_key_context_from_row(row, global_key)` — row-level helper in
  `flexohr.codecs.analysisgnn`
- `build_ohrs_from_dataframe(df, global_key)` — bulk OHR construction with confidence
  snaks, also in `flexohr.codecs.analysisgnn`

**Validation:**
- `InversionBassConsistency()` — validator checking bass/inversion/interval agreement;
  returns `ValidationResult(is_valid, messages)`. Validators compose via `&`, `|`, `~`.
- `validate_ohrs(df, ohrs)` — cross-checks resolved OHRs against redundant columns
  (tonkey, root, bass); returns DataFrame with `*_ok` columns

**Rendering:**
- `ohr.to_format('dcml')` — renders as DCML label string
  (e.g., `V65/V/I/G` = dominant 6/5 of V in I in G)

**Mutation / annotation:**
- `ohr.with_(reference=..., bass=SPC(...))` — returns modified copy
- `ohr.with_snak("chord_quality", confidence=0.91)` — attaches confidence metadata
- OHR is `@dataclass(frozen=True)` — hashable, equality-comparable, usable as dict keys

**Introspection:**
- `ohr.get_property("chord_quality")` → `ChordQuality` enum member
- `ohr.get_property("inversion")` → `Inversion` enum member
- `ohr.component("r")` → root Component
- `ohr.components("b", depth=1, tone_function=ToneFunction.root)` → iterator
- `ohr.resolve()` → new OHR with all relative values resolved to absolute

**Legality tables** (in `flexohr.harmony.chord_tables`):
- `CHORD_QUALITY_INTERVALS` — maps each `ChordQuality` to its interval structure
- `CHORD_QUALITY_TO_CLASS` — maps quality to `ChordClass` (triad, seventh, aug6, ...)
- `CHORD_CLASS_MAX_INVERSION` — `{triad: 2, seventh: 3, augmented_sixth: 3}`;
  classes not in this dict (dyad, suspended, unclassified) do not support inversion
- `INVERSION_TO_INDEX` / `INDEX_TO_INVERSION` — bidirectional int ↔ `Inversion` enum

#### AnalysisGNN Task Vocabulary Summary (RN-relevant tasks)

The model produces per-note probability distributions over these classes:

| Task | # Classes | Content |
|------|-----------|---------|
| `quality` | 15 | Verbose English labels (see mismatch below) |
| `inversion` | 4 | Integer 0–3 |
| `degree1` | 22 | `-1`..`-7`, `1`..`7`, `#1`..`#7`, `None` |
| `degree2` | 22 | Same as degree1 (NaN = no tonicization) |
| `localkey` | 50 | 28 major + 22 minor key names (case = mode) |
| `romanNumeral` | 185 | Simple RN labels (no slash), e.g., `I`, `viio7`, `bVI` |
| `root` | 38 | Pitch spellings + `None` |
| `bass` | 38 | Pitch spellings + `None` |
| `tonkey` | 50 | Same vocab as localkey |

**Task roles in enumeration:**
- **Core five** (define the OHR): `quality`, `degree1`, `inversion`, `localkey`, `degree2`
- **Redundant/validating** (cross-check): `romanNumeral` (encodes root + chord type),
  `root`, `bass`, `tonkey`

#### Quality Vocabulary Mismatch

The FlexOHR `analysisgnn` codec registers 18 quality labels; the model predicts 15.

**Model-only** (no FlexOHR mapping yet — to be added to the codec):
- `"incomplete dominant-seventh chord"` — map to `dominant_seventh` or new enum member
- `"augmented sixth"` (generic) — map to `italian_sixth` or a generic aug6 member
- `"minor-augmented tetrachord"` — needs new enum member or mapping

**Codec-only** (model never predicts): `minor-major seventh chord`,
`augmented seventh chord`, `augmented major seventh chord`, `suspended second`,
`suspended fourth`, `power chord` — these exist in the FlexOHR enum but the
model has no class for them. No action needed.

**Decision:** Extend the FlexOHR `analysisgnn` codec to handle the 3 model-only labels.

#### Existing Usage Patterns

**Step 4b** (`_build_complete_rn_column` in the Gradio app) constructs one OHR per row
from the argmax predictions of the five core tasks. Localkey and tonkey mode are read
directly from prediction case (the 50-class vocabulary encodes mode via uppercase =
major, lowercase = minor). No enumeration, no top-k — pure argmax.

**The notebook** (`flexohr_project/flexohr/docs/notebooks/analysisgnn_tasks.py`) goes
further: section 6 validates OHRs against redundant columns (tonkey, root, bass) via
`validate_ohrs()`, and section 7 (`compare_alternatives`) builds alternative OHRs by
trusting contradicting columns, computes likelihoods as products of per-task
confidences, and picks the higher-likelihood interpretation. This is the prototype
for the enumeration logic.

#### Enumeration Design

**Strategy:** Cartesian product of top-k predictions per core task, filtered for
legality, scored by composite likelihood. The `romanNumeral` task is treated as an
additional evidence source for root + chord type (not as an enumeration anchor).

**Top-k per task (default k=3, configurable):** For each of the 5 core tasks, take
the top-k class labels with their probabilities. The cartesian product of 5 tasks at
k=3 yields at most 3^5 = 243 combinations before pruning. After legality filtering
(inversion ≤ max for quality class), this drops to roughly 100–150 valid combos.
After deduplication (different input combos producing the same DCML label), typically
fewer than 50 unique candidates remain. This is computationally trivial — sub-second
for a single note — and eliminates the need for threshold-based pruning.

**Why fixed top-k, not probability thresholds:** The 9 relevant tasks have
vocabulary sizes ranging from 4 (inversion) to 185 (romanNumeral). A task with 4
classes will typically have a top-1 probability of 0.7–0.95 and a top-2 of 0.1–0.25,
while a 185-class task might have top-1 at 0.1–0.5 and top-2 at 0.03–0.10. Any
absolute probability threshold (e.g., "only consider P > 0.1") would systematically
over-prune high-vocabulary tasks and under-prune low-vocabulary ones. A fixed k
avoids this entirely: the scoring step naturally ranks low-probability candidates
low unless other tasks corroborate them. Since the combinatorial space at k=3 is
already small, there is no computational pressure to prune more aggressively.

**Note on tonicization and task count:** Non-tonicized chords use `degree2 = "None"`
(class 21 in the 22-class vocabulary), which carries its own probability. Every
candidate therefore uses exactly 5 core tasks — the tonicization dimension does not
create a task-count asymmetry in the core score.

**Steps for a single note:**
1. For each of the 5 core tasks, extract top-k class labels with probabilities
   (from the `probabilities` Delta Lake table or in-memory distributions)
2. Enumerate all `(quality, degree1, inversion, localkey, degree2)` combinations
   from the cartesian product
3. **Prune illegal combinations:** inversion must be ≤ max for the quality's chord
   class (via `CHORD_QUALITY_TO_CLASS` + `CHORD_CLASS_MAX_INVERSION`)
4. **Construct OHR:** `OHR.from_(quality, sd, inversion=inv, reference_ohr=ctx)`
5. **Validate:** `InversionBassConsistency()(ohr)` — discard if invalid
6. **Score:** compute composite likelihood from per-task probabilities (core score)
7. **Cross-validate:** look up implied root, bass, tonkey from the resolved OHR;
   find these values in the corresponding task distributions and incorporate as
   additional evidence (validation score, reported separately)
8. **Deduplicate:** OHRs are hashable; group by DCML label string for human
   deduplication (different input combos may produce the same label)
9. **Rank and return top-k** candidates with composite likelihoods

**For a group of notes (aggregation):**
The per-note top-k candidate lists are combined across all notes in the group.
Candidates supported by multiple notes accumulate evidence; candidates contradicted
by other notes are penalized. The exact combination method (product, mean, vote, etc.)
is the V3 Aggregation Operator from the variable-parts table.

#### Composite Likelihood: The Task-Count Bias Problem

The core score (from the 5 core tasks) is always composed from the same number of
factors, so it is directly comparable across candidates. The bias problem arises when
incorporating the **redundant/validating tasks** (romanNumeral, root, bass, tonkey).

**The problem:** A candidate that finds corroboration from all 4 redundant tasks has
its score multiplied by 4 additional factors (each ≤ 1), while a candidate matched by
only 2 redundant tasks has fewer multiplicands. Under a raw product, the candidate
with *more* supporting evidence can receive a *lower* composite score. For example,
if `root` and `romanNumeral` both support a given degree1 prediction, and `bass` and
`tonkey` further confirm the candidate, the 4-factor product may be lower than a
2-factor product for a different candidate that simply wasn't checkable against bass
or tonkey.

**Vocabulary-size disparity in scores:** Even within the core 5, the tasks have very
different numbers of classes (inversion: 4, quality: 15, degree1/2: 22, localkey: 50).
A probability of 0.5 for a 4-class task is far less informative than 0.5 for a
50-class task (the uniform baseline is 0.25 vs. 0.02). This does not affect the
ranking of candidates that all use the same 5 tasks — the product is still
monotonically related to how well each task supports the candidate. But it matters
when **interpreting** scores and when **weighting** tasks: a task with a larger
vocabulary contributes more bits of information per probability unit and arguably
deserves more weight.

**Required:** A modular scoring framework that supports multiple approaches:
- **Simple product** — baseline, easy to inspect, biased by task count when
  incorporating redundant tasks
- **Log-probability sum** — numerically stable equivalent of product, same bias
- **Geometric mean** (= exp(log-sum / n)) — normalizes by number of contributing
  tasks, removing the task-count bias for redundant-task incorporation
- **Weighted product** — per-task weights allow tuning for vocabulary size and
  task reliability (e.g., weight romanNumeral higher because a match against 185
  classes is more informative)
- **Separate core vs. validation scores** — report the core-5 likelihood and the
  redundant-task support score independently, let the user decide how to combine;
  this cleanly avoids the task-count bias by keeping the two score dimensions
  orthogonal

The scoring function should be a pluggable component (V7 Certainty/Confidence Model
in the variable-parts table), and the number of contributing tasks should always be
reported alongside the score so that comparisons are interpretable.

#### What Needs to Be Built (in `analysisgnn/aggregation/`)

1. **`roman_numeral.py`** — the enumerator:
   - `enumerate_candidates(probs_df, global_key, top_k_per_task, threshold)` →
     list of `(OHR, score, metadata)` tuples
   - Legal-combination filter using FlexOHR chord tables
   - OHR construction via FlexOHR `OHR.from_()`
   - Validation via `InversionBassConsistency`
   - Cross-validation against redundant tasks

2. **`scoring.py`** — pluggable scoring functions:
   - `ProductScorer`, `LogSumScorer`, `GeometricMeanScorer`, `WeightedScorer`
   - Each takes a dict of `{task: probability}` and returns a composite score
   - Reports task count alongside score

3. **FlexOHR codec extension** — add the 3 model-only quality labels to the
   `analysisgnn` codec in `flexohr/codecs/analysisgnn.py`

4. **Integration with aggregation framework** — register as a new strategy in the
   existing `analysisgnn/aggregation/registry.py`, callable from the Gradio app

5. **Group-level aggregation** — extend single-note enumeration to note groups by
   combining per-note distributions before or after enumeration

---

## Verovio Integration Architecture

### Overview

The Gradio app embeds a Verovio-based score viewer in an iframe. Verovio renders
MusicXML into SVG; the app's JS then overlays graph edges, note coloring, RN labels,
and click-to-inspect interactivity on top of the rendered SVG.

### Version & CDN

Currently using **Verovio 5.0.0** via the CDN at
`https://www.verovio.org/javascript/5.0.0/verovio-toolkit-wasm.js`
(loaded in `examples/assets/verovio_score_graph.html`). To update, change the version
number in the `<script src>` tag; available versions are at
`https://www.verovio.org/javascript/<version>/` (not all releases have JS builds —
e.g., 6.1.1 was CocoaPods-only). **Do not upgrade to Verovio 6.x without adapting
the JS**: version 6.0.0 changed the SVG styling structure ("Remove default css
scoping"), which breaks the `.page-margin` selector used by our JS to anchor edge
overlays and RN labels.

### File layout

| File | Role |
|------|------|
| `examples/assets/verovio_score_graph.html` | Template: loads CSS, JS, payload, Verovio CDN script |
| `examples/assets/verovio_score_graph.js` | All rendering logic: Verovio init, note mapping, edge overlay, RN labels, click handler |
| `examples/assets/verovio_score_graph.css` | Styling: toolbar, note panel, `.agn-colored`/`.agn-active` note highlighting, edges, RN labels |

### Data flow

1. Python builds a **payload** dict (`_build_visual_payload()` in the Gradio app):
   - `score_xml`: the **original** MusicXML text (read directly from the uploaded
     file via `_read_score_xml_text()`; partitura re-export is only used as fallback
     for non-XML formats like `.mxl`)
   - `notes`: per-note metadata (pitch, onset, tasks, confidence, RN)
   - `edges`: graph edge lists per type
   - `meta.roman_spans`: RN label spans for the JS overlay
   - `note_colors`: optional per-note CSS color strings (e.g., NCT coloring)
2. The payload is JSON-serialized into the HTML template as `window.__AGN_PAYLOAD__`.
3. The HTML is escaped and embedded as an `<iframe srcdoc="...">`.
4. Inside the iframe, the JS:
   - Waits for the Verovio WASM runtime to initialize (`ensureVerovioReady()`)
   - Calls `tk.setOptions(...)` then `tk.loadData(payload.score_xml)`
   - Renders SVG pages via `tk.renderToSVG(page)`
   - Maps payload notes to SVG `<g class="note">` elements by ID (first by
     `findNoteElementById()` which tries exact match, endsWith, includes; then
     falls back to sequential order for unmapped notes)
   - Draws edge overlays as SVG `<path>` elements
   - Draws RN labels as SVG `<text>` elements from `meta.roman_spans`
   - Applies `note_colors` via CSS custom property `--agn-note-color`

### Important: always use the original MusicXML

The score XML sent to Verovio **must** be the original uploaded MusicXML, not a
partitura re-export. Partitura's `save_musicxml()` strips `<accidental>` elements
(it writes `<alter>` inside `<pitch>` but not the display-controlling `<accidental>`),
causing Verovio to render notes without accidental symbols. The original file
preserves both. RN annotations are handled entirely by the JS overlay (not embedded
in the MusicXML), so there is no reason to re-export.

### Verovio options (current)

```js
tk.setOptions({
  breaks: "none",        // horizontal continuous layout
  footer: "none",
  header: "none",
  adjustPageHeight: true,
  adjustPageWidth: true,
  pageMarginBottom: 0,
  pageMarginTop: 0,
});
```

Full option reference: https://book.verovio.org/toolkit-reference/toolkit-options.html

### Extending

- **Note coloring**: set `payload.note_colors = { "<noteIndex>": "<css-color>" }`;
  the JS applies class `.agn-colored` + CSS variable `--agn-note-color`
- **Edge visibility**: controlled by `payload.meta.visible_edge_types` (list of
  edge type strings); toggled via the Gradio checkboxes
- **New overlays**: add SVG elements to `pageMargin` containers in the JS
  (same pattern as `renderRomanNumeralOverlay()`)

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
