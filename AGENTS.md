# AGENTS.md — AnalysisGNN Aggregation Research

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
- **One table per task** for clean separation and manageable widths
- **Resolved human-readable column names** (not integer indices; whitespace replaced
  by underscores, e.g., `major_triad` instead of `major triad`)
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
├── probs_<task>/          # Delta table per task: one row per note
│   ├── _delta_log/        #   columns = resolved class labels (underscored)
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

#### `probs_<task>` tables (one per task)

One row per note (same length and order as `notes`). Columns are the **resolved
class labels** from the task's vocabulary, with whitespace replaced by underscores
(e.g., for `probs_quality`: `major_triad`, `minor_triad`, `diminished_triad`, etc.),
plus metadata columns.

| Column            | Type    | Description                              |
|-------------------|---------|------------------------------------------|
| `note_id`         | string  | Note ID (join key to `notes` table)      |
| `<class_label_0>` | float32 | Probability for class 0                  |
| `<class_label_1>` | float32 | Probability for class 1                  |
| ...               | ...     | ...                                      |
| `<class_label_k>` | float32 | Probability for class k                  |
| `argmax`          | string  | Resolved label of the most probable class|
| `confidence`      | float32 | max(probabilities) for this note         |
| `entropy`         | float32 | Shannon entropy of the distribution      |

The class label columns are derived from each task's `classList` (from
`available_representations` in `analysisgnn/utils/chord_representations.py`) or
`CadenceEncoder.accepted_cadences` for the cadence task. All labels are converted
to strings and whitespace is replaced with underscores to produce valid column names.

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
df = pq.read_table("outputs/score_id/probs_romanNumeral").to_pandas()
df.to_csv("romanNumeral_probs.csv", index=False)

# Or with argmax-only summary
notes_df = pq.read_table("outputs/score_id/notes").to_pandas()
for task in tasks:
    probs_df = pq.read_table(f"outputs/score_id/probs_{task}").to_pandas()
    notes_df[task] = probs_df["argmax"]
    notes_df[f"{task}_confidence"] = probs_df["confidence"]
notes_df.to_csv("notes_summary.csv", index=False)
```

### Dependencies to Add

```
deltalake>=0.22.0
pyarrow>=14.0.0
```

---

## Phase 1: Implementation Plan

### Step 1: Delta Lake Writer Module

Create `analysisgnn/storage/delta_writer.py`:

- `write_analysis_results(output_dir, score, predictions, data, task_dict, metadata)`
  - Writes `notes/`, `edges/`, `probs_<task>/` tables, `metadata.json`
  - Resolves class labels via `available_representations` / `CadenceEncoder`
  - Converts whitespace in class labels to underscores for column names
  - Computes entropy alongside argmax and confidence
  - Populates the `hyperedges` table with default groupings (`onset`, `beat`, `measure`)
    derived from the graph's onset_div, beat cluster, and measure cluster attributes

### Step 2: Delta Lake Reader Module

Create `analysisgnn/storage/delta_reader.py`:

- `load_notes(output_dir) -> pd.DataFrame`
- `load_edges(output_dir, edge_types=None) -> pd.DataFrame`
- `load_task_probs(output_dir, task) -> pd.DataFrame`
- `load_groups(output_dir, edge_type=None) -> pd.DataFrame`
- `list_group_types(output_dir) -> List[str]`
- `reconstruct_pyg_graph(output_dir, edge_types=None) -> HeteroData`
- `reconstruct_rustworkx_graph(output_dir, edge_types=None) -> rx.PyDiGraph`
- `list_aggregation_tables(output_dir) -> List[str]`
- `export_table_to_csv(output_dir, table_name, csv_path)`

### Step 3: Aggregation Runtime Refactoring

Extend the existing aggregation runtime in `analysisgnn/models/analysis.py` so that:

1. **No aggregation by default**: `aggregation_mode` defaults to `"none"` instead of
   `"mean"`. The raw per-note softmax outputs are returned unchanged unless an
   aggregation is explicitly requested.

2. **`"mean"` becomes optional**: The current onset → beat → measure mean pipeline is
   still available under the name `"mean"` but is no longer the default.

3. **Named registry**: Aggregation strategies are registered by keyword name. The
   existing `"mean"` and `"voter"` modes are the first two entries. New strategies
   can be added by registering a name and a callable. The `aggregation_spec` dict
   gains a `"mode"` key that can be any registered name (not just `"mean"` or
   `"voter"`).

4. **Backward compatibility**: Training and test steps that rely on `"mean"` continue
   to work. The CLI `--aggregation_mode` flag accepts the expanded set of registered
   names.

The integration point in `predict()` (line 5173) is adapted so that:
- `note_predictions` (raw softmax) is always captured
- Aggregation is applied only if `aggregation_spec["mode"] != "none"`
- When Delta Lake output is enabled, raw probabilities are always written; aggregated
  probabilities are written only if aggregation was applied

### Step 4: Gradio Integration

In `examples/gradio_hybrid_analysis_app.py`:

- Add an output directory selector or auto-generate path from score name
- After inference, write Delta Lake automatically
- Add a "Browse Delta Lake" section: list tables, inspect contents, export to CSV
- Aggregation experiments: load raw probs from Delta Lake, apply new aggregation,
  write result as new `agg_*` table
- **Verovio note colouring**: Support flexible per-note colouring in the visual score
  display, e.g., to show homogeneous regions (where nearby notes' predictions agree)
  in green shades and heterogeneous/contradictory regions in red shades. The colouring
  data can be computed from prediction agreement metrics and passed in the visual
  payload.
- The aggregation mode dropdown is extended to include `"none"` and any registered
  strategies.

### Step 5: Aggregation Experimentation Framework

Create `analysisgnn/aggregation/` package, integrating with the existing aggregation
runtime:

- `base.py`: Abstract `AggregationStrategy` interface, compatible with the signature
  expected by `_aggregate_note_probs()` (accepts `note_prob_dict`, `data`,
  `batch_size`, returns `Dict[str, torch.Tensor]`)
- `registry.py`: Named registry mapping strategy keywords to callables. Pre-registers
  `"none"` (passthrough), `"mean"` (current `_aggregate_note_probs` with mean mode),
  and `"voter"` (current voter bundle path). `_resolve_aggregation_runtime()` is
  updated to look up strategies from this registry.
- `mean.py`: Current mean aggregation, refactored from `analysis.py` module-level
  functions into the strategy interface
- `roman_numeral.py`: Legal RN enumeration from task distributions (future)
- Each strategy can read from Delta Lake (raw probs + graph) and write results back
  as a new `agg_*` table

---

## Conventions

- Always say **note**, never "vertex" or "node" — since groupings are hyperedges (not
  hypernodes), there is no ambiguity
- Tables are named `probs_<task>` for raw distributions, `agg_<method>_<task>` for
  aggregated results, `hyperedges` for group memberships (all grouping types)
- All per-note tables share the same row ordering (by `onset_div, pitch` sort)
  and are joined via `note_id`
- Class label columns use the **resolved human-readable names** from the task vocabulary,
  with whitespace replaced by underscores (e.g., `major_triad`, not `major triad`)
- Delta Lake versioning tracks the evolution of aggregation experiments within a run
