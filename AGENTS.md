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

Additional "hyperedge" node types (`beat`, `measure`) group notes into metrical
positions via `connects` edges. These are optional and can be regenerated with different
parameters (e.g., different beat granularity or measure grouping).

### Training Data Background

The training data are expert-labelled scores using Roman-numeral analysis (DCML
standard: keys, modulations, phrase boundaries, cadences; RomanText: keys + Roman
numerals only). For GNN training, these segment-level labels were **decomposed** into
atomic per-note, per-task labels. The research goal is essentially the **inverse
operation**: reconstructing musically sensible, non-contradictory Roman-numeral
analyses from atomic note-task probability distributions.

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
- **Resolved human-readable column names** (not integer indices)
- **Organic growth**: new aggregation tables are added to the same Delta Lake over time

Additionally, **CSV export** is available as a user-triggered convenience from any
Delta Lake table (e.g., for inspection, sharing, or git-diffable summaries).

### Delta Lake Directory Layout

```
outputs/<score_id>/
├── vertices/              # Delta table: one row per note
│   ├── _delta_log/
│   └── *.parquet
├── edges/                 # Delta table: one row per edge
│   ├── _delta_log/
│   └── *.parquet
├── probs_<task>/          # Delta table per task: one row per note
│   ├── _delta_log/        #   columns = resolved class labels
│   └── *.parquet
├── agg_<method>_<task>/   # Aggregation result tables (added over time)
│   ├── _delta_log/
│   └── *.parquet
└── metadata.json          # Score ID, checkpoint paths, inference params,
                           #   task_dict, class vocabularies, creation timestamp
```

### Table Schemas

#### `vertices` table

One row per note, sorted by `(onset_div, pitch)` to match graph node indices.

| Column           | Type    | Description                              |
|------------------|---------|------------------------------------------|
| `vertex_idx`     | int     | 0-based graph vertex index (= row index) |
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

One row per directed edge.

| Column      | Type   | Description                                      |
|-------------|--------|--------------------------------------------------|
| `src_idx`   | int    | Source vertex index                               |
| `dst_idx`   | int    | Destination vertex index                          |
| `edge_type` | string | One of: `onset`, `consecutive`, `during`, `rest`  |

Optional beat/measure hyperedges (stored with different edge_type values):

| `edge_type`          | Description                          |
|----------------------|--------------------------------------|
| `beat_connects`      | Beat hypernode → note                |
| `measure_connects`   | Measure hypernode → note             |

When beat/measure hyperedges are included, the `src_idx` values reference virtual
beat/measure node indices (offset above the note vertex range). A `hypernodes` table
may be added to describe them.

#### `probs_<task>` tables (one per task)

One row per note (same length and order as `vertices`). Columns are the **resolved
class labels** from the task's vocabulary (e.g., for `probs_quality`: `"major triad"`,
`"minor triad"`, `"diminished triad"`, etc.), plus metadata columns.

| Column            | Type    | Description                              |
|-------------------|---------|------------------------------------------|
| `vertex_idx`      | int     | Graph vertex index (join key)            |
| `<class_label_0>` | float32 | Probability for class 0                  |
| `<class_label_1>` | float32 | Probability for class 1                  |
| ...               | ...     | ...                                      |
| `<class_label_k>` | float32 | Probability for class k                  |
| `argmax`          | string  | Resolved label of the most probable class|
| `confidence`      | float32 | max(probabilities) for this note         |
| `entropy`         | float32 | Shannon entropy of the distribution      |

The class label columns are derived from each task's `classList` (from
`available_representations` in `analysisgnn/utils/chord_representations.py`) or
`CadenceEncoder.accepted_cadences` for the cadence task.

#### `agg_<method>_<task>` tables (added over time)

Same schema as `probs_<task>` but with aggregated probability distributions.
Additional columns may include:

| Column          | Type   | Description                                   |
|-----------------|--------|-----------------------------------------------|
| `group_id`      | int    | Which aggregation group this note belongs to   |
| `group_size`    | int    | Number of notes in this group                  |
| `aggregation`   | string | Name of the aggregation method used            |

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
    ...
  },
  "edge_types_included": ["onset", "consecutive", "during", "rest"],
  "beat_edges_included": false,
  "measure_edges_included": false,
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

vertices_df = pq.read_table("outputs/score_id/vertices").to_pandas()
edges_df = pq.read_table("outputs/score_id/edges").to_pandas()

data = HeteroData()
data["note"].x = ...  # reconstruct features from vertices_df columns
data["note"].onset_div = torch.tensor(vertices_df["onset_div"].values)
# ... other note attributes

for etype in ["onset", "consecutive", "during", "rest"]:
    mask = edges_df["edge_type"] == etype
    src = edges_df.loc[mask, "src_idx"].values
    dst = edges_df.loc[mask, "dst_idx"].values
    data["note", etype, "note"].edge_index = torch.tensor([src, dst])
```

#### rustworkx

```python
import rustworkx as rx

graph = rx.PyDiGraph()
n = len(vertices_df)
vertex_indices = graph.add_nodes_from(range(n))

for _, row in edges_df.iterrows():
    graph.add_edge(row["src_idx"], row["dst_idx"], row["edge_type"])
```

### CSV Export (User-Triggered)

Any Delta Lake table can be exported to CSV on demand:

```python
# From the Gradio app or CLI
df = pq.read_table("outputs/score_id/probs_romanNumeral").to_pandas()
df.to_csv("romanNumeral_probs.csv", index=False)

# Or with argmax-only summary
vertices_df = ...
for task in tasks:
    probs_df = pq.read_table(f"outputs/score_id/probs_{task}").to_pandas()
    vertices_df[task] = probs_df["argmax"]
    vertices_df[f"{task}_confidence"] = probs_df["confidence"]
vertices_df.to_csv("vertices_summary.csv", index=False)
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
  - Writes `vertices/`, `edges/`, `probs_<task>/` tables, `metadata.json`
  - Resolves class labels via `available_representations` / `CadenceEncoder`
  - Computes entropy alongside argmax and confidence
  - Optionally includes beat/measure hyperedges

### Step 2: Delta Lake Reader Module

Create `analysisgnn/storage/delta_reader.py`:

- `load_vertices(output_dir) -> pd.DataFrame`
- `load_edges(output_dir, edge_types=None) -> pd.DataFrame`
- `load_task_probs(output_dir, task) -> pd.DataFrame`
- `reconstruct_pyg_graph(output_dir, edge_types=None) -> HeteroData`
- `reconstruct_rustworkx_graph(output_dir, edge_types=None) -> rx.PyDiGraph`
- `list_aggregation_tables(output_dir) -> List[str]`
- `export_table_to_csv(output_dir, table_name, csv_path)`

### Step 3: Integration Point in predict()

In `analysisgnn/models/analysis.py`, the `predict()` method (line 5032+):

- **Before** aggregation (line 5173): capture `note_predictions` (raw softmax outputs)
- **After** aggregation (line 5174): capture `predictions` (aggregated)
- Add an option `save_delta=True` to write both raw and aggregated results
- The `data` (HeteroData) object is available for edge extraction

In `analysisgnn/inference/hybrid_predictor.py`:

- Thread `save_delta` / `output_dir` through `HybridAnalysisPredictor.predict()`

### Step 4: Gradio Integration

In `examples/gradio_hybrid_analysis_app.py`:

- Add an output directory selector or auto-generate path from score name
- After inference, write Delta Lake automatically
- Add a "Browse Delta Lake" section: list tables, inspect contents, export to CSV
- Aggregation experiments: load raw probs from Delta Lake, apply new aggregation,
  write result as new `agg_*` table

### Step 5: Aggregation Experimentation Framework

Create `analysisgnn/aggregation/` package:

- `base.py`: Abstract `AggregationStrategy` interface
- `mean.py`: Current mean aggregation, refactored
- `roman_numeral.py`: Legal RN enumeration from task distributions
- `registry.py`: Register and discover aggregation strategies
- Each strategy reads from Delta Lake (raw probs + graph), writes results back as a
  new `agg_*` table

---

## Conventions

- **Vertex** (not "node") to avoid confusion with musical notes
- Tables are named `probs_<task>` for raw distributions, `agg_<method>_<task>` for
  aggregated results
- All tables share the same row ordering (by `vertex_idx` = `onset_div, pitch` sort)
- Class label columns use the **resolved human-readable names** from the task vocabulary
- Delta Lake versioning tracks the evolution of aggregation experiments within a run
