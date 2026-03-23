# PROMPT: Phase 1, Step 2 — Delta Lake Reader + Demo Notebook

## Goal

Create `analysisgnn/storage/delta_reader.py` — a reader module that loads Delta Lake
tables written by `write_analysis_results()` — and a Jupyter notebook in `notebooks/`
that demonstrates the full predict → store → load → inspect workflow using the
Mozart K.1 score.

## What to Build

### 1. Reader Module: `analysisgnn/storage/delta_reader.py`

Thin convenience wrappers around `deltalake.DeltaTable`. Each function takes
`output_dir` (the per-score directory, e.g., `outputs/Minuet_in_G_Major_K.1/`)
and returns a pandas DataFrame or dict.

```python
def load_notes(output_dir: str) -> pd.DataFrame
```
Load the `notes/` table. Returns a DataFrame with columns: `note_id`, `onset_div`,
`onset_beat`, `duration_div`, `duration_beat`, `pitch_midi`, `pitch_spelling`,
`staff`, `voice`, `measure`, `ts_beats`.

```python
def load_edges(output_dir: str, edge_types: Optional[List[str]] = None) -> pd.DataFrame
```
Load the `edges/` table. If `edge_types` is given (e.g., `["onset", "consecutive"]`),
filter to only those types. Returns columns: `src`, `dst`, `edge_type`.

```python
def load_probabilities(
    output_dir: str,
    task: Optional[str] = None,
    top_k: Optional[int] = None,
) -> pd.DataFrame
```
Load the `probabilities/` table. If `task` is given, filter to that task only.
If `top_k` is given, filter to rows with `rank <= top_k` (e.g., `top_k=3` for
top-3 predictions per note per task). Returns columns: `note_id`, `task`,
`class_id`, `class_label`, `probability`, `is_argmax`, `rank`.

```python
def load_hyperedges(
    output_dir: str,
    edge_type: Optional[str] = None,
) -> pd.DataFrame
```
Load the `hyperedges/` table. If `edge_type` is given (e.g., `"beat"`), filter
to that grouping type. Returns columns: `group_id`, `note_id`, `edge_type`,
`parent_group_id`.

```python
def load_metadata(output_dir: str) -> dict
```
Load `metadata.json` and return as a Python dict.

```python
def list_group_types(output_dir: str) -> List[str]
```
Return the distinct `edge_type` values present in the `hyperedges/` table.

```python
def list_tables(output_dir: str) -> List[str]
```
Return names of all Delta tables in the directory (by scanning for subdirectories
containing `_delta_log/`).

```python
def export_table_to_csv(output_dir: str, table_name: str, csv_path: str) -> str
```
Load a named table and write it to CSV. Returns the CSV path.

```python
def argmax_summary(output_dir: str, tasks: Optional[List[str]] = None) -> pd.DataFrame
```
Convenience: pivot the `probabilities` table to produce a wide-format summary where
each task becomes a column containing the argmax class_label, plus a `<task>_confidence`
column with the argmax probability. Joined with the notes table on `note_id`.

### 2. Tests: `tests/test_delta_reader.py`

Use the Delta Lake already written at `outputs/Minuet_in_G_Major_K.1/` (by the
Step 1 tests). No new inference needed — just test the reader functions:

- `test_load_notes` — correct shape, expected columns, note_ids unique
- `test_load_edges` — filtering by edge_type works
- `test_load_probabilities` — full load, filter by task, filter by top_k
- `test_load_probabilities_sums_to_one` — per (note, task) groups sum to 1.0
- `test_load_hyperedges` — full load, filter by edge_type
- `test_load_metadata` — expected keys, task_dict matches
- `test_list_group_types` — returns `["onset", "beat", "measure"]` or subset
- `test_list_tables` — returns at least `["notes", "edges", "probabilities", "hyperedges"]`
- `test_export_csv` — exports to a temp CSV, reads back, matches
- `test_argmax_summary` — correct shape, one row per note, task columns present

**Prerequisite**: These tests depend on `outputs/Minuet_in_G_Major_K.1/` existing.
If it does not exist, skip the tests with `pytest.mark.skipif`. Or alternatively,
use a session-scoped fixture that calls the writer first (reusing the Step 1 fixture
pattern).

### 3. Jupyter Notebook: `notebooks/delta_lake_demo.ipynb`

A demonstration notebook that walks through the full workflow. Structure:

**Cell 1: Setup**
```python
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath("")), ".."))
# or however the project root is best added for notebook context
```

**Cell 2: Run inference (with no aggregation)**
```python
from analysisgnn.models.analysis import ContinualAnalysisGNN
model = ContinualAnalysisGNN.load_from_checkpoint(CHECKPOINT_PATH, map_location="cpu", strict=False)
model.eval()
predictions, intermediates = model.predict(SCORE_PATH, aggregation_spec={"mode": "none"}, return_intermediates=True)
```

**Cell 3: Write Delta Lake**
```python
from analysisgnn.storage.delta_writer import write_analysis_results
output_dir = write_analysis_results(
    output_dir="outputs/Minuet_in_G_Major_K.1",
    score=intermediates["score"],
    note_array=intermediates["note_array"],
    predictions=predictions,
    data=intermediates["data"],
    task_dict=model.task_dict,
    metadata={"score_path": SCORE_PATH, "full_checkpoint": CHECKPOINT_PATH, "device": "cpu"},
)
```

**Cell 4: Load and inspect notes**
```python
from analysisgnn.storage.delta_reader import load_notes
notes = load_notes(output_dir)
notes.head(10)
```

**Cell 5: Inspect edges**
```python
from analysisgnn.storage.delta_reader import load_edges
edges = load_edges(output_dir)
edges.groupby("edge_type").size()
```

**Cell 6: Query top-3 predictions for romanNumeral**
```python
from analysisgnn.storage.delta_reader import load_probabilities
top3_rn = load_probabilities(output_dir, task="romanNumeral", top_k=3)
top3_rn.head(15)  # Shows top-3 candidates for the first 5 notes
```

**Cell 7: Argmax summary (wide format)**
```python
from analysisgnn.storage.delta_reader import argmax_summary
summary = argmax_summary(output_dir, tasks=["romanNumeral", "localkey", "quality", "cadence"])
summary.head(10)
```

**Cell 8: Inspect hyperedge groupings**
```python
from analysisgnn.storage.delta_reader import load_hyperedges, list_group_types
print("Group types:", list_group_types(output_dir))
beat_groups = load_hyperedges(output_dir, edge_type="beat")
beat_groups.groupby("group_id").size().describe()
```

**Cell 9: Metadata**
```python
from analysisgnn.storage.delta_reader import load_metadata
meta = load_metadata(output_dir)
print(f"Score: {meta['score_id']}, {meta['num_notes']} notes, {len(meta['task_dict'])} tasks")
```

**Cell 10: CSV export**
```python
from analysisgnn.storage.delta_reader import export_table_to_csv
csv_path = export_table_to_csv(output_dir, "notes", "/tmp/notes.csv")
print(f"Exported to {csv_path}")
```

The notebook should use **relative paths** (relative to the repo root) for the score
and checkpoint, so it works when run from the `notebooks/` directory.

## Files to Create/Modify

| File | Action |
|------|--------|
| `analysisgnn/storage/delta_reader.py` | Create |
| `tests/test_delta_reader.py` | Create |
| `notebooks/delta_lake_demo.ipynb` | Create |

No modifications to existing files needed.

## How to Verify

1. `conda run -n analysisgnn python -m pytest tests/test_delta_reader.py -v` — all pass
2. Open the notebook in Jupyter and run all cells — no errors, output tables display
   correctly
3. The CSV export produces a valid file

## Reference

- Writer: `analysisgnn/storage/delta_writer.py`
- Existing Delta Lake output: `outputs/Minuet_in_G_Major_K.1/`
- Score: `notebooks/Minuet_in_G_Major_K.1.musicxml`
- Checkpoint: `artifacts/gradio_checkpoints/uocj8f6y_full_last.ckpt`
- AGENTS.md schema docs: lines 228-362 (table schemas + graph reconstruction)
