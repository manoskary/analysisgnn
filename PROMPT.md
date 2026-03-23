# PROMPT: Phase 1 — Delta Lake Storage for Analysis Results

## Step 1: Delta Lake Writer — COMPLETED

Created `analysisgnn/storage/delta_writer.py` with `write_analysis_results()`.

### What was built

1. **`analysisgnn/storage/__init__.py`** — package scaffold
2. **`analysisgnn/storage/delta_writer.py`** — main writer module
   - `write_analysis_results()` — writes 4 Delta tables + metadata.json
   - `resolve_task_vocabulary()` — resolves class labels for all 21 tasks through
     a 7-step resolution chain (available_representations -> aliases -> cadence ->
     NoteDegree49 -> binary tasks -> integer tasks -> fallback)
3. **`requirements.txt`** — added `deltalake>=0.22.0`, `pyarrow>=14.0.0`
4. **`tests/test_delta_writer.py`** — 25 integration tests using real Mozart K.1
   inference, all passing
5. **`analysisgnn/models/analysis.py`** — two changes:
   - `predict()` gains `return_intermediates=True` flag (returns score, note_array,
     data alongside predictions)
   - `_resolve_aggregation_runtime()` and `_aggregate_note_probs()` support
     `"none"` mode to skip aggregation entirely

### Design decisions made

- **Single long-format `probabilities` table** instead of multiple `probs_<task>/`
  wide tables. One row per (note, task, class) combination. Columns: `note_id`,
  `task`, `class_id`, `class_label`, `probability`, `is_argmax`, `rank`.
- **`class_label` can be empty string** (e.g., cadence class 0 = no cadence)
  but is never null/NaN.
- **Class vocabularies are ordered** — `class_vocabularies[task][i]` resolves
  `class_id = i` correctly.
- Output goes to `outputs/<score_name>/` (e.g., `outputs/Minuet_in_G_Major_K.1/`).

### Output structure (verified)

```
outputs/Minuet_in_G_Major_K.1/
├── notes/           (256 rows — one per note)
├── edges/           (1094 edges: 528 onset, 519 consecutive, 47 during)
├── probabilities/   (N * T * C rows in long format)
├── hyperedges/      (onset, beat, measure groupings)
└── metadata.json    (task_dict, class_vocabularies, edge/note counts)
```

---

## Step 2: Delta Lake Reader — NEXT

Create `analysisgnn/storage/delta_reader.py` with functions to load tables back
and reconstruct the graph. Then build the Jupyter notebook that demonstrates
predict -> store -> load -> inspect.

### What to build

1. **`analysisgnn/storage/delta_reader.py`**:
   - `load_notes(output_dir) -> pd.DataFrame`
   - `load_edges(output_dir, edge_types=None) -> pd.DataFrame`
   - `load_probabilities(output_dir, task=None) -> pd.DataFrame`
   - `load_hyperedges(output_dir, edge_type=None) -> pd.DataFrame`
   - `load_metadata(output_dir) -> dict`
   - `list_group_types(output_dir) -> List[str]`
   - `export_table_to_csv(output_dir, table_name, csv_path)`

2. **`notebooks/delta_lake_demo.ipynb`** — Jupyter notebook that:
   - Loads the Mozart K.1 score
   - Runs inference with no aggregation
   - Writes Delta Lake to `outputs/Minuet_in_G_Major_K.1/`
   - Reads back each table and inspects contents
   - Shows how to query: e.g., "top-3 predictions per note for romanNumeral"
   - Shows the argmax summary (pivot from long to wide)
   - Inspects the hyperedges groupings

3. **Tests** for the reader functions in `tests/test_delta_reader.py`
