# PROMPT: Phase 1, Step 4 — Gradio GUI Restructuring + Delta Lake Integration

## Goal

Restructure the Gradio app into three clearly separated modules with proper data flow,
integrate Delta Lake persistence, and enable post-hoc re-aggregation without re-running
inference.  Replace the redundant `_extract_graph_edges_from_score` call with data from
`intermediates["data"]`.

## Current App Structure (flat, to be replaced)

The app (`examples/gradio_hybrid_analysis_app.py`, ~1060 lines) currently has a flat
layout with 28 widgets jumbled together: score upload, checkpoints, task selection,
inference mode, iterative refinement, aggregation mode, voter path, edit-conditioned
controls, predictions table, and Verovio — all stacked vertically with minimal
separation.  See `AGENTS.md` Step 4 description for context.

## New Layout

Three modules, top to bottom.  Each module is visually separated with a header.

### Module 1: Data Source (tabs)

Two tabs: **"Analyse Score"** (1a) and **"Load Delta Lake"** (1b).

**Tab 1a — Analyse Score:**
- Checkpoint paths (`full_ckpt`, `masked_ckpt`), device dropdown
- Score file upload
- Task selection (checkbox group + CSV override)
- "Enable Iterative Refinement" checkbox + steps/percentile controls
  (the old `mode_selector` dropdown is **removed** — the checkbox suffices to
  distinguish Full vs Iterative)
- **"Run Inference"** button (green) — always runs with `aggregation_spec={"mode":
  "none"}` and `return_intermediates=True`.  Stores the raw predictions, score,
  note_array, and data in `gr.State`.  Writes Delta Lake to
  `outputs/<score_id>/`.  Populates Module 2.

**Tab 1b — Load Delta Lake:**
- A path textbox or directory browser to select an existing Delta Lake output dir
- A **"Load"** button that reads notes, probabilities, hyperedges, and metadata
  from the Delta Lake and populates Module 2

In both cases, after the button click, Module 2 becomes active and shows the results.

### Module 2: Analysis Results

Sits **below** Module 1.  Has a header bar with:
- Aggregation strategy dropdown (populated from `list_strategies()`, default `"None"`)
- **"Aggregate!"** button — re-applies the selected strategy to the raw probabilities
  (stored in `gr.State` from Module 1) and updates both the table and Verovio views
- **"Export CSV"** button — saves the current table to a user-specified path
- **"Save Delta Lake"** button — writes/updates the Delta Lake, including any
  aggregation results generated during this session (cached in state)

Below the header bar: two tabs, **"Analysis Results"** (renamed from "Inference &
Edits") and **"Verovio Visual Score"**.

**Tab: Analysis Results**
- Editable `gr.Dataframe` ("Predictions") — shows the current aggregation result
  (or raw predictions if aggregation is `"None"`)
- Status textbox

**Tab: Verovio Visual Score**
- Edge type visibility checkboxes
- **"Refresh Visual"** button (for Verovio-specific display settings like edges;
  the prediction data shown on note-click updates automatically with aggregation)
- Verovio HTML iframe
- Visual status textbox

**Aggregation data flow**: Whenever the user clicks "Aggregate!", the selected
strategy's `aggregate()` runs on the stored raw probabilities.  The resulting
DataFrame updates the predictions table immediately.  The Verovio note-click payload
also updates (so clicking a note shows the aggregated values).  If the Verovio tab
has been opened, the rendered score reflects the new data; if not, it renders on
first visit.  The "Refresh Visual" button only re-renders Verovio display settings
(edge types, etc.), not the underlying data.

### Module 3: Edit-Conditioned Re-Inference

Sits **below** Module 2.  **Grayed out** (all widgets `interactive=False`) until
Module 1a has been used (i.e., inference has been run — not when loading from Delta
Lake).

Contains:
- "Target-only overwrite (partial mode)" checkbox
- "Known Rows" and "Target Rows" textboxes
- **"Update Analysis"** button (big, orange) — runs edit-conditioned masked inference
  using the edits made in the Module 2 table.  Updates Module 2 with the new
  predictions.

### Bottom: Diagnostic Output

Below Module 3:
- "Show Iteration Trace" checkbox
- Iteration trace textbox (used by both iterative inference and edit-conditioned
  updates)

### Voter Checkpoint Path

**Comment out** the `voter_checkpoint_path` textbox and all voter-related logic.
The voter aggregation is not implemented in the new `analysisgnn/aggregation/`
package yet.

## Key Implementation Changes

### 1. Capture intermediates from inference

`HybridAnalysisPredictor.predict()` needs to support `return_intermediates=True`.
Currently it does not forward this to `model.predict()`.  Add the passthrough so
that `run_full_inference()` receives `intermediates` containing `score`, `note_array`,
and `data`.

Store these in `gr.State` objects:
- `raw_predictions_state`: the raw `Dict[str, torch.Tensor]` (always `"none"` mode)
- `intermediates_state`: `{"score": ..., "note_array": ..., "data": ...}`

### 2. Replace `_extract_graph_edges_from_score`

The function at line 385 re-creates the score graph independently to extract edge
indices.  This duplicates what `model.predict()` already does.  Replace it:

- `_build_graph_overlay_payload()` should take edges from
  `intermediates["data"].edge_index_dict` instead of calling
  `_extract_graph_edges_from_score(score, note_array)`
- This eliminates a redundant `create_score_graph` call and ensures the visual
  overlay matches exactly what the model used

When loading from Delta Lake (Module 1b), edges come from `delta_reader.load_edges()`.

### 3. Run inference always with `aggregation_spec={"mode": "none"}`

The "Run Inference" button always gets raw per-note probabilities (no aggregation).
Aggregation is applied post-hoc in Module 2 via the "Aggregate!" button.  This means:

- Remove `aggregation_mode` from inference controls (it moves to Module 2)
- `_build_aggregation_spec()` always returns `{"mode": "none"}` for inference
- The default aggregation in Module 2 is `"None"` (raw predictions)

### 4. Post-hoc aggregation in Module 2

When "Aggregate!" is clicked:

1. Read the selected strategy name from the dropdown
2. If raw Delta Lake data is available (from `gr.State`):
   a. Convert raw predictions to the long-format probabilities DataFrame
   b. Run `get_strategy(name).aggregate(probabilities, notes, hyperedges, metadata)`
   c. Apply `format_table_output()` and update the predictions table
3. Cache each aggregation result so that "Save Delta Lake" can persist all of them

### 5. Delta Lake writing

**On initial inference** (Module 1a "Run Inference"):
- Write the raw predictions, notes, edges, hyperedges to Delta Lake under
  `outputs/<score_id>/`
- Guard: only write if Delta Lake does not already exist

**On "Save Delta Lake"** (Module 2):
- Write/update the Delta Lake with any new aggregation results generated during the
  session

### 6. Loading from Delta Lake (Module 1b)

When "Load" is clicked:
- Read notes, probabilities, hyperedges, edges, metadata from the given path
- Store them in the same `gr.State` objects used by inference
- Populate the Module 2 table with the `"none"` aggregation (raw argmax)
- **Set Module 3 to disabled** (no model available for edit-conditioned inference)

## Files to Modify

| File | Action |
|------|--------|
| `examples/gradio_hybrid_analysis_app.py` | Major restructure: 3-module layout, Delta Lake integration, intermediates capture |
| `analysisgnn/inference/hybrid_predictor.py` | Add `return_intermediates` passthrough to `HybridAnalysisPredictor.predict()` |

Files that should NOT need changes: `delta_writer.py`, `delta_reader.py`,
`analysisgnn/aggregation/`, `verovio_score_graph.js`, `verovio_score_graph.css`.

## How to Verify

1. Launch the Gradio app, upload Mozart K.1, run inference → see raw predictions in
   Module 2, Delta Lake written to `outputs/Minuet_in_G_Major_K.1/`
2. Change aggregation to "Mean", click "Aggregate!" → table updates, Verovio
   note-click data updates
3. Switch back to "None" → raw predictions again
4. Click "Export CSV" → CSV matches what the table shows
5. In a new session, use "Load Delta Lake" tab → load the output dir → see results
   in Module 2, Module 3 is grayed out
6. All 52 tests still pass:
   ```
   conda run -n analysisgnn python -m pytest tests/test_delta_writer.py tests/test_delta_reader.py tests/test_aggregation_mean.py -v
   ```

## Reference

- Current Gradio app: `examples/gradio_hybrid_analysis_app.py` (~1060 lines)
- `HybridAnalysisPredictor.predict()`: `analysisgnn/inference/hybrid_predictor.py:379-414`
- `_extract_graph_edges_from_score()`: `examples/gradio_hybrid_analysis_app.py:385-438`
- `_build_graph_overlay_payload()`: `examples/gradio_hybrid_analysis_app.py:441-516`
- `run_full_inference()`: `examples/gradio_hybrid_analysis_app.py:603-675`
- `run_partial_rerender()`: `examples/gradio_hybrid_analysis_app.py:678-764`
- `build_demo()` (all widgets): `examples/gradio_hybrid_analysis_app.py:821-1051`
- Delta Lake writer: `analysisgnn/storage/delta_writer.py`
- Delta Lake reader: `analysisgnn/storage/delta_reader.py`
- Aggregation package: `analysisgnn/aggregation/`
