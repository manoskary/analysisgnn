# PROMPT: Phase 1, Step 3 — Reference CSVs + Aggregation Package

## Goal

Produce ground-truth reference CSVs for two aggregation modes (`none` and `mean`)
using the existing inference pipeline on Mozart K.1, then create a new
`analysisgnn/aggregation/` package whose `mean` strategy reproduces the exact same
values from the pre-aggregation Delta Lake. Along the way, unify the label decoding
across the codebase.

## Context

### Two decoding systems exist today

1. **Gradio / `_decode_task_predictions()`** in
   `analysisgnn/inference/hybrid_predictor.py:248-295` — used for display. It decodes
   argmax class IDs into human-readable labels via `available_representations[task].decode()`
   or `CadenceEncoder.decode()`, falling back to raw integer IDs for tasks it doesn't
   recognize.

2. **Delta Lake / `resolve_task_vocabulary()`** in
   `analysisgnn/storage/delta_writer.py:44-103` — used when writing the `class_label`
   column. It uses a richer resolution chain (aliases, binary task vocabularies,
   `NoteDegree49`, integer-label tasks, fallback).

The Delta Lake approach is more complete but has one legacy artefact: it was designed
for wide-format tables where labels become column names, so spaces were replaced by
underscores (e.g., `major_triad`). The Gradio approach uses the original `decode()`
output, which preserves spaces (e.g., `major triad`).

### Known differences (per task)

| Task            | Gradio output                                                                                      | Delta output                           | Issue                                                                                                       |
|-----------------|----------------------------------------------------------------------------------------------------|----------------------------------------|-------------------------------------------------------------------------------------------------------------|
| `quality`       | `"major triad"` (space)                                                                            | `"major triad"` (space)                | Actually same — Delta uses `_class_list_to_strings(classList)` which calls `str()`, preserving the original |
| `hrythm`        | Raw int `0`/`1`                                                                                    | `"True"`/`"False"`                     | Gradio has no alias `hrythm`→`hrhythm`; falls through to int                                                |
| `section`       | Raw int `0`/`1`                                                                                    | `"no_section_start"`/`"section_start"` | Not in `available_representations`; Delta has `_BINARY_TASK_VOCABULARIES`                                   |
| `phrase`        | Raw int `0`/`1`                                                                                    | `"no_boundary"`/`"phrase_end"`         | Same                                                                                                        |
| `organ_point`   | Raw int `0`/`1`                                                                                    | `"no_pedal"`/`"pedal_present"`         | Same                                                                                                        |
| `tpc_in_label`  | Raw int `0`/`1` → Gradio display remaps to `"NCT"`/`"Chord Tone"`                                  | `"NCT"`/`"chord_tone"`                 | Three-way inconsistency                                                                                     |
| `tpc_is_root`   | Raw int `0`/`1`                                                                                    | `"not_root"`/`"is_root"`               | Same pattern                                                                                                |
| `tpc_is_bass`   | Raw int `0`/`1`                                                                                    | `"not_bass"`/`"is_bass"`               | Same pattern                                                                                                |
| `note_degree`   | Raw int `0`..`48`                                                                                  | `"bbb1"`, `"bb1"`, ..., `"###7"`       | Not in `available_representations`; Delta has `_EXTRA_REPRESENTATION_CLASSES`                               |
| `inversion`     | `int` (0,1,2,3)                                                                                    | `str` ("0","1","2","3")                | Type difference only                                                                                        |
| `downbeat`      | `int`                                                                                              | `str`                                  | Type difference only                                                                                        |
| `staff`         | `int`                                                                                              | `str`                                  | Type difference only                                                                                        |
| `romanNumeral`  | Works for class_ids 0..183; crashes for 184 (`classList` has 184 entries due to missing-comma bug) | Pads with `"class_184"` fallback       | Delta is more robust                                                                                        |
| Everything else | Identical                                                                                          | Identical                              |                                                                                                             |

### The `romanNumeral` missing-comma bug

`SIMPLE_NUMERAL_VOCABULARY` in `analysisgnn/utils/globals.py` has a missing comma at
lines 2351-2354, concatenating `'#VII'` and `'bvio7'` into `'#VIIbvio7'`, producing
184 items instead of 185. This means `SimpleRomanNumeral185.classList` actually has 184
entries. When the model predicts class_id 184 (the 185th class), the Gradio decoder
crashes and falls back to all-integer labels. The Delta writer pads with `"class_184"`.

## What to Build

### 1. Unify Label Decoding

**Do NOT modify files yet.** First, present the proposed unified vocabulary for each
task (especially the binary tasks and `tpc_in_label`) so the user can review the
musicological correctness before any code changes. The key decisions to get approval
on:

- `tpc_in_label`: Currently three variants: int `0`/`1`, Delta `"NCT"`/`"chord_tone"`,
  Gradio display `"NCT"`/`"Chord Tone"`. Propose one canonical form.
- Binary tasks (`section`, `phrase`, `organ_point`, `tpc_is_root`, `tpc_is_bass`):
  Are the Delta labels (`"no_boundary"`/`"phrase_end"`, etc.) musicologically accurate?
- `hrythm`: The Delta label `"True"`/`"False"` comes from
  `HarmonicRhythm2.classList = [True, False]`. Is there a better label?
- `note_degree`: The Delta labels (`"bbb1"`, ..., `"###7"`) come from
  `NoteDegree49.classList`. Are these correct?
- `romanNumeral` class 184: Should this be `"class_184"`, or can we fix the
  missing-comma bug and restore the intended label?

After getting approval, update `_decode_task_predictions()` in `hybrid_predictor.py`
to use `resolve_task_vocabulary()` as its label source. This means:

- Remove the per-task `if task in available_representations: cls.decode(...)` chain
- Instead: `vocab = resolve_task_vocabulary(task, num_classes)` then
  `decoded = np.array([vocab[i] for i in class_ids])`
- Keep the confidence/argmax computation unchanged
- Remove the `_pcset_to_str` special case (handled by `_class_list_to_strings`)
- Keep cadence's empty-string-for-class-0 semantics

Also update `_convert_tpc_column_inplace()` in the Gradio app to match the unified
vocabulary (or remove it if the unified vocab already handles tpc_in_label correctly).

### 2. Generate Reference CSVs

Create a script `scripts/generate_reference_csvs.py` that:

1. Loads the model and runs inference on `notebooks/Minuet_in_G_Major_K.1.musicxml`
2. For each mode in `["none", "mean"]`:
   a. Calls `model.predict(score, aggregation_spec={"mode": mode}, return_intermediates=True)`
   b. Calls `predictions_to_dataframe(score, predictions, tasks=ALL_TASKS, include_confidence=True, include_class_ids=False)`
   c. Applies `_format_table_output(df, tasks)` for consistent column ordering
   d. Saves to `outputs/Minuet_in_G_Major_K.1/reference_<mode>.csv`

The CSVs have columns: `row`, `note_id`, `onset_beat`, `measure`, `duration_beat`,
`pitch_spelling`, `pitch_midi`, then for each of the 21 tasks: `<task>`, `<task>_confidence`.

### 3. Create `analysisgnn/aggregation/` Package

```
analysisgnn/aggregation/
├── __init__.py
├── base.py          # Abstract AggregationStrategy interface
├── registry.py      # Named strategy registry
└── mean.py          # Mean aggregation (onset → beat → measure)
```

#### `base.py`

```python
class AggregationStrategy(ABC):
    """Abstract base class for aggregation strategies."""

    @abstractmethod
    def aggregate(
        self,
        probabilities: pd.DataFrame,   # Long-format probabilities from Delta Lake
        notes: pd.DataFrame,            # Notes table from Delta Lake
        hyperedges: pd.DataFrame,       # Hyperedges table from Delta Lake
        metadata: dict,                 # metadata.json contents
        tasks: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Aggregate probabilities and return an argmax summary DataFrame.

        The returned DataFrame has one row per note and columns:
        - All columns from the notes table
        - For each task: <task> (argmax label) and <task>_confidence (max prob)

        This matches the format of the Gradio display DataFrame and the
        reference CSVs.
        """
        ...
```

The key insight: the new aggregation strategies operate on **Delta Lake data**
(pandas DataFrames), not on PyTorch tensors + PyG graphs. This decouples
aggregation experimentation from the model inference pipeline.

#### `mean.py`

Re-implements the existing onset → beat → measure mean pipeline, but operating on
the long-format `probabilities` DataFrame + `hyperedges` DataFrame rather than
PyTorch tensors.

The algorithm for each level (onset, beat, measure):

1. Load the group memberships from `hyperedges` (filtered by `edge_type`)
2. For each task in the level's task list:
   a. Join probabilities with group memberships on `note_id`
   b. For eligible notes (filtered by `tpc_in_label` argmax for RNA tasks at
      onset/beat level):
      - Compute the mean probability for each `(group_id, class_id)` combination
      - Broadcast the group mean back to all member notes
      - Replace the original probabilities with the group means
3. The onset level also performs change-point detection (as in
   `onsetwise_logit_aggregation` lines 420-452): after onset grouping, find
   contiguous spans of identical argmax predictions and broadcast each span's
   onset-group distribution to all notes in the span.

The level → tasks mapping (from `DEFAULT_TASKS_BY_LEVEL` in `posthoc_aggregator.py`):

```python
{
    "onset": ["cadence", "phrase", "root", "localkey", "quality",
              "inversion", "degree1", "degree2", "romanNumeral", "section"],
    "beat":  ["root", "localkey", "quality", "inversion",
              "degree1", "degree2", "romanNumeral",
              "cadence", "phrase", "section"],
    "measure": ["localkey"],
}
```

#### `registry.py`

```python
_STRATEGIES: Dict[str, Type[AggregationStrategy]] = {}

def register(name: str, cls: Type[AggregationStrategy]): ...
def get(name: str) -> Type[AggregationStrategy]: ...
def list_strategies() -> List[str]: ...
```

Pre-registers `"none"` (passthrough — returns `argmax_summary()` from
`delta_reader`) and `"mean"` (the new mean aggregation).

### 4. Validation Test

Create `tests/test_aggregation_mean.py`:

- Load the Delta Lake at `outputs/Minuet_in_G_Major_K.1/`
- Load the reference CSV `outputs/Minuet_in_G_Major_K.1/reference_mean.csv`
- Run the new `MeanAggregation` strategy on the Delta Lake data
- Assert that the resulting DataFrame matches the reference CSV exactly
  (for all 21 task columns and all 21 confidence columns, within float tolerance)

Similarly, test `"none"`:
- Run the `NoneAggregation` strategy
- Assert it matches `reference_none.csv`

Use `pytestmark = pytest.mark.skipif` if the output directory or reference CSVs
don't exist.

## Files to Create/Modify

| File | Action |
|------|--------|
| `analysisgnn/inference/hybrid_predictor.py` | Modify: update `_decode_task_predictions()` to use `resolve_task_vocabulary()` |
| `examples/gradio_hybrid_analysis_app.py` | Modify: update `_convert_tpc_column_inplace()` to match unified vocab |
| `scripts/generate_reference_csvs.py` | Create: script to generate reference CSVs |
| `analysisgnn/aggregation/__init__.py` | Create |
| `analysisgnn/aggregation/base.py` | Create |
| `analysisgnn/aggregation/registry.py` | Create |
| `analysisgnn/aggregation/mean.py` | Create |
| `tests/test_aggregation_mean.py` | Create |

Optional if the missing-comma bug should be fixed:
| `analysisgnn/utils/globals.py` | Modify: fix the missing comma in `SIMPLE_NUMERAL_VOCABULARY` |

## How to Verify

1. `conda run -n analysisgnn python scripts/generate_reference_csvs.py` — produces
   two CSVs without errors
2. `conda run -n analysisgnn python -m pytest tests/test_aggregation_mean.py -v` —
   all tests pass (new aggregation matches reference CSVs)
3. The Gradio app still works correctly (no display regressions)
4. `conda run -n analysisgnn python -m pytest tests/test_delta_writer.py tests/test_delta_reader.py -v` —
   existing tests still pass

## Reference

- Existing aggregation code: `analysisgnn/models/analysis.py` lines 296-540
  (`_groupwise_mean_broadcast`, `_aggregate_with_mode`, `onsetwise_logit_aggregation`,
  `beatwise_logit_aggregation`, `measurewise_logit_aggregation`,
  `_aggregate_note_probs`, `_resolve_aggregation_runtime`)
- `DEFAULT_TASKS_BY_LEVEL`: `analysisgnn/models/posthoc_aggregator.py` lines 11-37
- Delta Lake reader: `analysisgnn/storage/delta_reader.py`
- Delta Lake output: `outputs/Minuet_in_G_Major_K.1/`
- Score: `notebooks/Minuet_in_G_Major_K.1.musicxml`
- Checkpoint: `artifacts/gradio_checkpoints/uocj8f6y_full_last.ckpt`
- Gradio app: `examples/gradio_hybrid_analysis_app.py`
- `_decode_task_predictions`: `analysisgnn/inference/hybrid_predictor.py:248-295`
- `resolve_task_vocabulary`: `analysisgnn/storage/delta_writer.py:44-103`
- `CadenceEncoder`: `analysisgnn/utils/music.py:208-276`
- `available_representations`: `analysisgnn/utils/chord_representations.py:529-541`
- `NoteDegree49`: `analysisgnn/utils/chord_representations.py:489-491`
- `SIMPLE_NUMERAL_VOCABULARY` (missing-comma bug): `analysisgnn/utils/globals.py:2314-2359`
