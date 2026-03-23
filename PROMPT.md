# PROMPT: Phase 1, Step 5 — Aggregation Experimentation Framework

## Goal

Build out the `analysisgnn/aggregation/` package beyond `"none"` and `"mean"` with
concrete aggregation strategies that address the research goals: cross-task consistency
enforcement, Roman-numeral reconstruction, and dynamic groupings.  Each strategy must
work end-to-end in the Gradio app (select from dropdown, click Aggregate!, see results).

## Current State

- `analysisgnn/aggregation/` has `base.py` (ABC), `registry.py`, `mean.py`, `__init__.py`
- Two strategies registered: `"none"` (passthrough argmax) and `"mean"` (onset → beat →
  measure scatter_mean, matching the model's built-in pipeline)
- The Gradio app populates the aggregation dropdown from `list_strategies()` and runs
  `get_strategy(name).aggregate(probs_df, notes_df, hyperedges_df, metadata, tasks=tasks)`
- Aggregation results are cached in-memory per strategy name
- Raw predictions are precomputed into long-format DataFrames (`_precompute_delta_dfs()`)
  once after inference, stored in `delta_dfs_state`

## New Strategies to Implement

### 1. `roman_numeral.py` — Legal Roman Numeral Reconstruction

Given per-note probability distributions over atomic tasks (root, quality, inversion,
bass, degree1, degree2, localkey), enumerate the **legal** (internally consistent)
Roman-numeral analyses for each note/group and compute a composite probability.

**"Legal" means**: the combination of root + quality + inversion deterministically
implies a bass note.  If the predicted bass disagrees with what root+quality+inversion
imply, that combination is illegal.  Similarly, degree1+degree2+localkey must be
consistent with root.

Steps:
1. For each note (or group of notes), extract the top-k candidates per task
2. Enumerate combinations and filter for internal consistency
3. Score each legal combination by the product (or geometric mean) of individual
   task probabilities
4. Return the top-k legal Roman numerals per note/group

This strategy should produce a new column `romanNumeral_reconstructed` alongside
the standard per-task argmax columns.

### 2. `weighted_mean.py` — Confidence-Weighted Mean

Like `"mean"` but weights each note's contribution by its prediction confidence
(max probability across classes) or by inverse entropy.  Configurable via a
`weight_mode` parameter: `"confidence"`, `"inverse_entropy"`, `"duration"`.

### 3. `chord_change.py` — Change-Point Detection Grouping

Instead of fixed onset/beat/measure groupings, detect chord change points from
the raw predictions and create dynamic groups:

1. Compute an onset-level summary (argmax label per onset)
2. Detect boundaries where the label changes
3. Create `chord_change` hyperedge groups in the hyperedges table
4. Aggregate within each chord-change segment

This addresses V1 (context window) and V6 (temporal smoothing) from AGENTS.md.

## Base Class Interface

```python
class AggregationStrategy(ABC):
    @abstractmethod
    def aggregate(
        self,
        probabilities: pd.DataFrame,   # long-format (note_id, task, class_id, ...)
        notes: pd.DataFrame,            # notes table
        hyperedges: pd.DataFrame,       # hyperedges table
        metadata: dict,
        tasks: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Return a wide-format DataFrame with one row per note and per-task
        argmax label + confidence columns."""
```

All new strategies must conform to this interface and be registered via
`@register_strategy("name")`.

## Files to Create/Modify

| File | Action |
|------|--------|
| `analysisgnn/aggregation/roman_numeral.py` | New: legal RN enumeration strategy |
| `analysisgnn/aggregation/weighted_mean.py` | New: confidence/entropy/duration weighted mean |
| `analysisgnn/aggregation/chord_change.py` | New: change-point detection + within-segment aggregation |
| `analysisgnn/aggregation/__init__.py` | Add imports for new strategies |
| `tests/test_aggregation_strategies.py` | Tests for new strategies (using existing Delta Lake at `outputs/Minuet_in_G_Major_K.1/`) |

Files that should NOT need changes: `delta_writer.py`, `delta_reader.py`,
`gradio_hybrid_analysis_app.py` (strategies auto-appear via `list_strategies()`).

## How to Verify

1. All new strategies appear in the Gradio aggregation dropdown automatically
2. Each strategy produces valid output (correct columns, no NaN labels, correct row count)
3. `roman_numeral` strategy produces a `romanNumeral_reconstructed` column with legal
   RN strings
4. `chord_change` strategy produces fewer unique groups than `measure` (more granular
   than measure-level but groups consecutive identical predictions)
5. All tests pass:
   ```
   conda run -n analysisgnn python -m pytest tests/ -v
   ```

## Key Decisions Needed

- What consistency rules to use for legal RN filtering (root+quality+inversion → bass?
  degree+key → root? both?)
- How many top-k candidates per task to consider in the RN enumeration (affects speed)
- Whether `chord_change` should operate on a single task (e.g., `romanNumeral`) or on
  a composite of multiple tasks
- Whether `weighted_mean` should support custom weight functions or just the three
  built-in modes

## Reference

- Aggregation base class: `analysisgnn/aggregation/base.py`
- Registry: `analysisgnn/aggregation/registry.py`
- Mean strategy: `analysisgnn/aggregation/mean.py`
- Roman numeral decoding: `analysisgnn/utils/roman_decode.py`
- Chord representations / vocabularies: `analysisgnn/utils/chord_representations.py`
- Existing Delta Lake output: `outputs/Minuet_in_G_Major_K.1/`
