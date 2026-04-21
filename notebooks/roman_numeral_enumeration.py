# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Roman Numeral Enumeration from GNN Predictions
#
# This notebook demonstrates the top-k Roman-numeral candidate enumeration
# pipeline: starting from raw per-note probability distributions stored in
# a Delta Lake, building a `ScoringContext` for note groups, and enumerating
# the legal Roman-numeral candidates ranked by score.
#
# ## Contents
#
# 1. [Setup & Data Loading](#setup)
# 2. [Building a ScoringContext](#scoring-context)
# 3. [Single-Group Walkthrough](#single-group)
# 4. [Scoring Candidates](#scoring)
# 5. [Batch Enumeration](#batch)
# 6. [Cross-Validation vs Argmax](#cross-validation)
# 7. [Interesting Cases](#interesting)

# %%
from __future__ import annotations

import os
import pandas as pd

os.chdir(os.path.join(os.path.dirname(os.path.abspath(""))))

OUTPUT_DIR = "outputs/Minuet_in_G_Major_K.1"
GLOBAL_KEY = "G"

# %% [markdown]
# ## 2. Building a ScoringContext <a id="scoring-context"></a>
#
# `ScoringContext.from_delta()` loads all Delta Lake tables and wraps them
# in a single object.  All accessors filter on the fly -- no data is copied.

# %%
from analysisgnn.aggregation.scoring import ScoringContext

full_ctx = ScoringContext.from_delta(OUTPUT_DIR)
full_ctx

# %%
# Distribution for a single note and task
sample_note = full_ctx.note_ids[4]
full_ctx.distribution(sample_note, "quality").sort_values(ascending=False).iloc[:5]

# %%
# Top-3 for quality
full_ctx.top_k("quality", 3).sort_values("note_id")

# %%
# Argmax for localkey
full_ctx.argmax("localkey")["class_label"].value_counts()

# %% [markdown]
# ## 3. Single-Group Walkthrough <a id="single-group"></a>
#
# Pick one beat group, build a sub-context, and show the top-k labels
# per core task that the enumerator will use.

# %%
from analysisgnn.storage import delta_reader

beat_groups = delta_reader.load_hyperedges(OUTPUT_DIR, edge_type="beat")
group_ids = sorted(beat_groups["group_id"].unique())

target_group = group_ids[2]
group_notes = beat_groups.loc[beat_groups["group_id"] == target_group, "note_id"].tolist()

group_ctx = full_ctx.subcontext(group_notes)
group_ctx

# %%
# Top-k labels per core task (the enumerator's input)
from analysisgnn.aggregation.roman_numeral import _group_top_k_labels

K = 3
{task: _group_top_k_labels(group_ctx, task, K)
 for task in ("quality", "degree1", "inversion", "localkey", "degree2")
 if task in group_ctx.tasks}

# %%
# Cartesian product size (before pruning)
from functools import reduce
from operator import mul

task_labels = {
    task: [l for l in _group_top_k_labels(group_ctx, task, K) if not (task == "quality" and l == "None")]
    if task in group_ctx.tasks else ["None"]
    for task in ("quality", "degree1", "inversion", "localkey", "degree2")
}
reduce(mul, (len(v) for v in task_labels.values()))

# %% [markdown]
# ## 4. Scoring Candidates <a id="scoring"></a>
#
# Run the full enumeration pipeline: top-k extraction, Cartesian product,
# legality pruning, OHR construction, validation, scoring, dedup.

# %%
from analysisgnn.aggregation.roman_numeral import enumerate_roman_numerals
from analysisgnn.aggregation.scoring import GeometricMeanScorer, ProductScorer

candidates, trace = enumerate_roman_numerals(
    group_ctx, GLOBAL_KEY, k=3, top_n=10, scorer=GeometricMeanScorer()
)
trace

# %%
# Top candidates as a DataFrame
pd.DataFrame([
    {"rank": c.rank, "dcml": c.dcml, "core_score": c.result.core.score,
     "n_notes": c.result.core.num_notes, **c.candidate}
    for c in candidates
])

# %%
# Compare scorers: GeometricMean vs Product
cands_geom, _ = enumerate_roman_numerals(
    group_ctx, GLOBAL_KEY, k=3, top_n=5, scorer=GeometricMeanScorer()
)
cands_prod, _ = enumerate_roman_numerals(
    group_ctx, GLOBAL_KEY, k=3, top_n=5, scorer=ProductScorer()
)

n = min(len(cands_geom), len(cands_prod))
pd.DataFrame({
    "rank": range(1, n + 1),
    "geom_dcml": [c.dcml for c in cands_geom[:n]],
    "geom_score": [c.result.core.score for c in cands_geom[:n]],
    "prod_dcml": [c.dcml for c in cands_prod[:n]],
    "prod_score": [c.result.core.score for c in cands_prod[:n]],
})

# %% [markdown]
# ## 5. Batch Enumeration <a id="batch"></a>
#
# Run the enumerator over all beat groups and collect the top-1 result
# per group into a summary table.

# %%
# Vectorised group-note lookup
group_note_map = beat_groups.groupby("group_id")["note_id"].apply(list).to_dict()

batch_results = {
    gid: enumerate_roman_numerals(full_ctx.subcontext(gnotes), GLOBAL_KEY, k=3, top_n=3)
    for gid, gnotes in group_note_map.items()
}

summary = pd.DataFrame([
    {"group": gid, "top1_dcml": cands[0].dcml if cands else "(none)",
     "score": cands[0].result.core.score if cands else 0.0,
     "n_notes": cands[0].result.core.num_notes if cands else 0,
     "n_candidates": tr.num_after_dedup,
     "n_pruned": tr.num_raw_combos - tr.num_after_dedup}
    for gid, (cands, tr) in batch_results.items()
])
summary

# %%
{"groups_with_candidates": (summary["n_candidates"] > 0).sum(),
 "total_groups": len(summary),
 "avg_candidates": summary["n_candidates"].mean()}

# %% [markdown]
# ## 6. Cross-Validation vs Argmax <a id="cross-validation"></a>
#
# Compare the enumerated top-1 Roman numeral against the raw argmax
# `romanNumeral` prediction from the probabilities table.

# %%
# Majority-vote argmax romanNumeral per beat group
argmax_rn = full_ctx.argmax("romanNumeral").rename(columns={"class_label": "argmax_rn"})

group_argmax = (
    beat_groups[["group_id", "note_id"]]
    .merge(argmax_rn[["note_id", "argmax_rn"]], on="note_id")
    .groupby("group_id")["argmax_rn"]
    .agg(lambda s: s.mode().iloc[0])
    .rename("argmax_majority_rn")
)

cross = summary.merge(group_argmax, left_on="group", right_index=True)
cross["enum_chord"] = cross["top1_dcml"].str.split("/").str[0]
cross["match"] = cross["enum_chord"] == cross["argmax_majority_rn"]

# %%
{"agreement": cross["match"].sum(),
 "disagreement": (~cross["match"]).sum(),
 "pct_agreement": f"{100 * cross['match'].mean():.1f}%"}

# %%
# Disagreements
cross.loc[~cross["match"], ["group", "top1_dcml", "enum_chord", "argmax_majority_rn", "score"]]

# %% [markdown]
# ## 7. Interesting Cases <a id="interesting"></a>
#
# Look at groups where the top-2 candidates are close in score
# (competitive alternatives), or where the enumerated result disagrees
# with the raw argmax.

# %%
# Groups with competitive top-2
CLOSE_THRESHOLD = 0.8

close_df = pd.DataFrame([
    {"group": gid,
     "top1": cands[0].dcml, "top1_score": cands[0].result.core.score,
     "top2": cands[1].dcml, "top2_score": cands[1].result.core.score,
     "ratio": cands[1].result.core.score / cands[0].result.core.score
             if cands[0].result.core.score > 0 else 0}
    for gid, (cands, _) in batch_results.items()
    if len(cands) >= 2
    and (cands[1].result.core.score / cands[0].result.core.score
         if cands[0].result.core.score > 0 else 0) >= CLOSE_THRESHOLD
]).sort_values("ratio", ascending=False)
close_df

# %%
# Detailed look at the most competitive case
if len(close_df) > 0:
    case_gid = close_df.iloc[0]["group"]
    case_cands = batch_results[case_gid][0]
    pd.DataFrame([
        {"rank": c.rank, "dcml": c.dcml, "score": c.result.core.score, **c.candidate}
        for c in case_cands
    ])
