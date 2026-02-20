import pandas as pd

from analysisgnn.inference.hybrid_predictor import (
    HybridAnalysisPredictor,
    build_mask_inputs_from_table_edits,
    parse_index_expression,
)


def test_parse_index_expression_1_based_ranges():
    indices = parse_index_expression("1-3, 5, 10-9, x", max_len=12)
    assert indices == [0, 1, 2, 4, 8, 9]


def test_parse_index_expression_bounds_filtered():
    indices = parse_index_expression("0, 1, 2, 99", max_len=3)
    assert indices == [0, 1]


def test_should_use_masked_model_from_user_edits():
    assert HybridAnalysisPredictor.should_use_masked_model(
        user_edits={"label_overrides": {"romanNumeral": {"indices": [0], "labels": ["V"]}}}
    )
    assert HybridAnalysisPredictor.should_use_masked_model(
        user_edits={"node_mask": {"targets": [0], "context": [1]}},
        masked_spec=None,
    )


def test_should_not_use_masked_model_without_context():
    assert not HybridAnalysisPredictor.should_use_masked_model(user_edits=None, masked_spec=None)


def test_build_mask_inputs_from_table_edits():
    df = pd.DataFrame(
        {
            "row": [0, 1, 2],
            "romanNumeral": ["I", "V", "ii6"],
            "localkey": ["C", "C", "C"],
        }
    )
    user_edits, masked_spec, info = build_mask_inputs_from_table_edits(
        edited_df=df,
        masked_tasks=["romanNumeral", "localkey"],
        known_rows_expr="1,3",
        target_rows_expr="2",
    )

    assert user_edits is not None
    assert masked_spec is not None
    assert info["num_known"] == 2
    assert info["num_targets"] == 1
    assert user_edits["node_mask"]["context"] == [0, 2]
    assert user_edits["node_mask"]["targets"] == [1]
    assert "romanNumeral" in user_edits["label_overrides"]
    assert user_edits["label_overrides"]["romanNumeral"]["indices"] == [0, 2]
