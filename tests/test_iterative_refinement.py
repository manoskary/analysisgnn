import torch
import pytorch_lightning as pl

from analysisgnn.models.analysis import ContinualAnalysisGNN


def _lightweight_model():
    model = ContinualAnalysisGNN.__new__(ContinualAnalysisGNN)
    pl.LightningModule.__init__(model)
    model.task_dict = {"romanNumeral": 3, "localkey": 3}
    model.masked_tasks = ["romanNumeral", "localkey"]
    model.structured_refine_tasks = ["romanNumeral", "localkey"]
    model.structured_refine_keep_ratio = 0.35
    model.structured_refine_conf_min = 0.80
    model.structured_refine_boundary_max = 0.50
    return model


def _component_model():
    model = ContinualAnalysisGNN.__new__(ContinualAnalysisGNN)
    pl.LightningModule.__init__(model)
    model.task_dict = {
        "localkey": 3,
        "degree1": 3,
        "degree2": 3,
        "quality": 3,
        "inversion": 3,
    }
    model.masked_tasks = ["localkey", "degree1", "degree2", "quality", "inversion"]
    model.structured_refine_tasks = list(model.masked_tasks)
    model.structured_refine_keep_ratio = 0.25
    model.structured_refine_conf_min = 0.80
    model.structured_refine_boundary_max = 0.50
    return model


def test_joint_mean_confidence_prefers_high_conf_nodes():
    model = _lightweight_model()
    probs = {
        "romanNumeral": torch.tensor(
            [
                [0.90, 0.05, 0.05],
                [0.40, 0.30, 0.30],
            ],
            dtype=torch.float32,
        ),
        "localkey": torch.tensor(
            [
                [0.80, 0.10, 0.10],
                [0.45, 0.45, 0.10],
            ],
            dtype=torch.float32,
        ),
    }
    idx = torch.tensor([0, 1], dtype=torch.long)
    conf = model._joint_mean_confidence(probs, ["romanNumeral", "localkey"], idx)
    assert conf.shape[0] == 2
    assert float(conf[0]) > float(conf[1])


def test_normalize_iterative_spec_defaults():
    model = _lightweight_model()
    cfg = model._normalize_iterative_spec(iterative_spec={"enabled": True}, conditioning=None)
    assert cfg["enabled"] is True
    assert cfg["steps"] == 10
    assert cfg["keep_percentile_per_step"] == 10.0
    assert cfg["masked_tasks"] == ["romanNumeral", "localkey"]
    assert cfg["freeze_confidence"] == "joint_mean"


def test_merge_target_only_predictions_leaves_non_targets_unchanged():
    model = _lightweight_model()
    base = {
        "romanNumeral": torch.tensor([[0.9, 0.1], [0.2, 0.8]], dtype=torch.float32),
        "localkey": torch.tensor([[0.7, 0.3], [0.6, 0.4]], dtype=torch.float32),
    }
    refined = {
        "romanNumeral": torch.tensor([[0.1, 0.9], [0.6, 0.4]], dtype=torch.float32),
        "localkey": torch.tensor([[0.2, 0.8], [0.1, 0.9]], dtype=torch.float32),
    }
    merged = model._merge_target_only_predictions(
        base_predictions=base,
        refined_predictions=refined,
        target_indices=torch.tensor([1], dtype=torch.long),
        tasks=["romanNumeral", "localkey"],
    )
    assert torch.allclose(merged["romanNumeral"][0], base["romanNumeral"][0])
    assert torch.allclose(merged["localkey"][0], base["localkey"][0])
    assert torch.allclose(merged["romanNumeral"][1], refined["romanNumeral"][1])
    assert torch.allclose(merged["localkey"][1], refined["localkey"][1])


def test_normalize_iterative_spec_structured_v2_defaults():
    model = _lightweight_model()
    cfg = model._normalize_iterative_spec(
        iterative_spec={"enabled": True, "version": "structured_v2"},
        conditioning=None,
    )
    assert cfg["enabled"] is True
    assert cfg["version"] == "structured_v2"
    assert cfg["masked_tasks"] == ["romanNumeral", "localkey"]
    assert cfg["segment_keep_ratio"] == 0.35


def test_merge_onset_segments_respects_boundaries():
    model = _lightweight_model()
    onset_labels = {
        "romanNumeral": torch.tensor([1, 1, 1, 2], dtype=torch.long),
        "localkey": torch.tensor([0, 0, 0, 1], dtype=torch.long),
    }
    stable_mask = torch.tensor([True, True, True, True])
    boundary_flags = torch.tensor([False, True, False, False])
    segments = model._merge_onset_segments(
        onset_labels=onset_labels,
        stable_mask=stable_mask,
        boundary_flags=boundary_flags,
        tasks=["romanNumeral", "localkey"],
        onset_confidence=torch.tensor([0.9, 0.9, 0.95, 0.99]),
    )
    assert len(segments) == 4
    assert segments[0]["onset_indices"] == [0]
    assert segments[1]["onset_indices"] == [1]
    assert segments[2]["onset_indices"] == [2]
    assert segments[3]["onset_indices"] == [3]


def test_normalize_iterative_spec_component_v3_defaults():
    model = _component_model()
    cfg = model._normalize_iterative_spec(
        iterative_spec={"enabled": True, "version": "component_v3"},
        conditioning=None,
    )
    assert cfg["enabled"] is True
    assert cfg["version"] == "component_v3"
    assert cfg["steps"] == 3
    assert cfg["masked_tasks"] == ["localkey", "degree1", "degree2", "quality", "inversion"]
    assert cfg["segment_margin_min"] == 0.10
