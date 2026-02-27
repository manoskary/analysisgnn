import torch
import pytorch_lightning as pl

from analysisgnn.models.analysis import ContinualAnalysisGNN


def _lightweight_model():
    model = ContinualAnalysisGNN.__new__(ContinualAnalysisGNN)
    pl.LightningModule.__init__(model)
    model.task_dict = {"romanNumeral": 3, "localkey": 3}
    model.masked_tasks = ["romanNumeral", "localkey"]
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
