import torch

from analysisgnn.models.analysis import combined_degree_accuracy


def test_combined_degree_accuracy_exact_match_with_valid_mask():
    logits = {
        "degree1": torch.tensor(
            [
                [0.1, 0.8, 0.1],
                [0.7, 0.2, 0.1],
                [0.2, 0.2, 0.6],
            ]
        ),
        "degree2": torch.tensor(
            [
                [0.9, 0.1],
                [0.1, 0.9],
                [0.8, 0.2],
            ]
        ),
    }
    labels = {
        "degree1": torch.tensor([1, 0, -1]),
        "degree2": torch.tensor([0, 0, 1]),
    }
    task_dict = {"degree1": 3, "degree2": 2}

    metric = combined_degree_accuracy(logits, labels, task_dict)

    assert metric is not None
    assert torch.isclose(metric, torch.tensor(0.5))


def test_combined_degree_accuracy_returns_none_without_valid_joint_labels():
    logits = {
        "degree1": torch.tensor([[0.1, 0.9]]),
        "degree2": torch.tensor([[0.9, 0.1]]),
    }
    labels = {
        "degree1": torch.tensor([-1]),
        "degree2": torch.tensor([3]),
    }
    task_dict = {"degree1": 2, "degree2": 2}

    metric = combined_degree_accuracy(logits, labels, task_dict)

    assert metric is None
