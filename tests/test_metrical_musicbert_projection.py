import torch

from analysisgnn.models.analysis import TorchAnalysisGNN


def test_metrical_projection_uses_base_width_for_non_note_nodes():
    model = TorchAnalysisGNN(
        metadata=(
            ["note", "beat", "measure"],
            [("note", "onset", "note")],
        ),
        in_channels=1047,
        base_in_channels=23,
        hidden_channels=32,
        out_channels=16,
        task_dict={"romanNumeral": 185},
        num_layers=1,
        use_graph_encoder=False,
        logit_fusion=False,
    )

    assert model.project_dict["note"][0].in_features == 1175
    assert model.project_dict["beat"][0].in_features == 23
    assert model.project_dict["measure"][0].in_features == 23

    x = model.encode(
        pitch_spelling=torch.zeros(5, dtype=torch.long),
        key_signature=torch.zeros(5, dtype=torch.long),
        x_dict={
            "note": torch.zeros(5, 1047),
            "beat": torch.zeros(3, 23),
            "measure": torch.zeros(2, 23),
        },
        edge_index_dict={
            ("note", "onset", "note"): torch.empty((2, 0), dtype=torch.long),
        },
        batch_dict={
            "note": torch.zeros(5, dtype=torch.long),
        },
        batch_size=5,
        neighbor_mask_node=None,
        neighbor_mask_edge=None,
    )

    assert x.shape == (5, 16)
