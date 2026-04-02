import re

from analysisgnn.train.train_analysisgnn import _clip_wandb_label


def test_clip_wandb_label_clips_without_hash_suffix():
    original = "train+eval-HybridGNN-heads-only-tasks=all.rna-feat=simple-mb-lora-no-gnn-aug-nomasked-mtasks=none-nopreserve-noiterrefine"
    clipped = _clip_wandb_label(original, max_len=64, field_name="group")

    assert len(clipped) <= 64
    assert clipped == original[:64].rstrip("-_.")
    assert re.search(r"-[0-9a-f]{10}$", clipped) is None


def test_clip_wandb_label_trims_trailing_separators():
    original = "a" * 64 + "-extra"
    clipped = _clip_wandb_label(original, max_len=64, field_name="group")

    assert clipped == "a" * 64
