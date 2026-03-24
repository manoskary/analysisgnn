from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.backends.opt_einsum import strategy
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import StochasticWeightAveraging
from analysisgnn.models.analysis import ContinualAnalysisGNN
from analysisgnn.models.musicbert_backbone import MusicBertAdapterConfig
from analysisgnn.models.musicbert_note_encoder import MusicBertNoteEncoder
from analysisgnn.data.datamodules.analysis import AnalysisDataModule
import torch
import argparse
import wandb
import os
import hashlib
import re
from pathlib import Path
import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.tuner import Tuner


# for repeatability
seed_everything(0, workers=True)
torch.multiprocessing.set_sharing_strategy("file_system")


class WarmupEarlyStopping(EarlyStopping):
    """EarlyStopping that is inactive before a minimum epoch."""

    def __init__(self, *args, start_epoch: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = max(0, int(start_epoch))

    def _should_skip_check(self, trainer) -> bool:
        if trainer.current_epoch < self.start_epoch:
            return True
        return super()._should_skip_check(trainer)

TASK_DICT = {
        "cadence": 4,
        "localkey": 50,
        "tonkey": 50,
        "quality": 15,
        "inversion": 4,
        "root": 38,
        "bass": 38,
        "degree1": 22,
        "degree2": 22,
        "hrythm": 2,
        "pcset": 94,
        "romanNumeral": 185,
        "section": 2,
        "phrase": 2,
        "organ_point": 2,
        "tpc_in_label": 2,
        "tpc_is_root": 2,
        "tpc_is_bass": 2,
        "organ_point": 2,
        "downbeat": 45,
        "note_degree": 49,
        "staff": 4,
    }

WANDB_TASK_ABBR = {
    "all": "all",
    "rna": "rna",
    "cadence": "cad",
    "localkey": "lk",
    "tonkey": "tk",
    "quality": "ql",
    "inversion": "inv",
    "root": "rt",
    "bass": "bs",
    "degree1": "d1",
    "degree2": "d2",
    "romanNumeral": "rn",
    "section": "sec",
    "phrase": "phr",
    "organ_point": "op",
    "tpc_in_label": "nct",
    "tpc_is_root": "nct_rt",
    "tpc_is_bass": "nct_bs",
    "downbeat": "db",
    "note_degree": "nd",
    "staff": "stf",
}


def _clip_wandb_label(value: str, *, max_len: int = 128, field_name: str = "label") -> str:
    """Clip long W&B identifiers while preserving uniqueness."""
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    keep = max(1, max_len - len(digest) - 1)
    clipped = f"{text[:keep]}-{digest}"
    print(
        f"Warning: W&B {field_name} exceeded {max_len} chars; "
        f"using clipped value '{clipped}'."
    )
    return clipped


def _sanitize_wandb_token(value: str) -> str:
    """Normalize free-form text into a W&B-safe token."""
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = text.strip("._-")
    return text or "none"


def _compact_task_tag(tasks, *, field_name: str, max_items: int = 6, max_len: int = 48) -> str:
    """Build short deterministic task tags for W&B names/groups."""
    if isinstance(tasks, str):
        raw_items = [x.strip() for x in tasks.split(",") if x.strip()]
    else:
        raw_items = [str(x).strip() for x in (tasks or []) if str(x).strip()]

    compact = []
    seen = set()
    for task in raw_items:
        token = WANDB_TASK_ABBR.get(task, _sanitize_wandb_token(task))
        if token in seen:
            continue
        seen.add(token)
        compact.append(token)

    overflow = 0
    if len(compact) > max_items:
        overflow = len(compact) - max_items
        compact = compact[:max_items]
    if overflow > 0:
        compact.append(f"plus{overflow}")

    label = ".".join(compact) if compact else "none"
    return _clip_wandb_label(label, max_len=max_len, field_name=field_name)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default="-1",
                        help="GPUs to use, for multiple separate by comma, i.e. 0,1,2. Use -1 for CPU. (Default: -1)")
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="Trainer precision (e.g., 16-mixed, bf16-mixed, 32-true). Defaults to bf16-mixed on Ampere+ CUDA, otherwise 16-mixed on CUDA.",
    )
    parser.add_argument('--num_layers', type=int, default=3,
                        help="Number of layers on the Graph Convolutional Encoder Network")
    parser.add_argument('--hidden_channels', type=int, default=256, help="Number of hidden units")
    parser.add_argument('--out_channels', type=int, default=128, help="Number of output units")
    parser.add_argument('--num_epochs', type=str, default="50", help="Number of epochs")
    parser.add_argument('--dropout', type=float, default=0.3, help="Dropout")
    parser.add_argument('--lr', type=float, default=0.005, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=5e-3, help="Weight decay")
    parser.add_argument("--grad_clip_val", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of workers")
    parser.add_argument(
        "--num_sanity_val_steps",
        type=int,
        default=0,
        help="Sanity validation steps before training starts (set 0 for faster startup).",
    )
    parser.add_argument(
        "--reload_dataloaders_every_n_epochs",
        type=int,
        default=0,
        help="How often to rebuild dataloaders. 0 avoids per-epoch reload overhead.",
    )
    parser.add_argument("--lambda_dctn", type=float, default=0.5, help="Lambda for the distilation loss")
    parser.add_argument("--lambda_featl", type=float, default=0.1, help="Lambda for the feature regularization loss")
    parser.add_argument("--lambda_ewc", type=float, default=2.0, help="Lambda for the Elastic Weight Consolidation loss")
    parser.add_argument("--lambda_edge", type=float, default=0.1, help="Lambda for the edge classification loss")
    parser.add_argument("--use_edge_loss", action="store_true", help="Enable edge-based loss training")
    parser.add_argument("--load_from_checkpoint", action="store_true", help="Load model from WANDB checkpoint")
    parser.add_argument("--force_reload", action="store_true", help="Force reload of the data")
    parser.add_argument("--model", type=str, default="HybridGNN", help="Encoder type to use",
                        choices=["HybridGNN", "HGT", "MetricalGNN"])
    parser.add_argument(
        "--disable_graph_encoder",
        action="store_true",
        help="Disable graph message passing and train classification heads directly on note representations.",
    )
    parser.add_argument("--use_jk", help="Use Jumping Knowledge", action="store_true")
    parser.add_argument("--tags", type=str, default="", help="Tags to add to the WandB run api")
    parser.add_argument("--homogeneous", action="store_true", help="Use homogeneous graphs")
    parser.add_argument("--reg_loss_type", type=str, default="la", help="Use different regularization loss")
    parser.add_argument("--raw_dir", type=str, default=None, help="Raw directory to use")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--auto_batch_size", type=bool, help="Automatically find optimal batch size", default=True)
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients before stepping the optimizer",
    )
    parser.add_argument("--use_reledge", action="store_true", help="Use reledge")
    parser.add_argument("--use_wandb", help="Use wandb", action="store_true",)
    parser.add_argument("--wandb_project", type=str, default="AnalysisGNN-MusicBERT",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default="melkisedeath",
                        help="W&B entity/team name")
    parser.add_argument("--use_metrical", action="store_true", help="Use metrical graphs")
    parser.add_argument("--subgraph_size", type=int, default=500, help="Subgraph size")
    parser.add_argument("--add_beats", action="store_true", help="Add beats to the graph")
    parser.add_argument("--add_measures", action="store_true", help="Add measures to the graph")
    parser.add_argument("--mt_strategy", type=str, default="wloss", help="Multi-task strategy")
    parser.add_argument("--feat_norm_scale", type=float, default=0.0,
                        help="Scale factor for the feature normalization loss")
    parser.add_argument("--compile", action="store_true", help="Compile the model with Pytorch>2.0")
    parser.add_argument("--use_swa", action="store_true", help="Use Stochastic Weight Averaging")
    # parser.add_argument("--task_dict", type=dict, default={
    #     "cadence": 4, "metrical_strength": 5, "localkey": 35, "tonkey": 35, "quality": 15, "inversion": 4, "root": 35,
    #     "bass": 35, "degree1": 22, "degree2": 22, "hrythm": 2, "pcset": 94, "romanNumeral": 185, "section": 2,
    #     "phrase": 2, "tpc_in_label": 2, "tpc_is_root": 2, "tpc_is_bass": 2, "organ_point": 2
    # }, help="Task dictionary")
    parser.add_argument("--main_tasks", type=str, default="all,cadence,rna", help="Main tasks")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to use for training, mainly used for debuging purposes")
    parser.add_argument("--verbose", action="store_true", help="Verbose")
    parser.add_argument("--random_split", action="store_true", help="random_split")
    parser.add_argument("--logit_fusion", action="store_true", help="In case of multiple tasks, use logit fusion")
    parser.add_argument("--has_memories", help="Use memories", type=bool, default=False,)
    parser.add_argument("--feature_type", type=str, default="simple", choices=["cadence", "simple"], help="Input feature type")
    parser.add_argument("--config_path", type=str, default=None, help="Path to the config file")
    parser.add_argument("--do_train", action="store_true", help="Train the model")
    parser.add_argument("--do_eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint")
    parser.add_argument("--use_transpositions", help="Use transpositions", action="store_true")
    parser.add_argument("--use_ewc", action="store_true", help="Use Elastic Weight Consolidation")
    parser.add_argument("--cl_training", action="store_true", help="Use Continual Learning Training")
    parser.add_argument("--use_smote", action="store_true", help="Use SMOTE")
    parser.add_argument("--train_with_masking", action="store_true", 
                        help="Enable semi-supervised node masking with random masking during training")
    parser.add_argument("--mask_ratio", type=float, default=0.15,
                        help="Ratio of nodes to mask as context during training (default: 0.15)")
    parser.add_argument(
        "--masked_prediction_train",
        action="store_true",
        help="Enable label-conditioned masked prediction training mode.",
    )
    parser.add_argument(
        "--masked_tasks",
        type=str,
        default="",
        help="Comma-separated tasks for masked prediction conditioning (required when --masked_prediction_train).",
    )
    parser.add_argument(
        "--known_ratio",
        type=float,
        default=None,
        help="Known-label/context ratio for masked prediction (defaults to mask_ratio when omitted).",
    )
    parser.add_argument(
        "--mask_sampling_policy",
        type=str,
        default="hybrid",
        choices=["random", "span", "hybrid"],
        help="Sampling policy for context nodes in masked prediction mode.",
    )
    parser.add_argument(
        "--mask_span_min_onsets",
        type=int,
        default=2,
        help="Minimum onset-span length when mask_sampling_policy uses spans.",
    )
    parser.add_argument(
        "--mask_span_max_onsets",
        type=int,
        default=8,
        help="Maximum onset-span length when mask_sampling_policy uses spans.",
    )
    parser.add_argument(
        "--constraint_mode",
        type=str,
        default="hard",
        choices=["hard", "soft"],
        help="Constraint policy for known labels in masked prediction mode.",
    )
    parser.add_argument(
        "--feedback_mode",
        type=str,
        default="single_pass",
        choices=["single_pass"],
        help="Feedback loop mode for label conditioning.",
    )
    parser.add_argument(
        "--iterative_refine_train",
        action="store_true",
        help="Enable 2-step self-conditioning refinement during masked-prediction training.",
    )
    parser.add_argument(
        "--iterative_train_steps",
        type=int,
        default=2,
        help="Number of refinement passes during training (v1 supports up to 2).",
    )
    parser.add_argument(
        "--iterative_train_keep_ratio",
        type=float,
        default=0.5,
        help="Fraction of highest-confidence target nodes frozen as pseudo-known for pass-2.",
    )
    parser.add_argument(
        "--iterative_train_pass2_weight",
        type=float,
        default=1.0,
        help="Loss weight for pass-2 masked loss.",
    )
    parser.add_argument(
        "--iterative_train_consistency_lambda",
        type=float,
        default=0.02,
        help="KL consistency weight between pass-1 and pass-2 on pseudo-frozen nodes.",
    )
    parser.add_argument(
        "--iterative_train_start_epoch",
        type=int,
        default=8,
        help="Epoch to start iterative pass-2 refinement (before this, pass-2 is disabled).",
    )
    parser.add_argument(
        "--iterative_train_keep_ratio_start",
        type=float,
        default=0.05,
        help="Pass-2 pseudo-freeze ratio at iterative_train_start_epoch.",
    )
    parser.add_argument(
        "--iterative_train_keep_ratio_end",
        type=float,
        default=0.30,
        help="Pass-2 pseudo-freeze ratio at end of training.",
    )
    parser.add_argument(
        "--iterative_train_conf_min",
        type=float,
        default=0.80,
        help="Minimum confidence for pseudo-label freezing candidates in pass-2.",
    )
    parser.add_argument(
        "--iterative_train_min_remaining_ratio",
        type=float,
        default=0.20,
        help="Minimum target-node ratio that must remain unfrozen for pass-2.",
    )
    parser.add_argument(
        "--iterative_train_pass2_weight_start",
        type=float,
        default=0.25,
        help="Pass-2 loss weight at iterative_train_start_epoch.",
    )
    parser.add_argument(
        "--iterative_train_pass2_weight_end",
        type=float,
        default=0.75,
        help="Pass-2 loss weight at end of training.",
    )
    parser.add_argument(
        "--iterative_eval",
        action="store_true",
        help="Enable LLaDA-style iterative refinement during validation/test (no known labels at step 1).",
    )
    parser.add_argument(
        "--iterative_eval_during_fit",
        dest="iterative_eval_during_fit",
        action="store_true",
        help="Also compute iterative metrics during validation epochs (logged under val_iter/*).",
    )
    parser.add_argument(
        "--no_iterative_eval_during_fit",
        dest="iterative_eval_during_fit",
        action="store_false",
        help="Skip iterative validation during fit; keep iterative benchmarking for test only.",
    )
    parser.set_defaults(iterative_eval_during_fit=False)
    parser.add_argument(
        "--iterative_eval_steps",
        type=int,
        default=10,
        help="Number of refinement steps for iterative evaluation.",
    )
    parser.add_argument(
        "--iterative_eval_keep_percentile",
        type=float,
        default=10.0,
        help="Percentile of highest-confidence remaining nodes to freeze per step in iterative evaluation.",
    )
    parser.add_argument(
        "--iterative_eval_tasks",
        type=str,
        default="",
        help="Deprecated alias for --iterative_eval_masked_tasks.",
    )
    parser.add_argument(
        "--iterative_eval_masked_tasks",
        type=str,
        default="",
        help="Comma-separated task list for iterative evaluation refinement; defaults to masked_tasks or all available.",
    )
    parser.add_argument(
        "--iterative_eval_target_only_update",
        action="store_true",
        default=False,
        help="If set, iterative evaluation only updates target nodes and preserves baseline predictions elsewhere.",
    )
    parser.add_argument(
        "--iterative_eval_zero_known",
        dest="iterative_eval_zero_known",
        action="store_true",
        help="Force iterative evaluation to start with zero known labels (fair apples-to-apples against full inference).",
    )
    parser.add_argument(
        "--no_iterative_eval_zero_known",
        dest="iterative_eval_zero_known",
        action="store_false",
        help="Allow iterative evaluation to use known labels from masked conditioning.",
    )
    parser.set_defaults(iterative_eval_zero_known=True)
    parser.add_argument(
        "--beam_eval",
        action="store_true",
        help="Enable onset-level constrained beam decoding evaluation in parallel with baseline metrics.",
    )
    parser.add_argument(
        "--beam_eval_during_fit",
        action="store_true",
        default=False,
        help="Also compute beam-decoded validation metrics during fit (val_full_beam/*).",
    )
    parser.add_argument(
        "--beam_width",
        type=int,
        default=8,
        help="Beam width for onset-level RNA decoding.",
    )
    parser.add_argument(
        "--beam_spec_json",
        type=str,
        default=None,
        help="Optional JSON file with beam decoder overrides (topk_by_task, weights, penalties).",
    )
    parser.add_argument(
        "--aggregation_mode",
        type=str,
        default="mean",
        choices=["mean", "voter", "voter_consistent_beat"],
        help="Post-hoc aggregation mode for onset/beat/measure pooling.",
    )
    parser.add_argument(
        "--aggregation_voter_path",
        type=str,
        default=None,
        help="Path to post-hoc voter checkpoint artifact (used when aggregation_mode=voter|voter_consistent_beat).",
    )
    parser.add_argument(
        "--aggregation_compare_mean_in_test",
        action="store_true",
        help="During test, log both mean and voter aggregation metrics for apples-to-apples comparison.",
    )
    parser.add_argument("--use_musicbert", action="store_true", help="Use MusicBERT note encoder")
    parser.add_argument("--musicbert_model_name", type=str, default="manoskary/musicbert-large", help="MusicBERT model name")
    parser.add_argument(
        "--musicbert_unfreeze_backbone",
        action="store_false",
        dest="musicbert_freeze_backbone",
        default=True,
        help="Unfreeze MusicBERT backbone parameters",
    )
    parser.add_argument("--musicbert_use_lora", action="store_true", help="Enable LoRA adapters for MusicBERT")
    parser.add_argument("--musicbert_lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--musicbert_lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--musicbert_lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--musicbert_alignment_dir", type=str, default=None, help="Directory with MusicBERT alignment .npz files")
    parser.add_argument(
        "--musicbert_cached_embeddings_dir",
        type=str,
        default=None,
        help="Directory with precomputed note-level MusicBERT embeddings (.npz).",
    )
    parser.add_argument(
        "--musicbert_require_cached_embeddings",
        action="store_true",
        help="Require cached MusicBERT embeddings for every graph (no runtime fallback).",
    )
    parser.add_argument(
        "--musicbert_embedding_dim",
        type=int,
        default=None,
        help="Override cached MusicBERT embedding dimension (auto-inferred when omitted).",
    )
    parser.add_argument(
        "--musicbert_fusion",
        type=str,
        default="replace",
        choices=["replace", "concat", "gate"],
        help="How to fuse MusicBERT embeddings with note features.",
    )
    parser.add_argument(
        "--mt_conflict_method",
        type=str,
        default="pcgrad",
        choices=["none", "pcgrad", "cagrad", "gradnorm"],
        help="Conflict mitigation for multitask losses.",
    )
    parser.add_argument(
        "--cagrad_c",
        type=float,
        default=0.4,
        help="CAGrad conflict-aversion strength (only used when mt_conflict_method=cagrad).",
    )
    parser.add_argument(
        "--cagrad_max_iter",
        type=int,
        default=25,
        help="Max simplex optimization steps for CAGrad (only used when mt_conflict_method=cagrad).",
    )
    parser.add_argument(
        "--gradnorm_alpha",
        type=float,
        default=1.5,
        help="GradNorm alpha parameter (only used when mt_conflict_method=gradnorm).",
    )
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="cosine_warmup",
        choices=["cosine_warmup", "plateau"],
        help="Learning-rate scheduler type.",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio for cosine scheduler.")
    parser.add_argument("--min_lr_ratio", type=float, default=0.02, help="Minimum LR ratio for cosine scheduler.")
    parser.add_argument("--plateau_factor", type=float, default=0.5, help="ReduceLROnPlateau factor.")
    parser.add_argument("--plateau_patience", type=int, default=6, help="ReduceLROnPlateau patience.")
    parser.add_argument("--plateau_min_lr", type=float, default=1e-6, help="ReduceLROnPlateau minimum LR.")
    parser.add_argument("--monitor_metric", type=str, default="val_full/total_loss",
                        help="Metric used by checkpointing/early stopping and plateau scheduler.")
    parser.add_argument("--monitor_mode", type=str, default="min", choices=["min", "max"],
                        help="Optimization mode for monitor_metric.")
    parser.add_argument("--early_stop_patience", type=int, default=12, help="Early stopping patience.")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.002, help="Early stopping min_delta.")
    parser.add_argument(
        "--early_stop_start_epoch",
        type=int,
        default=12,
        help="Do not activate early stopping before this epoch.",
    )
    parser.add_argument("--optimizer_stats_log_every_n_steps", type=int, default=50,
                        help="Log LR and gradient norms every N optimizer steps.")
    parser.add_argument(
        "--preserve_pretrained",
        action="store_true",
        help="Enable no-regression preservation losses (teacher KD + feature anchor + L2-SP).",
    )
    parser.add_argument(
        "--preserve_kd_lambda",
        type=float,
        default=1.0,
        help="Lambda for teacher logit distillation loss.",
    )
    parser.add_argument(
        "--preserve_feat_lambda",
        type=float,
        default=0.1,
        help="Lambda for teacher feature-anchor loss.",
    )
    parser.add_argument(
        "--preserve_l2sp_lambda",
        type=float,
        default=1e-4,
        help="Lambda for L2-SP drift penalty against pretrained initialization.",
    )
    parser.add_argument(
        "--preserve_temperature",
        type=float,
        default=2.0,
        help="Distillation temperature for pretrained-preservation KD.",
    )
    parser.add_argument(
        "--preserve_tasks",
        type=str,
        default="all_nonmasked",
        help="Comma-separated tasks for preservation KD, or 'all_nonmasked'.",
    )
    parser.add_argument(
        "--preserve_teacher_checkpoint",
        type=str,
        default=None,
        help="Optional teacher checkpoint path; defaults to checkpoint_path when resuming.",
    )
    parser.add_argument(
        "--unmasked_batch_prob",
        type=float,
        default=0.30,
        help="Probability of running a fully-unmasked supervised batch during masked training.",
    )
    parser.add_argument(
        "--freeze_graph_encoder_stage_epochs",
        type=int,
        default=5,
        help="When preserve_pretrained is enabled, freeze graph encoder for this many initial epochs.",
    )
    parser.add_argument(
        "--preserve_stage_b_epochs",
        type=int,
        default=25,
        help="Epoch count for stage-B LR in preserve_pretrained mode.",
    )
    parser.add_argument(
        "--preserve_stage_a_lr",
        type=float,
        default=5e-5,
        help="Stage-A learning rate (preserve_pretrained).",
    )
    parser.add_argument(
        "--preserve_stage_b_lr",
        type=float,
        default=3e-5,
        help="Stage-B learning rate (preserve_pretrained).",
    )
    parser.add_argument(
        "--preserve_stage_c_lr",
        type=float,
        default=2e-5,
        help="Stage-C learning rate (preserve_pretrained).",
    )
    parser.add_argument(
        "--preserve_max_regression_abs",
        type=float,
        default=0.015,
        help="Maximum allowed absolute drop for val/nonmasked_total_acc vs teacher baseline.",
    )
    parser.add_argument("--robust_profile", action="store_true",
                        help="Apply robust defaults for augmented multitask runs.")
    parser.add_argument("--early_stopping", dest="early_stopping", action="store_true",
                        help="Enable early stopping.")
    parser.add_argument("--no_early_stopping", dest="early_stopping", action="store_false",
                        help="Disable early stopping.")
    parser.set_defaults(early_stopping=True)
    return parser


def _get_primary_cuda_major() -> int:
    if not torch.cuda.is_available():
        return 0
    try:
        major, _ = torch.cuda.get_device_capability(0)
        return int(major)
    except Exception:
        return 0


def _validate_selected_cuda_devices(gpus_arg: str) -> None:
    if gpus_arg == "-1" or not torch.cuda.is_available():
        return
    try:
        device_ids = [int(eval(gpu)) for gpu in gpus_arg.split(",")]
    except Exception as exc:
        raise ValueError(f"Invalid --gpus value '{gpus_arg}'. Expected '-1' or comma-separated integers.") from exc
    device_count = torch.cuda.device_count()
    for device_id in device_ids:
        if device_id < 0 or device_id >= device_count:
            raise ValueError(
                f"Requested CUDA device index {device_id} is out of range for visible device count={device_count}. "
                "If you are using CUDA_VISIBLE_DEVICES, --gpus indices are relative to the visible subset."
            )
    supported_arches = set(torch.cuda.get_arch_list())
    supported_sms = []
    for arch in supported_arches:
        if not arch.startswith("sm_"):
            continue
        try:
            supported_sms.append(int(arch.split("_", 1)[1]))
        except ValueError:
            continue
    supported_majors = {sm // 10 for sm in supported_sms}
    max_supported_major = max(supported_majors) if supported_majors else None
    unsupported = []
    for device_id in device_ids:
        major, minor = torch.cuda.get_device_capability(device_id)
        sm = f"sm_{major}{minor}"
        # Be tolerant of minor revisions (e.g., sm_89 vs sm_90) as long as major arch
        # is known/supported by the build. Hard-fail only on truly new major arches.
        if max_supported_major is None or major not in supported_majors or major > max_supported_major:
            unsupported.append((device_id, torch.cuda.get_device_name(device_id), sm))
    if unsupported:
        unsupported_desc = ", ".join(
            f"index={idx} name='{name}' capability={sm}" for idx, name, sm in unsupported
        )
        raise RuntimeError(
            "Selected CUDA device is not supported by this PyTorch build. "
            f"Unsupported device(s): {unsupported_desc}. "
            f"PyTorch supports: {sorted(supported_arches)}. "
            "Pick a different GPU, or install a PyTorch build that supports your GPU architecture."
        )


def _infer_cached_embedding_dim(cache_dir: str) -> int:
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        raise ValueError(f"Cached MusicBERT embedding directory not found: {cache_dir}")
    for npz_path in sorted(cache_path.glob("*.npz")):
        with np.load(npz_path) as data:
            if "note_embeddings" not in data:
                continue
            emb = data["note_embeddings"]
            if emb.ndim != 2:
                continue
            return int(emb.shape[1])
    raise ValueError(
        f"Could not infer cached embedding dimension from {cache_dir}. "
        "Expected at least one .npz file containing `note_embeddings`."
    )


def main():
    parser = get_parser()

    args = parser.parse_args()
    args.main_tasks = args.main_tasks.split(",")
    args.masked_tasks = [t.strip() for t in args.masked_tasks.split(",") if t.strip()]
    args.iterative_eval_tasks = [t.strip() for t in args.iterative_eval_tasks.split(",") if t.strip()]
    args.iterative_eval_masked_tasks = [
        t.strip() for t in args.iterative_eval_masked_tasks.split(",") if t.strip()
    ]
    args.preserve_tasks = [t.strip() for t in args.preserve_tasks.split(",") if t.strip()]
    args.num_epochs = args.num_epochs.split(",")
    if len(args.num_epochs) == 1:
        args.num_epochs = int(args.num_epochs[0])
        args.epochs_per_task = [args.num_epochs // len(args.main_tasks)] * len(args.main_tasks)
    else:
        args.epochs_per_task = [int(n) for n in args.num_epochs]
        args.num_epochs = sum(args.epochs_per_task)
    # tranform args to dict
    config = vars(args)
    config["task_dict"] = TASK_DICT
    config["use_edge_loss"] = config.get("use_edge_loss", False)
    _validate_selected_cuda_devices(config["gpus"])
    use_cuda = config["gpus"] != "-1" and torch.cuda.is_available()
    cuda_major = _get_primary_cuda_major() if use_cuda else 0
    ampere_or_newer = cuda_major >= 8
    if config.get("precision") is None:
        if use_cuda:
            if ampere_or_newer and torch.cuda.is_bf16_supported():
                config["precision"] = "bf16-mixed"
            else:
                config["precision"] = "16-mixed"
        else:
            config["precision"] = "32-true"
    if use_cuda and ampere_or_newer:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        print("Enabled Ampere+ fast math (TF32 + high matmul precision).")

    if args.config_path is not None:
        import json
        args_config = config.copy()
        with open(args.config_path, "r") as f:
            config = json.load(f)

        for k, v in args_config.items():
            if k not in config.keys():
                config[k] = v

    if isinstance(config.get("masked_tasks", []), str):
        config["masked_tasks"] = [t.strip() for t in config["masked_tasks"].split(",") if t.strip()]
    if isinstance(config.get("iterative_eval_tasks", []), str):
        config["iterative_eval_tasks"] = [
            t.strip() for t in config["iterative_eval_tasks"].split(",") if t.strip()
        ]
    if isinstance(config.get("iterative_eval_masked_tasks", []), str):
        config["iterative_eval_masked_tasks"] = [
            t.strip() for t in config["iterative_eval_masked_tasks"].split(",") if t.strip()
        ]
    if not config.get("iterative_eval_masked_tasks"):
        config["iterative_eval_masked_tasks"] = list(config.get("iterative_eval_tasks", []))
    if isinstance(config.get("preserve_tasks", []), str):
        config["preserve_tasks"] = [t.strip() for t in config["preserve_tasks"].split(",") if t.strip()]
    if not config.get("preserve_tasks"):
        config["preserve_tasks"] = ["all_nonmasked"]

    if config.get("robust_profile", False):
        print("Applying robust profile defaults.")
        config["scheduler_type"] = "cosine_warmup"
        if config.get("mt_conflict_method", "none") == "none":
            config["mt_conflict_method"] = "pcgrad"
        config["monitor_metric"] = "val_full/total_loss"
        config["monitor_mode"] = "min"
        config["early_stopping"] = True
        config["lr"] = 1e-3
        config["weight_decay"] = 1e-2
        config["dropout"] = 0.4
        config["grad_clip_val"] = 0.5
        if use_cuda and ampere_or_newer:
            config["precision"] = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
        if config.get("preserve_pretrained", False):
            # No-regression finetuning should start from conservative rates.
            config["lr"] = float(config.get("preserve_stage_a_lr", 1e-4))

    if config.get("mt_conflict_method") in {"pcgrad", "cagrad", "gradnorm"} and config.get("mt_strategy") == "wloss":
        print(
            "Warning: mt_conflict_method with mt_strategy=wloss is not supported cleanly. "
            "Switching mt_strategy to fixed (non-learned task weights)."
        )
        config["mt_strategy"] = "fixed"
    if config.get("mt_conflict_method") in {"pcgrad", "cagrad", "gradnorm"} and config.get("mt_strategy") == "famo":
        print(
            "Warning: mt_conflict_method and mt_strategy=famo are incompatible. "
            "Switching mt_strategy to fixed."
        )
        config["mt_strategy"] = "fixed"

    if config.get("masked_prediction_train", False):
        if not config.get("masked_tasks"):
            raise ValueError(
                "--masked_prediction_train requires --masked_tasks "
                "(e.g. romanNumeral,localkey,quality,inversion,degree1,degree2)."
            )
        unknown_tasks = [t for t in config["masked_tasks"] if t not in TASK_DICT]
        if unknown_tasks:
            raise ValueError(f"Unknown masked task(s): {unknown_tasks}")
        if config.get("feedback_mode", "single_pass") != "single_pass":
            raise ValueError("Only --feedback_mode single_pass is currently supported.")
    if config.get("iterative_refine_train", False):
        keep_ratio_legacy = float(config.get("iterative_train_keep_ratio", 0.5))
        keep_ratio_start_cfg = float(config.get("iterative_train_keep_ratio_start", 0.05))
        keep_ratio_end_cfg = float(config.get("iterative_train_keep_ratio_end", 0.30))
        if keep_ratio_legacy != 0.5 and keep_ratio_start_cfg == 0.05 and keep_ratio_end_cfg == 0.30:
            config["iterative_train_keep_ratio_start"] = keep_ratio_legacy
            config["iterative_train_keep_ratio_end"] = keep_ratio_legacy
        pass2_weight_legacy = float(config.get("iterative_train_pass2_weight", 1.0))
        pass2_weight_start_cfg = float(config.get("iterative_train_pass2_weight_start", 0.25))
        pass2_weight_end_cfg = float(config.get("iterative_train_pass2_weight_end", 0.75))
        if pass2_weight_legacy != 1.0 and pass2_weight_start_cfg == 0.25 and pass2_weight_end_cfg == 0.75:
            config["iterative_train_pass2_weight_start"] = pass2_weight_legacy
            config["iterative_train_pass2_weight_end"] = pass2_weight_legacy
        steps = int(config.get("iterative_train_steps", 2))
        if steps < 1:
            raise ValueError("--iterative_train_steps must be >= 1.")
        if steps > 2:
            print("Warning: iterative_train_steps>2 not supported in v1; clamping to 2.")
            config["iterative_train_steps"] = 2
        keep_ratio = float(config.get("iterative_train_keep_ratio", 0.5))
        if keep_ratio < 0 or keep_ratio > 1:
            raise ValueError("--iterative_train_keep_ratio must be in [0, 1].")
        keep_ratio_start = float(config.get("iterative_train_keep_ratio_start", 0.05))
        keep_ratio_end = float(config.get("iterative_train_keep_ratio_end", 0.30))
        if keep_ratio_start < 0 or keep_ratio_start > 1:
            raise ValueError("--iterative_train_keep_ratio_start must be in [0, 1].")
        if keep_ratio_end < 0 or keep_ratio_end > 1:
            raise ValueError("--iterative_train_keep_ratio_end must be in [0, 1].")
        conf_min = float(config.get("iterative_train_conf_min", 0.80))
        if conf_min < 0 or conf_min > 1:
            raise ValueError("--iterative_train_conf_min must be in [0, 1].")
        min_rem = float(config.get("iterative_train_min_remaining_ratio", 0.20))
        if min_rem < 0 or min_rem > 1:
            raise ValueError("--iterative_train_min_remaining_ratio must be in [0, 1].")
        pass2_w_start = float(config.get("iterative_train_pass2_weight_start", 0.25))
        pass2_w_end = float(config.get("iterative_train_pass2_weight_end", 0.75))
        if pass2_w_start < 0 or pass2_w_end < 0:
            raise ValueError("--iterative_train_pass2_weight_start/end must be >= 0.")
        if not config.get("masked_prediction_train", False):
            print("Warning: --iterative_refine_train requires masked prediction; disabling iterative refine train.")
            config["iterative_refine_train"] = False
    if config.get("iterative_eval", False):
        eval_steps = int(config.get("iterative_eval_steps", 10))
        if eval_steps < 1:
            raise ValueError("--iterative_eval_steps must be >= 1.")
        keep_pct = float(config.get("iterative_eval_keep_percentile", 10.0))
        if keep_pct < 0 or keep_pct > 100:
            raise ValueError("--iterative_eval_keep_percentile must be in [0, 100].")
        unknown_eval_tasks = [
            t for t in config.get("iterative_eval_masked_tasks", []) if t not in TASK_DICT
        ]
        if unknown_eval_tasks:
            raise ValueError(f"Unknown iterative eval task(s): {unknown_eval_tasks}")

    if int(config.get("beam_width", 8)) < 1:
        raise ValueError("--beam_width must be >= 1.")
    config["beam_width"] = int(config.get("beam_width", 8))
    if config.get("beam_spec_json"):
        beam_spec_path = Path(str(config["beam_spec_json"])).expanduser()
        if not beam_spec_path.exists():
            raise ValueError(f"--beam_spec_json path not found: {beam_spec_path}")
        import json

        with open(beam_spec_path, "r", encoding="utf-8") as f:
            loaded_beam_spec = json.load(f)
        if not isinstance(loaded_beam_spec, dict):
            raise ValueError("--beam_spec_json must contain a JSON object.")
        config["beam_spec_config"] = loaded_beam_spec
    else:
        config["beam_spec_config"] = None
    if config.get("beam_eval_during_fit", False) and not config.get("beam_eval", False):
        print("Warning: --beam_eval_during_fit requires --beam_eval; disabling beam_eval_during_fit.")
        config["beam_eval_during_fit"] = False
    if config.get("beam_eval", False) and config.get("iterative_eval", False):
        print(
            "Warning: --beam_eval with --iterative_eval is not supported in v1 test logging. "
            "Disabling beam_eval for this run."
        )
        config["beam_eval"] = False
        config["beam_eval_during_fit"] = False

    aggregation_mode = str(config.get("aggregation_mode", "mean")).lower().strip()
    if aggregation_mode not in {"mean", "voter", "voter_consistent_beat"}:
        print(f"Warning: unknown aggregation_mode '{aggregation_mode}', falling back to 'mean'.")
        aggregation_mode = "mean"
    aggregation_voter_path = config.get("aggregation_voter_path")
    if aggregation_mode in {"voter", "voter_consistent_beat"}:
        if not aggregation_voter_path:
            print(
                "Warning: aggregation_mode requires aggregation_voter_path but none was provided; using mean."
            )
            aggregation_mode = "mean"
        elif not os.path.exists(aggregation_voter_path):
            print(
                f"Warning: aggregation_voter_path not found at '{aggregation_voter_path}'; using mean."
            )
            aggregation_mode = "mean"
    if config.get("aggregation_compare_mean_in_test", False):
        if not aggregation_voter_path or not os.path.exists(str(aggregation_voter_path)):
            print(
                "Warning: aggregation_compare_mean_in_test requires a valid aggregation_voter_path; "
                "disabling compare mode."
            )
            config["aggregation_compare_mean_in_test"] = False
    config["aggregation_mode"] = aggregation_mode
    if config.get("preserve_pretrained", False):
        if not config.get("masked_prediction_train", False):
            print(
                "Warning: --preserve_pretrained is enabled without --masked_prediction_train. "
                "Preservation losses will still run, but mixed masked/unmasked sampling is inactive."
            )
        if not (0.0 <= float(config.get("unmasked_batch_prob", 0.30)) <= 1.0):
            raise ValueError("--unmasked_batch_prob must be in [0, 1].")
    if config.get("known_ratio") is not None:
        config["mask_ratio"] = float(config["known_ratio"])
    else:
        config["known_ratio"] = float(config.get("mask_ratio", 0.15))

    if config.get("use_musicbert", False):
        has_alignments = bool(config.get("musicbert_alignment_dir"))
        has_cached_embeddings = bool(config.get("musicbert_cached_embeddings_dir"))
        if not has_alignments and not has_cached_embeddings:
            raise ValueError(
                "MusicBERT training requires either --musicbert_alignment_dir (runtime encoding) "
                "or --musicbert_cached_embeddings_dir (precomputed note embeddings)."
            )
    if config.get("disable_graph_encoder", False) and not config.get("use_musicbert", False):
        print("Warning: --disable_graph_encoder is enabled without --use_musicbert.")

    use_cached_embeddings = (
        config.get("use_musicbert", False)
        and bool(config.get("musicbert_cached_embeddings_dir"))
        and config.get("musicbert_freeze_backbone", True)
        and not config.get("musicbert_use_lora", False)
    )
    if config.get("use_musicbert", False) and config.get("musicbert_cached_embeddings_dir"):
        if not config.get("musicbert_freeze_backbone", True):
            print(
                "Warning: cached MusicBERT embeddings were provided but backbone is unfrozen; "
                "falling back to runtime MusicBERT forward passes."
            )
        if config.get("musicbert_use_lora", False):
            print(
                "Warning: cached MusicBERT embeddings were provided with LoRA enabled; "
                "falling back to runtime MusicBERT forward passes."
            )
    if config.get("musicbert_require_cached_embeddings", False) and not use_cached_embeddings:
        raise ValueError(
            "--musicbert_require_cached_embeddings requires frozen backbone without LoRA and "
            "--musicbert_cached_embeddings_dir."
        )
    if use_cached_embeddings and not config.get("musicbert_require_cached_embeddings", False):
        config["musicbert_require_cached_embeddings"] = True
        print("Enabling strict cached-embedding mode (all graphs must have cached MusicBERT embeddings).")
    config["musicbert_use_cached_embeddings"] = use_cached_embeddings
    if config.get("preserve_pretrained", False) and config.get("preserve_teacher_checkpoint") is None:
        if config.get("load_from_checkpoint", False) and config.get("checkpoint_path"):
            config["preserve_teacher_checkpoint"] = config["checkpoint_path"]

    if config["gpus"] == "-1":
        devices = 1
        accelerator = "cpu"
        use_ddp = False
    else:
        devices = [eval(gpu) for gpu in config["gpus"].split(",")]
        accelerator = "auto"
        use_ddp = len(devices) > 1

    if not config["cl_training"]:
        config["training_dataloader_type"] = "combined"

    datamodule = AnalysisDataModule(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        subgraph_size=config["subgraph_size"],
        num_neighbors=[5]*(config["num_layers"]-1),
        raw_dir=config["raw_dir"],
        force_reload=config["force_reload"],
        verbose=config["verbose"],
        tasks=list(config["task_dict"].keys()),
        random_split=config.get("random_split", False),
        max_samples=config.get("max_samples", None),
        main_tasks=config.get("main_tasks", ["cadence", "rna", "all"]),
        remove_beats=not config.get("add_beats", False),
        remove_measures= not config.get("add_measures", False),
        feature_type=config.get("feature_type", "cadence"),
        augment=config.get("use_transpositions", True),
        training_dataloader_type=config.get("training_dataloader_type", "sequential"),
        alignment_dir=config.get("musicbert_alignment_dir") if not use_cached_embeddings else None,
        require_alignment=config.get("use_musicbert", False) and not use_cached_embeddings,
        musicbert_embedding_cache_dir=config.get("musicbert_cached_embeddings_dir") if use_cached_embeddings else None,
        require_cached_embeddings=config.get("musicbert_require_cached_embeddings", False),
    )
    datamodule.setup()

    if datamodule.main_tasks != config.get("main_tasks"):
        config["main_tasks"] = datamodule.main_tasks
        if config.get("cl_training", False):
            config["epochs_per_task"] = [config["num_epochs"] // len(config["main_tasks"])] * len(config["main_tasks"])

    config["metadata"] = datamodule.metadata
    config["base_in_channels"] = datamodule.features
    config["in_channels"] = datamodule.features

    note_encoder = None
    if config.get("use_musicbert", False):
        if config.get("musicbert_use_cached_embeddings", False):
            musicbert_dim = config.get("musicbert_embedding_dim")
            if musicbert_dim is None:
                musicbert_dim = _infer_cached_embedding_dim(config["musicbert_cached_embeddings_dir"])
            config["musicbert_embedding_dim"] = int(musicbert_dim)
            config["musicbert_hidden_size"] = int(musicbert_dim)
            print(f"Using cached MusicBERT note embeddings (dim={musicbert_dim}).")
        else:
            adapter_cfg = MusicBertAdapterConfig(
                use_lora=config.get("musicbert_use_lora", False),
                lora_r=config.get("musicbert_lora_r", 8),
                lora_alpha=config.get("musicbert_lora_alpha", 16),
                lora_dropout=config.get("musicbert_lora_dropout", 0.1),
            )
            note_encoder = MusicBertNoteEncoder(
                pretrained_name=config.get("musicbert_model_name", "manoskary/musicbert-large"),
                adapter_cfg=adapter_cfg,
                freeze_backbone=config.get("musicbert_freeze_backbone", True),
            )
            musicbert_dim = note_encoder.backbone.model.config.hidden_size
            config["musicbert_hidden_size"] = int(musicbert_dim)
        fusion = config.get("musicbert_fusion", "replace")
        if fusion == "concat":
            config["in_channels"] = config["base_in_channels"] + musicbert_dim
        elif fusion == "gate":
            config["in_channels"] = config["base_in_channels"]
        else:
            config["in_channels"] = musicbert_dim

    if config["load_from_checkpoint"] and config["checkpoint_path"] is not None:
        # if checkpoint_path is url from wandb, download it
        if not os.path.exists(config["checkpoint_path"]):
            print(f"Checkpoint path {config['checkpoint_path']} does not exist! Trying to download from WANDB...")
            try:                
                run = wandb.init()
                artifact = run.use_artifact('melkisedeath/AnalysisGNN/model-zun976rt:v0', type='model')
                artifact_dir = artifact.download()
                config["checkpoint_path"] = artifact_dir + "/model.ckpt"
            except Exception:
                raise ValueError(f"Checkpoint path {config['checkpoint_path']} does not exist!")
        # Load model from checkpoint
        print(f"Loading model from checkpoint {config['checkpoint_path']}")
        if (
            config.get("preserve_pretrained", False)
            and (
                not config.get("preserve_teacher_checkpoint")
                or not os.path.exists(config.get("preserve_teacher_checkpoint"))
            )
        ):
            config["preserve_teacher_checkpoint"] = config["checkpoint_path"]
        ckpt = torch.load(config["checkpoint_path"], map_location="cpu")
        ckpt_state = ckpt.get("state_dict", {})
        ckpt_hparams = ckpt.get("hyper_parameters", {}) if isinstance(ckpt, dict) else {}
        # Align architecture-critical knobs to checkpoint values so eval/resume
        # does not fail on linear shape mismatches when CLI defaults differ.
        if isinstance(ckpt_hparams, dict):
            ckpt_fusion = ckpt_hparams.get("musicbert_fusion", None)
            if ckpt_fusion is not None and config.get("musicbert_fusion") != ckpt_fusion:
                print(
                    "Warning: overriding musicbert_fusion from checkpoint "
                    f"({config.get('musicbert_fusion')} -> {ckpt_fusion}) for shape compatibility."
                )
                config["musicbert_fusion"] = ckpt_fusion
            ckpt_base_in = ckpt_hparams.get("base_in_channels", None)
            if ckpt_base_in is not None:
                try:
                    ckpt_base_in = int(ckpt_base_in)
                    if config.get("base_in_channels") != ckpt_base_in:
                        print(
                            "Warning: overriding base_in_channels from checkpoint "
                            f"({config.get('base_in_channels')} -> {ckpt_base_in})."
                        )
                        config["base_in_channels"] = ckpt_base_in
                except Exception:
                    pass
            ckpt_in = ckpt_hparams.get("in_channels", None)
            if ckpt_in is not None:
                try:
                    ckpt_in = int(ckpt_in)
                    if config.get("in_channels") != ckpt_in:
                        print(
                            "Warning: overriding in_channels from checkpoint "
                            f"({config.get('in_channels')} -> {ckpt_in})."
                        )
                        config["in_channels"] = ckpt_in
                except Exception:
                    pass
            for k in ("musicbert_hidden_size", "musicbert_embedding_dim"):
                v = ckpt_hparams.get(k, None)
                if v is not None:
                    try:
                        config[k] = int(v)
                    except Exception:
                        pass
        ckpt_note_proj = ckpt_state.get("model.project_dict.note.0.weight", None)
        if isinstance(ckpt_note_proj, torch.Tensor) and ckpt_note_proj.ndim == 2:
            # project_dict.note consumes [x_note || pitch_emb || key_emb], where
            # pitch/key contribute 128 dims in total.
            inferred_in = int(ckpt_note_proj.shape[1]) - 128
            if inferred_in > 0 and int(config.get("in_channels", inferred_in)) != inferred_in:
                print(
                    "Warning: inferred in_channels from checkpoint note projection "
                    f"({config.get('in_channels')} -> {inferred_in})."
                )
                config["in_channels"] = inferred_in
        has_note_encoder = any(key.startswith("note_encoder.") for key in ckpt_state.keys())
        has_label_conditioning = any(
            key.startswith("label_condition_embeddings.") or key.startswith("label_condition_fusion.")
            for key in ckpt_state.keys()
        )
        has_wloss_params = any(key == "clf_loss.params" or key.startswith("clf_loss.params.") for key in ckpt_state.keys())
        ckpt_mt_strategy = ckpt_hparams.get("mt_strategy", None)
        current_mt_strategy = config.get("mt_strategy", None)
        mt_strategy_mismatch = (
            (current_mt_strategy == "wloss" and not has_wloss_params)
            or (current_mt_strategy != "wloss" and has_wloss_params)
            or (ckpt_mt_strategy is not None and ckpt_mt_strategy != current_mt_strategy)
        )
        if mt_strategy_mismatch:
            print(
                "Warning: checkpoint/model mt_strategy mismatch "
                f"(checkpoint={ckpt_mt_strategy}, current={current_mt_strategy}, "
                f"checkpoint_has_wloss_params={has_wloss_params}). "
                "Loading with strict=False."
            )
        if has_note_encoder and note_encoder is None and not config.get("musicbert_use_cached_embeddings", False):
            raise ValueError(
                "Checkpoint contains MusicBERT note encoder weights, but --use_musicbert "
                "is not enabled. Re-run with --use_musicbert (and LoRA flags if needed)."
            )
        if note_encoder is not None:
            strict = True
            if config.get("musicbert_use_lora", False) or not has_note_encoder:
                strict = False
            if config.get("masked_prediction_train", False) and not has_label_conditioning:
                strict = False
            if has_label_conditioning and not config.get("masked_prediction_train", False):
                strict = False
            if mt_strategy_mismatch:
                strict = False
            model = ContinualAnalysisGNN.load_from_checkpoint(
                config["checkpoint_path"],
                hparams=config,
                note_encoder=note_encoder,
                strict=strict,
            )
        else:
            strict = not has_note_encoder
            if config.get("masked_prediction_train", False) and not has_label_conditioning:
                strict = False
            if has_label_conditioning and not config.get("masked_prediction_train", False):
                strict = False
            if mt_strategy_mismatch:
                strict = False
            model = ContinualAnalysisGNN.load_from_checkpoint(
                config["checkpoint_path"],
                hparams=config,
                strict=strict,
            )
        model.note_encoder = note_encoder
        model.musicbert_use_cached_embeddings = config.get("musicbert_use_cached_embeddings", False)
        model.current_task = config["main_tasks"][0]
    else:
        model = ContinualAnalysisGNN(config, note_encoder=note_encoder)

    if not config.get("disable_graph_encoder", False) and config["model"] == "MetricalGNN":
        if config["add_measures"] and config["add_beats"]:
            config["model"] = "MetricalGNN"
        elif config["add_measures"]:
            config["model"] = "MeasureGNN"
        elif config["add_beats"]:
            config["model"] = "BeatGNN"
        else:
            config["model"] = "NoteGNN"


    if config["compile"]:
        model = torch.compile(model, dynamic=True)

    model_arch = f"{config['model']}-heads-only" if config.get("disable_graph_encoder", False) else config["model"]
    model_name = f"{model_arch}_{config['num_layers']}x{config['hidden_channels']}-dropout={config['dropout']}-lr={config['lr']}-wd={config['weight_decay']}"

    wandb_logger = None
    if config["use_wandb"]:

        task_group = "-".join(config["main_tasks"])
        task_group_tag = _compact_task_tag(
            config.get("main_tasks", []), field_name="main_tasks_group"
        )
        musicbert_tag = "mb"
        if not config.get("use_musicbert", False):
            musicbert_tag = "no-mb"
        elif config.get("musicbert_use_cached_embeddings", False):
            musicbert_tag = "mb-cache"
        elif config.get("musicbert_use_lora", False):
            musicbert_tag = "mb-lora"
        elif config.get("musicbert_freeze_backbone", True):
            musicbert_tag = "mb-frozen"
        else:
            musicbert_tag = "mb-unfrozen"

        aug = "aug" if config.get("use_transpositions", True) else "noaug"
        feature_tag = _sanitize_wandb_token(config.get("feature_type", "cadence"))
        arch_tag = "no-gnn" if config.get("disable_graph_encoder", False) else "gnn"
        masked_tag = "masked" if config.get("masked_prediction_train", False) else "nomasked"
        masked_tasks_tag = (
            _compact_task_tag(
                config.get("masked_tasks", []),
                field_name="masked_tasks_group",
                max_items=4,
                max_len=36,
            )
            if config.get("masked_prediction_train", False)
            else "none"
        )
        preserve_tag = "preserve" if config.get("preserve_pretrained", False) else "nopreserve"
        iterative_tag = "iterrefine" if config.get("iterative_refine_train", False) else "noiterrefine"
        iterative_eval_tag = "itereval" if config.get("iterative_eval", False) else "noitereval"
        iterative_eval_zero_known_tag = (
            "iterzero"
            if config.get("iterative_eval", False) and config.get("iterative_eval_zero_known", True)
            else "iternonzero"
        )
        beam_tag = "beam_eval" if config.get("beam_eval", False) else "no_beam_eval"
        beam_width_tag = f"beam_w{int(config.get('beam_width', 8))}"
        beam_cfg_tag = ""
        if config.get("beam_eval", False) and isinstance(config.get("beam_spec_config"), dict):
            try:
                beam_cfg_json = str(config["beam_spec_config"])
                beam_cfg_hash = hashlib.sha1(beam_cfg_json.encode("utf-8")).hexdigest()[:8]
                beam_cfg_tag = f"beam_cfg_{beam_cfg_hash}"
            except Exception:
                beam_cfg_tag = ""
        phase = "train+eval" if args.do_train and args.do_eval else ("train" if args.do_train else ("eval" if args.do_eval else "run"))
        ckpt_tag = ""
        if args.do_eval and not args.do_train and config.get("checkpoint_path"):
            ckpt_parent = os.path.basename(os.path.dirname(os.path.dirname(config["checkpoint_path"])))
            ckpt_tag = f"-ckpt={ckpt_parent}"
        run_name = (
            f"{phase}-{model_arch}"
            f"-tasks={task_group_tag}"
            f"-feat={feature_tag}"
                f"-{musicbert_tag}"
                f"-{arch_tag}"
                f"-{aug}"
                f"-{masked_tag}"
                f"-mtasks={masked_tasks_tag}"
                f"-{preserve_tag}"
                f"-{iterative_tag}"
                f"-{iterative_eval_tag}"
                f"-{iterative_eval_zero_known_tag}"
                f"-sched={_sanitize_wandb_token(config['scheduler_type'])}"
                f"-conf={_sanitize_wandb_token(config['mt_conflict_method'])}"
                f"-ep={config['num_epochs']}"
                f"-bs={config['batch_size']}"
                f"-lr={config['lr']}{ckpt_tag}"
            )
        run_name = _clip_wandb_label(run_name, max_len=128, field_name="name")
        group = (
            f"{task_group_tag}-{feature_tag}-{musicbert_tag}-{arch_tag}-{aug}-"
            f"{masked_tag}-{masked_tasks_tag}-{preserve_tag}-{iterative_tag}-"
            f"{iterative_eval_tag}-{iterative_eval_zero_known_tag}-"
            f"{_sanitize_wandb_token(config['scheduler_type'])}-"
            f"{_sanitize_wandb_token(config['mt_conflict_method'])}"
        )
        group = _clip_wandb_label(group, max_len=128, field_name="group")
        job_type = phase
        user_tags = args.tags.split(",") if args.tags != "" else []
        tags = [
            t
            for t in [
                phase,
                task_group_tag,
                feature_tag,
                musicbert_tag,
                arch_tag,
                aug,
                masked_tag,
                masked_tasks_tag,
                preserve_tag,
                iterative_tag,
                iterative_eval_tag,
                iterative_eval_zero_known_tag,
                beam_tag,
                beam_width_tag,
                beam_cfg_tag,
                config["scheduler_type"],
                config["mt_conflict_method"],
            ] + user_tags
            if t
        ]

        wandb_logger_kwargs = dict(
            config=config,
            project=config["wandb_project"],
            entity=config["wandb_entity"],
            group=group,
            job_type=job_type,
            name=run_name,
            tags=tags,
            log_model=True,
        )
        try:
            wandb_logger = WandbLogger(**wandb_logger_kwargs)
            wandb_logger.log_hyperparams(args)
        except Exception as exc:
            print(
                "Warning: W&B logger initialization failed "
                f"({type(exc).__name__}: {exc}). Retrying in offline mode."
            )
            try:
                if wandb.run is not None:
                    wandb.finish()
            except Exception:
                pass
            os.environ["WANDB_MODE"] = "offline"
            try:
                wandb_logger = WandbLogger(**wandb_logger_kwargs)
                wandb_logger.log_hyperparams(args)
                print("W&B offline logging enabled.")
            except Exception as offline_exc:
                print(
                    "Warning: W&B offline fallback failed "
                    f"({type(offline_exc).__name__}: {offline_exc}). Continuing without W&B logger."
                )
                wandb_logger = None
                config["use_wandb"] = False

    monitor_metric = config.get("monitor_metric", "val_full/total_loss")
    monitor_mode = config.get("monitor_mode", "min")
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor=monitor_metric, mode=monitor_mode, save_last=True)
    # Set up spawn strategy
    # strategy = DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True) if use_ddp else "auto"

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [checkpoint_callback, lr_monitor]
    if config.get("use_swa", False):
        swa = StochasticWeightAveraging(swa_lrs=5e-5, swa_epoch_start=50)
        callbacks.append(swa)
    if config.get("early_stopping", True):
        callbacks.append(
            WarmupEarlyStopping(
                monitor=monitor_metric,
                mode=monitor_mode,
                patience=config.get("early_stop_patience", 12),
                min_delta=config.get("early_stop_min_delta", 0.002),
                start_epoch=config.get("early_stop_start_epoch", 12),
                check_finite=True,
                strict=False,
            )
        )
    manual_optimization = config.get("mt_strategy") == "famo" or config.get("mt_conflict_method") in {"pcgrad", "cagrad", "gradnorm"}
    if manual_optimization and int(config.get("accumulate_grad_batches", 1)) > 1:
        print(
            "Warning: manual optimization (pcgrad/cagrad/gradnorm/famo) does not support "
            "Trainer(accumulate_grad_batches>1). Forcing accumulate_grad_batches=1."
        )
        config["accumulate_grad_batches"] = 1
    gradient_clip_val = 0.0 if manual_optimization else float(config.get("grad_clip_val", 1.0))
    trainer = Trainer(
        max_epochs=config["num_epochs"], accelerator=accelerator, devices=devices,
        # strategy=strategy,
        num_sanity_val_steps=int(config.get("num_sanity_val_steps", 0)),
        logger=wandb_logger if config["use_wandb"] else None,
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=int(config.get("reload_dataloaders_every_n_epochs", 0)),
        log_every_n_steps=1,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=config["accumulate_grad_batches"],
        precision=config["precision"],
    )

    print(f"Training model {model_name} with config: {config}")

    if args.do_train:
        trainer.fit(model, datamodule=datamodule)

    if args.do_eval:
        # always test on all tasks, to do that reload the datamodule and update the main tasks
        datamodule = AnalysisDataModule(
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            subgraph_size=config["subgraph_size"],
            num_neighbors=[5]*(config["num_layers"]-1),
            raw_dir=config["raw_dir"],
            force_reload=config["force_reload"],
            verbose=config["verbose"],
            tasks=list(config["task_dict"].keys()),
            random_split=config.get("random_split", False),
            max_samples=config.get("max_samples", None),
            main_tasks=["cadence", "rna", "all"],
            remove_beats=not config.get("add_beats", False),
            remove_measures= not config.get("add_measures", False),
            feature_type=config.get("feature_type", "cadence"),
            augment=config.get("use_transpositions", True),
            training_dataloader_type=config.get("training_dataloader_type", "sequential"),
            alignment_dir=config.get("musicbert_alignment_dir") if not config.get("musicbert_use_cached_embeddings", False) else None,
            require_alignment=config.get("use_musicbert", False) and not config.get("musicbert_use_cached_embeddings", False),
            musicbert_embedding_cache_dir=config.get("musicbert_cached_embeddings_dir") if config.get("musicbert_use_cached_embeddings", False) else None,
            require_cached_embeddings=config.get("musicbert_require_cached_embeddings", False),
        )
        datamodule.setup()
        # Test on the checkpoint produced by the current run when training happened in this process.
        if args.do_train:
            ckpt_path = checkpoint_callback.best_model_path
            if not ckpt_path:
                ckpt_path = checkpoint_callback.last_model_path or "last"
            trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)
        elif config["load_from_checkpoint"] and config["checkpoint_path"] is not None:
            # For eval-only with masked conditioning enabled, keep the in-memory model loaded
            # with strict=False compatibility instead of forcing strict checkpoint restore.
            if config.get("masked_prediction_train", False):
                trainer.test(model, datamodule=datamodule, ckpt_path=None)
            else:
                trainer.test(model, datamodule=datamodule, ckpt_path=config["checkpoint_path"])
        else:
            ckpt_path = checkpoint_callback.best_model_path if checkpoint_callback.best_model_path else "last"
            trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
