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
from pathlib import Path
import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.tuner import Tuner


# for repeatability
seed_everything(0, workers=True)
torch.multiprocessing.set_sharing_strategy("file_system")

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
    parser.add_argument("--monitor_metric", type=str, default="val/total_loss",
                        help="Metric used by checkpointing/early stopping and plateau scheduler.")
    parser.add_argument("--monitor_mode", type=str, default="min", choices=["min", "max"],
                        help="Optimization mode for monitor_metric.")
    parser.add_argument("--early_stop_patience", type=int, default=12, help="Early stopping patience.")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.002, help="Early stopping min_delta.")
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
        default=15,
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
        default=1e-4,
        help="Stage-A learning rate (preserve_pretrained).",
    )
    parser.add_argument(
        "--preserve_stage_b_lr",
        type=float,
        default=5e-5,
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
    if isinstance(config.get("preserve_tasks", []), str):
        config["preserve_tasks"] = [t.strip() for t in config["preserve_tasks"].split(",") if t.strip()]
    if not config.get("preserve_tasks"):
        config["preserve_tasks"] = ["all_nonmasked"]

    if config.get("robust_profile", False):
        print("Applying robust profile defaults.")
        config["scheduler_type"] = "cosine_warmup"
        if config.get("mt_conflict_method", "none") == "none":
            config["mt_conflict_method"] = "pcgrad"
        config["monitor_metric"] = "val/total_loss"
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
        feature_tag = config.get("feature_type", "cadence")
        arch_tag = "no-gnn" if config.get("disable_graph_encoder", False) else "gnn"
        masked_tag = "masked" if config.get("masked_prediction_train", False) else "nomasked"
        masked_tasks_tag = "-".join(config.get("masked_tasks", [])) if config.get("masked_prediction_train", False) else "none"
        preserve_tag = "preserve" if config.get("preserve_pretrained", False) else "nopreserve"
        phase = "train+eval" if args.do_train and args.do_eval else ("train" if args.do_train else ("eval" if args.do_eval else "run"))
        ckpt_tag = ""
        if args.do_eval and not args.do_train and config.get("checkpoint_path"):
            ckpt_parent = os.path.basename(os.path.dirname(os.path.dirname(config["checkpoint_path"])))
            ckpt_tag = f"-ckpt={ckpt_parent}"
        run_name = (
            f"{phase}-{model_arch}"
            f"-tasks={task_group}"
            f"-feat={feature_tag}"
                f"-{musicbert_tag}"
                f"-{arch_tag}"
                f"-{aug}"
                f"-{masked_tag}"
                f"-mtasks={masked_tasks_tag}"
                f"-{preserve_tag}"
                f"-sched={config['scheduler_type']}"
                f"-conf={config['mt_conflict_method']}"
                f"-ep={config['num_epochs']}"
                f"-bs={config['batch_size']}"
                f"-lr={config['lr']}{ckpt_tag}"
            )
        group = (
            f"{task_group}-{feature_tag}-{musicbert_tag}-{arch_tag}-{aug}-"
            f"{masked_tag}-{masked_tasks_tag}-{preserve_tag}-{config['scheduler_type']}-{config['mt_conflict_method']}"
        )
        job_type = phase
        user_tags = args.tags.split(",") if args.tags != "" else []
        tags = [
            t
            for t in [
                phase,
                task_group,
                feature_tag,
                musicbert_tag,
                arch_tag,
                aug,
                masked_tag,
                masked_tasks_tag,
                preserve_tag,
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

    monitor_metric = config.get("monitor_metric", "val/total_loss")
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
            EarlyStopping(
                monitor=monitor_metric,
                mode=monitor_mode,
                patience=config.get("early_stop_patience", 12),
                min_delta=config.get("early_stop_min_delta", 0.002),
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
        num_sanity_val_steps=3,
        logger=wandb_logger if config["use_wandb"] else None,
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
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
