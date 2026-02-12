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
        help="Trainer precision (e.g., 16-mixed, bf16-mixed, 32-true). Defaults to 16-mixed on CUDA.",
    )
    parser.add_argument('--num_layers', type=int, default=3,
                        help="Number of layers on the Graph Convolutional Encoder Network")
    parser.add_argument('--hidden_channels', type=int, default=256, help="Number of hidden units")
    parser.add_argument('--out_channels', type=int, default=128, help="Number of output units")
    parser.add_argument('--num_epochs', type=str, default="50", help="Number of epochs")
    parser.add_argument('--dropout', type=float, default=0.3, help="Dropout")
    parser.add_argument('--lr', type=float, default=0.005, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=5e-3, help="Weight decay")
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
        "--musicbert_fusion",
        type=str,
        default="replace",
        choices=["replace", "concat", "gate"],
        help="How to fuse MusicBERT embeddings with note features.",
    )
    parser.add_argument(
        "--mt_conflict_method",
        type=str,
        default="none",
        choices=["none", "pcgrad", "gradnorm"],
        help="Conflict mitigation for multitask losses.",
    )
    parser.add_argument(
        "--gradnorm_alpha",
        type=float,
        default=1.5,
        help="GradNorm alpha parameter (only used when mt_conflict_method=gradnorm).",
    )
    return parser


def main():
    parser = get_parser()

    args = parser.parse_args()
    args.main_tasks = args.main_tasks.split(",")
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
    if config.get("precision") is None:
        if config["gpus"] != "-1" and torch.cuda.is_available():
            config["precision"] = "16-mixed"
        else:
            config["precision"] = "32-true"

    if args.config_path is not None:
        import json
        args_config = config.copy()
        with open(args.config_path, "r") as f:
            config = json.load(f)

        for k, v in args_config.items():
            if k not in config.keys():
                config[k] = v

    if config.get("use_musicbert", False) and not config.get("musicbert_alignment_dir"):
        raise ValueError("MusicBERT training requires --musicbert_alignment_dir with .npz alignments.")
    if config.get("disable_graph_encoder", False) and not config.get("use_musicbert", False):
        print("Warning: --disable_graph_encoder is enabled without --use_musicbert.")

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
        alignment_dir=config.get("musicbert_alignment_dir"),
        require_alignment=config.get("use_musicbert", False),
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
        ckpt = torch.load(config["checkpoint_path"], map_location="cpu")
        ckpt_state = ckpt.get("state_dict", {})
        has_note_encoder = any(key.startswith("note_encoder.") for key in ckpt_state.keys())
        if has_note_encoder and note_encoder is None:
            raise ValueError(
                "Checkpoint contains MusicBERT note encoder weights, but --use_musicbert "
                "is not enabled. Re-run with --use_musicbert (and LoRA flags if needed)."
            )
        if note_encoder is not None:
            strict = True
            if config.get("musicbert_use_lora", False) or not has_note_encoder:
                strict = False
            model = ContinualAnalysisGNN.load_from_checkpoint(
                config["checkpoint_path"],
                note_encoder=note_encoder,
                strict=strict,
            )
        else:
            model = ContinualAnalysisGNN.load_from_checkpoint(config["checkpoint_path"])
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

    if config["use_wandb"]:

        task_group = "-".join(config["main_tasks"])
        musicbert_tag = "mb"
        if not config.get("use_musicbert", False):
            musicbert_tag = "no-mb"
        elif config.get("musicbert_use_lora", False):
            musicbert_tag = "mb-lora"
        elif config.get("musicbert_freeze_backbone", True):
            musicbert_tag = "mb-frozen"
        else:
            musicbert_tag = "mb-unfrozen"

        aug = "aug" if config.get("use_transpositions", True) else "noaug"
        feature_tag = config.get("feature_type", "cadence")
        arch_tag = "no-gnn" if config.get("disable_graph_encoder", False) else "gnn"
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
            f"-ep={config['num_epochs']}"
            f"-bs={config['batch_size']}"
            f"-lr={config['lr']}{ckpt_tag}"
        )
        group = f"{task_group}-{feature_tag}-{musicbert_tag}-{arch_tag}-{aug}"
        job_type = phase
        user_tags = args.tags.split(",") if args.tags != "" else []
        tags = [t for t in [phase, task_group, feature_tag, musicbert_tag, arch_tag, aug] + user_tags if t]

        wandb_logger = WandbLogger(
            config=config,
            project=config["wandb_project"],
            entity=config["wandb_entity"],
            group=group,
            job_type=job_type,
            name=run_name,
            tags=tags,
            log_model=True,
        )
        wandb_logger.log_hyperparams(args)

    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val/total_loss", mode="min", save_last=True)
    # Set up spawn strategy
    # strategy = DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True) if use_ddp else "auto"

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [checkpoint_callback, lr_monitor]
    if config.get("use_swa", False):
        swa = StochasticWeightAveraging(swa_lrs=5e-5, swa_epoch_start=50)
        callbacks.append(swa)
    manual_optimization = config.get("mt_strategy") == "famo" or config.get("mt_conflict_method") in {"pcgrad", "gradnorm"}
    gradient_clip_val = 0.0 if manual_optimization else 1.0
    trainer = Trainer(
        max_epochs=config["num_epochs"]+1, accelerator=accelerator, devices=devices,
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
            alignment_dir=config.get("musicbert_alignment_dir"),
            require_alignment=config.get("use_musicbert", False),
        )
        datamodule.setup()
        # Test on best model
        if config["load_from_checkpoint"] and config["checkpoint_path"] is not None:
            trainer.test(model, datamodule=datamodule, ckpt_path=config["checkpoint_path"])
        else:
            trainer.test(model, datamodule=datamodule, ckpt_path=checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()
