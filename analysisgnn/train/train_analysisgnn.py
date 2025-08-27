from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.backends.opt_einsum import strategy
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import StochasticWeightAveraging
from analysisgnn.models.analysis import ContinualAnalysisGNN
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
    parser.add_argument("--use_jk", help="Use Jumping Knowledge", action="store_true")
    parser.add_argument("--tags", type=str, default="", help="Tags to add to the WandB run api")
    parser.add_argument("--homogeneous", action="store_true", help="Use homogeneous graphs")
    parser.add_argument("--reg_loss_type", type=str, default="la", help="Use different regularization loss")
    parser.add_argument("--raw_dir", type=str, default=None, help="Raw directory to use")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--auto_batch_size", type=bool, help="Automatically find optimal batch size", default=True)
    parser.add_argument("--use_reledge", action="store_true", help="Use reledge")
    parser.add_argument("--use_wandb", help="Use wandb", action="store_true",)
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

    if args.config_path is not None:
        import json
        args_config = config.copy()
        with open(args.config_path, "r") as f:
            config = json.load(f)

        for k, v in args_config.items():
            if k not in config.keys():
                config[k] = v

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
    )
    datamodule.setup()

    config["metadata"] = datamodule.metadata
    config["in_channels"] = datamodule.features

    if config["load_from_checkpoint"] and config["checkpoint_path"] is not None:
        # if checkpoint_path is url from wandb, download it
        if not os.path.exists(config["checkpoint_path"]):
            print(f"Checkpoint path {config['checkpoint_path']} does not exist! Trying to download from WANDB...")
            try:                
                run = wandb.init()
                artifact = run.use_artifact('melkisedeath/AnalysisGNN/model-zun976rt:v0', type='model')
                artifact_dir = artifact.download()
                config["checkpoint_path"] = artifact_dir + "/model.ckpt"
            except :
                raise ValueError(f"Checkpoint path {config['checkpoint_path']} does not exist!")
        # Load model from checkpoint
        print(f"Loading model from checkpoint {config['checkpoint_path']}")                    
        model = ContinualAnalysisGNN.load_from_checkpoint(config["checkpoint_path"])
        model.current_task = config["main_tasks"][0]
    else:
        model = ContinualAnalysisGNN(config)

    if config["model"] == "MetricalGNN":
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

    model_name = f"{config['model']}_{config['num_layers']}x{config['hidden_channels']}-dropout={config['dropout']}-lr={config['lr']}-wd={config['weight_decay']}"

    if config["use_wandb"]:

        group = f"{'_'.join(config['main_tasks'])}"

        ft = config.get("feature_type", "cadence")
        aug = "aug" if config.get("use_transpositions", True) else "noaug"
        job_type = f"{aug}-{ft}_features"        

        wandb.init(
            project="AnalysisGNN",
            entity="melkisedeath",
            group=group,
            job_type=job_type,
            name=model_name,
            tags=args.tags.split(",") if args.tags != "" else None,
            config=config,
        )

        wandb_logger = WandbLogger(
            config=config,
            project="AnalysisGNN",
            entity="melkisedeath",
            group=group,
            job_type=f"{aug}/{ft}-features",
            name=model_name,
            tags=args.tags.split(",") if args.tags != "" else None,
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
    trainer = Trainer(
        max_epochs=config["num_epochs"]+1, accelerator=accelerator, devices=devices,
        # strategy=strategy,
        num_sanity_val_steps=3,
        logger=wandb_logger if config["use_wandb"] else None,
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=1,
        gradient_clip_val=1.0,
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
        )
        datamodule.setup()
        # Test on best model
        if config["load_from_checkpoint"] and config["checkpoint_path"] is not None:
            trainer.test(model, datamodule=datamodule, ckpt_path=config["checkpoint_path"])
        else:
            trainer.test(model, datamodule=datamodule, ckpt_path=checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()