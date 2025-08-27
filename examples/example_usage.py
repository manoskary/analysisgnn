#!/usr/bin/env python3
"""
Example usage of AnalysisGNN for training and prediction

This script demonstrates how to:
1. Set up a training configuration
2. Train an AnalysisGNN model
3. Use the trained model for prediction
"""

import os
import json
from analysisgnn.models.analysis import ContinualAnalysisGNN
from analysisgnn.data.datamodules.analysis import AnalysisDataModule
from pytorch_lightning import Trainer


def example_training():
    """Example of how to train an AnalysisGNN model."""
    
    # Configuration for training
    config = {
        "model": "HybridGNN",
        "num_layers": 3,
        "hidden_channels": 256,
        "out_channels": 128,
        "dropout": 0.3,
        "lr": 0.005,
        "weight_decay": 0.005,
        "batch_size": 32,
        "num_epochs": 10,  # Small for demo
        "main_tasks": ["cadence", "localkey"],
        "add_beats": True,
        "add_measures": False,
        "feature_type": "cadence",
        "gpus": "-1",  # CPU only for demo
        "num_workers": 2,
        "subgraph_size": 200,  # Smaller for demo
        "cl_training": False,
        "use_transpositions": True,
        "verbose": True,
        "max_samples": 100,  # Limit samples for demo
        "task_dict": {
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
            "downbeat": 45,
            "note_degree": 49,
            "staff": 4,
        }
    }
    
    print("Setting up data module...")
    # Set up data module
    datamodule = AnalysisDataModule(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        subgraph_size=config["subgraph_size"],
        num_neighbors=[5] * (config["num_layers"] - 1),
        force_reload=False,
        verbose=config["verbose"],
        tasks=list(config["task_dict"].keys()),
        main_tasks=config["main_tasks"],
        remove_beats=not config.get("add_beats", False),
        remove_measures=not config.get("add_measures", False),
        feature_type=config.get("feature_type", "cadence"),
        augment=config.get("use_transpositions", True),
        max_samples=config.get("max_samples", None),
    )
    
    try:
        datamodule.setup()
        print("✓ Data module setup successful")
        
        # Update config with data info
        config["metadata"] = datamodule.metadata
        config["in_channels"] = datamodule.features
        
        print("Creating model...")
        # Create model
        model = ContinualAnalysisGNN(config)
        print("✓ Model created successfully")
        
        print("Setting up trainer...")
        # Set up trainer
        trainer = Trainer(
            max_epochs=config["num_epochs"],
            accelerator="cpu",
            devices=1,
            num_sanity_val_steps=1,
            log_every_n_steps=10,
            limit_train_batches=10,  # Limit for demo
            limit_val_batches=5,     # Limit for demo
        )
        print("✓ Trainer setup successful")
        
        print("Starting training...")
        # Train model (commented out to avoid long training time)
        # trainer.fit(model, datamodule=datamodule)
        print("✓ Training completed (skipped for demo)")
        
        return model, datamodule
        
    except Exception as e:
        print(f"✗ Error during training setup: {e}")
        return None, None


def example_prediction():
    """Example of how to use AnalysisGNN for prediction."""
    
    print("\nPrediction Example:")
    print("For prediction, you would typically:")
    print("1. Load a trained checkpoint")
    print("2. Load a musicXML score") 
    print("3. Use the prediction script:")
    print("   analysisgnn-predict --checkpoint_path model.ckpt --input_score score.musicxml")
    
    # Note: Actual prediction requires a trained model and musicXML file


def main():
    """Run the example."""
    print("AnalysisGNN Example Usage")
    print("=" * 40)
    
    print("\n1. Training Example:")
    model, datamodule = example_training()
    
    if model is not None:
        print("✓ Training example completed successfully")
    else:
        print("✗ Training example failed")
    
    example_prediction()
    
    print("\n" + "=" * 40)
    print("Example completed!")
    print("\nNext steps:")
    print("- Install dependencies: pip install -r requirements.txt")
    print("- Train a model: analysisgnn-train --do_train --main_tasks cadence,localkey")
    print("- Make predictions: analysisgnn-predict --checkpoint_path model.ckpt --input_score score.musicxml")


if __name__ == "__main__":
    main()
