#!/usr/bin/env python3
"""
AnalysisGNN Prediction Script

This script performs music analysis using the AnalysisGNN model.
It can predict multiple analysis tasks including cadence detection, 
key analysis, harmonic analysis, and more.
"""

import pandas as pd
import copy
import os
import numpy as np
import partitura as pt
import torch
import argparse
from typing import Dict, List, Optional

# AnalysisGNN imports  
from analysisgnn.models.analysis import ContinualAnalysisGNN
from analysisgnn.utils.chord_representations_latest import available_representations
from analysisgnn.utils.chord_representations import formatRomanNumeral


def get_parser() -> argparse.ArgumentParser:
    """Get command line argument parser."""
    parser = argparse.ArgumentParser(
        description="AnalysisGNN: Multi-task Music Analysis with Graph Neural Networks"
    )
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        required=False,
        help="Path to AnalysisGNN checkpoint file (.ckpt)"
    )
    parser.add_argument(
        "--input_score", 
        type=str, 
        required=False,
        help="Path to input musicXML score"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./outputs",
        help="Directory to save outputs (default: ./outputs)"
    )
    parser.add_argument(
        "--tasks", 
        type=str, 
        default="cadence,localkey,tonkey,quality,romanNumeral",
        help="Comma-separated list of analysis tasks to perform"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu",
        help="Device to use for inference (cpu, cuda)"
    )
    parser.add_argument(
        "--wandb_artifact", 
        type=str, 
        default="melkisedeath/AnalysisGNN/model-uvj2ddun:v1",
        help="W&B artifact path (e.g., 'user/project/model-id:version')"
    )
    parser.add_argument(
        "--export_roman_numerals", 
        action="store_true",
        help="Export roman numeral analysis as musicXML"
    )
    parser.add_argument(
        "--export_csv", 
        action="store_true",
        help="Export analysis results as CSV"
    )
    return parser


def download_wandb_checkpoint(artifact_path: str) -> str:
    """Download checkpoint from Weights & Biases, or use cached version if available."""
    # Create artifacts directory structure
    artifacts_dir = "./artifacts/models"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Check if checkpoint already exists directly in artifacts/models
    checkpoint_path = os.path.join(artifacts_dir, "model.ckpt")
    if os.path.exists(checkpoint_path):
        print(f"Using cached checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    # Check for any .ckpt file in the artifacts/models directory
    if os.path.exists(artifacts_dir):
        for fname in os.listdir(artifacts_dir):
            if fname.endswith('.ckpt'):
                checkpoint_path = os.path.join(artifacts_dir, fname)
                print(f"Using cached checkpoint: {checkpoint_path}")
                return checkpoint_path
    
    # Check artifact-specific subdirectory
    artifact_dir = os.path.join(artifacts_dir, os.path.basename(artifact_path))
    checkpoint_path = os.path.join(artifact_dir, "model.ckpt")
    if os.path.exists(checkpoint_path):
        print(f"Using cached checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    # Only import and use wandb if checkpoint is not cached
    try:
        import wandb
        print(f"Downloading checkpoint from W&B: {artifact_path}")
        run = wandb.init()
        artifact = run.use_artifact(artifact_path, type='model')
        artifact_dir = artifact.download(root=artifacts_dir)
        wandb.finish()
        
        # Find the checkpoint file
        checkpoint_path = os.path.join(artifact_dir, "model.ckpt")
        if not os.path.exists(checkpoint_path):
            for fname in os.listdir(artifact_dir):
                if fname.endswith('.ckpt'):
                    checkpoint_path = os.path.join(artifact_dir, fname)
                    break
        
        return checkpoint_path
    except ImportError:
        raise ImportError("wandb package is required to download artifacts. Install with: pip install wandb")


def download_default_score() -> str:
    """Download the default Mozart quartet score."""
    import urllib.request
    
    url = "https://raw.githubusercontent.com/manoskary/humdrum-mozart-quartets/refs/heads/master/musicxml/k158-01.musicxml"
    
    # Create artifacts directory if it doesn't exist
    artifacts_dir = "./artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    default_score_path = os.path.join(artifacts_dir, "k158-01.musicxml")
    
    if not os.path.exists(default_score_path):
        print(f"Downloading default score from: {url}")
        urllib.request.urlretrieve(url, default_score_path)
        print(f"Default score saved to: {default_score_path}")
    else:
        print(f"Using cached default score: {default_score_path}")
    
    return default_score_path


def load_model(checkpoint_path: str, device: str = "cpu") -> ContinualAnalysisGNN:
    """Load AnalysisGNN model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    model = ContinualAnalysisGNN.load_from_checkpoint(
        checkpoint_path, 
        map_location=device
    )
    model.eval()
    model.to(device)
    return model


def predict_analysis(
    model: ContinualAnalysisGNN, 
    score: pt.score.Score, 
    tasks: List[str],
    device: str = "cpu"
) -> Dict[str, np.ndarray]:
    """Perform music analysis prediction."""
    
    with torch.no_grad():
        # Get predictions from model
        predictions = model.predict(score)
    
    # Decode predictions
    decoded_predictions = {}
    for task in tasks:
        if task in predictions:
            pred_tensor = predictions[task]
            if len(pred_tensor.shape) > 1:
                pred_onehot = torch.argmax(pred_tensor, dim=-1)
            else:
                pred_onehot = pred_tensor
            
            # Decode using available representations
            if task in available_representations:
                try:
                    decoded = available_representations[task].decode(
                        pred_onehot.reshape(-1, 1)
                    )
                    # Convert to numpy array if it's a list
                    if isinstance(decoded, list):
                        decoded_predictions[task] = np.array(decoded).flatten()
                    else:
                        decoded_predictions[task] = decoded.flatten()
                except (IndexError, ValueError) as e:
                    print(f"Warning: Error decoding {task} predictions: {e}")
                    # Fallback to raw indices
                    decoded_predictions[task] = pred_onehot.cpu().numpy()
            else:
                decoded_predictions[task] = pred_onehot.cpu().numpy()
    
    # Add timing information if available
    if "onset" in predictions:
        decoded_predictions["onset"] = predictions["onset"].cpu().numpy()
    else:
        decoded_predictions["onset"] = score.note_array()["onset_beat"]
    if "s_measure" in predictions:
        decoded_predictions["s_measure"] = predictions["s_measure"].cpu().numpy()
    else:
        decoded_predictions["s_measure"] = score[0].measure_number_map(score.note_array()["onset_div"])

    return decoded_predictions


def export_to_csv(
    predictions: Dict[str, np.ndarray], 
    output_path: str
) -> None:
    """Export predictions to CSV file."""
    df = pd.DataFrame(predictions)
    df.to_csv(output_path, index=False)
    print(f"Analysis results saved to: {output_path}")


def export_roman_numerals_to_musicxml(
    predictions: Dict[str, np.ndarray],
    score: pt.score.Score,
    output_path: str
) -> None:
    """Export Roman numeral analysis to musicXML."""
    
    if "romanNumeral" not in predictions or "localkey" not in predictions:
        print("Warning: Roman numeral export requires 'romanNumeral' and 'localkey' predictions")
        return
    
    # Create a copy of the score
    score_copy = copy.deepcopy(score)
    
    # Get the bass part for timing reference
    bass_part = score_copy.parts[-1]
    
    # Create a new part for Roman numerals
    rn_part = pt.score.Part(
        id="RNA", 
        part_name="Roman Numerals", 
        quarter_duration=bass_part._quarter_durations[0]
    )
    rn_part.add(pt.score.Clef(staff=1, sign="percussion", line=2, octave_change=0), 0)
    rn_part.add(pt.score.Staff(number=1, lines=1), 0)
    
    # Add roman numeral annotations
    onsets = predictions.get("onset", np.arange(len(predictions["romanNumeral"])))
    
    annotations = []
    prev_key = ""
    
    for i, (rn, key, onset) in enumerate(zip(
        predictions["romanNumeral"], 
        predictions["localkey"], 
        onsets
    )):
        if key != prev_key:
            rn_formatted = f"{key}:{rn}"
            prev_key = key
        else:
            rn_formatted = rn
        
        formatted_rn = formatRomanNumeral(rn_formatted, key)
        onset_div = int(bass_part.inv_beat_map(onset).item()) if hasattr(bass_part, 'inv_beat_map') else int(onset * 4)
        annotations.append((formatted_rn, onset_div))
    
    # Remove consecutive duplicates
    unique_annotations = []
    for i, (rn, onset) in enumerate(annotations):
        if i == 0 or rn != annotations[i-1][0]:
            unique_annotations.append((rn, onset))
    
    # Add annotations to the score
    for i, (rn, onset) in enumerate(unique_annotations):
        duration = (unique_annotations[i+1][1] - onset) if i+1 < len(unique_annotations) else 4
        
        note = pt.score.UnpitchedNote(step="F", octave=5, staff=1)
        word = pt.score.RomanNumeral(rn)
        rn_part.add(note, onset, onset + duration)
        rn_part.add(word, onset)
    
    # Copy time signatures and measures
    for item in bass_part.iter_all(pt.score.TimeSignature):
        rn_part.add(item, item.start.t)
    for item in bass_part.measures:
        rn_part.add(item, item.start.t, item.end.t)
    
    # Add the Roman numeral part to the score
    score_copy.parts.append(rn_part)
    
    # Save the annotated score
    pt.save_musicxml(score_copy, output_path)
    print(f"Roman numeral analysis saved to: {output_path}")


def main():
    """Main prediction function."""
    parser = get_parser()
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Handle checkpoint loading
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    else:
        # Use default wandb artifact
        print("No checkpoint path provided, using default W&B artifact...")
        checkpoint_path = download_wandb_checkpoint(args.wandb_artifact)
    
    # Handle input score
    if args.input_score:
        input_score_path = args.input_score
        if not os.path.exists(input_score_path):
            raise FileNotFoundError(f"Input score not found: {input_score_path}")
    else:
        # Use default score
        print("No input score provided, using default Mozart quartet...")
        input_score_path = download_default_score()
    
    # Load model
    print(f"Loading model from: {checkpoint_path}")
    model = load_model(checkpoint_path, args.device)
    
    # Load score
    print(f"Loading score: {input_score_path}")
    score = pt.load_score(input_score_path)
    
    # Parse tasks
    tasks = [task.strip() for task in args.tasks.split(",")]
    print(f"Performing analysis tasks: {tasks}")
    
    # Perform analysis
    predictions = predict_analysis(model, score, tasks, args.device)
    
    # Generate output filename base
    input_basename = os.path.splitext(os.path.basename(input_score_path))[0]
    
    # Export results
    if args.export_csv:
        csv_path = os.path.join(args.output_dir, f"{input_basename}_analysis.csv")
        export_to_csv(predictions, csv_path)
    
    if args.export_roman_numerals and "romanNumeral" in predictions:
        rn_path = os.path.join(args.output_dir, f"{input_basename}_roman_numerals.musicxml")
        export_roman_numerals_to_musicxml(predictions, score, rn_path)
    
    # Always export a basic CSV with all predictions
    if not args.export_csv and not args.export_roman_numerals:
        csv_path = os.path.join(args.output_dir, f"{input_basename}_analysis.csv")
        export_to_csv(predictions, csv_path)
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()
