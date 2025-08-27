# AnalysisGNN: A Unified Music Analysis Model with Graph Neural Networks

AnalysisGNN is a comprehensive framework for multi-task music analysis using Graph Neural Networks (GNNs). It can perform various music analysis tasks including cadence detection, Phrase Detection, key analysis, harmonic analysis, and Roman numeral analysis on musical scores and detection of Non-Chord Notes.

This work was published on the 17th International Symposium on
Computer Music Multidisciplinary Research 2025 (CMMR).

## Features

- **Multi-task Learning**: Simultaneously predict multiple music analysis tasks
- **Graph-based Representation**: Uses heterogeneous graphs to represent musical scores
- **State-of-the-art Models**: Implements HybridGNN, HGT, and MetricalGNN architectures
- **Continual Learning**: Support for sequential learning of different analysis tasks
- **Flexible Training**: Supports both single-task and multi-task training strategies
- **Easy Inference**: Simple prediction interface for new musical scores

## Supported Analysis Tasks

- **Cadence Detection**: Identify cadence types in musical passages
- **Key Analysis**: Local and global key detection
- **Harmonic Analysis**: Chord quality, inversion, root, and bass note prediction
- **Roman Numeral Analysis**: Functional harmonic analysis
- **Rhythmic Analysis**: Downbeat and metrical analysis
- **Voice Leading**: Analysis of voice leading patterns

### Disclaimer

This repository is under construction, despite the many features some things might not yet work properly.

#### TODOs
- Add resolve function for Roman Numerals
- Add onset and beat prediction aggregation in predict
- Support all training modes
- Simplify requirements


## Installation

### Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/manoskary/analysisgnn.git
cd analysisgnn

# Create conda environment
conda env create -f environment.yml
conda activate analysisgnn

# Install the package
pip install -e .
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/manoskary/analysisgnn.git
cd analysisgnn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install pyg-lib first (required for graphmuse)
TORCH=2.5.0
CUDA=cu124  # or cu118, cu121 depending on your CUDA version
pip install pyg-lib -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

# Install dependencies
pip install -r requirements.txt

# Install graphmuse (with specific pyg-lib dependency)
pip install graphmuse

# Install the package
pip install -e .
```

**Notes:**
- Downloaded models and scores are cached in the `./artifacts/` folder to avoid re-downloading. This folder is excluded from git tracking.
- If you encounter issues with `graphmuse` installation, make sure you have installed `pyg-lib` for your specific PyTorch and CUDA versions first (see the pip installation section above).

## Quick Start

### Easy Prediction (No Setup Required)

The fastest way to try AnalysisGNN is to use the default pre-trained model and example score:

```bash
# Install the package and run prediction immediately
analysisgnn-predict
```

This will:
1. Download a pre-trained AnalysisGNN model from Weights & Biases (cached in `./artifacts/models/`)
2. Download a Mozart string quartet example (K. 158, mvt. 1) (cached in `./artifacts/`)
3. Perform music analysis (cadence, key, harmony, Roman numerals)
4. Save results to `./outputs/k158-01_analysis.csv`

### Training a Model

```bash
# Basic training with default parameters
analysisgnn-train --do_train --main_tasks cadence,rna --num_epochs 50

# Training with GPU support
analysisgnn-train --do_train --gpus 0 --main_tasks cadence,localkey,romanNumeral --batch_size 32

# Training with W&B logging
analysisgnn-train --do_train --use_wandb --main_tasks all --num_epochs 100
```

### Using the Prediction Script

```bash
# Quick start - uses default pre-trained model and example score
analysisgnn-predict

# Predict analysis from a specific trained model
analysisgnn-predict --checkpoint_path model.ckpt --input_score score.musicxml

# Use your own score with the default pre-trained model
analysisgnn-predict --input_score my_score.musicxml --export_roman_numerals

# Download a specific model from W&B and predict
analysisgnn-predict --wandb_artifact user/project/model-id:v0 --input_score score.musicxml --export_roman_numerals

# Specify analysis tasks
analysisgnn-predict --input_score score.musicxml --tasks cadence,localkey,romanNumeral --export_csv
```

## Training Parameters

### Model Architecture

- `--model`: Choose architecture (`HybridGNN`, `HGT`, `MetricalGNN`)
- `--num_layers`: Number of GNN layers (default: 3)
- `--hidden_channels`: Hidden dimension size (default: 256)
- `--dropout`: Dropout rate (default: 0.3)

### Training Configuration

- `--num_epochs`: Number of training epochs
- `--batch_size`: Batch size for training (default: 100)
- `--lr`: Learning rate (default: 0.005)
- `--weight_decay`: Weight decay for regularization (default: 5e-3)

### Multi-task Learning

- `--main_tasks`: Comma-separated list of tasks to train on
- `--cl_training`: Enable continual learning mode
- `--mt_strategy`: Multi-task learning strategy (`wloss`, `gradnorm`)

### Graph Configuration

- `--add_beats`: Include beat nodes in the graph
- `--add_measures`: Include measure nodes in the graph
- `--subgraph_size`: Maximum subgraph size for sampling (default: 500)

## Advanced Usage

### Custom Training Configuration

Create a JSON configuration file:

```json
{
    "model": "HybridGNN",
    "num_layers": 4,
    "hidden_channels": 512,
    "dropout": 0.2,
    "lr": 0.001,
    "batch_size": 64,
    "main_tasks": ["cadence", "localkey", "romanNumeral"],
    "add_beats": true,
    "add_measures": true
}
```

Then run:
```bash
analysisgnn-train --config_path config.json --do_train
```

### Evaluation

```bash
# Evaluate a trained model
analysisgnn-train --do_eval --checkpoint_path best_model.ckpt

# Load from W&B and evaluate
analysisgnn-train --do_eval --load_from_checkpoint --checkpoint_path user/project/model-id:v0
```

## Model Architecture

AnalysisGNN uses heterogeneous graph neural networks to represent musical scores. The graph includes:

- **Note nodes**: Individual notes with features like pitch, duration, onset
- **Beat nodes**: Metrical positions (optional)
- **Measure nodes**: Bar-level structure (optional)
- **Edges**: Various relationships between musical elements

### Supported GNN Architectures

1. **HybridGNN**: Combines multiple GNN layers with attention mechanisms
2. **HGT (Heterogeneous Graph Transformer)**: Uses transformer-style attention for heterogeneous graphs
3. **MetricalGNN**: Specialized for metrical and rhythmic analysis

## Data Format

The model expects musical scores in any quantized and spelled symbolic format that partitura can read. The graph construction automatically extracts:

- Note-level features (pitch, duration, voice, etc.)

And the model predicts on the note level:

- Phrases
- Cadences
- Sections
- Pedal Notes
- Local Key
- Tonalized Key
- Chord Degree
- Inversion
- Quality
- The bass note
- The root note
- Non-Chord Tote (NCT) identification

## Citation

If you use AnalysisGNN in your research, please cite:

```bibtex
@inproceedings{karystinaios2024analysisgnn,
    title={AnalysisGNN: A Unified Music Analysis Model with Graph Neural Networks},
    author={Karystinaios, Emmanouil and Hentschel, Johannes and Neuwirth, Markus and Widmer, Gerhard},
    booktitle={International Symposium on Computer Music Multidisciplinary Research (CMMR)},
    year={2025}
}
```


## Acknowledgments

- Built with [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/)
- Heavily Inspired and used data by [AugmentedNet](https://github.com/napulen/AugmentedNet)
- Uses datasets for training from the [Distant Listening Corpus](https://github.com/DCMLab/distant_listening_corpus)
- Graph operations powered by [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- Music processing with [Partitura](https://github.com/CPJKU/partitura)
- Experiment tracking with [Weights & Biases](https://wandb.ai/)
