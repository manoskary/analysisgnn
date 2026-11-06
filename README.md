# AnalysisGNN: A Unified Music Analysis Model with Graph Neural Networks

AnalysisGNN is a comprehensive framework for multi-task music analysis using Graph Neural Networks (GNNs). It can perform various music analysis tasks including cadence detection, Phrase Detection, key analysis, harmonic analysis, and Roman numeral analysis on musical scores and detection of Non-Chord Notes.

This repository includes the implementation of **[From Prediction to Collaboration:
Interactive Symbolic Music Analysis](https://arxiv.org/abs/2607.13587)**,
in the Proceedings of the 27th International Society for Music Information
Retrieval Conference (**ISMIR 2026**), Abu Dhabi, UAE. It combines MusicBERT
representations with AnalysisGNN for full-score Roman numeral analysis,
masked completion, and interactive revision.

The original AnalysisGNN work was published at the 17th International Symposium
on Computer Music Multidisciplinary Research (**CMMR 2025**). Citations for both
papers are provided below.

## ISMIR 2026 Paper and Checkpoints

The paper implementation was developed on the
[`gradio` branch](https://github.com/manoskary/analysisgnn/tree/gradio).
The [Hugging Face demo](https://huggingface.co/spaces/manoskary/analysisgnn)
also installs this branch and uses the released
[`manoskary/analysisgnn-hybrid` model bundle](https://huggingface.co/manoskary/analysisgnn-hybrid).
The interactive interface is available at [analysisgnn.com](https://analysisgnn.com).

The bundle's `bundle.json` manifest identifies these artifacts:

| Artifact | Path in the Hugging Face repository | Purpose |
| --- | --- | --- |
| Full checkpoint | `checkpoints/full.ckpt` | Blind full-score inference |
| Masked checkpoint | `checkpoints/masked.ckpt` | Completion and revision conditioned on known labels |
| Note voter | `voters/note_voter.pt` | Post-hoc aggregation |
| Beat voter | `voters/beat_voter.pt` | Beat-level aggregation |

After installation, load the released checkpoints through the bundle resolver:

```python
import partitura as pt
from analysisgnn.inference.hf_bundle import resolve_hybrid_bundle
from analysisgnn.inference.hybrid_predictor import HybridAnalysisPredictor

bundle = resolve_hybrid_bundle(
    "manoskary/analysisgnn-hybrid",
    revision="aa2b81c4453a586985ef6746076f8111af2653fe",
)
predictor = HybridAnalysisPredictor(
    full_checkpoint_path=bundle.full_ckpt,
    masked_checkpoint_path=bundle.masked_ckpt,
    device="auto",
)
score = pt.load_score("my_score.musicxml")
predictions = predictor.predict(score, force_route="full")
```

The revision pins the released bundle; omit it to use the latest version.
The bundle excludes the MusicBERT backbone and tokenizer, which are downloaded
separately from [`manoskary/musicbert-large`](https://huggingface.co/manoskary/musicbert-large)
and [`manoskary/miditok-REMI`](https://huggingface.co/manoskary/miditok-REMI)
when needed. The masked checkpoint is intended for partial-label completion;
use the full checkpoint for blind inference.

## Features

- **Multi-task Learning**: Simultaneously predict multiple music analysis tasks
- **Graph-based Representation**: Uses heterogeneous graphs to represent musical scores
- **State-of-the-art Models**: Implements HybridGNN, HGT, and MetricalGNN architectures
- **Continual Learning**: Support for sequential learning of different analysis tasks
- **Flexible Training**: Supports both single-task and multi-task training strategies
- **Semi-Supervised Node Masking**: Selective prediction with context labels for improved efficiency
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

For the ISMIR 2026 hybrid model, use the Hugging Face checkpoint example above.
The CLI examples below use the original AnalysisGNN prediction workflow and its
Weights & Biases checkpoint defaults.

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

### Semi-Supervised Node Masking

AnalysisGNN supports selective prediction where only a subset of nodes need predictions while others provide context:

```python
from analysisgnn.utils.node_masking import create_node_mask

# Create mask: T (target), C (context), U (unlabeled)
mask = create_node_mask(
    num_nodes=100,
    target_indices=torch.arange(0, 40),    # Predict these
    context_indices=torch.arange(40, 70),  # Use as context
    context_weight=0.1                      # Context contributes 10% loss
)

# Add to your graph data
graph["note"].node_mask = mask

# Training automatically uses the mask
# - Context nodes: labels fixed during training, guide predictions
# - Target nodes: full loss, primary prediction focus
# - Unlabeled nodes: ignored, no loss contribution
```

**Use cases:**
- Phrase boundary analysis with harmonic context
- Transfer learning with pseudo-labels as context
- Interactive annotation assistance
- Semi-supervised learning with limited annotations

See `examples/semi_supervised_masking_example.py` for detailed usage.

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
- Tonicized Key
- Chord Degree
- Inversion
- Quality
- The bass note
- The root note
- Non-Chord Tote (NCT) identification

## Citation

For the hybrid model, masked completion, or interactive analysis framework,
please cite the ISMIR 2026 paper:

```bibtex
@inproceedings{karystinaios2026prediction,
    title={From Prediction to Collaboration: Interactive Symbolic Music Analysis},
    author={Karystinaios, Emmanouil and Hentschel, Johannes and Neuwirth, Markus and Widmer, Gerhard},
    booktitle={Proceedings of the 27th International Society for Music Information Retrieval Conference (ISMIR)},
    year={2026},
    address={Abu Dhabi, UAE},
    url={https://arxiv.org/abs/2607.13587}
}
```

For the original AnalysisGNN model, please also cite:

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
