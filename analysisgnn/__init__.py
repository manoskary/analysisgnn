"""
AnalysisGNN: A Unified Music Analysis Model with Graph Neural Networks

This package provides a comprehensive framework for multi-task music analysis 
using Graph Neural Networks (GNNs).
"""

__version__ = "1.0.0"
__author__ = "Emmanouil Karystinaios"
__email__ = "manos.karyss@gmail.com"

from . import data
from . import train
from . import models
from . import descriptors
from . import utils
from . import metrics
from . import inference

# Key imports for easy access
from .models.analysis import ContinualAnalysisGNN
from .data.datamodules.analysis import AnalysisDataModule

__all__ = [
    "ContinualAnalysisGNN",
    "AnalysisDataModule", 
    "data",
    "train", 
    "models",
    "descriptors",
    "utils",
    "metrics",
    "inference"
]