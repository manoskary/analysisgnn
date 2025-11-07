"""
AnalysisGNN: A Unified Music Analysis Model with Graph Neural Networks

This package provides a comprehensive framework for multi-task music analysis 
using Graph Neural Networks (GNNs).
"""

__version__ = "1.0.0"
__author__ = "Emmanouil Karystinaios"

from . import data
from . import train
from . import models
from . import descriptors
from . import utils
from . import metrics

# analysisgnn/__init__.py
import importlib

__all__ = ["inference"]  # optional

def __getattr__(name):
    if name == "inference":
        mod = importlib.import_module(".inference", __name__)
        globals()[name] = mod  # cache
        return mod
    raise AttributeError(f"module {__name__} has no attribute {name!r}")


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
