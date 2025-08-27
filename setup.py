#!/usr/bin/env python3
"""
Setup script for AnalysisGNN: A Unified Music Analysis Model with Graph Neural Networks
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="analysisgnn",
    version="1.0.0",
    author="Emmanouil Karystinaios",
    author_email="manos.karyss@gmail.com",
    description="A Unified Music Analysis Model with Graph Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/manoskary/analysisgnn",
    project_urls={
        "Bug Tracker": "https://github.com/manoskary/analysisgnn/issues",
        "Documentation": "https://github.com/manoskary/analysisgnn",
        "Source Code": "https://github.com/manoskary/analysisgnn",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8,<3.12",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0", 
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "analysisgnn-train=analysisgnn.train.train_analysisgnn:main",
            "analysisgnn-predict=analysisgnn.inference.predict_analysis:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="music analysis, graph neural networks, deep learning, pytorch, musicology",
)
