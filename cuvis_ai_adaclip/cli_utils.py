"""Reusable CLI utilities for AdaCLIP examples.

This module provides reusable Click CLI components that can be used across
different AdaCLIP examples to maintain consistency and reduce code duplication.
"""

from pathlib import Path
import click
import torch
from cuvis_ai_adaclip import (
    AdaCLIPDetector,
    download_weights,
    list_available_weights,
)
from loguru import logger

# Available model backbones - can be imported by other examples
AVAILABLE_BACKBONES = [
    "ViT-L-14-336",
    "ViT-L-14",
    "ViT-B-16",
    "ViT-B-32",
    "ViT-H-14",
]

class AdaCLIPCLI:
    """Base CLI class for AdaCLIP examples with common options."""

    def __init__(self, name="AdaCLIP Example"):
        self.name = name
        self.cli = click.Group(name=self.name)

    def add_common_options(self, command):
        """Add common AdaCLIP options to a Click command."""
        available_weights = list_available_weights()
        options = [
            click.option("--output-dir", type=str, default="outputs/example",
                        help="Output directory for results"),
            click.option("--backbone-name", type=click.Choice(AVAILABLE_BACKBONES), default="ViT-L-14-336",
                        help="Backbone name for AdaCLIP"),
            click.option("--weight-name", type=click.Choice(available_weights), default="pretrained_all",
                        help="Weight name for AdaCLIP"),
            click.option("--prompt-text", type=str, default="anomaly",
                        help="Prompt text for AdaCLIP"),
            click.option("--target-class-id", type=int, default=2,
                        help="Target anomaly class ID"),
            click.option("--quantile", type=float, default=0.95,
                        help="Quantile for binary decider"),
            click.option("--gaussian-sigma", type=float, default=4.0,
                        help="Gaussian sigma for AdaCLIP"),
            click.option("--use-half-precision", is_flag=True,
                        help="Use half precision for optimization"),
            click.option("--enable-warmup", is_flag=True,
                        help="Enable warmup for optimization"),
            click.option("--batch-size", type=int, default=4,
                        help="Batch size for data loading"),
        ]

        # Apply options in reverse order (last to first)
        for option in reversed(options):
            command = option(command)
        return command

    def add_data_options(self, command):
        """Add common data configuration options to a Click command."""
        options = [
            click.option("--cu3s-file-path", type=str, default="data/Lentils/Lentils_000.cu3s",
                        help="Path to CU3S file"),
            click.option("--annotation-json-path", type=str, default="data/Lentils/Lentils_000.json",
                        help="Path to annotation JSON file"),
            click.option("--train-ids", type=str, default="0,2",
                        help="Comma-separated train IDs"),
            click.option("--val-ids", type=str, default="1",
                        help="Comma-separated validation IDs"),
            click.option("--test-ids", type=str, default="3,5",
                        help="Comma-separated test IDs"),
            click.option("--processing-mode", type=str, default="Reflectance",
                        help="Processing mode for data"),
            click.option("--normal-class-ids", type=str, default="0,1",
                        help="Comma-separated normal class IDs"),
        ]

        # Apply options in reverse order (last to first)
        for option in reversed(options):
            command = option(command)
        return command

    def add_visualization_options(self, command):
        """Add common visualization options to a Click command."""
        options = [
            click.option("--visualize-upto", type=int, default=10,
                        help="Maximum number of visualizations to generate"),
        ]

        # Apply options in reverse order (last to first)
        for option in reversed(options):
            command = option(command)
        return command

    def add_wavelength_options(self, command):
        """Add wavelength-specific options to a Click command."""
        options = [
            click.option("--target-wavelengths", type=str, default="650,550,450",
                        help="Comma-separated target wavelengths for R,G,B channels"),
        ]

        # Apply options in reverse order (last to first)
        for option in reversed(options):
            command = option(command)
        return command

    def parse_data_config(self, **kwargs):
        """Parse data configuration from CLI arguments."""
        return {
            "cu3s_file_path": kwargs.get("cu3s_file_path", "data/Lentils/Lentils_000.cu3s"),
            "annotation_json_path": kwargs.get("annotation_json_path", "data/Lentils/Lentils_000.json"),
            "train_ids": [int(x.strip()) for x in kwargs.get("train_ids", "0,2").split(",")],
            "val_ids": [int(x.strip()) for x in kwargs.get("val_ids", "1").split(",")],
            "test_ids": [int(x.strip()) for x in kwargs.get("test_ids", "3,5").split(",")],
            "batch_size": kwargs.get("batch_size", 4),
            "processing_mode": kwargs.get("processing_mode", "Reflectance")
        }

    def parse_normal_class_ids(self, class_ids_str):
        """Parse normal class IDs from comma-separated string."""
        return [int(x.strip()) for x in class_ids_str.split(",")]

    def parse_target_wavelengths(self, wavelengths_str):
        """Parse target wavelengths from comma-separated string."""
        return tuple(float(w.strip()) for w in wavelengths_str.split(","))

    def setup_logging(self):
        """Set up consistent logging for AdaCLIP examples."""
        logger.remove()  # Remove default logger
        logger.add(lambda msg: print(msg), level="INFO")

    def get_device(self):
        """Get the appropriate device (CUDA if available, else CPU)."""
        return "cuda" if torch.cuda.is_available() else "cpu"
