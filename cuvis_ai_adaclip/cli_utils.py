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
            click.option("--pretrained-adaclip", type=click.Choice(available_weights), default="pretrained_all",
                        help="Weight name for AdaCLIP"),
            click.option("--prompt-text", type=str, default="normal: lentils, anomaly: stones",
                        help="Prompt text for AdaCLIP"),
            click.option("--target-class-id", type=int, default=2,
                        help="Target anomaly class ID"),
            click.option("--quantile", type=float, default=0.995,
                        help="Quantile for binary decider"),
            click.option("--gaussian-sigma", type=float, default=4.0,
                        help="Gaussian sigma for AdaCLIP"),
            click.option("--use-half-precision", default=True,
                        help="Use half precision for optimization"),
            click.option("--enable-warmup", default=True,
                        help="Enable warmup for optimization"),
            click.option("--use-torch-preprocess/--no-use-torch-preprocess", default=True,
                        help="Use fast tensor preprocessing (default: True). Use --no-use-torch-preprocess for PIL preprocessing (exact match)"),
            click.option("--batch-size", type=int, default=1,
                        help="Batch size for data loading"),
        ]

        # Apply options in reverse order (last to first)
        for option in reversed(options):
            command = option(command)
        return command

    def add_data_options(self, command):
        """Add common data configuration options to a Click command."""
        options = [
            click.option("--cu3s-file-path", type=str, default="C:/Users/anish.raj/projects/gitlab_cuvis_ai_3/cuvis.ai/data/Lentils/Lentils_000.cu3s",
                        help="Path to CU3S file"),
            click.option("--annotation-json-path", type=str, default="C:/Users/anish.raj/projects/gitlab_cuvis_ai_3/cuvis.ai/data/Lentils/Lentils_000.json",
                        help="Path to annotation JSON file"),
            click.option("--train-ids", type=str, default="0,2",
                        help="Comma-separated train IDs"),
            click.option("--val-ids", type=str, default="2,4",
                        help="Comma-separated validation IDs"),
            click.option("--test-ids", type=str, default="1,3,5",
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
            click.option("--visualize-upto", type=int, default=3,
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

    def add_cir_options(self, command):
        """Add CIR false-color specific options to a Click command."""
        options = [
            click.option("--nir-nm", type=float, default=860.0,
                        help="Near-infrared wavelength in nm"),
            click.option("--red-nm", type=float, default=670.0,
                        help="Red wavelength in nm"),
            click.option("--green-nm", type=float, default=560.0,
                        help="Green wavelength in nm"),
        ]

        # Apply options in reverse order (last to first)
        for option in reversed(options):
            command = option(command)
        return command

    def add_cir_false_rg_options(self, command):
        """Add CIR false-RG specific options (NIR->R, Red->G, Green(visible)->B)."""
        options = [
            click.option("--nir-nm", type=float, default=860.0,
                        help="Near-infrared wavelength in nm"),
            click.option("--red-nm", type=float, default=670.0,
                        help="Red wavelength in nm"),
            click.option("--green-nm", type=float, default=450.0,
                        help="Green (visible) wavelength in nm"),
        ]

        # Apply options in reverse order (last to first)
        for option in reversed(options):
            command = option(command)
        return command

    def add_high_contrast_options(self, command):
        """Add high-contrast band selection options."""
        options = [
            click.option("--hc-windows", type=str, default="440-500,500-580,610-700",
                        help="Comma-separated wavelength windows as start-end pairs (e.g., 440-500,500-580,610-700)"),
            click.option("--hc-alpha", type=float, default=0.1,
                        help="Weight for Laplacian energy term"),
        ]

        # Apply options in reverse order (last to first)
        for option in reversed(options):
            command = option(command)
        return command

    def parse_hc_windows(self, windows_str):
        """Parse high-contrast windows from comma-separated string (e.g., '440-500,500-580,610-700')."""
        windows = []
        for window in windows_str.split(","):
            start, end = window.strip().split("-")
            windows.append((float(start), float(end)))
        return tuple(windows)

    def add_supervised_cir_options(self, command):
        """Add supervised CIR band selection options."""
        options = [
            click.option("--sup-windows", type=str, default="840-910,650-720,500-570",
                        help="Comma-separated wavelength windows as start-end pairs (NIR,Red,Green)"),
            click.option("--sup-score-weights", type=str, default="1.0,1.0,1.0",
                        help="Comma-separated score weights (Fisher, AUC, MI)"),
            click.option("--sup-lambda-penalty", type=float, default=0.5,
                        help="Lambda penalty for supervised band selection"),
        ]

        # Apply options in reverse order (last to first)
        for option in reversed(options):
            command = option(command)
        return command

    def parse_sup_windows(self, windows_str):
        """Parse supervised CIR windows from comma-separated string."""
        windows = []
        for window in windows_str.split(","):
            start, end = window.strip().split("-")
            windows.append((float(start), float(end)))
        return tuple(windows)

    def parse_sup_score_weights(self, weights_str):
        """Parse score weights from comma-separated string."""
        return tuple(float(w.strip()) for w in weights_str.split(","))

    def add_supervised_full_spectrum_options(self, command):
        """Add supervised full-spectrum band selection options (no windows, global selection)."""
        options = [
            click.option("--sup-fs-score-weights", type=str, default="1.0,1.0,1.0",
                        help="Comma-separated score weights (Fisher, AUC, MI)"),
            click.option("--sup-fs-lambda-penalty", type=float, default=0.5,
                        help="Lambda penalty for supervised band selection"),
        ]

        # Apply options in reverse order (last to first)
        for option in reversed(options):
            command = option(command)
        return command

    def add_supervised_windowed_false_rgb_options(self, command):
        """Add supervised windowed false-RGB band selection options (visible RGB windows)."""
        options = [
            click.option("--sup-wf-windows", type=str, default="440-500,500-580,610-700",
                        help="Comma-separated wavelength windows as start-end pairs (Blue,Green,Red)"),
            click.option("--sup-wf-score-weights", type=str, default="1.0,1.0,1.0",
                        help="Comma-separated score weights (Fisher, AUC, MI)"),
            click.option("--sup-wf-lambda-penalty", type=float, default=0.5,
                        help="Lambda penalty for supervised band selection"),
        ]

        # Apply options in reverse order (last to first)
        for option in reversed(options):
            command = option(command)
        return command

    def parse_data_config(self, **kwargs):
        """Parse data configuration from CLI arguments."""
        return {
            "cu3s_file_path": kwargs.get("cu3s_file_path", "C:/Users/anish.raj/projects/gitlab_cuvis_ai_3/cuvis.ai/data/Lentils/Lentils_000.cu3s"),
            "annotation_json_path": kwargs.get("annotation_json_path", "C:/Users/anish.raj/projects/gitlab_cuvis_ai_3/cuvis.ai/data/Lentils/Lentils_000.json"),
            "train_ids": [int(x.strip()) for x in kwargs.get("train_ids", "0,2").split(",")],
            "val_ids": [int(x.strip()) for x in kwargs.get("val_ids", "2,4").split(",")],
            "test_ids": [int(x.strip()) for x in kwargs.get("test_ids", "1,3,5").split(",")],
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
