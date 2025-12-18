"""High-contrast AdaCLIP example (variance + Laplacian windowed RGB).

This script mirrors the updated AdaCLIP examples and uses the shared experiment
config in `configs/experiment/adaclip_high_contrast.yaml`.

It:
  * Builds a CuvisPipeline explicitly.
  * Uses LentilsAnomalyDataNode → HighContrastBandSelector → AdaCLIPDetector.
  * HighContrastBandSelector selects bands with highest variance + Laplacian energy per window.
  * Adds a quantile-based decider, generic anomaly metrics, and visualizations.
  * Logs everything via TensorBoardMonitorNode and saves the pipeline + experiment config.
"""
from __future__ import annotations

from pathlib import Path

import torch
import click
from cuvis_ai_adaclip import (
    AdaCLIPDetector,
    download_weights,
    list_available_weights,
)
from loguru import logger

from cuvis_ai.data.lentils_anomaly import SingleCu3sDataModule
from cuvis_ai.deciders.binary_decider import QuantileBinaryDecider
from cuvis_ai.node.band_selection import HighContrastBandSelector
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.metrics import AnomalyDetectionMetrics
from cuvis_ai.node.monitor import TensorBoardMonitorNode
from cuvis_ai.node.visualizations import RGBAnomalyMask, ScoreHeatmapVisualizer
from cuvis_ai.pipeline.pipeline import CuvisPipeline
from cuvis_ai.training import StatisticalTrainer
from cuvis_ai.training.config import (
    PipelineMetadata,
    TrainingConfig,
    TrainRunConfig,
)

from cuvis_ai_adaclip.cli_utils import AdaCLIPCLI

# Create reusable CLI instance
cli = AdaCLIPCLI("AdaCLIP High Contrast")

@cli.add_common_options
@cli.add_data_options
@cli.add_visualization_options
@click.command()
def main(**kwargs):
    """Run AdaCLIP high-contrast (statistical) with Click CLI."""
    logger.info("Run: AdaCLIP high-contrast (statistical)")

    # Parse configuration using CLI utilities
    output_dir = Path(kwargs["output_dir"])
    data_config = cli.parse_data_config(**kwargs)
    logger.info("Output: {}", output_dir)

    # ----------------------------
    # Data & weights
    # ----------------------------
    datamodule = SingleCu3sDataModule(**data_config)
    datamodule.setup(stage=None)

    wavelengths = datamodule.train_ds.wavelengths
    logger.info("Wavelengths: {:.1f}-{:.1f} nm", wavelengths.min(), wavelengths.max())

    model_name = kwargs["backbone_name"]
    weight_name = kwargs["pretrained_adaclip"]
    prompt_text = kwargs["prompt_text"]

    logger.info("Available weights: {}", list_available_weights())
    download_weights(weight_name)

    # ----------------------------
    # Resolve example-specific config
    # ----------------------------
    quantile = kwargs["quantile"]
    visualize_upto = kwargs["visualize_upto"]
    gaussian_sigma = kwargs["gaussian_sigma"]

    # High-contrast band selection parameters (default values)
    windows = ((400.0, 500.0), (500.0, 600.0), (600.0, 700.0), (700.0, 800.0), (800.0, 900.0), (900.0, 1000.0))
    alpha = 0.5

    # Read optimization flags from config (default to False for non-optimized comparison)
    use_half_precision = kwargs.get("use_half_precision", False)
    enable_warmup = kwargs.get("enable_warmup", False)

    logger.info(
        "Splits: train={}, val={}, test={}", data_config["train_ids"], data_config["val_ids"], data_config["test_ids"]
    )
    logger.info("Model: {} | Weights: {}", model_name, weight_name)
    logger.info("Prompt: {}", prompt_text)
    logger.info("Selector windows: {}", windows)
    logger.info("Selector alpha: {}", alpha)
    logger.info("AdaCLIP opts: fp16={}, warmup={}", use_half_precision, enable_warmup)

    # ----------------------------
    # Build pipeline
    # ----------------------------
    pipeline = CuvisPipeline("AdaCLIP_HighContrast")

    data_node = LentilsAnomalyDataNode(
        normal_class_ids=[0, 1],
    )
    band_selector = HighContrastBandSelector(windows=windows, alpha=alpha)

    adaclip = AdaCLIPDetector(
        weight_name=weight_name,
        backbone=model_name,
        prompt_text=prompt_text,
        gaussian_sigma=gaussian_sigma,
        use_half_precision=use_half_precision,
        enable_warmup=enable_warmup,
    )

    decider = QuantileBinaryDecider(quantile=quantile)
    standard_metrics = AnomalyDetectionMetrics(name="detection_metrics")
    score_viz = ScoreHeatmapVisualizer(normalize_scores=True, up_to=visualize_upto)
    mask_viz = RGBAnomalyMask(up_to=visualize_upto)
    monitor = TensorBoardMonitorNode(
        run_name=pipeline.name,
        output_dir=str(Path(kwargs["output_dir"])/ "tensorboard"),
    )

    # Wiring: cube → band selector → AdaCLIP → decider → metrics + viz + TB
    pipeline.connect(
        # hyperspectral → high-contrast RGB
        (data_node.outputs.cube, band_selector.inputs.cube),
        (data_node.outputs.wavelengths, band_selector.inputs.wavelengths),
        # RGB → AdaCLIP
        (band_selector.outputs.rgb_image, adaclip.inputs.rgb_image),
        # AdaCLIP scores → decider + visualizations
        (adaclip.outputs.scores, decider.inputs.logits),
        (adaclip.outputs.scores, score_viz.inputs.scores),
        (adaclip.outputs.scores, mask_viz.inputs.scores),
        # decisions + GT for metrics + overlay
        (decider.outputs.decisions, standard_metrics.inputs.decisions),
        (data_node.outputs.mask, standard_metrics.inputs.targets),
        (decider.outputs.decisions, mask_viz.inputs.decisions),
        (data_node.outputs.mask, mask_viz.inputs.mask),
        (band_selector.outputs.rgb_image, mask_viz.inputs.rgb_image),
        # send metrics + artifacts to TensorBoard
        (standard_metrics.outputs.metrics, monitor.inputs.metrics),
        (score_viz.outputs.artifacts, monitor.inputs.artifacts),
        (mask_viz.outputs.artifacts, monitor.inputs.artifacts),
    )

    # ----------------------------
    # Move pipeline to GPU if available
    # ----------------------------
    device = cli.get_device()
    logger.info("Device: {}", device)
    pipeline.to(device)

    # ----------------------------
    # Visualize and run
    # ----------------------------
    pipeline.visualize(
        format="render_graphviz",
        output_path=str(output_dir / "pipeline" / f"{pipeline.name}.png"),
        show_execution_stage=True,
    )

    pipeline.visualize(
        format="render_mermaid",
        output_path=str(output_dir / "pipeline" / f"{pipeline.name}.md"),
        direction="LR",
        include_node_class=True,
        wrap_markdown=True,
        show_execution_stage=True,
    )

    # No statistical fit is required for HighContrastBandSelector / AdaCLIP
    # but we can still use StatisticalTrainer to run val/test passes.
    trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)

    if data_config["val_ids"]:
        logger.info("Validate: start")
        trainer.validate()
    else:
        logger.info("Validate: skipped (no val_ids)")

    logger.info("Test: start")
    trainer.test()

    # ----------------------------
    # Save pipeline and experiment config
    # ----------------------------
    results_dir = output_dir / "trained_models"
    pipeline_metadata = PipelineMetadata(
        name=pipeline.name,
        description=(
            "Statistical AdaCLIP high-contrast pipeline "
            "(LentilsAnomalyDataNode → HighContrastBandSelector → AdaCLIPDetector)"
        ),
        tags=["statistical", "adaclip", "high_contrast"],
        author="cuvis.ai",
    )

    # Save to trained_models/ (for this specific run)
    pipeline_output_path = results_dir / f"{pipeline.name}.yaml"
    logger.info("Save pipeline: {}", pipeline_output_path)
    pipeline.save_to_file(str(pipeline_output_path), metadata=pipeline_metadata)


    # Create and save complete trainrun config for reproducibility    

    trainrun_config = TrainRunConfig(
        name="adaclip_high_contrast_cli",
        pipeline=pipeline.serialize(),
        data=data_config,
        training=TrainingConfig(),
        output_dir=str(output_dir),
        loss_nodes=[],  # no learnable loss nodes
        metric_nodes=["detection_metrics"],
        freeze_nodes=[],
        unfreeze_nodes=[],
    )

    trainrun_output_path = results_dir / "adaclip_high_contrast_cli_trainrun.yaml"
    logger.info("Save trainrun config: {}", trainrun_output_path)
    trainrun_config.save_to_file(str(trainrun_output_path))

    logger.info("TensorBoard cmd: uv run tensorboard --logdir={}", monitor.output_dir)

if __name__ == "__main__":
    main()
