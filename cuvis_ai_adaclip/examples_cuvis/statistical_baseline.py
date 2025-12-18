"""Baseline AdaCLIP example using fixed false-RGB (650/550/450 nm).

This script provides a Click CLI for running the AdaCLIP baseline example.
It uses fixed target wavelengths (650/550/450 nm) for R/G/B channels.

It:
  * Builds a CuvisPipeline explicitly.
  * Uses LentilsAnomalyDataNode → BaselineFalseRGBSelector → AdaCLIPDetector.
  * BaselineFalseRGBSelector uses fixed target wavelengths (650/550/450 nm) for R/G/B.
  * Adds a quantile-based decider, generic anomaly metrics, and visualizations.
  * Logs everything via TensorBoardMonitorNode and saves the pipeline + experiment config.
"""
from __future__ import annotations

from pathlib import Path
import click
import torch
from cuvis_ai_adaclip import (
    AdaCLIPDetector,
    download_weights,
    list_available_weights,
)
from cuvis_ai_adaclip.cli_utils import AdaCLIPCLI, AVAILABLE_BACKBONES
from loguru import logger

from cuvis_ai.data.lentils_anomaly import SingleCu3sDataModule
from cuvis_ai.deciders.binary_decider import QuantileBinaryDecider
from cuvis_ai.node.band_selection import BaselineFalseRGBSelector
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

# Create reusable CLI instance
cli = AdaCLIPCLI("AdaCLIP Baseline")

@cli.add_common_options
@cli.add_data_options
@cli.add_wavelength_options
@cli.add_visualization_options
@click.command()
def main(**kwargs):
    """Run AdaCLIP baseline (statistical) with Click CLI."""
    logger.info("Run: AdaCLIP baseline (statistical)")

    # Parse configuration using CLI utilities
    output_dir = Path(kwargs["output_dir"]) / "adaclip_baseline"
    data_config = cli.parse_data_config(**kwargs)
    target_wavelengths = cli.parse_target_wavelengths(kwargs["target_wavelengths"])

    logger.info("Output: {}", output_dir)
    logger.info(
        "Splits: train={}, val={}, test={}",
        data_config["train_ids"],
        data_config["val_ids"],
        data_config["test_ids"],
    )
    logger.info("Model: {} | Weights: {}", kwargs["backbone_name"], kwargs["pretrained_adaclip"])
    logger.info("Prompt: {}", kwargs["prompt_text"])
    logger.info("RGB wavelengths (nm): {}", target_wavelengths)

    # ----------------------------
    # Data & weights
    # ----------------------------
    datamodule = SingleCu3sDataModule(**data_config)
    datamodule.setup(stage=None)

    wavelengths = datamodule.train_ds.wavelengths
    logger.info("Wavelengths: {:.1f}-{:.1f} nm", wavelengths.min(), wavelengths.max())

    logger.info("Available weights: {}", list_available_weights())
    download_weights(kwargs["pretrained_adaclip"])

    # ----------------------------
    # Build pipeline with named nodes
    # ----------------------------
    pipeline = CuvisPipeline("AdaCLIP_Baseline")

    # Named nodes for better tracking
    normal_class_ids = cli.parse_normal_class_ids(kwargs["normal_class_ids"])
    logger.info("Normal class_ids: {}", normal_class_ids)
    data_node = LentilsAnomalyDataNode(
        name="lentils_data_node",
        normal_class_ids=normal_class_ids,
    )
    band_selector = BaselineFalseRGBSelector(
        name="baseline_rgb_selector",
        target_wavelengths=target_wavelengths
    )

    adaclip_detector = AdaCLIPDetector(
        name="adaclip_detector",
        weight_name=kwargs["pretrained_adaclip"],
        backbone=kwargs["backbone_name"],
        prompt_text=kwargs["prompt_text"],
        gaussian_sigma=kwargs["gaussian_sigma"],
        use_half_precision=kwargs["use_half_precision"],
        enable_warmup=kwargs["enable_warmup"],
    )

    binary_decider = QuantileBinaryDecider(
        name="quantile_decider",
        quantile=kwargs["quantile"]
    )
    detection_metrics = AnomalyDetectionMetrics(
        name="anomaly_metrics"
    )
    score_visualizer = ScoreHeatmapVisualizer(
        name="score_heatmap_viz",
        normalize_scores=True,
        up_to=kwargs["visualize_upto"]
    )
    mask_visualizer = RGBAnomalyMask(
        name="rgb_anomaly_mask_viz",
        up_to=kwargs["visualize_upto"]
    )
    tensorboard_monitor = TensorBoardMonitorNode(
        name="tensorboard_monitor",
        run_name=pipeline.name,
        output_dir=str(Path(kwargs["output_dir"])/ "tensorboard"),
    )

    # Wiring: cube → band selector → AdaCLIP → decider → metrics + viz + TB
    pipeline.connect(
        # hyperspectral → RGB
        (data_node.outputs.cube, band_selector.inputs.cube),
        (data_node.outputs.wavelengths, band_selector.inputs.wavelengths),
        # RGB → AdaCLIP
        (band_selector.outputs.rgb_image, adaclip_detector.inputs.rgb_image),
        # AdaCLIP scores → decider + visualizations
        (adaclip_detector.outputs.scores, binary_decider.inputs.logits),
        (adaclip_detector.outputs.scores, score_visualizer.inputs.scores),
        (adaclip_detector.outputs.scores, mask_visualizer.inputs.scores),
        # decisions + GT for metrics + overlay
        (binary_decider.outputs.decisions, detection_metrics.inputs.decisions),
        (data_node.outputs.mask, detection_metrics.inputs.targets),
        (binary_decider.outputs.decisions, mask_visualizer.inputs.decisions),
        (data_node.outputs.mask, mask_visualizer.inputs.mask),
        (band_selector.outputs.rgb_image, mask_visualizer.inputs.rgb_image),
        # send metrics + artifacts to TensorBoard
        (detection_metrics.outputs.metrics, tensorboard_monitor.inputs.metrics),
        (score_visualizer.outputs.artifacts, tensorboard_monitor.inputs.artifacts),
        (mask_visualizer.outputs.artifacts, tensorboard_monitor.inputs.artifacts),
    )

    # ----------------------------
    # Move pipeline to GPU if available
    # ----------------------------
    device = cli.get_device()
    logger.info(f"Moved the pipeline to Device: {device}")
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

    # Run validation and testing
    trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)

    if data_config["val_ids"]:
        logger.info("Starting with the validation dataset.")
        trainer.validate()
    else:
        logger.info("Validate: skipped (no val_ids)")

    logger.info("Starting with the test dataset.")
    trainer.test()

    # ----------------------------
    # Save pipeline and experiment config
    # ----------------------------
    results_dir = output_dir / "trained_models"
    pipeline_metadata = PipelineMetadata(
        name=pipeline.name,
        description=(
            "Statistical AdaCLIP baseline pipeline "
            "(LentilsAnomalyDataNode → BaselineFalseRGBSelector → AdaCLIPDetector)"
        ),
        tags=["statistical", "adaclip", "baseline"],
        author="cuvis.ai",
    )

    # Save pipeline
    pipeline_output_path = results_dir / f"{pipeline.name}.yaml"
    logger.info("Save pipeline: {}", pipeline_output_path)
    pipeline.save_to_file(str(pipeline_output_path), metadata=pipeline_metadata)

    # Save to configs/pipeline/
    pipeline_config_dir = Path("configs/pipeline")
    pipeline_config_dir.mkdir(parents=True, exist_ok=True)
    pipeline_config_path = pipeline_config_dir / "adaclip_baseline.yaml"
    logger.info("Save pipeline config: {}", pipeline_config_path)
    pipeline.save_to_file(str(pipeline_config_path), metadata=pipeline_metadata)

    # Save experiment config
    trainrun_config = TrainRunConfig(
        name="adaclip_baseline_cli",
        pipeline=pipeline.serialize(),
        data=data_config,
        training=TrainingConfig(),
        output_dir=str(output_dir),
        loss_nodes=[],
        metric_nodes=["anomaly_metrics"],
        freeze_nodes=[],
        unfreeze_nodes=[],
    )

    trainrun_output_path = results_dir / "adaclip_baseline_cli_trainrun.yaml"
    logger.info("Save trainrun config: {}", trainrun_output_path)
    trainrun_config.save_to_file(str(trainrun_output_path))
    logger.info("TensorBoard: {}", tensorboard_monitor.output_dir)
    logger.info("TensorBoard cmd: uv run tensorboard --logdir={}", tensorboard_monitor.output_dir)

if __name__ == "__main__":
    main()
