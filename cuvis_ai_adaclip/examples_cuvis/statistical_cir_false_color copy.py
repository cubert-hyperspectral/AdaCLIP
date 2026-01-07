"""CIR false-color AdaCLIP example using NIR->R, R->G, G->B mapping.

This script mirrors the style of:
  - examples/lad_statistical_training.py
  - examples/deep_svdd_gradient_training.py
  - examples/adaclip/statistical_baseline.py

It:
  * Builds a CuvisPipeline explicitly.
  * Uses LentilsAnomalyDataNode → CIRFalseColorSelector → AdaCLIPDetector.
  * Adds a quantile-based decider, generic anomaly metrics, and visualizations.
  * Logs everything via TensorBoardMonitorNode.
"""
from __future__ import annotations

from pathlib import Path

import time
import torch
import click
from cuvis_ai_adaclip import (
    AdaCLIPDetector,
    download_weights,
    list_available_weights,
)
from loguru import logger

from cuvis_ai.data.lentils_anomaly import SingleCu3sDataModule
# TEMPORARY: Commented out for performance testing - DELETE after testing
# from cuvis_ai.deciders.binary_decider import QuantileBinaryDecider
from cuvis_ai.node.band_selection import CIRFalseColorSelector
from cuvis_ai.node.data import LentilsAnomalyDataNode
# from cuvis_ai.node.metrics import AnomalyDetectionMetrics
# from cuvis_ai.node.monitor import TensorBoardMonitorNode
# from cuvis_ai.node.visualizations import RGBAnomalyMask, ScoreHeatmapVisualizer
# END TEMPORARY
from cuvis_ai.pipeline.pipeline import CuvisPipeline
from cuvis_ai.training import StatisticalTrainer
from cuvis_ai.training.config import (
    PipelineMetadata,
    TrainingConfig,
    TrainRunConfig,
)

from cuvis_ai_adaclip.cli_utils import AdaCLIPCLI

# Create reusable CLI instance
cli = AdaCLIPCLI("AdaCLIP CIR False Color")

@cli.add_common_options
@cli.add_data_options
@cli.add_cir_options
@cli.add_visualization_options
@click.command()
def main(**kwargs):
    """Run AdaCLIP CIR false-color (statistical) with Click CLI."""
    logger.info("=== AdaCLIP CIR false-color (statistical) ===")
    run_start = time.perf_counter()

    # Parse configuration using CLI utilities
    output_dir = Path(kwargs["output_dir"])
    data_config = cli.parse_data_config(**kwargs)

    # ----------------------------
    # Data & weights
    # ----------------------------
    datamodule = SingleCu3sDataModule(**data_config)
    datamodule.setup(stage=None)

    wavelengths = datamodule.train_ds.wavelengths
    logger.info("Wavelength range: {:.1f}-{:.1f} nm", wavelengths.min(), wavelengths.max())

    model_name = kwargs["backbone_name"]
    weight_name = kwargs["pretrained_adaclip"]
    prompt_text = kwargs["prompt_text"]
    target_class_id = kwargs["target_class_id"]

    logger.info("Available AdaCLIP weights: {}", list_available_weights())
    download_weights(weight_name)

    # ----------------------------
    # Resolve example-specific config
    # ----------------------------
    quantile = kwargs["quantile"]
    visualize_upto = kwargs["visualize_upto"]
    gaussian_sigma = kwargs["gaussian_sigma"]

    # CIR false-color wavelengths from CLI options
    nir_nm = kwargs["nir_nm"]
    red_nm = kwargs["red_nm"]
    green_nm = kwargs["green_nm"]

    logger.info(
        "Splits: train={}, val={}, test={}", data_config["train_ids"], data_config["val_ids"], data_config["test_ids"]
    )
    logger.info("Model: {} | Weights: {}", model_name, weight_name)
    logger.info("Prompt: {}", prompt_text)
    logger.info("Target anomaly class_id: {}", target_class_id)
    logger.info(
        "CIR wavelengths: NIR={:.1f} nm, Red={:.1f} nm, Green={:.1f} nm", nir_nm, red_nm, green_nm
    )

    # ----------------------------
    # Build pipeline
    # ----------------------------
    pipeline = CuvisPipeline("AdaCLIP_CIR_FalseColor")

    data_node = LentilsAnomalyDataNode(
        normal_class_ids=[0, 1],
    )
    band_selector = CIRFalseColorSelector(nir_nm=nir_nm, red_nm=red_nm, green_nm=green_nm)

    # Read optimization flags from config (default to False for non-optimized comparison)
    use_half_precision = kwargs.get("use_half_precision", True)
    enable_warmup = kwargs.get("enable_warmup", True)
    use_torch_preprocess = kwargs.get("use_torch_preprocess", True)  # Default to True for fast path

    # Use image_size=336 for faster inference (smaller input = faster processing)
    image_size = 518
    
    logger.info(f"AdaCLIP optimizations: FP16={use_half_precision}, Warmup={enable_warmup}, TorchPreprocess={use_torch_preprocess}")
    logger.info(f"AdaCLIP image_size: {image_size} (smaller = faster inference)")

    adaclip = AdaCLIPDetector(
        weight_name=weight_name,
        backbone=model_name,
        prompt_text=prompt_text,
        image_size=image_size,
        gaussian_sigma=gaussian_sigma,
        use_half_precision=use_half_precision,
        enable_warmup=enable_warmup,
        use_torch_preprocess=use_torch_preprocess,
    )

    # TEMPORARY: Commented out for performance testing - DELETE after testing
    # These nodes add ~800-900ms overhead per image (decider, metrics, visualizations, TensorBoard)
    # decider = QuantileBinaryDecider(quantile=quantile)
    # standard_metrics = AnomalyDetectionMetrics(name="detection_metrics")
    # score_viz = ScoreHeatmapVisualizer(normalize_scores=True, up_to=visualize_upto)
    # mask_viz = RGBAnomalyMask(up_to=visualize_upto)
    # monitor = TensorBoardMonitorNode(
    #     run_name=pipeline.name,
    #     output_dir=str(output_dir / ".." / "tensorboard"),
    # )
    # END TEMPORARY

    # TEMPORARY: Minimal wiring for performance testing - only data → band selector → AdaCLIP
    # This isolates AdaCLIP timing from pipeline overhead
    pipeline.connect(
        # hyperspectral → CIR false-color RGB
        (data_node.outputs.cube, band_selector.inputs.cube),
        (data_node.outputs.wavelengths, band_selector.inputs.wavelengths),
        # RGB → AdaCLIP (this is all we need for timing)
        (band_selector.outputs.rgb_image, adaclip.inputs.rgb_image),
        # TEMPORARY: Commented out all downstream nodes (decider, metrics, viz, TB)
        # (adaclip.outputs.scores, decider.inputs.logits),
        # (adaclip.outputs.scores, score_viz.inputs.scores),
        # (adaclip.outputs.scores, mask_viz.inputs.scores),
        # (decider.outputs.decisions, standard_metrics.inputs.decisions),
        # (data_node.outputs.mask, standard_metrics.inputs.targets),
        # (decider.outputs.decisions, mask_viz.inputs.decisions),
        # (data_node.outputs.mask, mask_viz.inputs.mask),
        # (band_selector.outputs.rgb_image, mask_viz.inputs.rgb_image),
        # (standard_metrics.outputs.metrics, monitor.inputs.metrics),
        # (score_viz.outputs.artifacts, monitor.inputs.artifacts),
        # (mask_viz.outputs.artifacts, monitor.inputs.artifacts),
    )
    # END TEMPORARY

    # ----------------------------
    # Move pipeline to GPU if available
    # ----------------------------
    device = cli.get_device()
    logger.info(f"Moving pipeline to device: {device}")
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

    # No statistical fit is required for CIRFalseColorSelector / AdaCLIP
    # but we can still use StatisticalTrainer to run val/test passes.
    trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)

    val_duration = 0.0
    if data_config["val_ids"]:
        val_start = time.perf_counter()
        logger.info("Running validation...")
        trainer.validate()
        val_duration = time.perf_counter() - val_start
        logger.info(f"Validation duration: {val_duration:.2f} seconds")
    else:
        logger.info("Skipping validation (no val_ids provided)")

    test_start = time.perf_counter()
    logger.info("Running test...")
    trainer.test()
    test_duration = time.perf_counter() - test_start
    logger.info(f"Test duration: {test_duration:.2f} seconds")

    # ----------------------------
    # Save pipeline and experiment config
    # ----------------------------
    results_dir = output_dir / "trained_models"
    pipeline_metadata = PipelineMetadata(
        name=pipeline.name,
        description=(
            "Statistical AdaCLIP CIR false-color pipeline "
            "(LentilsAnomalyDataNode → CIRFalseColorSelector → AdaCLIPDetector)"
        ),
        tags=["statistical", "adaclip", "cir_false_color"],
        author="cuvis.ai",
    )

    # Save to trained_models/ (for this specific run)
    pipeline_output_path = results_dir / f"{pipeline.name}.yaml"
    logger.info(f"Saving pipeline to: {pipeline_output_path}")
    pipeline.save_to_file(str(pipeline_output_path), metadata=pipeline_metadata)
    logger.info(f"  Created: {pipeline_output_path}")
    logger.info(f"  Weights: {pipeline_output_path.with_suffix('.pt')}")

    # Also save to configs/pipeline/ (for reference by experiment configs)
    pipeline_config_dir = Path("configs/pipeline")
    pipeline_config_dir.mkdir(parents=True, exist_ok=True)
    pipeline_config_path = pipeline_config_dir / "adaclip_cir_false_color.yaml"
    logger.info(f"Saving pipeline config to: {pipeline_config_path}")
    pipeline.save_to_file(str(pipeline_config_path), metadata=pipeline_metadata)
    logger.info(f"  Created: {pipeline_config_path}")

    # Create and save complete experiment config for reproducibility
    # Pipeline is embedded here for programmatic creation, but the saved YAML will
    # reference the pipeline config via defaults (updated separately)
    pipeline_config = pipeline.serialize()
    training_cfg = TrainingConfig()

    trainrun_config = TrainRunConfig(
        name="adaclip_cir_false_color_cli",
        pipeline=pipeline_config,
        data=data_config,
        training=training_cfg,
        output_dir=str(output_dir),
        loss_nodes=[],
        metric_nodes=[],  # TEMPORARY: Empty since we commented out metrics node for performance testing
        freeze_nodes=[],
        unfreeze_nodes=[],
    )

    trainrun_output_path = results_dir / "adaclip_cir_false_color_cli_trainrun.yaml"
    logger.info(f"Saving trainrun config to: {trainrun_output_path}")
    trainrun_config.save_to_file(str(trainrun_output_path))

    total_duration = time.perf_counter() - run_start
    logger.info("=== Experiment Complete ===")
    logger.info(f"Trained pipeline saved: {pipeline_output_path}")
    logger.info(f"TrainRun config saved: {trainrun_output_path}")
    # TEMPORARY: Commented out TensorBoard reference since monitor is disabled for performance testing
    # logger.info(f"TensorBoard logs: {monitor.output_dir}")
    # logger.info(f"View logs: uv run tensorboard --logdir={output_dir}")
    logger.info(
        f"Timing: validation {val_duration:.2f}s, "
        f"test {test_duration:.2f}s, "
        f"total {total_duration:.2f}s"
    )

if __name__ == "__main__":
    main()
