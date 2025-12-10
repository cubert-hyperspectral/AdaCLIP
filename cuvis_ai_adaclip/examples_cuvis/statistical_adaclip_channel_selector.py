"""AdaCLIP + SoftChannelSelector example on Lentils data.

This script combines:
  - AdaCLIP zero-shot anomaly detection (plugin)
  - Learnable SoftChannelSelector (61→3 channels) trained with gradients

Compared to the fixed band-selectors (e.g. BaselineFalseRGBSelector, CIRFalseColorSelector),
this example learns an optimal 3-channel projection optimized for AdaCLIP anomaly detection
using ground-truth masks. The channel selector is trained to optimize AdaCLIP performance.

High-level pipeline:

    LentilsAnomalyDataNode
        → SoftChannelSelector (61→3, gradient-trained)
            └→ AdaCLIPDetector (frozen, no gradients) → BCE loss + metrics
            └→ RXGlobal (optional, for comparison only)

We log:
  - AdaCLIP-based anomaly detection metrics (used for training)
  - RX-based anomaly detection metrics (optional comparison)
  - RGB + anomaly masks/overlays via TensorBoard
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import hydra
from cuvis_ai.anomaly.rx_detector import RXGlobal
from cuvis_ai.anomaly.rx_logit_head import RXLogitHead
from cuvis_ai.data.lentils_anomaly import SingleCu3sDataModule
from cuvis_ai.deciders.binary_decider import BinaryDecider
from cuvis_ai.deciders.binary_decider import QuantileBinaryDecider
from cuvis_ai.node import Node
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.pipeline.ports import PortSpec
from cuvis_ai.utils.types import Context
from cuvis_ai.node.losses import (
    AnomalyBCEWithLogits,
    SelectorDiversityRegularizer,
    SelectorEntropyRegularizer,
)
from cuvis_ai.node.metrics import AnomalyDetectionMetrics
from cuvis_ai.node.monitor import TensorBoardMonitorNode
from cuvis_ai.node.selector import SoftChannelSelector
from cuvis_ai.node.visualizations import RGBAnomalyMask, ScoreHeatmapVisualizer
from cuvis_ai.pipeline.canvas import CuvisCanvas
from cuvis_ai.training import GradientTrainer, StatisticalTrainer
from cuvis_ai.training.config import (
    CallbacksConfig,
    EarlyStoppingConfig,
    LearningRateMonitorConfig,
    ModelCheckpointConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainerConfig,
    TrainingConfig,
)
from loguru import logger
from omegaconf import DictConfig

from cuvis_ai_adaclip import (
    AdaCLIPDetector,
    download_weights,
    list_available_weights,
)

# Ensure cuvis.ai project root on sys.path when run from this repo
# From: AdaCLIP-cuvis/cuvis_ai_adaclip/examples_cuvis/statistical_adaclip_channel_selector.py
# Go up to: gitlab_cuvis_ai_3/
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # gitlab_cuvis_ai_3
CUVIS_AI_ROOT = PROJECT_ROOT / "cuvis.ai"
if str(CUVIS_AI_ROOT) not in sys.path:
    sys.path.insert(0, str(CUVIS_AI_ROOT))


DEFAULT_DATA_ROOT = CUVIS_AI_ROOT / "data" / "Lentils"
DEFAULT_TRAIN_IDS = [0, 2]
DEFAULT_VAL_IDS = [1]
DEFAULT_TEST_IDS = [3, 5]
DEFAULT_MODEL_NAME = "ViT-L-14-336"
DEFAULT_WEIGHT_NAME = "pretrained_all"
DEFAULT_PROMPT = "normal: lentils, anomaly: stones"
DEFAULT_EXPERIMENT_NAME = "statistical_adaclip_channel_selector_plugin"
DEFAULT_MONITOR_ROOT = CUVIS_AI_ROOT / "outputs" / "tensorboard"
DEFAULT_MASK_CHANNEL = 0
DEFAULT_QUANTILE = 0.995

AVAILABLE_BACKBONES = [
    "ViT-L-14-336",
    "ViT-L-14",
    "ViT-B-16",
    "ViT-B-32",
    "ViT-H-14",
]


def _expand_path(value: str | Path) -> Path:
    return Path(str(value)).expanduser()


class NormalizeRGBNode(Node):
    """Per-image, per-channel min-max normalization for RGB [B, H, W, 3].

    This node normalizes RGB images using the same logic as BandSelectorBase._compose_rgb:
    per-batch, per-channel min/max normalization to [0, 1] range.

    This ensures the selected channels from SoftChannelSelector are properly normalized
    for visualization and AdaCLIP input, matching the behavior of fixed band selectors.
    """

    INPUT_SPECS = {
        "rgb_in": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="Unnormalized RGB [B, H, W, 3]",
        )
    }

    OUTPUT_SPECS = {
        "rgb_out": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="Normalized RGB [B, H, W, 3] in [0, 1]",
        )
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(
        self,
        rgb_in: torch.Tensor,
        context: Context | None = None,  # noqa: ARG002
        **_: dict,
    ) -> dict[str, torch.Tensor]:
        """Normalize RGB using per-image, per-channel min/max normalization.

        Parameters
        ----------
        rgb_in : torch.Tensor
            Unnormalized RGB [B, H, W, 3]

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with "rgb_out" containing normalized RGB [B, H, W, 3] in [0, 1]
        """
        # rgb_in: [B, H, W, 3]
        rgb = rgb_in

        # Per-batch, per-channel min/max normalization (same logic as _compose_rgb)
        # Keep dims so broadcasting works: [B, 1, 1, 3]
        rgb_min = rgb.amin(dim=(1, 2), keepdim=True)
        rgb_max = rgb.amax(dim=(1, 2), keepdim=True)
        denom = (rgb_max - rgb_min).clamp_min(1e-8)

        rgb = (rgb - rgb_min) / denom
        rgb = rgb.clamp_(0.0, 1.0)

        return {"rgb_out": rgb}


@hydra.main(
    version_base=None,
    config_path="../../../cuvis.ai/cuvis_ai/conf",
    config_name="general",
)
def main(cfg: DictConfig) -> None:
    """Main training function with AdaCLIP + SoftChannelSelector."""

    # ----------------------------
    # Resolve configuration
    # ----------------------------
    data_root = _expand_path(getattr(cfg, "data_root", DEFAULT_DATA_ROOT))
    train_ids = list(getattr(cfg, "train_ids", DEFAULT_TRAIN_IDS))
    val_ids = list(getattr(cfg, "val_ids", DEFAULT_VAL_IDS))
    test_ids = list(getattr(cfg, "test_ids", DEFAULT_TEST_IDS))

    model_name = str(getattr(cfg, "model_name", DEFAULT_MODEL_NAME))
    weight_name = str(getattr(cfg, "weight_name", DEFAULT_WEIGHT_NAME))
    prompt_text = str(getattr(cfg, "prompt", DEFAULT_PROMPT))
    experiment_name = str(getattr(cfg, "experiment_name", DEFAULT_EXPERIMENT_NAME))
    monitor_root = _expand_path(getattr(cfg, "monitor_root", DEFAULT_MONITOR_ROOT))

    mask_channel = int(getattr(cfg, "mask_viz_channel", DEFAULT_MASK_CHANNEL))
    quantile = float(getattr(cfg, "quantile", DEFAULT_QUANTILE))
    gaussian_sigma = float(getattr(cfg, "gaussian_sigma", 4.0))

    if model_name not in AVAILABLE_BACKBONES:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {', '.join(AVAILABLE_BACKBONES)}",
        )

    if not data_root.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_root}. "
            "Set data_root or download the Lentils dataset.",
        )

    logger.info("=== AdaCLIP + SoftChannelSelector example (plugin) ===")
    logger.info("Data root: {}", data_root)
    logger.info("Splits: train={}, val={}, test={}", train_ids, val_ids, test_ids)
    logger.info("Model: {} | Weights: {}", model_name, weight_name)
    logger.info("Prompt: {}", prompt_text)

    # ----------------------------
    # Data & AdaCLIP weights
    # ----------------------------
    datamodule = SingleCu3sDataModule(
        data_dir=str(data_root),
        dataset_name="Lentils",
        batch_size=4,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        processing_mode="Reflectance",
        normalize_to_unit=False,
    )
    datamodule.setup(stage="fit")

    wavelengths = datamodule.train_ds.wavelengths

    logger.info("Dataset wavelengths: {}", wavelengths.shape)
    logger.info(
        "Wavelength range: {:.1f} - {:.1f} nm",
        wavelengths.min(),
        wavelengths.max(),
    )

    logger.info("Available AdaCLIP weights (plugin): {}", list_available_weights())
    download_weights(weight_name)

    # ----------------------------
    # Build computation graph
    # ----------------------------
    canvas_name = (
        f"{experiment_name}_{model_name}_{Path(weight_name).stem}".replace("-", "_")
    )
    canvas = CuvisCanvas(canvas_name)

    data_node = LentilsAnomalyDataNode(
        wavelengths=wavelengths,
        normal_class_ids=[
            0,
            1,
        ],  # {0: 'Unlabeled', 1: 'Lentils_black', 2: 'Lentils_brown', 3: 'Stone', 4: 'Background'}
    )
    selector = SoftChannelSelector(
        n_select=3,
        input_channels=61,
        init_method="variance",  # uses streaming stats utilities in cuvis.ai
        temperature_init=5.0,
        temperature_min=0.1,
        temperature_decay=0.9,
        hard=False,
        eps=1.0e-6,
    )
    # Normalize the 3 selected channels before feeding into AdaCLIP / visualization
    # This matches the normalization done by BandSelectorBase._compose_rgb
    rgb_normalizer = NormalizeRGBNode(name="rgb_norm")

    rx = RXGlobal(eps=1.0e-6)
    logit_head = RXLogitHead(init_scale=1.0, init_bias=0.0)

    # AdaCLIP branch: consumes the learned 3-channel RGB from the selector
    adaclip = AdaCLIPDetector(
        weight_name=weight_name,
        backbone=model_name,
        prompt_text=prompt_text,
        gaussian_sigma=gaussian_sigma,
    )

    # Student branch: RX + BCE supervised by ground-truth masks
    student_decider = BinaryDecider(threshold=0.5)
    bce_loss = AnomalyBCEWithLogits(name="bce", weight=10.0, pos_weight=None)
    entropy_loss = SelectorEntropyRegularizer(
        name="entropy",
        weight=0.1,
        target_entropy=None,
    )
    diversity_loss = SelectorDiversityRegularizer(
        name="diversity",
        weight=0.01,
    )
    metrics_student = AnomalyDetectionMetrics(name="metrics_student")

    # Teacher-style AdaCLIP metrics on the same learned RGB
    teacher_decider = QuantileBinaryDecider(quantile=quantile)
    metrics_adaclip = AnomalyDetectionMetrics(name="metrics_adaclip")

    # Visualizations: RGB input + GT + overlays for student + AdaCLIP
    score_viz_student = ScoreHeatmapVisualizer(
        name="scores_student",
        normalize_scores=True,
        up_to=3,
    )
    score_viz_adaclip = ScoreHeatmapVisualizer(
        name="scores_adaclip",
        normalize_scores=True,
        up_to=3,
    )
    mask_viz_student = RGBAnomalyMask(name="mask_student", up_to=3)
    mask_viz_adaclip = RGBAnomalyMask(name="mask_adaclip", up_to=3)

    tensorboard_node = TensorBoardMonitorNode(
        output_dir=str(monitor_root),
        run_name=canvas_name,
    )

    # ----------------------------
    # Wire the graph
    # ----------------------------
    canvas.connect(
        # Shared preprocessing + selector
        (data_node.outputs.cube, selector.data),
        # Normalize selected 3 channels for AdaCLIP & visualization
        (selector.selected_channels, rgb_normalizer.rgb_in),
        # AdaCLIP branch (for training the selector)
        # Use normalized RGB for AdaCLIP
        (rgb_normalizer.rgb_out, adaclip.inputs.rgb_image),
        (adaclip.outputs.scores, bce_loss.predictions),
        (data_node.outputs.mask, bce_loss.targets),
        (selector.weights, entropy_loss.weights),
        (selector.weights, diversity_loss.weights),
        # AdaCLIP metrics (for training monitoring)
        (adaclip.outputs.scores, teacher_decider.inputs.logits),
        (teacher_decider.outputs.decisions, metrics_adaclip.decisions),
        (data_node.outputs.mask, metrics_adaclip.targets),
        # Optional RX branch (for comparison only, not used in training)
        # Use selected (reweighted 61 channels) for RX
        (selector.selected, rx.data),
        (rx.scores, logit_head.scores),
        (logit_head.logits, student_decider.logits),
        (student_decider.outputs.decisions, metrics_student.decisions),
        (data_node.outputs.mask, metrics_student.targets),
        # Visualizations: AdaCLIP (main) - use normalized RGB
        (adaclip.outputs.scores, score_viz_adaclip.inputs.scores),
        (teacher_decider.outputs.decisions, mask_viz_adaclip.inputs.decisions),
        (data_node.outputs.mask, mask_viz_adaclip.inputs.mask),
        (rgb_normalizer.rgb_out, mask_viz_adaclip.inputs.rgb_image),
        # Visualizations: RX (optional comparison) - same normalized RGB
        (rx.scores, score_viz_student.inputs.scores),
        (student_decider.outputs.decisions, mask_viz_student.inputs.decisions),
        (data_node.outputs.mask, mask_viz_student.inputs.mask),
        (rgb_normalizer.rgb_out, mask_viz_student.inputs.rgb_image),
        # Send everything to TensorBoard
        (metrics_adaclip.metrics, tensorboard_node.metrics),
        (metrics_student.metrics, tensorboard_node.metrics),
        (score_viz_adaclip.artifacts, tensorboard_node.artifacts),
        (score_viz_student.artifacts, tensorboard_node.artifacts),
        (mask_viz_adaclip.artifacts, tensorboard_node.artifacts),
        (mask_viz_student.artifacts, tensorboard_node.artifacts),
    )

    # ----------------------------
    # Visualize graph
    # ----------------------------
    canvas.visualize(
        format="render_graphviz",
        output_path=f"outputs/canvases/{canvas.name}.png",
        show_execution_stage=True,
    )
    canvas.visualize(
        format="render_mermaid",
        output_path=f"outputs/canvases/{canvas.name}.md",
        direction="LR",
        include_node_class=True,
        wrap_markdown=True,
        show_execution_stage=True,
    )

    # ----------------------------
    # Phase 1: Statistical initialization
    # ----------------------------
    logger.info("Phase 1: Statistical initialization of selector...")
    stat_trainer = StatisticalTrainer(canvas=canvas, datamodule=datamodule)
    stat_trainer.fit()

    # ----------------------------
    # Freeze AdaCLIP and unfreeze selector
    # ----------------------------
    logger.info("Freezing AdaCLIP (no gradients) and unfreezing selector for gradient training...")
    adaclip.freeze()  # Ensure AdaCLIP stays frozen
    selector.unfreeze()  # Only selector gets gradients

    # ----------------------------
    # Phase 2: Gradient training
    # ----------------------------
    logger.info("Phase 2: Gradient training of SoftChannelSelector based on AdaCLIP outputs...")

    training_cfg = TrainingConfig(
        seed=42,
        trainer=TrainerConfig(
            max_epochs=20,
            accelerator="auto",
            callbacks=CallbacksConfig(
                early_stopping=[
                    EarlyStoppingConfig(
                        monitor="train/bce",
                        mode="min",
                    ),
                    EarlyStoppingConfig(
                        monitor="metrics_adaclip/iou",
                        min_delta=0.01,
                    ),
                ],
                model_checkpoint=ModelCheckpointConfig(
                    dirpath="./outputs/adaclip_channel_selector_checkpoints",
                    monitor="metrics_adaclip/iou",
                    verbose=True,
                ),
                learning_rate_monitor=LearningRateMonitorConfig(
                    logging_interval="epoch",
                ),
            ),
        ),
        optimizer=OptimizerConfig(
            lr=0.001,
            scheduler=SchedulerConfig(
                name="reduce_on_plateau",
                monitor="metrics_adaclip/iou",
                mode="max",
                factor=0.5,
                patience=5,
                threshold=0.01,
            ),
        ),
    )

    grad_trainer = GradientTrainer(
        canvas=canvas,
        datamodule=datamodule,
        loss_nodes=[bce_loss, entropy_loss, diversity_loss],
        metric_nodes=[metrics_student, metrics_adaclip],
        trainer_config=training_cfg.trainer,
        optimizer_config=training_cfg.optimizer,
        monitors=[tensorboard_node],
    )
    grad_trainer.fit()

    logger.info("Running validation evaluation with best checkpoint...")
    val_results = grad_trainer.validate()
    logger.info("Validation results: {}", val_results)

    logger.info("Running test evaluation with best checkpoint...")
    test_results = grad_trainer.test()
    logger.info("Test results: {}", test_results)

    logger.info("=== Training Complete ===")
    logger.info(
        "Checkpoints saved to: ./outputs/adaclip_channel_selector_checkpoints",
    )
    logger.info(f"TensorBoard logs: {tensorboard_node.output_dir}")
    logger.info(
        "View logs: uv run tensorboard "
        f"--logdir={tensorboard_node.output_dir}",
    )


if __name__ == "__main__":
    main()


