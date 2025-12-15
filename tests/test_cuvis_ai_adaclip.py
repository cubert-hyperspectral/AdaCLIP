"""Tests for cuvis_ai_adaclip plugin (AdaCLIP wrapper + node).

These tests mirror the behavior of cuvis.ai's own AdaCLIP tests but
operate on the plugin package that lives inside the forked AdaCLIP repo.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from cuvis_ai_adaclip import (
    ADACLIP_WEIGHTS,
    AdaCLIPDetector,
    AdaCLIPModel,
    download_weights,
    get_weights_dir,
    list_available_weights,
)

# ============================================================================
# Weight Manager Tests
# ============================================================================


class TestWeightManager:
    """Tests for plugin weight download manager."""

    def test_list_available_weights(self) -> None:
        """Test that available weights are listed correctly."""
        weights = list_available_weights()
        assert isinstance(weights, list)
        assert len(weights) >= 3
        assert "pretrained_all" in weights
        assert "pretrained_mvtec_colondb" in weights
        assert "pretrained_visa_clinicdb" in weights

    def test_get_weights_dir_creates_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that weights directory is created if it doesn't exist."""
        # Mock the home directory
        mock_home = tmp_path / "home"
        monkeypatch.setattr(Path, "home", lambda: mock_home)

        weights_dir = get_weights_dir()
        assert weights_dir.exists()
        assert "cuvis_ai" in str(weights_dir)
        assert "adaclip" in str(weights_dir)

    def test_adaclip_weights_registry(self) -> None:
        """Test that weight registry has correct structure."""
        for _name, cfg in ADACLIP_WEIGHTS.items():
            assert "gdrive_id" in cfg
            assert "description" in cfg
            assert "filename" in cfg
            assert cfg["gdrive_id"]  # Non-empty
            assert cfg["filename"].endswith(".pth")

    def test_download_weights_invalid_name(self) -> None:
        """Test that invalid weight name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown weight"):
            download_weights("invalid_weight_name")


# ============================================================================
# AdaCLIP Detector Node Tests
# ============================================================================


class TestAdaCLIPDetectorNode:
    """Tests for plugin AdaCLIPDetector node."""

    def test_detector_initialization(self) -> None:
        """Test that AdaCLIPDetector initializes with correct parameters."""
        detector = AdaCLIPDetector(
            weight_name="pretrained_all",
            prompt_text="test prompt",
            gaussian_sigma=4.0,
        )

        assert detector.weight_name == "pretrained_all"
        assert detector.prompt_text == "test prompt"
        assert detector.gaussian_sigma == 4.0
        # Verify initialization buffer is registered
        assert "_initialized_flag" in dict(detector.named_buffers())
        assert detector._initialized_flag.item() is False

    def test_detector_port_specs(self) -> None:
        """Test that AdaCLIPDetector has correct port specs."""
        detector = AdaCLIPDetector()

        # Check input specs
        assert "rgb_image" in detector.INPUT_SPECS
        input_spec = detector.INPUT_SPECS["rgb_image"]
        assert input_spec.shape == (-1, -1, -1, 3)

        # Check output specs
        assert "scores" in detector.OUTPUT_SPECS
        assert "anomaly_score" in detector.OUTPUT_SPECS
        output_spec = detector.OUTPUT_SPECS["scores"]
        assert output_spec.shape == (-1, -1, -1, 1)

    def test_detector_serialization(self) -> None:
        """Test that AdaCLIPDetector can be serialized via hparams."""
        detector = AdaCLIPDetector(
            weight_name="pretrained_all",
            prompt_text="my prompt",
            gaussian_sigma=2.0,
        )

        # Check hparams directly (as per serialization guide - avoid custom serialize())
        assert detector.hparams["weight_name"] == "pretrained_all"
        assert detector.hparams["prompt_text"] == "my prompt"
        assert detector.hparams["gaussian_sigma"] == 2.0

    def test_detector_preprocess_handles_float32_input(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that AdaCLIPDetector handles float32 RGB inputs (0-1 range)."""
        detector = AdaCLIPDetector()

        # Mock preprocessing and model to avoid downloading weights
        mock_preprocess = lambda pil: torch.rand(3, 518, 518)  # noqa: E731
        detector._preprocess = mock_preprocess  # type: ignore[assignment]

        class MockModel:
            device = torch.device("cpu")

            def predict(
                self,
                image: torch.Tensor,
                prompt: str = "",
                sigma: float = 4.0,
                aggregation: bool = True,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return torch.rand(1, 32, 32), torch.tensor([0.5])

        detector._adaclip_model = MockModel()  # type: ignore[assignment]

        # Float32 input in 0-1 range
        rgb_float = torch.rand(1, 64, 64, 3, dtype=torch.float32)
        result = detector.forward(rgb_image=rgb_float)
        assert "scores" in result
        assert "anomaly_score" in result

    def test_detector_preprocess_handles_uint8_input(self) -> None:
        """Test that AdaCLIPDetector handles uint8 RGB inputs (0-255 range)."""
        detector = AdaCLIPDetector()

        # Mock preprocessing and model to avoid downloading weights
        detector._preprocess = MagicMock(
            return_value=torch.rand(1, 3, 518, 518)
        )  # type: ignore[assignment]

        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_model.predict.return_value = (
            torch.rand(1, 32, 32),
            torch.tensor([0.5]),
        )
        detector._adaclip_model = mock_model  # type: ignore[assignment]

        # Uint8 input in 0-255 range
        rgb_uint8 = torch.randint(0, 256, (1, 64, 64, 3), dtype=torch.uint8)
        result = detector.forward(rgb_image=rgb_uint8)
        assert "scores" in result
        assert "anomaly_score" in result

    def test_detector_outputs_on_model_device(self) -> None:
        """Test that AdaCLIPDetector returns outputs on the model's device.

        Device placement is handled by cuvis.ai's pipeline.to(); the node should not
        move outputs to match the input device.
        """
        detector = AdaCLIPDetector()

        # Mock the model to avoid downloading weights
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_model.predict.return_value = (
            torch.rand(1, 32, 32, device="cpu"),
            torch.tensor([0.5], device="cpu"),
        )
        detector._adaclip_model = mock_model  # type: ignore[assignment]
        detector._preprocess = MagicMock(
            return_value=torch.rand(1, 3, 518, 518)
        )  # type: ignore[assignment]

        # Input on CPU
        rgb_input = torch.rand(1, 64, 64, 3, dtype=torch.float32)
        result = detector.forward(rgb_image=rgb_input)

        # Outputs should be on the model's device
        assert result["scores"].device.type == "cpu"
        assert result["anomaly_score"].device.type == "cpu"


# ============================================================================
# Core Model Wrapper Tests
# ============================================================================


class TestAdaCLIPModelWrapper:
    """Tests for AdaCLIPModel wrapper class in the plugin."""

    def test_adaclip_model_initialization(self) -> None:
        """Test AdaCLIPModel initialization (without loading weights)."""
        model = AdaCLIPModel(
            backbone="ViT-L-14-336",
            image_size=518,
            prompting_depth=4,
            prompting_length=5,
        )

        assert model.backbone == "ViT-L-14-336"
        assert model.image_size == 518
        assert model._model is None  # Lazy initialization


# ============================================================================
# Integration Tests with cuvis.ai pipeline
# ============================================================================


class TestIntegrationWithCuvisAI:
    """Integration tests for AdaCLIPDetector with cuvis.ai pipeline components."""

    def test_band_selector_to_detector_pipeline(self) -> None:
        """Test a simple BaselineFalseRGBSelector -> AdaCLIPDetector pipeline wiring.

        Uses CuvisPipeline (current cuvis.ai API) instead of the legacy CuvisCanvas.
        """
        try:
            from cuvis_ai.node import BaselineFalseRGBSelector
            from cuvis_ai.pipeline.pipeline import CuvisPipeline
        except Exception as exc:  # pragma: no cover - optional dependency
            pytest.skip(f"cuvis.ai integration not available: {exc}")

        pipeline = CuvisPipeline("test_band_to_adaclip_plugin")

        selector = BaselineFalseRGBSelector()
        detector = AdaCLIPDetector()

        # This should not raise any errors; just check PortSpec compatibility.
        pipeline.connect(
            (selector.outputs.rgb_image, detector.inputs.rgb_image),
        )

        # Check that nodes are in the pipeline graph
        assert len(list(pipeline._graph.nodes())) == 2

    def test_canvas_with_baseline_strategy(self) -> None:
        """Test pipeline setup with baseline band selector and plugin AdaCLIPDetector."""
        try:
            from cuvis_ai.node import BaselineFalseRGBSelector
            from cuvis_ai.node.data import LentilsAnomalyDataNode
            from cuvis_ai.pipeline.pipeline import CuvisPipeline
        except Exception as exc:  # pragma: no cover - optional dependency
            pytest.skip(f"cuvis.ai integration not available: {exc}")

        pipeline = CuvisPipeline("test_baseline_adaclip_plugin")
        data_node = LentilsAnomalyDataNode(
            normal_class_ids=[0, 1],
            wavelengths=np.linspace(400, 900, 50),
        )
        band_selector = BaselineFalseRGBSelector()
        detector = AdaCLIPDetector()

        pipeline.connect(
            (data_node.outputs.cube, band_selector.inputs.cube),
            (data_node.outputs.wavelengths, band_selector.inputs.wavelengths),
            (band_selector.outputs.rgb_image, detector.inputs.rgb_image),
        )

        assert len(list(pipeline._graph.nodes())) == 3
