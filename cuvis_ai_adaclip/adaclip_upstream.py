"""High-level AdaCLIP model wrapper backed by the forked AdaCLIP repo.

This module mirrors the behavior of the ``cuvis_ai.anomaly.adaclip.model``
wrapper, but imports AdaCLIP directly from the upstream implementation in
this repository (``method/*.py``) instead of any code inside ``cuvis_ai``.
"""

from __future__ import annotations

from pathlib import Path
import time

import torch
from loguru import logger
from torch import nn
from torchvision.transforms import Compose

from method.adaclip import AdaCLIP
from method.custom_clip import (
    OPENAI_DATASET_MEAN,
    OPENAI_DATASET_STD,
    create_model_and_transforms,
    get_model_config,
)

from .weights import download_weights, list_available_weights


class AdaCLIPModel(nn.Module):
    """High-level AdaCLIP model for zero-shot anomaly detection (plugin version).

    This class wraps the upstream AdaCLIP implementation and provides a
    simplified interface for inference. It handles model loading,
    preprocessing, and inference.

    NOTE: This model follows the cuvis.ai convention of NOT calling .to(device)
    inside forward/predict. Device placement is handled externally by calling
    .to(device) on the model or the containing pipeline.
    """

    def __init__(
        self,
        backbone: str = "ViT-L-14-336",
        image_size: int = 518,
        prompting_depth: int = 4,
        prompting_length: int = 5,
        prompting_branch: str = "VL",
        prompting_type: str = "SD",
        use_hsf: bool = True,
        k_clusters: int = 20,
        output_layers: list[int] | None = None,
        device: str | None = None,  # Kept for backward compat but prefer using .to()
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.image_size = image_size
        self.prompting_depth = prompting_depth
        self.prompting_length = prompting_length
        self.prompting_branch = prompting_branch
        self.prompting_type = prompting_type
        self.use_hsf = use_hsf
        self.k_clusters = k_clusters
        self.output_layers = output_layers or [6, 12, 18, 24]

        # Store initial device hint (used only during lazy init if no tensors exist yet)
        self._init_device_hint = device

        self._clip_model: AdaCLIP | None = None
        self._model: AdaCLIP | None = None
        self._preprocess: Compose | None = None
        self._initialized = False

    @property
    def device(self) -> torch.device:
        """Discover current device from model parameters/buffers.

        This property dynamically queries the device from the model's tensors,
        ensuring correct device after .to() calls on the model or pipeline.
        """
        # First check if we have any parameters
        for param in self.parameters():
            return param.device
        # Then check buffers
        for buf in self.buffers():
            return buf.device
        # Fallback to init hint or CPU
        if self._init_device_hint is not None:
            return torch.device(self._init_device_hint)
        return torch.device("cpu")

    def _init_model(self) -> None:
        """Initialize the CLIP backbone and AdaCLIP model.

        The model is initialized on the current device (discovered from existing
        tensors or using the init hint). After initialization, the model can be
        moved to a different device using .to(device).
        """
        if self._initialized:
            return

        # Determine device for initialization
        init_device = self.device  # Uses the dynamic property
        device_str = str(init_device)

        logger.info(f"[cuvis_ai_adaclip] Initializing AdaCLIP with {self.backbone} backbone on {device_str}...")

        # Create CLIP model and transforms using upstream helpers
        # NOTE: create_model_and_transforms creates tensors on the specified device
        clip_model, preprocess_train, preprocess_val = create_model_and_transforms(
            model_name=self.backbone,
            img_size=self.image_size,
            pretrained="openai",
            device=device_str,
        )

        # Match the behavior of the trainer: use fixed-size Resize and CenterCrop
        from torchvision import transforms as T

        preprocess_val.transforms[0] = T.Resize(
            size=(self.image_size, self.image_size),
            interpolation=T.InterpolationMode.BICUBIC,
        )
        preprocess_val.transforms[1] = T.CenterCrop(size=(self.image_size, self.image_size))

        # Get channel dimensions from upstream config
        model_cfg = get_model_config(self.backbone)
        if model_cfg is None:
            raise ValueError(f"Model config for {self.backbone} not found")

        text_channel = model_cfg["embed_dim"]
        visual_channel = model_cfg["vision_cfg"]["width"]

        # Instantiate AdaCLIP as in the upstream training code
        # NOTE: AdaCLIP stores device internally but we'll update it to use dynamic discovery
        self._clip_model = AdaCLIP(
            freeze_clip=clip_model,
            text_channel=text_channel,
            visual_channel=visual_channel,
            prompting_length=self.prompting_length,
            prompting_depth=self.prompting_depth,
            prompting_branch=self.prompting_branch,
            prompting_type=self.prompting_type,
            use_hsf=self.use_hsf,
            k_clusters=self.k_clusters,
            output_layers=self.output_layers,
            device=device_str,  # Initial device hint for AdaCLIP
            image_size=self.image_size,
        )

        # Move all submodules (ProjectLayer, PromptLayer, etc.) to the correct device.
        # This is initialization-time device placement, NOT forward-time movement.
        # It ensures all newly created layers are on the same device as the CLIP backbone.
        self._clip_model.to(init_device)

        # Register as submodule so it moves with subsequent .to() calls on the parent
        self._model = self._clip_model

        self._preprocess = preprocess_val
        self._initialized = True

        logger.info(f"[cuvis_ai_adaclip] AdaCLIP initialized on {device_str}")

    def load_weights(self, weight_path: str | Path) -> None:
        """Load pretrained AdaCLIP weights from a checkpoint file."""
        self._init_model()

        weight_path = Path(weight_path)
        if not weight_path.exists():
            raise FileNotFoundError(f"Weight file not found: {weight_path}")

        logger.info(f"[cuvis_ai_adaclip] Loading AdaCLIP weights from {weight_path}")

        # PyTorch 2.6+ defaults weights_only=True, which breaks older checkpoints.
        # Explicitly request full checkpoint loading, with backward-compatible fallback.
        try:
            checkpoint = torch.load(weight_path, map_location=self.device, weights_only=False)
        except TypeError:  # Older torch versions without weights_only argument
            checkpoint = torch.load(weight_path, map_location=self.device)

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # The checkpoint keys may have 'clip_model.' prefix; our model expects keys without it.
        state_dict = {k.replace("clip_model.", ""): v for k, v in state_dict.items()}

        missing, unexpected = self._clip_model.load_state_dict(state_dict, strict=False)  # type: ignore[arg-type]
        if missing:
            logger.debug(f"[cuvis_ai_adaclip] Missing keys (first few): {missing[:5]}...")
        if unexpected:
            logger.debug(f"[cuvis_ai_adaclip] Unexpected keys (first few): {unexpected[:5]}...")

        logger.info("[cuvis_ai_adaclip] AdaCLIP weights loaded successfully")

    def get_preprocess(self) -> Compose:
        """Return the preprocessing transform to apply to PIL images."""
        self._init_model()
        return self._preprocess  # type: ignore[return-value]

    def predict(
        self,
        image: torch.Tensor,
        prompt: str = "",
        sigma: float = 4.0,
        aggregation: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run anomaly detection on an image batch.

        Parameters
        ----------
        image :
            Preprocessed image tensor ``[B, C, H, W]``. Must already be on the
            same device as the model (use pipeline.to(device) to move everything).
        prompt :
            Text prompt describing the object class.
        sigma :
            Gaussian smoothing sigma for the anomaly map.
        aggregation :
            Whether to aggregate multi-scale features.

        NOTE: This method does NOT call .to(device) on the input. The caller
        is responsible for ensuring the input is on the correct device.
        This follows cuvis.ai conventions where pipeline.to(device) handles
        all device placement.
        """
        self._init_model()

        if not self._initialized:
            raise RuntimeError("Model not initialized. Call load_weights() first.")

        # NOTE: Removed image.to(self.device) - input must already be on correct device
        # The pipeline.to(device) call handles device placement for all tensors

        # Debug: Log input device and model device
        logger.debug(f"[AdaCLIPModel.predict] Input image device: {image.device}, dtype: {image.dtype}, shape: {image.shape}")
        if self._clip_model is not None:
            model_params = list(self._clip_model.parameters())
            if model_params:
                model_device = model_params[0].device
                model_dtype = model_params[0].dtype
                logger.debug(f"[AdaCLIPModel.predict] Model device: {model_device}, dtype: {model_dtype}")

        # Use empty string as prompt if not provided
        cls_name = [prompt] if prompt else [""]

        # Run AdaCLIP in batched mode (matching cuvis.ai behavior)
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_start = time.perf_counter()
        with torch.no_grad():
            anomaly_map, anomaly_score = self._clip_model(image, cls_name, aggregation=aggregation)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_end = time.perf_counter()
        elapsed = time.time() - start_time
        inference_time_ms = (inference_end - inference_start) * 1000.0

        # Log inference time for visibility
        logger.info(f"[cuvis_ai_adaclip] AdaCLIP inference time: {elapsed:.3f}s ({inference_time_ms:.1f}ms), batch_size={image.shape[0]}, per_image={inference_time_ms/image.shape[0]:.1f}ms")

        # Ensure anomaly_score is 1D [B]
        if anomaly_score.dim() > 1:
            anomaly_score = anomaly_score.squeeze(-1)
        elif anomaly_score.dim() == 0:
            anomaly_score = anomaly_score.unsqueeze(0)

        # Optional Gaussian smoothing
        if sigma > 0:
            anomaly_map = self._gaussian_smooth_2d(anomaly_map, sigma)

        return anomaly_map, anomaly_score

    @staticmethod
    def _gaussian_smooth_2d(anomaly_map: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply 2D Gaussian smoothing on ``[B, H, W]`` anomaly maps."""
        b, h, w = anomaly_map.shape
        # Choose kernel size as 2*ceil(3*sigma)+1 (covers ~99% of Gaussian)
        kernel_size = int(2 * (int(3 * sigma) + 1) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        coords = torch.arange(kernel_size, device=anomaly_map.device, dtype=anomaly_map.dtype)
        coords = coords - (kernel_size - 1) / 2.0
        gauss_1d = torch.exp(-(coords**2) / (2 * float(sigma) ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
        gauss_2d = gauss_2d / gauss_2d.sum()

        weight = gauss_2d.view(1, 1, kernel_size, kernel_size)

        anomaly_map = anomaly_map.unsqueeze(1)  # [B, 1, H, W]
        anomaly_map = torch.nn.functional.conv2d(
            anomaly_map,
            weight,
            padding=kernel_size // 2,
        )
        anomaly_map = anomaly_map.squeeze(1)  # [B, H, W]
        return anomaly_map

    def forward(
        self,
        image: torch.Tensor,
        prompt: str = "",
        sigma: float = 4.0,
        aggregation: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Alias for :meth:`predict` to support standard nn.Module usage."""
        return self.predict(image, prompt, sigma, aggregation)


def create_adaclip_model(
    weight_name: str = "pretrained_all",
    backbone: str = "ViT-L-14-336",
    image_size: int = 518,
    prompting_depth: int = 4,
    prompting_length: int = 5,
    device: str | None = None,
) -> AdaCLIPModel:
    """Create and load an AdaCLIP model with pretrained weights."""
    model = AdaCLIPModel(
        backbone=backbone,
        image_size=image_size,
        prompting_depth=prompting_depth,
        prompting_length=prompting_length,
        device=device,
    )

    weight_path = download_weights(weight_name)
    model.load_weights(weight_path)
    model.eval()
    return model


__all__ = [
    "AdaCLIPModel",
    "create_adaclip_model",
    "download_weights",
    "list_available_weights",
    "OPENAI_DATASET_MEAN",
    "OPENAI_DATASET_STD",
]
