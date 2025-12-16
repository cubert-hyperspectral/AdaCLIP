"""AdaCLIP detector node for the cuvis_ai_adaclip plugin.

This node is a **self-contained** AdaCLIP integration that:

- Uses the upstream AdaCLIP implementation from this repository
  (via :mod:`cuvis_ai_adaclip.adaclip_upstream`).
- Does **not** import any adaclip-related code from the main ``cuvis_ai``
  package (only generic node/port/typing utilities).
"""

from __future__ import annotations

import time
from typing import Any
from contextlib import nullcontext

import numpy as np
import torch
from cuvis_ai.node.node import Node
from cuvis_ai.pipeline.ports import PortSpec
from cuvis_ai.utils.types import Context
from PIL import Image
from torchvision.transforms import Compose

from cuvis_ai_adaclip.adaclip_upstream import (
    AdaCLIPModel,
    OPENAI_DATASET_MEAN,
    OPENAI_DATASET_STD,
    download_weights,
)
from loguru import logger


class AdaCLIPDetector(Node):
    """AdaCLIP zero-shot anomaly detector node (plugin version).

    This node applies AdaCLIP for anomaly detection on RGB images.
    It takes RGB images (either uint8 or float32) and outputs pixel-level
    anomaly scores and image-level anomaly scores.

    The node uses lazy loading to avoid initializing the model until
    it's actually needed (first forward pass). The underlying AdaCLIP model
    is registered as a submodule so that ``state_dict()`` captures its weights.
    """

    INPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="RGB image [B, H, W, 3] in float32 (0-1 or 0-255 range)",
        ),
    }

    OUTPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Pixel-level anomaly scores [B, H, W, 1]",
        ),
        "anomaly_score": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Image-level anomaly score [B]",
        ),
    }

    def __init__(
        self,
        weight_name: str = "pretrained_all",
        backbone: str = "ViT-L-14-336",
        prompt_text: str = "",
        image_size: int = 518,
        prompting_depth: int = 4,
        prompting_length: int = 5,
        gaussian_sigma: float = 4.0,
        use_half_precision: bool = True,  # Enable FP16 for faster inference
        enable_warmup: bool = True,  # Warmup runs to optimize CUDA kernels
        enable_gradients: bool = False,  # If True, allow gradients to flow through AdaCLIP
        **kwargs: Any,
    ) -> None:
        # Pass all serializable arguments to super().__init__ for proper hparams capture
        super().__init__(
            weight_name=weight_name,
            backbone=backbone,
            prompt_text=prompt_text,
            image_size=image_size,
            prompting_depth=prompting_depth,
            prompting_length=prompting_length,
            gaussian_sigma=gaussian_sigma,
            use_half_precision=use_half_precision,
            enable_warmup=enable_warmup,
            enable_gradients=enable_gradients,
            **kwargs,
        )

        self.weight_name = weight_name
        self.backbone = backbone
        self.prompt_text = prompt_text
        self.image_size = image_size
        self.prompting_depth = prompting_depth
        self.prompting_length = prompting_length
        self.gaussian_sigma = gaussian_sigma
        self.use_half_precision = use_half_precision
        self.enable_warmup = enable_warmup
        self.enable_gradients = enable_gradients
        self._warmup_done = False

        # Lazy initialization - will be registered as submodule when loaded
        self._adaclip_model: AdaCLIPModel | None = None
        self._preprocess: Compose | None = None

        # Track initialization state as a buffer (survives state_dict)
        self.register_buffer(
            "_initialized_flag", torch.tensor(False, dtype=torch.bool), persistent=True
        )

    @property
    def current_device(self) -> torch.device:
        """Discover current device from module parameters/buffers."""
        for param in self.parameters():
            return param.device
        for buf in self.buffers():
            return buf.device
        return torch.device("cpu")

    def _ensure_model_loaded(self) -> None:
        """Lazy load model on first forward pass.

        The model is created and then moved to the current device of this node.
        Subsequent .to() calls on the pipeline will move the registered submodule.
        """
        if self._adaclip_model is not None:
            return

        # Download weights if not cached
        weight_path = download_weights(self.weight_name)

        # Get current device from this node's tensors
        device = self.current_device

        # Create model with device hint for initial creation
        model = AdaCLIPModel(
            backbone=self.backbone,
            image_size=self.image_size,
            prompting_depth=self.prompting_depth,
            prompting_length=self.prompting_length,
            device=str(device),  # Initial device hint
        )
        model.load_weights(weight_path)
        model.eval()

        # Freeze AdaCLIP weights by default. Even when enable_gradients is True,
        # we want gradients to flow THROUGH AdaCLIP to upstream nodes, but we
        # don't want to update AdaCLIP parameters themselves.
        for param in model.parameters():
            param.requires_grad_(False)

        # Register as a submodule so it moves with .to() calls on the pipeline
        # Using add_module ensures it's part of the module tree
        self.add_module("adaclip_model", model)
        self._adaclip_model = model
        self._preprocess = model.get_preprocess()
        self._initialized_flag.fill_(True)

        # Debug: Log device information
        model_device = next(self._adaclip_model.parameters()).device if list(self._adaclip_model.parameters()) else torch.device("cpu")
        logger.info(f"[AdaCLIPDetector] Model initialized on device: {model_device}")
        logger.info(f"[AdaCLIPDetector] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"[AdaCLIPDetector] CUDA device: {torch.cuda.get_device_name(0)}")

        # Apply optimizations after model is loaded
        if self.use_half_precision and torch.cuda.is_available():
            try:
                logger.info("[AdaCLIPDetector] Converting model to half precision (FP16) for faster inference...")
                self._adaclip_model = self._adaclip_model.half()
                if hasattr(self._adaclip_model, "_clip_model") and self._adaclip_model._clip_model is not None:
                    self._adaclip_model._clip_model = self._adaclip_model._clip_model.half()
                logger.info("[AdaCLIPDetector] ✅ Model converted to FP16")
            except Exception as e:
                logger.warning(f"[AdaCLIPDetector] ⚠️  FP16 conversion failed: {e}, continuing with FP32")
                self.use_half_precision = False

        # Enable cuDNN benchmarking for consistent input sizes
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

    def _preprocess_rgb(self, rgb_bhwc: torch.Tensor) -> torch.Tensor:
        """Preprocess RGB tensor for model input.

        When ``enable_gradients`` is False (default), this method mirrors the
        original behavior: it converts the tensor to a NumPy array, goes
        through PIL, and applies the upstream CLIP preprocessing pipeline
        (Compose of Resize/CenterCrop/ToTensor/Normalize).

        When ``enable_gradients`` is True, a fully differentiable preprocessing
        path implemented in pure PyTorch is used instead. This:

        - Keeps the inputs in tensor form (no detach / NumPy / PIL),
        - Resizes and center-crops to ``self.image_size``,
        - Normalizes using the OpenAI dataset statistics.

        The differentiable path is designed to numerically approximate the
        original preprocessing while allowing gradients to flow back to
        upstream nodes (e.g., SoftChannelSelector) during training.
        """
        if self._preprocess is None:
            raise RuntimeError("AdaCLIPDetector model not initialized")

        # Non-differentiable, exact upstream behavior for standard inference
        if not self.enable_gradients:
            b = rgb_bhwc.shape[0]

            # Convert to uint8 numpy for PIL processing (happens on CPU)
            rgb_np = rgb_bhwc.detach().cpu().numpy()

            # Handle different input ranges
            if rgb_np.max() <= 1.0:
                rgb_np = (rgb_np * 255).astype(np.uint8)
            else:
                rgb_np = rgb_np.astype(np.uint8)

            # Process each image through CLIP preprocessing (creates CPU tensors)
            preprocessed = []
            for i in range(b):
                pil_img = Image.fromarray(rgb_np[i], mode="RGB")
                img_tensor = self._preprocess(pil_img)
                preprocessed.append(img_tensor)

            batch_tensor = torch.stack(preprocessed, dim=0)

            target_device = self.current_device
            return batch_tensor.to(target_device)

        # Differentiable preprocessing path for training mode
        # rgb_bhwc: [B, H, W, 3], float32 in [0,1] or [0,255]
        b, h, w, c = rgb_bhwc.shape
        # assert c == 3, f"Expected 3 channels for RGB, got {c}"

        # Normalize to [0,1]
        rgb = rgb_bhwc
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        rgb = torch.clamp(rgb, 0.0, 1.0)

        # Convert to BCHW
        rgb_bchw = rgb.permute(0, 3, 1, 2)  # [B, 3, H, W]

        # Resize with preserved aspect ratio then center-crop to image_size
        target_size = self.image_size

        # Compute scale to make the smaller side == target_size, then crop center
        in_h, in_w = h, w
        if in_h == target_size and in_w == target_size:
            resized = rgb_bchw
        else:
            scale = target_size / min(in_h, in_w)
            new_h = int(round(in_h * scale))
            new_w = int(round(in_w * scale))
            resized = torch.nn.functional.interpolate(
                rgb_bchw,
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            )

        _, _, rh, rw = resized.shape
        top = max(0, (rh - target_size) // 2)
        left = max(0, (rw - target_size) // 2)
        bottom = top + target_size
        right = left + target_size
        cropped = resized[:, :, top:bottom, left:right]  # [B, 3, S, S]

        # If input is smaller than target_size in some dimension, pad reflectively
        if cropped.shape[2] != target_size or cropped.shape[3] != target_size:
            pad_h = max(0, target_size - cropped.shape[2])
            pad_w = max(0, target_size - cropped.shape[3])
            padding = (
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2,
            )  # (left, right, top, bottom)
            cropped = torch.nn.functional.pad(cropped, padding, mode="reflect")
            cropped = cropped[:, :, :target_size, :target_size]

        # Normalize using OpenAI dataset mean/std
        mean = torch.tensor(OPENAI_DATASET_MEAN, dtype=cropped.dtype, device=cropped.device).view(
            1, 3, 1, 1
        )
        std = torch.tensor(OPENAI_DATASET_STD, dtype=cropped.dtype, device=cropped.device).view(
            1, 3, 1, 1
        )
        normalized = (cropped - mean) / std

        return normalized

    def forward(
        self,
        rgb_image: torch.Tensor,
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Run AdaCLIP inference on RGB images.

        Parameters
        ----------
        rgb_image :
            RGB image tensor in BHWC format with 3 channels.
        """
        self._ensure_model_loaded()
        assert self._adaclip_model is not None

        b, h, w, _ = rgb_image.shape

        # Preprocess images
        img_tensor = self._preprocess_rgb(rgb_image)
        
        # Convert to half precision if enabled
        if self.use_half_precision and torch.cuda.is_available():
            img_tensor = img_tensor.half()

        # Warmup runs (only once, on first forward pass)
        if self.enable_warmup and not self._warmup_done and torch.cuda.is_available():
            logger.info("[AdaCLIPDetector]  Running warmup inference to optimize CUDA kernels...")
            try:
                with torch.no_grad():
                    if self.use_half_precision:
                        with torch.cuda.amp.autocast():
                            _ = self._adaclip_model.predict(
                                img_tensor[:1] if img_tensor.shape[0] > 1 else img_tensor,
                                prompt=self.prompt_text,
                                sigma=self.gaussian_sigma,
                            )
                    else:
                        _ = self._adaclip_model.predict(
                            img_tensor[:1] if img_tensor.shape[0] > 1 else img_tensor,
                            prompt=self.prompt_text,
                            sigma=self.gaussian_sigma,
                        )
                    torch.cuda.synchronize()
                self._warmup_done = True
                logger.info("[AdaCLIPDetector]  Warmup complete")
            except Exception as e:
                logger.warning(f"[AdaCLIPDetector]   Warmup failed: {e}, continuing without warmup")

        # Run inference
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_start = time.perf_counter()
        # Gradients are disabled by default (inference mode). When
        # enable_gradients=True, we allow autograd to track operations so that
        # upstream nodes (e.g., channel selectors) can be trained using losses
        # defined on AdaCLIP outputs, while AdaCLIP weights remain frozen.
        grad_ctx = nullcontext if self.enable_gradients else torch.no_grad
        with grad_ctx():
            # Use autocast for additional speedup (works with FP16)
            if self.use_half_precision and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    anomaly_map, anomaly_score = self._adaclip_model.predict(
                        img_tensor,
                        prompt=self.prompt_text,
                        sigma=self.gaussian_sigma,
                        enable_gradients=self.enable_gradients,
                    )
            else:
                anomaly_map, anomaly_score = self._adaclip_model.predict(
                    img_tensor,
                    prompt=self.prompt_text,
                    sigma=self.gaussian_sigma,
                    enable_gradients=self.enable_gradients,
                )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_end = time.perf_counter()
        inference_time_ms = (inference_end - inference_start) * 1000.0
        
        logger.info(
            f"[AdaCLIPDetector] Inference time: {inference_time_ms:.1f}ms "
            f"(batch_size={b}, per_image={inference_time_ms/b:.1f}ms)"
        )

        # Resize anomaly map back to original size if needed
        if anomaly_map.shape[1] != h or anomaly_map.shape[2] != w:
            anomaly_map = torch.nn.functional.interpolate(
                anomaly_map.unsqueeze(1),  # [B, 1, h, w]
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)  # [B, H, W]

        scores = anomaly_map.unsqueeze(-1)  # [B, H, W, 1]

        # Convert outputs back to FP32 for compatibility with downstream nodes
        # This allows FP16 computation internally for speed while maintaining FP32 interface
        # NOTE: This conversion is necessary because OUTPUT_SPECS specifies float32, and
        # downstream nodes (decider, visualizers) expect float32 inputs.
        if scores.dtype == torch.float16:
            scores = scores.float()
        if anomaly_score.dtype == torch.float16:
            anomaly_score = anomaly_score.float()

        return {
            "scores": scores,
            "anomaly_score": anomaly_score,
        }

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load state dict with key remapping for _clip_model/_model compatibility.
        
        Handles backward compatibility when saved weights use different attribute names.
        The AdaCLIPModel wrapper now only uses _clip_model (not _model), but older
        saved weights might have _model keys that need to be remapped.
        """
        # Create a remapped state_dict that handles both _clip_model and _model keys
        remapped_state_dict = {}
        for key, value in state_dict.items():
            # If key has adaclip_model._model, remap to adaclip_model._clip_model
            if "adaclip_model._model." in key:
                new_key = key.replace("adaclip_model._model.", "adaclip_model._clip_model.")
                remapped_state_dict[new_key] = value
                # Also keep original in case both exist
                remapped_state_dict[key] = value
            else:
                remapped_state_dict[key] = value
        
        # Call parent load_state_dict with remapped keys
        # Use strict=False to allow partial loading if some keys don't match
        result = super().load_state_dict(remapped_state_dict, strict=False)
        
        # Log any remaining mismatches
        if hasattr(result, "missing_keys") and result.missing_keys:
            logger.warning(
                f"[AdaCLIPDetector] Missing keys after remapping (first 5): "
                f"{list(result.missing_keys)[:5]}..."
            )
        if hasattr(result, "unexpected_keys") and result.unexpected_keys:
            logger.debug(
                f"[AdaCLIPDetector] Unexpected keys (first 5): "
                f"{list(result.unexpected_keys)[:5]}..."
            )
        
        return result

    def load(self, params: dict, serial_dir: str) -> None:  # noqa: D401, ARG002
        """Legacy no-op; configuration is restored via constructor/state_dict elsewhere."""
        return


__all__ = ["AdaCLIPDetector"]
