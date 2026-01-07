"""AdaCLIP detector node for the cuvis_ai_adaclip plugin.

This node is a **self-contained** AdaCLIP integration that:

- Uses the upstream AdaCLIP implementation from this repository
  (via :mod:`cuvis_ai_adaclip.adaclip_upstream`).
- Does **not** import any adaclip-related code from the main ``cuvis_ai``
  package (only generic node/port/typing utilities).
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cuvis_ai.node.node import Node
from cuvis_ai.pipeline.ports import PortSpec
from cuvis_ai.utils.types import Context
from PIL import Image
from torchvision.transforms import Compose

try:
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.v2 import functional as F_v2
    HAS_TORCHVISION_V2 = True
except ImportError:
    HAS_TORCHVISION_V2 = False
    InterpolationMode = None
    F_v2 = None

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
        use_torch_preprocess: bool = True,  # If True, use fast tensor preprocessing; if False, use PIL (exact match)
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
            use_torch_preprocess=use_torch_preprocess,
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
        self.use_torch_preprocess = use_torch_preprocess
        self._warmup_done = False

        # Lazy initialization - will be registered as submodule when loaded
        self._adaclip_model: AdaCLIPModel | None = None
        self._preprocess: Compose | None = None

        # Track initialization state as a buffer (survives state_dict)
        self.register_buffer(
            "_initialized_flag", torch.tensor(False, dtype=torch.bool), persistent=True
        )

        # Cache mean/std as buffers for efficient preprocessing (don't recreate every call)
        self.register_buffer(
            "_clip_mean",
            torch.tensor(OPENAI_DATASET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_clip_std",
            torch.tensor(OPENAI_DATASET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
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
        logger.info(f"[AdaCLIPDetector] Preprocessing method: {'tensor (fast)' if self.use_torch_preprocess else 'PIL (exact match)'}")

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

    def _preprocess_rgb_torch_pil_match(self, rgb_bhwc: torch.Tensor) -> torch.Tensor:
        """Torch implementation that closely matches the PIL pipeline.

        Pipeline: Resize((S,S), BICUBIC) -> CenterCrop(S) -> ToTensor() -> Normalize(mean,std)

        This matches upstream PIL preprocessing exactly:
        - Fixed-size resize (no aspect ratio preservation)
        - Bicubic interpolation with antialias
        - uint8 quantization before resizing (like PIL sees it) - SKIPPED when enable_gradients=True
        - ToTensor() equivalent: uint8 -> float in [0,1]
        - Normalize with cached mean/std
        
        NOTE: When enable_gradients=True, we skip uint8 conversion to preserve gradient flow.
        uint8 is not a floating-point type and breaks the computation graph.
        """
        x = rgb_bhwc  # [B, H, W, 3]

        # Bring to [0,255] like image pixels
        if x.max() <= 1.0:
            x = x * 255.0

        # When enable_gradients=True, we MUST stay in float to preserve gradient flow.
        # torch.uint8 is not a floating-point type and breaks the computation graph.
        if self.enable_gradients:
            # Differentiable path: clamp to [0,255] but stay in float32
            x = x.clamp(0, 255).to(torch.float32)  # [B, H, W, 3] float32
        else:
            # Non-differentiable path: exact PIL match with uint8 quantization
            x = x.round().clamp(0, 255).to(torch.uint8)  # [B, H, W, 3] uint8

        # BHWC -> BCHW
        x = x.permute(0, 3, 1, 2)  # [B, 3, H, W]

        S = self.image_size

        # Resize to fixed size (S,S) with bicubic, antialias if available
        if HAS_TORCHVISION_V2 and not self.enable_gradients:
            # torchvision.transforms.v2 can handle uint8 directly
            try:
                x = F_v2.resize(
                    x,
                    size=[S, S],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                )
                # CenterCrop (mostly no-op because it's already SxS)
                x = F_v2.center_crop(x, output_size=[S, S])
            except Exception:
                # Fallback: use functional interpolate
                x = x.to(torch.float32)
                x = torch.nn.functional.interpolate(
                    x, size=(S, S), mode="bicubic", align_corners=False
                )
        else:
            # Differentiable path or fallback: use interpolate (needs float)
            if x.dtype != torch.float32:
                x = x.to(torch.float32)
            try:
                x = torch.nn.functional.interpolate(
                    x,
                    size=(S, S),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
            except TypeError:
                # Older PyTorch versions don't support antialias
                x = torch.nn.functional.interpolate(
                    x, size=(S, S), mode="bicubic", align_corners=False
                )

        # ToTensor(): uint8 -> float in [0,1]
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = x / 255.0

        # Normalize using cached buffers
        mean = self._clip_mean.to(device=x.device, dtype=x.dtype)
        std = self._clip_std.to(device=x.device, dtype=x.dtype)
        x = (x - mean) / std

        return x

    def _preprocess_rgb(self, rgb_bhwc: torch.Tensor) -> torch.Tensor:
        """Preprocess RGB tensor for model input.

        When ``use_torch_preprocess`` is True (default), uses fast tensor-based
        preprocessing that closely matches PIL preprocessing with bicubic interpolation.
        This is faster than PIL preprocessing and works well for inference.

        When ``use_torch_preprocess`` is False, uses exact PIL-based preprocessing
        that matches the original CLIP preprocessing pipeline.

        The ``enable_gradients`` flag controls whether gradients flow through the
        preprocessing, but the preprocessing method choice is independent.
        """
        if self._preprocess is None:
            raise RuntimeError("AdaCLIPDetector model not initialized")

        # Fast tensor preprocessing path (default) - PIL-matching implementation
        if self.use_torch_preprocess:
            return self._preprocess_rgb_torch_pil_match(rgb_bhwc)

        # PIL preprocessing path (exact match with original CLIP preprocessing)
        else:
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
        preprocess_start = time.perf_counter()
        img_tensor = self._preprocess_rgb(rgb_image)
        preprocess_end = time.perf_counter()
        preprocess_time_ms = (preprocess_end - preprocess_start) * 1000.0
        logger.info(
            f"[AdaCLIPDetector] Preprocessing time: {preprocess_time_ms:.2f}ms "
            f"(method={'tensor' if self.use_torch_preprocess else 'PIL'})"
        )
        
        # Convert to half precision if enabled
        if self.use_half_precision and torch.cuda.is_available():
            img_tensor = img_tensor.half()

        # Warmup runs (only once, on first forward pass)
        if self.enable_warmup and not self._warmup_done and torch.cuda.is_available():
            logger.info("[AdaCLIPDetector]  Running warmup inference to optimize CUDA kernels...")
            try:
                with torch.no_grad():
                    if self.use_half_precision:
                        with torch.cuda.amp.autocast(dtype=torch.float16):
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
                with torch.cuda.amp.autocast(dtype=torch.float16):
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
            input_dtype = anomaly_map.dtype
            anomaly_map = torch.nn.functional.interpolate(
                anomaly_map.unsqueeze(1),  # [B, 1, h, w]
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)  # [B, H, W]
        anomaly_map = anomaly_map.to(input_dtype)

        scores = anomaly_map.unsqueeze(-1)  # [B, H, W, 1]

        # Convert outputs back to FP32 for compatibility with downstream nodes
        # This allows FP16 computation internally for speed while maintaining FP32 interface
        # NOTE: This conversion is necessary because OUTPUT_SPECS specifies float32, and
        # downstream nodes (decider, visualizers) expect float32 inputs.
        if scores.dtype in (torch.float16, torch.bfloat16):
            scores = scores.float()
        if anomaly_score.dtype in (torch.float16, torch.bfloat16):
            anomaly_score = anomaly_score.float()

        # DEBUG: Save intermediate tensors
        # if hasattr(self, "_debug_save_dir") and self._debug_save_dir:
        #     if context is not None:
        #         for frame_idx in range(b):
        #             self._save_debug_tensor(rgb_image[frame_idx], "input_rgb", context, frame_idx)
        #             self._save_debug_tensor(scores[frame_idx], "output_scores", context, frame_idx)
        #     else:
        #         if hasattr(self, "_debug") and self._debug:
        #             print(f"[AdaCLIPDetector] DEBUG: context is None, skipping debug save")
        # else:
        #     if hasattr(self, "_debug") and self._debug:
        #         print(f"[AdaCLIPDetector] DEBUG: _debug_save_dir not set or empty. "
        #               f"hasattr={hasattr(self, '_debug_save_dir')}, "
        #               f"value={getattr(self, '_debug_save_dir', 'NOT_SET')}")

        # # DEBUG: Print output info
        # if hasattr(self, "_debug") and self._debug:
        #     print(f"[AdaCLIPDetector] Input RGB: shape={rgb_image.shape}, "
        #           f"min={rgb_image.min().item():.4f}, max={rgb_image.max().item():.4f}, "
        #           f"mean={rgb_image.mean().item():.4f}, requires_grad={rgb_image.requires_grad}")
        #     print(f"[AdaCLIPDetector] Output scores: shape={scores.shape}, "
        #           f"min={scores.min().item():.4f}, max={scores.max().item():.4f}, "
        #           f"mean={scores.mean().item():.4f}, requires_grad={scores.requires_grad}")

        return {
            "scores": scores,
            "anomaly_score": anomaly_score,
        }

    # def _save_debug_tensor(
    #     self, tensor: torch.Tensor, name: str, context: Context | None, frame_idx: int
    # ) -> None:
    #     """Save tensor for debugging if debug mode is enabled."""
    #     if not (hasattr(self, "_debug_save_dir") and self._debug_save_dir):
    #         if hasattr(self, "_debug") and self._debug:
    #             print(f"[AdaCLIPDetector._save_debug_tensor] Skipping: _debug_save_dir not set")
    #         return

    #     if context is None:
    #         if hasattr(self, "_debug") and self._debug:
    #             print(f"[AdaCLIPDetector._save_debug_tensor] Skipping: context is None")
    #         return

    #     # Create directory structure: {stage}/epoch_{epoch}/batch_{batch_idx}/frame_{frame_idx}/
    #     # Convert ExecutionStage enum to string (e.g., ExecutionStage.TRAIN -> "train")
    #     stage_str = context.stage.value if hasattr(context.stage, "value") else str(context.stage)
    #     save_dir = (
    #         Path(self._debug_save_dir)
    #         / stage_str
    #         / f"epoch_{context.epoch:03d}"
    #         / f"batch_{context.batch_idx:03d}"
    #         / f"frame_{frame_idx:03d}"
    #     )
    #     save_dir.mkdir(parents=True, exist_ok=True)

        # Convert tensor to numpy and save
        # try:
        #     tensor_np = tensor.detach().cpu().numpy()
        #     save_path = save_dir / f"{self.name}_{name}.npy"
        #     np.save(save_path, tensor_np)
        #     if hasattr(self, "_debug") and self._debug:
        #         print(f"[AdaCLIPDetector._save_debug_tensor] Saved: {save_path}")
        # except Exception as e:
        #     if hasattr(self, "_debug") and self._debug:
        #         print(f"[AdaCLIPDetector._save_debug_tensor] ERROR saving {name}: {e}")

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


__all__ = ["AdaCLIPDetector"]
