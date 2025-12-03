"""AdaCLIP detector node for the cuvis_ai_adaclip plugin.

This node is a **self-contained** AdaCLIP integration that:

- Uses the upstream AdaCLIP implementation from this repository
  (via :mod:`cuvis_ai_adaclip.adaclip_upstream`).
- Does **not** import any adaclip-related code from the main ``cuvis_ai``
  package (only generic node/port/typing utilities).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from cuvis_ai.node.node import Node
from cuvis_ai.pipeline.ports import PortSpec
from cuvis_ai.utils.types import Context
from PIL import Image
from torchvision.transforms import Compose

from cuvis_ai_adaclip.adaclip_upstream import AdaCLIPModel, download_weights


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
            **kwargs,
        )

        self.weight_name = weight_name
        self.backbone = backbone
        self.prompt_text = prompt_text
        self.image_size = image_size
        self.prompting_depth = prompting_depth
        self.prompting_length = prompting_length
        self.gaussian_sigma = gaussian_sigma

        # Lazy initialization - will be registered as submodule when loaded
        self._adaclip_model: AdaCLIPModel | None = None
        self._preprocess: Compose | None = None

        # Track initialization state as a buffer (survives state_dict)
        self.register_buffer(
            "_initialized_flag", torch.tensor(False, dtype=torch.bool), persistent=True
        )

    def _ensure_model_loaded(self) -> None:
        """Lazy load model on first forward pass."""
        if self._adaclip_model is not None:
            return

        # Download weights if not cached
        weight_path = download_weights(self.weight_name)

        # Determine device from current module parameters/buffers or default
        device: str | torch.device | None = None
        for param in self.parameters():
            device = param.device
            break
        if device is None:
            for buf in self.buffers():
                device = buf.device
                break
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create and register the model as a submodule
        model = AdaCLIPModel(
            backbone=self.backbone,
            image_size=self.image_size,
            prompting_depth=self.prompting_depth,
            prompting_length=self.prompting_length,
            device=str(device),
        )
        model.load_weights(weight_path)
        model.eval()

        self._adaclip_model = model
        self._preprocess = model.get_preprocess()
        self._initialized_flag.fill_(True)

    def _preprocess_rgb(self, rgb_bhwc: torch.Tensor) -> torch.Tensor:
        """Preprocess RGB tensor for model input.

        Converts BHWC tensor to preprocessed BCHW tensor suitable for the model.
        The output tensor is created on the same device as the AdaCLIP model.
        """
        if self._preprocess is None:
            raise RuntimeError("AdaCLIPDetector model not initialized")

        b = rgb_bhwc.shape[0]

        # Convert to uint8 numpy for PIL processing
        rgb_np = rgb_bhwc.detach().cpu().numpy()

        # Handle different input ranges
        if rgb_np.max() <= 1.0:
            rgb_np = (rgb_np * 255).astype(np.uint8)
        else:
            rgb_np = rgb_np.astype(np.uint8)

        # Process each image through CLIP preprocessing
        preprocessed = []
        for i in range(b):
            pil_img = Image.fromarray(rgb_np[i], mode="RGB")
            img_tensor = self._preprocess(pil_img)
            preprocessed.append(img_tensor)

        batch_tensor = torch.stack(preprocessed, dim=0)
        return batch_tensor

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

        # Run inference
        with torch.no_grad():
            anomaly_map, anomaly_score = self._adaclip_model.predict(
                img_tensor,
                prompt=self.prompt_text,
                sigma=self.gaussian_sigma,
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

        return {
            "scores": scores,
            "anomaly_score": anomaly_score,
        }

    def load(self, params: dict, serial_dir: str) -> None:  # noqa: D401, ARG002
        """Legacy no-op; configuration is restored via constructor/state_dict elsewhere."""
        return


__all__ = ["AdaCLIPDetector"]
