"""cuvis_ai_adaclip: AdaCLIP wrapper and cuvis.ai plugin package.

This package lives inside the forked AdaCLIP repository and provides:

- A high-level AdaCLIP model wrapper backed by the upstream implementation
  (see :mod:`cuvis_ai_adaclip.adaclip_upstream`).
- A cuvis.ai-compatible Node, :class:`cuvis_ai_adaclip.node.AdaCLIPDetector`,
  which plugs into the cuvis.ai canvas/Node system.
"""

from .adaclip_upstream import (  # noqa: F401
    OPENAI_DATASET_MEAN,
    OPENAI_DATASET_STD,
    AdaCLIPModel,
    create_adaclip_model,
    download_weights,
    list_available_weights,
)
from .node import AdaCLIPDetector  # noqa: F401
from .weights import ADACLIP_WEIGHTS, get_weights_dir  # noqa: F401


def register_all_nodes() -> int:
    """Register all cuvis_ai_adaclip nodes in the cuvis.ai NodeRegistry.

    Returns
    -------
    int
        The number of node classes that were registered.
    """
    # Local import to avoid importing cuvis_ai at plugin import time
    from cuvis_ai.node.node_registry import auto_register_package

    return auto_register_package("cuvis_ai_adaclip")


__all__ = [
    "ADACLIP_WEIGHTS",
    "AdaCLIPDetector",
    "AdaCLIPModel",
    "create_adaclip_model",
    "download_weights",
    "get_weights_dir",
    "list_available_weights",
    "OPENAI_DATASET_MEAN",
    "OPENAI_DATASET_STD",
    "register_all_nodes",
]
