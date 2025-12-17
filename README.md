# AdaCLIP Plugin for cuvis.ai

This repository provides a **cuvis.ai plugin** for [AdaCLIP](https://arxiv.org/abs/2407.15795), a zero-shot anomaly detection method that adapts CLIP with hybrid learnable prompts.

> **Note**: For the original AdaCLIP repository and training code, see [README_UPSTREAM.md](README_UPSTREAM.md).

## Overview

This plugin integrates AdaCLIP into the cuvis.ai framework, allowing you to:

- Use AdaCLIP as a `Node` in cuvis.ai pipelines
- Combine AdaCLIP with cuvis.ai's band selection nodes (CIR, supervised, etc.)
- Run statistical training workflows with AdaCLIP
- Visualize results via TensorBoard

## Installation

### Prerequisites

- Python 3.10-3.13
- CUDA-capable GPU (recommended)
- For local development: [cuvis.ai](https://github.com/cubert-hyperspectral/cuvis.ai) repository cloned at `../cuvis.ai` (relative to this repo)

> **Note**: `cuvis_ai` is automatically installed as a dependency. The `pyproject.toml` is pre-configured with a path dependency (`[tool.uv.sources]`) that points to `../cuvis.ai` for local development.

### Install the Plugin

1. **Clone this repository:**
   ```bash
   git clone <repository-url>
   cd AdaCLIP-cuvis
   ```

2. **For local development (recommended):**
   
   Ensure `cuvis.ai` is cloned at the same level as `AdaCLIP-cuvis`:
   ```
   my_project/
   ├── cuvis.ai/
   └── AdaCLIP-cuvis/
   ```
   
   The `pyproject.toml` is pre-configured with a path dependency to `../cuvis.ai`.
   Install with uv (recommended):
   ```bash
   uv pip install -e .
   ```
   
   Or with pip:
   ```bash
   pip install -e .
   ```
   
   This will automatically install `cuvis_ai` from the local path as an editable dependency.

3. **For production/standalone installation:**
   
   If `cuvis_ai` is available from PyPI or another source, you can install normally:
   ```bash
   uv pip install -e .
   ```
   
   The plugin will install `cuvis_ai` from the configured source. To override the path dependency, you can modify `pyproject.toml` or use environment-specific configuration.

4. **Verify installation:**
   ```python
   from cuvis_ai_adaclip import AdaCLIPDetector, list_available_weights
   print(list_available_weights())
   ```

## Quick Start

### Basic Usage

```python
from cuvis_ai_adaclip import AdaCLIPDetector
from cuvis_ai.pipeline.canvas import CuvisCanvas

# Create the detector node
adaclip = AdaCLIPDetector(
    weight_name="pretrained_all",
    backbone="ViT-L-14-336",
    prompt_text="normal: lentils, anomaly: stones",
    gaussian_sigma=4.0,
)

# Use in a canvas
canvas = CuvisCanvas("my_pipeline")
canvas.add_node(adaclip)
# ... wire up your pipeline
```

### Download Pre-trained Weights

```python
from cuvis_ai_adaclip import download_weights, list_available_weights

# List available weights
print(list_available_weights())
# Output: ['pretrained_all', 'pretrained_mvtec_clinicdb', 'pretrained_visa_colondb']

# Download weights (automatically cached)
download_weights("pretrained_all")
```

## Node API

### `AdaCLIPDetector`

A cuvis.ai `Node` that performs zero-shot anomaly detection on RGB images.

#### Inputs

- **`rgb_image`**: `torch.Tensor` of shape `[B, H, W, 3]` (float32, 0-1 or 0-255 range)

#### Outputs

- **`scores`**: `torch.Tensor` of shape `[B, H, W, 1]` - Pixel-level anomaly scores
- **`anomaly_score`**: `torch.Tensor` of shape `[B]` - Image-level anomaly scores

#### Parameters

- **`weight_name`** (str, default: `"pretrained_all"`): Pre-trained weight identifier
- **`backbone`** (str, default: `"ViT-L-14-336"`): CLIP backbone model
  - Options: `"ViT-L-14-336"`, `"ViT-L-14"`, `"ViT-B-16"`, `"ViT-B-32"`, `"ViT-H-14"`
- **`prompt_text`** (str, default: `""`): Text prompt describing normal vs. anomaly classes
- **`image_size`** (int, default: `518`): Input image size for the model
- **`prompting_depth`** (int, default: `4`): Depth of prompt layers
- **`prompting_length`** (int, default: `5`): Length of learnable prompts
- **`gaussian_sigma`** (float, default: `4.0`): Gaussian smoothing sigma for anomaly maps

#### Example

```python
adaclip = AdaCLIPDetector(
    weight_name="pretrained_all",
    backbone="ViT-L-14-336",
    prompt_text="normal: lentils, anomaly: stones",
    gaussian_sigma=4.0,
)
```

## Examples

The plugin includes several example scripts demonstrating AdaCLIP integration with cuvis.ai:

### Available Examples

1. **`statistical_baseline.py`** - Fixed false-RGB (650/550/450 nm) band selection
2. **`statistical_cir_false_color.py`** - CIR false-color band selection (NIR→R, R→G, G→B)
3. **`statistical_cir_false_rg_color.py`** - CIR false-RG (NIR→R, R→G, visible Green→B)
4. **`statistical_high_contrast.py`** - High-contrast band selection (variance + Laplacian)
5. **`statistical_supervised_cir.py`** - Supervised CIR (windowed mRMR)
6. **`statistical_supervised_full_spectrum.py`** - Supervised full-spectrum (global mRMR)
7. **`statistical_supervised_windowed_false_rgb.py`** - Supervised windowed false-RGB

### Running Examples

All examples are located in `cuvis_ai_adaclip/examples_cuvis/` and can be run with:

```bash
# From the AdaCLIP-cuvis root directory
uv run python cuvis_ai_adaclip/examples_cuvis/statistical_baseline.py \
    data_root=../cuvis.ai/data/Lentils \
    model_name=ViT-L-14-336 \
    weight_name=pretrained_all
```

Examples support Hydra configuration overrides:

```bash
uv run python cuvis_ai_adaclip/examples_cuvis/statistical_cir_false_color.py \
    data_root=../cuvis.ai/data/Lentils \
    train_ids=[0,2] \
    test_ids=[1,3,5] \
    prompt="normal: lentils, anomaly: stones" \
    gaussian_sigma=4.0
```

## Integration with cuvis.ai

### Node Registration

The plugin automatically registers nodes when imported. For explicit registration:

```python
from cuvis_ai_adaclip import register_all_nodes

num_registered = register_all_nodes()
print(f"Registered {num_registered} nodes")
```

### Pipeline Example

```python
from cuvis_ai_adaclip import AdaCLIPDetector
from cuvis_ai.node.band_selection import CIRFalseColorSelector
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.pipeline.canvas import CuvisCanvas

canvas = CuvisCanvas("adaclip_pipeline")

# Create nodes
data_node = LentilsAnomalyDataNode(wavelengths=wavelengths, normal_class_ids=[0, 1])
band_selector = CIRFalseColorSelector(nir_nm=860.0, red_nm=670.0, green_nm=560.0)
adaclip = AdaCLIPDetector(
    weight_name="pretrained_all",
    prompt_text="normal: lentils, anomaly: stones",
)

# Wire up the pipeline
canvas.connect(
    (data_node.outputs.cube, band_selector.inputs.cube),
    (data_node.outputs.wavelengths, band_selector.inputs.wavelengths),
    (band_selector.outputs.rgb_image, adaclip.inputs.rgb_image),
)
```

## Available Weights

The plugin provides access to pre-trained AdaCLIP weights:

| Weight Name | Description | Google Drive |
|------------|-------------|--------------|
| `pretrained_all` | Trained on all datasets | [Link](https://drive.google.com/file/d/1Cgkfx3GAaSYnXPLolx-P7pFqYV0IVzZF/view) |
| `pretrained_mvtec_clinicdb` | Trained on MVTec AD & ClinicDB | [Link](https://drive.google.com/file/d/1xVXANHGuJBRx59rqPRir7iqbkYzq45W0/view) |
| `pretrained_visa_colondb` | Trained on VisA & ColonDB | [Link](https://drive.google.com/file/d/1QGmPB0ByPZQ7FucvGODMSz7r5Ke5wx9W/view) |

Weights are automatically downloaded and cached on first use.

## Architecture

This plugin repository contains:

- **`cuvis_ai_adaclip/`** - The plugin package
  - `adaclip_upstream.py` - High-level AdaCLIP model wrapper
  - `node/adaclip_node.py` - cuvis.ai Node implementation
  - `weights.py` - Weight download and management
  - `examples_cuvis/` - Example scripts for cuvis.ai integration
- **`method/`** - Upstream AdaCLIP core implementation
- **`dataset/`** - Dataset loaders (for training)
- **`adaclip_tools/`** - Utility functions

## Compatibility

- **Python**: 3.10-3.13
- **PyTorch**: Provided by cuvis.ai dependency

## Citation

If you use AdaCLIP in your research, please cite:

```bibtex
@inproceedings{AdaCLIP,
  title={AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection},
  author={Cao, Yunkang and Zhang, Jiangning and Frittoli, Luca and Cheng, Yuqi and Shen, Weiming and Boracchi, Giacomo},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```

## License

MIT License (see [LICENSE](LICENSE) file)

## Acknowledgments

- Original AdaCLIP implementation: [caoyunkang/AdaCLIP](https://github.com/caoyunkang/AdaCLIP)
- cuvis.ai framework: [cubert-hyperspectral/cuvis.ai](https://github.com/cubert-hyperspectral/cuvis.ai)

