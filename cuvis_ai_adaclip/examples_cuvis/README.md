# AdaCLIP Examples with Reusable CLI

This directory contains AdaCLIP examples that demonstrate various band selection strategies and anomaly detection pipelines. All examples now use a consistent, reusable Click CLI interface.

## 🚀 Quick Start

All examples support the same reusable CLI options for consistency:

### Common CLI Options (Available in all examples)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--backbone-name` | Choice | `ViT-L-14-336` | AdaCLIP backbone model |
| `--weight-name` | Choice | `pretrained_all` | Pretrained weights |
| `--prompt-text` | String | `anomaly` | Text prompt for AdaCLIP |
| `--output-dir` | String | `outputs/example` | Output directory |
| `--visualize-upto` | Integer | `10` | Maximum visualizations to generate |
| `--batch-size` | Integer | `4` | Batch size for data loading |
| `--quantile` | Float | `0.95` | Quantile for binary decider |
| `--gaussian-sigma` | Float | `4.0` | Gaussian sigma for AdaCLIP |
| `--use-half-precision` | Flag | `False` | Enable FP16 optimization |
| `--enable-warmup` | Flag | `False` | Enable warmup optimization |

## 📋 Available Examples

### 1. Baseline Example (Fixed False-RGB)
**File**: `statistical_baseline.py`

```bash
uv run python .\cuvis_ai_adaclip\examples_cuvis\statistical_baseline.py \
    --backbone-name ViT-L-14-336 \
    --weight-name pretrained_all \
    --visualize-upto 5 \
    --target-wavelengths 650,550,450
```

**Features**:
- Fixed false-RGB band selection (650/550/450 nm)
- Baseline performance comparison
- Simple statistical pipeline

### 2. Channel Selector with Gradient Training
**File**: `statistical_adaclip_channel_selector.py`

```bash
uv run python .\cuvis_ai_adaclip\examples_cuvis\statistical_adaclip_channel_selector.py \
    --backbone-name ViT-L-14-336 \
    --weight-name pretrained_all \
    --visualize-upto 3
```

**Features**:
- Learnable SoftChannelSelector (61→3 channels)
- Two-phase training: statistical initialization + gradient training
- RX comparison branch
- Advanced channel optimization

### 3. CIR False-Color (NIR-Red-Green)
**File**: `statistical_cir_false_color.py`

```bash
uv run python .\cuvis_ai_adaclip\examples_cuvis\statistical_cir_false_color.py \
    --backbone-name ViT-L-14-336 \
    --weight-name pretrained_all \
    --visualize-upto 5
```

**Features**:
- CIR false-color mapping (NIR→R, R→G, G→B)
- Default wavelengths: NIR=850nm, Red=650nm, Green=550nm
- Vegetation analysis optimized

### 4. CIR False-RG Color
**File**: `statistical_cir_false_rg_color.py`

```bash
uv run python .\cuvis_ai_adaclip\examples_cuvis\statistical_cir_false_rg_color.py \
    --backbone-name ViT-L-14-336 \
    --weight-name pretrained_all \
    --visualize-upto 5
```

**Features**:
- CIR false-RG mapping with visible green
- Default wavelengths: NIR=860nm, Red=670nm, Green=450nm
- Enhanced vegetation contrast

### 5. High-Contrast Band Selection
**File**: `statistical_high_contrast.py`

```bash
uv run python .\cuvis_ai_adaclip\examples_cuvis\statistical_high_contrast.py \
    --backbone-name ViT-L-14-336 \
    --weight-name pretrained_all \
    --visualize-upto 5
```

**Features**:
- Variance + Laplacian energy based selection
- Windowed spectral analysis
- Automatic high-contrast band detection

### 6. Supervised CIR (Windowed mRMR)
**File**: `statistical_supervised_cir.py`

```bash
uv run python .\cuvis_ai_adaclip\examples_cuvis\statistical_supervised_cir.py \
    --backbone-name ViT-L-14-336 \
    --weight-name pretrained_all \
    --visualize-upto 5
```

**Features**:
- Supervised band selection using mRMR
- Fisher + AUC + MI scoring
- Windowed spectral optimization
- Requires ground-truth masks

### 7. Supervised Full-Spectrum
**File**: `statistical_supervised_full_spectrum.py`

```bash
uv run python .\cuvis_ai_adaclip\examples_cuvis\statistical_supervised_full_spectrum.py \
    --backbone-name ViT-L-14-336 \
    --weight-name pretrained_all \
    --visualize-upto 5
```

**Features**:
- Global mRMR band selection
- Full-spectrum analysis
- Supervised learning with ground-truth
- Optimal band combination discovery

### 8. Supervised Windowed False-RGB
**File**: `statistical_supervised_windowed_false_rgb.py`

```bash
uv run python .\cuvis_ai_adaclip\examples_cuvis\statistical_supervised_windowed_false_rgb.py \
    --backbone-name ViT-L-14-336 \
    --weight-name pretrained_all \
    --visualize-upto 5
```

**Features**:
- Windowed false-RGB selection
- Visible spectrum optimization
- Supervised mRMR scoring
- Ground-truth guided band selection

## 🎯 Advanced Usage

### Custom Data Configuration
```bash
uv run python .\cuvis_ai_adaclip\examples_cuvis\statistical_baseline.py \
    --cu3s-file-path data/Lentils/Lentils_000.cu3s \
    --annotation-json-path data/Lentils/Lentils_000.json \
    --train-ids 0,2 \
    --val-ids 1 \
    --test-ids 3,5 \
    --normal-class-ids 0,1
```

### Performance Optimization
```bash
uv run python .\cuvis_ai_adaclip\examples_cuvis\statistical_baseline.py \
    --use-half-precision \
    --enable-warmup \
    --batch-size 8
```

### Different Backbone Models
```bash
uv run python .\cuvis_ai_adaclip\examples_cuvis\statistical_baseline.py \
    --backbone-name ViT-B-16 \
    --weight-name pretrained_all
```

## 📊 Output Structure

All examples generate consistent output in the specified `--output-dir`:

```
output_dir/
├── pipeline/                  # Pipeline visualizations
│   ├── {pipeline_name}.png    # Graphviz visualization
│   └── {pipeline_name}.md     # Mermaid diagram
├── tensorboard/               # TensorBoard logs
└── trained_models/            # Saved models and configs
    ├── {pipeline_name}.yaml   # Pipeline configuration
    └── *.yaml                 # Experiment configs
```

### 🎯 TensorBoard Visualization

After running any example, you can visualize the results using TensorBoard:

```bash
uv run tensorboard --logdir=outputs/example/../tensorboard
```

This will launch TensorBoard and display:
- **Metrics**: Anomaly detection performance metrics
- **Visualizations**: RGB images, anomaly masks, and score heatmaps
- **Artifacts**: Saved pipeline graphs and configurations
- **Training Progress**: Loss curves and evaluation metrics

**Note**: Replace `outputs/example` with your actual output directory path.

## 🖥️ TensorBoard Tips

1. **Access TensorBoard**: Open `http://localhost:6006` in your browser
2. **Compare Runs**: Use TensorBoard to compare different band selection strategies
3. **Monitor Training**: Track metrics across epochs for gradient-trained models
4. **Visual Debugging**: Examine anomaly masks and score heatmaps for quality assessment

## 🔧 Available Backbone Models

- `ViT-L-14-336` (default)
- `ViT-L-14`
- `ViT-B-16`
- `ViT-B-32`
- `ViT-H-14`

## 📈 Visualization Control

Use `--visualize-upto` to control the number of visualization outputs:
- `--visualize-upto 0`: No visualizations
- `--visualize-upto 5`: Up to 5 visualizations (recommended)
- `--visualize-upto 10`: Maximum visualizations (default)

## 🎨 Band Selection Strategies

| Example | Strategy | Learning Type | Ground Truth Required |
|---------|----------|---------------|-----------------------|
| Baseline | Fixed wavelengths | None | ❌ |
| Channel Selector | Learnable weights | Gradient | ❌ |
| CIR False-Color | Fixed CIR mapping | None | ❌ |
| High-Contrast | Variance + Laplacian | Statistical | ❌ |
| Supervised CIR | mRMR scoring | Statistical | ✅ |
| Supervised Full-Spectrum | Global mRMR | Statistical | ✅ |
| Supervised Windowed | Windowed mRMR | Statistical | ✅ |

## 💡 Tips

1. **Start with baseline**: Run `statistical_baseline.py` first to establish performance reference
2. **Compare strategies**: Use the same `--backbone-name` and `--weight-name` for fair comparisons
3. **Visualization**: Start with `--visualize-upto 3` for quick feedback, increase for detailed analysis
4. **GPU utilization**: Enable `--use-half-precision` for better GPU memory usage
5. **Reproducibility**: All examples save complete configurations for reproducible results

## 🔍 Troubleshooting

**Issue**: CUDA out of memory
**Solution**: Reduce `--batch-size` or enable `--use-half-precision`

**Issue**: Missing data files
**Solution**: Ensure CU3S files are in the correct path or specify with `--cu3s-file-path`

**Issue**: Slow performance
**Solution**: Enable `--enable-warmup` and use smaller `--batch-size`

Enjoy exploring the different AdaCLIP band selection strategies! 🚀
