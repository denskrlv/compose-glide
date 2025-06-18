# ComposeGlide
ComposeGlide is an enhanced text-to-image generation framework built on OpenAI's GLIDE diffusion models. It provides improved compositional control, attention visualization capabilities, and specialized models for face generation.

![ComposeGlide Example](images/thumbnail.png)

## Features

- 🖼️ High-quality text-to-image generation
- 🧩 Enhanced compositional control over image elements
- 👤 Specialized face generation capabilities
- 🔍 Attention map visualization for model interpretability
- 🔄 Fine-tuning support for custom datasets
- 🚀 Simple API for integration into downstream applications

## Installation

Prerequisites:

- Python 3.7+
- PyTorch 1.9+
- CUDA-compatible GPU (recommended)

```python
# Installation steps
git clone --recursive https://github.com/denskrlv/compose-glide.git
cd compose-glide

# First install the local dependency
pip install -e ./glide-text2im

# Then install the main package
pip install -e .
```

## Quick Start

```python
from compose_glide import ComposeGlide

# Initialize the model
model = ComposeGlide.from_pretrained("models/glide_faces.pt")

# Generate an image from text
image = model.generate(
    prompt="A portrait of a woman with blue eyes and blonde hair", 
    guidance_scale=3.0,
    steps=100
)

# Save the image
image.save("portrait.png")
```

## Fine-tuning

ComposeGlide supports fine-tuning on custom datasets:

```python
python -m src.scripts.fine_tune \
    --dataset_path /path/to/dataset \
    --output_dir ./models/custom \
    --batch_size 8 \
    --epochs 10
```

## Project Structure

```markdown
compose-glide/
├── src/                       # Core source code
│   └── compose_glide/         # Main package
├── models/                    # Model checkpoints
└── notebooks/                 # Jupyter notebooks
```

## References

- **[GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/abs/2112.10741)** - Nichol, A., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., McGrew, B., Sutskever, I., & Chen, M. (2022)

- **[Compositional Visual Generation with Composable Diffusion Models](https://arxiv.org/abs/2206.01714)** - Liu, N., Li, S., Du, Y., Torralba, A., & Tenenbaum, J.
