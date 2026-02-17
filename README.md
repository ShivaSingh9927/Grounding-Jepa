# Grounding-JEPA Architecture

This project implements the **Grounding-JEPA** architecture for Medical Image Grounding (MIG). The architecture is designed to leverage world-model features for precise anatomical understanding and spatial localization in X-ray scans.

## Architecture Overview

The model is divided into three functional blocks: **The Observer**, **The Bridge**, and **The Reasoner**.

### 1. The Observer (Vision)
- **Backbone**: I-JEPA (Image Joint-Embedding Predictive Architecture)
- **Input**: 224 x 224 high-resolution X-ray scans.
- **Logic**: Predicts missing anatomical structures in latent space, providing built-in anatomical continuity.
- **Output**: Patch-level embeddings (32 x 32 grid = 1024 tokens).
- **Configuration**: Uses `patch_size=7` to achieve high spatial density.

### 2. The Bridge (Alignment)
- **Module**: `C-Abstractor` (Convolutional Abstractor).
- **Structure**: Lightweight convolutional block (stride 2) followed by an MLP.
- **Function**: Reduces the 1024 patch tokens from I-JEPA down to a manageable **256 "Visual Words"** while maintaining the spatial grid.
- **Goal**: Summarizes local anatomy for the LLM.

### 3. The Reasoner (Language)
- **Backbone**: Qwen2.5-VL-7B.
- **Input**: Visual tokens from C-Abstractor + Grounding Prompt.
- **Output Format**: Chain-of-Box (CoT) reasoning with final coordinates.
  - Template: `<think> Reasoning... </think> <answer> [x1, y1, x2, y2] </answer>`

## Project Structure

```text
Grounding-Jepa/
├── modules/
│   ├── observer.py   # I-JEPA Vision Backbone
│   ├── bridge.py     # C-Abstractor implementation
│   └── reasoner.py   # Qwen2.5-VL Language Reasoner
├── model.py          # Unified GroundingJepa Model
└── inference_example.py # Usage demonstration
```

## Setup and Usage

### Requirements
- `torch >= 2.5.1`
- `transformers` (latest with Qwen2.5-VL support)
- Pretrained I-JEPA weights (refer to `/nuvodata/User_data/shiva/ijepa`)
- access to Qwen2.5-VL-7B-Instruct

### Example Usage

```python
from model import GroundingJepa
import torch

# Initialize model
model = GroundingJepa(
    ijepa_variant='vit_huge',
    patch_size=7,
    qwen_model_id='Qwen/Qwen2.5-VL-7B-Instruct'
)

# Load pretrained weights (Optional)
# model.observer.load_pretrained('path/to/ijepa_checkpoint.pth')

# Prepare input
image = torch.randn(1, 3, 224, 224).cuda()
prompt = "Identify the pediatric lung field."
# Tokenize prompt...

# Forward pass
# output = model(image, input_ids)
```

## References
This implementation refers to:
- **MedGround-R1**: For Grounding-reasoning logic and training strategies.
- **I-JEPA**: For the world-model vision backbone.
- **Qwen2.5-VL**: For the state-of-the-art vision-language reasoning backbone.
