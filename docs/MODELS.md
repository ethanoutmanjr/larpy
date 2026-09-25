# Available Models for Object Price Estimation

## Overview

Larpy supports multiple ML model architectures for estimating object prices on a **1-10 scale**. Each model has different trade-offs between speed, accuracy, and resource requirements.

---

## Model Comparison

| Model | Type | Params | Speed | Accuracy | Best Use Case |
|-------|------|--------|-------|----------|---------------|
| **EfficientNetB0** | CNN | 5.3M | Fast | Good | Quick inference, edge devices |
| **EfficientNetB3** | CNN | 12M | Medium | Great | Balanced speed/accuracy |
| **EfficientNetB7** | CNN | 66M | Slow | Best | Maximum accuracy |
| **ViT Base** | Transformer | 86M | Medium | Great | State-of-art visual features |
| **ViT Large** | Transformer | 307M | Slow | Best | Heavy-duty analysis |
| **ResNet50** | CNN | 25M | Fast | Good | Reliable baseline |
| **Swin Tiny** | Hierarchical | 28M | Medium | Great | Multi-scale features |
| **CLIP ViT-L/14** | Multimodal | 304M | Slow | Great | Image+text understanding |
| **MobileNet V3** | CNN | 2.5M | Very Fast | Fair | Mobile/embedded |
| **DenseNet121** | CNN | 8M | Fast | Good | Feature-rich baseline |

---

## Detailed Model Descriptions

### 1. EfficientNet (B0-B7)
**Best overall choice for most use cases.**

EfficientNet uses compound scaling to balance depth, width, and resolution. Pre-trained on ImageNet, these models are excellent at extracting visual features that correlate with object quality and value.

- **B0**: Fastest, smallest — great for real-time applications
- **B7**: Most accurate but requires more GPU memory
- **Recommended starting point**: `efficientnet_b0` or `efficientnet_b3`

```python
from larpy.models.price_estimator import priceEstimator
model = priceEstimator(model_name="efficientnet_b0", num_classes=10)
```

### 2. Vision Transformer (ViT)
**State-of-the-art visual understanding.**

ViT divides images into patches and processes them with self-attention. They capture more global context than CNNs and often produce better embeddings.

- **vit_base**: Good balance of accuracy and speed
- **vit_large**: Maximum accuracy for complex objects
- **Best for**: High-value item assessment where accuracy matters most

```python
model = priceEstimator(model_name="vit_base", num_classes=10)
```

### 3. CLIP (ViT-L/14)
**Multimodal — understands both images and text.**

CLIP can process both images and text descriptions, making it ideal when you have product descriptions alongside images. This is the most powerful option but also the most resource-intensive.

- **Use when**: You have product names/descriptions alongside images
- **Benefit**: Can distinguish between a "cheap plastic watch" and "luxury rolex"
- **Model**: `clipPriceEstimator` class

```python
from larpy.models.price_estimator import clipPriceEstimator
model = clipPriceEstimator(clip_model_name="openai/clip-vit-large-patch14")
```

### 4. ResNet (34/50)
**Reliable and well-tested baseline.**

Classic CNN architecture that's proven across countless vision tasks. Good starting point if you want something stable and well-documented.

- **ResNet34**: Lighter, faster
- **ResNet50**: More features, slightly better accuracy
- **Best for**: Baseline comparisons or when training from scratch

### 5. Swin Transformer
**Hierarchical vision transformer.**

Swin processes images at multiple scales, making it excellent for detecting both fine details (e.g., material quality) and overall shape/structure.

- **Best for**: Objects where both macro and micro features matter
- **Use case**: Furniture, art, collectibles

### 6. Hybrid Approaches

You can also combine models for even better results:

| Approach | Description | Accuracy |
|----------|-------------|----------|
| **CNN + XGBoost** | Extract features with CNN, classify with XGBoost | High |
| **CNN + LightGBM** | Similar to above, faster training | High |
| **CLIP + DeBERTa** | Image features + text embeddings fusion | Very High |
| **EfficientNet + TF-IDF** | Simple and effective | Good |

---

## Model Selection Guide

```
                    ┌──────────────────┐
                    │  What's your     │
                    │  priority?       │
                    └─────┬────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
         Speed        Accuracy    Simplicity
              │           │           │
              ▼           ▼           ▼
      EfficientNetB0   ViT Large  EfficientNetB0
      MobileNet V3     CLIP ViT   ResNet50
              │           │           │
              └─────┬─────┘─────┬───┘
                    │           │
                    ▼           ▼
            Start here → efficientnet_b0
            Then scale up as needed
```

---

## Training Recommendations

### Dataset Size Guidelines

| Dataset Size | Recommended Model | Notes |
|-------------|-------------------|-------|
| < 1,000 images | EfficientNetB0, ResNet34 | Use transfer learning heavily |
| 1,000-10,000 | EfficientNetB3, ViT Base | Fine-tune last layers |
| 10,000-100,000 | EfficientNetB7, ViT Large | Full fine-tuning viable |
| > 100,000 | CLIP ViT-L/14 | Leverage multimodal |

### Data Augmentation

Always use augmentation to improve generalization:
- Random horizontal flip
- Random rotation (+/-15 degrees)
- Color jitter
- Random cropping

---

## References

- EfficientNet: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- Vision Transformer: [Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929)
- CLIP: [Radford et al., 2021](https://arxiv.org/abs/2103.00020)
- Swin Transformer: [Liu et al., 2021](https://arxiv.org/abs/2103.14030)
