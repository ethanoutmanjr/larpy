# Larpy — Object Price Estimator

A machine learning-powered application that estimates how expensive an object is on a scale of 1–10. Built with Python, Redis Queue (RQ), and Streamlit.

## Features

- **Image-based Price Estimation** — Upload a photo and get a price tier (1–10)
- **Async Processing** — Redis Queue (RQ) handles ML inference in the background
- **Streamlit Frontend** — Beautiful, interactive UI
- **Multiple ML Models** — Choose from EfficientNet, ViT, CLIP, and more

## Quick Start

### Prerequisites

- Python 3.10+
- Redis server running locally
- Docker (optional, for containerized deployment)

### Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/larpy.git
cd larpy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Redis
redis-server

# Run the RQ worker
python -m larpy.workers.worker

# Start the Streamlit frontend
streamlit run larpy/frontend/app.py
```

## Project Structure

```
larpy/
├── larpy/
│   ├── __init__.py
│   ├── config/          # Configuration and settings
│   ├── models/          # ML model definitions and training
│   ├── workers/         # RQ worker tasks
│   ├── api/             # API clients and service layer
│   ├── utils/           # Utility functions
│   ├── data/            # Data loading and preprocessing
│   └── frontend/        # Streamlit application
├── tests/
│   ├── unit/
│   └── integration/
├── artifacts/           # Trained models and embeddings
├── docker/              # Docker configuration
├── docs/                # Documentation
├── scripts/             # Utility scripts
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Available Models

| Model | Type | Best For | Scale |
|-------|------|----------|-------|
| **EfficientNetB0–B7** | CNN | General object price estimation | 1–10 |
| **Vision Transformer (ViT)** | Transformer | High-accuracy visual features | 1–10 |
| **CLIP (ViT-L/14)** | Multimodal | Image+text understanding | 1–10 |
| **ResNet34/50** | CNN | Lightweight baseline | 1–10 |
| **Swin Transformer** | Hierarchical | Fine-grained object analysis | 1–10 |
| **EfficientNet + XGBoost** | Hybrid | Fast inference, good accuracy | 1–10 |
| **CLIP + DeBERTa** | Multimodal | Text+image fusion pricing | 1–10 |

## License

MIT
