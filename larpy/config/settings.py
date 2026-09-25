"""Application configuration settings."""

import os
from dataclasses import dataclass


@dataclass
class redisConfig:
    """Redis connection settings."""
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    db: int = int(os.getenv("REDIS_DB", "0"))
    decode_responses: bool = True


@dataclass
class modelConfig:
    """Model configuration."""
    name: str = os.getenv("MODEL_NAME", "efficientnet_b0")
    scale: int = int(os.getenv("MODEL_SCALE", "10"))
    model_path: str = os.getenv("MODEL_PATH", "artifacts/models/")
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    available_models = {
        "efficientnet_b0": "efficientnet_b0",
        "efficientnet_b3": "efficientnet_b3",
        "efficientnet_b7": "efficientnet_b7",
        "vit_base": "google/vit-base-patch16-224",
        "vit_large": "google/vit-large-patch16-224",
        "resnet34": "resnet34",
        "resnet50": "resnet50",
        "swin_tiny": "microsoft/swin-tiny-patch4-window7-224",
        "swin_small": "microsoft/swin-small-patch4-window7-224",
        "clip_vit_l14": "openai/clip-vit-large-patch14",
        "densenet121": "densenet121",
        "mobilenet_v3": "google/mobilenet_v3_small_100",
    }


@dataclass
class trainingConfig:
    """Training hyperparameters."""
    data_path: str = os.getenv("DATA_PATH", "data/")
    batch_size: int = int(os.getenv("BATCH_SIZE", "32"))
    epochs: int = int(os.getenv("EPOCHS", "50"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "0.001"))
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    mixed_precision: bool = True
    num_classes: int = int(os.getenv("MODEL_SCALE", "10"))
    patience: int = 10
    min_delta: float = 0.001


@dataclass
class apiConfig:
    """API server settings."""
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    cors_origins: list = None

    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["http://localhost:8501", "http://127.0.0.1:8501"]


@dataclass
class appConfig:
    """Main application configuration."""
    redis: redisConfig = redisConfig()
    model: modelConfig = modelConfig()
    training: trainingConfig = trainingConfig()
    api: apiConfig = apiConfig()
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "true").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


app_config = appConfig()
