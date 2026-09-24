"""RQ task definitions for price estimation."""

import time
import torch
from typing import Optional, Dict, Any
from redis import Redis
from rq import Queue
from PIL import Image
import io
import numpy as np

from larpy.config.settings import app_config
from larpy.models.price_estimator import PriceEstimator

# Redis connection
redis_conn = Redis(
    host=app_config.redis.host,
    port=app_config.redis.port,
    db=app_config.redis.db,
    decode_responses=True,
)

# Create RQ queue
price_queue = Queue("price_estimation", connection=redis_conn)

# Global model instance
_model_instance: Optional[PriceEstimator] = None


def get_model() -> PriceEstimator:
    """Get or load the price estimation model (lazy loading)."""
    global _model_instance
    if _model_instance is None:
        model_path = f"{app_config.model.model_path}/{app_config.model.name}_trained.pth"
        if os_path_exists(model_path):
            _model_instance = PriceEstimator.load(
                model_name=app_config.model.name,
                path=model_path,
                num_classes=app_config.model.scale,
            )
        else:
            # Initialize fresh model for inference
            _model_instance = PriceEstimator(
                model_name=app_config.model.name,
                num_classes=app_config.model.scale,
            )
    return _model_instance


def os_path_exists(path: str) -> bool:
    """Check if file exists."""
    import os
    return os.path.exists(path)


@price_queue.job("default", timeout=300)
def estimate_price_job(
    image_data: bytes,
    image_format: str = "JPEG",
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Background job to estimate object price from image bytes.

    Args:
        image_data: Raw image bytes
        image_format: Image format (JPEG, PNG, etc.)
        model_name: Optional model override

    Returns:
        Dictionary with price estimation results
    """
    start_time = time.time()

    try:
        # Decode image
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Load model
        model = get_model()
        if model_name:
            model = PriceEstimator.load(model_name=model_name, path=model_name)

        # Preprocess
        from torchvision import transforms

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image_tensor = preprocess(image).unsqueeze(0)

        # Predict
        model.eval()
        with torch.no_grad():
            score = model(image_tensor).item()

        # Round to nearest integer on 1-10 scale
        price_tier = max(1, min(10, round(score)))

        processing_time = time.time() - start_time

        result = {
            "status": "success",
            "price_tier": price_tier,
            "raw_score": round(score, 4),
            "processing_time_ms": round(processing_time * 1000, 2),
            "model_used": model_name or app_config.model.name,
        }

        return result

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "processing_time_ms": round((time.time() - start_time) * 1000, 2),
        }


@price_queue.job("default", timeout=60)
def batch_estimate_price_job(
    image_batch: list,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process a batch of images for price estimation.

    Args:
        image_batch: List of image byte dicts
        model_name: Optional model override

    Returns:
        List of estimation results
    """
    results = []
    for img_data in image_batch:
        result = estimate_price_job(
            image_data=img_data["data"],
            image_format=img_data.get("format", "JPEG"),
            model_name=model_name,
        )
        results.append(result)
    return results


@price_queue.job("default", timeout=60)
def health_check() -> Dict[str, Any]:
    """Health check task."""
    return {
        "status": "healthy",
        "model_loaded": _model_instance is not None,
        "queue": "price_estimation",
    }


# Worker entry point helper
def enqueue_price_estimation(image_data: bytes, image_format: str = "JPEG") -> str:
    """Enqueue a price estimation job and return the job ID."""
    job = estimate_price_job.delay(image_data, image_format)
    return job.id


def enqueue_batch_estimation(image_batch: list) -> str:
    """Enqueue a batch price estimation job."""
    formatted_batch = [{"data": img, "format": "JPEG"} for img in image_batch]
    job = batch_estimate_price_job.delay(formatted_batch)
    return job.id


def main():
    """Worker entry point for RQ."""
    from larpy.workers.worker import Worker
    worker = Worker()
    worker.start()
