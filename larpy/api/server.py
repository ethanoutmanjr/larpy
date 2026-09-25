"""FastAPI server for price estimation endpoints."""

import os
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import requests

from larpy.config.settings import app_config
from larpy.workers.tasks import estimate_price_job, enqueue_price_estimation


app = FastAPI(
    title="Larpy API",
    description="ML-powered object price estimation API (1-10 scale)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin for origin in app_config.api.cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class priceEstimateRequest(BaseModel):
    image_url: Optional[str] = None
    image_data: Optional[str] = None
    model_name: Optional[str] = None


class priceEstimateResponse(BaseModel):
    status: str
    price_tier: Optional[int] = None
    raw_score: Optional[float] = None
    processing_time_ms: Optional[float] = None
    job_id: Optional[str] = None
    error: Optional[str] = None


class batchEstimateRequest(BaseModel):
    images: List[str]
    model_name: Optional[str] = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "larpy-api"}


@app.post("/estimate", response_model=priceEstimateResponse)
async def estimate_price(
    file: UploadFile = File(...),
    model_name: Optional[str] = None,
):
    """Estimate the price of an object from an uploaded image."""
    try:
        image_data = await file.read()
        job_id = enqueue_price_estimation(image_data, file.content_type or "JPEG")
        return priceEstimateResponse(
            status="queued",
            job_id=job_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/estimate/sync")
async def estimate_price_sync(
    file: UploadFile = File(...),
    model_name: Optional[str] = None,
):
    """
    Synchronous price estimation (blocking).
    Use for quick, single-image estimates.
    """
    try:
        from larpy.workers.tasks import get_model
        from torchvision import transforms
        from PIL import Image
        import torch

        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        model = get_model()
        if model_name:
            model = model.__class__.load(model_name=model_name, path=model_name)

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image_tensor = preprocess(image).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            score = model(image_tensor).item()

        price_tier = max(1, min(10, round(score)))

        return priceEstimateResponse(
            status="success",
            price_tier=price_tier,
            raw_score=round(score, 4),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/queue/status/{job_id}")
async def get_job_status(job_id: str):
    """Check the status of a queued job."""
    from redis import Redis
    from rq import Queue

    redis_conn = Redis(host=app_config.redis.host, port=app_config.redis.port)
    queue = Queue("price_estimation", connection=redis_conn)
    job = queue.fetch_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": job_id,
        "status": job.get_status(),
        "result": job.result,
    }


def main():
    """CLI entry point."""
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="Larpy API Server")
    parser.add_argument("--host", default=app_config.api.host)
    parser.add_argument("--port", type=int, default=app_config.api.port)

    args = parser.parse_args()
    uvicorn.run("larpy.api.server:app", host=args.host, port=args.port, reload=True)


if __name__ == "__main__":
    main()
