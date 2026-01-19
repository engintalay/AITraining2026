from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ..config import AppConfig
from ..evaluator import Evaluator
import os
import contextlib

# Global objects
evaluator = None

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global evaluator
    # Load config and initialize evaluator on startup
    # Assuming config.yaml is in the root where we run this
    config_path = os.getenv("CONFIG_PATH", "config.yaml")
    if os.path.exists(config_path):
        config = AppConfig.load_from_yaml(config_path)
        evaluator = Evaluator(config)
    else:
        print(f"Warning: {config_path} not found. API generic mode.")
    yield
    # Cleanup if needed

app = FastAPI(title="LLM Fine-Tuning Platform API", lifespan=lifespan)

class CompareRequest(BaseModel):
    question: str

class ModelStats(BaseModel):
    answer: str
    tokens_used: int
    response_time_ms: float

class CompareResponse(BaseModel):
    question: str
    base_model: ModelStats
    finetuned_model: ModelStats

@app.post("/compare", response_model=CompareResponse)
async def compare_models(request: CompareRequest):
    if not evaluator:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")
    
    results = evaluator.compare(request.question)
    return results

@app.get("/health")
def health_check():
    return {"status": "ok"}
