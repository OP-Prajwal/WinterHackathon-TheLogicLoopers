from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
from typing import List, Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    prompt: str

class AnalysisResponse(BaseModel):
    is_safe: bool
    confidence: float
    issues: List[str]

@app.get("/api/metrics")
async def get_metrics():
    return {
        "totalScans": 12345 + random.randint(0, 100),
        "securityScore": 98.2,
        "threatsDetected": 24,
        "databaseSize": "1.2TB"
    }

@app.get("/api/rank-history")
async def get_rank_history():
    return [
        {"name": "Jan", "rank": random.randint(300, 450)},
        {"name": "Feb", "rank": random.randint(250, 350)},
        {"name": "Mar", "rank": random.randint(200, 300)},
        {"name": "Apr", "rank": 278},
        {"name": "May", "rank": 189},
        {"name": "Jun", "rank": 239},
        {"name": "Jul", "rank": 349},
    ]

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_prompt(request: AnalysisRequest):
    is_safe = "poison" not in request.prompt.lower()
    return {
        "is_safe": is_safe,
        "confidence": 0.99 if not is_safe else 0.95,
        "issues": ["Potential poisoning attempt detected"] if not is_safe else []
    }

@app.post("/api/purify")
async def start_purification():
    return {"status": "started", "estimated_time": "30s"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
