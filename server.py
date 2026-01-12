import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Import our pipeline components to ensure they are loadable and ready for future use
from poison_guard.pipeline.trainer import PoisonGuardTrainer
# In a real scenario, we would load a trained model here.

app = FastAPI(title="Poison Guard API", version="0.1.0")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"], # Common Vite/React ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Metric(BaseModel):
    label: str
    value: str
    change: str
    trend: str

class RankPoint(BaseModel):
    name: str
    rank: int

@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "poison-guard-backend"}

@app.get("/api/metrics")
def get_metrics() -> List[Metric]:
    # TODO: Replace with real metrics from PoisonGuardTrainer/Model
    return [
        Metric(label="Total Scans", value="12,345", change="+12% from last month", trend="up"),
        Metric(label="Security Score", value="98.2%", change="+2.1% from last week", trend="up"),
        Metric(label="Threats Detected", value="24", change="-5% from last month", trend="down"),
        Metric(label="Database Size", value="1.2TB", change="+0.5% from last month", trend="neutral"),
    ]

@app.get("/api/effective-rank")
def get_effective_rank() -> List[RankPoint]:
    # TODO: Replace with real effective rank calculations
    return [
        {"name": 'Jan', "rank": 400},
        {"name": 'Feb', "rank": 300},
        {"name": 'Mar', "rank": 200},
        {"name": 'Apr', "rank": 278},
        {"name": 'May', "rank": 189},
        {"name": 'Jun', "rank": 239},
        {"name": 'Jul', "rank": 349},
    ]

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
