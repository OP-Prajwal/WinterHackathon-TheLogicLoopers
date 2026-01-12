"""
PoisonGuard API Server
FastAPI WebSocket server for real-time training monitoring.
"""
import asyncio
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import torch
import numpy as np
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
import shutil
import io
import os

MODEL_PATH = "model_checkpoint.pt"

# Import PoisonGuard components
from src.poison_guard.data.adapters.tabular import BRFSSDiabetesAdapter
from src.poison_guard.baselines.calculator import FingerprintCalculator
from src.poison_guard.models.encoders.tabular_mlp import TabularMLPEncoder
from src.poison_guard.models.heads.mlp import ProjectionHead
from src.poison_guard.pipeline.auditor import PipelineAuditor
from src.poison_guard.pipeline.trainer import PoisonGuardTrainer
from src.poison_guard.validation.injector import TabularPoisonInjector


# Global state
class TrainingState:
    def __init__(self):
        self.is_running = False
        self.is_poisoned = False
        self.batch = 0
        self.trainer: Optional[PoisonGuardTrainer] = None
        self.clean_tensors: Optional[torch.Tensor] = None
        self.poisoned_tensors: Optional[torch.Tensor] = None
        self.adapter: Optional[BRFSSDiabetesAdapter] = None
        self.fingerprint = None
        self.connections: list[WebSocket] = []
        self.task: Optional[asyncio.Task] = None
        
state = TrainingState()


def load_data():
    """Load and prepare BRFSS dataset."""
    print("[Server] Loading BRFSS dataset...")
    df = pd.read_csv("data/diabetes_brfss.csv")
    print(f"[Server] Loaded {len(df)} records.")
    
    # Initialize adapter
    state.adapter = BRFSSDiabetesAdapter()
    state.adapter.fit(df)
    state.clean_tensors = state.adapter.transform(df)
    
    # Create poisoned version
    injector = TabularPoisonInjector()
    trigger = {'BMI': 99.0, 'MentHlth': 30.0}
    poisoned_df = injector.inject_backdoor(df, trigger, poison_fraction=0.10)
    state.poisoned_tensors = state.adapter.transform(poisoned_df)
    
    # Create encoder and compute fingerprint
    encoder = TabularMLPEncoder(input_dim=state.clean_tensors.shape[1], output_dim=64)
    with torch.no_grad():
        initial_h = encoder(state.clean_tensors[:5000])  # Sample for fingerprint
    
    fp_calc = FingerprintCalculator()
    state.fingerprint = fp_calc.compute("BRFSS_Diabetes", "v1.0", initial_h)
    
    # Initialize trainer
    head = ProjectionHead(input_dim=64, hidden_dim=32, output_dim=32)
    auditor = PipelineAuditor(baseline=state.fingerprint, monitor_window=20)
    state.trainer = PoisonGuardTrainer(encoder, head, auditor=auditor)
    
    print("[Server] Data loaded and trainer initialized.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data and model on startup."""
    load_data()
    
    # Try to load existing model
    if os.path.exists(MODEL_PATH):
        print(f"[Server] Loading saved model from {MODEL_PATH}...")
        try:
            checkpoint = torch.load(MODEL_PATH)
            state.trainer.encoder.load_state_dict(checkpoint['encoder'])
            state.trainer.head.load_state_dict(checkpoint['head'])
            print("[Server] Model weights loaded successfully.")
            
            # CRITICAL: Recompute fingerprint with the LOADED weights
            # Otherwise baseline matches random init weights -> everything looks like anomaly
            print("[Server] Recomputing security baseline for loaded model...")
            with torch.no_grad():
                # Ensure eval mode for consistent fingerprint
                state.trainer.encoder.eval() 
                state.trainer.encoder.eval() 
                clean_sample = state.clean_tensors[:5000].to(state.trainer.device) # Use trainer's device property
                # Note: clean_tensors is CPU by default in load_data, encoder might be CUDA
                # keeping it simple:
                initial_h = state.trainer.encoder(clean_sample)
                state.trainer.encoder.train() # Revert to train mode just in case
            
            fp_calc = FingerprintCalculator()
            state.fingerprint = fp_calc.compute("BRFSS_Diabetes", "v1.0", initial_h)
            # Update auditor's baseline
            state.trainer.auditor.baseline = state.fingerprint
            state.trainer.auditor._drift.baseline = state.fingerprint
            print("[Server] Baseline synchronized.")
            
        except Exception as e:
            print(f"[Server] Failed to load model: {e}")
            
    yield
    
    # Save model on shutdown
    print(f"[Server] Saving model to {MODEL_PATH}...")
    torch.save({
        'encoder': state.trainer.encoder.state_dict(),
        'head': state.trainer.head.state_dict()
    }, MODEL_PATH)
    print("[Server] Model saved.")


app = FastAPI(title="PoisonGuard API", lifespan=lifespan)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def broadcast(message: Dict[str, Any]):
    """Send message to all connected clients."""
    data = json.dumps(message, default=str)
    disconnected = []
    for ws in state.connections:
        try:
            await ws.send_text(data)
        except:
            disconnected.append(ws)
    for ws in disconnected:
        state.connections.remove(ws)


async def training_loop():
    """Main training loop that runs in background."""
    print("[Server] Training loop started.")
    
    while state.is_running:
        state.batch += 1
        
        # Choose data source
        data = state.poisoned_tensors if state.is_poisoned else state.clean_tensors
        
        # Sample random batch
        indices = torch.randint(0, len(data), (128,))
        batch1 = data[indices]
        indices2 = torch.randint(0, len(data), (128,))
        batch2 = data[indices2]
        
        # Train step
        result = state.trainer.train_step(batch1, batch2)
        
        # Extract metrics
        audit = result.get('audit', {})
        metrics = audit.get('metrics', {})
        
        action = audit.get('action', 'CONTINUE')
        effective_rank = metrics.get('current_rank', 50.0)
        density = metrics.get('current_density', 0.3)
        drift_score = audit.get('total_score', 0.0)
        
        # Build message
        message = {
            "type": "metrics",
            "data": {
                "dataset": "BRFSS-2015",
                "batch": state.batch,
                "effective_rank": round(effective_rank, 2),
                "density": round(density, 4),
                "drift_score": round(drift_score, 4),
                "action": action,
                "timestamp": datetime.now().isoformat(),
                "is_poisoned": state.is_poisoned,
            }
        }
        
        await broadcast(message)
        
        # Stop if HALTED
        if action == "HALT":
            state.is_running = False
            await broadcast({"type": "event", "data": {
                "severity": "danger",
                "message": "⛔ TRAINING HALTED - Attack Detected!",
                "batch": state.batch,
            }})
            break
        
        # Broadcast events
        if action == "ALERT":
            await broadcast({"type": "event", "data": {
                "severity": "warning",
                "message": f"⚠️ Anomaly detected - Score: {drift_score:.2f}",
                "batch": state.batch,
            }})
        
        # Pace the training
        await asyncio.sleep(0.6)
    
    print("[Server] Training loop stopped.")


@app.get("/api/status")
async def get_status():
    """Health check endpoint."""
    return {
        "status": "ok",
        "training": state.is_running,
        "batch": state.batch,
        "poisoned": state.is_poisoned,
    }


@app.post("/api/training/start")
async def start_training():
    """Start training session."""
    if state.is_running:
        return {"status": "already_running"}
    
    # Reset state
    state.is_running = True
    state.is_poisoned = False
    state.batch = 0
    
    # Reinitialize trainer with fresh weights
    encoder = TabularMLPEncoder(input_dim=state.clean_tensors.shape[1], output_dim=64)
    head = ProjectionHead(input_dim=64, hidden_dim=32, output_dim=32)
    auditor = PipelineAuditor(baseline=state.fingerprint, monitor_window=20)
    state.trainer = PoisonGuardTrainer(encoder, head, auditor=auditor)
    
    # Start training in background
    state.task = asyncio.create_task(training_loop())
    
    await broadcast({"type": "event", "data": {
        "severity": "info",
        "message": "🚀 Training started",
        "batch": 0,
    }})
    
    return {"status": "started"}


@app.post("/api/training/inject")
async def inject_poison():
    """Inject poison into training data."""
    if not state.is_running:
        return {"status": "not_running"}
    
    state.is_poisoned = True
    
    await broadcast({"type": "event", "data": {
        "severity": "warning", 
        "message": "💉 Poison injection triggered!",
        "batch": state.batch,
    }})
    
    return {"status": "injected", "batch": state.batch}


@app.post("/api/training/stop")
async def stop_training():
    """Stop training session."""
    state.is_running = False
    if state.task:
        state.task.cancel()
        try:
            await state.task
        except asyncio.CancelledError:
            pass
    
    await broadcast({"type": "event", "data": {
        "severity": "info",
        "message": "⏹️ Training stopped",
        "batch": state.batch,
    }})
    
    return {"status": "stopped"}


class DiabetesInput(BaseModel):
    """Input model for manual testing, requires either text or specific fields."""
    text: Optional[str] = None
    HighBP: float = 0.0
    HighChol: float = 0.0
    CholCheck: float = 1.0
    BMI: float = 28.0
    Smoker: float = 0.0
    Stroke: float = 0.0
    HeartDiseaseorAttack: float = 0.0
    PhysActivity: float = 1.0
    Fruits: float = 1.0
    Veggies: float = 1.0
    HvyAlcoholConsump: float = 0.0
    AnyHealthcare: float = 1.0
    NoDocbcCost: float = 0.0
    GenHlth: float = 2.0
    MentHlth: float = 3.0
    PhysHlth: float = 0.0
    DiffWalk: float = 0.0
    Sex: float = 0.0
    Age: float = 8.0 # Age bracket 35-39
    Education: float = 5.0
    Income: float = 6.0

def parse_natural_language(text: str, current_data: Dict[str, float]) -> Dict[str, float]:
    """
    Simple heuristic parser for hackathon demo.
    Updates current_data based on keywords in text.
    """
    text = text.lower()
    data = current_data.copy()
    
    # Simple trigger keywords
    if "high bp" in text or "high blood pressure" in text: data['HighBP'] = 1.0
    if "high chol" in text or "high cholesterol" in text: data['HighChol'] = 1.0
    if "smoker" in text or "smoke" in text: data['Smoker'] = 1.0
    if "stroke" in text: data['Stroke'] = 1.0
    if "heart disease" in text or "heart attack" in text: data['HeartDiseaseorAttack'] = 1.0
    if "alcohol" in text and "heavy" in text: data['HvyAlcoholConsump'] = 1.0
    if "walk" in text and "difficult" in text: data['DiffWalk'] = 1.0
    
    # Numeric extraction (naive)
    import re
    
    # BMI extraction
    bmi_match = re.search(r'bmi\s*(?:is|of)?\s*(\d+)', text)
    if bmi_match:
        data['BMI'] = float(bmi_match.group(1))
        
    # Mental Health extraction
    ment_match = re.search(r'mental\s*(?:health)?\s*(?:is|of)?\s*(\d+)', text)
    if ment_match:
        data['MentHlth'] = float(ment_match.group(1))

    # General Health (1-5)
    gen_match = re.search(r'health\s*(?:is|of)?\s*([1-5])', text)
    if gen_match:
        data['GenHlth'] = float(gen_match.group(1))
        
    return data

def clean_audit_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy/torch types to python native for JSON serialization."""
    import numpy as np
    clean = {}
    for k, v in result.items():
        # Handle Recursion for nested dicts
        if isinstance(v, dict):
            clean[k] = clean_audit_result(v)
            continue
            
        # Handle PyTorch
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                clean[k] = v.item()
            else:
                clean[k] = v.tolist()
            continue
            
        # Handle Numpy Arrays
        if isinstance(v, np.ndarray):
            clean[k] = v.tolist()
            continue
            
        # Handle Numpy Scalars (float32, float64, int64, bool_, etc)
        if isinstance(v, (np.generic, np.bool_)):
            clean[k] = v.item()
            continue
            
        # Fallback for standard python types
        clean[k] = v
        
    return clean





def scan_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Hybrid Defense to a full dataframe.
    Returns dataframe with 'anomaly_score' and 'is_anomalous' columns.
    """
    # 1. Transform to Embedding Space
    try:
        # Pre-fill missing cols with defaults if needed (simple robustification)
        # Fix: Adapter does not have feature_names, use num_cols + cat_cols
        required_cols = state.adapter.num_cols + state.adapter.cat_cols
        
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0 # Default fallback
                
        tensor_x = state.adapter.transform(df)
        tensor_x = tensor_x.to(state.trainer.device)
        
        with torch.no_grad():
            state.trainer.encoder.eval()
            h = state.trainer.encoder(tensor_x)
            if state.is_running: state.trainer.encoder.train()
            
            # 2. Geometric Score (Cosine Distance)
            centroid = state.fingerprint.mean_embedding.to(state.trainer.device)
            # Expand centroid to batch size
            centroid_batch = centroid.unsqueeze(0).expand(h.shape[0], -1)
            
            cos_sim = torch.nn.functional.cosine_similarity(h, centroid_batch)
            distances = 1.0 - cos_sim
            
            # Base Score
            scores = distances * 5.0
            
            # 3. Biometric Heuristics (Vectorized)
            # Penalize BMI > 50
            if 'BMI' in df.columns:
                bad_bmi = torch.tensor((df['BMI'] > 50).values, device=state.trainer.device)
                scores = torch.where(bad_bmi, scores * 2.0, scores)
                
            # Penalize MentHlth > 30
            if 'MentHlth' in df.columns:
                bad_ment = torch.tensor((df['MentHlth'] > 30).values, device=state.trainer.device)
                scores = torch.where(bad_ment, scores * 2.0, scores)
                
            # threshold 2.2
            is_anomalous = scores > 2.2
            
            # Attach to DF
            df['drift_score'] = scores.cpu().numpy()
            df['is_anomalous'] = is_anomalous.cpu().numpy()
            
            return df
            
    except Exception as e:
        print(f"Scan failed: {e}")
        # Re-raise so the caller knows it failed, instead of finding missing columns
        raise e


@app.post("/api/dataset/scan")
async def scan_uploaded_file(file: UploadFile = File(...)):
    """
    Scan an uploaded CSV file for poisoning.
    """
    if not state.trainer:
         raise HTTPException(status_code=503, detail="System not initialized.")
         
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        print(f"[Server] Scanning uploaded dataset: {len(df)} rows")
        
        # Run Scan
        scored_df = scan_dataset(df)
        
        # Split
        poison_df = scored_df[scored_df['is_anomalous'] == True]
        safe_df = scored_df[scored_df['is_anomalous'] == False]
        
        # Save temp files
        poison_df.to_csv("scan_poison.csv", index=False)
        safe_df.to_csv("scan_safe.csv", index=False)
        
        return {
            "status": "complete",
            "total_rows": len(df),
            "poison_count": len(poison_df),
            "safe_count": len(safe_df),
            "poison_file": "/api/dataset/download/poison",
            "safe_file": "/api/dataset/download/safe"
        }
        
    except Exception as e:
        print(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dataset/download/{type}")
async def download_scanned(type: str):
    """Download the purified or poisoned dataset."""
    if type == "poison":
        if os.path.exists("scan_poison.csv"):
            return FileResponse("scan_poison.csv", filename="poisoned_data.csv")
    elif type == "safe":
        if os.path.exists("scan_safe.csv"):
            return FileResponse("scan_safe.csv", filename="purified_data.csv")
            
    raise HTTPException(status_code=404, detail="File not found")


    if not state.trainer:
         raise HTTPException(status_code=503, detail="System not initialized. Please wait for startup.")
         
    # 1. Prepare Data
    data_dict = input_data.model_dump(exclude={'text'})
    
    # 2. Apply NLP Overlay if text is present
    if input_data.text:
        print(f"[Server] Parsing NLP input: {input_data.text}")
        data_dict = parse_natural_language(input_data.text, data_dict)
        
    # Add dummy target column as expected by adapter
    data_dict['Diabetes_binary'] = 0.0 
    
    # 3. Create DataFrame
    df = pd.DataFrame([data_dict])
    
    try:
        # Transform
        # We need to use the adapter to transform this single row
        
        # 1. Transform
        tensor_x = state.adapter.transform(df)
        tensor_x = tensor_x.to(state.trainer.device)
        
        # 2. Audit (Custom Single-Sample Logic)
        # The PipelineAuditor is designed for BATCH statistics (Rank, Density).
        # A single sample always has Rank=1, which looks like "Collapse" (Drift).
        # So we use a simple Distance-to-Centroid metric for single samples.
        with torch.no_grad():
            # Ensure model is in eval mode
            state.trainer.encoder.eval()
            
            # Get representation
            h = state.trainer.encoder(tensor_x)
            
            # Restore training mode if needed
            if state.is_running:
                state.trainer.encoder.train()
            
            # Compute Distance to Clean Centroid
            # "Normal" data should be close to the mean embedding of the clean set
            # Use trainer.device to avoid AttributeError
            centroid = state.fingerprint.mean_embedding.to(state.trainer.device)
            
            # Calculate Cosine Distance (0 to 2)
            # 1.0 - Similarity
            cos_sim = torch.nn.functional.cosine_similarity(h, centroid)
            distance = 1.0 - cos_sim.item()
            
            # Scoring Heuristic
            # Clean data usually clusters well (dist < 0.3)
            # Sick Patient (Outlier) dist ≈ 0.41 (Score 2.07)
            # Poison (BMI 99) dist ≈ 0.32 (Score 1.6) -> "Stealthy"
            
            score = distance * 5.0 
            
            # --- HYBRID DEFENSE ---
            # Pure distance check is struggling to separate "Sick" from "Stealthy Poison".
            # We add a Domain Knowledge check: Unrealistic biological values should boost score.
            # BMI > 60 is extremely rare. BMI 99 is virtually impossible.
            # Mental Health > 30 is impossible (scale is 0-30).
            
            # Check for extreme values in the original input logic, 
            # but since we only have 'tensor_x' here, we rely on the embedding.
            # Actually, we passed 'data_dict' earlier but didn't pass it to this scope.
            
            # However, we can trust that if the embedding is stealthy, we rely on the heuristic.
            # Let's adjust the Sensitivity strictly.
            # If we raise threshold to 2.2:
            # - Sick (2.07) -> SAFE
            # - Poison (1.6) -> SAFE (False Negative)
            
            # Since we can't easily access the raw features here without refactoring,
            # We will use the 'data_dict' from the outer scope!
            
            if data_dict.get('BMI', 0) > 50 or data_dict.get('MentHlth', 0) > 30:
                print(f"[Server] Domain Heuristic Triggered: Extreme Values Detected")
                score *= 2.0 # Boost anomaly score for biologically impossible values
            
            print(f"[Server] Single-Sample Debug - Cosine Dist: {distance:.4f}, Final Score: {score:.4f}")

            # Threshold of 2.2 clears "Sick" patients but catches "Poison" (1.6 * 2 = 3.2)
            is_anomalous = score > 2.2
            
            audit_result = {
                "action": "ALERT" if is_anomalous else "CONTINUE",
                "total_score": score,
                "metrics": {
                    "distance": distance,
                    "similarity": cos_sim.item()
                }
            }
            
        # 3. Serialize Safely
        # IMPORTANT: Convert any numpy/torch float32 to python float for JSON
        clean_result = clean_audit_result(audit_result)
            
        return {
            "status": "ok",
            "result": clean_result,
            "parsed_data": data_dict # Return what we inferred so frontend can show it
        }
        
    except Exception as e:
        print(f"Check failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



@app.websocket("/ws/metrics")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time metrics streaming."""
    await websocket.accept()
    state.connections.append(websocket)
    print(f"[Server] Client connected. Total: {len(state.connections)}")
    
    # Send current state
    await websocket.send_text(json.dumps({
        "type": "connected",
        "data": {
            "training": state.is_running,
            "batch": state.batch,
            "poisoned": state.is_poisoned,
        }
    }))
    
    try:
        while True:
            # Keep connection alive, handle any incoming messages
            data = await websocket.receive_text()
            msg = json.loads(data)
            
            # Handle control messages from frontend
            if msg.get("action") == "start":
                await start_training()
            elif msg.get("action") == "stop":
                await stop_training()
            elif msg.get("action") == "inject":
                await inject_poison()
                
    except WebSocketDisconnect:
        state.connections.remove(websocket)
        print(f"[Server] Client disconnected. Total: {len(state.connections)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
