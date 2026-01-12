import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import uvicorn
import asyncio
import json
import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, HTTPException, Depends, status
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Form
from pydantic import BaseModel
import io
from pydantic import BaseModel
from typing import List, Dict
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import model components
from poison_guard.models.encoders.tabular_mlp import TabularMLPEncoder
from poison_guard.models.heads.mlp import ProjectionHead
from poison_guard.db import connect_to_mongo, close_mongo_connection, get_database
from poison_guard.auth import get_password_hash, verify_password, create_access_token, get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES
from datetime import timedelta

app = FastAPI(title="Poison Guard API", version="0.1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_db_client():
    await connect_to_mongo()

@app.on_event("shutdown")
async def shutdown_db_client():
    await close_mongo_connection()

# --- Configuration & Model Loading ---
MODEL_PATH = "trained_model_v2.pt"
DATA_PATH = "data/diabetes_brfss.csv"

# These must match train_model.py
INPUT_DIM = 21 
HIDDEN_DIM = 256
OUTPUT_DIM = 64

encoder = TabularMLPEncoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)
head = ProjectionHead(input_dim=OUTPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)

# Load real data for streaming
print("Loading dataset for monitoring stream...")
try:
    df = pd.read_csv(DATA_PATH)
    if 'Diabetes_binary' in df.columns:
        X_test = df.drop('Diabetes_binary', axis=1).values
    else:
        X_test = df.values
    
    # Scale it (using a fitted scaler ideally, but here fitting on test set for demo purposes or strictly we should load the scaler)
    # For demo, just fit_transform is okay but technically leakage.
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    print(f"Dataset loaded: {X_test_tensor.shape}")
except Exception as e:
    print(f"Error loading data: {e}")
    X_test_tensor = torch.randn(1000, INPUT_DIM)

# Load Checkpoint
try:
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        if 'encoder' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder'])
            print("Encoder loaded successfully.")
        else:
            print("Invalid checkpoint format.")
    else:
        print(f"Model file {MODEL_PATH} not found. Waiting for training to finish...")
except Exception as e:
    print(f"Error loading model: {e}")

encoder.eval()

def calculate_effective_rank(embeddings: torch.Tensor) -> float:
    if embeddings.size(0) < 2:
        return 1.0
    embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
    try:
        _, S, _ = torch.svd(embeddings)
        p = S / S.sum()
        p = p[p > 0]
        entropy = -torch.sum(p * torch.log(p))
        erank = torch.exp(entropy).item()
        return erank
    except:
        return 1.0

# --- Data Models ---
class Metric(BaseModel):
    label: str
    value: str
    change: str
    trend: str

class RankPoint(BaseModel):
    name: str
    rank: int

class TrainingStatus(BaseModel):
    status: str
    training: bool
    batch: int
    poisoned: bool
    
class CheckRequest(BaseModel):
    text: str


# --- Auth Models ---
class UserCreate(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

# --- Global State ---
# "training" flag here now means "monitoring active"
state = {
    "status": "IDLE",
    "training": False,
    "batch": 0,
    "poisoned": False,
    "strict_mode": True,  # If True, HALT immediately on attack. If False, just alert.
    "halted": False,       # Track if system has halted due to strict mode
    "sensitivity": 1.0,    # Detection Sensitivity (0.5=Low, 1.0=Medium, 1.5=High)
    "speed": 1.0           # Simulation Speed (delay in seconds)
}

# --- Streaming Data Storage (for user-uploaded files) ---
streaming_data = {
    "tensor": None,        # User uploaded data as tensor
    "filename": None,      # Original filename
    "total_rows": 0        # Number of rows in uploaded file
}

# --- WebSocket Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# --- Global storage for monitoring results ---
monitoring_results = {
    "scan_id": None,
    "clean_indices": [],
    "poison_indices": [],
    "clean_count": 0,
    "poison_count": 0,
    "complete": False
}

# --- Monitoring/Prediction Loop ---
async def monitoring_loop():
    global state, streaming_data, monitoring_results
    batch_size = 10  # Process 10 rows at a time
    
    # Reset batch counter and results at start
    state["batch"] = 0
    monitoring_results = {
        "scan_id": str(uuid.uuid4())[:8],
        "clean_indices": [],
        "poison_indices": [],
        "clean_count": 0,
        "poison_count": 0,
        "complete": False
    }
    
    # Use uploaded data if available, otherwise use pre-loaded data
    if streaming_data["tensor"] is not None:
        data_tensor = streaming_data["tensor"]
        data_source = streaming_data["filename"]
        print(f"[STREAM] Using uploaded file: {data_source}", flush=True)
    else:
        data_tensor = X_test_tensor
        data_source = "diabetes_brfss.csv (default)"
        print(f"[STREAM] Using default data: {data_source}", flush=True)
    
    total_samples = len(data_tensor)
    total_batches = (total_samples + batch_size - 1) // batch_size  # Ceiling division
    idx = 0
    
    # Track batch results
    poisoned_batches = []
    clean_batches = []
    
    # First pass: calculate baseline statistics from first batch for calibration
    with torch.no_grad():
        sample_embeddings = encoder(data_tensor[:min(50, total_samples)])
        sample_norms = torch.norm(sample_embeddings, dim=1)
        baseline_mean = float(sample_norms.mean().item())
        baseline_std = float(sample_norms.std().item())
        # Threshold: mean + 2*std (captures outliers)
        dynamic_threshold = baseline_mean + 2.0 * max(baseline_std, 1.0)
    
    print(f"[STREAM] Total: {total_samples} rows = {total_batches} batches", flush=True)
    print(f"[STREAM] Detection threshold: {dynamic_threshold:.2f} (mean={baseline_mean:.2f}, std={baseline_std:.2f})", flush=True)
    
    while state["training"] and idx < total_samples:
        state["batch"] += 1
        batch_start_idx = idx
        
        # Get Batch of 10 rows from Data
        next_idx = min(idx + batch_size, total_samples)
        batch_data = data_tensor[idx:next_idx]
        batch_rows = len(batch_data)
        idx = next_idx
        
        # Run Prediction (Forward pass) - detect poison naturally
        with torch.no_grad():
            embeddings = encoder(batch_data)
            rank = calculate_effective_rank(embeddings)
            density = float(embeddings.std().item())
            
            # Calculate embedding norms to detect anomalies per row
            norms = torch.norm(embeddings, dim=1)
            
            # Per-row classification
            row_is_poison = (norms > dynamic_threshold).tolist()
            
        # Track individual row results
        poison_count_in_batch = 0
        for i, is_poison in enumerate(row_is_poison):
            global_idx = batch_start_idx + i
            if is_poison:
                monitoring_results["poison_indices"].append(global_idx)
                monitoring_results["poison_count"] += 1
                poison_count_in_batch += 1
            else:
                monitoring_results["clean_indices"].append(global_idx)
                monitoring_results["clean_count"] += 1
        
        # Batch classification
        is_batch_poisoned = poison_count_in_batch > 0
        
        # Calculate drift score based on actual detection
        if poison_count_in_batch > batch_rows // 2:
            drift = 0.9
            poisoned_batches.append(state["batch"])
        elif poison_count_in_batch > 0:
            drift = 0.6
            poisoned_batches.append(state["batch"])
        else:
            drift = 0.1
            clean_batches.append(state["batch"])
            
        metric_data = {
            "dataset": data_source,
            "batch": state["batch"],
            "total_batches": total_batches,
            "effective_rank": rank,
            "density": density,
            "drift_score": drift,
            "action": "POISON" if is_batch_poisoned else "CLEAN",
            "timestamp": "now",
            "is_poisoned": is_batch_poisoned,
            "poison_count": poison_count_in_batch,
            "batch_size": batch_rows,
            "total_poison": monitoring_results["poison_count"],
            "total_clean": monitoring_results["clean_count"]
        }
        
        await manager.broadcast({
            "type": "metrics",
            "data": metric_data
        })
        
        # Emit event for each batch result
        if is_batch_poisoned:
            await manager.broadcast({
                "type": "event",
                "data": {
                    "severity": "danger",
                    "message": f"🚨 Batch {state['batch']}/{total_batches}: {poison_count_in_batch} POISONED rows found",
                    "batch": state['batch']
                }
            })
        else:
            await manager.broadcast({
                "type": "event",
                "data": {
                    "severity": "info",
                    "message": f"✅ Batch {state['batch']}/{total_batches}: ALL CLEAN",
                    "batch": state['batch']
                }
            })
        
        await asyncio.sleep(state["speed"])
    
    # Monitoring complete - save separated data
    monitoring_results["complete"] = True
    state["training"] = False
    state["status"] = "COMPLETE"
    
    # Save clean and poison data to files
    scan_id = monitoring_results["scan_id"]
    clean_indices = monitoring_results["clean_indices"]
    poison_indices = monitoring_results["poison_indices"]
    
    # Get original dataframe from streaming data
    if streaming_data["tensor"] is not None:
        # Reconstruct indices to save
        os.makedirs("downloads", exist_ok=True)
        
        # Save as numpy arrays (indices)
        np.save(os.path.join("downloads", f"{scan_id}_clean_indices.npy"), np.array(clean_indices))
        np.save(os.path.join("downloads", f"{scan_id}_poison_indices.npy"), np.array(poison_indices))
        
        print(f"[STREAM] Saved indices - Clean: {len(clean_indices)}, Poison: {len(poison_indices)}", flush=True)
    
    # Broadcast completion with download info
    await manager.broadcast({
        "type": "complete",
        "data": {
            "scan_id": scan_id,
            "clean_count": len(clean_indices),
            "poison_count": len(poison_indices),
            "message": f"📊 COMPLETE: {len(clean_indices)} clean, {len(poison_indices)} poisoned rows"
        }
    })
    
    await manager.broadcast({
        "type": "event",
        "data": {
            "severity": "warning",
            "message": f"📊 COMPLETE: {len(clean_indices)} clean, {len(poison_indices)} poisoned rows",
            "batch": state['batch']
        }
    })
    
    print(f"[STREAM] Complete - Clean: {len(clean_indices)}, Poisoned: {len(poison_indices)}", flush=True)

# --- API Endpoints ---
@app.websocket("/ws/metrics")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if "action" in message:
                    if message["action"] == "start":
                        if not state["training"]:
                            state["training"] = True
                            state["status"] = "MONITORING"
                            # Start the background task
                            asyncio.create_task(monitoring_loop())
                    elif message["action"] == "stop":
                        state["training"] = False
                        state["status"] = "IDLE"
                    elif message["action"] == "inject":
                        state["poisoned"] = not state["poisoned"]
            except Exception as e:
                print(f"Error processing WS message: {e}")
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/auth/register", response_model=Token)
async def register(user: UserCreate):
    db = get_database()
    # Check if user exists
    existing_user = await db.users.find_one({"username": user.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    user_dict = {"username": user.username, "hashed_password": hashed_password}
    await db.users.insert_one(user_dict)
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/auth/login", response_model=Token)
async def login(user: UserCreate):
    db = get_database()
    db_user = await db.users.find_one({"username": user.username})
    if not db_user or not verify_password(user.password, db_user["hashed_password"]):
         raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/users/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return current_user
@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "poison-guard-backend"}

@app.get("/api/downloads/{filename}")
def download_file(filename: str):
    """Serve CSV files from downloads directory"""
    file_path = os.path.join("downloads", filename)
    if os.path.exists(file_path):
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="text/csv"
        )
    return {"error": "File not found"}

# --- Settings Endpoints ---
@app.get("/api/settings")
def get_settings():
    """Get current system settings"""
    return {
        "strict_mode": state["strict_mode"],
        "halted": state["halted"],
        "sensitivity": state["sensitivity"],
        "speed": state["speed"],
        "status": state["status"]
    }

@app.post("/api/settings/speed")
def set_speed(value: float):
    """Set simulation delay in seconds (0.1 to 5.0)"""
    state["speed"] = max(0.05, min(5.0, value))
    print(f"[SETTINGS] Speed set to: {state['speed']}s delay", flush=True)
    return {"speed": state["speed"]}

@app.post("/api/settings/sensitivity")
def set_sensitivity(value: float):
    """Set detection sensitivity (0.1 to 3.0)"""
    state["sensitivity"] = max(0.1, min(3.0, value))
    print(f"[SETTINGS] Sensitivity set to: {state['sensitivity']}", flush=True)
    return {"sensitivity": state["sensitivity"]}

@app.post("/api/settings/strict-mode")
def set_strict_mode(enabled: bool = True):
    """Toggle strict mode on/off"""
    state["strict_mode"] = enabled
    print(f"[SETTINGS] Strict Mode set to: {enabled}", flush=True)
    return {"strict_mode": state["strict_mode"]}

@app.post("/api/settings/reset-halt")
def reset_halt():
    """Reset system from halted state"""
    state["halted"] = False
    state["status"] = "IDLE"
    print("[SETTINGS] System HALT reset", flush=True)
    return {"halted": False, "status": "IDLE"}

# --- Training Control Endpoints ---
@app.post("/api/training/start")
async def start_training():
    if not state["training"]:
        state["training"] = True
        state["status"] = "MONITORING"
        asyncio.create_task(monitoring_loop())
    return {"status": "started", "state": state}

@app.post("/api/stream/upload")
async def upload_streaming_data(file: UploadFile = File(...)):
    print(f"[STREAM] Receiving file for streaming: {file.filename}", flush=True)
    try:
        # Load and process data (using consistent logic with scan_dataset)
        df = await load_dataset_multi_format(file)
        
        if 'Diabetes_binary' in df.columns:
            X = df.drop('Diabetes_binary', axis=1).values
        else:
            X = df.values
            
        # Ensure input dim
        if X.shape[1] != INPUT_DIM:
            if X.shape[1] > INPUT_DIM:
                X = X[:, :INPUT_DIM]
            else:
                padding = np.zeros((X.shape[0], INPUT_DIM - X.shape[1]))
                X = np.hstack([X, padding])
        
        # Scale
        ref_means = np.array([0.5]*INPUT_DIM)
        ref_means[3] = 28.0 # BMI 
        ref_means[14] = 3.0 # MentHlth
        
        ref_stds = np.array([0.5]*INPUT_DIM)
        ref_stds[3] = 6.0 
        ref_stds[14] = 5.0 
        
        X_scaled = (X - ref_means) / (ref_stds + 1e-6)
        X_tensor = torch.FloatTensor(X_scaled)
        
        # update global state
        streaming_data["tensor"] = X_tensor
        streaming_data["filename"] = file.filename
        streaming_data["total_rows"] = len(X_tensor)
        
        # Reset streaming state
        state["training"] = False
        state["batch"] = 0
        state["status"] = "IDLE"
        
        print(f"[STREAM] Loaded {len(X_tensor)} rows from {file.filename}", flush=True)
        return {
            "status": "loaded", 
            "filename": file.filename,
            "rows": len(X_tensor)
        }
    except Exception as e:
        print(f"[STREAM] Error loading file: {e}", flush=True)
        return {"error": str(e)}

@app.post("/api/training/stop")
async def stop_training():
    state["training"] = False
    state["status"] = "IDLE"
    return {"status": "stopped", "state": state}

@app.get("/api/monitoring/results")
def get_monitoring_results():
    """Get the current monitoring results"""
    return {
        "scan_id": monitoring_results["scan_id"],
        "clean_count": monitoring_results["clean_count"],
        "poison_count": monitoring_results["poison_count"],
        "complete": monitoring_results["complete"],
        "total": monitoring_results["clean_count"] + monitoring_results["poison_count"]
    }

@app.get("/api/monitoring/download")
def download_separated_data(type: str, format: str = "csv"):
    """Download separated clean or poison data from monitoring"""
    global streaming_data, monitoring_results
    
    if not monitoring_results["complete"]:
        raise HTTPException(status_code=400, detail="Monitoring not complete yet")
    
    if type not in ['clean', 'poison']:
        raise HTTPException(status_code=400, detail="Type must be 'clean' or 'poison'")
    
    # Get the indices
    indices = monitoring_results["clean_indices"] if type == "clean" else monitoring_results["poison_indices"]
    
    if len(indices) == 0:
        raise HTTPException(status_code=404, detail=f"No {type} data found")
    
    # Get original tensor and convert to dataframe
    if streaming_data["tensor"] is None:
        raise HTTPException(status_code=400, detail="No uploaded data available")
    
    # Create dataframe from tensor
    tensor = streaming_data["tensor"]
    data_array = tensor.numpy()
    
    # Select only the rows with matching indices
    selected_data = data_array[indices]
    
    # Create DataFrame
    columns = [f"feature_{i}" for i in range(selected_data.shape[1])]
    df = pd.DataFrame(selected_data, columns=columns)
    
    # Add status column
    df['_status'] = type.upper()
    
    # Save in requested format
    os.makedirs("downloads", exist_ok=True)
    scan_id = monitoring_results["scan_id"]
    
    format = format.lower()
    if format not in ["csv", "json", "xlsx", "parquet"]:
        format = "csv"
    
    filename = f"logicloopers_{type}_{scan_id}.{format}"
    file_path = os.path.join("downloads", filename)
    
    if format == "csv":
        df.to_csv(file_path, index=False)
    elif format == "json":
        df.to_json(file_path, orient="records", indent=2)
    elif format == "xlsx":
        df.to_excel(file_path, index=False)
    elif format == "parquet":
        df.to_parquet(file_path, index=False)
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )

@app.post("/api/training/inject")
async def inject_poison():
    state["poisoned"] = not state["poisoned"]
    return {"status": "toggled", "poisoned": state["poisoned"]}



# --- Data Handler Helper Functions ---
async def load_dataset_multi_format(file: UploadFile) -> pd.DataFrame:
    """Load dataset from various formats"""
    filename = file.filename.lower()
    contents = await file.read()
    buffer = io.BytesIO(contents)
    
    try:
        if filename.endswith('.csv'):
            return pd.read_csv(buffer)
        elif filename.endswith('.json'):
            return pd.read_json(buffer)
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            return pd.read_excel(buffer)
        elif filename.endswith('.parquet'):
            return pd.read_parquet(buffer)
        elif filename.endswith('.txt'):
            # Try parsing as CSV first
            try:
                buffer.seek(0)
                return pd.read_csv(buffer)
            except:
                raise ValueError("Text file could not be parsed as CSV.")
        else:
             # Default try CSV
             buffer.seek(0)
             return pd.read_csv(buffer)
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        raise ValueError(f"Failed to parse file: {str(e)}")

def save_dataframe_multi_format(df: pd.DataFrame, base_name: str, format: str = "csv") -> tuple[str, str]:
    """Save dataframe in requested format. Returns (full_path, relative_url)"""
    os.makedirs("downloads", exist_ok=True)
    
    # Normalize format
    format = format.lower()
    if format not in ["csv", "json", "excel", "xlsx", "parquet"]:
        format = "csv"
        
    if format == "excel": format = "xlsx"
        
    filename = f"{base_name}.{format}"
    file_path = os.path.join("downloads", filename)
    
    if format == "csv":
        df.to_csv(file_path, index=False)
    elif format == "json":
        df.to_json(file_path, orient="records", indent=2)
    elif format == "xlsx":
        df.to_excel(file_path, index=False)
    elif format == "parquet":
        df.to_parquet(file_path, index=False)
        
    return file_path, f"/api/downloads/{filename}"

import uuid
import shutil

# Temp storage
TEMP_SCAN_DIR = os.path.join(os.getcwd(), "temp_scans")
os.makedirs(TEMP_SCAN_DIR, exist_ok=True)

@app.post("/api/dataset/scan")
async def scan_dataset(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    print(f"!!! RECEIVING REQUEST: {file.filename} from {current_user['username']} !!!", flush=True)
    try:
        # Load data using multi-format handler from feat/multi-format-io
        df = await load_dataset_multi_format(file)
        
        # Preprocess logic (existing logic preserved)
        if 'Diabetes_binary' in df.columns:
            X = df.drop('Diabetes_binary', axis=1).values
        else:
            X = df.values
            
        # Ensure input dim
        if X.shape[1] != INPUT_DIM:
            if X.shape[1] > INPUT_DIM:
                X = X[:, :INPUT_DIM]
            else:
                padding = np.zeros((X.shape[0], INPUT_DIM - X.shape[1]))
                X = np.hstack([X, padding])
        
        # Scale using fixed reference stats (from HEAD)
        ref_means = np.array([0.5]*INPUT_DIM)
        ref_means[3] = 28.0 # BMI 
        ref_means[14] = 3.0 # MentHlth
        
        ref_stds = np.array([0.5]*INPUT_DIM)
        ref_stds[3] = 6.0 
        ref_stds[14] = 5.0 
        
        X_scaled = (X - ref_means) / (ref_stds + 1e-6)
        X_tensor = torch.FloatTensor(X_scaled)
        
        # Run Model
        poison_count = 0
        safe_count = 0
        
        # Batch inference
        batch_size = 256
        dataset_status = [] # List of strings: "SAFE" or "POISON"
        
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                embeddings = encoder(batch)
                norms = torch.norm(embeddings, dim=1)
                
                # Check anomaly again (Using logic from feat/multi-format-io which seems updated/consistent)
                threshold = 150.0 
                is_poison_batch = (norms > threshold).bool().tolist()
                
                for is_p in is_poison_batch:
                    if is_p:
                        poison_count += 1
                        dataset_status.append("POISON")
                    else:
                        safe_count += 1
                        dataset_status.append("SAFE")

        # Create poison and safe DataFrames
        df['_status'] = dataset_status[:len(df)]
        
        poison_df = df[df['_status'] == 'POISON'].drop(columns=['_status'])
        safe_df = df[df['_status'] == 'SAFE'].drop(columns=['_status'])
        
        # Save to files (Local Storage for download) - Preserving structure from HEAD for frontend compatibility
        import uuid
        scan_id = str(uuid.uuid4())[:8]
        poison_filename = f"poison_{scan_id}.csv"
        safe_filename = f"safe_{scan_id}.csv"
        
        os.makedirs("downloads", exist_ok=True)
        poison_path = os.path.join("downloads", poison_filename)
        safe_path = os.path.join("downloads", safe_filename)
        
        poison_df.to_csv(poison_path, index=False)
        safe_df.to_csv(safe_path, index=False)
        
        # Also save Parquet backups for the new export feature (from feat/multi-format-io)
        os.makedirs(TEMP_SCAN_DIR, exist_ok=True)
        safe_df.to_parquet(os.path.join(TEMP_SCAN_DIR, f"{scan_id}_safe.parquet"), index=False)
        poison_df.to_parquet(os.path.join(TEMP_SCAN_DIR, f"{scan_id}_poison.parquet"), index=False)

        # --- MongoDB Storage --- (From HEAD)
        scan_record = {
            "scan_id": scan_id,
            "filename": file.filename,
            "uploaded_by": current_user['username'],
            "timestamp": datetime.utcnow(),
            "total_rows": len(df),
            "poison_count": poison_count,
            "safe_count": safe_count,
            "poison_file_path": poison_path,
            "safe_file_path": safe_path
        }
        
        db = get_database()
        await db.scans.insert_one(scan_record)
        print(f"DEBUG: Saved scan record {scan_id} to MongoDB", flush=True)

        return {
            "scan_id": scan_id,
            "total_rows": len(df),
            "poison_count": poison_count,
            "safe_count": safe_count,
            "poison_file": f"/api/downloads/{poison_filename}",
            "safe_file": f"/api/downloads/{safe_filename}",
            "message": "Scan completed and logged to database."
        }
    except Exception as e:
        print(f"Error scanning dataset: {e}")
        return {"error": str(e)}

@app.get("/api/dataset/export")
async def export_dataset(scan_id: str, type: str, format: str):
    """
    Export specific dataset (safe/poison) in requested format.
    """
    if type not in ['safe', 'poison']:
        raise HTTPException(status_code=400, detail="Invalid type. Use 'safe' or 'poison'.")
        
    source_path = os.path.join(TEMP_SCAN_DIR, f"{scan_id}_{type}.parquet")
    
    if not os.path.exists(source_path):
        raise HTTPException(status_code=404, detail="Scan data not found or expired.")
        
    try:
        # Load
        df = pd.read_parquet(source_path)
        
        # Use helper to save to format
        base_name = f"logicloopers_{type}_{scan_id[:8]}"
        
        # Note: save_dataframe_multi_format puts things in 'downloads'
        file_path, url = save_dataframe_multi_format(df, base_name, format)
        
        filename = os.path.basename(file_path)
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/octet-stream' 
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.get("/api/scans")
async def get_scans(current_user: dict = Depends(get_current_user)):
    db = get_database()
    # Fetch scans for the user, sorted by timestamp desc
    cursor = db.scans.find({"uploaded_by": current_user['username']}).sort("timestamp", -1).limit(50)
    scans = await cursor.to_list(length=50)
    
    # Convert ObjectId to string and datetime to isoformat if needed
    results = []
    for scan in scans:
        scan["_id"] = str(scan["_id"])
        results.append(scan)
        
    return results

@app.post("/api/check")
def check_sample(req: CheckRequest):
    """
    Analyze patient description using LLM-based parsing.
    No hardcoded keywords - uses AI to understand natural language.
    """
    from llm_parser import get_parser
    
    data = req.dict()
    text = data.get("text", "")
    
    print(f"[check_sample] Received: {text[:100]}...", flush=True)
    
    # Use LLM parser for intelligent extraction
    parser = get_parser()
    llm_result = parser.parse(text)
    
    parsed = llm_result["parsed_data"]
    contradictions = llm_result.get("contradictions", [])
    anomalies = llm_result.get("anomalies", [])
    llm_risk = llm_result.get("risk_assessment", "SAFE")
    llm_used = llm_result.get("llm_used", False)
    
    print(f"[check_sample] LLM Used: {llm_used}, Risk: {llm_risk}", flush=True)
    print(f"[check_sample] Contradictions: {contradictions}", flush=True)
    print(f"[check_sample] Anomalies: {anomalies}", flush=True)
    print(f"[check_sample] RAW PARSED BMI: {parsed.get('BMI')}", flush=True)
    
    # Build input vector for model inference (21 features)
    input_vec = np.array([[
        parsed["HighBP"], parsed["HighChol"], parsed.get("CholCheck", 1), parsed["BMI"],
        parsed["Smoker"], parsed["Stroke"], parsed["HeartDisease"], parsed.get("PhysActivity", 1),
        parsed.get("Fruits", 1), parsed.get("Veggies", 1), parsed["HvyAlcohol"], parsed.get("Healthcare", 1),
        parsed.get("NoDocCost", 0), parsed["GenHlth"], parsed["MentHlth"], parsed["PhysHlth"],
        parsed["DiffWalk"], parsed.get("Sex", 0), parsed["Age"], parsed.get("Education", 4), parsed.get("Income", 5)
    ]], dtype=np.float32)
    
    # Apply scaling
    ref_means = np.array([0.5]*INPUT_DIM, dtype=np.float32)
    ref_means[3] = 28.0   # BMI
    ref_means[14] = 3.0   # MentHlth
    ref_means[15] = 3.0   # PhysHlth
    
    ref_stds = np.array([0.5]*INPUT_DIM, dtype=np.float32)
    ref_stds[3] = 6.0
    ref_stds[14] = 5.0
    ref_stds[15] = 5.0
    
    input_scaled = (input_vec - ref_means) / (ref_stds + 1e-6)
    
    # Run Model for embedding-based anomaly detection
    input_tensor = torch.FloatTensor(input_scaled)
    with torch.no_grad():
        emb = encoder(input_tensor)
        score = torch.norm(emb).item()
    
    print(f"[check_sample] Model Score: {score:.4f}", flush=True)
    
    # Combine LLM analysis with model score
    has_contradictions = len(contradictions) > 0
    has_anomalies = len(anomalies) > 0
    
    # Threshold Tuning:
    # Normal data yields norms ~0.5 - 1.5
    # Outliers (like BMI 1000) yielded ~10.7
    # Setting threshold to 3.0 to catch these without hardcoding features
    model_detected = score > 3.0
    
    llm_detected = llm_risk in ["SUSPICIOUS", "DANGEROUS"]
    
    is_poison = model_detected or has_anomalies or has_contradictions or llm_detected
    
    # Generate risk factors from LLM results
    risk_factors = []
    risk_factors.extend(anomalies)
    risk_factors.extend(contradictions)
    
    # Add detected conditions
    if parsed["BMI"] > 30 and parsed["BMI"] <= 80:
        risk_factors.append(f"High BMI ({parsed['BMI']:.1f})")
    if parsed["HighBP"]:
        risk_factors.append("High Blood Pressure")
    if parsed["HighChol"]:
        risk_factors.append("High Cholesterol")
    if parsed["Smoker"]:
        risk_factors.append("Smoker")
    if parsed["Stroke"]:
        risk_factors.append("Stroke History")
    if parsed["HeartDisease"]:
        risk_factors.append("Heart Disease")
    
    # Generate verdict
    if has_anomalies:
        verdict = f"🚨 DATA INTEGRITY FAILURE - {len(anomalies)} physiological impossibilities detected!"
    elif has_contradictions:
        verdict = f"🚨 LOGICAL CONTRADICTION - {len(contradictions)} inconsistencies found!"
    elif model_detected:
        verdict = "🚨 ANOMALOUS DATA DETECTED - Model flagged unusual patterns!"
    elif llm_detected:
        verdict = f"⚠️ AI ANALYSIS: {llm_risk} - Suspicious patterns detected."
    else:
        verdict = "✅ Data point appears normal and within expected ranges."
    
    return {
        "result": {
            "action": "HALT" if is_poison else "SAFE",
            "total_score": score,
            "is_anomalous": is_poison,
            "risk_factors": risk_factors,
            "verdict": verdict,
            "detection_method": {
                "llm_used": llm_used,
                "llm_risk": llm_risk,
                "model_score": score,
                "has_contradictions": has_contradictions,
                "has_anomalies": has_anomalies
            }
        },
        "parsed_data": parsed,
        "raw_text": text
    }

# Ensure DEFAULT_CLEAN is defined or dummy
DEFAULT_CLEAN = {k:0.0 for k in range(INPUT_DIM)} 

@app.get("/api/metrics")
def get_metrics() -> List[Metric]:
    return [
        Metric(label="Total Scans", value=str(state['batch'] * 64), change="+running", trend="up"),
        Metric(label="Protection Level", value="High", change="Active", trend="neutral"),
    ]


@app.get("/api/effective-rank")
def get_effective_rank() -> List[RankPoint]:
    return [{"name": f"Batch {i}", "rank": 400} for i in range(10)]

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
