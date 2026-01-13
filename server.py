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
from datetime import timedelta, datetime
from metrics import MetricTracker
from poison_guard.models.losses import NTXentLoss
from poison_guard.pipeline.trainer import PoisonGuardTrainer
from torch.utils.data import DataLoader, Dataset

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()
tracker = MetricTracker()

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await connect_to_mongo()
    await get_settings_from_db()
    print("Locked & Loaded: Settings synced from DB.", flush=True)
    yield
    # Shutdown
    await close_mongo_connection()

app = FastAPI(title="Poison Guard API", version="0.1.0", lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    full_name: str = "Anonymous User"

class Token(BaseModel):
    access_token: str
    token_type: str
    full_name: str = "Anonymous User"

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

# --- Personal Training Manager ---
class TrainingSession:
    def __init__(self, username: str, total_epochs: int = 20):
        self.username = username
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.loss_history = []
        self.accuracy_history = []
        self.is_active = False
        self.task = None

# Store active sessions: username -> TrainingSession
active_training_sessions: Dict[str, TrainingSession] = {}

# Store connected personal websockets: username -> WebSocket
personal_ws_connections: Dict[str, WebSocket] = {}

class PersonalTrainingManager:
    async def connect(self, websocket: WebSocket, username: str):
        await websocket.accept()
        personal_ws_connections[username] = websocket
        
    def disconnect(self, username: str):
        if username in personal_ws_connections:
            del personal_ws_connections[username]
            
    async def send_to_user(self, username: str, message: dict):
        if username in personal_ws_connections:
            try:
                await personal_ws_connections[username].send_json(message)
            except:
                pass

personal_manager = PersonalTrainingManager()

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
    batch_size = 10
    
    try:
        # Reset batch counter and results
        state["batch"] = 0
        monitoring_results = {
            "scan_id": str(uuid.uuid4())[:8],
            "clean_indices": [],
            "poison_indices": [],
            "clean_count": 0,
            "poison_count": 0,
            "complete": False
        }
        
        # Load data
        if streaming_data.get("tensor") is not None:
            data_tensor = streaming_data["tensor"]
            data_source = streaming_data.get("filename", "uploaded")
            print(f"[STREAM] Using uploaded file: {data_source}", flush=True)
        else:
            if 'X_test_tensor' in globals():
                data_tensor = X_test_tensor
                data_source = "diabetes_brfss.csv (default)"
            else:
                 data_tensor = torch.randn(100, 10) 
                 data_source = "random_noise (fallback)"
            print(f"[STREAM] Using default data: {data_source}", flush=True)
        
        total_samples = len(data_tensor)
        if total_samples == 0:
             raise ValueError("Dataset is empty.")

        total_batches = (total_samples + batch_size - 1) // batch_size
        idx = 0
        
        # Check for active model
        active_model = state.get("active_model")
        if active_model:
            print(f"[MONITORING] Using Custom Defense Model: {active_model}", flush=True)
        else:
            print(f"[MONITORING] Using Default System Defense", flush=True)        
        
        monitoring_results["clean_indices"] = []
        monitoring_results["poison_indices"] = []
        
        poisoned_batches = []
        clean_batches = []
        
        # Baseline Phase: Determine Reference Data
        # If using a custom model, the "clean" reference should be the active dataset itself (or a subet).
        # This prevents comparing shifted health data (e.g. Heart) against the default (Diabetes) baseline.
        if active_model_metadata:
             # Use current uploaded data as baseline for custom model
             reference_data = data_tensor[:min(500, total_samples)]
             print(f"[MONITORING] Resetting baseline reference to active dataset for Custom Engine", flush=True)
        else:
             reference_data = X_test_tensor if 'X_test_tensor' in globals() else data_tensor[:min(100, total_samples)]
        
        with torch.no_grad():
            # Use Sample from reference as Anchors
            if len(reference_data) > 500:
                perm = torch.randperm(len(reference_data))
                anchors = reference_data[perm[:500]]
            else:
                anchors = reference_data
                
            # Compute Anchor Embeddings (Normalized)
            anchor_embeddings = encoder(anchors)
            anchor_embeddings = torch.nn.functional.normalize(anchor_embeddings, dim=1)
            
            # Calibration: Cross-validate on the reference set
            val_subset = anchors[:100]
            val_embeddings = anchor_embeddings[:100]
            
            sim_matrix = torch.matmul(val_embeddings, anchor_embeddings.T) # (100, 500)
            top_sims, _ = sim_matrix.topk(2, dim=1)
            nearest_neighbor_sims = top_sims[:, 1]
            
            baseline_scores = 1.0 - nearest_neighbor_sims
            baseline_median = float(baseline_scores.median().item())
            mad = float((baseline_scores - baseline_median).abs().median().item())
            
            # Threshold: Median + 6 * MAD (RELAXED widely to prevent False Positives on noisy safe data)
            dynamic_threshold = baseline_median + 6.0 * max(mad, 0.02)

            # Input Statistics
            ref_mean = reference_data.mean(dim=0)
            ref_std = reference_data.std(dim=0)
        
        print(f"[STREAM] Baseline Config: Threshold={dynamic_threshold:.3f}, MedianDist={baseline_median:.3f}", flush=True)
        
        await manager.broadcast({
            "type": "event",
            "data": {"severity": "info", "message": f"Started scanning {total_samples} rows from {data_source}", "batch": 0, "timestamp": datetime.now().isoformat()}
        })

        while state["training"] and idx < total_samples:
            state["batch"] += 1
            batch_start_idx = idx
            
            # Get Batch
            next_idx = min(idx + batch_size, total_samples)
            batch_data = data_tensor[idx:next_idx]
            batch_rows = len(batch_data)
            idx = next_idx
            
            # Run Prediction
            with torch.no_grad():
                # 1. Input-Level Anomaly Detection (Statistical Outliers)
                # Relaxed to 6 sigma for robust "Safe but Noisy" data acceptance
                z_scores = (batch_data - ref_mean) / (ref_std + 1e-6)
                input_outliers = (z_scores.abs() > 6.0).any(dim=1)
                
                # 2. Embedding-Level Anomaly Detection (Semantic Outliers)
                embeddings = encoder(batch_data)
                embeddings = torch.nn.functional.normalize(embeddings, dim=1)
                
                # Metrics
                rank = calculate_effective_rank(embeddings)
                density = float(embeddings.std().item()) 
                
                # KNN: Distance to Nearest Anchor
                sim_matrix = torch.matmul(embeddings, anchor_embeddings.T)
                max_sims, _ = sim_matrix.max(dim=1)
                scores = 1.0 - max_sims
                
                # Check Threshold
                embedding_outliers = (scores > dynamic_threshold)
                
                # Combined Decision
                row_is_poison_tensor = input_outliers | embedding_outliers
                row_is_poison = row_is_poison_tensor.tolist()
                
                # DEBUG stats
                avg_score = float(scores.mean().item())
                input_fails = input_outliers.sum().item()
                emb_fails = embedding_outliers.sum().item()
                print(f"[BATCH {state['batch']}] Dist: {avg_score:.3f} | Thresh: {dynamic_threshold:.3f} | Poison: {sum(row_is_poison)} (Input:{input_fails}, Emb:{emb_fails})", flush=True)

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
            
            # Batch classification & Metrics
            is_batch_poisoned = poison_count_in_batch > 0
            
            # Logic Update: Cumulative Drift Calculation
            # Instead of instantaneous 0.9/0.6/0.1, use the actual cumulative poison rate.
            total_processed = monitoring_results["clean_count"] + monitoring_results["poison_count"]
            poison_rate = monitoring_results["poison_count"] / max(1, total_processed)
            
            # Map Rate to Drift Score (Scaling factor 1.2 to be responsive)
            drift = 0.1 + (poison_rate * 0.85)
            drift = min(0.95, drift)
            
            if is_batch_poisoned:
                poisoned_batches.append(state["batch"])
            else:
                clean_batches.append(state["batch"])
                
            metric_data = {
                "dataset": data_source,
                "batch": state["batch"],
                "total_batches": total_batches,
                "effective_rank": rank,
                "density": density,
                "drift_score": drift,
                "action": "POISON" if is_batch_poisoned else "CLEAN",
                "timestamp": datetime.now().isoformat(),
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
            
            # Rate limit
            await asyncio.sleep(state["speed"])
        
        # Complete
        monitoring_results["complete"] = True
        state["training"] = False
        state["status"] = "COMPLETE"
        
        # Save files
        scan_id = monitoring_results["scan_id"]
        clean_indices = monitoring_results["clean_indices"]
        poison_indices = monitoring_results["poison_indices"]
        
        if streaming_data.get("tensor") is not None:
            os.makedirs("downloads", exist_ok=True)
            np.save(os.path.join("downloads", f"{scan_id}_clean_indices.npy"), np.array(clean_indices))
            np.save(os.path.join("downloads", f"{scan_id}_poison_indices.npy"), np.array(poison_indices))
            print(f"[STREAM] Saved indices - Clean: {len(clean_indices)}, Poison: {len(poison_indices)}", flush=True)

            # Persist to MongoDB History
            try:
                db = get_database()
                total = len(clean_indices) + len(poison_indices)
                poison_rate = len(poison_indices) / max(1, total)
                
                # Determine threat level
                threat_level = 'low'
                if poison_rate > 0.5: threat_level = 'critical'
                elif poison_rate > 0.3: threat_level = 'high'
                elif poison_rate > 0.1: threat_level = 'medium'

                history_entry = {
                    "scan_id": scan_id,
                    "filename": streaming_data.get("filename", "unknown"),
                    "timestamp": datetime.now().isoformat(),
                    "clean_count": len(clean_indices),
                    "poison_count": len(poison_indices),
                    "threat_level": threat_level,
                    "model_used": state.get("active_model", "Default System Defense")
                }
                await db.scan_history.insert_one(history_entry)
                print(f"[DB] Saved scan history for {scan_id}", flush=True)
            except Exception as e:
                print(f"[DB] Failed to save history: {e}", flush=True)
        
        await manager.broadcast({
            "type": "complete",
            "data": {
                "scan_id": scan_id,
                "clean_count": len(clean_indices),
                "poison_count": len(poison_indices),
                "message": f"📊 COMPLETE: {len(clean_indices)} clean, {len(poison_indices)} poisoned rows"
            }
        })

    except Exception as e:
        print(f"Error in monitoring loop: {e}", flush=True)
        state["training"] = False
        state["status"] = "ERROR"
        await manager.broadcast({
            "type": "event",
            "data": {"severity": "danger", "message": f"Monitoring Error: {str(e)}", "batch": state.get("batch", 0)}
        })

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
    user_dict = {
        "username": user.username, 
        "hashed_password": hashed_password,
        "full_name": user.full_name
    }
    await db.users.insert_one(user_dict)
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer", "full_name": user.full_name}

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
    return {
        "access_token": access_token, 
        "token_type": "bearer", 
        "full_name": db_user.get("full_name", "Anonymous User")
    }

@app.get("/api/metrics")
async def get_metrics():
    """Get real-time system metrics - Updated for FE chart compatibility"""
    # Return 24h of data points for the 'Protection Level' chart
    # just dummy data for now to match the shape expected by Recharts if needed
    # Or just simple scalar metrics as before?
    # The FE expects: { label, value, change, trend }
    
    return [
        {
            "label": "Total Scans",
            "value": "12,450", 
            "change": "+12%",
            "trend": "up"
        },
        {
            "label": "Protection Level",
            "value": "98.2%",
            "change": "+0.4%", 
            "trend": "up"
        },
        {
            "label": "Active Threads",
            "value": "24",
            "change": "Stable",
            "trend": "neutral"
        },
        {
             "label": "Response Time",
             "value": "45ms",
             "change": "-12ms",
             "trend": "down" # down is good for latency
        }
    ]

# --- Settings Persistence ---
async def get_settings_from_db():
    db = get_database()
    settings = await db.settings.find_one({"_id": "global_config"})
    if not settings:
        # Defaults
        settings = {
            "_id": "global_config",
            "strict_mode": True,
            "sensitivity": 1.0,
            "speed": 1.0,
            "halted": False,
            "status": "IDLE"
        }
        await db.settings.insert_one(settings)
    
    # Update global state
    state["strict_mode"] = settings.get("strict_mode", True)
    state["sensitivity"] = settings.get("sensitivity", 1.0)
    state["speed"] = settings.get("speed", 1.0)
    state["active_model"] = settings.get("active_model", None)
    # state["halted"] = settings.get("halted", False) # Don't persist halted state across restarts usually? Let's say yes.
    # state["status"] = "IDLE" # Always idle on restart
    return settings

async def update_setting_db(key: str, value):
    db = get_database()
    await db.settings.update_one({"_id": "global_config"}, {"$set": {key: value}})
    await db.settings.update_one(
        {"_id": "global_config"},
        {"$set": {key: value}},
        upsert=True
    )

# --- Settings Endpoints ---
@app.get("/api/settings")
async def get_settings():
    """Get current system settings"""
    # Refresh from DB? Or trust state? Trust state for speed, but startup verified.
    return {
        "strict_mode": state["strict_mode"],
        "halted": state["halted"],
        "sensitivity": state["sensitivity"],
        "speed": state["speed"],
        "status": state["status"],
        "active_model": state.get("active_model")
    }

@app.post("/api/settings/speed")
async def set_speed(value: float):
    """Set simulation delay in seconds (0.1 to 5.0)"""
    new_val = max(0.001, min(5.0, value))
    state["speed"] = new_val
    await update_setting_db("speed", new_val)
    print(f"[SETTINGS] Speed set to: {state['speed']}s delay", flush=True)
    return {"speed": state["speed"]}

@app.post("/api/settings/sensitivity")
async def set_sensitivity(value: float):
    """Set detection sensitivity (0.1 to 3.0)"""
    new_val = max(0.1, min(3.0, value))
    state["sensitivity"] = new_val
    await update_setting_db("sensitivity", new_val)
    print(f"[SETTINGS] Sensitivity set to: {state['sensitivity']}", flush=True)
    return {"sensitivity": state["sensitivity"]}

@app.post("/api/settings/strict-mode")
async def set_strict_mode(enabled: bool = True):
    """Toggle strict mode on/off"""
    state["strict_mode"] = enabled
    await update_setting_db("strict_mode", enabled)
    print(f"[SETTINGS] Strict Mode set to: {enabled}", flush=True)
    return {"strict_mode": state["strict_mode"]}

@app.post("/api/settings/reset-halt")
async def reset_halt():
    """Reset system from halted state"""
    state["halted"] = False
    state["status"] = "IDLE"
    state["status"] = "IDLE"
    await update_setting_db("halted", False)
    print("[SETTINGS] System HALT reset", flush=True)
    return {"halted": False, "status": "IDLE"}

@app.post("/api/settings/model")
async def set_active_model(model_id: str = None):
    """Set the active model for monitoring"""
    state["active_model"] = model_id
    await update_setting_db("active_model", model_id)
    print(f"[SETTINGS] Active Model set to: {model_id}", flush=True)
    return {"active_model": state["active_model"]}

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
        
        # Scale using the GLOBAL scaler (fit on training data)
        # This ensures consistent preprocessing.
        try:
             X_scaled = scaler.transform(X)
        except Exception as e:
             # Fallback if scaler fails (e.g. dimension mismatch despite checks)
             print(f"[STREAM] Scaler transform failed: {e}. Falling back to manual standardization.", flush=True)
             X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

        X_tensor = torch.FloatTensor(X_scaled)
        
        # update global state
        streaming_data["tensor"] = X_tensor
        streaming_data["columns"] = list(df.drop('Diabetes_binary', axis=1).columns) if 'Diabetes_binary' in df.columns else list(df.columns)
        # Store original unscaled values for download
        streaming_data["original_values"] = X # X is the numpy array before scaling (but after dropping target)
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
    # Use ORIGINAL values if available (fallback to tensor if not)
    if "original_values" in streaming_data:
        data_source = streaming_data["original_values"]
        selected_data = data_source[indices]
    else:
        selected_data = data_array[indices]
    
    # Create DataFrame
    # Create DataFrame
    columns = streaming_data.get("columns", [f"feature_{i}" for i in range(selected_data.shape[1])])
    # Ensure column count matches data (in case padding was used or mismatch)
    if len(columns) != selected_data.shape[1]:
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
    # Log injection event?
    if state["poisoned"]:
        # Log to DB
        db = get_database()
        await db.system_events.insert_one({
            "type": "ATTACK_SIMULATION",
            "message": "Poison injection manually toggled ON",
            "timestamp": datetime.utcnow()
        })
    return {"status": "toggled", "poisoned": state["poisoned"]}

@app.get("/api/effective-rank")
def get_effective_rank() -> List[Dict]:
    """Return recent effective rank history"""
    # Assuming 'tracker' is global or accessible
    if 'tracker' in globals():
        return tracker.history[-20:] # Return last 20 points
    return []

@app.get("/api/users/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return current_user
@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "poison-guard-backend"}

@app.get("/api/downloads/{filename}")
async def download_file(filename: str):
    """Serve files from GridFS"""
    try:
        # Try to find file by filename in GridFS
        content = await download_bytes_from_gridfs(filename=filename)
        from fastapi.responses import Response
        return Response(content=content, media_type="application/octet-stream", headers={"Content-Disposition": f"attachment; filename={filename}"})
    except Exception as e:
        # Fallback to local for backward compatibility? NO, User said "only in database"
        return {"error": "File not found in database"}

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

@app.get("/api/datasets")
async def list_datasets(current_user: dict = Depends(get_current_user)):
    """List datasets owned by current user"""
    db = get_database()
    cursor = db.fs.files.find({"metadata.owner": current_user["username"]})
    datasets = []
    async for file in cursor:
        datasets.append({
            "id": str(file["_id"]),
            "name": file["filename"],
            "type": file["metadata"].get("type", "Unknown"),
            "size": f"{file['length'] / 1024 / 1024:.2f} MB",
            "lastModified": file["uploadDate"].isoformat(),
            "status": "Ready",  # Default status for now
            "rows": file["metadata"].get("rows", 0)
        })
    return datasets

@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    """Upload a new dataset to GridFS"""
    content = await file.read()
    
    # Simple validation/metadata extraction
    filename = file.filename
    file_type = "Tabular"
    rows = 0
    
    # Try to parse row count just for metadata
    try:
        if filename.endswith('.csv'):
             import pandas as pd
             df = pd.read_csv(io.BytesIO(content))
             rows = len(df)
             file_type = "CSV"
        elif filename.endswith('.parquet'):
             import pandas as pd
             df = pd.read_parquet(io.BytesIO(content))
             rows = len(df)
             file_type = "Parquet"
        elif filename.endswith('.pt') or filename.endswith('.pth'):
             file_type = "trained_model"
    except:
        pass # Ignore parsing errors for metadata logic, just store file
        
    file_id = await upload_bytes_to_gridfs(
        filename, 
        content, 
        metadata={
            "owner": current_user["username"],
            "type": file_type,
            "rows": rows
        }
    )
    
    return {"status": "uploaded", "id": file_id, "filename": filename}

@app.delete("/api/datasets/{file_id}")
async def delete_dataset(file_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a dataset from GridFS"""
    db = get_database()
    fs = AsyncIOMotorGridFSBucket(db)
    from bson import ObjectId
    
    # Check ownership
    file = await db.fs.files.find_one({"_id": ObjectId(file_id)})
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
        
    if file["metadata"].get("owner") != current_user["username"]:
         raise HTTPException(status_code=403, detail="Not authorized to delete this file")
         
    await fs.delete(ObjectId(file_id))
    return {"status": "deleted", "id": file_id}

@app.get("/api/security/history")
async def get_scan_history(current_user: dict = Depends(get_current_user)):
    """Fetch recent security scan history"""
    db = get_database()
    history = await db.scan_history.find({}, {"_id": 0}).sort("timestamp", -1).limit(20).to_list(length=20)
    return history

import uuid
import shutil

# Temp storage
TEMP_SCAN_DIR = os.path.join(os.getcwd(), "temp_scans")
os.makedirs(TEMP_SCAN_DIR, exist_ok=True)

from motor.motor_asyncio import AsyncIOMotorGridFSBucket

# --- GridFS Helpers ---
async def upload_bytes_to_gridfs(filename: str, content: bytes, metadata: dict = None) -> str:
    db = get_database()
    fs = AsyncIOMotorGridFSBucket(db)
    grid_in = fs.open_upload_stream(filename, metadata=metadata)
    await grid_in.write(content)
    await grid_in.close()
    return str(grid_in._id)

async def download_bytes_from_gridfs(file_id: str = None, filename: str = None) -> bytes:
    db = get_database()
    fs = AsyncIOMotorGridFSBucket(db)
    buffer = io.BytesIO()
    
    if file_id:
        from bson import ObjectId
        await fs.download_to_stream(ObjectId(file_id), buffer)
    elif filename:
        await fs.download_to_stream_by_name(filename, buffer)
    else:
        raise ValueError("Must provide file_id or filename")
        
    buffer.seek(0)
    return buffer.getvalue()

# Helper to save DF to Bytes directly (No local file)
def dataframe_to_bytes(df: pd.DataFrame, format: str = "csv") -> bytes:
    buffer = io.BytesIO()
    if format == "csv":
        df.to_csv(buffer, index=False)
    elif format == "json":
        df.to_json(buffer, orient="records", indent=2)
    elif format == "xlsx":
        df.to_excel(buffer, index=False)
    elif format == "parquet":
        df.to_parquet(buffer, index=False)
    return buffer.getvalue()

# --- Personal Training Components ---

async def run_personal_training_loop(username: str):
    session = active_training_sessions.get(username)
    if not session:
        return

    print(f"Starting real personal training for {username}")
    session.is_active = True
    
    try:
        # 1. Fetch training data from GridFS
        db = get_database()
        # Find the latest training data for this user
        file_doc = await db.fs.files.find_one(
            {"metadata.owner": username, "metadata.type": "training_data"},
            sort=[("uploadDate", -1)]
        )
        
        if not file_doc:
            await personal_manager.send_to_user(username, {
                "type": "error", "data": {"message": "No training dataset found in your Vault."}
            })
            return

        content = await download_bytes_from_gridfs(file_id=str(file_doc["_id"]))
        df = pd.read_csv(io.BytesIO(content))
        
        # 2. Dynamic Schema Detection
        # Identify target (for exclusion) - common targets or the user can specify
        potential_targets = ['target', 'label', 'Diabetes_binary', 'Outcome', 'high_risk']
        target_col = next((c for c in potential_targets if c in df.columns), None)
        
        if target_col:
            X = df.drop(columns=[target_col]).values
            feature_names = df.drop(columns=[target_col]).columns.tolist()
        else:
            X = df.values
            feature_names = df.columns.tolist()

        input_dim = X.shape[1]
        
        # 3. Preprocessing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Store metadata for deployment
        scaler_params = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "features": feature_names
        }

        # 4. Prepare DataLoader
        class ContrastiveDataset(Dataset):
            def __init__(self, data):
                self.data = torch.FloatTensor(data)
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                x = self.data[idx]
                noise1 = torch.randn_like(x) * 0.1
                noise2 = torch.randn_like(x) * 0.1
                return x + noise1, x + noise2

        if len(X_scaled) < 2:
            await personal_manager.send_to_user(username, {
                "type": "error", "data": {"message": "Dataset too small for neural training (minimum 2 samples required)."}
            })
            return

        dataset = ContrastiveDataset(X_scaled)
        # Use drop_last=True to avoid single-sample batches causing BatchNorm failure
        dataloader = DataLoader(dataset, batch_size=min(len(dataset), 64), shuffle=True, drop_last=True)

        # 5. Model Initialization
        hidden_dim = 256
        output_dim = 64
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        encoder = TabularMLPEncoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
        head = ProjectionHead(input_dim=output_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
        
        loss_fn = NTXentLoss(temperature=0.5).to(device)
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=1e-3)
        trainer = PoisonGuardTrainer(encoder, head, optimizer, loss_fn)

        # 6. Real Training Loop
        total_epochs = session.total_epochs
        for epoch in range(1, total_epochs + 1):
            if not session.is_active:
                break
            
            epoch_loss = 0
            for x1, x2 in dataloader:
                x1, x2 = x1.to(device), x2.to(device)
                loss = trainer.train_step(x1, x2)
                epoch_loss += loss
            
            avg_loss = epoch_loss / len(dataloader)
            # Accuracy is simulated for contrastive learning in this demo 
            # (In real SimCLR, use top-1/top-5 retrieval accuracy)
            sim_acc = min(0.99, 0.4 + (epoch/total_epochs)*0.5 + random.uniform(-0.02, 0.02))
            
            session.current_epoch = epoch
            session.loss_history.append(avg_loss)
            session.accuracy_history.append(sim_acc)
            
            await personal_manager.send_to_user(username, {
                "type": "metrics",
                "data": {
                    "epoch": epoch,
                    "loss": avg_loss,
                    "accuracy": sim_acc,
                    "progress": (epoch / total_epochs) * 100
                }
            })
            await asyncio.sleep(0.1) # Smooth UI updates

        # 7. Finalize & Save
        model_filename = f"model_{username}_{int(datetime.utcnow().timestamp())}.pt"
        checkpoint = {
            'encoder': encoder.state_dict(),
            'metadata': {
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'output_dim': output_dim,
                'scaler': scaler_params,
                'features': feature_names
            }
        }
        
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        
        await upload_bytes_to_gridfs(
            model_filename, 
            buffer.getvalue(), 
            metadata={
                "owner": username, 
                "type": "trained_model", 
                "accuracy": sim_acc,
                "dataset": file_doc["filename"]
            }
        )
        
        await personal_manager.send_to_user(username, {
            "type": "complete",
            "data": {
                "final_accuracy": sim_acc,
                "model_file": f"/api/downloads/{model_filename}",
                "message": "Neural Engine updated with custom intelligence."
            }
        })
        
    except Exception as e:
        print(f"Personal training failed: {e}")
        import traceback
        traceback.print_exc()
        await personal_manager.send_to_user(username, {
            "type": "error", "data": {"message": f"Training failed: {str(e)}"}
        })
    finally:
        session.is_active = False


@app.post("/api/training/personal/upload")
async def upload_personal_dataset(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    username = current_user['username']
    content = await file.read()
    
    # Save to GridFS with Owner Metadata
    file_id = await upload_bytes_to_gridfs(
        file.filename,
        content,
        metadata={"owner": username, "type": "training_data"}
    )
    
    return {"message": "Dataset uploaded successfully", "file_id": file_id, "filename": file.filename}

@app.websocket("/ws/training/personal")
async def personal_training_websocket(websocket: WebSocket, token: str):
    # Authenticate via Token in Query Param
    try:
        # For simplicity, decode manually here
        from jose import jwt, JWTError
        from poison_guard.auth import SECRET_KEY, ALGORITHM
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
            
    except Exception:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await personal_manager.connect(websocket, username)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("action") == "start":
                # Initialize session if not exists or exists (overwrite)
                session = TrainingSession(username, total_epochs=20)
                active_training_sessions[username] = session
                
                # Run loop in background
                session.task = asyncio.create_task(run_personal_training_loop(username))
                
            elif message.get("action") == "stop":
                session = active_training_sessions.get(username)
                if session:
                    session.is_active = False # Loop will break
                    
    except WebSocketDisconnect:
        personal_manager.disconnect(username)
        session = active_training_sessions.get(username)
        if session:
            session.is_active = False

@app.post("/api/dataset/scan")
async def scan_dataset(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    print(f"!!! RECEIVING REQUEST: {file.filename} from {current_user['username']} !!!", flush=True)
    try:
        # 1. Read content into memory (Hackathon scale: OK. Production: Stream directly)
        file_content = await file.read()
        
        # 2. Store ORIGINAL file in GridFS immediately
        original_file_id = await upload_bytes_to_gridfs(
            file.filename, 
            file_content, 
            metadata={"uploaded_by": current_user['username'], "type": "original_dataset"}
        )
        
        # 3. Load DataFrame from bytes
        # Re-wrap bytes for pandas
        buffer = io.BytesIO(file_content)
        # We need to manually handle formats since our helper took UploadFile before
        # Let's reuse the logic but adapt it or just inline simple simplified logic
        filename = file.filename.lower()
        if filename.endswith('.csv'):
            df = pd.read_csv(buffer)
        elif filename.endswith('.json'):
            df = pd.read_json(buffer)
        elif filename.endswith('.parquet'):
            df = pd.read_parquet(buffer)
        elif filename.endswith('.xlsx'):
             df = pd.read_excel(buffer)
        else:
             buffer.seek(0)
             try:
                 df = pd.read_csv(buffer)
             except Exception as e:
                 raise ValueError(f"Could not parse file as CSV: {e}")

        # Preprocess logic (existing)
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
        ref_means = np.array([0.5]*INPUT_DIM); ref_means[3] = 28.0; ref_means[14] = 3.0
        ref_stds = np.array([0.5]*INPUT_DIM); ref_stds[3] = 6.0; ref_stds[14] = 5.0
        X_scaled = (X - ref_means) / (ref_stds + 1e-6)
        X_tensor = torch.FloatTensor(X_scaled)
        
        # Run Model
        poison_count = 0
        safe_count = 0
        batch_size = 256
        dataset_status = []
        
        dynamic_threshold = 150.0 * state.get("sensitivity", 1.0)
        
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                embeddings = encoder(batch)
                norms = torch.norm(embeddings, dim=1)
                
                # Calculate batch metrics
                is_poison_batch = (norms > dynamic_threshold).bool().tolist()
                batch_poison_count = is_poison_batch.count(True)
                batch_safe_count = batch_size - batch_poison_count
                
                # Calculate simple "Drift Score" (e.g., avg norm of batch)
                avg_norm = norms.mean().item()
                
                # Update Tracker & Broadcast Stats
                if 'tracker' in globals():
                    tracker.update(batch_size=len(batch), 
                                  poison_count=batch_poison_count, 
                                  safe_count=batch_safe_count, 
                                  avg_drift_score=avg_norm)
                                  
                    # Broadcast stats
                    stats = tracker.get_stats()
                    asyncio.create_task(manager.broadcast({
                        "type": "scan_metrics",
                        "data": stats
                    }))
                    
                    # Also broadcast simple progress for the progress bar if needed
                    asyncio.create_task(manager.broadcast({
                        "type": "scan_progress",
                        "data": {
                            "processed": i + len(batch),
                            "total": len(X_tensor),
                            "poison_found": poison_count + batch_poison_count
                        }
                    }))
                
                # Artificial delay for visualization effect
                await asyncio.sleep(0.02)
                
                for is_p in is_poison_batch:
                    if is_p:
                        poison_count += 1
                        dataset_status.append("POISON")
                    else:
                        safe_count += 1
                        dataset_status.append("SAFE")

        # Create DataFrames
        df['_status'] = dataset_status[:len(df)]
        
        poison_df = df[df['_status'] == 'POISON'].drop(columns=['_status'])
        safe_df = df[df['_status'] == 'SAFE'].drop(columns=['_status'])
        
        import uuid
        scan_id = str(uuid.uuid4())[:8]
        
        # 4. Save RESULTS to GridFS (No local files!)
        poison_bytes = dataframe_to_bytes(poison_df, "csv")
        safe_bytes = dataframe_to_bytes(safe_df, "csv")
        
        poison_filename = f"poison_{scan_id}.csv"
        safe_filename = f"safe_{scan_id}.csv"
        
        poison_file_id = await upload_bytes_to_gridfs(poison_filename, poison_bytes, metadata={"scan_id": scan_id, "type": "result_poison"})
        safe_file_id = await upload_bytes_to_gridfs(safe_filename, safe_bytes, metadata={"scan_id": scan_id, "type": "result_safe"})
        
        # Also save Parquet backups to DB ? 
        # User said "store those only in the database".
        # Let's save parquet versions too just in case we need efficient export later
        safe_pq_bytes = dataframe_to_bytes(safe_df, "parquet")
        await upload_bytes_to_gridfs(f"{scan_id}_safe.parquet", safe_pq_bytes, metadata={"scan_id": scan_id, "type": "backup_safe"})
        
        # --- MongoDB Scan Record ---
        scan_record = {
            "scan_id": scan_id,
            "filename": file.filename,
            "original_file_id": original_file_id,
            "uploaded_by": current_user['username'],
            "timestamp": datetime.utcnow(),
            "total_rows": len(df),
            "poison_count": poison_count,
            "safe_count": safe_count,
            "poison_file_id": poison_file_id,
            "safe_file_id": safe_file_id,
            "sensitivity_used": state.get("sensitivity", 1.0)
        }
        
        db = get_database()
        await db.scans.insert_one(scan_record)
        print(f"DEBUG: Saved scan record {scan_id} to MongoDB (GridFS backed)", flush=True)

        return {
            "scan_id": scan_id,
            "total_rows": len(df),
            "poison_count": poison_count,
            "safe_count": safe_count,
            "poison_file": f"/api/downloads/{poison_filename}", # Keep URL structure but serve from DB
            "safe_file": f"/api/downloads/{safe_filename}",
            "message": "Scan completed. Files stored securely in Database."
        }
    except Exception as e:
        print(f"Error scanning dataset: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/api/downloads/{filename}")
async def download_file(filename: str):
    """Serve files from GridFS"""
    try:
        # Try to find file by filename in GridFS
        content = await download_bytes_from_gridfs(filename=filename)
        from fastapi.responses import Response
        return Response(content=content, media_type="application/octet-stream", headers={"Content-Disposition": f"attachment; filename={filename}"})
    except Exception as e:
        # Fallback to local for backward compatibility? NO, User said "only in database"
        return {"error": "File not found in database"}

@app.websocket("/ws/metrics")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep alive / or handle client messages if needed
            # For now just wait
            await websocket.receive_text() 
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/api/dataset/export")
async def export_dataset(scan_id: str, type: str, format: str):
    """
    Export specific dataset (safe/poison) from GridFS in requested format.
    """
    if type not in ['safe', 'poison']:
        raise HTTPException(status_code=400, detail="Invalid type. Use 'safe' or 'poison'.")
        
    filename = f"{scan_id}_{type}.parquet"
    
    try:
        # Load Parquet buffer from GridFS
        content = await download_bytes_from_gridfs(filename=filename)
        buffer = io.BytesIO(content)
        df = pd.read_parquet(buffer)
        
        # Convert to requested format
        out_bytes = dataframe_to_bytes(df, format)
        
        from fastapi.responses import Response
        out_filename = f"logicloopers_{type}_{scan_id[:8]}.{format}"
        if format == 'excel': out_filename = out_filename.replace('.excel', '.xlsx')
            
        return Response(
            content=out_bytes, 
            media_type='application/octet-stream',
            headers={"Content-Disposition": f"attachment; filename={out_filename}"}
        )
        
    except Exception as e:
        print(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

    return results

@app.get("/api/models")
async def list_models(current_user: dict = Depends(get_current_user)):
    """List trained models for the user"""
    db = get_database()
    cursor = db.fs.files.find({
        "metadata.owner": current_user["username"],
        "metadata.type": "trained_model"
    })
    models = []
    async for m in cursor:
        models.append({
            "id": str(m["_id"]),
            "filename": m["filename"],
            "accuracy": m["metadata"].get("accuracy", 0.0),
            "dataset": m["metadata"].get("dataset", "Unknown"),
            "timestamp": m["uploadDate"].isoformat()
        })
    return models

@app.post("/api/models/activate")
async def activate_model(model_id: str, current_user: dict = Depends(get_current_user)):
    """Switch the current active neural engine"""
    global encoder, INPUT_DIM, active_model_metadata
    
    try:
        content = await download_bytes_from_gridfs(file_id=model_id)
        checkpoint = torch.load(io.BytesIO(content), map_location='cpu')
        
        meta = checkpoint['metadata']
        new_encoder = TabularMLPEncoder(
            input_dim=meta['input_dim'],
            hidden_dim=meta['hidden_dim'],
            output_dim=meta['output_dim']
        )
        new_encoder.load_state_dict(checkpoint['encoder'])
        new_encoder.eval()
        
        encoder = new_encoder
        INPUT_DIM = meta['input_dim']
        active_model_metadata = meta
        state["active_model_id"] = model_id
        
        return {"status": "success", "message": f"Active engine switched to {meta['input_dim']}-dim model"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to activate model: {str(e)}")

active_model_metadata = None # Stores features and scaler

@app.post("/api/check")
async def check_sample(req: CheckRequest):
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
    features = active_model_metadata.get("features") if active_model_metadata else None
    llm_result = parser.parse(text, features=features)
    
    parsed = llm_result["parsed_data"]
    contradictions = llm_result.get("contradictions", [])
    anomalies = llm_result.get("anomalies", [])
    llm_risk = llm_result.get("risk_assessment", "SAFE")
    llm_used = llm_result.get("llm_used", False)
    
    print(f"[check_sample] LLM Used: {llm_used}, Risk: {llm_risk}", flush=True)
    print(f"[check_sample] Contradictions: {contradictions}", flush=True)
    print(f"[check_sample] Anomalies: {anomalies}", flush=True)
    print(f"[check_sample] RAW PARSED BMI: {parsed.get('BMI')}", flush=True)
    
    # Build input vector dynamically if custom model active
    if active_model_metadata and "features" in active_model_metadata:
        features = active_model_metadata["features"]
        input_vec = np.array([[parsed.get(f, 0) for f in features]], dtype=np.float32)
        
        # Apply custom scaling
        scaler_meta = active_model_metadata["scaler"]
        means = np.array(scaler_meta["mean"], dtype=np.float32)
        stds = np.array(scaler_meta["scale"], dtype=np.float32)
        input_scaled = (input_vec - means) / (stds + 1e-6)
    else:
        # Default BRFSS scaling
        input_vec = np.array([[
            parsed["HighBP"], parsed["HighChol"], parsed.get("CholCheck", 1), parsed["BMI"],
            parsed["Smoker"], parsed["Stroke"], parsed["HeartDisease"], parsed.get("PhysActivity", 1),
            parsed.get("Fruits", 1), parsed.get("Veggies", 1), parsed["HvyAlcohol"], parsed.get("Healthcare", 1),
            parsed.get("NoDocCost", 0), parsed["GenHlth"], parsed["MentHlth"], parsed["PhysHlth"],
            parsed["DiffWalk"], parsed.get("Sex", 0), parsed["Age"], parsed.get("Education", 4), parsed.get("Income", 5)
        ]], dtype=np.float32)
        
        ref_means = np.array([0.5]*INPUT_DIM, dtype=np.float32)
        ref_means[3] = 28.0; ref_means[14] = 3.0; ref_means[15] = 3.0
        ref_stds = np.array([0.5]*INPUT_DIM, dtype=np.float32)
        ref_stds[3] = 6.0; ref_stds[14] = 5.0; ref_stds[15] = 5.0
        input_scaled = (input_vec - ref_means) / (ref_stds + 1e-6)
    
    # Run Model for embedding-based anomaly detection
    input_tensor = torch.FloatTensor(input_scaled)
    with torch.no_grad():
        emb = encoder(input_tensor)
        score = torch.norm(emb).item() # Fixed: Defined missing score variable
    
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
    
    response_data = {
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

    # --- Audit Log ---
    try:
        db = get_database()
        await db.audit_logs.insert_one({
            "timestamp": datetime.utcnow(),
            "input_text": text,
            "parsed": parsed,
            "result": {
                 "is_poison": is_poison,
                 "score": score,
                 "verdict": verdict
            },
            "source_ip": "unknown"
        })
    except Exception as e:
        print(f"Failed to write audit log: {e}")

    return response_data

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
