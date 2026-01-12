# Invisible Data Poisoning Detection System

### AI Training Firewall for Preventing Silent Model Corruption

---

## Description

Modern AI systems are increasingly vulnerable to **silent data poisoning attacks**, where attackers inject a small percentage of malicious but valid-looking data into the training pipeline. These attacks bypass traditional defenses because training accuracy and validation metrics remain normal, while the model’s internal behavior is subtly corrupted—leading to biased predictions, backdoors, or malicious behavior after deployment.

The **Invisible Data Poisoning Detection System** is an AI-driven **training-time security firewall** that prevents such attacks *before* the model is damaged. Instead of relying on labels or accuracy, the system monitors **representation geometry and learning dynamics inside the model** using contrastive learning, spectral analysis, and drift detection.

The system is:

* Fully **unsupervised**
* **Dataset-agnostic**
* Effective across **medical data, census data, network security data, and system logs**
* Capable of **real-time attack detection**
* Able to **automatically halt training** before irreversible corruption

> **We don’t detect poisoned data.
> We detect poisoned learning.**

---

## 🎥 Demo Video Link

👉 **Demo Video:** <INSERT GOOGLE DRIVE LINK HERE>

---

## Features

*  **Training-Time AI Firewall** – Detects attacks while the model is learning
*  **Invisible Attack Detection** – Catches clean-label and backdoor poisoning attacks
*  **Intelligent NLP Validator (The Judge's Test)** – Uses **Google Gemini AI** to detect logical contradictions in natural language input (e.g., "healthy athlete with heart disease").
*  **Representation Geometry Monitoring** – Tracks effective rank, isotropy, and density
*  **Automatic HALT Mechanism** – Stops training before damage propagates
*  **Dataset-Agnostic Architecture** – Same defense logic across multiple domains
*  **Real-Time Monitoring** – Detects poisoning within early training batches

---

## Tech Stack

### AI / Machine Learning

* Python
* PyTorch
* **Google Gemini 2.0 Flash / Pro** (LLM-based Logic Verification)
* Contrastive Learning (NT-Xent Loss)
* Spectral Analysis (Effective Rank)
* Representation Geometry Monitoring

### Backend & Infrastructure

* FastAPI
* Hugging Face, Kaggle Datasets
* NumPy, Pandas, Scikit-learn
* Postgres Database

### Frontend (Advanced Dashboard)

* React
* Framer Motion
* Recharts
* Tailwind CSS

---

## Google Technologies Used (Mandatory)

> ⚠️ Use of Google technologies is mandatory for this hackathon.

* **Google Gemini API (Gemini 2.0 Flash)** – Powering the "Judge's Data Test" feature. The system uses Gemini to perform zero-shot logical reasoning on natural language inputs, detecting semantic contradictions (e.g., "Active marathon runner in a wheelchair") that traditional rule-based parsers miss. This acts as a cognitive security layer for unstructured data intake.
* **TensorFlow** – Used as a Google-backed deep learning framework for experimentation, validation, and interoperability checks alongside PyTorch, ensuring the system remains framework-agnostic and compatible with TensorFlow-based training pipelines.

---

## Datasets Used

* **BRFSS 2015 Diabetes Dataset** (~253,680 rows)


All datasets are **real, public, and anonymized**, suitable for research and hackathon use.

---

## Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone <https://github.com/OP-Prajwal/WinterHackathon-TheLogicLoopers>
cd WinterHackathon-TheLogicLoopers
```

### 2️⃣ Create Virtual Environment & Install Dependencies

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 3️⃣ Configure Environment Variables

Create a `.env` file in the root directory (or rename `.env.example`).
**Required Variables:**

```env
# Google Gemini API Key (for LLM Validator)
GEMINI_API_KEY=your-api-key-here

# MongoDB Connection (for GridFS & Settings)
# Example: mongodb://localhost:27017 or your Atlas URL
MONGO_URL=mongodb+srv://<user>:<password>@cluster.mongodb.net/?retryWrites=true&w=majority

# JWT Secret (for Authentication)
SECRET_KEY=your-secret-key-change-me
```

### 4️⃣ Start the Backend

The server runs on port **8000** (Documentation: http://localhost:8000/docs).

```bash
# Make sure venv is active
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### 5️⃣ Start the Frontend

Open a new terminal:

```bash
cd frontend
npm install
npm run dev
```

### 6️⃣ Run the Defense System

1.  **Dashboard:** Open http://localhost:5173
2.  **Login/Register:** Create an account to access the dashboard.
3.  **Upload & Scan:** Go to the dashboard, upload a dataset (CSV/Parquet), and watch the real-time metrics.
4.  **Test Logic:** Use the "Judge's Test" feature to validate natural language inputs with Gemini.

---

## Project Structure (High Level)

```
src/
 └── poison_guard/
     ├── core/              # Core interfaces
     ├── data/              # Dataset adapters
     ├── models/            # Encoders & contrastive heads
     ├── baselines/         # Behavioral fingerprints
     ├── security_engine/   # Spectral & geometric detection
     └── pipeline/          # Trainer & Auditor

server.py                   # FastAPI Backend (Defense Firewall)
llm_parser.py               # Gemini AI Integration Module
train_model.py              # Main Training Pipeline
generate_test_data.py       # Syntax-checker / Data Generator
.env                        # API Key Configuration
```

---

## Why This Project Matters

* Accuracy-based defenses **fail** against modern poisoning attacks
* Label inspection is **too late**
* This system secures AI **before it learns the wrong behavior**
* **LLM-Enhanced Security**: We use AI to protect AI, adding a layer of semantic understanding that statistical methods lack.

By monitoring **how data behaves inside the model**, the system stops attacks that are invisible to humans and traditional ML pipelines.

---

## Team Members – *The Logic Loopers*

* **Prajwal Gaonkar**
* **Karthik**
* **Mohit**
* **Kanak Tanwar**

---

## Final Status

✅ Architecture Locked & Implemented
✅ Multi-Domain Validation Passed
✅ Real-Time Poison Detection Proven
✅ **Gemini LLM Integration Complete**
✅ Auto-HALT Defense Working
✅ Ready for Demo & Deployment

---

*Winter Hackathon – Organized by Sceptix & GDG SJEC* ✨
