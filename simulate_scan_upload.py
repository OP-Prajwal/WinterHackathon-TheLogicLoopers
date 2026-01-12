import requests
import os

API_BASE = "http://localhost:8000"
USERNAME = "mockuser"
PASSWORD = "mockpass"
FILE_PATH = "test_poison.csv"

def login():
    try:
        response = requests.post(f"{API_BASE}/api/auth/login", json={
            "username": USERNAME,
            "password": PASSWORD
        })
        if response.status_code == 200:
            return response.json()["access_token"]
        print(f"Login failed: {response.text}")
        return None
    except Exception as e:
        print(f"Login error: {e}")
        return None

def upload_scan(token):
    if not os.path.exists(FILE_PATH):
        print(f"File {FILE_PATH} not found.")
        return

    try:
        with open(FILE_PATH, "rb") as f:
            files = {"file": (FILE_PATH, f, "text/csv")}
            headers = {"Authorization": f"Bearer {token}"}
            print(f"Uploading {FILE_PATH}...")
            response = requests.post(f"{API_BASE}/api/dataset/scan", files=files, headers=headers)
            
            if response.status_code == 200:
                print("✅ Scan uploaded successfully!")
                print(response.json())
            else:
                print(f"❌ Upload failed: {response.text}")
    except Exception as e:
        print(f"Upload error: {e}")

if __name__ == "__main__":
    print("Simulating User Scan...")
    # Ensure user exists (signup just in case, mock db might be empty)
    requests.post(f"{API_BASE}/api/auth/register", json={"username": USERNAME, "password": PASSWORD})
    
    token = login()
    if token:
        upload_scan(token)
