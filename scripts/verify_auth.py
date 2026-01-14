import requests
import sys

BASE_URL = "http://localhost:8000"

def test_register():
    print(f"Checking {BASE_URL}/api/auth/register...")
    try:
        # Preflight check (OPTIONS)
        opt_resp = requests.options(f"{BASE_URL}/api/auth/register")
        print(f"OPTIONS Status: {opt_resp.status_code}")
        
        # Actual Register
        payload = {"username": "test_verification_user", "password": "securepassword"}
        resp = requests.post(f"{BASE_URL}/api/auth/register", json=payload)
        
        print(f"POST Status: {resp.status_code}")
        print(f"Response: {resp.text}")
        
        if resp.status_code in [200, 400]: # 400 if user exists is also "Success" in terms of endpoint existing
            print("✅ Endpoint exists.")
            return True
        elif resp.status_code == 404:
            print("❌ Endpoint NOT FOUND (404).")
            return False
        else:
            print(f"⚠️ Unexpected status: {resp.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return False

if __name__ == "__main__":
    test_register()
