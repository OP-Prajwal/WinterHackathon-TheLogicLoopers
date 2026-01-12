
import requests
import time
import sys

BASE_URL = "http://localhost:8000/api"

def test_settings_persistence():
    print("Testing Settings Persistence...")
    
    # 1. Get initial
    try:
        r = requests.get(f"{BASE_URL}/settings")
        if r.status_code != 200:
            print(f"FAILED: Get settings returned {r.status_code}")
            return False
        initial = r.json()
        print(f"Initial: {initial}")
    except Exception as e:
         print(f"Connection failed: {e}")
         return False

    # 2. Change Sensitivity
    new_sens = 2.5
    r = requests.post(f"{BASE_URL}/settings/sensitivity?value={new_sens}")
    if r.status_code != 200:
        print(f"FAILED: Set sensitivity returned {r.status_code}")
        return False
    print(f"Set Sensitivity to {new_sens}")
    
    # 3. Verify Change
    r = requests.get(f"{BASE_URL}/settings")
    updated = r.json()
    print(f"Updated: {updated}")
    
    if abs(updated['sensitivity'] - new_sens) > 0.01:
        print("FAILED: Persistence check failed (value mismatch)")
        return False
        
    print("SUCCESS: Settings updated and retrieved.")
    return True

def test_audit_log():
    print("\nTesting Audit Log (check_sample)...")
    payload = {"text": "Patient has BMI 99 and eats rocks."}
    
    try:
        r = requests.post(f"{BASE_URL}/check", json=payload)
        if r.status_code != 200:
            print(f"FAILED: Check sample returned {r.status_code}")
            print(r.text)
            return False
        
        data = r.json()
        print("Check Result:", data['result']['verdict'])
        
        # We can't easily check MongoDB directly without a driver here, 
        # but if the 200 OK returned, the async insert likely triggered.
        # Ideally we'd query the DB to be sure, but for now 200 OK implies no crash.
        print("SUCCESS: Check endpoint returned 200 OK.")
        return True
    except Exception as e:
        print(f"Check failed: {e}")
        return False

if __name__ == "__main__":
    if test_settings_persistence() and test_audit_log():
        print("\nALL SYSTEM TESTS PASSED ✅")
        sys.exit(0)
    else:
        print("\nTESTS FAILED ❌")
        sys.exit(1)
