import requests
import os
import time

url_scan = "http://localhost:8003/api/dataset/scan"
url_export = "http://localhost:8003/api/dataset/export"
csv_path = 'test_scan.csv'

if not os.path.exists(csv_path):
    print("Test file not found")
    exit(1)

# 1. Scan
print(f"Uploading {csv_path}...")
files = {'file': open(csv_path, 'rb')}
# No download_format needed now
try:
    r = requests.post(url_scan, files=files)
    print("Scan Status:", r.status_code)
    print("Full Response:", r.text) 
    
    if r.status_code != 200:
        print("Scan Failed:", r.text)
        exit(1)
        
    resp = r.json()
    scan_id = resp.get('scan_id')
    print("Scan ID:", scan_id)
    print(f"Safe: {resp.get('safe_count')}, Poison: {resp.get('poison_count')}")
    
    if not scan_id:
        print("No scan_id returned!")
        exit(1)
        
    # 2. Export as JSON
    print("\nRequesting Export (JSON)...")
    params = {'scan_id': scan_id, 'type': 'safe', 'format': 'json'}
    r_export = requests.get(url_export, params=params)
    
    print("Export Status:", r_export.status_code)
    if r_export.status_code == 200:
        content = r_export.text
        print("Export Content Preview:", content[:100])
        if "{" in content and "}" in content:
            print("SUCCESS: JSON Export received.")
        else:
            print("WARNING: Content might not be JSON.")
    else:
        print("Export Failed:", r_export.text)

    # 3. Export as Parquet
    print("\nRequesting Export (Parquet)...")
    params['format'] = 'parquet'
    r_export = requests.get(url_export, params=params)
    print("Export Status:", r_export.status_code)
    if r_export.status_code == 200:
        print("SUCCESS: Parquet Export received (binary).")
    else:
        print("Export Failed:", r_export.text)

except Exception as e:
    print("Error:", e)
