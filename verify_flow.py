import requests
import os

url = "http://localhost:8002/api/dataset/scan"
csv_path = 'test_scan.csv'

if not os.path.exists(csv_path):
    print("Test file not found")
    exit(1)

print(f"Uploading {csv_path}...")
files = {'file': open(csv_path, 'rb')}
# Request JSON output
data = {'download_format': 'json'}

try:
    r = requests.post(url, files=files, data=data)
    print("Status:", r.status_code)
    resp = r.json()
    print("Response keys:", resp.keys())
    print("Total:", resp.get('total_rows'))
    print("Poison:", resp.get('poison_count'))
    print("Safe:", resp.get('safe_count'))
    print("Poison URL:", resp.get('poison_file'))
    print("Safe URL:", resp.get('safe_file'))
except Exception as e:
    print("Error:", e)
