import urllib.request
import urllib.parse
import json
import mimetypes
import uuid

url = "http://localhost:8001/api/dataset/scan"
filepath = "test_poison.csv"

# Build Multipart Form Data (Manual is painful, let's just use simple read if endpoint allows plain bytes? No it expects UploadFile)
# Actually, installing requests is safer/easier than writing manual multipart in urllib.
# But wait, I can just use curl if requests is missing! 
# Let's try curl. It's windows. `curl` works in powershell.

print("Running curl via subprocess...")
import subprocess

try:
    # PowerShell curl is actually Invoke-WebRequest, we need real curl or use cmd
    # Just use python to install requests quickly? No, slower.
    # Let's use simple python multipart logic helper or just install requests.
    pass
except:
    pass
    
# Actually, I'll just install requests. It's standard.
import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import requests
except ImportError:
    print("Installing requests...")
    install("requests")
    import requests

try:
    files = {'file': open('test_poison.csv', 'rb')}
    print(f"Sending request to {url}...")
    response = requests.post(url, files=files)
    print(f"Status: {response.status_code}")
    print(requests.get("http://localhost:8000/api/health").json()) # Check health too
except Exception as e:
    print(e)

