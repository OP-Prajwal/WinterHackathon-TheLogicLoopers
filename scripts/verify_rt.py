import asyncio
import websockets
import json
import requests
import time
import os

async def check_metrics():
    uri = "ws://localhost:8003/ws/metrics"
    # Wait for server
    time.sleep(2) 
    
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected! Triggering scan now...")
            
            # Start scan in background (using requests, non-async for simplicity here or async via aiohttp if I had it, but requests is fine if just triggering)
            # Actually requests blocks, so let's use a thread? Or just simple post before listening?
            # Scan takes time, so we should trigger it and then listen.
            
            import threading
            def run_scan():
                try:
                    if not os.path.exists('test_scan.csv'):
                        # Create dummy if not exists
                        with open('test_scan.csv', 'w') as f:
                            f.write("f1,f2,f3\n1,2,3")
                            
                    files = {'file': open('test_scan.csv', 'rb')}
                    print("POST /api/dataset/scan...")
                    r = requests.post("http://localhost:8003/api/dataset/scan", files=files)
                    print("Scan Response:", r.status_code)
                except Exception as e:
                    print(f"Scan trigger failed: {e}")
                    
            t = threading.Thread(target=run_scan)
            t.start()
            
            print("Listening for metrics...")
            # Wait for multiple messages
            for _ in range(5):
                msg = await asyncio.wait_for(websocket.recv(), timeout=10)
                data = json.loads(msg)
                print(f"Received Metric: Throughput={data.get('current_throughput')}, Drift={data.get('drift_score_window')}")
                if 'drift_score_window' in data:
                     print("SUCCESS: Valid metric data received.")
                     return
            
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(check_metrics())
