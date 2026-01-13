"""
Check the exact type values of files in GridFS
"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")

async def check_types():
    client = AsyncIOMotorClient(MONGO_URL)
    db = client.poison_guard_db
    
    print("--- File Types in GridFS ---")
    cursor = db.fs.files.find({})
    async for doc in cursor:
        meta = doc.get("metadata", {})
        ftype = meta.get("type", "NO_TYPE")
        print(f"  Name: {doc['filename'][:40]:40} | Type: '{ftype}'")
    
    client.close()

if __name__ == "__main__":
    asyncio.run(check_types())
