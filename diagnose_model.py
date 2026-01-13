"""
Diagnostic script to check model file integrity in MongoDB GridFS
"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from bson import ObjectId
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")
MODEL_ID = "69657530bbc02a3779f5867a"  # The ID from your logs

async def diagnose():
    print(f"Connecting to MongoDB...")
    client = AsyncIOMotorClient(MONGO_URL)
    db = client.poison_guard
    
    print(f"\n--- Checking Model File: {MODEL_ID} ---\n")
    
    # 1. Check fs.files
    file_doc = await db.fs.files.find_one({"_id": ObjectId(MODEL_ID)})
    if file_doc:
        print(f"✅ File Metadata Found:")
        print(f"   Filename: {file_doc.get('filename')}")
        print(f"   Length: {file_doc.get('length')} bytes")
        print(f"   Chunk Size: {file_doc.get('chunkSize')}")
        print(f"   Upload Date: {file_doc.get('uploadDate')}")
        print(f"   Metadata: {file_doc.get('metadata')}")
    else:
        print(f"❌ File Metadata NOT FOUND in fs.files!")
        print("   The model ID does not exist in the database.")
        client.close()
        return
    
    # 2. Check fs.chunks
    chunk_count = await db.fs.chunks.count_documents({"files_id": ObjectId(MODEL_ID)})
    print(f"\n   Chunks in fs.chunks: {chunk_count}")
    
    if chunk_count == 0:
        print(f"❌ NO CHUNKS FOUND! The file is EMPTY/CORRUPTED!")
        print("   This is why 'pop from empty list' occurs.")
        print("\n   SOLUTION: You need to RE-TRAIN the model in Personal Training.")
    else:
        print(f"✅ Chunks exist. Attempting to read...")
        
        try:
            fs = AsyncIOMotorGridFSBucket(db)
            grid_out = await fs.open_download_stream(ObjectId(MODEL_ID))
            content = await grid_out.read()
            
            print(f"   Downloaded Size: {len(content)} bytes")
            
            if len(content) > 0:
                print(f"✅ Model file is VALID and contains data!")
                
                # Try to load with torch
                import torch
                import io
                try:
                    checkpoint = torch.load(io.BytesIO(content), map_location='cpu')
                    print(f"✅ PyTorch loaded successfully!")
                    print(f"   Keys: {list(checkpoint.keys())}")
                except Exception as e:
                    print(f"❌ PyTorch load failed: {e}")
            else:
                print(f"❌ Downloaded content is empty!")
                
        except Exception as e:
            print(f"❌ Download failed: {e}")
    
    # 3. List all trained models for this user
    print(f"\n--- All Trained Models in Database ---")
    cursor = db.fs.files.find({"metadata.type": "trained_model"})
    async for doc in cursor:
        chunks = await db.fs.chunks.count_documents({"files_id": doc["_id"]})
        status = "✅" if doc.get("length", 0) > 0 and chunks > 0 else "⚠️ EMPTY"
        print(f"   {status} ID: {doc['_id']} | Name: {doc['filename']} | Size: {doc.get('length', 0)} bytes | Chunks: {chunks}")
    
    client.close()
    print("\n--- Diagnosis Complete ---")

if __name__ == "__main__":
    asyncio.run(diagnose())
