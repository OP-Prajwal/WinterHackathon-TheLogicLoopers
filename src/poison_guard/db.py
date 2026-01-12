import os
from motor.motor_asyncio import AsyncIOMotorClient

# Default to local mongo if not set
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = "poison_guard_db"

class Database:
    client: AsyncIOMotorClient = None
    db = None

db = Database()

async def connect_to_mongo():
    try:
        db.client = AsyncIOMotorClient(MONGO_URL)
        db.db = db.client[DB_NAME]
        print(f"Connected to MongoDB at {MONGO_URL}")
    except Exception as e:
        print(f"Could not connect to MongoDB: {e}")

async def close_mongo_connection():
    if db.client:
        db.client.close()
        print("Closed MongoDB connection")

def get_database():
    return db.db
