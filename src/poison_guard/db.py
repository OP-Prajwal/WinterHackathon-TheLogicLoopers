import os
from motor.motor_asyncio import AsyncIOMotorClient

# Default to local mongo if not set
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = "poison_guard_db"

class Database:
    client: AsyncIOMotorClient = None
    db = None


# Mock Database Implementation for Fallback
class MockCursor:
    def __init__(self, data):
        self.data = data
        self._sort = None
        self._limit = None

    def sort(self, key, direction):
        self._sort = (key, direction)
        return self

    def limit(self, limit):
        self._limit = limit
        return self

    async def to_list(self, length=None):
        result = list(self.data)
        if self._sort:
            key, direction = self._sort
            reverse = direction == -1
            try:
                 result.sort(key=lambda x: x.get(key, ""), reverse=reverse)
            except:
                pass # Best effort sort
        
        if self._limit:
            result = result[:self._limit]
        
        if length:
            result = result[:length]
            
        return result

class MockCollection:
    def __init__(self, name):
        self.name = name
        self.data = []

    async def find_one(self, query):
        for doc in self.data:
            match = True
            for k, v in query.items():
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                return doc
        return None

    async def insert_one(self, document):
        # Simulate ObjectId
        if "_id" not in document:
            from bson import ObjectId
            document["_id"] = ObjectId()
        self.data.append(document)
        return document

    def find(self, query={}):
        # Simple query filtering
        filtered = []
        for doc in self.data:
            match = True
            for k, v in query.items():
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                filtered.append(doc)
        return MockCursor(filtered)

class MockDatabase:
    def __init__(self):
        self.collections = {}

    def __getattr__(self, name):
        if name not in self.collections:
            self.collections[name] = MockCollection(name)
        return self.collections[name]

db = Database()

async def connect_to_mongo():
    try:
        # Short timeout to fail fast if no DB
        db.client = AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=2000)
        # Force connection check
        await db.client.server_info()
        db.db = db.client[DB_NAME]
        print(f"✅ Connected to MongoDB at {MONGO_URL}", flush=True)
    except Exception as e:
        print(f"⚠️ Could not connect to real MongoDB: {e}", flush=True)
        print("⚠️ SWITCHING TO IN-MEMORY MOCK DATABASE. Data will NOT persist after restart.", flush=True)
        db.db = MockDatabase()

async def close_mongo_connection():
    if db.client:
        db.client.close()
        print("Closed MongoDB connection")

def get_database():
    if db.db is None:
         # Fallback if accessed before connect (shouldn't happen in fastAPI startup usually)
         print("Warning: Database accessed before connection init. Returning Mock.", flush=True)
         return MockDatabase()
    return db.db
