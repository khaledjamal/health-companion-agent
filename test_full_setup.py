# test_full_setup.py
from src.config import settings
from src.memory.database import DatabaseManager
from src.memory.vector_store import VectorStoreManager

print("=== Testing Database Connection ===")
db = DatabaseManager()
db.create_tables()
print("✅ Database tables created")

# Create test user
user = db.create_user(name="Test User", age=30, goals="Improve sleep and fitness")
print(f"✅ Created user: {user.name} (ID: {user.id})")

print("\n=== Testing Vector Store ===")
vector_store = VectorStoreManager()
print("✅ Vector store initialized")

# Load knowledge base
if vector_store.load_from_directory(settings.knowledge_base_dir) > 0:
    print("✅ Knowledge base loaded")
else:
    print("⚠️  No knowledge base files found - create data/knowledge_base/*.txt files")

# Test vector search
vector_store.add_document(
    content="Getting 7-9 hours of sleep is essential for recovery and HRV optimization.",
    topic="sleep"
)
results = vector_store.retrieve("sleep tips", top_k=1)
print(f"✅ Vector search working: found {len(results)} results")

print("\n=== Collection Info ===")
info = vector_store.get_collection_info()
print(f"Total documents: {info['total_documents']}")
print(f"Topics: {info['topics']}")

print("\n✅ All systems operational!")