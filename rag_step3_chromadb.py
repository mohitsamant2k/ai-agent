"""
Step 3: ChromaDB — Your First Vector Database

In Steps 1-2, we stored embeddings in Python lists/arrays.
Problem: Lists don't scale. Searching 1M documents takes 19 seconds!

ChromaDB solves this:
  ✅ Stores embeddings on disk (persistent)
  ✅ Fast search using HNSW indexing (milliseconds, not seconds)
  ✅ Adds metadata filtering (search by topic + similarity)
  ✅ Handles embedding automatically (no manual model.encode!)
  ✅ Scales to millions of documents

Think of it like:
  - Python list = sticky notes on your desk (works for 30 notes)
  - ChromaDB = a library with a catalog system (works for millions of books)

This file demonstrates:
  1. Creating a collection and adding documents
  2. Querying with natural language
  3. Metadata filtering (search WITHIN a topic)
  4. Updating and deleting documents
  5. Persistent storage (data survives restarts)
  6. ChromaDB vs manual search — speed comparison
"""

import chromadb
import time
import numpy as np
from rag_utils import (
    cosine_sim, load_embedding_model, distance_to_similarity, similarity_emoji,
    TECH_DOCUMENTS_25 as documents, TECH_TOPICS_25 as topics,
)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: Your first ChromaDB collection
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PART 1: Creating your first vector database")
print("=" * 65)

# Create a ChromaDB client (in-memory for now)
client = chromadb.Client()

# A "collection" is like a table in a regular database
# ChromaDB will automatically embed text using its default model!
collection = client.create_collection(
    name="tech_knowledge",
    metadata={"description": "Tech knowledge base for learning RAG"}
)

print(f"\n✅ Created collection: '{collection.name}'")
print(f"   ChromaDB handles embeddings automatically!")

# Documents and topics imported from rag_utils (TECH_DOCUMENTS_25, TECH_TOPICS_25)

# Each document needs a unique ID
ids = [f"doc_{i}" for i in range(len(documents))]

# Metadata — extra info we can filter on later! (topics imported from rag_utils)
metadatas = [{"topic": topic, "index": i} for i, topic in enumerate(topics)]

# Add everything to ChromaDB in one call
print(f"\nAdding {len(documents)} documents...")
start = time.time()
collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas,
)
elapsed = time.time() - start
print(f"Done in {elapsed:.2f}s! ChromaDB embedded + indexed everything.")
print(f"Documents in collection: {collection.count()}")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: Querying — just ask in natural language!
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("PART 2: Natural language search")
print("=" * 65)

queries = [
    "How do I build a REST API?",
    "What is the best way to store data?",
    "How does GPT work?",
    "How to deploy my application?",
    "What is async programming?",
]

for query in queries:
    # ChromaDB does EVERYTHING: embed query → search → rank → return
    results = collection.query(
        query_texts=[query],
        n_results=3,
    )
    
    print(f"\n🔍 \"{query}\"")
    for i in range(len(results["documents"][0])):
        doc = results["documents"][0][i]
        distance = results["distances"][0][i]
        doc_id = results["ids"][0][i]
        topic = results["metadatas"][0][i]["topic"]
        
        # ChromaDB returns DISTANCE (lower = better), not similarity
        # Convert: similarity ≈ 1 - (distance² / 2) for normalized vectors
        similarity = 1 - (distance ** 2 / 2)
        emoji = "🟢" if similarity > 0.5 else "🟡" if similarity > 0.3 else "🔴"
        print(f"   {emoji} [{similarity:.4f}] [{topic:>8}] {doc}")

print("""
  💡 Notice:
     - We just wrote query_texts=["..."] — that's it!
     - No model.encode(), no cosine_sim(), no sorting
     - ChromaDB did everything automatically
     - It even returns metadata (topic) with each result!
""")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: Metadata filtering — search WITHIN a topic
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PART 3: Metadata filtering (the killer feature!)")
print("=" * 65)

print("\n--- Without filter: search ALL documents ---\n")

results = collection.query(
    query_texts=["How to make my code better?"],
    n_results=5,
)
print(f"🔍 \"How to make my code better?\" (all topics)")
for i in range(len(results["documents"][0])):
    doc = results["documents"][0][i]
    topic = results["metadatas"][0][i]["topic"]
    distance = results["distances"][0][i]
    similarity = 1 - (distance ** 2 / 2)
    print(f"   [{similarity:.4f}] [{topic:>8}] {doc}")

print("\n--- With filter: ONLY Python documents ---\n")

results = collection.query(
    query_texts=["How to make my code better?"],
    n_results=5,
    where={"topic": "python"},  # ← Only search Python docs!
)
print(f"🔍 \"How to make my code better?\" (Python only)")
for i in range(len(results["documents"][0])):
    doc = results["documents"][0][i]
    topic = results["metadatas"][0][i]["topic"]
    distance = results["distances"][0][i]
    similarity = 1 - (distance ** 2 / 2)
    print(f"   [{similarity:.4f}] [{topic:>8}] {doc}")

print("\n--- Filter with multiple conditions ---\n")

# Search AI/ML OR Database topics only
results = collection.query(
    query_texts=["How do machines learn patterns?"],
    n_results=5,
    where={"$or": [{"topic": "ai_ml"}, {"topic": "database"}]},
)
print(f"🔍 \"How do machines learn patterns?\" (AI/ML + Database only)")
for i in range(len(results["documents"][0])):
    doc = results["documents"][0][i]
    topic = results["metadatas"][0][i]["topic"]
    distance = results["distances"][0][i]
    similarity = 1 - (distance ** 2 / 2)
    print(f"   [{similarity:.4f}] [{topic:>8}] {doc}")

print("""
  💡 Metadata filtering is HUGE for RAG:
     - User asks about Python? → Only search Python docs
     - User is a beginner? → Only search beginner-level docs
     - Document from 2024? → Filter by date metadata
     - Combine: topic="python" AND level="beginner" AND year>=2024
     
     This is like SQL WHERE clause + vector similarity together!
""")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: Update and delete documents
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PART 4: Update and delete operations")
print("=" * 65)

print(f"\n📊 Documents before: {collection.count()}")

# Update a document
collection.update(
    ids=["doc_0"],
    documents=["Python list comprehensions and generator expressions create efficient iterables"],
    metadatas=[{"topic": "python", "index": 0, "updated": True}],
)
print("✏️  Updated doc_0 with better content + 'updated' metadata")

# Verify the update
result = collection.get(ids=["doc_0"])
print(f"   New text: \"{result['documents'][0][:60]}...\"")
print(f"   Metadata: {result['metadatas'][0]}")

# Add a new document
collection.add(
    ids=["doc_25"],
    documents=["GitHub Actions automates CI/CD workflows in the cloud"],
    metadatas=[{"topic": "devops", "index": 25}],
)
print(f"\n➕ Added doc_25 (GitHub Actions)")
print(f"📊 Documents after add: {collection.count()}")

# Delete a document
collection.delete(ids=["doc_25"])
print(f"🗑️  Deleted doc_25")
print(f"📊 Documents after delete: {collection.count()}")

print("""
  💡 CRUD operations just like a regular database:
     - Create: collection.add(...)
     - Read:   collection.get(...) or collection.query(...)
     - Update: collection.update(...)
     - Delete: collection.delete(...)
""")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: Persistent storage — data survives restarts!
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PART 5: Persistent storage")
print("=" * 65)

# In-memory client (what we've been using) — data disappears when script ends
# Persistent client — data saved to disk!

import os
persist_dir = os.path.join(os.path.dirname(__file__), "chroma_db")

print(f"\n📁 Saving to: {persist_dir}")

# Create a persistent client
persistent_client = chromadb.PersistentClient(path=persist_dir)

# Create (or get existing) collection
persistent_collection = persistent_client.get_or_create_collection(
    name="persistent_knowledge"
)

# Only add documents if collection is empty (avoid duplicates on re-run)
if persistent_collection.count() == 0:
    print("   First run — adding documents...")
    persistent_collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas,
    )
    print(f"   Added {persistent_collection.count()} documents to disk!")
else:
    print(f"   Collection already has {persistent_collection.count()} documents (from previous run)")

# Query the persistent collection
results = persistent_collection.query(
    query_texts=["How does machine learning work?"],
    n_results=3,
)
print(f"\n🔍 Persistent search: \"How does machine learning work?\"")
for i in range(len(results["documents"][0])):
    doc = results["documents"][0][i]
    topic = results["metadatas"][0][i]["topic"]
    distance = results["distances"][0][i]
    similarity = 1 - (distance ** 2 / 2)
    print(f"   [{similarity:.4f}] [{topic:>8}] {doc}")

print("""
  💡 Persistent vs In-Memory:
     
     chromadb.Client()           → Data lost when script ends
     chromadb.PersistentClient() → Data saved to disk forever!
     
     Run this script twice — second time it won't re-add documents.
     Your vector database persists across restarts!
""")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 6: Speed comparison — ChromaDB vs manual search
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PART 6: Speed comparison — ChromaDB vs manual loop")
print("=" * 65)

from sentence_transformers import SentenceTransformer

print("\nLoading sentence-transformers model for manual comparison...")
st_model = load_embedding_model()

# Manual approach (from Step 2)
manual_embeddings = st_model.encode(documents)

# cosine_sim imported from rag_utils

query_text = "How to build a web application?"
query_emb = st_model.encode(query_text)

num_iterations = 500

# Manual search timing
start = time.time()
for _ in range(num_iterations):
    scores = [cosine_sim(query_emb, doc_emb) for doc_emb in manual_embeddings]
    top_3 = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:3]
manual_time = time.time() - start

# ChromaDB search timing (embedding already done at add time)
start = time.time()
for _ in range(num_iterations):
    results = collection.query(query_texts=[query_text], n_results=3)
chroma_time = time.time() - start

print(f"\n  Query: \"{query_text}\"")
print(f"  Documents: {len(documents)}")
print(f"  Iterations: {num_iterations}")
print(f"\n  📏 Manual loop:")
print(f"     Total: {manual_time:.4f}s | Per search: {manual_time/num_iterations*1000:.4f}ms")
print(f"\n  ⚡ ChromaDB:")
print(f"     Total: {chroma_time:.4f}s | Per search: {chroma_time/num_iterations*1000:.4f}ms")

if chroma_time < manual_time:
    print(f"\n  🏆 ChromaDB is {manual_time/chroma_time:.1f}x faster!")
else:
    print(f"\n  ℹ️  With only {len(documents)} docs, manual loop can be faster.")
    print(f"     ChromaDB shines at 10,000+ documents where HNSW indexing kicks in!")

print("""
  💡 Why ChromaDB wins at scale:
     25 documents    → Manual might be faster (less overhead)
     1,000 documents → ChromaDB starts winning
     100,000+ docs   → ChromaDB is 100x+ faster (HNSW indexing)
     
     Plus ChromaDB gives you:
       ✅ Persistent storage
       ✅ Metadata filtering
       ✅ Automatic embedding
       ✅ CRUD operations
       ✅ No need to manage numpy arrays yourself
""")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 7: Interactive ChromaDB search
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PART 7: Try ChromaDB yourself! (type 'quit' to exit)")
print("=" * 65)

print(f"\n  Topics: python, ai_ml, webdev, database, devops")
print(f"  Documents: {collection.count()}")
print(f"  Tip: type 'topic:python how to code' to filter by topic\n")

while True:
    user_input = input("  🔍 Search: ").strip()
    if user_input.lower() in ("quit", "exit", "q"):
        print("\n  Goodbye! 👋")
        break
    if not user_input:
        continue
    
    # Check for topic filter
    topic_filter = None
    query = user_input
    if user_input.startswith("topic:"):
        parts = user_input.split(" ", 1)
        topic_filter = parts[0].replace("topic:", "")
        query = parts[1] if len(parts) > 1 else ""
        if not query:
            print("  ⚠️  Please provide a search query after the topic filter")
            continue
    
    # Build the query
    kwargs = {
        "query_texts": [query],
        "n_results": 5,
    }
    if topic_filter:
        kwargs["where"] = {"topic": topic_filter}
        print(f"  📁 Filtering: topic={topic_filter}")
    
    try:
        results = collection.query(**kwargs)
        print()
        for i in range(len(results["documents"][0])):
            doc = results["documents"][0][i]
            topic = results["metadatas"][0][i]["topic"]
            distance = results["distances"][0][i]
            similarity = 1 - (distance ** 2 / 2)
            emoji = "🟢" if similarity > 0.5 else "🟡" if similarity > 0.3 else "🔴"
            print(f"   {emoji} [{similarity:.4f}] [{topic:>8}] {doc}")
        print()
    except Exception as e:
        print(f"  ❌ Error: {e}\n")
