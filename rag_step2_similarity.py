"""
Step 2: Compare Vectors (Similarity)

In Step 1, we turned text into numbers (embeddings).
Now: HOW do you compare two sets of numbers to find "similar meaning"?

THREE METHODS:
  1. Cosine Similarity — compares DIRECTION (most popular for text)
  2. Dot Product — compares direction AND magnitude
  3. Euclidean Distance — compares actual DISTANCE between points

Think of it like comparing two people walking:
  - Cosine: "Are they walking in the same direction?" (ignores speed)
  - Dot Product: "Same direction AND similar speed?"
  - Euclidean: "How far apart are they right now?"

This file demonstrates:
  1. All 3 methods with visual intuition
  2. When each method gives different answers
  3. Building a proper search engine with ranking
  4. Speed comparison for real-world use
"""

import numpy as np
import time
from rag_utils import load_embedding_model, cosine_sim

model = load_embedding_model()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: The three similarity methods — explained with numbers
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PART 1: Three ways to compare vectors")
print("=" * 65)

# Let's use tiny 2D vectors first to build intuition
# (Real embeddings are 384D, but the math is the same)

print("\n--- Simple 2D example first ---\n")

# Imagine a 2D space: x = "techiness", y = "foodiness"
vec_python = np.array([0.9, 0.1])   # very tech, not food
vec_coding = np.array([0.8, 0.15])  # very tech, not food
vec_recipe = np.array([0.1, 0.95])  # not tech, very food
vec_both   = np.array([0.7, 0.7])   # somewhat both

vectors = {
    "Python":  vec_python,
    "Coding":  vec_coding,
    "Recipe":  vec_recipe,
    "TechFood": vec_both,
}

# cosine_sim imported from rag_utils

def dot_product(a, b):
    """Direction + magnitude: higher if both point same way AND are large"""
    return np.dot(a, b)

def euclidean_dist(a, b):
    """Straight-line distance: 0 = same point, higher = further apart"""
    return np.linalg.norm(a - b)

print(f"  {'Pair':<22} {'Cosine':>8} {'Dot Prod':>10} {'Euclidean':>10}")
print(f"  {'─' * 22} {'─' * 8} {'─' * 10} {'─' * 10}")

pairs = [
    ("Python", "Coding"),   # similar topic
    ("Python", "Recipe"),   # different topic
    ("Python", "TechFood"), # partially related
    ("Recipe", "TechFood"), # partially related
]

for name1, name2 in pairs:
    v1, v2 = vectors[name1], vectors[name2]
    cos = cosine_sim(v1, v2)
    dot = dot_product(v1, v2)
    euc = euclidean_dist(v1, v2)
    print(f"  {name1 + ' ↔ ' + name2:<22} {cos:>8.4f} {dot:>10.4f} {euc:>10.4f}")

print("""
  💡 Key insight:
     Cosine: Python↔Coding = 0.99 (almost identical direction!)
     Euclidean: Python↔Coding = 0.11 (very close in space!)
     Both agree: these are similar.
     
     But Cosine ONLY cares about direction, not magnitude.
     Euclidean cares about actual distance.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: Real embeddings — comparing all 3 methods
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PART 2: Real embeddings — all 3 methods compared")
print("=" * 65)

sentences = [
    # Group A: Programming
    "Python is great for building web applications",
    "JavaScript runs in the browser and on servers",
    "Writing clean code makes maintenance easier",
    
    # Group B: Food
    "Italian pasta with fresh tomato sauce is delicious",
    "Sushi requires fresh fish and seasoned rice",
    "Baking a chocolate cake needs cocoa and butter",
    
    # Group C: Science
    "DNA contains the genetic instructions for life",
    "Quantum physics describes subatomic particles",
    "The theory of relativity changed our understanding of time",
]

labels = [
    "Python web apps", "JavaScript", "Clean code",
    "Italian pasta", "Sushi", "Chocolate cake",
    "DNA genetics", "Quantum physics", "Relativity",
]

embeddings = model.encode(sentences)

print(f"\n📊 Comparing: \"{sentences[0]}\" against all others\n")
print(f"  {'Sentence':<20} {'Cosine':>8} {'Dot Prod':>10} {'Euclidean':>10}  {'Verdict'}")
print(f"  {'─' * 20} {'─' * 8} {'─' * 10} {'─' * 10}  {'─' * 10}")

base = embeddings[0]
for i in range(1, len(sentences)):
    cos = cosine_sim(base, embeddings[i])
    dot = dot_product(base, embeddings[i])
    euc = euclidean_dist(base, embeddings[i])
    
    if cos > 0.5:
        verdict = "🟢 Related"
    elif cos > 0.25:
        verdict = "🟡 Weak"
    else:
        verdict = "🔴 Different"
    
    print(f"  {labels[i]:<20} {cos:>8.4f} {dot:>10.4f} {euc:>10.4f}  {verdict}")

print("""
  💡 Notice:
     - All 3 methods AGREE on ranking (programming > food/science)
     - Cosine is between -1 and 1 (easy to interpret)
     - Dot product can be any number (harder to set thresholds)
     - Euclidean: LOWER = more similar (opposite of cosine!)
     
  🏆 For text search, Cosine Similarity wins because:
     1. Scale doesn't matter (normalized)
     2. Easy threshold: >0.5 = similar, <0.2 = unrelated
     3. Works best with sentence-transformers models
""")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: Build a proper search engine with ranking
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PART 3: Proper search engine with Top-K ranking")
print("=" * 65)

# Bigger knowledge base — 30 documents
knowledge_base = [
    # Python (0-5)
    "Python list comprehensions create lists in a single line of code",
    "Django REST framework is used for building APIs in Python",
    "Virtual environments isolate Python project dependencies",
    "Python decorators modify the behavior of functions",
    "The GIL in Python prevents true multi-threading",
    "Async and await keywords enable asynchronous programming in Python",
    
    # AI/ML (6-11)
    "Transformers architecture revolutionized natural language processing",
    "GPT models predict the next token in a sequence",
    "Fine-tuning adapts a pre-trained model to a specific task",
    "RAG combines retrieval with generation for better AI answers",
    "Embeddings represent words as dense numerical vectors",
    "Attention mechanism helps models focus on relevant parts of input",
    
    # Web Dev (12-17)
    "React components use JSX to describe user interfaces",
    "REST APIs use HTTP methods like GET POST PUT DELETE",
    "CSS Flexbox makes responsive layouts easy to build",
    "WebSocket enables real-time bidirectional communication",
    "OAuth 2.0 is the standard protocol for authorization",
    "Docker containers package applications with their dependencies",
    
    # Database (18-23)
    "SQL JOIN combines rows from two or more tables",
    "MongoDB stores data as flexible JSON-like documents",
    "Redis is an in-memory data store used for caching",
    "Database indexing speeds up query performance significantly",
    "ACID properties ensure reliable database transactions",
    "PostgreSQL supports both relational and JSON data types",
    
    # DevOps (24-29)
    "CI/CD pipelines automate building testing and deploying code",
    "Kubernetes orchestrates containerized applications at scale",
    "Terraform provisions infrastructure using code",
    "Git branches allow parallel development workflows",
    "Load balancers distribute traffic across multiple servers",
    "Monitoring with Prometheus helps detect system issues early",
]

# Pre-compute all embeddings
print(f"\nEmbedding {len(knowledge_base)} documents...")
kb_embeddings = model.encode(knowledge_base)
print(f"Done! Shape: {kb_embeddings.shape}\n")


def search(query, method="cosine", top_k=5):
    """
    Search the knowledge base using different similarity methods.
    
    Args:
        query: The search text
        method: "cosine", "dot", or "euclidean"
        top_k: How many results to return
    """
    query_emb = model.encode(query)
    
    scores = []
    for i, doc_emb in enumerate(kb_embeddings):
        if method == "cosine":
            score = cosine_sim(query_emb, doc_emb)
        elif method == "dot":
            score = dot_product(query_emb, doc_emb)
        elif method == "euclidean":
            # Negate so higher = better (closer = more similar)
            score = -euclidean_dist(query_emb, doc_emb)
        scores.append((i, score))
    
    # Sort by score (highest first)
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# Test with multiple queries
test_queries = [
    "How do I build a REST API?",
    "What is the best way to store data?",
    "How does GPT work?",
    "How to deploy my application?",
    "What is async programming?",
]

for query in test_queries:
    print(f"🔍 \"{query}\"")
    results = search(query, method="cosine", top_k=3)
    for rank, (idx, score) in enumerate(results, 1):
        emoji = "🟢" if score > 0.5 else "🟡" if score > 0.3 else "🔴"
        print(f"   {emoji} #{rank} [{score:.4f}] {knowledge_base[idx]}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: Do the methods ever DISAGREE?
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PART 4: When methods disagree (ranking differences)")
print("=" * 65)

query = "How to make my Python code faster?"
query_emb = model.encode(query)

print(f"\n🔍 Query: \"{query}\"\n")

# Get rankings from all 3 methods
methods = {
    "Cosine": lambda q, d: cosine_sim(q, d),
    "Dot Product": lambda q, d: dot_product(q, d),
    "Euclidean": lambda q, d: -euclidean_dist(q, d),
}

for method_name, score_fn in methods.items():
    scores = [(i, score_fn(query_emb, kb_embeddings[i])) for i in range(len(knowledge_base))]
    scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"  📏 {method_name} ranking:")
    for rank, (idx, score) in enumerate(scores[:3], 1):
        print(f"     #{rank} [{score:>8.4f}] {knowledge_base[idx][:60]}...")
    print()

print("""  💡 In practice, all 3 methods usually give the SAME top results
     for text embeddings. This is because sentence-transformers
     produces normalized vectors (length ≈ 1.0), which makes:
       - Cosine similarity ≈ Dot product
       - Euclidean distance ∝ inverse of cosine similarity
     
     So just use Cosine Similarity — it's the standard! ✅
""")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: Speed test — how fast is vector search?
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PART 5: Speed benchmark")
print("=" * 65)

# Test search speed with our 30-document knowledge base
query = "How to build a web application?"
query_emb = model.encode(query)

# Time the similarity computation (not the embedding step)
num_iterations = 1000

start = time.time()
for _ in range(num_iterations):
    scores = [cosine_sim(query_emb, doc_emb) for doc_emb in kb_embeddings]
elapsed = time.time() - start

print(f"\n  Documents: {len(knowledge_base)}")
print(f"  Iterations: {num_iterations}")
print(f"  Total time: {elapsed:.4f} seconds")
print(f"  Per search: {elapsed/num_iterations*1000:.4f} ms")
print(f"  Searches/second: {num_iterations/elapsed:.0f}")

# Now time with NumPy matrix operations (the fast way)
start = time.time()
for _ in range(num_iterations):
    # This does ALL comparisons at once using matrix math
    norms_q = np.linalg.norm(query_emb)
    norms_d = np.linalg.norm(kb_embeddings, axis=1)
    scores = np.dot(kb_embeddings, query_emb) / (norms_d * norms_q)
elapsed_fast = time.time() - start

print(f"\n  ⚡ With NumPy matrix ops:")
print(f"  Per search: {elapsed_fast/num_iterations*1000:.4f} ms")
print(f"  Searches/second: {num_iterations/elapsed_fast:.0f}")
print(f"  Speedup: {elapsed/elapsed_fast:.1f}x faster!")

print(f"""
  💡 With 30 documents, even the slow way is fast.
     But with 1 MILLION documents, you'd need:
       Loop: ~{elapsed/num_iterations * 1_000_000:.0f} seconds per search 😱
       NumPy: ~{elapsed_fast/num_iterations * 1_000_000:.1f} seconds per search
       
     That's why we need vector databases (ChromaDB, Pinecone, etc.)
     — they use smart indexing to search millions in milliseconds!
     
     → That's Step 3! 🚀
""")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 6: Interactive search (try your own queries!)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PART 6: Try it yourself! (type 'quit' to exit)")
print("=" * 65)

print("\n  Knowledge base topics: Python, AI/ML, Web Dev, Databases, DevOps")
print(f"  Total documents: {len(knowledge_base)}\n")

while True:
    query = input("  🔍 Search: ").strip()
    if query.lower() in ("quit", "exit", "q"):
        print("\n  Goodbye! 👋")
        break
    if not query:
        continue
    
    results = search(query, method="cosine", top_k=5)
    print()
    for rank, (idx, score) in enumerate(results, 1):
        emoji = "🟢" if score > 0.5 else "🟡" if score > 0.3 else "🔴"
        print(f"   {emoji} #{rank} [{score:.4f}] {knowledge_base[idx]}")
    print()
