"""
Step 6: Hybrid Search — Vector + Keyword (BM25)

Building on Step 5 (complete RAG pipeline), we now UPGRADE retrieval.

THE PROBLEM with vector-only search:
  ✅ Vector search finds SIMILAR MEANING (great!)
  ❌ But misses EXACT KEYWORDS sometimes

  Example: User asks "What is BM25?"
    Vector search → might return "information retrieval" docs (similar meaning)
    But MISSES a doc that literally says "BM25 is a ranking function..."

THE SOLUTION — Hybrid Search:
  1. Vector Search: "What does this MEAN?" (semantic similarity)
  2. BM25 Keyword Search: "Does this contain these WORDS?" (exact match)
  3. Reciprocal Rank Fusion (RRF): Merge both result lists smartly

This file demonstrates:
  Part 1: BM25 keyword search from scratch
  Part 2: Vector search (ChromaDB — we already know this)
  Part 3: Reciprocal Rank Fusion (RRF) — combining results
  Part 4: Head-to-head comparison: Vector vs BM25 vs Hybrid
  Part 5: Full Hybrid RAG pipeline with Azure OpenAI
"""

import os
import time
import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from rag_utils import (
    recursive_chunk, distance_to_similarity, similarity_emoji,
    PYTHON_DOC, AI_DOC, WEBDEV_DOC,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SETUP: Build knowledge base (same as Step 5)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("SETUP: Building Knowledge Base")
print("=" * 70)

documents = {
    "python_guide": PYTHON_DOC,
    "ai_ml_guide": AI_DOC,
    "webdev_guide": WEBDEV_DOC,
}

# ── Chunk all documents ─────────────────────────────────────────────────────

all_chunks = []
all_ids = []
all_metadatas = []

for doc_name, content in documents.items():
    chunks = recursive_chunk(content, chunk_size=400, chunk_overlap=50)
    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        all_ids.append(f"{doc_name}::chunk_{i}")
        all_metadatas.append({
            "source": doc_name,
            "chunk_index": i,
        })

print(f"📚 Total chunks: {len(all_chunks)} from {len(documents)} documents\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: BM25 Keyword Search — How It Works
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 1: BM25 Keyword Search — Finding Exact Words")
print("=" * 70)
print()
print("BM25 (Best Matching 25) is a KEYWORD search algorithm.")
print("It works like Google before AI — counting how often words appear.")
print()
print("How BM25 scores a document for a query:")
print("  1. Term Frequency (TF): How often does the query word appear?")
print("     More occurrences → higher score")
print("  2. Inverse Document Frequency (IDF): Is the word rare or common?")
print("     Rare words ('asyncio') → high value")
print("     Common words ('the', 'is') → low value")
print("  3. Document Length: Longer docs get penalized slightly")
print("     (prevents long documents from always winning)")
print()

# ── Create BM25 index ───────────────────────────────────────────────────────

# Step 1: Tokenize each chunk (split into words, lowercase)
tokenized_chunks = [chunk.lower().split() for chunk in all_chunks]

# Step 2: Build BM25 index
bm25 = BM25Okapi(tokenized_chunks)

print("✅ BM25 index built!")
print(f"   Vocabulary covers {len(all_chunks)} chunks")
print(f"   Average chunk length: {np.mean([len(t) for t in tokenized_chunks]):.0f} words")
print()

# ── Demo: BM25 search ───────────────────────────────────────────────────────

def bm25_search(query, top_k=5):
    """Search using BM25 keyword matching."""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    # Get top-k indices sorted by score (descending)
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "chunk": all_chunks[idx],
            "id": all_ids[idx],
            "source": all_metadatas[idx]["source"],
            "score": float(scores[idx]),
        })
    return results


print("── BM25 Demo: Searching for specific terms ──\n")

demo_queries_bm25 = [
    "Python decorators",
    "async await asyncio",
    "REST API HTTP methods",
]

for query in demo_queries_bm25:
    results = bm25_search(query, top_k=3)
    print(f"🔍 Query: \"{query}\"")
    for i, r in enumerate(results):
        # Normalize BM25 score for emoji (different scale than cosine)
        emoji = "🟢" if r["score"] > 5 else "🟡" if r["score"] > 2 else "🔴"
        print(f"   {i+1}. {emoji} [{r['source']}] score={r['score']:.2f}")
        preview = r["chunk"][:80].replace("\n", " ")
        print(f"      \"{preview}...\"")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: Vector Search (ChromaDB) — Quick Refresher
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 2: Vector Search (ChromaDB) — Semantic Meaning")
print("=" * 70)
print()
print("Vector search finds MEANING, not just words.")
print("  'automobile' → matches 'car' (same meaning!)")
print("  'ML' → matches 'machine learning' (abbreviation!)")
print("BM25 would MISS these because the words are different.")
print()

# ── Build ChromaDB collection ───────────────────────────────────────────────

client = chromadb.Client()
collection = client.create_collection(name="hybrid_search_kb")

collection.add(
    documents=all_chunks,
    ids=all_ids,
    metadatas=all_metadatas,
)
print(f"✅ ChromaDB ready: {collection.count()} chunks indexed\n")


def vector_search(query, top_k=5):
    """Search using ChromaDB vector similarity."""
    results = collection.query(query_texts=[query], n_results=top_k)
    
    output = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        similarity = distance_to_similarity(distance)
        output.append({
            "chunk": results["documents"][0][i],
            "id": results["ids"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "score": similarity,
        })
    return output


print("── Vector Demo: Same queries for comparison ──\n")

for query in demo_queries_bm25:
    results = vector_search(query, top_k=3)
    print(f"🔍 Query: \"{query}\"")
    for i, r in enumerate(results):
        emoji = similarity_emoji(r["score"])
        print(f"   {i+1}. {emoji} [{r['source']}] sim={r['score']:.3f}")
        preview = r["chunk"][:80].replace("\n", " ")
        print(f"      \"{preview}...\"")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: Reciprocal Rank Fusion (RRF) — The Magic Merger
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 3: Reciprocal Rank Fusion (RRF) — Merging Results")
print("=" * 70)
print()
print("We have two ranked lists. How to combine them?")
print()
print("BAD idea: Average the scores")
print("  ❌ BM25 scores range 0-20+, cosine similarity 0-1")
print("  ❌ Scales are incompatible, averaging is meaningless")
print()
print("GOOD idea: Reciprocal Rank Fusion (RRF)")
print("  ✅ Only uses RANK positions, ignores raw scores")
print("  ✅ Formula: RRF_score = Σ (1 / (k + rank))")
print("  ✅ k=60 is standard (prevents top result from dominating)")
print()
print("Example:")
print("  Doc X is rank 1 in vector, rank 3 in BM25:")
print("  RRF = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323")
print()
print("  Doc Y is rank 2 in both:")
print("  RRF = 1/(60+2) + 1/(60+2) = 0.0161 + 0.0161 = 0.0323")
print()
print("  Both get similar RRF scores because Y is consistently good!")
print()


def reciprocal_rank_fusion(result_lists, k=60):
    """
    Reciprocal Rank Fusion — merge multiple ranked result lists.
    
    For each document, RRF score = sum(1 / (k + rank)) across all lists.
    k=60 is the standard constant from the original paper.
    
    Args:
        result_lists: List of result lists, each sorted by relevance
                      Each result must have an 'id' field
        k: Smoothing constant (default 60)
    
    Returns:
        Merged results sorted by RRF score (highest first)
    """
    rrf_scores = {}  # id → total RRF score
    doc_data = {}    # id → document data (chunk, source, etc.)
    
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            doc_id = result["id"]
            rrf_score = 1.0 / (k + rank)
            
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
                doc_data[doc_id] = result
            rrf_scores[doc_id] += rrf_score
    
    # Sort by RRF score (descending)
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    merged = []
    for doc_id in sorted_ids:
        entry = doc_data[doc_id].copy()
        entry["rrf_score"] = rrf_scores[doc_id]
        merged.append(entry)
    
    return merged


# ── Demo: RRF in action ─────────────────────────────────────────────────────

print("── RRF Demo ──\n")

demo_query = "How does Python handle errors and exceptions?"

vector_results = vector_search(demo_query, top_k=5)
bm25_results = bm25_search(demo_query, top_k=5)

print(f"🔍 Query: \"{demo_query}\"\n")

print("📊 Vector Search Ranking:")
for i, r in enumerate(vector_results):
    preview = r["chunk"][:60].replace("\n", " ")
    print(f"   Rank {i+1}: [{r['source']}] sim={r['score']:.3f}  \"{preview}...\"")

print()
print("📊 BM25 Keyword Ranking:")
for i, r in enumerate(bm25_results):
    preview = r["chunk"][:60].replace("\n", " ")
    print(f"   Rank {i+1}: [{r['source']}] score={r['score']:.2f}  \"{preview}...\"")

print()
hybrid_results = reciprocal_rank_fusion([vector_results, bm25_results])

print("📊 Hybrid (RRF) Merged Ranking:")
for i, r in enumerate(hybrid_results[:5]):
    preview = r["chunk"][:60].replace("\n", " ")
    # Show which search methods found this result
    in_vector = any(v["id"] == r["id"] for v in vector_results[:5])
    in_bm25 = any(b["id"] == r["id"] for b in bm25_results[:5])
    sources = []
    if in_vector: sources.append("Vector")
    if in_bm25:   sources.append("BM25")
    found_in = " + ".join(sources)
    print(f"   Rank {i+1}: [{r['source']}] RRF={r['rrf_score']:.4f} ({found_in})")
    print(f"           \"{preview}...\"")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: Head-to-Head Comparison — Where Each Method Wins
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 4: Head-to-Head — Vector vs BM25 vs Hybrid")
print("=" * 70)
print()
print("Let's test queries where each method has an advantage:\n")


def hybrid_search(query, top_k=5):
    """Full hybrid search: Vector + BM25 + RRF."""
    v_results = vector_search(query, top_k=top_k)
    b_results = bm25_search(query, top_k=top_k)
    return reciprocal_rank_fusion([v_results, b_results])[:top_k]


# Test cases designed to show strengths/weaknesses
test_cases = [
    {
        "query": "What is pip and how do I install packages?",
        "why": "KEYWORD query — 'pip' is a specific term BM25 should catch",
    },
    {
        "query": "How do computers learn from data without being told what to do?",
        "why": "SEMANTIC query — no ML jargon, vector search shines",
    },
    {
        "query": "Python asyncio concurrent programming with async await",
        "why": "MIXED query — has both specific terms AND meaning",
    },
    {
        "query": "JWT OAuth authentication security",
        "why": "KEYWORD query — specific tech terms",
    },
    {
        "query": "How to make AI give better answers using external knowledge?",
        "why": "SEMANTIC query — describes RAG without using the word 'RAG'",
    },
]

for tc in test_cases:
    query = tc["query"]
    print(f"🔍 \"{query}\"")
    print(f"   💡 {tc['why']}")
    print()
    
    v_top = vector_search(query, top_k=3)
    b_top = bm25_search(query, top_k=3)
    h_top = hybrid_search(query, top_k=3)
    
    # Show #1 result from each method
    print(f"   Vector #1: [{v_top[0]['source']}] sim={v_top[0]['score']:.3f}")
    preview_v = v_top[0]["chunk"][:70].replace("\n", " ")
    print(f"              \"{preview_v}...\"")
    
    print(f"   BM25   #1: [{b_top[0]['source']}] score={b_top[0]['score']:.2f}")
    preview_b = b_top[0]["chunk"][:70].replace("\n", " ")
    print(f"              \"{preview_b}...\"")
    
    print(f"   Hybrid #1: [{h_top[0]['source']}] RRF={h_top[0]['rrf_score']:.4f}")
    preview_h = h_top[0]["chunk"][:70].replace("\n", " ")
    print(f"              \"{preview_h}...\"")
    
    # Check if hybrid found something different
    v_id = v_top[0]["id"]
    b_id = b_top[0]["id"]
    h_id = h_top[0]["id"]
    
    if h_id == v_id and h_id == b_id:
        print(f"   ✅ All three agree!")
    elif h_id == v_id:
        print(f"   📊 Hybrid sided with Vector search")
    elif h_id == b_id:
        print(f"   📊 Hybrid sided with BM25 keyword search")
    else:
        print(f"   🔀 Hybrid found a unique best result!")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: Full Hybrid RAG Pipeline with Azure OpenAI
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 5: Full Hybrid RAG Pipeline + LLM")
print("=" * 70)
print()

# ── Setup Azure OpenAI (same as Step 5) ─────────────────────────────────────

from openai import OpenAI

load_dotenv()

azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
model_name = os.environ.get("AZURE_OPENAI_MODEL", "gpt-4.1-mini")

# Azure OpenAI endpoint format: strip /chat/completions if present
base_url = azure_endpoint.replace("/chat/completions", "")

llm_client = OpenAI(base_url=base_url, api_key=azure_api_key)

print(f"🤖 LLM: {model_name}")
print(f"🔗 Endpoint: {base_url[:50]}...")
print()


def call_llm(prompt, system_msg="You are a helpful assistant."):
    """Call Azure OpenAI LLM."""
    response = llm_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=500,
    )
    return response.choices[0].message.content


# ── Compare: Vector-only RAG vs Hybrid RAG ──────────────────────────────────

def rag_query(query, search_method="hybrid"):
    """
    Full RAG pipeline with selectable search method.
    
    Args:
        query: User's question
        search_method: 'vector', 'bm25', or 'hybrid'
    
    Returns:
        dict with answer, sources, and search details
    """
    # Step 1: Retrieve context using chosen method
    if search_method == "vector":
        results = vector_search(query, top_k=3)
    elif search_method == "bm25":
        results = bm25_search(query, top_k=3)
    else:  # hybrid
        results = hybrid_search(query, top_k=3)
    
    # Step 2: Build context from retrieved chunks
    context_parts = []
    sources = []
    for i, r in enumerate(results):
        context_parts.append(f"[Source {i+1}: {r['source']}]\n{r['chunk']}")
        sources.append(r["source"])
    
    context = "\n\n".join(context_parts)
    
    # Step 3: Build RAG prompt
    system_msg = """You are a knowledgeable assistant. Answer questions using ONLY 
the provided context. If the context doesn't contain the answer, say 
"I don't have enough information to answer that."
Always mention which source(s) you used."""
    
    prompt = f"""Context from knowledge base:
---
{context}
---

Question: {query}

Answer based on the context above:"""
    
    # Step 4: Generate answer
    answer = call_llm(prompt, system_msg)
    
    return {
        "answer": answer,
        "sources": list(set(sources)),
        "method": search_method,
        "chunks_used": len(results),
    }


# ── Side-by-side comparison ─────────────────────────────────────────────────

comparison_queries = [
    "What is pip and how do I use virtual environments in Python?",
    "How do AI systems learn to give better answers using external documents?",
    "What are async, await, and decorators in Python?",
]

for query in comparison_queries:
    print(f"{'─' * 70}")
    print(f"🔍 Question: \"{query}\"")
    print(f"{'─' * 70}\n")
    
    for method in ["vector", "bm25", "hybrid"]:
        start = time.time()
        result = rag_query(query, search_method=method)
        elapsed = time.time() - start
        
        label = {"vector": "🟦 VECTOR", "bm25": "🟨 BM25", "hybrid": "🟩 HYBRID"}
        print(f"{label[method]} RAG ({elapsed:.1f}s):")
        print(f"   Sources: {', '.join(result['sources'])}")
        # Show first 200 chars of answer
        answer_preview = result["answer"][:200].replace("\n", " ")
        print(f"   Answer: {answer_preview}...")
        print()
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 6: Architecture Diagram
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 6: Hybrid Search Architecture")
print("=" * 70)
print("""

  HYBRID RAG ARCHITECTURE
  ════════════════════════

  User Query: "How does Python handle async programming?"
       │
       ├──────────────────┐
       │                  │
       ▼                  ▼
  ┌──────────┐     ┌──────────┐
  │  Vector  │     │   BM25   │
  │  Search  │     │  Keyword │
  │ (ChromaDB)│    │  Search  │
  └────┬─────┘     └────┬─────┘
       │                 │
       │ Rank by         │ Rank by
       │ meaning         │ word match
       │                 │
       ▼                 ▼
  ┌─────────────────────────┐
  │  Reciprocal Rank Fusion │
  │                         │
  │  RRF = Σ 1/(k + rank)  │
  │  k = 60                 │
  │                         │
  │  Combines BOTH rankings │
  │  into one merged list   │
  └───────────┬─────────────┘
              │
              ▼ Top-3 chunks
  ┌─────────────────────────┐
  │    Build RAG Prompt     │
  │                         │
  │  Context: [chunk1]      │
  │           [chunk2]      │
  │           [chunk3]      │
  │                         │
  │  Question: user query   │
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │   Azure OpenAI LLM     │
  │   (gpt-4.1-mini)       │
  │                         │
  │   Generates answer      │
  │   using retrieved       │
  │   context only          │
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │   Answer + Sources      │
  │                         │
  │   "Python uses asyncio  │
  │   module with async and │
  │   await keywords..."    │
  │                         │
  │   Source: python_guide   │
  └─────────────────────────┘
""")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 7: Summary — When to Use What
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 7: Summary — When to Use What")
print("=" * 70)
print("""
  ┌──────────────┬────────────────────┬─────────────────────┐
  │   Method     │   Best For         │   Weakness          │
  ├──────────────┼────────────────────┼─────────────────────┤
  │ Vector Only  │ Semantic meaning   │ Misses exact terms  │
  │              │ Paraphrased queries│ "pip" → may miss    │
  │              │ "How to learn?"    │                     │
  ├──────────────┼────────────────────┼─────────────────────┤
  │ BM25 Only   │ Exact keywords     │ No understanding    │
  │              │ Technical terms    │ "automobile" ≠ "car"│
  │              │ "asyncio decorator"│                     │
  ├──────────────┼────────────────────┼─────────────────────┤
  │ Hybrid (RRF)│ BOTH!              │ Slightly slower     │
  │              │ Best of both worlds│ (2 searches)        │
  │              │ Production choice  │                     │
  └──────────────┴────────────────────┴─────────────────────┘

  INDUSTRY STANDARD:
  • Elasticsearch + vector = hybrid search
  • Pinecone has built-in hybrid mode
  • Most production RAG uses hybrid search
  • RRF with k=60 is the most common fusion method

  NEXT STEP → Step 7: Re-ranking
    Use a cross-encoder to RE-SCORE results
    More accurate than embeddings but slower
    Applied AFTER retrieval (vector or hybrid)
""")

print("✅ Step 6 complete! You now understand Hybrid Search.\n")
