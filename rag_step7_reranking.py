"""
Step 7: Re-ranking — The Precision Booster

Building on Step 6 (Hybrid Search), we add a RE-RANKING stage.

THE PROBLEM with retrieval (vector or hybrid):
  Step 6 retrieves the top-K chunks — but the ORDER might be wrong!
  
  Why? Bi-encoders (what we've been using) encode query and document
  SEPARATELY, then compare vectors. Fast, but approximate.

THE SOLUTION — Cross-Encoder Re-ranking:
  A cross-encoder takes BOTH query AND document TOGETHER as input,
  and outputs a single relevance score. Much more accurate!

  ┌─────────────────────────────────────────────────────────┐
  │  Bi-encoder (Step 1-6):        Cross-encoder (Step 7):  │
  │                                                         │
  │  query → [vector]              query + doc → [score]    │
  │  doc   → [vector]              (processes TOGETHER)     │
  │  compare vectors               direct relevance score   │
  │                                                         │
  │  ✅ Fast (encode once)         ✅ Very accurate          │
  │  ❌ Less precise               ❌ Slow (can't pre-index) │
  └─────────────────────────────────────────────────────────┘

  That's why we use BOTH:
    1. Retrieval (fast): Get top-20 candidates with hybrid search
    2. Re-rank (accurate): Score those 20 with cross-encoder
    3. Return top-3 after re-ranking

This file demonstrates:
  Part 1: What is a cross-encoder and how it differs from bi-encoder
  Part 2: Re-ranking retrieved results
  Part 3: Before vs After — measuring re-ranking improvement
  Part 4: Full pipeline: Hybrid Search → Re-rank → LLM
  Part 5: Speed vs Accuracy tradeoff analysis
"""

import os
import time
import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from rag_utils import (
    recursive_chunk, distance_to_similarity, similarity_emoji,
    PYTHON_DOC, AI_DOC, WEBDEV_DOC,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SETUP: Build knowledge base (reused from Step 6)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("SETUP: Building Knowledge Base + Search Indexes")
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

print(f"📚 {len(all_chunks)} chunks from {len(documents)} documents")

# ── BM25 index ──────────────────────────────────────────────────────────────

tokenized_chunks = [chunk.lower().split() for chunk in all_chunks]
bm25 = BM25Okapi(tokenized_chunks)
print("✅ BM25 index ready")

# ── ChromaDB index ──────────────────────────────────────────────────────────

client = chromadb.Client()
collection = client.create_collection(name="reranking_kb")
collection.add(documents=all_chunks, ids=all_ids, metadatas=all_metadatas)
print(f"✅ ChromaDB ready: {collection.count()} chunks")

# ── Search functions (from Step 6) ──────────────────────────────────────────

def bm25_search(query, top_k=5):
    """BM25 keyword search."""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [{
        "chunk": all_chunks[idx],
        "id": all_ids[idx],
        "source": all_metadatas[idx]["source"],
        "score": float(scores[idx]),
    } for idx in top_indices]


def vector_search(query, top_k=5):
    """ChromaDB vector search."""
    results = collection.query(query_texts=[query], n_results=top_k)
    return [{
        "chunk": results["documents"][0][i],
        "id": results["ids"][0][i],
        "source": results["metadatas"][0][i]["source"],
        "score": distance_to_similarity(results["distances"][0][i]),
    } for i in range(len(results["ids"][0]))]


def reciprocal_rank_fusion(result_lists, k=60):
    """Merge multiple ranked lists using RRF."""
    rrf_scores = {}
    doc_data = {}
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            doc_id = result["id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
                doc_data[doc_id] = result
            rrf_scores[doc_id] += 1.0 / (k + rank)
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    merged = []
    for doc_id in sorted_ids:
        entry = doc_data[doc_id].copy()
        entry["rrf_score"] = rrf_scores[doc_id]
        merged.append(entry)
    return merged


def hybrid_search(query, top_k=5):
    """Hybrid search: Vector + BM25 + RRF."""
    v = vector_search(query, top_k=top_k)
    b = bm25_search(query, top_k=top_k)
    return reciprocal_rank_fusion([v, b])[:top_k]


print("✅ Search functions ready\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: Cross-Encoder — What Is It?
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 1: Cross-Encoder vs Bi-Encoder — The Key Difference")
print("=" * 70)
print("""
  BI-ENCODER (what we've used so far):
  ════════════════════════════════════
  
  Query: "Python error handling"     Doc: "try-except blocks..."
       │                                  │
       ▼                                  ▼
  ┌──────────┐                      ┌──────────┐
  │ Encoder  │                      │ Encoder  │
  │ (same    │                      │ (same    │
  │  model)  │                      │  model)  │
  └────┬─────┘                      └────┬─────┘
       │                                  │
       ▼                                  ▼
  [0.2, -0.1, ...]                  [0.3, -0.2, ...]
       │                                  │
       └──────── cosine sim ──────────────┘
                    │
                 0.85 ← similarity score
  
  ✅ Fast: encode docs ONCE, reuse forever
  ❌ Query and doc never "see" each other


  CROSS-ENCODER (new in this step):
  ═════════════════════════════════
  
  Query + Doc → TOGETHER as one input
       │
       ▼
  ┌─────────────────────────────────────┐
  │  "Python error handling [SEP]       │
  │   try-except blocks catch errors    │
  │   in Python programs..."            │
  │                                     │
  │  Transformer sees BOTH at once      │
  │  Can do word-level cross-attention  │
  └──────────────┬──────────────────────┘
                 │
                 ▼
              0.92 ← relevance score (more accurate!)
  
  ✅ Sees query + doc together → understands context
  ❌ Slow: must re-encode for every (query, doc) pair
  ❌ Cannot pre-index documents
""")

# ── Load cross-encoder model ────────────────────────────────────────────────

print("Loading cross-encoder model...")
print("(Model: ms-marco-MiniLM-L-6-v2 — trained on 500K real search queries)\n")

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("✅ Cross-encoder loaded!\n")

# ── Demo: Cross-encoder scoring ─────────────────────────────────────────────

print("── Demo: Cross-encoder scores (query, document) pairs ──\n")

demo_query = "How does Python handle errors?"

demo_docs = [
    "Error handling in Python uses try-except blocks. You can catch specific exceptions like ValueError or TypeError.",
    "Python is a high-level programming language created by Guido van Rossum.",
    "React is a popular JavaScript library for building user interfaces.",
    "The finally block runs cleanup code regardless of whether an exception occurred.",
]

print(f"🔍 Query: \"{demo_query}\"\n")

# Cross-encoder takes pairs of (query, document)
pairs = [(demo_query, doc) for doc in demo_docs]
scores = cross_encoder.predict(pairs)

# Sort by score
ranked = sorted(zip(scores, demo_docs), reverse=True)

for i, (score, doc) in enumerate(ranked):
    emoji = "🟢" if score > 3 else "🟡" if score > 0 else "🔴"
    preview = doc[:80]
    print(f"  {i+1}. {emoji} score={score:+.2f}  \"{preview}...\"")

print()
print("Notice how the cross-encoder:")
print("  🟢 Scores error-handling docs HIGH (directly relevant)")
print("  🔴 Scores Python-general doc LOW (same topic, wrong content)")
print("  🔴 Scores React doc VERY LOW (completely irrelevant)")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: Re-ranking Retrieved Results
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 2: Re-ranking — Fixing Retrieval Order")
print("=" * 70)
print()
print("Strategy: Retrieve MANY (top-10) → Re-rank → Keep BEST (top-3)")
print("  This is called the 'retrieve-then-rerank' pattern.")
print()


def rerank(query, results, top_k=3):
    """
    Re-rank retrieved results using cross-encoder.
    
    Takes pre-retrieved results and re-scores them with the
    more accurate cross-encoder model.
    
    Args:
        query: The user's question
        results: List of retrieved chunks (from hybrid/vector/bm25 search)
        top_k: How many to return after re-ranking
    
    Returns:
        Re-ranked results with cross-encoder scores
    """
    if not results:
        return []
    
    # Create (query, chunk) pairs for cross-encoder
    pairs = [(query, r["chunk"]) for r in results]
    
    # Score all pairs
    ce_scores = cross_encoder.predict(pairs)
    
    # Attach scores and sort
    for i, result in enumerate(results):
        result["ce_score"] = float(ce_scores[i])
    
    # Sort by cross-encoder score (descending)
    reranked = sorted(results, key=lambda x: x["ce_score"], reverse=True)
    
    return reranked[:top_k]


# ── Demo: Before vs After re-ranking ────────────────────────────────────────

print("── Demo: Hybrid Search → Re-rank ──\n")

test_queries = [
    "What is the difference between supervised and unsupervised learning?",
    "How do Python decorators work and when should I use them?",
    "What is REST API and how does authentication work?",
]

for query in test_queries:
    print(f"🔍 \"{query}\"\n")
    
    # Get more candidates than we need (top-10)
    hybrid_results = hybrid_search(query, top_k=10)
    
    # Show top-3 BEFORE re-ranking
    print("  📋 BEFORE re-ranking (hybrid search order):")
    for i, r in enumerate(hybrid_results[:3]):
        preview = r["chunk"][:70].replace("\n", " ")
        print(f"     {i+1}. [{r['source']}] RRF={r['rrf_score']:.4f}")
        print(f"        \"{preview}...\"")
    
    # Re-rank
    reranked = rerank(query, hybrid_results, top_k=3)
    
    print()
    print("  📋 AFTER re-ranking (cross-encoder order):")
    for i, r in enumerate(reranked):
        preview = r["chunk"][:70].replace("\n", " ")
        # Find original rank
        orig_rank = next(
            j+1 for j, h in enumerate(hybrid_results) if h["id"] == r["id"]
        )
        moved = f"was #{orig_rank}" if orig_rank != i+1 else "same"
        print(f"     {i+1}. [{r['source']}] CE={r['ce_score']:+.2f} ({moved})")
        print(f"        \"{preview}...\"")
    
    # Check if order changed
    before_ids = [r["id"] for r in hybrid_results[:3]]
    after_ids = [r["id"] for r in reranked]
    if before_ids == after_ids:
        print(f"\n  ✅ Same order — hybrid search was already right!")
    else:
        print(f"\n  🔄 Order CHANGED — cross-encoder found a better ranking!")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: Measuring Improvement — Does Re-ranking Help?
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 3: Measuring Improvement — Relevance Scores")
print("=" * 70)
print()
print("Let's see how cross-encoder scores the top-3 from each method.\n")

eval_queries = [
    {
        "query": "How does transfer learning reduce training cost?",
        "expected_topic": "ai_ml_guide",
    },
    {
        "query": "What are Python's special dunder methods like __init__ and __str__?",
        "expected_topic": "python_guide",
    },
    {
        "query": "How does React virtual DOM make websites fast?",
        "expected_topic": "webdev_guide",
    },
    {
        "query": "What is RAG and how does it reduce hallucinations?",
        "expected_topic": "ai_ml_guide",
    },
    {
        "query": "How does CSS Flexbox help with responsive layouts?",
        "expected_topic": "webdev_guide",
    },
]

# Track scores across methods
method_scores = {"vector": [], "bm25": [], "hybrid": [], "hybrid+rerank": []}

for eq in eval_queries:
    query = eq["query"]
    expected = eq["expected_topic"]
    
    print(f"🔍 \"{query}\"")
    print(f"   Expected source: {expected}\n")
    
    # Get results from each method
    v_results = vector_search(query, top_k=10)
    b_results = bm25_search(query, top_k=10)
    h_results = hybrid_search(query, top_k=10)
    r_results = rerank(query, h_results.copy(), top_k=3)
    
    # Score top-1 from each method with cross-encoder
    methods = {
        "vector":         v_results[:3],
        "bm25":           b_results[:3],
        "hybrid":         h_results[:3],
        "hybrid+rerank":  r_results,
    }
    
    for name, results in methods.items():
        # Get cross-encoder scores for top-3
        if results:
            pairs = [(query, r["chunk"]) for r in results]
            ce_scores = cross_encoder.predict(pairs)
            avg_score = float(np.mean(ce_scores))
            top_source = results[0]["source"]
            correct = "✓" if top_source == expected else "✗"
            method_scores[name].append(avg_score)
            
            label = f"{name:>15}"
            print(f"   {label}: avg_CE={avg_score:+.2f}  top=[{top_source}] {correct}")
    
    print()

# ── Summary table ───────────────────────────────────────────────────────────

print("── Average Cross-Encoder Scores (higher = better) ──\n")
print(f"  {'Method':<20} {'Avg CE Score':>12} {'Improvement':>12}")
print(f"  {'─' * 44}")

baseline = np.mean(method_scores["vector"])
for name in ["vector", "bm25", "hybrid", "hybrid+rerank"]:
    avg = np.mean(method_scores[name])
    improvement = ((avg - baseline) / abs(baseline)) * 100 if baseline != 0 else 0
    bar = "█" * int(max(0, (avg + 5)) * 2)  # visual bar
    marker = " ← BEST" if name == "hybrid+rerank" else ""
    print(f"  {name:<20} {avg:>+10.2f}   {improvement:>+8.1f}%  {bar}{marker}")

print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: Full Pipeline — Hybrid + Rerank + LLM
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 4: Full Pipeline — Hybrid → Re-rank → LLM")
print("=" * 70)
print()

# ── Setup Azure OpenAI ──────────────────────────────────────────────────────

from openai import OpenAI

load_dotenv()

azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
model_name = os.environ.get("AZURE_OPENAI_MODEL", "gpt-4.1-mini")

base_url = azure_endpoint.replace("/chat/completions", "")
llm_client = OpenAI(base_url=base_url, api_key=azure_api_key)

print(f"🤖 LLM: {model_name}")
print(f"🔗 Endpoint: {base_url[:50]}...\n")


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


def rag_query(query, use_reranking=True):
    """
    Full RAG pipeline: Hybrid Search → (optional) Re-rank → LLM.
    
    Args:
        query: User's question
        use_reranking: Whether to re-rank with cross-encoder
    
    Returns:
        dict with answer, sources, and timing info
    """
    t_start = time.time()
    
    # Step 1: Retrieve candidates (get more if re-ranking)
    retrieve_k = 10 if use_reranking else 3
    candidates = hybrid_search(query, top_k=retrieve_k)
    t_retrieve = time.time()
    
    # Step 2: Re-rank (optional)
    if use_reranking:
        results = rerank(query, candidates, top_k=3)
        t_rerank = time.time()
    else:
        results = candidates[:3]
        t_rerank = t_retrieve  # no reranking time
    
    # Step 3: Build context
    context_parts = []
    sources = []
    for i, r in enumerate(results):
        ce_info = f" (relevance: {r['ce_score']:+.1f})" if "ce_score" in r else ""
        context_parts.append(f"[Source {i+1}: {r['source']}{ce_info}]\n{r['chunk']}")
        sources.append(r["source"])
    context = "\n\n".join(context_parts)
    
    # Step 4: Generate answer
    system_msg = """You are a knowledgeable assistant. Answer using ONLY the provided context.
If the context doesn't have the answer, say "I don't have enough information."
Mention which source(s) you used."""
    
    prompt = f"""Context:
---
{context}
---

Question: {query}

Answer:"""
    
    answer = call_llm(prompt, system_msg)
    t_end = time.time()
    
    return {
        "answer": answer,
        "sources": list(set(sources)),
        "reranked": use_reranking,
        "timing": {
            "retrieve": t_retrieve - t_start,
            "rerank": t_rerank - t_retrieve,
            "llm": t_end - t_rerank,
            "total": t_end - t_start,
        },
    }


# ── Side-by-side: With vs Without Re-ranking ────────────────────────────────

comparison_queries = [
    "What is transfer learning and why is it useful?",
    "How do Python decorators and the dunder model relate?",
    "What is the difference between REST APIs and WebSockets?",
]

for query in comparison_queries:
    print(f"{'─' * 70}")
    print(f"🔍 \"{query}\"")
    print(f"{'─' * 70}\n")
    
    # Without re-ranking
    result_no = rag_query(query, use_reranking=False)
    print(f"🟦 WITHOUT Re-ranking ({result_no['timing']['total']:.1f}s):")
    print(f"   Sources: {', '.join(result_no['sources'])}")
    preview = result_no["answer"][:200].replace("\n", " ")
    print(f"   Answer: {preview}...")
    print(f"   ⏱ Retrieve: {result_no['timing']['retrieve']:.3f}s | LLM: {result_no['timing']['llm']:.1f}s")
    print()
    
    # With re-ranking
    result_yes = rag_query(query, use_reranking=True)
    print(f"🟩 WITH Re-ranking ({result_yes['timing']['total']:.1f}s):")
    print(f"   Sources: {', '.join(result_yes['sources'])}")
    preview = result_yes["answer"][:200].replace("\n", " ")
    print(f"   Answer: {preview}...")
    print(f"   ⏱ Retrieve: {result_yes['timing']['retrieve']:.3f}s | Rerank: {result_yes['timing']['rerank']:.3f}s | LLM: {result_yes['timing']['llm']:.1f}s")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: Speed vs Accuracy Tradeoff
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 5: Speed vs Accuracy — When to Re-rank")
print("=" * 70)
print()

# Time each stage
print("── Timing Benchmark ──\n")

bench_query = "How does Python handle asynchronous programming?"

# Time retrieval methods
t0 = time.time()
_ = vector_search(bench_query, top_k=10)
t_vector = time.time() - t0

t0 = time.time()
_ = bm25_search(bench_query, top_k=10)
t_bm25 = time.time() - t0

t0 = time.time()
h_results = hybrid_search(bench_query, top_k=10)
t_hybrid = time.time() - t0

t0 = time.time()
_ = rerank(bench_query, h_results, top_k=3)
t_rerank = time.time() - t0

print(f"  Vector search (10 results):   {t_vector*1000:>8.1f} ms")
print(f"  BM25 search (10 results):     {t_bm25*1000:>8.1f} ms")
print(f"  Hybrid search (10 results):   {t_hybrid*1000:>8.1f} ms")
print(f"  Cross-encoder re-rank (10→3): {t_rerank*1000:>8.1f} ms")
print()

print(f"  Re-ranking adds ~{t_rerank*1000:.0f}ms — worth it for better accuracy!")
print()

print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │              WHEN TO USE RE-RANKING                            │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                 │
  │  ✅ USE re-ranking when:                                        │
  │     • Accuracy matters more than speed                          │
  │     • User is waiting for a response (chatbot, search)          │
  │     • You retrieve < 50 candidates to re-rank                   │
  │     • Domain-specific queries need precision                    │
  │                                                                 │
  │  ❌ SKIP re-ranking when:                                       │
  │     • Latency is critical (< 100ms requirement)                 │
  │     • Re-ranking hundreds/thousands of documents                │
  │     • Simple/broad queries where top results are obvious        │
  │     • Batch processing where speed matters more                 │
  │                                                                 │
  │  PRODUCTION PATTERN:                                            │
  │     Retrieve 20-50 → Re-rank → Keep top 3-5                    │
  │     Re-ranking 20 docs ≈ 50-100ms (totally acceptable)          │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
""")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 6: Architecture — The Complete RAG Pipeline So Far
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 6: Complete RAG Pipeline Architecture")
print("=" * 70)
print("""
  THE FULL RAG PIPELINE (Steps 1-7)
  ═════════════════════════════════

  ┌─────────────────────────────────────────────────┐
  │               INDEXING (one-time)                │
  ├─────────────────────────────────────────────────┤
  │                                                  │
  │  Documents → Chunk (Step 4) → Embed → ChromaDB  │
  │                                → Tokenize → BM25 │
  │                                                  │
  └──────────────────────┬──────────────────────────┘
                         │ pre-built indexes
                         ▼
  ┌─────────────────────────────────────────────────┐
  │            QUERY TIME (per question)             │
  ├─────────────────────────────────────────────────┤
  │                                                  │
  │  User Query                                      │
  │       │                                          │
  │       ├──── Vector Search ──── top 10 ──┐        │
  │       │     (Step 2-3)                  │        │
  │       │                                 ▼        │
  │       ├──── BM25 Search ─────  top 10 ──┤        │
  │       │     (Step 6)                    │        │
  │       │                                 ▼        │
  │       │                          RRF Fusion      │
  │       │                          (Step 6)        │
  │       │                              │           │
  │       │                         top 10 merged    │
  │       │                              │           │
  │       │                              ▼           │
  │       │                      ┌──────────────┐    │
  │       │                      │ Cross-Encoder │   │
  │       │                      │  Re-ranking   │   │
  │       │                      │  (Step 7) ←── NEW │
  │       │                      └──────┬───────┘    │
  │       │                             │            │
  │       │                        top 3 best        │
  │       │                             │            │
  │       │                             ▼            │
  │       │                      Build RAG Prompt    │
  │       │                      (Step 5)            │
  │       │                             │            │
  │       │                             ▼            │
  │       │                      Azure OpenAI LLM    │
  │       │                      (Step 5)            │
  │       │                             │            │
  │       │                             ▼            │
  │       └──────────────────── Answer + Sources     │
  │                                                  │
  └──────────────────────────────────────────────────┘
""")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 7: Summary
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 7: Summary — What We Learned")
print("=" * 70)
print("""
  ┌──────────────────┬──────────────────┬───────────────────┐
  │   Component      │   How It Works   │   When Added      │
  ├──────────────────┼──────────────────┼───────────────────┤
  │ Bi-encoder       │ Encode separately│ Steps 1-3         │
  │ (embeddings)     │ Compare vectors  │ (fast, reusable)  │
  ├──────────────────┼──────────────────┼───────────────────┤
  │ BM25             │ Count keywords   │ Step 6            │
  │                  │ TF-IDF scoring   │ (exact matches)   │
  ├──────────────────┼──────────────────┼───────────────────┤
  │ RRF Fusion       │ Merge rank lists │ Step 6            │
  │                  │ 1/(k + rank)     │ (best of both)    │
  ├──────────────────┼──────────────────┼───────────────────┤
  │ Cross-encoder    │ Score pairs      │ Step 7 (THIS!)    │
  │ (re-ranker)      │ together         │ (precision boost) │
  └──────────────────┴──────────────────┴───────────────────┘

  KEY INSIGHT:
    Retrieval = RECALL (find all relevant docs)
    Re-ranking = PRECISION (put the best ones first)
    You need BOTH for production RAG!

  MODEL USED:
    cross-encoder/ms-marco-MiniLM-L-6-v2
    • Trained on 500K real search queries (MS MARCO dataset)
    • 22MB model — small and fast
    • No GPU needed!

  NEXT STEP → Step 8: Source Citations
    Track exactly which document + passage answered the question
    Show: "Based on python_guide, paragraph 3..."
""")

print("✅ Step 7 complete! You now understand Re-ranking.\n")
