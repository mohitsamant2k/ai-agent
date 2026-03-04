"""
Step 12: Contextual Retrieval — Add Context to Chunks BEFORE Embedding

THE PROBLEM:
  When we chunk a document, each chunk LOSES its context.
  
  Original document: "Python Guide — Error Handling Section"
    Chunk 7: "The finally block runs cleanup code regardless 
              of whether an exception occurred."
  
  This chunk is now ORPHANED — it doesn't say:
    - Which document it came from
    - What section/topic it belongs to
    - What came before it
  
  So when someone searches "Python cleanup after errors",
  chunk 7 might not rank high because it never says "Python" or "error"!

THE FIX (Anthropic's technique):
  Before embedding, prepend a short context to each chunk:
  
  BEFORE: "The finally block runs cleanup code regardless..."
  AFTER:  "From the Python programming guide, section on error handling
           and exceptions: The finally block runs cleanup code regardless..."
  
  Now the chunk is SELF-CONTAINED — it carries its own context.
  Vector search finds it for "Python error cleanup" because
  the words "Python", "error", and "cleanup" are all present!

HOW:
  Use the LLM to read the FULL document + the chunk,
  and generate a 1-2 sentence context prefix.
  This is done at INDEXING TIME (once), not at query time.

This file demonstrates:
  Part 1: The problem — chunks without context fail
  Part 2: Generating context for each chunk (LLM call)
  Part 3: Search comparison — normal vs contextual chunks
  Part 4: Cost analysis and practical considerations
  Part 5: Architecture diagram + Summary
"""

import os
import time
import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from openai import OpenAI
from rag_utils import (
    recursive_chunk, distance_to_similarity,
    PYTHON_DOC, AI_DOC, WEBDEV_DOC,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("SETUP: Preparing Normal + Contextual Pipelines")
print("=" * 70)

# Document metadata (so the LLM knows what each doc is about)
documents = {
    "python_guide": {
        "content": PYTHON_DOC,
        "title": "Python Programming Guide",
        "description": "Comprehensive guide covering Python basics, data types, "
                       "error handling, decorators, async programming, testing, "
                       "and the package ecosystem.",
    },
    "ai_ml_guide": {
        "content": AI_DOC,
        "title": "AI and Machine Learning Guide",
        "description": "Guide covering supervised, unsupervised, and reinforcement "
                       "learning, deep learning, transformers, transfer learning, "
                       "and RAG (Retrieval-Augmented Generation).",
    },
    "webdev_guide": {
        "content": WEBDEV_DOC,
        "title": "Modern Web Development Guide",
        "description": "Guide covering HTML, CSS, JavaScript, React, REST APIs, "
                       "Node.js, responsive design, and web frameworks.",
    },
}

# Azure OpenAI setup
load_dotenv()
azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
model_name = os.environ.get("AZURE_OPENAI_MODEL", "gpt-4.1-mini")
base_url = azure_endpoint.replace("/chat/completions", "")
llm_client = OpenAI(base_url=base_url, api_key=azure_api_key)


def call_llm(prompt, system_msg="You are a helpful assistant.", temperature=0.0,
             max_tokens=150):
    response = llm_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


# Cross-encoder for re-ranking
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print(f"   LLM ready: {model_name}")
print(f"   Cross-encoder ready")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: The Problem — Chunks Without Context
# ═══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("PART 1: The Problem -- Orphaned Chunks")
print("=" * 70)
print()
print("When we chunk documents, each chunk becomes an orphan.")
print("It loses track of WHERE it came from and WHAT it's about.")
print()

# Chunk all docs normally
normal_chunks = []
normal_ids = []
normal_metadatas = []

for doc_name, doc_info in documents.items():
    chunks = recursive_chunk(doc_info["content"], chunk_size=400, chunk_overlap=50)
    for i, chunk in enumerate(chunks):
        normal_chunks.append(chunk)
        normal_ids.append(f"{doc_name}::chunk_{i}")
        normal_metadatas.append({
            "source": doc_name,
            "chunk_index": i,
            "doc_title": doc_info["title"],
        })

print(f"   {len(normal_chunks)} chunks from {len(documents)} documents\n")

# Show examples of orphaned chunks
print("-- Examples of orphaned chunks --\n")
orphan_examples = [
    (5, "Doesn't mention 'Python' at all!"),
    (8, "Says 'Python' but not what section"),
]

for idx, problem in orphan_examples:
    if idx < len(normal_chunks):
        preview = normal_chunks[idx][:150].replace("\n", " ")
        print(f"  Chunk {idx} [{normal_metadatas[idx]['source']}]:")
        print(f"    \"{preview}...\"")
        print(f"    Problem: {problem}")
        print()

# Build normal search pipeline
normal_tokenized = [c.lower().split() for c in normal_chunks]
normal_bm25 = BM25Okapi(normal_tokenized)

chroma_client = chromadb.Client()
normal_collection = chroma_client.create_collection(name="normal_chunks")
normal_collection.add(
    documents=normal_chunks,
    ids=normal_ids,
    metadatas=normal_metadatas,
)

print("   Normal pipeline ready (no context added)\n")


# Show a problematic search
print("-- Problem demo: searching for 'Python cleanup after errors' --\n")

demo_query = "Python cleanup after errors"

# BM25 search
bm25_scores = normal_bm25.get_scores(demo_query.lower().split())
top_bm25 = np.argsort(bm25_scores)[::-1][:3]

print(f"  BM25 results for \"{demo_query}\":")
for rank, idx in enumerate(top_bm25, 1):
    preview = normal_chunks[idx][:80].replace("\n", " ")
    has_python = "Python" in normal_chunks[idx] or "python" in normal_chunks[idx]
    has_finally = "finally" in normal_chunks[idx]
    print(f"    {rank}. [{normal_metadatas[idx]['source']}] "
          f"(score={bm25_scores[idx]:.2f}) "
          f"{'✅ has Python' if has_python else '❌ no Python'} "
          f"{'✅ has finally' if has_finally else ''}")
    print(f"       \"{preview}...\"")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: Generating Context for Each Chunk
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 2: Adding Context to Chunks (Anthropic's Technique)")
print("=" * 70)
print()
print("For each chunk, we ask the LLM:")
print("  'Given the full document and this specific chunk,")
print("   write a 1-2 sentence context that explains where")
print("   this chunk fits in the document.'")
print()
print("This context is PREPENDED to the chunk before embedding.")
print()

CONTEXT_SYSTEM = """You are a document context generator. Given a document and a 
specific chunk from it, write a short context prefix (1-2 sentences) that situates 
the chunk within the full document.

Rules:
1. Mention the document title/topic
2. Mention the specific section or subtopic this chunk covers  
3. Keep it under 50 words
4. Write in a neutral, descriptive tone
5. Output ONLY the context prefix, nothing else
6. Do NOT repeat the chunk content"""

CONTEXT_PROMPT = """Document title: {doc_title}
Document description: {doc_description}

Full document (for reference):
{full_document}

---

Specific chunk to contextualize:
{chunk}

---

Write a 1-2 sentence context prefix for this chunk:"""


def generate_chunk_context(chunk, doc_info):
    """
    Generate a context prefix for a chunk using the LLM.
    
    The LLM sees the FULL document + the specific chunk,
    and writes a brief context that explains where this chunk fits.
    
    Args:
        chunk: The text chunk to contextualize
        doc_info: Dict with 'title', 'description', 'content'
    
    Returns:
        str: Context prefix (1-2 sentences)
    """
    prompt = CONTEXT_PROMPT.format(
        doc_title=doc_info["title"],
        doc_description=doc_info["description"],
        full_document=doc_info["content"][:2000],  # First 2000 chars for context
        chunk=chunk,
    )
    context = call_llm(prompt, CONTEXT_SYSTEM, temperature=0.0, max_tokens=100)
    return context.strip()


def create_contextual_chunk(chunk, context):
    """
    Prepend context to chunk, creating a self-contained passage.
    
    Before: "The finally block runs cleanup code..."
    After:  "From the Python programming guide, section on error handling:
             The finally block runs cleanup code..."
    """
    return f"{context}\n\n{chunk}"


# ── Generate context for all chunks ─────────────────────────────────────────

print("-- Generating context for all chunks (LLM call per chunk) --\n")

contextual_chunks = []
contextual_ids = []
contextual_metadatas = []
contexts_generated = []

chunk_idx = 0
total_chunks = len(normal_chunks)
t_start = time.time()

for doc_name, doc_info in documents.items():
    chunks = recursive_chunk(doc_info["content"], chunk_size=400, chunk_overlap=50)
    
    print(f"  Processing {doc_info['title']} ({len(chunks)} chunks)...")
    
    for i, chunk in enumerate(chunks):
        # Generate context using LLM
        context = generate_chunk_context(chunk, doc_info)
        contexts_generated.append(context)
        
        # Create contextual chunk
        ctx_chunk = create_contextual_chunk(chunk, context)
        
        contextual_chunks.append(ctx_chunk)
        contextual_ids.append(f"{doc_name}::ctx_chunk_{i}")
        contextual_metadatas.append({
            "source": doc_name,
            "chunk_index": i,
            "doc_title": doc_info["title"],
            "context": context,
        })
        
        chunk_idx += 1

t_total = time.time() - t_start
print(f"\n   Generated context for {chunk_idx} chunks in {t_total:.1f}s")
print(f"   Average: {t_total/chunk_idx:.2f}s per chunk\n")

# Show before/after examples
print("-- Before vs After: context added --\n")

show_indices = [0, 5, 10]  # Show first chunk from each doc (roughly)
for idx in show_indices:
    if idx < len(normal_chunks):
        print(f"  Chunk {idx} [{normal_metadatas[idx]['source']}]:")
        
        normal_preview = normal_chunks[idx][:100].replace("\n", " ")
        print(f"    BEFORE: \"{normal_preview}...\"")
        
        ctx = contexts_generated[idx]
        print(f"    CONTEXT: \"{ctx}\"")
        
        ctx_preview = contextual_chunks[idx][:120].replace("\n", " ")
        print(f"    AFTER:  \"{ctx_preview}...\"")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: Search Comparison — Normal vs Contextual
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 3: Search Comparison -- Normal vs Contextual Chunks")
print("=" * 70)
print()

# Build contextual search pipeline
ctx_tokenized = [c.lower().split() for c in contextual_chunks]
ctx_bm25 = BM25Okapi(ctx_tokenized)

ctx_collection = chroma_client.create_collection(name="contextual_chunks")
ctx_collection.add(
    documents=contextual_chunks,
    ids=contextual_ids,
    metadatas=contextual_metadatas,
)

print("   Contextual pipeline ready\n")


# ── Search helpers ──────────────────────────────────────────────────────────

def search_normal(query, top_k=3):
    """Search in normal (no-context) chunks."""
    # Vector search
    v_results = normal_collection.query(query_texts=[query], n_results=top_k)
    # BM25 search
    bm25_scores = normal_bm25.get_scores(query.lower().split())
    top_bm25_idx = np.argsort(bm25_scores)[::-1][:top_k]
    
    # Combine with RRF
    rrf_scores = {}
    doc_data = {}
    
    for rank in range(len(v_results["ids"][0])):
        did = v_results["ids"][0][rank]
        rrf_scores[did] = rrf_scores.get(did, 0) + 1.0 / (60 + rank + 1)
        doc_data[did] = {
            "chunk": v_results["documents"][0][rank],
            "id": did,
            "source": v_results["metadatas"][0][rank]["source"],
        }
    
    for rank, idx in enumerate(top_bm25_idx):
        did = normal_ids[idx]
        rrf_scores[did] = rrf_scores.get(did, 0) + 1.0 / (60 + rank + 1)
        doc_data[did] = {
            "chunk": normal_chunks[idx],
            "id": did,
            "source": normal_metadatas[idx]["source"],
        }
    
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    results = [doc_data[did] for did in sorted_ids[:top_k * 2]]
    
    # Re-rank
    if results:
        pairs = [(query, r["chunk"]) for r in results]
        ce_scores = cross_encoder.predict(pairs)
        for i, r in enumerate(results):
            r["ce_score"] = float(ce_scores[i])
        results = sorted(results, key=lambda x: x["ce_score"], reverse=True)
    
    return results[:top_k]


def search_contextual(query, top_k=3):
    """Search in contextual (with-context) chunks."""
    # Vector search
    v_results = ctx_collection.query(query_texts=[query], n_results=top_k)
    # BM25 search
    bm25_scores = ctx_bm25.get_scores(query.lower().split())
    top_bm25_idx = np.argsort(bm25_scores)[::-1][:top_k]
    
    # Combine with RRF
    rrf_scores = {}
    doc_data = {}
    
    for rank in range(len(v_results["ids"][0])):
        did = v_results["ids"][0][rank]
        rrf_scores[did] = rrf_scores.get(did, 0) + 1.0 / (60 + rank + 1)
        doc_data[did] = {
            "chunk": v_results["documents"][0][rank],
            "id": did,
            "source": v_results["metadatas"][0][rank]["source"],
            "context": v_results["metadatas"][0][rank].get("context", ""),
        }
    
    for rank, idx in enumerate(top_bm25_idx):
        did = contextual_ids[idx]
        rrf_scores[did] = rrf_scores.get(did, 0) + 1.0 / (60 + rank + 1)
        doc_data[did] = {
            "chunk": contextual_chunks[idx],
            "id": did,
            "source": contextual_metadatas[idx]["source"],
            "context": contextual_metadatas[idx].get("context", ""),
        }
    
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    results = [doc_data[did] for did in sorted_ids[:top_k * 2]]
    
    # Re-rank against original query
    if results:
        pairs = [(query, r["chunk"]) for r in results]
        ce_scores = cross_encoder.predict(pairs)
        for i, r in enumerate(results):
            r["ce_score"] = float(ce_scores[i])
        results = sorted(results, key=lambda x: x["ce_score"], reverse=True)
    
    return results[:top_k]


def generate_answer(query, results):
    """Generate answer from results."""
    context = "\n\n---\n\n".join([r["chunk"] for r in results])
    prompt = f"Context:\n---\n{context}\n---\n\nQuestion: {query}\n\nAnswer briefly:"
    return call_llm(prompt, "Answer using ONLY the provided context.",
                    temperature=0.3, max_tokens=300)


# ── Run comparisons ────────────────────────────────────────────────────────

comparison_queries = [
    {
        "query": "Python cleanup after errors",
        "why": "Chunk about 'finally' block doesn't mention 'Python'",
    },
    {
        "query": "how neural networks learn patterns",
        "why": "AI doc chunks may not repeat 'neural networks' in every chunk",
    },
    {
        "query": "making websites work on mobile phones",
        "why": "User says 'mobile phones' but doc says 'responsive design'",
    },
    {
        "query": "testing code before deploying",
        "why": "Testing chunk might not mention which language",
    },
]

print("-- Side-by-side: Normal vs Contextual retrieval --\n")

for test in comparison_queries:
    query = test["query"]
    print(f"  {'─' * 56}")
    print(f"  Query: \"{query}\"")
    print(f"  Challenge: {test['why']}")
    print(f"  {'─' * 56}\n")
    
    # Normal search
    t0 = time.time()
    normal_results = search_normal(query, top_k=3)
    normal_answer = generate_answer(query, normal_results)
    t_normal = time.time() - t0
    
    normal_sources = set(r["source"] for r in normal_results)
    
    print(f"  [NORMAL] ({t_normal:.1f}s)")
    print(f"    Sources: {normal_sources}")
    for r in normal_results[:2]:
        preview = r["chunk"][:70].replace("\n", " ")
        print(f"    • [{r['source']}] \"{preview}...\"")
    print(f"    Answer: {normal_answer[:120].replace(chr(10), ' ')}...")
    print()
    
    # Contextual search
    t0 = time.time()
    ctx_results = search_contextual(query, top_k=3)
    ctx_answer = generate_answer(query, ctx_results)
    t_ctx = time.time() - t0
    
    ctx_sources = set(r["source"] for r in ctx_results)
    
    print(f"  [CONTEXTUAL] ({t_ctx:.1f}s)")
    print(f"    Sources: {ctx_sources}")
    for r in ctx_results[:2]:
        # Show the context that was added
        ctx_text = r.get("context", "")
        if ctx_text:
            print(f"    • Context: \"{ctx_text[:80]}...\"")
        preview = r["chunk"][:70].replace("\n", " ")
        print(f"      [{r['source']}] \"{preview}...\"")
    print(f"    Answer: {ctx_answer[:120].replace(chr(10), ' ')}...")
    print()
    
    # Compare which found different results
    normal_ids_set = set(r["id"].replace("ctx_chunk", "chunk") for r in normal_results)
    ctx_ids_set = set(r["id"].replace("ctx_chunk", "chunk") for r in ctx_results)
    same = normal_ids_set & ctx_ids_set
    ctx_unique = ctx_ids_set - normal_ids_set
    
    if ctx_unique:
        print(f"  📊 Contextual found {len(ctx_unique)} DIFFERENT chunk(s)!")
    else:
        print(f"  📊 Same chunks retrieved (context helped with ranking)")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: Cost Analysis and Practical Considerations
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 4: Cost Analysis & When to Use")
print("=" * 70)
print()

# Calculate stats
avg_normal_len = np.mean([len(c) for c in normal_chunks])
avg_ctx_len = np.mean([len(c) for c in contextual_chunks])
avg_context_len = np.mean([len(c) for c in contexts_generated])
size_increase = (avg_ctx_len - avg_normal_len) / avg_normal_len * 100

print(f"  CHUNK SIZE ANALYSIS:")
print(f"    Average normal chunk:     {avg_normal_len:.0f} characters")
print(f"    Average context prefix:   {avg_context_len:.0f} characters")
print(f"    Average contextual chunk: {avg_ctx_len:.0f} characters")
print(f"    Size increase:            {size_increase:.0f}%")
print()

print(f"  COST ANALYSIS (for {len(normal_chunks)} chunks):")
print(f"    LLM calls at indexing:    {len(normal_chunks)} (one per chunk)")
print(f"    Indexing time:            {t_total:.1f}s total")
print(f"    Per chunk:                {t_total/len(normal_chunks):.2f}s")
print(f"    Extra storage:            ~{size_increase:.0f}% more per chunk")
print(f"    Query-time cost:          $0 extra (context already embedded)")
print()

print("  KEY INSIGHT:")
print("    The cost is at INDEXING time (once), not QUERY time (every search).")
print("    You add context when building the database,")
print("    then every search benefits for free!")
print()

print("  WHEN TO USE CONTEXTUAL RETRIEVAL:")
print("    ✅ Large documents where chunks lose context")
print("    ✅ Multiple docs where chunks need to identify their source")
print("    ✅ Technical docs where section headers matter")
print("    ✅ Legal/financial docs ('20% growth' needs company context)")
print()
print("  WHEN TO SKIP:")
print("    ❌ Short documents (< 1 page) — chunks already have context")
print("    ❌ Chunks are already self-contained (FAQ style)")
print("    ❌ Cost-sensitive: N chunks = N LLM calls at indexing time")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: Architecture Diagram + Summary
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 5: Architecture + Summary")
print("=" * 70)
print("""
  CONTEXTUAL RETRIEVAL — How It Works
  ====================================

  INDEXING TIME (once, when building the database):
  
  Full Document ──────────────────────────────────┐
       |                                          |
       v                                          |
  Chunk into pieces                               |
       |                                          |
       v                                          v
  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
  │  Chunk 1    │     │  Chunk 2    │     │  Chunk 3    │
  │  "The       │     │  "Virtual   │     │  "Decorators│
  │   finally   │     │   envs are  │     │   modify    │
  │   block..." │     │   essential"│     │   behavior" │
  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
         │                   │                   │
         v                   v                   v
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │ LLM: "What  │  │ LLM: "What  │  │ LLM: "What  │
  │ context does │  │ context does │  │ context does │
  │ this chunk  │  │ this chunk  │  │ this chunk  │
  │ need?"      │  │ need?"      │  │ need?"      │
  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
         │                │                │
         v                v                v
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │ "From Python │  │ "From Python │  │ "From Python │
  │  guide, on   │  │  guide, on   │  │  guide, on   │
  │  error       │  │  virtual     │  │  decorators   │
  │  handling:   │  │  environments│  │  and function │
  │  The finally │  │  : Virtual   │  │  modification│
  │  block..."   │  │  envs are.." │  │  : Decorators│
  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
         │                │                │
         v                v                v
      Embed           Embed            Embed
         │                │                │
         v                v                v
  ┌──────────────────────────────────────────────┐
  │              Vector Database                 │
  │  (chunks are now self-contained!)            │
  └──────────────────────────────────────────────┘

  QUERY TIME (every search — no extra cost!):
  
  User: "Python cleanup after errors"
         │
         v
  Search normally — the context words
  "Python", "error handling" are now IN the chunk,
  so both BM25 and vector search find it easily!


  COMPARISON WITH OTHER TECHNIQUES:
  ┌────────────────────┬───────────────┬──────────────────┐
  │ Technique          │ Fixes         │ When             │
  ├────────────────────┼───────────────┼──────────────────┤
  │ Step 9:  CRAG      │ Bad documents │ After retrieval  │
  │ Step 10: Self-RAG  │ Bad answers   │ After generation │
  │ Step 11: Query     │ Bad queries   │ Before retrieval │
  │          Transform │               │                  │
  │ Step 12: Context   │ Bad CHUNKS    │ At indexing time │
  │          Retrieval │ (orphaned)    │ (once!)          │
  └────────────────────┴───────────────┴──────────────────┘

  The beauty: Step 12 costs NOTHING at query time.
  You pay once when building the database,
  then every search is better forever.
""")

print("  PROGRESS SO FAR:")
print("    Step 6:  Hybrid search (vector + BM25)")
print("    Step 7:  Re-ranking (cross-encoder)")
print("    Step 8:  Citations")
print("    Step 9:  CRAG (check docs before answering)")
print("    Step 10: Self-RAG (check answers after generating)")
print("    Step 11: Query transformation (fix the question)")
print("    Step 12: Contextual retrieval (fix the chunks)")
print()
print("  NEXT STEP -> Step 13: Agentic RAG")
print("    AI becomes the DRIVER — it decides what to search,")
print("    when to search again, and when it has enough info.")
print()
print("[OK] Step 12 complete! You now understand Contextual Retrieval.\n")
