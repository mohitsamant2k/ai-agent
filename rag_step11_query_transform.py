"""
Step 11: Query Transformation — Rewriting Questions for Better Retrieval

Steps 9-10 fixed bad DOCUMENTS and bad ANSWERS.
Step 11 fixes bad QUERIES — before we even search!

THE PROBLEM:
  User asks: "Compare Python and JavaScript for web development"
  
  This is actually 3 questions in one:
    1. "What is Python's role in web development?"
    2. "What is JavaScript's role in web development?"  
    3. "How do they compare?"
  
  Searching for the original query finds vague results.
  Searching for each sub-query finds precise, targeted results!

THREE TECHNIQUES:

  1. QUERY DECOMPOSITION
     Complex question → multiple simple sub-queries
     "Compare X and Y" → "What is X?" + "What is Y?" + "How do X and Y differ?"

  2. QUERY EXPANSION  
     Short query → add synonyms and related terms
     "ML" → "ML OR machine learning OR deep learning OR neural networks"

  3. HyDE (Hypothetical Document Embeddings)
     Generate a FAKE answer first → use that fake answer to search
     Why? The fake answer looks like a real document = better vector match!

This file demonstrates:
  Part 1: Query Decomposition — breaking complex questions apart
  Part 2: Query Expansion — adding related terms
  Part 3: HyDE — hypothetical document search
  Part 4: Comparison — original vs all 3 techniques
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
# SETUP: Full pipeline (reused from Steps 6-10)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("SETUP: Building Full RAG Pipeline")
print("=" * 70)

documents = {
    "python_guide": PYTHON_DOC,
    "ai_ml_guide": AI_DOC,
    "webdev_guide": WEBDEV_DOC,
}

all_chunks = []
all_ids = []
all_metadatas = []

for doc_name, content in documents.items():
    chunks = recursive_chunk(content, chunk_size=400, chunk_overlap=50)
    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        all_ids.append(f"{doc_name}::chunk_{i}")
        all_metadatas.append({"source": doc_name, "chunk_index": i})

print(f"   {len(all_chunks)} chunks from {len(documents)} documents")

tokenized_chunks = [chunk.lower().split() for chunk in all_chunks]
bm25 = BM25Okapi(tokenized_chunks)

client = chromadb.Client()
collection = client.create_collection(name="querytransform_kb")
collection.add(documents=all_chunks, ids=all_ids, metadatas=all_metadatas)

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("   BM25 + ChromaDB + Cross-encoder ready")

# ── Search functions ────────────────────────────────────────────────────────

def bm25_search(query, top_k=5):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [{
        "chunk": all_chunks[idx], "id": all_ids[idx],
        "source": all_metadatas[idx]["source"],
        "score": float(scores[idx]),
    } for idx in top_indices]

def vector_search(query, top_k=5):
    results = collection.query(query_texts=[query], n_results=top_k)
    return [{
        "chunk": results["documents"][0][i],
        "id": results["ids"][0][i],
        "source": results["metadatas"][0][i]["source"],
        "score": distance_to_similarity(results["distances"][0][i]),
    } for i in range(len(results["ids"][0]))]

def hybrid_search(query, top_k=5):
    v = vector_search(query, top_k=top_k)
    b = bm25_search(query, top_k=top_k)
    rrf_scores = {}
    doc_data = {}
    for results in [v, b]:
        for rank, result in enumerate(results, start=1):
            doc_id = result["id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
                doc_data[doc_id] = result
            rrf_scores[doc_id] += 1.0 / (60 + rank)
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    return [{**doc_data[did], "rrf_score": rrf_scores[did]} for did in sorted_ids][:top_k]

def rerank(query, results, top_k=3):
    if not results:
        return []
    pairs = [(query, r["chunk"]) for r in results]
    ce_scores = cross_encoder.predict(pairs)
    for i, result in enumerate(results):
        result["ce_score"] = float(ce_scores[i])
    return sorted(results, key=lambda x: x["ce_score"], reverse=True)[:top_k]

def retrieve(query, top_k=3):
    candidates = hybrid_search(query, top_k=10)
    return rerank(query, candidates, top_k=top_k)

# ── Azure OpenAI ────────────────────────────────────────────────────────────

load_dotenv()
azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
model_name = os.environ.get("AZURE_OPENAI_MODEL", "gpt-4.1-mini")
base_url = azure_endpoint.replace("/chat/completions", "")
llm_client = OpenAI(base_url=base_url, api_key=azure_api_key)

def call_llm(prompt, system_msg="You are a helpful assistant.", temperature=0.3):
    response = llm_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=600,
    )
    return response.choices[0].message.content

print(f"   LLM ready: {model_name}")
print("   Setup complete!\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: Query Decomposition — Breaking Complex Questions Apart
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 1: Query Decomposition")
print("=" * 70)
print()
print("Complex questions often contain MULTIPLE sub-questions.")
print("Breaking them apart gets better, more targeted retrieval.")
print()
print("  Original: 'Compare Python and JavaScript for web dev'")
print("  Sub-queries:")
print("    1. 'What is Python used for in web development?'")
print("    2. 'What is JavaScript used for in web development?'")
print("    3. 'How do Python and JavaScript compare for web dev?'")
print()

DECOMPOSE_SYSTEM = """You are a query decomposition assistant. Break complex questions 
into simpler sub-questions that can each be answered independently.

Rules:
1. Output ONLY the sub-questions, one per line
2. Each sub-question should be self-contained (no "it" or "they")
3. Generate 2-4 sub-questions
4. Number them: 1. 2. 3.
5. Keep them concise and specific"""

DECOMPOSE_PROMPT = """Break this complex question into simpler sub-questions:

Question: {query}

Sub-questions:"""


def decompose_query(query):
    """
    Break a complex question into simpler sub-questions.
    
    Returns:
        list of sub-query strings
    """
    prompt = DECOMPOSE_PROMPT.format(query=query)
    response = call_llm(prompt, DECOMPOSE_SYSTEM, temperature=0.3)
    
    # Parse numbered lines
    sub_queries = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove numbering: "1. ", "1) ", "- ", etc.
        cleaned = line
        for prefix in ["1.", "2.", "3.", "4.", "5.", "1)", "2)", "3)", "4)", "- "]:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break
        if cleaned:
            sub_queries.append(cleaned)
    
    return sub_queries


def retrieve_with_decomposition(query, top_k=3):
    """
    Decompose query → search each sub-query → merge & deduplicate results.
    
    Returns:
        dict with sub_queries, all results, and merged top results
    """
    sub_queries = decompose_query(query)
    
    all_results = {}  # id -> result (deduplication)
    sub_results = {}  # sub_query -> results
    
    for sq in sub_queries:
        results = retrieve(sq, top_k=top_k)
        sub_results[sq] = results
        for r in results:
            rid = r["id"]
            if rid not in all_results:
                all_results[rid] = r
                all_results[rid]["found_by"] = [sq]
            else:
                all_results[rid]["found_by"].append(sq)
    
    # Sort by how many sub-queries found the same chunk (more = more relevant)
    merged = sorted(
        all_results.values(),
        key=lambda x: len(x.get("found_by", [])),
        reverse=True,
    )
    
    return {
        "original_query": query,
        "sub_queries": sub_queries,
        "sub_results": sub_results,
        "merged_results": merged[:top_k * 2],  # Keep more since we merged
        "unique_chunks": len(all_results),
    }


# ── Demo: Query Decomposition ──────────────────────────────────────────────

print("-- Demo: Decomposing complex queries --\n")

decomposition_tests = [
    "Compare Python and JavaScript for building web applications",
    "What are the differences between supervised and unsupervised learning, and when should I use each?",
    "How do decorators work in Python and how are they similar to React higher-order components?",
]

for query in decomposition_tests:
    print(f"  Original: \"{query}\"\n")
    
    result = retrieve_with_decomposition(query, top_k=3)
    
    print(f"  Decomposed into {len(result['sub_queries'])} sub-queries:")
    for i, sq in enumerate(result["sub_queries"], 1):
        count = len(result["sub_results"].get(sq, []))
        print(f"    {i}. \"{sq}\" ({count} results)")
    
    print(f"\n  Merged: {result['unique_chunks']} unique chunks found")
    
    # Show top merged results
    for r in result["merged_results"][:3]:
        found_count = len(r.get("found_by", []))
        preview = r["chunk"][:60].replace("\n", " ")
        print(f"    [{r['source']}] (found by {found_count} sub-queries)")
        print(f"      \"{preview}...\"")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: Query Expansion — Adding Related Terms
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 2: Query Expansion")
print("=" * 70)
print()
print("Short queries miss relevant docs because of vocabulary mismatch.")
print()
print("  User searches: 'ML'")
print("  Docs say: 'machine learning', 'deep learning', 'neural networks'")
print("  No keyword match! BM25 misses these docs.")
print()
print("  Query expansion: 'ML' -> 'ML machine learning deep learning")
print("                           neural networks AI algorithms'")
print()

EXPAND_SYSTEM = """You are a search query expansion assistant. Given a query, add 
related terms, synonyms, and alternative phrasings to improve search recall.

Rules:
1. Keep the original query terms
2. Add 3-6 related terms or synonyms
3. Output as a single line of terms separated by spaces
4. Do NOT add explanations, just the expanded query
5. Focus on terms likely to appear in technical documents"""

EXPAND_PROMPT = """Expand this search query with related terms and synonyms:

Original query: {query}

Expanded query:"""


def expand_query(query):
    """
    Add related terms and synonyms to a query.
    
    Returns:
        str: expanded query with additional terms
    """
    prompt = EXPAND_PROMPT.format(query=query)
    response = call_llm(prompt, EXPAND_SYSTEM, temperature=0.3)
    return response.strip().strip('"')


def retrieve_with_expansion(query, top_k=3):
    """
    Expand query → search with both original + expanded → merge results.
    """
    expanded = expand_query(query)
    
    # Search with both
    original_results = retrieve(query, top_k=top_k)
    expanded_results = retrieve(expanded, top_k=top_k)
    
    # Merge and deduplicate
    seen = set()
    merged = []
    for r in original_results + expanded_results:
        if r["id"] not in seen:
            seen.add(r["id"])
            merged.append(r)
    
    # Re-rank the merged results against original query
    if len(merged) > top_k:
        merged = rerank(query, merged, top_k=top_k)
    
    return {
        "original_query": query,
        "expanded_query": expanded,
        "original_results": original_results,
        "expanded_results": expanded_results,
        "merged_results": merged,
    }


# ── Demo: Query Expansion ──────────────────────────────────────────────────

print("-- Demo: Expanding queries --\n")

expansion_tests = [
    "ML",
    "Python async",
    "CSS layout",
    "error handling",
    "REST API",
]

for query in expansion_tests:
    result = retrieve_with_expansion(query, top_k=3)
    
    orig_ids = {r["id"] for r in result["original_results"]}
    expanded_ids = {r["id"] for r in result["expanded_results"]}
    new_finds = expanded_ids - orig_ids
    
    print(f"  Original:  \"{query}\"")
    print(f"  Expanded:  \"{result['expanded_query'][:80]}...\"")
    print(f"  Original found: {len(orig_ids)} chunks")
    print(f"  Expansion found: {len(new_finds)} NEW chunks not in original")
    
    # Show what new chunks expansion found
    if new_finds:
        for r in result["expanded_results"]:
            if r["id"] in new_finds:
                preview = r["chunk"][:60].replace("\n", " ")
                print(f"    NEW: [{r['source']}] \"{preview}...\"")
                break
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: HyDE — Hypothetical Document Embeddings
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 3: HyDE -- Hypothetical Document Embeddings")
print("=" * 70)
print()
print("THE INSIGHT:")
print("  Questions and answers live in DIFFERENT vector spaces!")
print()
print("  Question: 'How does Python handle errors?'")
print("  Document: 'Python uses try-except blocks for error handling...'")
print()
print("  These have different sentence structures:")
print("    Question = interrogative ('How does...')")
print("    Document = declarative  ('Python uses...')")
print()
print("  HyDE solves this by generating a FAKE answer first,")
print("  then searching with the fake answer instead of the question.")
print("  The fake answer looks like a real document = better match!")
print()

HYDE_SYSTEM = """You are a technical writer. Write a short, factual passage that 
would answer the given question. Write as if this is from a textbook or documentation.

Rules:
1. Write 2-4 sentences
2. Use declarative statements (not questions)
3. Be specific and technical  
4. It's OK if you're not 100% accurate — this is just for search
5. Do NOT say "I think" or "I'm not sure" — write confidently"""

HYDE_PROMPT = """Write a short passage that would answer this question:

Question: {query}

Passage:"""


def generate_hypothetical_document(query):
    """
    Generate a hypothetical (fake) document that would answer the query.
    This fake doc is used as the search query instead of the question.
    
    Returns:
        str: hypothetical document passage
    """
    prompt = HYDE_PROMPT.format(query=query)
    response = call_llm(prompt, HYDE_SYSTEM, temperature=0.5)
    return response.strip()


def retrieve_with_hyde(query, top_k=3):
    """
    HyDE: Generate fake answer → use it to search → return real results.
    
    The fake answer has document-like structure, so vector search
    finds better matches than searching with a question.
    """
    # Step 1: Generate hypothetical document
    hypo_doc = generate_hypothetical_document(query)
    
    # Step 2: Search using the hypothetical doc as the query
    # Vector search benefits most from HyDE (embedding similarity)
    hyde_results = vector_search(hypo_doc, top_k=top_k * 2)
    
    # Step 3: Also do normal search for comparison
    normal_results = vector_search(query, top_k=top_k * 2)
    
    # Step 4: Merge and re-rank against original query
    seen = set()
    merged = []
    for r in hyde_results + normal_results:
        if r["id"] not in seen:
            seen.add(r["id"])
            merged.append(r)
    
    reranked = rerank(query, merged, top_k=top_k)
    
    return {
        "original_query": query,
        "hypothetical_doc": hypo_doc,
        "hyde_results": hyde_results[:top_k],
        "normal_results": normal_results[:top_k],
        "final_results": reranked,
    }


# ── Demo: HyDE in action ───────────────────────────────────────────────────

print("-- Demo: HyDE search --\n")

hyde_tests = [
    "How does Python handle errors?",
    "What is transfer learning in AI?",
    "How do you make a responsive website?",
]

for query in hyde_tests:
    result = retrieve_with_hyde(query, top_k=3)
    
    print(f"  Query: \"{query}\"\n")
    
    # Show the hypothetical document
    hypo_preview = result["hypothetical_doc"][:120].replace("\n", " ")
    print(f"  Hypothetical doc: \"{hypo_preview}...\"\n")
    
    # Compare what normal vs HyDE found
    normal_ids = [r["id"] for r in result["normal_results"]]
    hyde_ids = [r["id"] for r in result["hyde_results"]]
    
    # Check overlap
    overlap = set(normal_ids) & set(hyde_ids)
    hyde_unique = set(hyde_ids) - set(normal_ids)
    
    print(f"  Normal search found: {len(normal_ids)} chunks")
    print(f"  HyDE search found:   {len(hyde_ids)} chunks")
    print(f"  Overlap: {len(overlap)}  |  HyDE-only: {len(hyde_unique)}")
    
    # Show final reranked results
    print(f"  Final (re-ranked):")
    for r in result["final_results"][:2]:
        preview = r["chunk"][:60].replace("\n", " ")
        print(f"    [{r['source']}] \"{preview}...\"")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: Full Comparison — Original vs All 3 Techniques
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 4: Side-by-Side Comparison")
print("=" * 70)
print()
print("Testing all techniques on the same queries to see which helps most.")
print()


def generate_final_answer(query, results):
    """Generate answer from retrieved results."""
    context = "\n\n---\n\n".join([r["chunk"] for r in results])
    system_msg = "You are a helpful assistant. Answer using ONLY the provided context."
    prompt = f"Context:\n---\n{context}\n---\n\nQuestion: {query}\n\nAnswer briefly:"
    return call_llm(prompt, system_msg)


comparison_queries = [
    {
        "query": "Compare Python decorators with JavaScript higher-order functions",
        "why": "Complex query needing info from 2 different docs",
    },
    {
        "query": "ML algorithms",
        "why": "Short/vague query needing expansion",
    },
    {
        "query": "How do you handle async operations in web apps?",
        "why": "Question structure differs from doc structure (HyDE helps)",
    },
]

for test in comparison_queries:
    query = test["query"]
    print(f"  {'=' * 58}")
    print(f"  Query: \"{query}\"")
    print(f"  Why interesting: {test['why']}")
    print(f"  {'=' * 58}\n")
    
    # 1. Original (no transformation)
    t0 = time.time()
    orig_results = retrieve(query, top_k=3)
    orig_sources = set(r["source"] for r in orig_results)
    orig_answer = generate_final_answer(query, orig_results)
    t_orig = time.time() - t0
    
    print(f"  [ORIGINAL] ({t_orig:.1f}s)")
    print(f"    Sources: {orig_sources}")
    print(f"    Answer: {orig_answer[:100].replace(chr(10), ' ')}...")
    print()
    
    # 2. Decomposition
    t0 = time.time()
    decomp = retrieve_with_decomposition(query, top_k=3)
    decomp_sources = set(r["source"] for r in decomp["merged_results"][:3])
    decomp_answer = generate_final_answer(query, decomp["merged_results"][:3])
    t_decomp = time.time() - t0
    
    print(f"  [DECOMPOSED] ({t_decomp:.1f}s) -> {len(decomp['sub_queries'])} sub-queries")
    print(f"    Sources: {decomp_sources}")
    print(f"    Answer: {decomp_answer[:100].replace(chr(10), ' ')}...")
    print()
    
    # 3. Expansion
    t0 = time.time()
    expanded = retrieve_with_expansion(query, top_k=3)
    exp_sources = set(r["source"] for r in expanded["merged_results"][:3])
    exp_answer = generate_final_answer(query, expanded["merged_results"][:3])
    t_exp = time.time() - t0
    
    print(f"  [EXPANDED] ({t_exp:.1f}s)")
    print(f"    Expanded to: \"{expanded['expanded_query'][:60]}...\"")
    print(f"    Sources: {exp_sources}")
    print(f"    Answer: {exp_answer[:100].replace(chr(10), ' ')}...")
    print()
    
    # 4. HyDE
    t0 = time.time()
    hyde = retrieve_with_hyde(query, top_k=3)
    hyde_sources = set(r["source"] for r in hyde["final_results"])
    hyde_answer = generate_final_answer(query, hyde["final_results"])
    t_hyde = time.time() - t0
    
    print(f"  [HyDE] ({t_hyde:.1f}s)")
    print(f"    Hypothetical: \"{hyde['hypothetical_doc'][:60]}...\"")
    print(f"    Sources: {hyde_sources}")
    print(f"    Answer: {hyde_answer[:100].replace(chr(10), ' ')}...")
    print()
    
    # Summary: which found the most diverse sources?
    all_sources = orig_sources | decomp_sources | exp_sources | hyde_sources
    print(f"  Source coverage: Original={len(orig_sources)} "
          f"Decomp={len(decomp_sources)} "
          f"Expanded={len(exp_sources)} "
          f"HyDE={len(hyde_sources)} "
          f"(total unique={len(all_sources)})")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: Architecture Diagram + Summary
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 5: Architecture + Summary")
print("=" * 70)
print("""
  QUERY TRANSFORMATION TECHNIQUES
  ================================

  User Query: "Compare Python and JS for web dev"
       |
       +---> DECOMPOSITION
       |     Split into sub-questions:
       |       Q1: "Python for web dev?"
       |       Q2: "JavaScript for web dev?"
       |       Q3: "How do they compare?"
       |     Search each -> merge results
       |
       +---> EXPANSION
       |     Add synonyms and related terms:
       |       "Python JavaScript web development
       |        frameworks Flask Django React Node.js
       |        server-side client-side comparison"
       |     Search with expanded query
       |
       +---> HyDE
             Generate fake answer:
               "Python is used for web dev with frameworks
                like Django and Flask. JavaScript dominates
                front-end with React and Vue..."
             Search with fake answer (better vector match!)

  WHEN TO USE EACH:
  +------------------+-------------------------------+-----------+
  | Technique        | Best for                      | LLM calls |
  +------------------+-------------------------------+-----------+
  | Decomposition    | Complex/multi-part questions   | 1         |
  |                  | "Compare X and Y"              |           |
  |                  | "What are A, B, and C?"        |           |
  +------------------+-------------------------------+-----------+
  | Expansion        | Short/vague queries            | 1         |
  |                  | "ML" -> needs synonyms         |           |
  |                  | Vocabulary mismatch            |           |
  +------------------+-------------------------------+-----------+
  | HyDE             | Question vs document mismatch  | 1         |
  |                  | "How does X work?" searches    |           |
  |                  | better as "X works by..."      |           |
  +------------------+-------------------------------+-----------+

  You can COMBINE them:
    1. Decompose complex query into sub-queries
    2. Expand each sub-query with related terms
    3. Use HyDE on each expanded sub-query
    4. Merge all results -> re-rank -> answer

  COST: 1 extra LLM call per technique (cheap!)
        Unlike CRAG/Self-RAG which need 3-8 extra calls
""")

print("  PROGRESS SO FAR:")
print("    Step 6:  Hybrid search (vector + BM25)")
print("    Step 7:  Re-ranking (cross-encoder)")
print("    Step 8:  Citations")
print("    Step 9:  CRAG (check docs before answering)")
print("    Step 10: Self-RAG (check answers after generating)")
print("    Step 11: Query transformation (fix the question itself)")
print()
print("  NEXT STEP -> Step 12: Contextual Retrieval")
print("    Add context to chunks BEFORE embedding them")
print("    'Revenue grew 20%' -> 'From Acme 2025 report: Revenue grew 20%'")
print()
print("[OK] Step 11 complete! You now understand Query Transformation.\n")
