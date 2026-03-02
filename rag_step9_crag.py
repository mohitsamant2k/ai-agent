"""
Step 9: Corrective RAG (CRAG) — Self-Correcting Retrieval

Building on Steps 5-8, we make the RAG pipeline SELF-CORRECTING.

THE PROBLEM with basic RAG:
  User: "What is quantum computing?"
  RAG retrieves: chunks about "machine learning" (vaguely related)
  LLM: Makes up an answer using irrelevant context → HALLUCINATION!

  The retriever returns results even when they're WRONG.
  Basic RAG blindly trusts whatever the retriever gives it.

THE SOLUTION — Corrective RAG (CRAG):
  After retrieval, an AI "judge" CHECKS if the documents are relevant:
  
  ┌──────────────────────────────────────────────────────┐
  │  VERDICT        │  ACTION                            │
  ├──────────────────────────────────────────────────────┤
  │  ✅ CORRECT     │  Use the documents → generate      │
  │  🟡 AMBIGUOUS   │  Extract useful parts → generate   │
  │  ❌ INCORRECT   │  Reformulate query → search again  │
  └──────────────────────────────────────────────────────┘

  This prevents the LLM from using bad context!

This file demonstrates:
  Part 1: Relevance grading — AI judges retrieved documents
  Part 2: Query reformulation — rewriting bad queries
  Part 3: Knowledge extraction — filtering partial matches
  Part 4: Full CRAG pipeline with decision logic
  Part 5: Comparison: Basic RAG vs CRAG
  Part 6: Architecture diagram
"""

import os
import re
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
# SETUP: Full pipeline (reused from Steps 6-8)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("SETUP: Building Full RAG Pipeline")
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

print(f"   {len(all_chunks)} chunks from {len(documents)} documents")

# ── BM25 + ChromaDB + Cross-encoder ────────────────────────────────────────

tokenized_chunks = [chunk.lower().split() for chunk in all_chunks]
bm25 = BM25Okapi(tokenized_chunks)

client = chromadb.Client()
collection = client.create_collection(name="crag_kb")
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
    return [
        {**doc_data[did], "rrf_score": rrf_scores[did]} for did in sorted_ids
    ][:top_k]

def rerank(query, results, top_k=3):
    if not results:
        return []
    pairs = [(query, r["chunk"]) for r in results]
    ce_scores = cross_encoder.predict(pairs)
    for i, result in enumerate(results):
        result["ce_score"] = float(ce_scores[i])
    return sorted(results, key=lambda x: x["ce_score"], reverse=True)[:top_k]

def retrieve(query, top_k=3):
    """Full retrieval pipeline: Hybrid + Rerank."""
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
# PART 1: Relevance Grading — AI Judges the Retrieved Documents
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 1: Relevance Grading -- The AI Judge")
print("=" * 70)
print()
print("After retrieval, we ask the LLM: 'Is this document actually relevant?'")
print()
print("The LLM acts as a JUDGE with 3 verdicts:")
print("  CORRECT   -- Document directly answers the question")
print("  AMBIGUOUS -- Document is related but doesn't fully answer")
print("  INCORRECT -- Document is irrelevant to the question")
print()

GRADING_SYSTEM = """You are a relevance grading assistant. Your job is to judge whether 
a retrieved document is relevant to the user's question.

Grade the document as one of:
- CORRECT: The document directly contains information that answers the question
- AMBIGUOUS: The document is somewhat related but doesn't directly answer the question
- INCORRECT: The document is not relevant to the question at all

Respond with ONLY one word: CORRECT, AMBIGUOUS, or INCORRECT."""

GRADING_PROMPT = """Question: {query}

Retrieved Document:
---
{document}
---

Is this document relevant to answering the question? 
Respond with ONLY: CORRECT, AMBIGUOUS, or INCORRECT"""


def grade_document(query, document):
    """
    Use LLM to grade whether a document is relevant to the query.
    
    Returns: 'CORRECT', 'AMBIGUOUS', or 'INCORRECT'
    """
    prompt = GRADING_PROMPT.format(query=query, document=document)
    response = call_llm(prompt, GRADING_SYSTEM, temperature=0.0)
    
    # Parse the response — extract the verdict
    response_upper = response.strip().upper()
    if "CORRECT" in response_upper and "INCORRECT" not in response_upper:
        return "CORRECT"
    elif "AMBIGUOUS" in response_upper:
        return "AMBIGUOUS"
    elif "INCORRECT" in response_upper:
        return "INCORRECT"
    else:
        # Default to AMBIGUOUS if unclear
        return "AMBIGUOUS"


def grade_all_documents(query, results):
    """
    Grade all retrieved documents for relevance.
    
    Returns:
        List of results with 'grade' field added
    """
    graded = []
    for r in results:
        grade = grade_document(query, r["chunk"])
        r_copy = r.copy()
        r_copy["grade"] = grade
        graded.append(r_copy)
    return graded


# ── Demo: Grading retrieved documents ───────────────────────────────────────

print("-- Demo: Grading documents for relevance --\n")

grading_tests = [
    {
        "query": "How does Python handle errors?",
        "description": "Should find relevant Python error-handling chunks",
    },
    {
        "query": "What is quantum computing?",
        "description": "NOT in our docs -- should grade as INCORRECT",
    },
    {
        "query": "How do neural networks learn?",
        "description": "Partially covered in AI docs -- may be AMBIGUOUS",
    },
]

for test in grading_tests:
    query = test["query"]
    print(f"  Query: \"{query}\"")
    print(f"  ({test['description']})\n")
    
    results = retrieve(query, top_k=3)
    graded = grade_all_documents(query, results)
    
    for i, g in enumerate(graded):
        verdict_icon = {
            "CORRECT": "[OK]", "AMBIGUOUS": "[??]", "INCORRECT": "[NO]"
        }
        icon = verdict_icon.get(g["grade"], "[??]")
        preview = g["chunk"][:70].replace("\n", " ")
        print(f"    {i+1}. {icon} {g['grade']:<10} [{g['source']}]")
        print(f"       \"{preview}...\"")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: Query Reformulation — Fixing Bad Queries
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 2: Query Reformulation -- Rewriting Bad Queries")
print("=" * 70)
print()
print("When ALL documents are INCORRECT, the query might be:")
print("  1. Too vague -> make it more specific")
print("  2. Using wrong terms -> use better keywords")
print("  3. About something not in our docs -> acknowledge it")
print()

REFORMULATE_SYSTEM = """You are a search query optimizer. When a search query returns 
irrelevant results, your job is to rewrite the query to get better results.

The knowledge base contains documents about:
- Python programming (decorators, async, error handling, testing, etc.)
- AI/ML (supervised/unsupervised learning, deep learning, RAG, transformers)
- Web development (HTML, CSS, JavaScript, React, REST APIs, Node.js)

Rules:
1. Rewrite the query to better match what might be in these documents
2. Use more specific technical terms
3. If the topic is clearly NOT covered, respond with: NOT_IN_KB
4. Return ONLY the rewritten query or NOT_IN_KB"""

REFORMULATE_PROMPT = """Original query: {query}

The search returned irrelevant results for this query.
Rewrite the query to get better results from a knowledge base about 
Python, AI/ML, and Web Development.

Rewritten query:"""


def reformulate_query(query):
    """
    Rewrite a query that got poor retrieval results.
    
    Returns:
        tuple: (new_query, was_reformulated)
        If topic is not in KB, returns (None, False)
    """
    prompt = REFORMULATE_PROMPT.format(query=query)
    response = call_llm(prompt, REFORMULATE_SYSTEM, temperature=0.3)
    
    cleaned = response.strip().strip('"')
    
    if "NOT_IN_KB" in cleaned.upper():
        return None, False
    
    return cleaned, True


# ── Demo: Query reformulation ───────────────────────────────────────────────

print("-- Demo: Reformulating queries --\n")

reformulation_tests = [
    "What is quantum computing?",                    # Not in KB
    "How do computers learn stuff?",                  # Vague -> specific
    "making websites look good on phones",            # Casual -> technical
    "that Python thing with @ symbol above functions", # Informal -> proper
    "blockchain cryptocurrency mining",               # Not in KB
]

for query in reformulation_tests:
    new_query, was_reformulated = reformulate_query(query)
    
    if was_reformulated:
        print(f"  Original:     \"{query}\"")
        print(f"  Reformulated: \"{new_query}\"")
    else:
        print(f"  Original:     \"{query}\"")
        print(f"  Result:       NOT_IN_KB (topic not covered)")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: Knowledge Extraction — Filtering Partial Matches
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 3: Knowledge Extraction -- Filtering Useful Parts")
print("=" * 70)
print()
print("When documents are AMBIGUOUS (partially relevant):")
print("  Don't use the WHOLE chunk -> extract ONLY the useful parts.")
print()

EXTRACT_SYSTEM = """You are a knowledge extraction assistant. Given a question and a 
document that is partially relevant, extract ONLY the sentences from the document 
that are relevant to answering the question.

Rules:
1. Extract exact sentences from the document (do not paraphrase)
2. Only include sentences that help answer the question
3. If nothing is relevant, respond with: NO_RELEVANT_CONTENT
4. Separate extracted sentences with newlines"""

EXTRACT_PROMPT = """Question: {query}

Document:
---
{document}
---

Extract ONLY the sentences relevant to the question:"""


def extract_relevant_knowledge(query, document):
    """
    Extract only the relevant parts from a partially-relevant document.
    
    Returns:
        str: Extracted relevant content, or None if nothing useful
    """
    prompt = EXTRACT_PROMPT.format(query=query, document=document)
    response = call_llm(prompt, EXTRACT_SYSTEM, temperature=0.0)
    
    if "NO_RELEVANT_CONTENT" in response.upper():
        return None
    
    return response.strip()


# ── Demo: Knowledge extraction ──────────────────────────────────────────────

print("-- Demo: Extracting relevant knowledge --\n")

# A chunk that's partially relevant
demo_chunk = """Python's simplicity makes it an excellent first programming language. 
Variables don't need type declarations, indentation enforces clean code structure, 
and the syntax reads almost like English. List comprehensions, generator expressions, 
and built-in functions like map, filter, and zip make data processing concise and elegant."""

demo_questions = [
    "What makes Python easy for beginners?",          # Very relevant
    "How does Python handle data processing?",         # Partially relevant
    "What is Python's approach to error handling?",    # Not in this chunk
]

for question in demo_questions:
    extracted = extract_relevant_knowledge(question, demo_chunk)
    print(f"  Q: \"{question}\"")
    if extracted:
        # Show first 150 chars
        preview = extracted[:150].replace("\n", " | ")
        print(f"  Extracted: \"{preview}...\"")
    else:
        print(f"  Extracted: (nothing relevant)")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: Full CRAG Pipeline — Decision Logic
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 4: Full CRAG Pipeline -- Self-Correcting RAG")
print("=" * 70)
print()
print("The CRAG decision flow:")
print()
print("  1. Retrieve documents")
print("  2. Grade each document (CORRECT / AMBIGUOUS / INCORRECT)")
print("  3. Decide action:")
print("     - If ANY correct  -> use correct docs -> generate answer")
print("     - If ALL ambiguous -> extract useful parts -> generate")
print("     - If ALL incorrect -> reformulate query -> retry search")
print("     - If still bad     -> say 'I don't know'")
print()


def crag_pipeline(query, max_retries=1):
    """
    Corrective RAG pipeline with self-correction.
    
    Steps:
      1. Retrieve documents
      2. Grade relevance
      3. Decide: use as-is / extract / reformulate / give up
    
    Args:
        query: User's question
        max_retries: How many times to reformulate and retry
    
    Returns:
        dict with answer, decision path, and metadata
    """
    t_start = time.time()
    
    current_query = query
    decision_log = []
    attempt = 0
    
    while attempt <= max_retries:
        attempt += 1
        step_log = {"attempt": attempt, "query": current_query}
        
        # Step 1: Retrieve
        results = retrieve(current_query, top_k=3)
        step_log["retrieved"] = len(results)
        
        # Step 2: Grade each document
        graded = grade_all_documents(current_query, results)
        
        correct = [g for g in graded if g["grade"] == "CORRECT"]
        ambiguous = [g for g in graded if g["grade"] == "AMBIGUOUS"]
        incorrect = [g for g in graded if g["grade"] == "INCORRECT"]
        
        step_log["grades"] = {
            "correct": len(correct),
            "ambiguous": len(ambiguous),
            "incorrect": len(incorrect),
        }
        
        # Step 3: Decide action
        if correct:
            # BEST CASE: We have directly relevant documents
            step_log["action"] = "USE_CORRECT"
            context_chunks = [r["chunk"] for r in correct]
            sources = [r["source"] for r in correct]
            decision_log.append(step_log)
            break
            
        elif ambiguous and not incorrect:
            # PARTIAL: Extract useful parts from ambiguous docs
            step_log["action"] = "EXTRACT_FROM_AMBIGUOUS"
            context_chunks = []
            sources = []
            for r in ambiguous:
                extracted = extract_relevant_knowledge(current_query, r["chunk"])
                if extracted:
                    context_chunks.append(extracted)
                    sources.append(r["source"])
            if not context_chunks:
                # Extraction failed, try reformulation
                step_log["action"] = "EXTRACT_FAILED_REFORMULATE"
                decision_log.append(step_log)
                new_query, success = reformulate_query(current_query)
                if success and new_query:
                    current_query = new_query
                    continue
                else:
                    break
            decision_log.append(step_log)
            break
            
        else:
            # BAD CASE: All or mostly incorrect
            step_log["action"] = "REFORMULATE"
            decision_log.append(step_log)
            
            if attempt <= max_retries:
                new_query, success = reformulate_query(current_query)
                if success and new_query:
                    step_log["reformulated_to"] = new_query
                    current_query = new_query
                    continue
                else:
                    step_log["action"] = "NOT_IN_KNOWLEDGE_BASE"
                    break
            else:
                step_log["action"] = "MAX_RETRIES_REACHED"
                break
    
    t_retrieve = time.time()
    
    # Step 4: Generate answer (or decline)
    last_action = decision_log[-1]["action"] if decision_log else "UNKNOWN"
    
    if last_action in ("NOT_IN_KNOWLEDGE_BASE", "MAX_RETRIES_REACHED"):
        answer = ("I don't have enough information in my knowledge base to "
                  "answer this question. My documents cover Python programming, "
                  "AI/ML concepts, and web development.")
    elif context_chunks:
        context = "\n\n---\n\n".join(context_chunks)
        system_msg = """You are a helpful assistant. Answer using ONLY the provided context.
If the context doesn't fully answer the question, say what you can and note the limitation.
Add citation markers [1], [2] etc. for each source used."""
        
        prompt = f"""Context:
---
{context}
---

Question: {query}

Answer using the context above:"""
        
        answer = call_llm(prompt, system_msg)
    else:
        answer = ("I found some related information but couldn't extract "
                  "anything directly relevant to your question.")
    
    t_end = time.time()
    
    return {
        "query": query,
        "answer": answer,
        "decision_log": decision_log,
        "final_action": last_action,
        "sources": sources if 'sources' in dir() and sources else [],
        "timing": {
            "retrieve_grade": t_retrieve - t_start,
            "llm": t_end - t_retrieve,
            "total": t_end - t_start,
        },
    }


# ── Demo: Full CRAG pipeline ───────────────────────────────────────────────

print("-- Demo: CRAG in action --\n")

crag_tests = [
    {
        "query": "How does Python handle errors and exceptions?",
        "expect": "Should find CORRECT docs, answer directly",
    },
    {
        "query": "What is quantum computing and how do qubits work?",
        "expect": "NOT in KB, should say 'I don't know'",
    },
    {
        "query": "How do machines learn patterns from data?",
        "expect": "Should find AI/ML docs (might be AMBIGUOUS)",
    },
    {
        "query": "that thing with @ in Python above functions",
        "expect": "Vague query, may need reformulation",
    },
]

for test in crag_tests:
    query = test["query"]
    print(f"{'=' * 60}")
    print(f"  Query: \"{query}\"")
    print(f"  Expected: {test['expect']}")
    print(f"{'=' * 60}\n")
    
    result = crag_pipeline(query)
    
    # Show decision path
    for log in result["decision_log"]:
        grades = log.get("grades", {})
        grade_str = (f"C={grades.get('correct', 0)} "
                     f"A={grades.get('ambiguous', 0)} "
                     f"I={grades.get('incorrect', 0)}")
        
        action_icon = {
            "USE_CORRECT": "[OK] Use correct docs",
            "EXTRACT_FROM_AMBIGUOUS": "[>>] Extract from ambiguous",
            "REFORMULATE": "[??] Reformulate query",
            "NOT_IN_KNOWLEDGE_BASE": "[NO] Not in knowledge base",
            "MAX_RETRIES_REACHED": "[!!] Max retries reached",
            "EXTRACT_FAILED_REFORMULATE": "[>>] Extract failed, reformulate",
        }
        icon = action_icon.get(log["action"], log["action"])
        
        print(f"  Attempt {log['attempt']}: [{grade_str}] -> {icon}")
        if "reformulated_to" in log:
            print(f"    Reformulated to: \"{log['reformulated_to']}\"")
    
    # Show answer
    print(f"\n  Answer:")
    # Word-wrap at 60 chars
    words = result["answer"].split()
    line = "    "
    for word in words:
        if len(line) + len(word) + 1 > 70:
            print(line)
            line = "    " + word
        else:
            line += " " + word if line.strip() else "    " + word
    if line.strip():
        print(line)
    
    print(f"\n  Time: {result['timing']['total']:.1f}s")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: Comparison — Basic RAG vs CRAG
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 5: Basic RAG vs CRAG -- Side by Side")
print("=" * 70)
print()


def basic_rag(query):
    """Simple RAG without correction (for comparison)."""
    t_start = time.time()
    results = retrieve(query, top_k=3)
    context = "\n\n".join([r["chunk"] for r in results])
    
    system_msg = """You are a helpful assistant. Answer using ONLY the provided context.
If the context doesn't contain the answer, say "I don't have enough information."
Add citation markers [1], [2] etc."""
    
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
        "sources": [r["source"] for r in results],
        "timing": t_end - t_start,
    }


comparison_queries = [
    "How does Python handle errors?",                   # In KB - both should work
    "What is blockchain and how does mining work?",     # NOT in KB
    "that @ thing above Python functions",              # Vague query
]

for query in comparison_queries:
    print(f"  {'~' * 60}")
    print(f"  Query: \"{query}\"")
    print(f"  {'~' * 60}\n")
    
    # Basic RAG
    basic = basic_rag(query)
    print(f"  [BASIC RAG] ({basic['timing']:.1f}s)")
    basic_preview = basic["answer"][:180].replace("\n", " ")
    print(f"    {basic_preview}...")
    print()
    
    # CRAG
    corrective = crag_pipeline(query)
    print(f"  [CRAG] ({corrective['timing']['total']:.1f}s)")
    print(f"    Decision: {corrective['final_action']}")
    crag_preview = corrective["answer"][:180].replace("\n", " ")
    print(f"    {crag_preview}...")
    print()

print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 6: Architecture Diagram
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 6: CRAG Architecture")
print("=" * 70)
print("""
  CORRECTIVE RAG (CRAG) PIPELINE
  ===============================

  User Query
       |
       v
  +-----------+
  | Retrieve  |  Hybrid Search + Re-rank (Steps 6-7)
  | top-3     |
  +-----+-----+
        |
        v
  +-------------+
  | Grade Each  |  LLM judges: CORRECT / AMBIGUOUS / INCORRECT
  | Document    |
  +------+------+
         |
    +----+----+--------------------+
    |         |                    |
    v         v                    v
 CORRECT   AMBIGUOUS           INCORRECT
    |         |                    |
    v         v                    v
 Use docs  Extract only      Reformulate query
 as-is     useful parts      (rewrite + retry)
    |         |                    |
    +----+----+              +-----+-----+
         |                   |           |
         v                   v           v
  +-------------+      Retry with    NOT_IN_KB
  | Generate    |      new query     -> "I don't
  | Answer      |      (max 1x)        know"
  | with LLM    |
  +------+------+
         |
         v
  Answer + Sources

  KEY INSIGHT:
    Basic RAG:  Retrieve -> Generate (blindly trusts retriever)
    CRAG:       Retrieve -> GRADE -> Decide -> Generate
                            ^^^^^    ^^^^^^
                           The self-correction layer!
""")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 7: Summary
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 7: Summary -- What We Learned")
print("=" * 70)
print("""
  +------------------+----------------------------------+
  |   Component      |   Purpose                        |
  +------------------+----------------------------------+
  | Relevance Grader | LLM judges if docs are relevant  |
  |                  | CORRECT / AMBIGUOUS / INCORRECT   |
  +------------------+----------------------------------+
  | Query Reformer   | Rewrites bad queries to get       |
  |                  | better retrieval results          |
  +------------------+----------------------------------+
  | Knowledge        | Extracts only useful sentences    |
  | Extractor        | from partially-relevant docs      |
  +------------------+----------------------------------+
  | Decision Logic   | Routes to correct action based    |
  |                  | on grading results                |
  +------------------+----------------------------------+

  CRAG vs Basic RAG:
    Basic RAG: Fast but blindly trusts retriever
    CRAG:      Slower (extra LLM calls) but self-correcting

  WHEN TO USE CRAG:
    - When accuracy matters more than speed
    - When your knowledge base has gaps
    - When users ask questions outside your domain
    - When you need to prevent hallucinations

  COST: 1-4 extra LLM calls per query (for grading + reformulation)

  NEXT STEP -> Step 10: Self-RAG
    AI generates answer AND self-critiques it
    3 checks: Is doc relevant? Is answer supported? Is answer useful?
""")

print("[OK] Step 9 complete! You now understand Corrective RAG (CRAG).\n")
