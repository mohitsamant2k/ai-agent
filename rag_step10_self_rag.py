"""
Step 10: Self-RAG — AI That Critiques Its Own Answers

Building on Step 9 (CRAG), we add SELF-REFLECTION after generation.

CRAG vs Self-RAG:
  ┌──────────────────────────────────────────────────────────────┐
  │  CRAG (Step 9)           │  Self-RAG (Step 10)              │
  ├──────────────────────────────────────────────────────────────┤
  │  Checks BEFORE answering │  Checks AFTER answering          │
  │  "Are these docs good?"  │  "Is my answer good?"            │
  │  Grades the DOCUMENTS    │  Grades the ANSWER               │
  │  Decides: use/skip/retry │  Decides: keep/fix/regenerate    │
  └──────────────────────────────────────────────────────────────┘

Self-RAG runs 3 reflection checks on EVERY answer:

  Check 1: IsRel  — Is the document relevant to the question?
  Check 2: IsSup  — Is my answer actually supported by the document?
  Check 3: IsUse  — Is my answer useful to the user?

  If ANY check fails → retry with a different approach.

  Think of it like a student who:
    1. Writes an essay
    2. Re-reads it asking "Did I use the right sources?"
    3. "Did I actually back up my claims?"
    4. "Would the teacher find this helpful?"
    5. If not → rewrites before submitting

This file demonstrates:
  Part 1: The 3 reflection checks (IsRel, IsSup, IsUse)
  Part 2: Self-RAG pipeline with retry logic
  Part 3: Comparison: Basic RAG vs CRAG vs Self-RAG
  Part 4: When Self-RAG catches mistakes CRAG misses
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
# SETUP: Full pipeline (reused from Steps 6-9)
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
collection = client.create_collection(name="selfrag_kb")
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
# PART 1: The 3 Reflection Checks
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 1: The 3 Self-Reflection Checks")
print("=" * 70)
print()
print("After generating an answer, Self-RAG asks 3 questions:")
print()
print("  Check 1 (IsRel):  Is this document relevant to the question?")
print("  Check 2 (IsSup):  Is the answer supported by the document?")
print("  Check 3 (IsUse):  Is the answer actually useful?")
print()
print("Each check returns: YES, PARTIALLY, or NO")
print()

# ── Check 1: IsRel — Document Relevance ─────────────────────────────────────

ISREL_SYSTEM = """You are evaluating whether a retrieved document is relevant to a question.

Respond with ONLY one of:
- YES: The document contains information directly related to the question
- PARTIALLY: The document is somewhat related but missing key information
- NO: The document is not relevant to the question"""

ISREL_PROMPT = """Question: {query}

Document:
---
{document}
---

Is this document relevant to the question? Respond with ONLY: YES, PARTIALLY, or NO"""


def check_is_relevant(query, document):
    """Check 1: Is the document relevant to the question?"""
    prompt = ISREL_PROMPT.format(query=query, document=document)
    response = call_llm(prompt, ISREL_SYSTEM, temperature=0.0)
    upper = response.strip().upper()
    if "PARTIALLY" in upper:
        return "PARTIALLY"
    elif "YES" in upper:
        return "YES"
    else:
        return "NO"


# ── Check 2: IsSup — Answer Supported by Document ──────────────────────────

ISSUP_SYSTEM = """You are evaluating whether an answer is actually supported by the 
source document. An answer is supported if the document contains evidence for the 
claims made in the answer.

Respond with ONLY one of:
- YES: Every claim in the answer is supported by the document
- PARTIALLY: Some claims are supported, but some are not in the document
- NO: The answer makes claims not found in the document (hallucination)"""

ISSUP_PROMPT = """Question: {query}

Document (source):
---
{document}
---

Generated Answer:
---
{answer}
---

Is this answer supported by the document? Respond with ONLY: YES, PARTIALLY, or NO"""


def check_is_supported(query, document, answer):
    """Check 2: Is the answer actually backed by the document?"""
    prompt = ISSUP_PROMPT.format(query=query, document=document, answer=answer)
    response = call_llm(prompt, ISSUP_SYSTEM, temperature=0.0)
    upper = response.strip().upper()
    if "PARTIALLY" in upper:
        return "PARTIALLY"
    elif "YES" in upper:
        return "YES"
    else:
        return "NO"


# ── Check 3: IsUse — Answer Usefulness ──────────────────────────────────────

ISUSE_SYSTEM = """You are evaluating whether an answer is useful to the user.
A useful answer directly addresses the question, is clear, and provides actionable 
or informative content.

Respond with ONLY one of:
- YES: The answer fully addresses the question and is helpful
- PARTIALLY: The answer addresses part of the question or is vague
- NO: The answer does not help the user (too generic, off-topic, or wrong)"""

ISUSE_PROMPT = """Question: {query}

Answer:
---
{answer}
---

Is this answer useful for someone asking this question? 
Respond with ONLY: YES, PARTIALLY, or NO"""


def check_is_useful(query, answer):
    """Check 3: Is the answer useful to the user?"""
    prompt = ISUSE_PROMPT.format(query=query, answer=answer)
    response = call_llm(prompt, ISUSE_SYSTEM, temperature=0.0)
    upper = response.strip().upper()
    if "PARTIALLY" in upper:
        return "PARTIALLY"
    elif "YES" in upper:
        return "YES"
    else:
        return "NO"


# ── Demo: Run all 3 checks on sample answers ───────────────────────────────

print("-- Demo: Testing the 3 reflection checks --\n")

# Good answer scenario
demo_query = "How does Python handle errors?"
demo_doc = """Error handling in Python uses try-except blocks. You can catch specific 
exceptions like ValueError or TypeError. The finally block runs cleanup code 
regardless of whether an exception occurred. Custom exceptions can be created 
by inheriting from the Exception base class."""
demo_good_answer = ("Python handles errors using try-except blocks to catch "
                    "specific exceptions like ValueError. The finally block "
                    "runs cleanup code, and you can create custom exceptions "
                    "by inheriting from Exception.")
demo_bad_answer = ("Python handles errors using try-except blocks. Python also "
                   "supports async/await for concurrent programming and has "
                   "built-in support for multithreading with the GIL.")

print("  SCENARIO A: Good answer (faithful to source)\n")
print(f"  Q: \"{demo_query}\"")
print(f"  A: \"{demo_good_answer[:80]}...\"\n")

is_rel = check_is_relevant(demo_query, demo_doc)
is_sup = check_is_supported(demo_query, demo_doc, demo_good_answer)
is_use = check_is_useful(demo_query, demo_good_answer)

print(f"    IsRel (doc relevant?):     {is_rel}")
print(f"    IsSup (answer supported?): {is_sup}")
print(f"    IsUse (answer useful?):    {is_use}")
print()

print("  SCENARIO B: Bad answer (hallucinated content)\n")
print(f"  Q: \"{demo_query}\"")
print(f"  A: \"{demo_bad_answer[:80]}...\"\n")

is_rel_b = check_is_relevant(demo_query, demo_doc)
is_sup_b = check_is_supported(demo_query, demo_doc, demo_bad_answer)
is_use_b = check_is_useful(demo_query, demo_bad_answer)

print(f"    IsRel (doc relevant?):     {is_rel_b}")
print(f"    IsSup (answer supported?): {is_sup_b}  <-- Should catch hallucination!")
print(f"    IsUse (answer useful?):    {is_use_b}")
print()

print("  KEY INSIGHT:")
print("    IsSup catches when the LLM makes up facts not in the source!")
print("    This is the MAIN difference from CRAG.")
print("    CRAG only checks docs. Self-RAG checks the ANSWER too.")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: Full Self-RAG Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 2: Full Self-RAG Pipeline")
print("=" * 70)
print()
print("The pipeline:")
print("  1. Retrieve documents")
print("  2. Generate answer")
print("  3. Run 3 reflection checks")
print("  4. Score the answer (0-3 based on checks)")
print("  5. If score too low -> retry with different docs/prompt")
print()


def generate_answer(query, context_chunks):
    """Generate an answer from context chunks."""
    context = "\n\n---\n\n".join(context_chunks)
    system_msg = """You are a helpful assistant. Answer using ONLY the provided context.
If the context doesn't contain enough information, say so clearly.
Be specific and cite details from the context."""

    prompt = f"""Context:
---
{context}
---

Question: {query}

Answer using only the context above:"""

    return call_llm(prompt, system_msg)


def reflect_on_answer(query, documents_used, answer):
    """
    Run all 3 reflection checks on an answer.

    Args:
        query: The user's question
        documents_used: List of document chunks used as context
        answer: The generated answer

    Returns:
        dict with check results and overall score
    """
    combined_doc = "\n\n".join(documents_used)

    # Run all 3 checks
    is_rel = check_is_relevant(query, combined_doc)
    is_sup = check_is_supported(query, combined_doc, answer)
    is_use = check_is_useful(query, answer)

    # Score: YES=1, PARTIALLY=0.5, NO=0
    score_map = {"YES": 1.0, "PARTIALLY": 0.5, "NO": 0.0}
    total_score = (
        score_map.get(is_rel, 0)
        + score_map.get(is_sup, 0)
        + score_map.get(is_use, 0)
    )

    return {
        "is_relevant": is_rel,
        "is_supported": is_sup,
        "is_useful": is_use,
        "score": total_score,         # 0 to 3
        "max_score": 3.0,
        "passed": total_score >= 2.0,  # Need at least 2/3 to pass
    }


def self_rag_pipeline(query, max_attempts=2):
    """
    Self-RAG pipeline with reflection and retry.

    Flow:
      1. Retrieve → Generate → Reflect
      2. If reflection score < 2.0 → retry with more docs or stricter prompt
      3. Return best answer across attempts

    Args:
        query: User's question
        max_attempts: How many generate-reflect cycles to try

    Returns:
        dict with answer, reflections, and metadata
    """
    t_start = time.time()

    attempts = []
    best_attempt = None

    for attempt_num in range(1, max_attempts + 1):
        attempt = {"attempt": attempt_num}

        # Step 1: Retrieve (get more docs on retry)
        top_k = 3 if attempt_num == 1 else 5
        results = retrieve(query, top_k=top_k)
        chunks_used = [r["chunk"] for r in results]
        sources = [r["source"] for r in results]
        attempt["sources"] = sources
        attempt["num_docs"] = len(chunks_used)

        # Step 2: Generate answer
        if attempt_num == 1:
            answer = generate_answer(query, chunks_used)
        else:
            # On retry: use a stricter prompt
            context = "\n\n---\n\n".join(chunks_used)
            strict_system = """You are a precise assistant. Answer using ONLY facts 
explicitly stated in the context. Do NOT add any information not found in the context.
If the context is insufficient, clearly state what is missing.
Every claim must be directly traceable to the context."""

            strict_prompt = f"""Context:
---
{context}
---

Question: {query}

IMPORTANT: Only include facts explicitly stated in the context above. 
Do not add any external knowledge. Answer:"""

            answer = call_llm(strict_prompt, strict_system, temperature=0.1)

        attempt["answer"] = answer

        # Step 3: Reflect on the answer
        reflection = reflect_on_answer(query, chunks_used, answer)
        attempt["reflection"] = reflection

        attempts.append(attempt)

        # Track best attempt
        if best_attempt is None or reflection["score"] > best_attempt["reflection"]["score"]:
            best_attempt = attempt

        # If passed, no need to retry
        if reflection["passed"]:
            break

    t_end = time.time()

    return {
        "query": query,
        "answer": best_attempt["answer"],
        "reflection": best_attempt["reflection"],
        "attempts": attempts,
        "total_attempts": len(attempts),
        "timing": t_end - t_start,
    }


# ── Demo: Full Self-RAG pipeline ───────────────────────────────────────────

print("-- Demo: Self-RAG in action --\n")

selfrag_tests = [
    {
        "query": "What are Python decorators and how are they used?",
        "expect": "Good docs exist, should pass on first try",
    },
    {
        "query": "What is quantum computing?",
        "expect": "Not in KB, should score low on all checks",
    },
    {
        "query": "How does supervised learning work?",
        "expect": "AI docs should have this, should pass",
    },
]

for test in selfrag_tests:
    query = test["query"]
    print(f"  {'=' * 58}")
    print(f"  Query: \"{query}\"")
    print(f"  Expected: {test['expect']}")
    print(f"  {'=' * 58}\n")

    result = self_rag_pipeline(query)

    for att in result["attempts"]:
        ref = att["reflection"]
        print(f"  Attempt {att['attempt']} ({att['num_docs']} docs):")
        print(f"    IsRel: {ref['is_relevant']:<10}  "
              f"IsSup: {ref['is_supported']:<10}  "
              f"IsUse: {ref['is_useful']}")
        print(f"    Score: {ref['score']}/{ref['max_score']}  "
              f"{'PASSED' if ref['passed'] else 'FAILED'}")
        print()

    # Show final answer (truncated)
    answer_preview = result["answer"][:200].replace("\n", " ")
    print(f"  Final Answer:")
    words = result["answer"].split()
    line = "    "
    for word in words[:50]:
        if len(line) + len(word) + 1 > 70:
            print(line)
            line = "    " + word
        else:
            line += " " + word if line.strip() else "    " + word
    if line.strip():
        print(line)

    print(f"\n  Time: {result['timing']:.1f}s "
          f"({result['total_attempts']} attempt(s))")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: Comparison — Basic RAG vs CRAG vs Self-RAG
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 3: Basic RAG vs CRAG vs Self-RAG")
print("=" * 70)
print()


def basic_rag(query):
    """Simple RAG without any correction."""
    t_start = time.time()
    results = retrieve(query, top_k=3)
    context = "\n\n".join([r["chunk"] for r in results])
    system_msg = """You are a helpful assistant. Answer using ONLY the provided context.
If the context doesn't contain the answer, say so."""
    prompt = f"Context:\n---\n{context}\n---\n\nQuestion: {query}\n\nAnswer:"
    answer = call_llm(prompt, system_msg)
    return {"answer": answer, "timing": time.time() - t_start}


def crag_pipeline(query):
    """CRAG: Check docs BEFORE generating."""
    t_start = time.time()
    results = retrieve(query, top_k=3)

    # Grade each document
    grades = []
    for r in results:
        prompt = (f"Question: {query}\n\nDocument:\n---\n{r['chunk']}\n---\n\n"
                  "Is this document relevant? Respond ONLY: CORRECT, AMBIGUOUS, or INCORRECT")
        system = "You are a relevance judge. Respond with ONLY one word."
        resp = call_llm(prompt, system, temperature=0.0).strip().upper()
        if "INCORRECT" in resp:
            grade = "INCORRECT"
        elif "AMBIGUOUS" in resp:
            grade = "AMBIGUOUS"
        else:
            grade = "CORRECT"
        grades.append(grade)

    correct_docs = [r for r, g in zip(results, grades) if g == "CORRECT"]
    ambiguous_docs = [r for r, g in zip(results, grades) if g == "AMBIGUOUS"]

    if correct_docs:
        context = "\n\n".join([r["chunk"] for r in correct_docs])
        decision = "USE_CORRECT"
    elif ambiguous_docs:
        context = "\n\n".join([r["chunk"] for r in ambiguous_docs])
        decision = "USE_AMBIGUOUS"
    else:
        return {
            "answer": "I don't have relevant information for this question.",
            "decision": "REJECTED",
            "timing": time.time() - t_start,
        }

    system_msg = "You are a helpful assistant. Answer using ONLY the provided context."
    prompt = f"Context:\n---\n{context}\n---\n\nQuestion: {query}\n\nAnswer:"
    answer = call_llm(prompt, system_msg)
    return {"answer": answer, "decision": decision, "timing": time.time() - t_start}


# ── Side-by-side comparison ─────────────────────────────────────────────────

comparison_queries = [
    "How does Python handle errors and exceptions?",   # In KB
    "What is blockchain technology?",                   # NOT in KB
    "How do neural networks learn from data?",          # Partially in KB
]

for query in comparison_queries:
    print(f"  {'~' * 58}")
    print(f"  Query: \"{query}\"")
    print(f"  {'~' * 58}\n")

    # Basic RAG
    basic = basic_rag(query)
    preview = basic["answer"][:120].replace("\n", " ")
    print(f"  [BASIC RAG] ({basic['timing']:.1f}s)")
    print(f"    {preview}...")
    print()

    # CRAG
    crag = crag_pipeline(query)
    preview = crag["answer"][:120].replace("\n", " ")
    decision = crag.get("decision", "N/A")
    print(f"  [CRAG] ({crag['timing']:.1f}s) Decision: {decision}")
    print(f"    {preview}...")
    print()

    # Self-RAG
    selfrag = self_rag_pipeline(query, max_attempts=2)
    ref = selfrag["reflection"]
    preview = selfrag["answer"][:120].replace("\n", " ")
    print(f"  [SELF-RAG] ({selfrag['timing']:.1f}s) "
          f"Score: {ref['score']}/{ref['max_score']} "
          f"Attempts: {selfrag['total_attempts']}")
    print(f"    IsRel={ref['is_relevant']} "
          f"IsSup={ref['is_supported']} "
          f"IsUse={ref['is_useful']}")
    print(f"    {preview}...")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: When Self-RAG Catches What CRAG Misses
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 4: Self-RAG's Unique Strength -- Catching Hallucinations")
print("=" * 70)
print()
print("CRAG checks: 'Are the docs relevant?' (BEFORE generation)")
print("Self-RAG checks: 'Is the ANSWER faithful?' (AFTER generation)")
print()
print("Scenario where CRAG fails but Self-RAG catches it:")
print("  1. CRAG finds relevant docs -> grades CORRECT -> generates")
print("  2. But LLM adds facts NOT in the docs (hallucination!)")
print("  3. CRAG doesn't catch this -- it already approved the docs")
print("  4. Self-RAG's IsSup check catches: 'Answer not supported!'")
print()

# Demonstrate with a tricky query
print("-- Demo: Hallucination detection --\n")

tricky_query = "What testing frameworks does Python support?"

# Retrieve docs
tricky_results = retrieve(tricky_query, top_k=3)
tricky_chunks = [r["chunk"] for r in tricky_results]

# Generate two answers -- one faithful, one with hallucination
faithful_system = """Answer using ONLY facts from the context. Be brief."""
halluc_system = """Answer the question comprehensively. Use the context as a starting 
point but add relevant details you know about the topic."""

context_str = "\n\n".join(tricky_chunks)
prompt = f"Context:\n---\n{context_str}\n---\n\nQuestion: {tricky_query}\n\nAnswer:"

faithful_answer = call_llm(prompt, faithful_system, temperature=0.0)
halluc_answer = call_llm(prompt, halluc_system, temperature=0.7)

print(f"  Query: \"{tricky_query}\"\n")

# Check faithful answer
print("  Answer A (Faithful to source):")
faithful_preview = faithful_answer[:150].replace("\n", " ")
print(f"    \"{faithful_preview}...\"\n")
ref_a = reflect_on_answer(tricky_query, tricky_chunks, faithful_answer)
print(f"    IsRel={ref_a['is_relevant']}  "
      f"IsSup={ref_a['is_supported']}  "
      f"IsUse={ref_a['is_useful']}  "
      f"Score={ref_a['score']}/{ref_a['max_score']}")
print()

# Check hallucinated answer
print("  Answer B (Added external knowledge):")
halluc_preview = halluc_answer[:150].replace("\n", " ")
print(f"    \"{halluc_preview}...\"\n")
ref_b = reflect_on_answer(tricky_query, tricky_chunks, halluc_answer)
print(f"    IsRel={ref_b['is_relevant']}  "
      f"IsSup={ref_b['is_supported']}  "  
      f"IsUse={ref_b['is_useful']}  "
      f"Score={ref_b['score']}/{ref_b['max_score']}")
print()

if ref_b["is_supported"] != "YES":
    print("  [!!] Self-RAG caught the hallucination via IsSup check!")
    print("       CRAG would have missed this -- docs were relevant,")
    print("       but the ANSWER went beyond what docs actually said.")
else:
    print("  Note: The LLM stayed faithful even with the loose prompt.")
    print("  In practice, hallucination happens more with complex topics.")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: Architecture Diagram + Summary
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 5: Architecture + Summary")
print("=" * 70)
print("""
  SELF-RAG PIPELINE
  =================

  User Query
       |
       v
  +-----------+
  | Retrieve  |  Hybrid Search + Re-rank
  | top-3     |
  +-----+-----+
        |
        v
  +-----------+
  | Generate  |  LLM creates answer from docs
  | Answer    |
  +-----+-----+
        |
        v
  +-----+-----+
  | REFLECT   |  3 self-checks:
  |           |
  | IsRel?  --+--> Is the doc relevant?
  | IsSup?  --+--> Is answer supported by doc?
  | IsUse?  --+--> Is answer useful to user?
  |           |
  +-----+-----+
        |
   Score >= 2?
    /       \\
  YES        NO
   |          |
   v          v
  DONE     RETRY
  Return   (more docs + stricter prompt)
  answer      |
              v
           Generate again -> Reflect again
              |
           Best answer wins

  COMPARISON:
  +-----------+------------------+---------------------------+
  | Approach  | What it checks   | When it checks            |
  +-----------+------------------+---------------------------+
  | Basic RAG | Nothing          | Never                     |
  | CRAG      | Documents        | BEFORE generation         |
  | Self-RAG  | Documents +      | BEFORE + AFTER generation |
  |           | Answer quality   |                           |
  +-----------+------------------+---------------------------+

  COST per query:
  +-----------+------------+--------+
  | Approach  | LLM calls  | Time   |
  +-----------+------------+--------+
  | Basic RAG | 1          | ~2s    |
  | CRAG      | 4-8        | ~5s    |
  | Self-RAG  | 4-10       | ~7s    |
  +-----------+------------+--------+
""")

print("  WHEN TO USE EACH:")
print("    Basic RAG  -> Speed matters, docs are high quality")
print("    CRAG       -> Docs might be irrelevant, need filtering")
print("    Self-RAG   -> Need maximum accuracy, catch hallucinations")
print()
print("  Self-RAG's UNIQUE power:")
print("    It catches when the LLM makes up facts not in the source.")
print("    CRAG can't do this -- it only checks docs, not answers.")
print()
print("  NEXT STEP -> Step 11: Query Transformation")
print("    Break complex questions into sub-queries")
print("    Expand queries with related terms")
print("    HyDE: generate hypothetical answer to improve search")
print()
print("[OK] Step 10 complete! You now understand Self-RAG.\n")
