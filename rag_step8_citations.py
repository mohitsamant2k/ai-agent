"""
Step 8: Source Citations — Trust Through Transparency

Building on Step 7 (Re-ranking), we add CITATION TRACKING.

THE PROBLEM:
  User: "How does Python handle errors?"
  Bot:  "Python uses try-except blocks..." 
  User: "Says who? Where did you get that? Can I trust this?"

THE SOLUTION — Source Citations:
  Bot: "Python uses try-except blocks... [1]"
  
  Sources:
    [1] python_guide, chunk 5 (relevance: +7.8)
        "Error handling in Python uses try-except blocks.
         You can catch specific exceptions like ValueError..."
    [2] python_guide, chunk 6 (relevance: +3.2)
        "The finally block runs cleanup code regardless..."

  This lets users:
    ✅ Verify the answer against original text
    ✅ Read more context from the source
    ✅ Know WHICH document was used
    ✅ See HOW relevant the source was (confidence)

This file demonstrates:
  Part 1: Citation data model — what to track
  Part 2: LLM prompt engineering for citations
  Part 3: Full RAG pipeline with inline citations
  Part 4: Citation quality — verifying answers match sources
  Part 5: Interactive chatbot with citations
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
# SETUP: Full pipeline (Hybrid + Rerank from Steps 6-7)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("SETUP: Building Full RAG Pipeline")
print("=" * 70)

documents = {
    "python_guide": PYTHON_DOC,
    "ai_ml_guide": AI_DOC,
    "webdev_guide": WEBDEV_DOC,
}

# ── Chunk with paragraph tracking ───────────────────────────────────────────

all_chunks = []
all_ids = []
all_metadatas = []

for doc_name, content in documents.items():
    # Split into paragraphs first for citation tracking
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    
    chunks = recursive_chunk(content, chunk_size=400, chunk_overlap=50)
    for i, chunk in enumerate(chunks):
        # Find which paragraph this chunk best matches
        best_para = 0
        best_overlap = 0
        for p_idx, para in enumerate(paragraphs):
            # Count overlapping words
            chunk_words = set(chunk.lower().split())
            para_words = set(para.lower().split())
            overlap = len(chunk_words & para_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_para = p_idx + 1  # 1-indexed
        
        all_chunks.append(chunk)
        all_ids.append(f"{doc_name}::chunk_{i}")
        all_metadatas.append({
            "source": doc_name,
            "chunk_index": i,
            "paragraph": best_para,
            "char_start": content.find(chunk[:50]),  # approximate position
        })

print(f"📚 {len(all_chunks)} chunks from {len(documents)} documents")

# ── BM25 + ChromaDB + Cross-encoder ────────────────────────────────────────

tokenized_chunks = [chunk.lower().split() for chunk in all_chunks]
bm25 = BM25Okapi(tokenized_chunks)

client = chromadb.Client()
collection = client.create_collection(name="citations_kb")
collection.add(documents=all_chunks, ids=all_ids, metadatas=all_metadatas)

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("✅ BM25 + ChromaDB + Cross-encoder ready")

# ── Search functions ────────────────────────────────────────────────────────

def bm25_search(query, top_k=5):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [{
        "chunk": all_chunks[idx], "id": all_ids[idx],
        "source": all_metadatas[idx]["source"],
        "paragraph": all_metadatas[idx]["paragraph"],
        "chunk_index": all_metadatas[idx]["chunk_index"],
        "score": float(scores[idx]),
    } for idx in top_indices]

def vector_search(query, top_k=5):
    results = collection.query(query_texts=[query], n_results=top_k)
    return [{
        "chunk": results["documents"][0][i],
        "id": results["ids"][0][i],
        "source": results["metadatas"][0][i]["source"],
        "paragraph": results["metadatas"][0][i]["paragraph"],
        "chunk_index": results["metadatas"][0][i]["chunk_index"],
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
    merged = []
    for doc_id in sorted_ids:
        entry = doc_data[doc_id].copy()
        entry["rrf_score"] = rrf_scores[doc_id]
        merged.append(entry)
    return merged[:top_k]

def rerank(query, results, top_k=3):
    if not results:
        return []
    pairs = [(query, r["chunk"]) for r in results]
    ce_scores = cross_encoder.predict(pairs)
    for i, result in enumerate(results):
        result["ce_score"] = float(ce_scores[i])
    return sorted(results, key=lambda x: x["ce_score"], reverse=True)[:top_k]

# ── Azure OpenAI ────────────────────────────────────────────────────────────

load_dotenv()
azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
model_name = os.environ.get("AZURE_OPENAI_MODEL", "gpt-4.1-mini")
base_url = azure_endpoint.replace("/chat/completions", "")
llm_client = OpenAI(base_url=base_url, api_key=azure_api_key)

def call_llm(prompt, system_msg="You are a helpful assistant."):
    response = llm_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=600,
    )
    return response.choices[0].message.content

print(f"✅ LLM ready: {model_name}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: Citation Data Model — What to Track
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 1: Citation Data Model — What Makes a Good Citation")
print("=" * 70)
print("""
  A citation needs to answer 3 questions:
  
  1. WHERE did this come from?
     → Document name, paragraph/section, chunk index
  
  2. WHAT exactly was the source text?
     → The actual passage used (so user can verify)
  
  3. HOW confident are we?
     → Cross-encoder relevance score

  ┌─────────────────────────────────────────────────┐
  │  Citation Object                                │
  ├─────────────────────────────────────────────────┤
  │  {                                               │
  │    "id": 1,                                      │
  │    "source": "python_guide",                     │
  │    "paragraph": 6,                               │
  │    "relevance": +7.81,                           │
  │    "passage": "Error handling in Python uses..."  │
  │  }                                               │
  └─────────────────────────────────────────────────┘
""")

# ── Build citation objects from retrieved chunks ────────────────────────────

def build_citations(query, top_k=3):
    """
    Retrieve and build structured citation objects.
    
    Returns:
        citations: List of citation dicts with all tracking info
        context: Formatted context string for LLM prompt
    """
    # Retrieve → Re-rank (full pipeline from Steps 6-7)
    candidates = hybrid_search(query, top_k=10)
    reranked = rerank(query, candidates, top_k=top_k)
    
    citations = []
    context_parts = []
    
    for i, result in enumerate(reranked):
        citation_num = i + 1
        
        citation = {
            "id": citation_num,
            "source": result["source"],
            "paragraph": result["paragraph"],
            "chunk_index": result["chunk_index"],
            "relevance": result["ce_score"],
            "passage": result["chunk"].strip(),
        }
        citations.append(citation)
        
        # Format context with citation markers for LLM
        context_parts.append(
            f"[Source {citation_num}] (from: {result['source']}, "
            f"paragraph {result['paragraph']}):\n"
            f"{result['chunk'].strip()}"
        )
    
    context = "\n\n".join(context_parts)
    return citations, context


# Demo
print("── Demo: Building citations for a query ──\n")

demo_query = "How does Python handle errors and exceptions?"
citations, context = build_citations(demo_query)

print(f"🔍 Query: \"{demo_query}\"\n")
print(f"Found {len(citations)} cited sources:\n")

for c in citations:
    emoji = "🟢" if c["relevance"] > 3 else "🟡" if c["relevance"] > 0 else "🔴"
    print(f"  [{c['id']}] {emoji} {c['source']}, paragraph {c['paragraph']}")
    print(f"      Relevance: {c['relevance']:+.2f}")
    preview = c["passage"][:100].replace("\n", " ")
    print(f"      \"{preview}...\"")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: Prompt Engineering for Citations
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 2: Prompt Engineering — Making LLM Cite Its Sources")
print("=" * 70)
print()
print("The KEY is telling the LLM to use [1], [2], [3] markers in its answer.")
print("This way we can link each claim back to a specific source.\n")

CITATION_SYSTEM_PROMPT = """You are a knowledgeable assistant that ALWAYS cites sources.

RULES:
1. Answer ONLY using the provided sources
2. Add citation markers [1], [2], [3] after each claim
3. Each citation marker must reference a real source number
4. If sources don't contain the answer, say "I don't have enough information"
5. Keep answers clear and concise

EXAMPLE FORMAT:
Python uses try-except blocks for error handling [1]. The finally block 
ensures cleanup code runs regardless of exceptions [2]."""

CITATION_USER_PROMPT = """Sources:
---
{context}
---

Question: {query}

Answer with inline citations [1], [2], etc.:"""


def rag_with_citations(query):
    """
    Full RAG pipeline with citation tracking.
    
    Returns:
        dict with answer, citations, and metadata
    """
    t_start = time.time()
    
    # Step 1: Build citations (retrieve + rerank)
    citations, context = build_citations(query, top_k=3)
    t_retrieve = time.time()
    
    # Step 2: Generate answer with citation instructions
    prompt = CITATION_USER_PROMPT.format(context=context, query=query)
    answer = call_llm(prompt, CITATION_SYSTEM_PROMPT)
    t_end = time.time()
    
    return {
        "query": query,
        "answer": answer,
        "citations": citations,
        "timing": {
            "retrieve_rerank": t_retrieve - t_start,
            "llm": t_end - t_retrieve,
            "total": t_end - t_start,
        },
    }


# ── Demo: Citation prompt in action ─────────────────────────────────────────

print("── Demo: LLM answers with inline citations ──\n")

demo_queries = [
    "How does Python handle errors and exceptions?",
    "What is RAG and how does it help AI?",
    "How does React build user interfaces?",
]

for query in demo_queries:
    result = rag_with_citations(query)
    
    print(f"🔍 \"{query}\"\n")
    print(f"📝 Answer:")
    print(f"   {result['answer']}\n")
    print(f"📚 Sources cited:")
    for c in result["citations"]:
        emoji = "🟢" if c["relevance"] > 3 else "🟡" if c["relevance"] > 0 else "🔴"
        print(f"   [{c['id']}] {emoji} {c['source']}, ¶{c['paragraph']} "
              f"(relevance: {c['relevance']:+.1f})")
    print(f"\n   ⏱ {result['timing']['total']:.1f}s "
          f"(retrieve: {result['timing']['retrieve_rerank']:.2f}s, "
          f"llm: {result['timing']['llm']:.1f}s)")
    print(f"{'─' * 70}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: Rich Citation Display
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 3: Rich Citation Display — Production-Style Output")
print("=" * 70)
print()


def display_cited_answer(result):
    """
    Display a RAG answer with full citation details.
    Production-style output with expandable sources.
    """
    print(f"┌{'─' * 68}┐")
    print(f"│  🔍 Question: {result['query'][:52]:<52} │")
    print(f"├{'─' * 68}┤")
    
    # Answer section
    answer_lines = []
    words = result["answer"].split()
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 > 64:
            answer_lines.append(current_line)
            current_line = word
        else:
            current_line = f"{current_line} {word}".strip()
    if current_line:
        answer_lines.append(current_line)
    
    print(f"│  {'📝 Answer:':<66} │")
    for line in answer_lines:
        print(f"│  {line:<66} │")
    
    print(f"├{'─' * 68}┤")
    print(f"│  {'📚 Sources:':<66} │")
    
    for c in result["citations"]:
        emoji = "🟢" if c["relevance"] > 3 else "🟡" if c["relevance"] > 0 else "🔴"
        header = f"  [{c['id']}] {emoji} {c['source']}, paragraph {c['paragraph']}"
        print(f"│{header:<68} │")
        
        relevance_str = f"      Relevance: {c['relevance']:+.2f}"
        print(f"│{relevance_str:<68} │")
        
        # Show passage preview (truncated)
        passage_preview = c["passage"][:120].replace("\n", " ")
        passage_line = f"      \"{passage_preview}...\""
        if len(passage_line) > 66:
            passage_line = passage_line[:63] + "...\""
        print(f"│  {passage_line:<66} │")
        print(f"│{'':68} │")
    
    timing = (f"  ⏱ Total: {result['timing']['total']:.1f}s | "
              f"Retrieve: {result['timing']['retrieve_rerank']:.2f}s | "
              f"LLM: {result['timing']['llm']:.1f}s")
    print(f"│{timing:<68} │")
    print(f"└{'─' * 68}┘")


# Demo rich display
rich_queries = [
    "What are Python decorators and how do they modify functions?",
    "What is transfer learning and why does it reduce training cost?",
]

for query in rich_queries:
    result = rag_with_citations(query)
    display_cited_answer(result)
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: Citation Verification — Does the Answer Match Sources?
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 4: Citation Verification — Checking Answer Quality")
print("=" * 70)
print()
print("How do we know the LLM actually used the cited sources?")
print("We can check:")
print("  1. Does the answer contain citation markers [1], [2], [3]?")
print("  2. Are the cited source numbers valid?")
print("  3. Does the answer text overlap with cited passages?")
print()


def verify_citations(result):
    """
    Verify that the LLM's answer properly cites its sources.
    
    Checks:
      1. Citation markers present in answer
      2. All referenced citation numbers are valid
      3. Answer content overlaps with cited passages
    
    Returns:
        dict with verification results
    """
    answer = result["answer"]
    citations = result["citations"]
    num_sources = len(citations)
    
    # Check 1: Find citation markers in the answer
    import re
    cited_nums = set()
    for match in re.finditer(r'\[(\d+)\]', answer):
        cited_nums.add(int(match.group(1)))
    
    has_citations = len(cited_nums) > 0
    
    # Check 2: Are cited numbers valid?
    valid_nums = set(range(1, num_sources + 1))
    invalid_refs = cited_nums - valid_nums
    all_refs_valid = len(invalid_refs) == 0
    
    # Check 3: Content overlap — do answer words appear in cited passages?
    answer_words = set(answer.lower().split())
    # Remove common words
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on",
                  "at", "to", "for", "of", "and", "or", "but", "with", "as",
                  "by", "it", "its", "this", "that", "from", "be", "has",
                  "have", "had", "not", "can", "will", "do", "does", "did",
                  "i", "you", "he", "she", "we", "they", "1", "2", "3"}
    answer_words -= stop_words
    
    source_words = set()
    for c in citations:
        source_words.update(c["passage"].lower().split())
    source_words -= stop_words
    
    overlap = answer_words & source_words
    overlap_ratio = len(overlap) / len(answer_words) if answer_words else 0
    
    # Check 4: Unused sources — sources retrieved but not cited
    unused = valid_nums - cited_nums
    
    return {
        "has_citations": has_citations,
        "citation_count": len(cited_nums),
        "cited_numbers": sorted(cited_nums),
        "invalid_refs": sorted(invalid_refs),
        "all_refs_valid": all_refs_valid,
        "content_overlap": overlap_ratio,
        "unused_sources": sorted(unused),
        "grade": (
            "A" if has_citations and all_refs_valid and overlap_ratio > 0.3 else
            "B" if has_citations and all_refs_valid else
            "C" if has_citations else
            "F"
        ),
    }


# ── Run verification on test queries ────────────────────────────────────────

print("── Verifying citation quality ──\n")

verify_queries = [
    "How does Python handle errors and exceptions?",
    "What is the difference between supervised and unsupervised learning?",
    "How does CSS Flexbox help with responsive design?",
    "What are quantum computers?",  # Not in our docs — should say "I don't know"
    "What is RAG and how does it reduce hallucinations?",
]

grades = []

for query in verify_queries:
    result = rag_with_citations(query)
    verification = verify_citations(result)
    grades.append(verification["grade"])
    
    grade_emoji = {"A": "🌟", "B": "✅", "C": "⚠️", "F": "❌"}
    
    print(f"🔍 \"{query}\"")
    print(f"   Grade: {grade_emoji[verification['grade']]} {verification['grade']}")
    print(f"   Citations found: {verification['citation_count']} "
          f"(refs: {verification['cited_numbers']})")
    print(f"   All refs valid: {'✓' if verification['all_refs_valid'] else '✗'}"
          f"{' — invalid: ' + str(verification['invalid_refs']) if verification['invalid_refs'] else ''}")
    print(f"   Content overlap: {verification['content_overlap']:.0%}")
    if verification["unused_sources"]:
        print(f"   Unused sources: {verification['unused_sources']}")
    
    # Show short answer preview
    preview = result["answer"][:120].replace("\n", " ")
    print(f"   Answer: \"{preview}...\"")
    print()

# Summary
grade_counts = {g: grades.count(g) for g in ["A", "B", "C", "F"]}
print(f"── Citation Quality Summary ──")
print(f"   🌟 A (great): {grade_counts.get('A', 0)}")
print(f"   ✅ B (good):  {grade_counts.get('B', 0)}")
print(f"   ⚠️  C (weak):  {grade_counts.get('C', 0)}")
print(f"   ❌ F (none):  {grade_counts.get('F', 0)}")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: Interactive Chatbot with Citations
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 5: Interactive RAG Chatbot with Citations")
print("=" * 70)
print()
print("  Ask questions about Python, AI/ML, or Web Development!")
print("  Every answer includes verifiable source citations.")
print()
print("  Type 'quit' to exit")
print("  Type 'sources' to see full source text for last answer")
print(f"{'─' * 70}")

last_result = None

while True:
    try:
        query = input("\n🔍 You: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    
    if not query:
        continue
    if query.lower() in ("quit", "exit", "q"):
        print("\n👋 Goodbye!")
        break
    
    if query.lower() == "sources" and last_result:
        print(f"\n📚 Full sources for: \"{last_result['query']}\"\n")
        for c in last_result["citations"]:
            print(f"  ── Source [{c['id']}]: {c['source']}, ¶{c['paragraph']} ──")
            print(f"  Relevance: {c['relevance']:+.2f}")
            # Show full passage text
            for line in c["passage"].split("\n"):
                print(f"  │ {line}")
            print()
        continue
    
    # Get answer with citations
    result = rag_with_citations(query)
    last_result = result
    
    # Verify citations
    verification = verify_citations(result)
    grade_emoji = {"A": "🌟", "B": "✅", "C": "⚠️", "F": "❌"}
    
    # Display answer
    print(f"\n🤖 Answer:")
    print(f"   {result['answer']}")
    
    # Display citation summary
    print(f"\n📚 Sources:")
    for c in result["citations"]:
        emoji = "🟢" if c["relevance"] > 3 else "🟡" if c["relevance"] > 0 else "🔴"
        preview = c["passage"][:80].replace("\n", " ")
        print(f"   [{c['id']}] {emoji} {c['source']}, ¶{c['paragraph']} "
              f"(rel: {c['relevance']:+.1f})")
        print(f"       \"{preview}...\"")
    
    print(f"\n   Citation grade: {grade_emoji[verification['grade']]} "
          f"{verification['grade']} | "
          f"⏱ {result['timing']['total']:.1f}s | "
          f"Type 'sources' for full text")
