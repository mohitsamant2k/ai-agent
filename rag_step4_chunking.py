"""
Step 4: Document Chunking — Splitting Real Documents for RAG

In Steps 1-3, our "documents" were single sentences:
  "Python decorators modify the behavior of functions"

Real documents are PAGES long. Problem:
  ❌ Embedding a whole 10-page doc → averaged meaning → bad search
  ❌ One sentence at a time → too granular → loses context

Chunking is the solution:
  ✅ Split documents into meaningful pieces (200-1000 chars)
  ✅ Each chunk is self-contained enough to be useful
  ✅ Small enough to have a focused embedding
  ✅ Overlap between chunks preserves context at boundaries

Think of it like:
  - Whole book = one embedding → "What is this book about?" → vague answer
  - One word = one embedding → "What does 'the' mean?" → useless
  - One paragraph = one embedding → specific, searchable, useful! ✅

This file demonstrates:
  1. The problem — why we need chunking
  2. Naive chunking — fixed character splits (simple but dumb)
  3. Sentence-based chunking — split on sentence boundaries
  4. Recursive chunking — the industry standard (LangChain-style)
  5. Overlap — why chunks should share edges
  6. Chunk size impact — small vs medium vs large chunks on search quality
  7. Real document → ChromaDB pipeline (end-to-end)
"""

import time
import numpy as np
import re
import textwrap
import chromadb
from rag_utils import (
    load_embedding_model, cosine_sim, recursive_chunk,
    PYTHON_DOC as SAMPLE_DOCUMENT, AI_DOC as AI_DOCUMENT,
)


# We'll use this model throughout
model = load_embedding_model()


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE_DOCUMENT and AI_DOCUMENT imported from rag_utils
# (as PYTHON_DOC → SAMPLE_DOCUMENT, AI_DOC → AI_DOCUMENT)
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: The Problem — why we need chunking
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PART 1: Why chunking matters")
print("=" * 65)

print(f"\n📄 Our Python document:")
print(f"   Characters: {len(SAMPLE_DOCUMENT):,}")
print(f"   Words: {len(SAMPLE_DOCUMENT.split()):,}")
print(f"   Paragraphs: {len(SAMPLE_DOCUMENT.split(chr(10)+chr(10)))}")
print(f"   ≈ {len(SAMPLE_DOCUMENT.split()) // 250} pages")

# Embed the WHOLE document as one vector
whole_doc_embedding = model.encode(SAMPLE_DOCUMENT)

# Embed individual paragraphs
paragraphs = [p.strip() for p in SAMPLE_DOCUMENT.split("\n\n") if p.strip()]
paragraph_embeddings = model.encode(paragraphs)

# Now search!
query = "How do decorators work in Python?"
query_embedding = model.encode(query)

# cosine_sim imported from rag_utils

# Search against WHOLE document (1 embedding)
whole_doc_score = cosine_sim(query_embedding, whole_doc_embedding)

# Search against individual paragraphs
paragraph_scores = [(i, cosine_sim(query_embedding, emb)) 
                     for i, emb in enumerate(paragraph_embeddings)]
paragraph_scores.sort(key=lambda x: x[1], reverse=True)

print(f"\n🔍 Query: \"{query}\"")
print(f"\n  Whole document (1 embedding): similarity = {whole_doc_score:.4f}")
print(f"  → Vague! The embedding averages ALL topics together")

print(f"\n  Individual paragraphs (chunked):")
for idx, score in paragraph_scores[:3]:
    preview = paragraphs[idx][:80].replace("\n", " ")
    emoji = "🟢" if score > 0.5 else "🟡"
    print(f"   {emoji} [{score:.4f}] \"{preview}...\"")

best_idx, best_score = paragraph_scores[0]
print(f"\n  💡 Best paragraph score ({best_score:.4f}) vs whole doc ({whole_doc_score:.4f})")
print(f"     Chunked search is {best_score/whole_doc_score:.1f}x more relevant!")
print(f"     AND it returns the exact paragraph about decorators, not the whole doc!")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: Naive chunking — fixed character splits
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("PART 2: Naive chunking — fixed character splits")
print("=" * 65)

def naive_chunk(text, chunk_size=200):
    """Split text into fixed-size character chunks. Simple but dumb."""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

chunks = naive_chunk(SAMPLE_DOCUMENT, chunk_size=200)

print(f"\n📏 Document: {len(SAMPLE_DOCUMENT)} chars → {len(chunks)} chunks of ~200 chars\n")

# Show a few chunks to see the problem
for i, chunk in enumerate(chunks[:4]):
    display = chunk.replace("\n", "↵ ")
    print(f"  Chunk {i}: \"{display}\"")
    print(f"           [{len(chunk)} chars]\n")

print("  ❌ Problems with naive chunking:")
print("     1. Cuts mid-sentence: '...Guido van Ros' | 'sum and first...'")
print("     2. Cuts mid-word sometimes!")
print("     3. No awareness of paragraph boundaries")
print("     4. A chunk might start in the middle of a thought")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: Sentence-based chunking
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("PART 3: Sentence-based chunking")
print("=" * 65)

def sentence_chunk(text, max_chunk_size=500):
    """Split text into chunks along sentence boundaries."""
    # Simple sentence splitting (handles Mr., Dr., etc. imperfectly)
    sentences = re.split(r'(?<=[.!?])\s+', text.replace("\n", " "))
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed the limit, save current chunk
        if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += (" " + sentence if current_chunk else sentence)
    
    # Don't forget the last chunk!
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

chunks = sentence_chunk(SAMPLE_DOCUMENT, max_chunk_size=500)

print(f"\n📏 Document: {len(SAMPLE_DOCUMENT)} chars → {len(chunks)} chunks (max 500 chars)\n")

for i, chunk in enumerate(chunks):
    # Show first and last few words
    words = chunk.split()
    preview_start = " ".join(words[:8])
    preview_end = " ".join(words[-5:])
    print(f"  Chunk {i}: \"{preview_start} ... {preview_end}\"")
    print(f"           [{len(chunk)} chars, {len(words)} words]\n")

print("  ✅ Better! Each chunk ends at a sentence boundary")
print("  ✅ No words or sentences are cut in half")
print("  ❌ Still doesn't respect paragraph structure")
print("  ❌ No overlap — context at boundaries is lost")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: Recursive chunking — the industry standard
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("PART 4: Recursive chunking (the industry standard)")
print("=" * 65)

# recursive_chunk imported from rag_utils
# It tries to split on the LARGEST meaningful boundary first:
#   1. Double newline (paragraph boundary)    ← best
#   2. Single newline (line boundary)
#   3. Sentence ending (. ! ?)
#   4. Space (word boundary)
#   5. Character (last resort)                ← worst

chunks = recursive_chunk(SAMPLE_DOCUMENT, chunk_size=500, chunk_overlap=50)

print(f"\n📏 Document: {len(SAMPLE_DOCUMENT)} chars → {len(chunks)} chunks")
print(f"   Settings: chunk_size=500, overlap=50\n")

for i, chunk in enumerate(chunks):
    words = chunk.split()
    preview = " ".join(words[:10])
    print(f"  Chunk {i}: \"{preview}...\"")
    print(f"           [{len(chunk)} chars, {len(words)} words]")
    
    # Show overlap with next chunk
    if i < len(chunks) - 1:
        # Find common text between end of this chunk and start of next
        overlap_text = ""
        this_end = chunk[-80:]
        next_start = chunks[i + 1][:80]
        # Simple overlap detection
        for length in range(min(len(this_end), len(next_start)), 5, -1):
            if this_end.endswith(next_start[:length]):
                overlap_text = next_start[:length]
                break
        if overlap_text:
            print(f"           🔗 Overlaps with chunk {i+1}: \"{overlap_text[:50]}...\"")
    print()

print("""  💡 Why recursive chunking is the standard:
     1. Tries paragraph boundaries FIRST (most meaningful)
     2. Falls back to sentences, then words, then characters
     3. Each chunk is roughly the same size (balanced)
     4. Overlap ensures no context is lost at boundaries
     5. This is what LangChain's RecursiveCharacterTextSplitter does!
""")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: Overlap explained — why chunks share edges
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PART 5: Why overlap matters")
print("=" * 65)

# Create a document where important info spans a boundary
boundary_doc = (
    "Python virtual environments create isolated spaces for projects. "
    "They prevent package conflicts between different projects. "
    "To create a virtual environment, use the venv module: python -m venv myenv. "  # ← boundary here
    "After creating it, activate with: source myenv/bin/activate on Linux "
    "or myenv\\Scripts\\activate on Windows. "
    "Once activated, pip install only affects this environment."
)

# Chunk WITHOUT overlap
no_overlap = sentence_chunk(boundary_doc, max_chunk_size=180)

# Chunk WITH overlap (manual for demo)
def sentence_chunk_with_overlap(text, max_chunk_size=180, overlap_sentences=1):
    sentences = re.split(r'(?<=[.!?])\s+', text.replace("\n", " "))
    
    chunks = []
    current_sentences = []
    current_len = 0
    
    for sentence in sentences:
        if current_len + len(sentence) > max_chunk_size and current_sentences:
            chunks.append(" ".join(current_sentences))
            # Keep last N sentences for overlap
            current_sentences = current_sentences[-overlap_sentences:]
            current_len = sum(len(s) + 1 for s in current_sentences)
        
        current_sentences.append(sentence)
        current_len += len(sentence) + 1
    
    if current_sentences:
        chunks.append(" ".join(current_sentences))
    
    return chunks

with_overlap = sentence_chunk_with_overlap(boundary_doc, max_chunk_size=180, overlap_sentences=1)

print(f"\n📄 Document about virtual environments ({len(boundary_doc)} chars)")

print(f"\n--- Without overlap ({len(no_overlap)} chunks) ---\n")
for i, chunk in enumerate(no_overlap):
    print(f"  Chunk {i}: \"{chunk}\"")
    print(f"           [{len(chunk)} chars]\n")

print(f"--- With overlap ({len(with_overlap)} chunks) ---\n")
for i, chunk in enumerate(with_overlap):
    print(f"  Chunk {i}: \"{chunk}\"")
    print(f"           [{len(chunk)} chars]\n")

# Search to show the difference
query = "How to create and activate a virtual environment?"
query_emb = model.encode(query)

print(f"🔍 Query: \"{query}\"\n")

print("  Without overlap:")
for i, chunk in enumerate(no_overlap):
    score = cosine_sim(query_emb, model.encode(chunk))
    emoji = "🟢" if score > 0.6 else "🟡" if score > 0.4 else "🔴"
    preview = chunk[:60]
    print(f"   {emoji} [{score:.4f}] Chunk {i}: \"{preview}...\"")

print("\n  With overlap:")
for i, chunk in enumerate(with_overlap):
    score = cosine_sim(query_emb, model.encode(chunk))
    emoji = "🟢" if score > 0.6 else "🟡" if score > 0.4 else "🔴"
    preview = chunk[:60]
    print(f"   {emoji} [{score:.4f}] Chunk {i}: \"{preview}...\"")

print("""
  💡 Overlap ensures that information near chunk boundaries
     appears in BOTH chunks. Without overlap, a question about
     "create AND activate" might miss the answer because "create"
     is in chunk 1 and "activate" is in chunk 2!
     
     Typical overlap: 10-20% of chunk size
       chunk_size=500 → overlap=50-100 chars
""")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 6: Chunk size impact on search quality
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PART 6: Chunk size impact on search quality")
print("=" * 65)

# Try different chunk sizes on the same document
sizes = [100, 300, 500, 1000]
query = "How does transfer learning reduce training cost?"
query_emb = model.encode(query)

# Use the AI document which has a paragraph about transfer learning
test_doc = AI_DOCUMENT

print(f"\n📄 AI/ML document: {len(test_doc)} chars")
print(f"🔍 Query: \"{query}\"\n")

for size in sizes:
    chunks = recursive_chunk(test_doc, chunk_size=size, chunk_overlap=size // 10)
    
    # Find best chunk
    best_score = 0
    best_chunk = ""
    for chunk in chunks:
        score = cosine_sim(query_emb, model.encode(chunk))
        if score > best_score:
            best_score = score
            best_chunk = chunk
    
    preview = best_chunk[:70].replace("\n", " ")
    words_in_best = len(best_chunk.split())
    
    print(f"  chunk_size={size:>4} → {len(chunks):>2} chunks | "
          f"best score: {best_score:.4f} | "
          f"best chunk: {words_in_best} words")
    print(f"                    \"{preview}...\"\n")

print("""  💡 Chunk size tradeoffs:
     ┌──────────┬────────────────────┬────────────────────────┐
     │   Size   │    Pros            │    Cons                │
     ├──────────┼────────────────────┼────────────────────────┤
     │ Small    │ Very specific      │ Loses context          │
     │ (100)    │ High similarity    │ Too many chunks        │
     │          │ scores             │ May miss the answer    │
     ├──────────┼────────────────────┼────────────────────────┤
     │ Medium   │ Good balance       │ Sweet spot varies      │
     │ (300-500)│ Enough context     │ by document type       │
     │          │ Focused meaning    │                        │
     ├──────────┼────────────────────┼────────────────────────┤
     │ Large    │ Full context       │ Diluted embedding      │
     │ (1000+)  │ Complete thoughts  │ Lower similarity       │
     │          │                    │ Fewer chunks           │
     └──────────┴────────────────────┴────────────────────────┘
     
     Rule of thumb: Start with 500 chars, adjust based on results.
     Short docs (FAQ) → smaller chunks (200-300)
     Long docs (manuals) → larger chunks (500-1000)
""")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 7: End-to-end pipeline — Real document → chunks → ChromaDB → search
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PART 7: Complete chunking pipeline → ChromaDB")
print("=" * 65)

# Simulate loading multiple "files"
documents_to_load = {
    "python_guide.txt": SAMPLE_DOCUMENT,
    "ml_guide.txt": AI_DOCUMENT,
}

# Step 1: Chunk all documents
print("\n📋 Step 1: Chunking documents...\n")

all_chunks = []      # The text chunks
all_ids = []         # Unique IDs
all_metadatas = []   # Source info

for filename, content in documents_to_load.items():
    chunks = recursive_chunk(content, chunk_size=400, chunk_overlap=50)
    
    print(f"  📄 {filename}: {len(content)} chars → {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks):
        chunk_id = f"{filename}::chunk_{i}"
        all_chunks.append(chunk)
        all_ids.append(chunk_id)
        all_metadatas.append({
            "source": filename,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "char_count": len(chunk),
        })

print(f"\n  Total chunks to store: {len(all_chunks)}")

# Step 2: Store in ChromaDB
print("\n📋 Step 2: Storing in ChromaDB...\n")

client = chromadb.Client()

# Delete collection if it exists (clean start)
try:
    client.delete_collection("chunked_knowledge")
except:
    pass

collection = client.create_collection(
    name="chunked_knowledge",
    metadata={"description": "Chunked documents for RAG"}
)

start = time.time()
collection.add(
    documents=all_chunks,
    ids=all_ids,
    metadatas=all_metadatas,
)
elapsed = time.time() - start
print(f"  ✅ Stored {collection.count()} chunks in {elapsed:.2f}s")

# Step 3: Search!
print("\n📋 Step 3: Searching chunked documents...\n")

test_queries = [
    "How do Python decorators work?",
    "What is transfer learning?",
    "How to create a virtual environment?",
    "What are the types of machine learning?",
    "How does error handling work in Python?",
]

for query in test_queries:
    results = collection.query(query_texts=[query], n_results=2)
    
    print(f"  🔍 \"{query}\"")
    for i in range(len(results["documents"][0])):
        doc = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        similarity = 1 - (distance ** 2 / 2)
        
        source = meta["source"]
        chunk_idx = meta["chunk_index"]
        
        # Show first 80 chars of the chunk
        preview = doc[:80].replace("\n", " ")
        emoji = "🟢" if similarity > 0.5 else "🟡" if similarity > 0.3 else "🔴"
        print(f"   {emoji} [{similarity:.4f}] [{source} chunk {chunk_idx}]")
        print(f"      \"{preview}...\"")
    print()

# Step 4: Show the full pipeline
print("""
  ═══════════════════════════════════════════════════════
  THE COMPLETE RAG CHUNKING PIPELINE:
  ═══════════════════════════════════════════════════════
  
  📁 Load documents (PDF, TXT, HTML, etc.)
       ↓
  ✂️  Chunk (recursive split, 400 chars, 50 overlap)
       ↓
  🔢 Embed (each chunk → 384-dim vector, done by ChromaDB)
       ↓
  💾 Store (chunks + embeddings + metadata → ChromaDB)
       ↓
  🔍 Query ("How do decorators work?")
       ↓
  📎 Retrieve top-K chunks (most similar to query)
       ↓
  🤖 Send to LLM (chunks become context for the answer)
  
  We've now built Steps 1-4. Only the last step (LLM) remains!
  ═══════════════════════════════════════════════════════
""")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 8: Interactive search on chunked documents
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PART 8: Search your chunked documents! (type 'quit' to exit)")
print("=" * 65)

print(f"\n  Documents: {list(documents_to_load.keys())}")
print(f"  Total chunks: {collection.count()}")
print(f"  Tip: type 'source:python_guide.txt your query' to filter\n")

while True:
    user_input = input("  🔍 Search: ").strip()
    if user_input.lower() in ("quit", "exit", "q"):
        print("\n  Goodbye! 👋")
        break
    if not user_input:
        continue
    
    # Check for source filter
    source_filter = None
    query = user_input
    if user_input.startswith("source:"):
        parts = user_input.split(" ", 1)
        source_filter = parts[0].replace("source:", "")
        query = parts[1] if len(parts) > 1 else ""
        if not query:
            print("  ⚠️  Please provide a search query after the source filter")
            continue
    
    kwargs = {"query_texts": [query], "n_results": 3}
    if source_filter:
        kwargs["where"] = {"source": source_filter}
        print(f"  📁 Filtering: source={source_filter}")
    
    try:
        results = collection.query(**kwargs)
        print()
        for i in range(len(results["documents"][0])):
            doc = results["documents"][0][i]
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            similarity = 1 - (distance ** 2 / 2)
            
            emoji = "🟢" if similarity > 0.5 else "🟡" if similarity > 0.3 else "🔴"
            print(f"   {emoji} [{similarity:.4f}] [{meta['source']} chunk {meta['chunk_index']}]")
            # Show full chunk text (wrapped nicely)
            wrapped = textwrap.fill(doc, width=60, initial_indent="      ", 
                                     subsequent_indent="      ")
            print(wrapped)
            print()
    except Exception as e:
        print(f"  ❌ Error: {e}\n")
