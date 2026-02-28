"""
Step 5: Complete RAG — Retrieval + LLM Generation

This is the FINAL step! We connect everything together:
  Step 1: Embeddings (text → vectors)           ✅
  Step 2: Similarity (find relevant chunks)     ✅
  Step 3: ChromaDB (vector database)            ✅
  Step 4: Chunking (split real documents)       ✅
  Step 5: LLM Generation (THIS FILE)            ← NOW!

RAG = Retrieval-Augmented Generation
  1. RETRIEVAL: Find relevant chunks from your documents
  2. AUGMENTED: Stuff those chunks into the LLM's prompt
  3. GENERATION: LLM answers using YOUR data as context

Without RAG:
  User: "What are Python decorators?"
  LLM: Generic textbook answer (might be wrong or outdated)

With RAG:
  User: "What are Python decorators?"
  → Search vector DB → Find YOUR document's decorator paragraph
  → Send to LLM: "Based on this context: [your doc], answer: ..."
  LLM: Accurate answer based on YOUR specific documents!

This file demonstrates:
  1. Building the complete RAG pipeline
  2. Prompt engineering for RAG (how to structure the prompt)
  3. LLM without context vs with RAG context
  4. Handling "I don't know" (when context doesn't have the answer)
  5. Multi-turn conversation with RAG
  6. Interactive RAG chatbot
"""

import os
import time
import numpy as np
import chromadb
from dotenv import load_dotenv
from rag_utils import (
    load_embedding_model, recursive_chunk, distance_to_similarity,
    PYTHON_DOC, AI_DOC, WEBDEV_DOC,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Setup: Load model, create knowledge base (reusing Step 4's pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("SETUP: Building the knowledge base")
print("=" * 65)

print("\n📦 Loading embedding model...")
embed_model = load_embedding_model()

# Knowledge base documents imported from rag_utils (PYTHON_DOC, AI_DOC, WEBDEV_DOC)


# ── Chunking function imported from rag_utils ───────────────────────────────


# ── Build ChromaDB knowledge base ───────────────────────────────────────────

print("📚 Chunking documents...")

documents = {
    "python_guide": PYTHON_DOC,
    "ai_ml_guide": AI_DOC,
    "webdev_guide": WEBDEV_DOC,
}

client = chromadb.Client()
collection = client.create_collection(name="rag_knowledge_base")

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

collection.add(documents=all_chunks, ids=all_ids, metadatas=all_metadatas)
print(f"✅ Knowledge base ready: {collection.count()} chunks from {len(documents)} documents\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Setup OpenAI (or explain without it)
# ═══════════════════════════════════════════════════════════════════════════════

from openai import OpenAI

# Load Azure OpenAI credentials from .env file
load_dotenv()

azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
azure_model = os.environ.get("AZURE_OPENAI_MODEL", "gpt-4.1-mini")

if not azure_endpoint or not azure_api_key:
    raise EnvironmentError(
        "❌ Azure OpenAI credentials not found!\n"
        "   Create a .env file with:\n"
        "     AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/openai/v1/chat/completions\n"
        "     AZURE_OPENAI_API_KEY=your-api-key\n"
        "     AZURE_OPENAI_MODEL=gpt-4.1-mini"
    )

# Azure AI Foundry uses OpenAI-compatible endpoint
# Strip /chat/completions from the endpoint to get the base URL
base_url = azure_endpoint.rstrip("/")
if base_url.endswith("/chat/completions"):
    base_url = base_url[: -len("/chat/completions")]

llm_client = OpenAI(base_url=base_url, api_key=azure_api_key)
print(f"🔑 Azure OpenAI connected! Model: {azure_model}\n")


def call_llm(prompt, system_message="You are a helpful assistant.", temperature=0.3):
    """Call Azure OpenAI LLM with a prompt."""
    try:
        response = llm_client.chat.completions.create(
            model=azure_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[API Error: {e}]"


# ═══════════════════════════════════════════════════════════════════════════════
# The RAG Pipeline — the core function
# ═══════════════════════════════════════════════════════════════════════════════

def retrieve_context(query, n_results=3, source_filter=None):
    """Step 1 of RAG: Retrieve relevant chunks from the vector database."""
    kwargs = {"query_texts": [query], "n_results": n_results}
    if source_filter:
        kwargs["where"] = {"source": source_filter}
    
    results = collection.query(**kwargs)
    
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "distance": results["distances"][0][i],
            "similarity": distance_to_similarity(results["distances"][0][i]),
        })
    return chunks


def build_rag_prompt(question, context_chunks, max_context_chars=2000):
    """Step 2 of RAG: Build a prompt with retrieved context."""
    
    # Combine chunk texts into context string
    context_parts = []
    total_chars = 0
    for chunk in context_chunks:
        if total_chars + len(chunk["text"]) > max_context_chars:
            break
        context_parts.append(chunk["text"])
        total_chars += len(chunk["text"])
    
    context_str = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""Use the following context to answer the question. 
If the context does not have enough information to answer, say "I don't have enough information in my knowledge base to answer this."
Do NOT make up information that isn't in the context.
Keep your answer concise and focused.

CONTEXT:
{context_str}

QUESTION: {question}

ANSWER:"""
    
    return prompt


def rag_query(question, n_results=3, source_filter=None, show_details=True):
    """Complete RAG pipeline: Retrieve → Build Prompt → Generate Answer."""
    
    # Step 1: Retrieve
    chunks = retrieve_context(question, n_results=n_results, 
                               source_filter=source_filter)
    
    # Step 2: Build prompt
    prompt = build_rag_prompt(question, chunks)
    
    # Step 3: Generate answer
    system_msg = ("You are a helpful technical assistant. Answer questions "
                  "based ONLY on the provided context. Be concise and accurate.")
    answer = call_llm(prompt, system_message=system_msg)
    
    if show_details:
        print(f"\n  🔍 Retrieved {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            sim = chunk["similarity"]
            src = chunk["source"]
            preview = chunk["text"][:60].replace("\n", " ")
            emoji = "🟢" if sim > 0.5 else "🟡" if sim > 0.3 else "🔴"
            print(f"     {emoji} [{sim:.4f}] [{src}] \"{preview}...\"")
    
    return answer, chunks, prompt


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: The RAG Pipeline in Action
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PART 1: The complete RAG pipeline")
print("=" * 65)

question = "What are Python decorators and where are they used?"
print(f"\n  ❓ Question: \"{question}\"")

answer, chunks, prompt = rag_query(question)

print(f"\n  🤖 RAG Answer:")
for line in answer.split("\n"):
    print(f"     {line}")

print(f"""
  💡 What just happened:
     1. RETRIEVE: Searched {collection.count()} chunks → found {len(chunks)} relevant ones
     2. AUGMENT:  Built a prompt with context + question
     3. GENERATE: LLM answered using YOUR document's content
     
     The LLM didn't use its training data — it used YOUR knowledge base!
""")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: Prompt Engineering — what the LLM actually sees
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PART 2: What the LLM actually sees (the prompt)")
print("=" * 65)

question = "What is transfer learning?"
answer, chunks, prompt = rag_query(question, show_details=False)

print(f"\n  ❓ Question: \"{question}\"")
print(f"\n  📝 The FULL prompt sent to the LLM:")
print("  " + "─" * 55)
for line in prompt.split("\n"):
    print(f"  │ {line}")
print("  " + "─" * 55)

print(f"\n  🤖 LLM Response:")
for line in answer.split("\n"):
    print(f"     {line}")

print("""
  💡 The prompt has 3 key parts:
     1. INSTRUCTIONS: "Use context, don't make stuff up"
     2. CONTEXT: The actual chunks from your vector DB
     3. QUESTION: What the user asked
     
     This is the core of RAG — the LLM is GROUNDED in your data!
""")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: Without RAG vs With RAG
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PART 3: Without RAG vs With RAG")
print("=" * 65)

question = "What HTTP methods do REST APIs use?"

# Without RAG — LLM answers from its training data only
no_context_prompt = f"Answer this question concisely: {question}"
no_rag_answer = call_llm(no_context_prompt)

# With RAG — LLM answers using our documents
rag_answer, chunks, _ = rag_query(question, show_details=False)

print(f"\n  ❓ Question: \"{question}\"")

print(f"\n  ❌ WITHOUT RAG (LLM uses only training data):")
for line in no_rag_answer.split("\n"):
    print(f"     {line}")

print(f"\n  ✅ WITH RAG (LLM uses your documents):")
for line in rag_answer.split("\n"):
    print(f"     {line}")

print("""
  💡 Key differences:
     WITHOUT RAG → Generic answer from training data
       - Might be outdated
       - Might hallucinate (make up facts)
       - No source attribution
       
     WITH RAG → Answer grounded in YOUR documents
       - Always up-to-date (your docs are current)
       - Can't hallucinate (limited to context)
       - You know WHICH documents it used
       - Can be wrong only if your docs are wrong
""")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: Handling "I don't know"
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PART 4: When the knowledge base doesn't have the answer")
print("=" * 65)

# Ask something NOT in our knowledge base
out_of_scope_questions = [
    "What is the capital of France?",
    "How do I cook pasta?",
    "What was the score of yesterday's game?",
]

for question in out_of_scope_questions:
    print(f"\n  ❓ \"{question}\"")
    answer, chunks, _ = rag_query(question, show_details=False)
    
    # Show retrieved chunks (they'll have low similarity)
    best_sim = max(c["similarity"] for c in chunks)
    print(f"     Best chunk similarity: {best_sim:.4f} {'(low!)' if best_sim < 0.3 else ''}")
    print(f"     🤖 {answer[:100]}...")

print("""
  💡 A well-designed RAG system should say "I don't know" when:
     - Retrieved chunks have LOW similarity (< 0.3)
     - Context doesn't contain relevant information
     
     This is BETTER than a regular LLM which would just make up an answer!
     
     You can add a similarity threshold:
       if best_similarity < 0.3:
           return "I don't have info about this in my knowledge base"
""")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: RAG with source attribution
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PART 5: Source attribution — know WHERE the answer came from")
print("=" * 65)

questions = [
    "How does React work?",
    "What are the types of machine learning?",
    "How do I handle errors in Python?",
]

for question in questions:
    print(f"\n  ❓ \"{question}\"")
    answer, chunks, _ = rag_query(question, show_details=False)
    
    print(f"  🤖 {answer[:120]}...")
    
    # Show sources
    sources = set()
    for chunk in chunks:
        if chunk["similarity"] > 0.3:
            sources.add(chunk["source"])
    print(f"  📎 Sources: {', '.join(sources)}")

print("""
  💡 Source attribution is a HUGE advantage of RAG:
     - User can verify the answer by checking the source
     - You can show "Based on: python_guide, chunk 5"
     - Builds trust — the answer is traceable, not a black box
     - Legal/compliance teams love this!
""")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 6: The complete RAG architecture
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PART 6: The complete RAG architecture you've built!")
print("=" * 65)

print("""
  ╔═══════════════════════════════════════════════════════════════╗
  ║           THE RAG SYSTEM YOU BUILT (Steps 1-5)               ║
  ╠═══════════════════════════════════════════════════════════════╣
  ║                                                               ║
  ║  📁 INGESTION PIPELINE (happens once)                        ║
  ║  ─────────────────────────────────────                        ║
  ║  Documents → Chunk (Step 4) → Embed (Step 1) → Store (Step 3)║
  ║                                                               ║
  ║  📄 python_guide    ─┐                                       ║
  ║  📄 ai_ml_guide     ─┼→ Recursive Chunking → ChromaDB       ║
  ║  📄 webdev_guide    ─┘   (400 chars, 50 overlap)             ║
  ║                                                               ║
  ║  🔍 QUERY PIPELINE (happens every question)                  ║
  ║  ───────────────────────────────────────────                  ║
  ║                                                               ║
  ║  User Question                                                ║
  ║       ↓                                                       ║
  ║  Embed question (Step 1)                                      ║
  ║       ↓                                                       ║
  ║  Vector similarity search in ChromaDB (Steps 2+3)            ║
  ║       ↓                                                       ║
  ║  Retrieve top-K chunks                                        ║
  ║       ↓                                                       ║
  ║  Build prompt: instructions + context + question (Step 5)    ║
  ║       ↓                                                       ║
  ║  Send to LLM (GPT-4o-mini)                                   ║
  ║       ↓                                                       ║
  ║  Return answer + sources                                      ║
  ║                                                               ║
  ╚═══════════════════════════════════════════════════════════════╝
  
  You now understand every single piece of this system!
  This is the SAME architecture used by:
    - ChatGPT with file uploads
    - GitHub Copilot's @workspace
    - Enterprise knowledge bases
    - Customer support chatbots
    - Legal document search systems
""")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 7: Interactive RAG Chatbot
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PART 7: Your RAG Chatbot! (type 'quit' to exit)")
print("=" * 65)

print(f"""
  Knowledge base: {collection.count()} chunks from {len(documents)} documents
  Documents: {', '.join(documents.keys())}
  
  Commands:
    Just type a question → RAG search + LLM answer
    source:ai_ml_guide your question → Filter by document
    chunks → Show last retrieved chunks
    prompt → Show last prompt sent to LLM
    quit → Exit
""")

last_chunks = []
last_prompt = ""

while True:
    user_input = input("  💬 You: ").strip()
    if user_input.lower() in ("quit", "exit", "q"):
        print("\n  🎉 Congratulations! You've built a complete RAG system!")
        print("     From embeddings to LLM — you understand it all. 👋\n")
        break
    if not user_input:
        continue
    
    # Special commands
    if user_input.lower() == "chunks":
        if last_chunks:
            print("\n  📎 Last retrieved chunks:")
            for i, chunk in enumerate(last_chunks):
                sim = chunk["similarity"]
                src = chunk["source"]
                emoji = "🟢" if sim > 0.5 else "🟡" if sim > 0.3 else "🔴"
                print(f"     {emoji} [{sim:.4f}] [{src}]")
                text = chunk["text"][:100].replace("\n", " ")
                print(f"        \"{text}...\"")
            print()
        else:
            print("  No chunks yet — ask a question first!\n")
        continue
    
    if user_input.lower() == "prompt":
        if last_prompt:
            print("\n  📝 Last prompt sent to LLM:")
            print("  " + "─" * 55)
            for line in last_prompt.split("\n"):
                print(f"  │ {line}")
            print("  " + "─" * 55 + "\n")
        else:
            print("  No prompt yet — ask a question first!\n")
        continue
    
    # Parse source filter
    source_filter = None
    query = user_input
    if user_input.startswith("source:"):
        parts = user_input.split(" ", 1)
        source_filter = parts[0].replace("source:", "")
        query = parts[1] if len(parts) > 1 else ""
        if not query:
            print("  ⚠️  Please provide a question after the source filter\n")
            continue
        print(f"  📁 Filtering: {source_filter}")
    
    try:
        answer, last_chunks, last_prompt = rag_query(
            query, n_results=3, source_filter=source_filter, show_details=True
        )
        
        # Check if best similarity is too low
        best_sim = max(c["similarity"] for c in last_chunks)
        if best_sim < 0.25:
            print(f"\n  ⚠️  Low relevance ({best_sim:.4f}) — answer may not be accurate")
        
        print(f"\n  🤖 Answer:")
        for line in answer.split("\n"):
            print(f"     {line}")
        
        # Source attribution
        sources = set(c["source"] for c in last_chunks if c["similarity"] > 0.3)
        if sources:
            print(f"\n  📎 Sources: {', '.join(sources)}")
        print()
        
    except Exception as e:
        print(f"  ❌ Error: {e}\n")
