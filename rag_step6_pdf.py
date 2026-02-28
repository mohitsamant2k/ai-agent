"""
Step 6: PDF Support — Upload any PDF and ask questions about it

In Step 5, we hardcoded documents as Python strings.
Real RAG systems work with PDFs, Word docs, web pages, etc.

This step adds:
  ✅ Extract text from any PDF file (PyMuPDF)
  ✅ Handle multi-page documents with page tracking
  ✅ Preserve metadata: filename, page number, chunk index
  ✅ Ask questions and get answers WITH page citations
  ✅ Compare: which page answered? Was it accurate?

The pipeline becomes:
  📄 PDF file
    ↓
  📝 Extract text (per page)
    ↓
  ✂️  Chunk (recursive split, with page tracking)
    ↓
  💾 Store in ChromaDB (text + page metadata)
    ↓
  🔍 Query → retrieve relevant chunks
    ↓
  🤖 LLM answers with "Based on page X..."

This file demonstrates:
  1. PDF text extraction with PyMuPDF
  2. Page-aware chunking (each chunk knows its page number)
  3. Building a PDF knowledge base in ChromaDB
  4. RAG with page-level source citations
  5. Comparing results across different PDFs
  6. Interactive PDF Q&A chatbot

USAGE:
  - The script creates sample PDFs to demo the pipeline
  - You can also drop your own PDFs in the 'pdfs/' folder
"""

import os
import time
import chromadb
import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI
from rag_utils import recursive_chunk, distance_to_similarity


# ═══════════════════════════════════════════════════════════════════════════════
# Setup: Azure OpenAI
# ═══════════════════════════════════════════════════════════════════════════════

load_dotenv()

azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
azure_model = os.environ.get("AZURE_OPENAI_MODEL", "gpt-4.1-mini")

if not azure_endpoint or not azure_api_key:
    raise EnvironmentError(
        "❌ Azure OpenAI credentials not found!\n"
        "   Create a .env file with:\n"
        "     AZURE_OPENAI_ENDPOINT=https://...\n"
        "     AZURE_OPENAI_API_KEY=your-key\n"
        "     AZURE_OPENAI_MODEL=gpt-4.1-mini"
    )

base_url = azure_endpoint.rstrip("/")
if base_url.endswith("/chat/completions"):
    base_url = base_url[: -len("/chat/completions")]

llm_client = OpenAI(base_url=base_url, api_key=azure_api_key)


def call_llm(prompt, system_message="You are a helpful assistant.", temperature=0.3):
    """Call Azure OpenAI LLM."""
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
# PART 1: Create sample PDFs for demonstration
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PART 1: Creating sample PDFs")
print("=" * 65)

# Create a pdfs/ directory
pdf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdfs")
os.makedirs(pdf_dir, exist_ok=True)


def create_sample_pdf(filepath, title, pages_content):
    """Create a multi-page PDF with text content using PyMuPDF."""
    doc = fitz.open()  # new empty PDF

    for page_num, text in enumerate(pages_content):
        page = doc.new_page(width=595, height=842)  # A4 size

        # Title on first page
        if page_num == 0:
            page.insert_text(
                (50, 60),
                title,
                fontsize=18,
                fontname="helv",
                color=(0, 0, 0.6),
            )
            y_start = 100
        else:
            y_start = 50

        # Page number
        page.insert_text(
            (500, 820),
            f"Page {page_num + 1}",
            fontsize=9,
            fontname="helv",
            color=(0.5, 0.5, 0.5),
        )

        # Body text — wrap manually at ~85 chars per line
        lines = []
        for paragraph in text.split("\n"):
            if not paragraph.strip():
                lines.append("")
                continue
            words = paragraph.split()
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 > 85:
                    lines.append(current_line)
                    current_line = word
                else:
                    current_line = (current_line + " " + word).strip()
            if current_line:
                lines.append(current_line)

        y = y_start
        for line in lines:
            if y > 790:
                break
            page.insert_text(
                (50, y), line, fontsize=10, fontname="helv"
            )
            y += 14

    doc.save(filepath)
    doc.close()


# ── Sample PDF 1: Python Guide (3 pages) ────────────────────────────────────

python_pages = [
    # Page 1
    """Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991. It emphasizes code readability with its notable use of significant whitespace. Python supports multiple programming paradigms, including structured, object-oriented, and functional programming.

One of Python's key strengths is its extensive standard library, often described as "batteries included." This library provides modules for file I/O, system calls, sockets, and even interfaces to graphical user interface toolkits. The standard library reduces the need for external dependencies in many common programming tasks.

Python's package ecosystem is massive, with over 400,000 packages on PyPI (Python Package Index). Popular packages include NumPy for numerical computing, Pandas for data analysis, Flask and Django for web development, and TensorFlow and PyTorch for machine learning. The pip package manager makes it easy to install and manage these packages.

Virtual environments are essential for Python development. They create isolated spaces where you can install packages without affecting other projects. The venv module, included in Python 3.3+, creates lightweight virtual environments.""",

    # Page 2
    """Python's simplicity makes it an excellent first programming language. Variables don't need type declarations, indentation enforces clean code structure, and the syntax reads almost like English. List comprehensions, generator expressions, and built-in functions like map, filter, and zip make data processing concise.

Error handling in Python uses try-except blocks. You can catch specific exceptions like ValueError or TypeError, or catch all exceptions with a bare except clause (though this is discouraged). The finally block runs cleanup code regardless of whether an exception occurred. Python also supports custom exception classes.

Decorators are a powerful feature in Python that allow you to modify the behavior of functions or classes. A decorator is a function that takes another function as an argument and extends its behavior without explicitly modifying it. Common built-in decorators include @staticmethod, @classmethod, and @property.

Context managers, using the 'with' statement, ensure resources are properly managed. The most common use is file handling: 'with open("file.txt") as f:' guarantees the file is closed after use, even if an exception occurs.""",

    # Page 3
    """Asynchronous programming in Python is handled through the asyncio module. The async and await keywords make it possible to write concurrent code that is more readable than traditional threading. Asyncio is particularly useful for I/O-bound tasks like web scraping, API calls, and database queries.

Python's data model allows classes to emulate built-in types. By implementing special methods like __init__, __str__, __len__, and __getitem__, you can make objects work with built-in functions. This is what makes Python so flexible.

Testing in Python is well-supported through the unittest module (built-in) and pytest (third-party). Unit tests verify individual components, integration tests check that components work together, and end-to-end tests validate complete workflows. Test-driven development (TDD) is a popular methodology.

Type hints, introduced in Python 3.5 with PEP 484, allow developers to annotate function signatures and variable types. While Python remains dynamically typed at runtime, type hints enable static analysis tools like mypy to catch type-related bugs before execution. Modern Python codebases increasingly use type hints for better documentation and IDE support.""",
]

python_pdf_path = os.path.join(pdf_dir, "python_guide.pdf")
create_sample_pdf(python_pdf_path, "Python Programming Guide", python_pages)
print(f"\n  📄 Created: python_guide.pdf ({len(python_pages)} pages)")


# ── Sample PDF 2: Machine Learning Guide (3 pages) ──────────────────────────

ml_pages = [
    # Page 1
    """Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. The fundamental idea is that machines can identify patterns in data and make decisions with minimal human intervention.

There are three main types of machine learning. Supervised learning uses labeled training data where the algorithm learns from examples with correct answers provided. Common algorithms include linear regression, decision trees, random forests, and neural networks. These are used for spam detection, image classification, and price prediction.

Unsupervised learning works with unlabeled data, finding hidden patterns and structures. Clustering algorithms like K-means group similar data points together. Dimensionality reduction techniques like PCA compress high-dimensional data while preserving important information. These are used for customer segmentation and anomaly detection.""",

    # Page 2
    """Reinforcement learning is the third paradigm, where an agent learns by interacting with an environment. The agent receives rewards or penalties for its actions and learns to maximize cumulative reward. This approach has achieved remarkable results in game playing (AlphaGo), robotics, and autonomous driving.

Deep learning uses neural networks with many layers to learn complex patterns. Convolutional Neural Networks (CNNs) excel at image recognition. Recurrent Neural Networks (RNNs) and their variants like LSTM handle sequential data. Transformers have revolutionized natural language processing and are the foundation for GPT and BERT.

Transfer learning has dramatically reduced the cost of training models. Instead of training from scratch, you start with a pre-trained model and fine-tune it for your specific task. This works because early layers learn general features that are useful across many tasks.""",

    # Page 3
    """RAG (Retrieval-Augmented Generation) combines information retrieval with text generation. Instead of relying solely on training data, RAG retrieves relevant documents from an external knowledge base and uses them as context for generating responses. This reduces hallucinations and keeps answers up-to-date.

Embeddings are dense vector representations of text. Similar meanings produce similar vectors, enabling semantic search. Models like all-MiniLM-L6-v2 convert text into 384-dimensional vectors. Vector databases like ChromaDB, Pinecone, and Weaviate store and search these embeddings efficiently using algorithms like HNSW.

The AI development workflow typically involves: data collection, data preprocessing, feature engineering, model selection, training, evaluation, hyperparameter tuning, and deployment. MLOps practices bring DevOps principles to machine learning, ensuring reproducibility, monitoring, and continuous improvement of models in production.""",
]

ml_pdf_path = os.path.join(pdf_dir, "ml_guide.pdf")
create_sample_pdf(ml_pdf_path, "Machine Learning Guide", ml_pages)
print(f"  📄 Created: ml_guide.pdf ({len(ml_pages)} pages)")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: Extract text from PDFs
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("PART 2: Extracting text from PDFs (PyMuPDF)")
print("=" * 65)


def extract_pdf_pages(filepath):
    """
    Extract text from a PDF, page by page.

    Returns a list of dicts:
      [{"page": 1, "text": "...", "char_count": 1234}, ...]
    """
    doc = fitz.open(filepath)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")  # plain text extraction

        # Clean up: remove excessive whitespace
        text = text.strip()
        if text:
            pages.append({
                "page": page_num + 1,  # 1-indexed
                "text": text,
                "char_count": len(text),
            })

    doc.close()
    return pages


# Extract text from both PDFs
print()
for pdf_name in ["python_guide.pdf", "ml_guide.pdf"]:
    pdf_path = os.path.join(pdf_dir, pdf_name)
    pages = extract_pdf_pages(pdf_path)

    total_chars = sum(p["char_count"] for p in pages)
    print(f"  📄 {pdf_name}")
    print(f"     Pages: {len(pages)}")
    print(f"     Total characters: {total_chars:,}")
    for p in pages:
        preview = p["text"][:60].replace("\n", " ")
        print(f"     Page {p['page']}: {p['char_count']:,} chars — \"{preview}...\"")
    print()


print("""  💡 PyMuPDF extracts:
     - Plain text (what we need for RAG)
     - Also supports: images, tables, annotations, links
     - Very fast: can process 100+ page PDFs in seconds
     - Handles most PDF formats reliably
""")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: Page-aware chunking
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PART 3: Page-aware chunking")
print("=" * 65)


def chunk_pdf(filepath, chunk_size=400, chunk_overlap=50):
    """
    Extract and chunk a PDF with page-level metadata.

    Each chunk knows:
      - Which file it came from
      - Which page it was on
      - Its position in the chunk sequence

    Returns a list of dicts:
      [{"text": "...", "source": "file.pdf", "page": 2, "chunk_index": 5}, ...]
    """
    filename = os.path.basename(filepath)
    pages = extract_pdf_pages(filepath)

    all_chunks = []
    global_chunk_idx = 0

    for page_data in pages:
        page_num = page_data["page"]
        page_text = page_data["text"]

        # Chunk this page's text
        page_chunks = recursive_chunk(
            page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_text in page_chunks:
            all_chunks.append({
                "text": chunk_text,
                "source": filename,
                "page": page_num,
                "chunk_index": global_chunk_idx,
                "char_count": len(chunk_text),
            })
            global_chunk_idx += 1

    return all_chunks


# Show chunking results
print()
for pdf_name in ["python_guide.pdf", "ml_guide.pdf"]:
    pdf_path = os.path.join(pdf_dir, pdf_name)
    chunks = chunk_pdf(pdf_path, chunk_size=400, chunk_overlap=50)

    print(f"  📄 {pdf_name} → {len(chunks)} chunks")

    # Show page distribution
    page_counts = {}
    for c in chunks:
        page_counts[c["page"]] = page_counts.get(c["page"], 0) + 1

    for page, count in sorted(page_counts.items()):
        print(f"     Page {page}: {count} chunks")

    # Show a sample chunk
    sample = chunks[3] if len(chunks) > 3 else chunks[0]
    preview = sample["text"][:80].replace("\n", " ")
    print(f"     Sample chunk #{sample['chunk_index']}: "
          f"[page {sample['page']}] \"{preview}...\"")
    print()

print("""  💡 Page-aware chunking:
     - Each chunk is tagged with its source page
     - When answering a question, we can say "Based on page 2..."
     - This is essential for trust and verification
     - Long pages produce multiple chunks, short pages might be 1 chunk
""")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: Build the PDF knowledge base in ChromaDB
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PART 4: Building PDF knowledge base in ChromaDB")
print("=" * 65)

client = chromadb.Client()
collection = client.create_collection(name="pdf_knowledge_base")


def ingest_pdf(filepath, collection):
    """Add a PDF's chunks to ChromaDB with full metadata."""
    chunks = chunk_pdf(filepath, chunk_size=400, chunk_overlap=50)

    texts = [c["text"] for c in chunks]
    ids = [f"{c['source']}::p{c['page']}::chunk_{c['chunk_index']}" for c in chunks]
    metadatas = [{
        "source": c["source"],
        "page": c["page"],
        "chunk_index": c["chunk_index"],
        "char_count": c["char_count"],
    } for c in chunks]

    collection.add(documents=texts, ids=ids, metadatas=metadatas)
    return len(chunks)


# Ingest all PDFs
print()
total_chunks = 0
for pdf_name in ["python_guide.pdf", "ml_guide.pdf"]:
    pdf_path = os.path.join(pdf_dir, pdf_name)
    n = ingest_pdf(pdf_path, collection)
    total_chunks += n
    print(f"  ✅ {pdf_name}: {n} chunks ingested")

print(f"\n  📊 Total: {collection.count()} chunks in knowledge base")


# Also ingest any user PDFs in the pdfs/ folder
user_pdfs = [f for f in os.listdir(pdf_dir)
             if f.endswith(".pdf") and f not in ("python_guide.pdf", "ml_guide.pdf")]
if user_pdfs:
    print(f"\n  📁 Found {len(user_pdfs)} additional PDF(s):")
    for pdf_name in user_pdfs:
        pdf_path = os.path.join(pdf_dir, pdf_name)
        try:
            n = ingest_pdf(pdf_path, collection)
            print(f"     ✅ {pdf_name}: {n} chunks ingested")
        except Exception as e:
            print(f"     ❌ {pdf_name}: {e}")
    print(f"  📊 Total after user PDFs: {collection.count()} chunks")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: RAG with page citations
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("PART 5: RAG with page-level citations")
print("=" * 65)


def retrieve_from_pdf(query, n_results=3, source_filter=None, page_filter=None):
    """Retrieve relevant chunks with PDF metadata."""
    kwargs = {"query_texts": [query], "n_results": n_results}

    # Build where filter
    conditions = []
    if source_filter:
        conditions.append({"source": source_filter})
    if page_filter:
        conditions.append({"page": page_filter})

    if len(conditions) == 1:
        kwargs["where"] = conditions[0]
    elif len(conditions) > 1:
        kwargs["where"] = {"$and": conditions}

    results = collection.query(**kwargs)

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "page": results["metadatas"][0][i]["page"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "distance": results["distances"][0][i],
            "similarity": distance_to_similarity(results["distances"][0][i]),
        })
    return chunks


def pdf_rag_query(question, n_results=3, source_filter=None, show_details=True):
    """Complete RAG pipeline with PDF page citations."""
    # Step 1: Retrieve
    chunks = retrieve_from_pdf(question, n_results=n_results,
                                source_filter=source_filter)

    # Step 2: Build prompt with source info
    context_parts = []
    total_chars = 0
    for chunk in chunks:
        if total_chars + len(chunk["text"]) > 2000:
            break
        # Include source info in context so LLM can reference it
        source_tag = f"[Source: {chunk['source']}, Page {chunk['page']}]"
        context_parts.append(f"{source_tag}\n{chunk['text']}")
        total_chars += len(chunk["text"])

    context_str = "\n\n---\n\n".join(context_parts)

    prompt = f"""Use the following context to answer the question.
Each context section is labeled with its source file and page number.
When answering, mention which source and page the information came from.
If the context does not have enough information, say "I don't have enough information in my knowledge base."
Do NOT make up information.

CONTEXT:
{context_str}

QUESTION: {question}

ANSWER:"""

    # Step 3: Generate
    system_msg = ("You are a helpful technical assistant. Answer questions "
                  "based ONLY on the provided context. Always cite the source "
                  "file and page number in your answer.")
    answer = call_llm(prompt, system_message=system_msg)

    if show_details:
        print(f"\n  🔍 Retrieved {len(chunks)} chunks:")
        for chunk in chunks:
            sim = chunk["similarity"]
            src = chunk["source"]
            page = chunk["page"]
            preview = chunk["text"][:55].replace("\n", " ")
            emoji = "🟢" if sim > 0.5 else "🟡" if sim > 0.3 else "🔴"
            print(f"     {emoji} [{sim:.4f}] [{src} p.{page}] \"{preview}...\"")

    return answer, chunks


# ── Demo queries with page citations ─────────────────────────────────────────

print()
test_questions = [
    "What are Python decorators?",
    "What is reinforcement learning?",
    "How do virtual environments work in Python?",
    "What is RAG and how does it work?",
    "How does transfer learning reduce training cost?",
]

for question in test_questions:
    print(f"  ❓ \"{question}\"")
    answer, chunks = pdf_rag_query(question)

    print(f"\n  🤖 Answer:")
    for line in answer.split("\n"):
        print(f"     {line}")

    # Show cited pages
    cited = set()
    for c in chunks:
        if c["similarity"] > 0.3:
            cited.add(f"{c['source']} (p.{c['page']})")
    if cited:
        print(f"\n  📎 Sources: {', '.join(sorted(cited))}")
    print()
    print("  " + "─" * 60)
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 6: Compare — which page answered?
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PART 6: Page-level analysis — where does each answer come from?")
print("=" * 65)

analysis_questions = [
    ("What is Python's standard library?", "python_guide.pdf", 1),
    ("How do decorators work?", "python_guide.pdf", 2),
    ("What is asyncio used for?", "python_guide.pdf", 3),
    ("What are the types of machine learning?", "ml_guide.pdf", 1),
    ("What is deep learning?", "ml_guide.pdf", 2),
    ("What is RAG?", "ml_guide.pdf", 3),
]

print(f"\n  {'Question':<42} {'Expected':>18} {'Got':>18} {'Match':>6}")
print(f"  {'─' * 42} {'─' * 18} {'─' * 18} {'─' * 6}")

correct = 0
for question, expected_file, expected_page in analysis_questions:
    chunks = retrieve_from_pdf(question, n_results=1)
    top = chunks[0]
    got_file = top["source"]
    got_page = top["page"]

    match = got_file == expected_file and got_page == expected_page
    if match:
        correct += 1

    emoji = "✅" if match else "❌"
    expected_str = f"{expected_file[:12]}p.{expected_page}"
    got_str = f"{got_file[:12]}p.{got_page}"

    print(f"  {question:<42} {expected_str:>18} {got_str:>18} {emoji:>6}")

print(f"\n  📊 Accuracy: {correct}/{len(analysis_questions)} "
      f"({correct/len(analysis_questions)*100:.0f}%) — correct page retrieved")

print("""
  💡 Page-level retrieval accuracy:
     - Shows if the right page was found for each question
     - Perfect accuracy = chunking + embedding are well-tuned
     - Misses usually mean: chunk too big, or question is ambiguous
     - This is how you evaluate your RAG pipeline quality!
""")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 7: The complete PDF RAG pipeline
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PART 7: The complete PDF RAG pipeline")
print("=" * 65)

print("""
  ╔═══════════════════════════════════════════════════════════════╗
  ║           PDF RAG PIPELINE (Step 6)                          ║
  ╠═══════════════════════════════════════════════════════════════╣
  ║                                                               ║
  ║  📁 INGESTION                                                ║
  ║  ──────────                                                   ║
  ║  📄 PDF files                                                ║
  ║       ↓                                                       ║
  ║  📝 PyMuPDF: extract text per page                           ║
  ║       ↓                                                       ║
  ║  ✂️  Recursive chunking (400 chars, 50 overlap)              ║
  ║       ↓                                                       ║
  ║  💾 ChromaDB: store chunks + metadata                        ║
  ║       (source file, page number, chunk index)                ║
  ║                                                               ║
  ║  🔍 QUERY                                                    ║
  ║  ────────                                                     ║
  ║  User question                                                ║
  ║       ↓                                                       ║
  ║  Vector search in ChromaDB                                    ║
  ║       ↓                                                       ║
  ║  Retrieved chunks (with page metadata)                        ║
  ║       ↓                                                       ║
  ║  Prompt: context + source tags + question                    ║
  ║       ↓                                                       ║
  ║  Azure OpenAI → answer with page citations                   ║
  ║                                                               ║
  ║  ✨ "Based on ml_guide.pdf, page 2:                          ║
  ║      Transfer learning reduces training cost by..."          ║
  ║                                                               ║
  ╚═══════════════════════════════════════════════════════════════╝

  📌 To add your own PDFs:
     1. Drop PDF files into the 'pdfs/' folder
     2. Re-run this script
     3. They'll be automatically ingested and searchable!
""")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 8: Interactive PDF Q&A chatbot
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PART 8: PDF Q&A Chatbot (type 'quit' to exit)")
print("=" * 65)

# List available PDFs
available = set()
for c_id in collection.get()["ids"]:
    source = c_id.split("::")[0]
    available.add(source)

print(f"""
  📚 Knowledge base: {collection.count()} chunks
  📄 PDFs loaded: {', '.join(sorted(available))}

  Commands:
    Just type a question → search all PDFs
    file:python_guide.pdf your question → filter by file
    list → show loaded PDFs and chunk counts
    quit → exit
""")

while True:
    user_input = input("  💬 You: ").strip()
    if user_input.lower() in ("quit", "exit", "q"):
        print("\n  👋 Goodbye!")
        break
    if not user_input:
        continue

    if user_input.lower() == "list":
        print(f"\n  📚 Loaded PDFs:")
        # Get metadata for each source
        all_meta = collection.get()["metadatas"]
        source_stats = {}
        for meta in all_meta:
            src = meta["source"]
            page = meta["page"]
            if src not in source_stats:
                source_stats[src] = {"chunks": 0, "pages": set()}
            source_stats[src]["chunks"] += 1
            source_stats[src]["pages"].add(page)

        for src, stats in sorted(source_stats.items()):
            pages = sorted(stats["pages"])
            print(f"     📄 {src}: {stats['chunks']} chunks, "
                  f"{len(pages)} pages ({min(pages)}-{max(pages)})")
        print()
        continue

    # Parse file filter
    source_filter = None
    query = user_input
    if user_input.startswith("file:"):
        parts = user_input.split(" ", 1)
        source_filter = parts[0].replace("file:", "")
        query = parts[1] if len(parts) > 1 else ""
        if not query:
            print("  ⚠️  Please provide a question after the file filter\n")
            continue
        print(f"  📁 Filtering: {source_filter}")

    try:
        answer, chunks = pdf_rag_query(
            query, n_results=3, source_filter=source_filter, show_details=True
        )

        # Check relevance
        best_sim = max(c["similarity"] for c in chunks)
        if best_sim < 0.25:
            print(f"\n  ⚠️  Low relevance ({best_sim:.4f}) — answer may not be accurate")

        print(f"\n  🤖 Answer:")
        for line in answer.split("\n"):
            print(f"     {line}")

        # Source citations
        cited = set()
        for c in chunks:
            if c["similarity"] > 0.3:
                cited.add(f"{c['source']} (p.{c['page']})")
        if cited:
            print(f"\n  📎 Sources: {', '.join(sorted(cited))}")
        print()

    except Exception as e:
        print(f"  ❌ Error: {e}\n")
