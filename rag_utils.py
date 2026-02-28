"""
rag_utils.py — Shared utilities for RAG learning steps

Common functions used across multiple step files:
  - cosine_sim: Cosine similarity between two vectors
  - recursive_chunk: Industry-standard recursive text splitter
  - similarity_emoji: Convert similarity score to emoji indicator
  - distance_to_similarity: Convert ChromaDB distance to similarity
  - load_embedding_model: Load sentence-transformers model
  - get_knowledge_documents: Shared document corpus
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding Model
# ═══════════════════════════════════════════════════════════════════════════════

def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    """Load the sentence-transformers embedding model."""
    from sentence_transformers import SentenceTransformer
    print(f"Loading embedding model ({model_name})...")
    model = SentenceTransformer(model_name)
    print("Ready!")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Similarity Functions
# ═══════════════════════════════════════════════════════════════════════════════

def cosine_sim(a, b):
    """Cosine similarity: measures direction alignment between two vectors.
    Returns a value between -1 (opposite) and 1 (identical direction)."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def distance_to_similarity(distance):
    """Convert ChromaDB's L2 distance to cosine similarity.
    ChromaDB returns distance (lower = better), we want similarity (higher = better).
    Formula: similarity ≈ 1 - (distance² / 2) for normalized vectors."""
    return 1 - (distance ** 2 / 2)


def similarity_emoji(score, thresholds=(0.5, 0.3)):
    """Convert similarity score to emoji indicator.
    🟢 = high relevance, 🟡 = medium, 🔴 = low."""
    high, low = thresholds
    if score > high:
        return "🟢"
    elif score > low:
        return "🟡"
    else:
        return "🔴"


# ═══════════════════════════════════════════════════════════════════════════════
# Chunking
# ═══════════════════════════════════════════════════════════════════════════════

def recursive_chunk(text, chunk_size=500, chunk_overlap=50):
    """
    Recursive character text splitter — same idea as LangChain's.
    
    Strategy: Try to split on the LARGEST meaningful boundary first:
      1. Double newline (paragraph boundary)    ← best
      2. Single newline (line boundary)
      3. Sentence ending (. ! ?)
      4. Space (word boundary)
      5. Character (last resort)                ← worst
    
    Args:
        text: The text to split into chunks
        chunk_size: Maximum characters per chunk (default 500)
        chunk_overlap: Characters of overlap between chunks (default 50)
    
    Returns:
        List of text chunks with overlap applied
    """
    separators = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    
    def _split_text(text, separators):
        chunks = []
        
        # Find the best separator that exists in the text
        separator = separators[-1]  # default: empty string (char by char)
        for sep in separators:
            if sep in text:
                separator = sep
                break
        
        # Split using this separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)  # character by character
        
        # Merge small splits into chunks of the target size
        current_chunk = ""
        for split in splits:
            piece = split if not separator else split + separator
            
            if len(current_chunk) + len(piece) <= chunk_size:
                current_chunk += piece
            else:
                if current_chunk:
                    chunks.append(current_chunk.rstrip())
                
                # If this single piece is too large, recursively split it
                if len(piece) > chunk_size:
                    remaining = separators[separators.index(separator) + 1:]
                    if remaining:
                        sub_chunks = _split_text(piece, remaining)
                        chunks.extend(sub_chunks)
                        current_chunk = ""
                    else:
                        current_chunk = piece
                else:
                    current_chunk = piece
        
        if current_chunk.strip():
            chunks.append(current_chunk.rstrip())
        
        return chunks
    
    # Get initial chunks
    raw_chunks = _split_text(text, separators)
    
    # Add overlap between chunks
    if chunk_overlap > 0 and len(raw_chunks) > 1:
        overlapped_chunks = [raw_chunks[0]]
        for i in range(1, len(raw_chunks)):
            # Take the last chunk_overlap chars from previous chunk
            prev_tail = raw_chunks[i - 1][-chunk_overlap:]
            # Find a clean break point (word boundary)
            space_idx = prev_tail.find(" ")
            if space_idx != -1:
                prev_tail = prev_tail[space_idx + 1:]
            overlapped_chunks.append(prev_tail + " " + raw_chunks[i])
        return overlapped_chunks
    
    return raw_chunks


# ═══════════════════════════════════════════════════════════════════════════════
# Shared Document Corpus
# ═══════════════════════════════════════════════════════════════════════════════

PYTHON_DOC = """
Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991. It emphasizes code readability with its notable use of significant whitespace. Python supports multiple programming paradigms, including structured, object-oriented, and functional programming.

One of Python's key strengths is its extensive standard library, often described as "batteries included." This library provides modules for file I/O, system calls, sockets, and even interfaces to graphical user interface toolkits like Tk. The standard library reduces the need for external dependencies in many common programming tasks.

Python's package ecosystem is massive, with over 400,000 packages on PyPI (Python Package Index). Popular packages include NumPy for numerical computing, Pandas for data analysis, Flask and Django for web development, and TensorFlow and PyTorch for machine learning. The pip package manager makes it easy to install and manage these packages.

Virtual environments are essential for Python development. They create isolated spaces where you can install packages without affecting other projects. The venv module, included in Python 3.3+, creates lightweight virtual environments. Tools like conda provide more comprehensive environment management, including non-Python dependencies.

Python's simplicity makes it an excellent first programming language. Variables don't need type declarations, indentation enforces clean code structure, and the syntax reads almost like English. List comprehensions, generator expressions, and built-in functions like map, filter, and zip make data processing concise and elegant.

Error handling in Python uses try-except blocks. You can catch specific exceptions like ValueError or TypeError, or catch all exceptions with a bare except clause (though this is discouraged). The finally block runs cleanup code regardless of whether an exception occurred. Python also supports custom exception classes that inherit from the Exception base class.

Decorators are a powerful feature in Python that allow you to modify the behavior of functions or classes. A decorator is a function that takes another function as an argument and extends its behavior without explicitly modifying it. Common built-in decorators include @staticmethod, @classmethod, and @property. Decorators are widely used in web frameworks like Flask for routing.

Asynchronous programming in Python is handled through the asyncio module. The async and await keywords, introduced in Python 3.5, make it possible to write concurrent code that is more readable than traditional threading approaches. Asyncio is particularly useful for I/O-bound tasks like web scraping, API calls, and database queries where the program spends most of its time waiting for external responses.

Python's data model, sometimes called the "dunder" (double underscore) model, allows classes to emulate built-in types. By implementing special methods like __init__, __str__, __repr__, __len__, and __getitem__, you can make your objects work with built-in functions and operators. This is what makes Python so flexible — everything is an object, and objects can be customized to behave however you want.

Testing in Python is well-supported through the unittest module (built-in) and pytest (third-party). Unit tests verify individual components work correctly, integration tests check that components work together, and end-to-end tests validate complete workflows. Test-driven development (TDD) is a popular methodology where you write tests before writing the actual code.
""".strip()

AI_DOC = """
Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. The fundamental idea is that machines can identify patterns in data and make decisions with minimal human intervention.

There are three main types of machine learning. Supervised learning uses labeled training data — the algorithm learns from examples where the correct answer is provided. Common supervised learning algorithms include linear regression, decision trees, random forests, and neural networks. These are used for tasks like spam detection, image classification, and price prediction.

Unsupervised learning works with unlabeled data, finding hidden patterns and structures. Clustering algorithms like K-means group similar data points together. Dimensionality reduction techniques like PCA (Principal Component Analysis) compress high-dimensional data while preserving important information. These techniques are used for customer segmentation, anomaly detection, and recommendation systems.

Reinforcement learning is the third paradigm, where an agent learns by interacting with an environment. The agent receives rewards or penalties for its actions and learns to maximize cumulative reward over time. This approach has achieved remarkable results in game playing (AlphaGo, Atari games), robotics, and autonomous driving.

Deep learning, a subset of machine learning, uses neural networks with many layers (hence "deep") to learn complex patterns. Convolutional Neural Networks (CNNs) excel at image recognition, Recurrent Neural Networks (RNNs) and their variants like LSTM handle sequential data, and Transformers have revolutionized natural language processing. The transformer architecture, introduced in the "Attention Is All You Need" paper, is the foundation for models like BERT, GPT, and T5.

Transfer learning has dramatically reduced the cost and time of training models. Instead of training from scratch, you start with a pre-trained model (trained on massive datasets) and fine-tune it for your specific task. This approach works because early layers of neural networks learn general features (edges, shapes, basic grammar) that are useful across many tasks.

RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with text generation. Instead of relying solely on a model's training data, RAG retrieves relevant documents from an external knowledge base and uses them as context for generating responses. This approach reduces hallucinations, keeps answers up-to-date, and allows the model to access domain-specific information it wasn't trained on.
""".strip()

WEBDEV_DOC = """
Modern web development involves building applications that run in web browsers. The frontend (client-side) handles what users see and interact with, while the backend (server-side) manages data, logic, and security.

HTML (HyperText Markup Language) provides the structure of web pages. Elements like headings, paragraphs, links, images, and forms are defined using HTML tags. HTML5 introduced semantic elements like <header>, <nav>, <main>, and <footer> that make the document structure clearer and improve accessibility.

CSS (Cascading Style Sheets) controls the visual presentation. Modern CSS includes Flexbox for one-dimensional layouts, CSS Grid for two-dimensional layouts, and custom properties (variables) for maintainable design systems. Responsive design using media queries ensures websites work on screens of all sizes, from phones to desktops.

JavaScript is the programming language of the web. It enables dynamic behavior like form validation, animations, API calls, and real-time updates. Modern JavaScript (ES6+) includes features like arrow functions, destructuring, template literals, async/await, and modules. TypeScript adds static typing on top of JavaScript for better tooling and error catching.

React is a popular JavaScript library for building user interfaces. It uses a component-based architecture where the UI is broken into reusable pieces. Components can have state (data that changes) and props (data passed from parent). React's virtual DOM efficiently updates only the parts of the page that changed, making it fast.

REST APIs (Representational State Transfer) are the standard way for frontends to communicate with backends. They use HTTP methods: GET (read), POST (create), PUT (update), DELETE (remove). APIs return data in JSON format. Authentication is typically handled through JWT (JSON Web Tokens) or OAuth.

Node.js allows JavaScript to run on the server side. Express.js is the most popular Node.js web framework, providing routing, middleware, and request handling. For databases, developers commonly use PostgreSQL (relational), MongoDB (document-based), or Redis (in-memory caching).
""".strip()


# ═══════════════════════════════════════════════════════════════════════════════
# PDF Page Content (Step 6)
# ═══════════════════════════════════════════════════════════════════════════════

PYTHON_PDF_PAGES = [
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

ML_PDF_PAGES = [
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


# Short document collections for Steps 2-3
TECH_DOCUMENTS_25 = [
    # Python (5)
    "Python list comprehensions create lists in a single line of code",
    "Django REST framework is used for building APIs in Python",
    "Virtual environments isolate Python project dependencies",
    "Python decorators modify the behavior of functions",
    "Async and await keywords enable asynchronous programming in Python",
    # AI/ML (5)
    "Transformers architecture revolutionized natural language processing",
    "GPT models predict the next token in a sequence",
    "Fine-tuning adapts a pre-trained model to a specific task",
    "RAG combines retrieval with generation for better AI answers",
    "Embeddings represent words as dense numerical vectors",
    # Web Dev (5)
    "React components use JSX to describe user interfaces",
    "REST APIs use HTTP methods like GET POST PUT DELETE",
    "CSS Flexbox makes responsive layouts easy to build",
    "WebSocket enables real-time bidirectional communication",
    "Docker containers package applications with their dependencies",
    # Database (5)
    "SQL JOIN combines rows from two or more tables",
    "MongoDB stores data as flexible JSON-like documents",
    "Redis is an in-memory data store used for caching",
    "PostgreSQL supports both relational and JSON data types",
    "Database indexing speeds up query performance significantly",
    # DevOps (5)
    "CI/CD pipelines automate building testing and deploying code",
    "Kubernetes orchestrates containerized applications at scale",
    "Terraform provisions infrastructure using code",
    "Git branches allow parallel development workflows",
    "Load balancers distribute traffic across multiple servers",
]

TECH_TOPICS_25 = (["python"] * 5 + ["ai_ml"] * 5 + ["webdev"] * 5 +
                  ["database"] * 5 + ["devops"] * 5)
