"""
Step 1: Text → Numbers (Embeddings)

WHAT IS AN EMBEDDING?
  Every sentence gets converted to a list of numbers (a vector).
  Sentences with similar MEANING get similar numbers.

  "I love dogs"    → [0.12, -0.45, 0.78, 0.33, ...]
  "I adore puppies" → [0.11, -0.43, 0.80, 0.31, ...]  ← very close!
  "The stock market" → [0.89, 0.22, -0.15, -0.67, ...] ← very different!

HOW?
  A neural network (trained on billions of sentences) learned that
  "dogs" and "puppies" appear in similar contexts, so it places them
  close together in vector space. "stock market" appears in totally
  different contexts, so it's far away.

This file demonstrates:
  1. Converting text to vectors
  2. Seeing the raw numbers
  3. Comparing similar vs different sentences
  4. Building a simple search
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# ─── Load the embedding model ───────────────────────────────────────────────
# "all-MiniLM-L6-v2" is a small, fast model (80MB)
# It converts any sentence → 384 numbers
# First run downloads the model, then it's cached locally

print("Loading embedding model (first time downloads ~80MB)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded!\n")


# ─── Part 1: See what an embedding looks like ────────────────────────────────

print("=" * 60)
print("PART 1: What does an embedding look like?")
print("=" * 60)

sentence = "I love programming in Python"
embedding = model.encode(sentence)

print(f"\nSentence: \"{sentence}\"")
print(f"Embedding type: {type(embedding)}")
print(f"Embedding shape: {embedding.shape}")  # (384,) = 384 numbers
print(f"First 10 numbers: {embedding[:10].round(4)}")
print(f"Total numbers: {len(embedding)}")


# ─── Part 2: Similar sentences → similar vectors ────────────────────────────

print("\n" + "=" * 60)
print("PART 2: Similar meaning = similar vectors")
print("=" * 60)

# These mean the same thing (different words)
similar_sentences = [
    "I love programming in Python",
    "Python coding is my passion",
    "I enjoy writing Python code",
]

# These mean something completely different
different_sentences = [
    "The weather is sunny today",
    "I had pizza for lunch",
    "The stock market crashed",
]

all_sentences = similar_sentences + different_sentences
embeddings = model.encode(all_sentences)

# Cosine similarity: how "close" two vectors are (1.0 = identical, 0.0 = unrelated)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"\nBase sentence: \"{similar_sentences[0]}\"\n")
print("Comparing with similar sentences:")
for i in range(1, len(similar_sentences)):
    sim = cosine_similarity(embeddings[0], embeddings[i])
    print(f"  \"{similar_sentences[i]}\"")
    print(f"    → similarity: {sim:.4f}  {'🟢 HIGH' if sim > 0.5 else '🔴 LOW'}")

print("\nComparing with different sentences:")
for i, sent in enumerate(different_sentences):
    idx = len(similar_sentences) + i
    sim = cosine_similarity(embeddings[0], embeddings[idx])
    print(f"  \"{sent}\"")
    print(f"    → similarity: {sim:.4f}  {'🟢 HIGH' if sim > 0.5 else '🔴 LOW'}")


# ─── Part 3: Build a mini search engine ──────────────────────────────────────

print("\n" + "=" * 60)
print("PART 3: Mini search engine (50 documents)")
print("=" * 60)

# Our "knowledge base" — 20 documents about different topics
documents = [
    # Python programming
    "Python is a high-level programming language known for simplicity",
    "Lists, dictionaries, and tuples are Python data structures",
    "Django and Flask are popular Python web frameworks",
    "NumPy and Pandas are essential for data science in Python",
    "Python supports object-oriented and functional programming",

    # Machine Learning
    "Machine learning algorithms learn patterns from training data",
    "Neural networks are inspired by the human brain structure",
    "Deep learning uses multiple layers to process complex data",
    "Training a model requires labeled data and a loss function",
    "Overfitting occurs when a model memorizes training data",

    # Cooking
    "Pasta should be cooked in salted boiling water for 8-10 minutes",
    "A good curry needs onions, tomatoes, and fresh spices",
    "Baking requires precise measurements of flour and sugar",
    "Marinating meat for 2 hours improves flavor significantly",
    "Stir-frying vegetables on high heat keeps them crunchy",

    # Space
    "The Sun is a medium-sized star in the Milky Way galaxy",
    "Mars has the largest volcano in the solar system called Olympus Mons",
    "Black holes have gravity so strong that light cannot escape",
    "The International Space Station orbits Earth every 90 minutes",
    "Jupiter has 95 known moons including Europa and Ganymede",
]

# Embed all documents (batch processing — much faster)
print(f"\nEmbedding {len(documents)} documents...")
doc_embeddings = model.encode(documents)
print(f"Done! Each document → {doc_embeddings.shape[1]} numbers\n")

# Search function
def search(query, top_k=3):
    """Find the most similar documents to the query."""
    query_embedding = model.encode(query)
    
    # Calculate similarity with every document
    similarities = [
        cosine_similarity(query_embedding, doc_emb)
        for doc_emb in doc_embeddings
    ]
    
    # Sort by similarity (highest first)
    ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
    
    return [(documents[idx], score) for idx, score in ranked[:top_k]]

# Test searches
test_queries = [
    "How do I make Indian food?",
    "What are planets made of?",
    "How does AI learn from data?",
    "What is the best way to code in Python?",
    "How to cook noodles?",
]

for query in test_queries:
    print(f"🔍 Query: \"{query}\"")
    results = search(query)
    for rank, (doc, score) in enumerate(results, 1):
        emoji = "🟢" if score > 0.5 else "🟡" if score > 0.3 else "🔴"
        print(f"   {emoji} #{rank} (score: {score:.4f}): {doc}")
    print()


# ─── Part 4: Visualize the vector space (2D) ────────────────────────────────

print("=" * 60)
print("PART 4: How vectors cluster by topic")
print("=" * 60)

# Show which documents are close to each other
topics = (["Python"] * 5) + (["ML"] * 5) + (["Cooking"] * 5) + (["Space"] * 5)

print("\nCross-topic similarity (average):\n")
topic_names = ["Python", "ML", "Cooking", "Space"]
for i, t1 in enumerate(topic_names):
    for j, t2 in enumerate(topic_names):
        # Average similarity between all docs in topic i and topic j
        sims = []
        for di in range(5):
            for dj in range(5):
                if i != j or di != dj:  # skip self-comparison
                    sims.append(cosine_similarity(
                        doc_embeddings[i*5 + di],
                        doc_embeddings[j*5 + dj]
                    ))
        avg_sim = np.mean(sims)
        bar = "█" * int(avg_sim * 30)
        print(f"  {t1:>8} ↔ {t2:<8}: {avg_sim:.3f} {bar}")
    print()

print("=" * 60)
print("KEY TAKEAWAYS:")
print("  1. Same-topic documents have HIGH similarity (0.3-0.7)")
print("  2. Different-topic documents have LOW similarity (0.0-0.2)")
print("  3. The embedding model UNDERSTANDS meaning, not just words")
print('  4. "Indian food" matches "curry + spices" despite no shared words!')
print("=" * 60)
