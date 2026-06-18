# s1_embeddings.py
# WHAT: Turn text into vectors, measure meaning as geometric closeness
# WHY: The engine under all RAG - "find relevant text" becomes "find nearby points"

from sentence_transformers import SentenceTransformer
import numpy as np

# Loads a small, fast embedding model (downloads ~80MB on first run, then cached)
print("Loading embedding model (first run downloads it)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

phrases = [
    "I love my dog",
    "My puppy is adorable",
    "The stock market crashed",
    "Tax filing deadline",
]

# Each phrase -> a 384-number vector
embeddings = model.encode(phrases)
print(f"\nEach phrase became a vector of {embeddings.shape[1]} numbers.")
print(f"First phrase's first 8 numbers: {embeddings[0][:8]}\n")

def cosine_similarity(a, b):
    # cosine similarity = how aligned two vectors are (1.0 = identical meaning, 0 = unrelated)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("Similarity between every pair (1.0 = identical meaning):")
print("-" * 60)
for i in range(len(phrases)):
    for j in range(i + 1, len(phrases)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"{sim:.3f}  |  '{phrases[i]}'  <->  '{phrases[j]}'")
