# s2_vector_db.py
# WHAT: Store documents in a vector DB, retrieve by meaning at scale
# WHY: The storage layer of every RAG system - same engine as last session, industrialized

import chromadb

# An in-memory Chroma client (nothing saved to disk - fresh each run, perfect for learning)
client = chromadb.Client()

# A "collection" is a table of documents + their vectors.
# Chroma auto-embeds with a built-in model (all-MiniLM-L6-v2 - same one as last session)
collection = client.create_collection(name="knowledge")

# Our tiny knowledge base - 6 docs across 3 topics
documents = [
    "Puppies need frequent small meals and plenty of fresh water each day.",   # pet
    "Dogs should be walked daily and given regular vet checkups.",             # pet
    "Diversifying your portfolio reduces investment risk over time.",          # finance
    "Index funds typically have lower fees than actively managed funds.",      # finance
    "Sear the steak on high heat for two minutes per side.",                   # cooking
    "Let bread dough rise for about an hour before baking.",                   # cooking
]

# Store them - Chroma embeds each one automatically as it's added
collection.add(
    documents=documents,
    ids=[f"doc{i}" for i in range(len(documents))],
)
print(f"Stored {len(documents)} documents.\n")

# Now query by MEANING
query = "How do I care for a puppy?"
results = collection.query(
    query_texts=[query],
    n_results=3,   # return the 3 closest
)

print(f"QUERY: {query}\n")
print("TOP 3 MATCHES (closest meaning first):")
print("-" * 70)
for doc, dist in zip(results["documents"][0], results["distances"][0]):
    # distance: LOWER = more similar (it's the opposite of similarity - it's distance apart)
    print(f"distance {dist:.3f}  |  {doc}")
