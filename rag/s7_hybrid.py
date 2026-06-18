# s7_hybrid.py
# WHAT: Combine dense (vector) + sparse (BM25 keyword) retrieval via Reciprocal Rank Fusion
# WHY: Two methods fail in opposite directions - together they protect recall (catch the answer)

import chromadb
from rank_bm25 import BM25Okapi

KNOWLEDGE = [
    "Nordwind Logistics was founded in 2021 and is based in Hamburg, Germany.",
    "The Nordwind platform automatically matches available trucks to incoming freight loads.",
    "Pricing starts at two thousand dollars per month for small fleets.",
    "Nordwind offers a free trial that lasts fourteen days with no credit card required.",
    "Maria Vogel has served as the Chief Executive Officer of Nordwind since 2021.",
    "The company employs around two hundred people across three regional offices.",
]

# ---------- DENSE: vector search via Chroma ----------
chroma = chromadb.Client()
collection = chroma.create_collection(name="hybrid")
collection.add(documents=KNOWLEDGE, ids=[f"k{i}" for i in range(len(KNOWLEDGE))])

def dense_ranked(query, n):
    res = collection.query(query_texts=[query], n_results=n)
    return res["documents"][0]   # already ordered best-first

# ---------- SPARSE: BM25 keyword search ----------
# BM25 needs tokenized docs (just lowercase split for now)
tokenized = [doc.lower().split() for doc in KNOWLEDGE]
bm25 = BM25Okapi(tokenized)

def sparse_ranked(query, n):
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(zip(scores, KNOWLEDGE), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:n]]

# ---------- FUSION: Reciprocal Rank Fusion ----------
def rrf(dense_list, sparse_list, k=60):
    # Each doc scored by 1/(k + rank) in each list; sum across lists
    scores = {}
    for rank, doc in enumerate(dense_list):
        scores[doc] = scores.get(doc, 0) + 1 / (k + rank)
    for rank, doc in enumerate(sparse_list):
        scores[doc] = scores.get(doc, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def hybrid_search(query, n=6):
    dense = dense_ranked(query, n)
    sparse = sparse_ranked(query, n)

    print(f"\nQ: {query}")
    print("  DENSE (vector/meaning) order:")
    for i, doc in enumerate(dense):
        print(f"    {i+1}. {doc[:60]}")
    print("  SPARSE (BM25/keyword) order:")
    for i, doc in enumerate(sparse):
        print(f"    {i+1}. {doc[:60]}")

    fused = rrf(dense, sparse)
    print("  FUSED (RRF combined) order:")
    for doc, score in fused:
        print(f"    [{score:.4f}] {doc[:60]}")

hybrid_search("Who is the CEO?")
hybrid_search("caring for young animals")   # meaning query - watch dense win this one
