# s6_reranker.py
# WHAT: Two-stage retrieval - vector search casts a wide net, cross-encoder reranks for TRUE relevance
# WHY: Fixes the semantic-similarity vs answer-relevance gap (the 0.817 CEO failure)

from sentence_transformers import CrossEncoder
import chromadb

KNOWLEDGE = [
    "Nordwind Logistics was founded in 2021 and is based in Hamburg, Germany.",
    "The Nordwind platform automatically matches available trucks to incoming freight loads.",
    "Pricing starts at two thousand dollars per month for small fleets.",
    "Nordwind offers a free trial that lasts fourteen days with no credit card required.",
    "Maria Vogel has served as the Chief Executive Officer of Nordwind since 2021.",  # <- the real answer, added
    "The company employs around two hundred people across three regional offices.",
]

chroma = chromadb.Client()
collection = chroma.create_collection(name="nordwind_rerank")
collection.add(documents=KNOWLEDGE, ids=[f"k{i}" for i in range(len(KNOWLEDGE))])

# The cross-encoder: reads (query, document) PAIRS and scores true relevance.
# Downloads ~80MB first run, then cached. Runs locally on your M1.
print("Loading cross-encoder reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retrieve_and_rerank(question: str, wide_n=6, keep_top=2):
    # STAGE 1 - vector search casts a wide net (cheap, by distance)
    results = collection.query(query_texts=[question], n_results=wide_n)
    candidates = results["documents"][0]
    distances = results["distances"][0]

    print(f"\nQ: {question}")
    print(f"  STAGE 1 - vector candidates (by distance):")
    for doc, dist in zip(candidates, distances):
        print(f"    [{dist:.3f}] {doc[:65]}")

    # STAGE 2 - cross-encoder re-scores each (question, candidate) PAIR
    pairs = [(question, doc) for doc in candidates]
    scores = reranker.predict(pairs)   # higher = more truly relevant

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

    print(f"  STAGE 2 - reranked by TRUE relevance (cross-encoder score, higher=better):")
    for score, doc in ranked:
        print(f"    [{score:6.2f}] {doc[:65]}")

    print(f"  >>> TOP {keep_top} after rerank:")
    for score, doc in ranked[:keep_top]:
        print(f"      {doc}")

retrieve_and_rerank("Who is the CEO of Nordwind?")
retrieve_and_rerank("How long is the free trial?")
