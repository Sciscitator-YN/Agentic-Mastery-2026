# production_rag.py
# WHAT: Full production RAG funnel over an iced-coffee knowledge base, every stage visible:
#       hybrid retrieval (dense + sparse, weighted RRF) -> cross-encoder rerank -> threshold gate
#       -> grounded generation (or a no-LLM refusal), all traced in LangFuse.
# WHY:  Phase 3 capstone - assembles the primitives from s5/s6/s7 + rebuild_rag into one pipeline
#       that fails safe: no relevant evidence means no LLM call and no chance to hallucinate.

import os
from dotenv import load_dotenv
from langfuse import observe, get_client
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
langfuse = get_client()

# Tuning knobs (mirrors the spec)
WIDE_N = 18          # candidates pulled from EACH retriever before fusion
W_DENSE = 0.7        # weight on dense (meaning) ranking in RRF
W_SPARSE = 0.3       # weight on sparse (keyword) ranking in RRF
RRF_K = 60           # RRF damping constant
RERANK_THRESHOLD = 2.0   # keep only chunks scoring strictly above this

# ---------- INGEST: ~25 varied facts about iced coffee ----------
KNOWLEDGE = [
    # brewing methods
    "Cold brew is made by steeping coarse coffee grounds in cold or room-temperature water for 12 to 24 hours.",
    "Japanese-style iced coffee, also called flash brew, brews hot coffee directly onto ice to cool it instantly.",
    "Iced pour-over uses about half the usual water volume and brews straight onto a glass full of ice.",
    "Espresso over ice, the base of an iced latte, pulls a fresh shot directly onto cold milk and ice.",
    "Cold brew steeped longer than 24 hours can turn bitter and woody from over-extraction.",
    # ratios
    "A common cold brew concentrate ratio is one part coffee to eight parts water by weight.",
    "Cold brew concentrate is usually diluted one-to-one with water or milk before serving.",
    "Flash brew typically uses a 60-to-40 split of hot water to ice to keep the final strength balanced.",
    # types of drinks
    "An iced latte combines espresso with cold milk poured over ice.",
    "An iced americano is espresso diluted with cold water and ice, lighter than a latte.",
    "An iced mocha adds chocolate syrup to an iced latte for a sweeter, dessert-like drink.",
    "A frappe is a blended, frothy iced coffee drink that originated in Greece in 1957.",
    "A frappuccino is a trademarked Starbucks blended drink, distinct from the Greek frappe.",
    "Affogato is espresso poured over vanilla ice cream, a hot-meets-cold Italian treat.",
    # history
    "Mazagran, a sweetened cold coffee drink from 1840s Algeria, is considered an early iced coffee.",
    "Japanese kissaten cafes popularized flash-brewed iced coffee in the mid-20th century.",
    "Cold brew surged in popularity in the United States during the 2010s specialty coffee boom.",
    # equipment
    "The Toddy cold brew system uses a felt filter to produce a smooth, low-acid concentrate.",
    "A Hario V60 dripper is commonly used for the flash-brew iced pour-over method.",
    "A French press can make cold brew by steeping grounds and pressing the plunger after many hours.",
    "A fine-mesh sieve or paper filter is needed to remove the gritty sediment from cold brew.",
    # common mistakes
    "Pouring hot brewed coffee over ice without adjusting the recipe produces a watery, diluted cup.",
    "Using finely ground coffee for cold brew causes cloudiness and a harsh, over-extracted taste.",
    "Skipping dilution of cold brew concentrate leaves the drink unpleasantly strong and syrupy.",
    "Cold brew is lower in acidity than hot-brewed coffee because no heat is used during extraction.",
]

# Dense store (Chroma, in-memory)
chroma = chromadb.Client()
collection = chroma.create_collection(name="iced_coffee")
collection.add(documents=KNOWLEDGE, ids=[f"k{i}" for i in range(len(KNOWLEDGE))])

# Sparse store (BM25 over the SAME docs) - lowercase whitespace tokenization
tokenized = [doc.lower().split() for doc in KNOWLEDGE]
bm25 = BM25Okapi(tokenized)

# Cross-encoder reranker (downloads ~80MB first run, then cached; runs locally)
print("Loading cross-encoder reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print(f"Ingested {len(KNOWLEDGE)} iced-coffee facts into Chroma + BM25.\n")


# ---------- RETRIEVAL HELPERS ----------
def dense_ranked(query, n):
    res = collection.query(query_texts=[query], n_results=n)
    return res["documents"][0]   # already ordered best-first


def sparse_ranked(query, n):
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(zip(scores, KNOWLEDGE), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:n]]


def weighted_rrf(dense_list, sparse_list, w_dense=W_DENSE, w_sparse=W_SPARSE, k=RRF_K):
    # Each doc scored by weight * 1/(k + rank) in each list; summed across lists.
    scores = {}
    for rank, doc in enumerate(dense_list):
        scores[doc] = scores.get(doc, 0) + w_dense * (1 / (k + rank))
    for rank, doc in enumerate(sparse_list):
        scores[doc] = scores.get(doc, 0) + w_sparse * (1 / (k + rank))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------- QUERY TIME (runs per question) ----------
@observe(as_type="generation")
def rag_answer(question: str) -> str:
    print("=" * 80)
    print(f"Q: {question}")

    # STAGE 1 - HYBRID retrieval: dense + sparse, fused with weighted RRF
    dense = dense_ranked(question, WIDE_N)
    sparse = sparse_ranked(question, WIDE_N)
    fused = weighted_rrf(dense, sparse)

    print(f"\n  [STAGE 1] HYBRID retrieval (wide net = {WIDE_N}, RRF dense={W_DENSE}/sparse={W_SPARSE})")
    print(f"    DENSE (meaning) top 5:")
    for i, doc in enumerate(dense[:5]):
        print(f"      {i+1}. {doc[:70]}")
    print(f"    SPARSE (BM25 keyword) top 5:")
    for i, doc in enumerate(sparse[:5]):
        print(f"      {i+1}. {doc[:70]}")
    print(f"    FUSED (weighted RRF) top 8:")
    for doc, score in fused[:8]:
        print(f"      [{score:.5f}] {doc[:70]}")

    # STAGE 2 - cross-encoder reranks the fused candidates by TRUE relevance
    candidates = [doc for doc, _ in fused]
    pairs = [(question, doc) for doc in candidates]
    rerank_scores = reranker.predict(pairs)
    reranked = sorted(zip(rerank_scores, candidates), key=lambda x: x[0], reverse=True)

    print(f"\n  [STAGE 2] RERANK (cross-encoder, higher = more relevant)")
    for score, doc in reranked:
        print(f"      [{score:6.2f}] {doc[:70]}")

    # STAGE 3 - threshold gate: keep only chunks scoring strictly above the cutoff
    survivors = [(score, doc) for score, doc in reranked if score > RERANK_THRESHOLD]
    print(f"\n  [STAGE 3] THRESHOLD gate (keep score > {RERANK_THRESHOLD})")
    print(f"      {len(survivors)} of {len(reranked)} chunks survived:")
    for score, doc in survivors:
        print(f"      [{score:6.2f}] {doc[:70]}")

    # STAGE 4 - if nothing survives, refuse WITHOUT calling the LLM
    if not survivors:
        answer = "I don't have that information."
        print(f"\n  [STAGE 4] No survivors -> refusing without an LLM call.")
        print(f"  A: {answer}\n")
        return answer

    # STAGE 5 - grounded generation using ONLY the survivors
    context = "\n".join(f"- {doc}" for _, doc in survivors)
    system = (
        "You answer questions using ONLY the provided context. "
        "If the answer is not in the context, say exactly: 'I don't have that information.' "
        "Do not use any outside knowledge."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=150,
    )
    langfuse.update_current_generation(
        model="llama-3.3-70b-versatile",
        usage_details={"input": response.usage.prompt_tokens,
                       "output": response.usage.completion_tokens},
    )

    answer = response.choices[0].message.content
    print(f"\n  [STAGE 5] GROUNDED generation ({len(survivors)} chunks -> llama-3.3-70b)")
    print(f"  A: {answer}\n")
    return answer


@observe()
def main():
    rag_answer("What's the typical cold brew coffee-to-water ratio?")        # clearly answerable
    rag_answer("How many calories are in a Starbucks iced caramel macchiato?")  # right topic, not in docs
    rag_answer("What year did the Berlin Wall fall?")                        # totally unrelated


if __name__ == "__main__":
    main()
    langfuse.flush()
