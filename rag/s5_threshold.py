# s5_threshold.py
# WHAT: Filter retrieved chunks by a distance cutoff BEFORE they reach the model
# WHY: The retrieval-stage guardrail - returns only what's relevant, even if that's nothing

from dotenv import load_dotenv
load_dotenv()

from langfuse import observe, get_client
from groq import Groq
import chromadb
import os

groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
langfuse = get_client()

KNOWLEDGE = [
    "Nordwind Logistics was founded in 2021 and is based in Hamburg, Germany.",
    "The Nordwind platform automatically matches available trucks to incoming freight loads.",
    "Pricing starts at two thousand dollars per month for small fleets.",
    "Nordwind offers a free trial that lasts fourteen days with no credit card required.",
]

chroma = chromadb.Client()
collection = chroma.create_collection(name="nordwind")
collection.add(documents=KNOWLEDGE, ids=[f"k{i}" for i in range(len(KNOWLEDGE))])

DISTANCE_THRESHOLD = 1.2   # <-- the cutoff. Tune this empirically per dataset.

@observe(as_type="generation")
def rag_answer(question: str) -> str:
    results = collection.query(query_texts=[question], n_results=3)
    retrieved = results["documents"][0]
    distances = results["distances"][0]

    # THE FILTER: keep only chunks closer than the threshold
    kept = [(doc, dist) for doc, dist in zip(retrieved, distances) if dist <= DISTANCE_THRESHOLD]

    print(f"Q: {question}")
    print(f"  Retrieved {len(retrieved)} chunks, {len(kept)} survived the {DISTANCE_THRESHOLD} threshold:")
    for doc, dist in zip(retrieved, distances):
        mark = "KEEP" if dist <= DISTANCE_THRESHOLD else "DROP"
        print(f"    [{dist:.3f}] {mark}  {doc[:60]}")

    # SHORT-CIRCUIT: if nothing survived, don't even call the model
    if not kept:
        print("  A: I don't have that information. (no relevant context - skipped the LLM call)\n")
        return "I don't have that information."

    context = "\n".join(f"- {doc}" for doc, _ in kept)
    system = (
        "You answer questions using ONLY the provided context. "
        "If the answer is not in the context, say exactly: 'I don't have that information.' "
        "Do not use outside knowledge."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"

    response = groq.chat.completions.create(
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
    print(f"  A: {answer}\n")
    return answer

@observe()
def main():
    rag_answer("How long is the free trial?")   # answerable - chunk well under threshold
    rag_answer("Who is the CEO of Nordwind?")    # not answerable - all chunks above threshold
    rag_answer("What is the airspeed of a swallow?")  # totally unrelated - definitely all dropped

if __name__ == "__main__":
    main()
    langfuse.flush()
