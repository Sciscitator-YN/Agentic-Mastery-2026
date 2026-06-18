# s4_rag_loop.py
# WHAT: The complete RAG pipeline - retrieve relevant chunks, generate a grounded answer
# WHY: This IS the product. "AI that answers from our docs" = this, ~40 lines.

from dotenv import load_dotenv
load_dotenv()

from langfuse import observe, get_client
from groq import Groq
import chromadb
import os

groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
langfuse = get_client()

# ---------- INGESTION (build the library once) ----------
KNOWLEDGE = [
    "Nordwind Logistics was founded in 2021 and is based in Hamburg, Germany.",
    "The Nordwind platform automatically matches available trucks to incoming freight loads.",
    "Pricing starts at two thousand dollars per month for small fleets.",
    "Enterprise plans include dedicated support and custom integrations.",
    "Nordwind offers a free trial that lasts fourteen days with no credit card required.",
    "The platform reduced manual dispatch time by approximately forty percent for early customers.",
]

chroma = chromadb.Client()
collection = chroma.create_collection(name="nordwind")
collection.add(documents=KNOWLEDGE, ids=[f"k{i}" for i in range(len(KNOWLEDGE))])
print(f"Ingested {len(KNOWLEDGE)} documents into the knowledge base.\n")

# ---------- QUERY TIME (runs per question) ----------
@observe(as_type="generation")
def rag_answer(question: str) -> str:
    # 1. RETRIEVE: find the most relevant chunks
    results = collection.query(query_texts=[question], n_results=3)
    retrieved = results["documents"][0]
    distances = results["distances"][0]

    # 2. BUILD CONTEXT: stitch the retrieved chunks together
    context = "\n".join(f"- {chunk}" for chunk in retrieved)

    # 3. GENERATE: answer grounded ONLY in the retrieved context
    system = (
        "You answer questions using ONLY the provided context. "
        "If the answer is not in the context, say exactly: 'I don't have that information.' "
        "Do not use any outside knowledge."
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
    # Show what was retrieved so you can SEE the grounding
    print(f"Q: {question}")
    print(f"  Retrieved (closest distance {distances[0]:.3f}):")
    for d, c in zip(distances, retrieved):
        print(f"    [{d:.3f}] {c}")
    print(f"  A: {answer}\n")
    return answer

@observe()
def main():
    rag_answer("How long is the free trial?")     # answer IS in the docs
    rag_answer("Who is the CEO of Nordwind?")      # answer is NOT in the docs

if __name__ == "__main__":
    main()
    langfuse.flush()
