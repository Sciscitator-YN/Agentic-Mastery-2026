from dotenv import load_dotenv
from langfuse import observe, get_client
import chromadb
from groq import Groq
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
langfuse = get_client()

KNOWLEDGE = [ "Iron Man is a famous superhero from Marvel Comics. He was created by writer Stan Lee, with help from Larry Lieber, Don Heck, and Jack Kirby. Iron Man first appeared in a comic book called Tales of Suspense 39 in March 1963.",
"The hero's secret identity is Anthony Edward Tony Stark. He is a very rich and smart scientist and business magnate. Tony gets badly hurt when he is kidnapped and forced to build a dangerous weapon. Instead, he secretly builds a special suit of armor to save his life and escape.",
"Later, Tony makes his suit even better with more weapons and cool technology from his company, Stark Industries. He uses this suit to protect the world as Iron Man. For a long time, he kept his identity a secret. Iron Man was first created to explore ideas about American technology during the Cold War. Over time, the stories changed to talk about more modern issues.",]

chroma = chromadb.Client()
collection = chroma.create_collection(name="Ironman")
collection.add(documents=KNOWLEDGE, ids=[f"k{i}" for i in range(len(KNOWLEDGE))])
print(f"Ingested {len(KNOWLEDGE)} documents into the knowledge base.\n")


# ---------- QUERY TIME (runs per question) ----------
@observe(as_type="generation")
def rag_answer(question: str) -> str:
    
    results = collection.query(query_texts=[question], n_results=3)
    retrieved = results["documents"][0]
    distances = results["distances"][0]

   
    context = "\n".join(f"- {chunk}" for chunk in retrieved)
    
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
    # Show what was retrieved so you can SEE the grounding
    print(f"Q: {question}")
    print(f"  Retrieved (closest distance {distances[0]:.3f}):")
    for d, c in zip(distances, retrieved):
        print(f"    [{d:.3f}] {c}")
    print(f"  A: {answer}\n")
    return answer


@observe()
def main():
    rag_answer("Who created Iron Man?")
    rag_answer("What is Tony Stark's secret identity?")
    rag_answer("Who is the CEO of Nordwind?")  # not in the docs


if __name__ == "__main__":
    main()
    langfuse.flush()
