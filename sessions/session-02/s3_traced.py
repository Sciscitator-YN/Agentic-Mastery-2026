# s3_traced.py
# WHAT: First LLM call with full LangFuse tracing
# WHY: Truth lives in traces, not in asking the model about itself

from dotenv import load_dotenv
load_dotenv()

from langfuse import observe, get_client
from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
langfuse = get_client()

@observe(as_type="generation")
def ask_groq(question: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": question}],
        temperature=0.3,
        max_tokens=200,
    )
    # Attach real usage data to the trace
    langfuse.update_current_generation(
        model="llama-3.3-70b-versatile",
        usage_details={
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
        },
    )
    return response.choices[0].message.content

@observe()
def main():
    answer = ask_groq("In 2 sentences: why do LLMs hallucinate?")
    print(answer)

if __name__ == "__main__":
    main()
    langfuse.flush()  # ensure traces send before script exits
    print("\n>>> Now check cloud.langfuse.com -> Traces")
