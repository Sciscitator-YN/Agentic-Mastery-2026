# s1_chain_of_thought.py
# WHAT: Same hard problem, forced-fast vs forced-reasoning. Measure the accuracy gap.
# WHY: Models reason BY writing tokens - no hidden scratchpad. CoT is the base of all agent reasoning.

from dotenv import load_dotenv
load_dotenv()

from langfuse import observe, get_client
from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
langfuse = get_client()

PROBLEM = """A store had 120 apples. Monday they sold 25% of them.
Tuesday they sold 30% of what REMAINED after Monday.
Wednesday a delivery added 40 apples.
How many apples does the store have at the end of Wednesday?"""

@observe(as_type="generation")
def ask(system: str, label: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": PROBLEM},
        ],
        temperature=0.0,
        max_tokens=500,
    )
    langfuse.update_current_generation(
        model="llama-3.3-70b-versatile",
        usage_details={"input": response.usage.prompt_tokens,
                       "output": response.usage.completion_tokens},
    )
    print(f"\n{'='*55}\n{label}\n{'='*55}")
    print(response.choices[0].message.content)
    return response.choices[0].message.content

@observe()
def main():
    # Version 1: forbid reasoning, force an instant answer
    ask("You are a calculator. Reply with ONLY the final number, nothing else. No explanation.",
        "VERSION 1 - FORCED FAST (no room to compute)")

    # Version 2: demand explicit step-by-step reasoning first
    ask("Solve step by step. Show each calculation on its own line. State the final answer last.",
        "VERSION 2 - CHAIN OF THOUGHT (room to compute)")

if __name__ == "__main__":
    main()
    langfuse.flush()
    print(f"\n\nCorrect answer: 120 -> 90 -> 63 -> 103")
