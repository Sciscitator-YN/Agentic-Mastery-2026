# s3_memory.py
# WHAT: Stateless model + stateful application = the memory illusion
# WHY: This is how EVERY chatbot and agent works under the hood
# Watch the token count — that climbing number IS your context window filling

from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM = """You are D's sharp, concise AI assistant.
You remember everything said in this conversation.
If asked what you remember, summarize the full conversation so far."""

def chat(history: list, user_input: str) -> tuple[str, list]:
    history.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": SYSTEM}] + history,
        temperature=0.3,
        max_tokens=200,
    )

    reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": reply})

    total = response.usage.total_tokens
    print(f"  [tokens used this call: {total}]")

    return reply, history

def main():
    print("Multi-turn chat — type 'quit' to exit")
    print("Try these 3 messages in order and watch tokens grow:\n")
    print("  1. My name is D and I am building AI agents")
    print("  2. I am using Groq with llama-3.3-70b")
    print("  3. What do you remember about me?\n")

    history = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        if not user_input:
            continue

        reply, history = chat(history, user_input)
        print(f"AI:  {reply}\n")

    print(f"\nConversation: {len(history)} messages")
    print("The model stored NONE of this. We did. That's the whole trick.")

if __name__ == "__main__":
    main()
