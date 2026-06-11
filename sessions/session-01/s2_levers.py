# s2_levers.py
# WHAT: System prompts + temperature = the two most important levers
# WHY: These define every agent's personality and reliability in production

from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def call(system: str, user: str, temperature: float) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user}
        ],
        temperature=temperature,
        max_tokens=150,
    )
    return response.choices[0].message.content

question = "Give me a name for my new AI automation startup."

print("=" * 55)
print("EXPERIMENT 1 — System Prompt Changes Everything")
print("=" * 55)

personas = [
    ("Generic",           "You are a helpful assistant."),
    ("Silicon Valley VC", "You are a Silicon Valley VC. Name things for maximum hype and investor appeal."),
    ("Minimalist",        "You are a Swiss designer. You value clarity and timeless simplicity above all."),
]

for label, system in personas:
    print(f"\n[{label}]\n{call(system, question, 0.7)}")

print("\n" + "=" * 55)
print("EXPERIMENT 2 — Temperature Changes Randomness")
print("=" * 55)

system = "You are a creative naming expert. Give one name and one sentence of reasoning."

for temp in [0.0, 0.5, 1.0, 1.5]:
    print(f"\n[Temperature {temp}]\n{call(system, question, temp)}")
