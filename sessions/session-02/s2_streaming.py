# s2_streaming.py
# WHAT: Same prompt, non-streaming vs streaming - measure TTFT
# WHY: TTFT is the #1 UX metric in every AI product you'll ship

from dotenv import load_dotenv
from groq import Groq
import os
import time

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

PROMPT = "Explain how a car engine works in about 150 words."

print("=" * 50)
print("MODE 1 - NON-STREAMING (wait for everything)")
print("=" * 50)

start = time.time()
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": PROMPT}],
    temperature=0.3,
    max_tokens=300,
)
wait_time = time.time() - start

print(response.choices[0].message.content)
print(f"\n>>> You stared at a blank screen for {wait_time:.2f}s, then got it ALL at once")

print()
print("=" * 50)
print("MODE 2 - STREAMING (tokens as they generate)")
print("=" * 50)

start = time.time()
ttft = None

stream = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": PROMPT}],
    temperature=0.3,
    max_tokens=300,
    stream=True,
)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        if ttft is None:
            ttft = time.time() - start
        print(content, end="", flush=True)

total = time.time() - start
print(f"\n\n>>> First token in {ttft:.2f}s | total {total:.2f}s")
print(">>> The user was reading the whole time. Nothing felt frozen.")
