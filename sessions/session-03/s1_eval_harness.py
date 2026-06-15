# s1_eval_harness.py
# WHAT: Run a prompt against known-answer cases, score automatically, get a number
# WHY: Proving a prompt works > feeling it works. The amateur/professional line.

from dotenv import load_dotenv
load_dotenv()

from langfuse import observe, get_client
from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
langfuse = get_client()

# --- GOLDEN SET: inputs paired with ground-truth answers ---
# Company rule: lukewarm/neutral = negative
DATASET = [
    {"input": "Yeah it's pretty good, no major complaints.", "expected": "positive"},
    {"input": "This is the best thing I've ever bought!", "expected": "positive"},
    {"input": "Total garbage, broke in a day.",            "expected": "negative"},
    {"input": "It's fine, I guess.",                        "expected": "negative"},
    {"input": "Absolutely incredible, ten out of ten.",     "expected": "positive"},
    {"input": "Meh. It does the job.",                      "expected": "negative"},
    {"input": "I'm obsessed, can't recommend enough!",      "expected": "positive"},
    {"input": "Yeah it's pretty good, no major complaints.", "expected": "negative"},
    

]

# The prompt we're testing (few-shot teaches the lukewarm=negative rule)
def build_messages(text: str) -> list:
    return [
        {"role": "system", "content": "Classify sentiment as positive or negative. Reply with ONE word, lowercase."},
        {"role": "user", "content": "It's okay, nothing special."},
        {"role": "assistant", "content": "negative"},
        {"role": "user", "content": "Love it so much!"},
        {"role": "assistant", "content": "positive"},
     {"role": "user", "content": "It's pretty good, does what it should."},
        {"role": "assistant", "content": "negative"},

        {"role": "user", "content": text},
    ]
    

def score(output: str, expected: str) -> bool:
    # The engineering decision: normalize before comparing.
    # Strip whitespace, lowercase, remove trailing punctuation.
    cleaned = output.strip().lower().rstrip(".!")
    return cleaned == expected

@observe(as_type="generation")
def run_one(text: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=build_messages(text),
        temperature=0.3,
        max_tokens=10,
    )
    langfuse.update_current_generation(
        model="llama-3.3-70b-versatile",
        usage_details={"input": response.usage.prompt_tokens,
                       "output": response.usage.completion_tokens},
    )
    return response.choices[0].message.content

@observe()
def main():
    passed = 0
    print(f"{'INPUT':<45} {'EXPECTED':<10} {'GOT':<12} RESULT")
    print("-" * 80)
    for case in DATASET:
        raw = run_one(case["input"])
        ok = score(raw, case["expected"])
        passed += ok
        mark = "PASS" if ok else "FAIL"
        print(f"{case['input'][:43]:<45} {case['expected']:<10} {raw.strip()[:10]:<12} {mark}")

    total = len(DATASET)
    print("-" * 80)
    print(f"SCORE: {passed}/{total}  ({100*passed//total}%)")

if __name__ == "__main__":
    main()
    langfuse.flush()
