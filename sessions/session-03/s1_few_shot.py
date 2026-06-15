# s1_few_shot.py
# WHAT: Zero-shot (instruction only) vs few-shot (examples) on a task with a custom rule
# WHY: Showing a pattern beats describing it - and makes cheap models behave precisely

from dotenv import load_dotenv
load_dotenv()

from langfuse import observe, get_client
from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
langfuse = get_client()

TEST_INPUT = "It's okay I guess."

@observe(as_type="generation")
def classify(messages: list, label: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.0,
        max_tokens=10,
    )
    langfuse.update_current_generation(
        model="llama-3.3-70b-versatile",
        usage_details={"input": response.usage.prompt_tokens,
                       "output": response.usage.completion_tokens},
    )
    answer = response.choices[0].message.content
    print(f"{label}: {answer}")
    return answer

@observe()
def main():

    # VERSION 2 - FEW SHOT: demonstrate the custom rule (lukewarm = negative)
    few_shot = [
        {"role": "system", "content": "Classify the sentiment as positive or negative. Reply with one word."},
        {"role": "user", "content": "This is amazing, best purchase ever!"},
        {"role": "assistant", "content": "positive"},
        {"role": "user", "content": "It's fine, does the job."},
        {"role": "assistant", "content": "negative"},
        {"role": "user", "content": "Absolutely love it, exceeded expectations."},
        {"role": "assistant", "content": "positive"},
        {"role": "user", "content": TEST_INPUT},
    ]
    classify(few_shot, "FEW-SHOT   ('It's okay I guess.')")
    # VERSION 1 - ZERO SHOT: just an instruction
    zero_shot = [
        {"role": "system", "content": "Classify the sentiment as positive or negative. Reply with one word."},
        {"role": "user", "content": TEST_INPUT},
    ]
    classify(zero_shot, "ZERO-SHOT  ('It's okay I guess.')")

        

if __name__ == "__main__":
    main()
    langfuse.flush()
    print("\nNormal sentiment says 'okay' = positive/neutral.")
    print("This company's rule: lukewarm = negative. Only ONE version learned that.")
