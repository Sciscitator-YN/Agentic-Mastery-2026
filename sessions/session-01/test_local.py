# test_local.py — debug version
import requests

def ask_local(model: str, prompt: str):
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
    )
    print(f"Status code: {response.status_code}")
    data = response.json()
    print(f"Raw response: {data}")
    print()

    prompt_tokens = data.get("prompt_eval_count")
    completion_tokens = data.get("eval_count")
    if prompt_tokens is not None and completion_tokens is not None:
        total = prompt_tokens + completion_tokens
        print(f"  [tokens evaluated this call (cached prefix excluded): {total}]")
        print(f"  [prompt tokens: {prompt_tokens}]")
        print(f"  [completion tokens: {completion_tokens}]")
    else:
        print("  [token usage data was not provided by the local server]")
    print()

    return data["message"]["content"]

models = ["phi4-mini", "llama3.2:3b", "gemma3:4b"]

for model in models:
    print(f"=== {model} ===")
    try:
        answer = ask_local(model, "Say 'I am working'")
        print(answer)
    except Exception as e:
        print(f"ERROR: {e}")
