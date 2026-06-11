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
    print(f"Raw response: {response.json()}")
    print()

models = ["phi4-mini", "llama3.2:3b", "gemma3:4b"]

for model in models:
    print(f"=== {model} ===")
    try:
        ask_local(model, "Say 'I am working'")
    except Exception as e:
        print(f"ERROR: {e}")

