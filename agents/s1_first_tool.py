# s1_first_tool.py

import os
from dotenv import load_dotenv
from langfuse import observe, get_client
from groq import Groq
import json

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
langfuse = get_client()

MODEL = "llama-3.3-70b-versatile"

def get_weather(city: str) :
    temp =47
    return temp

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current temperature for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"],
            },
        },
    }
]




@observe(as_type="generation")
def _chat(messages: list, use_tools: bool):
    response = client.chat.completions.create(
        model=MODEL,
        tools=tools if use_tools else None,
        messages=messages,
        temperature=0.0,
        max_tokens=150,
    )

    langfuse.update_current_generation(
        model=MODEL,
        usage_details={"input": response.usage.prompt_tokens,
                       "output": response.usage.completion_tokens},
    )

    return response


@observe()
def ask(messages: list):
    response = _chat(messages, use_tools=True)

    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        # No tool needed — return the model's direct answer.
        answer = response.choices[0].message.content
        print(f"A: {answer}\n")
        return answer

    tool_call = tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    city = args["city"]

    result = get_weather(city)

    tool_message = {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": str(result)
    }

    messages.append(response.choices[0].message)
    messages.append(tool_message)

    final_response = _chat(messages, use_tools=False)

    answer = final_response.choices[0].message.content
    print(f"A: {answer}\n")

    return answer


@observe()
def main():
    question = "whats the weather in toronto"
    messages = [{"role": "system", "content": "You are a helpful assistant that uses tools when needed and Report ONLY the exact data returned by the tool. Do not add weather conditions, humidity, wind, unit conversions, or any detail not explicitly provided"},
                {"role": "user", "content": question}]
    ask(messages)


if __name__ == "__main__":
    main()
    langfuse.flush()
