# s2_react_loop.py
#
# WHAT: A ReAct-style agent loop — the model keeps calling tools, observing
#       results, and reasoning over them until it produces a final answer.
# WHY:  s1 was single-shot (one tool call, one answer). Real tasks need
#       multiple steps: call a tool, see the result, decide the next move.
#       The loop is what turns a tool-caller into an agent.

import os
from dotenv import load_dotenv
from langfuse import observe, get_client
from groq import Groq
import json
import ast, operator

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
langfuse = get_client()

MODEL = "llama-3.3-70b-versatile"
MAX_STEPS = 5  # safety cap so the loop can't run forever


# --- Tools -----------------------------------------------------------------

def get_weather(city: str):
    # Return labeled data so the model knows what the number means.
    # A bare `47` is ambiguous (units? what metric?) and the strict
    # system prompt makes the model refuse to interpret it.
    return {"city": city, "temperature_f": 47}

def calculator(expr: str) -> float:
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, 
           ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}
    
    def _eval(node):
        if isinstance(node, ast.Expression): return _eval(node.body)
        if isinstance(node, ast.Constant): return node.value
        if isinstance(node, ast.BinOp): return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp): return ops[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported operation")

    return _eval(ast.parse(expr.strip(), mode='eval'))



# Maps tool name -> python callable, so the loop can dispatch by name.
TOOL_FUNCS = {
    "get_weather": get_weather,
    "calculator": calculator,
}

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
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression (+, -, *, /, **)",
            "parameters": {
                "type": "object",
                "properties": {
                    "expr": {"type": "string", "description": "Arithmetic expression, e.g. '47 * 9 / 5 + 32'"}
                },
                "required": ["expr"],
            },
        },
    },
]


# --- LLM call 

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


# --- ReAct loop ------------------------------------------------------------

@observe()
def run_tool(tool_call):
    """Dispatch a single tool call and return its tool-role message."""
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    result = TOOL_FUNCS[name](**args)

    # Make the act -> observe cycle visible.
    arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
    print(f"   ├─ act:     {name}({arg_str})")
    print(f"   └─ observe: {result}")

    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": str(result),
    }


@observe()
def ask(messages: list):
    """Loop: think -> act -> observe, until the model answers without a tool."""
    for step in range(MAX_STEPS):
        # On the final allowed step, turn tools OFF so the model is forced
        # to answer from the observations it already has, instead of
        # re-calling tools forever and hitting MAX_STEPS with no answer.
        last_step = step == MAX_STEPS - 1
        print(f"── step {step + 1}/{MAX_STEPS} {'(tools off)' if last_step else ''}")
        response = _chat(messages, use_tools=not last_step)
        message = response.choices[0].message
        messages.append(message)

        tool_calls = message.tool_calls
        if not tool_calls:
            # No tool requested -> this is the final answer.
            answer = message.content
            print(f"\nA: {answer}\n")
            return answer

        # The model's reasoning before it decided to call tools, if any.
        if message.content:
            print(f"   think:   {message.content}")

        # Act on every tool the model asked for, append observations.
        for tool_call in tool_calls:
            messages.append(run_tool(tool_call))

    print("A: (stopped — reached MAX_STEPS)\n")
    return None


@observe()
def main():
    question = "What’s the weather in Tokyo, and what’s 47 times 3?"
    messages = [
        {"role": "system", "content": "You are a helpful assistant that uses tools when needed and return a answer as soon as you are satisfied  from  the tools result"},
        {"role": "user", "content": question},
    ]
    ask(messages)


if __name__ == "__main__":
    main()
    langfuse.flush()
