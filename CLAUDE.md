# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A learning/experimentation repo for building LLM agents ("Agentic Mastery 2026"). Each file is a
small, standalone, runnable script/experiment rather than part of a shared application — there is
no shared library code, package structure, or test suite.

## Setup & running

- Dependencies are installed in `.venv` (created with `uv`, no `requirements.txt`/`pyproject.toml`
  is checked in). Run scripts with `.venv/bin/python <path>` or after `source .venv/bin/activate`.
- Groq-based scripts read `GROQ_API_KEY` from a root-level `.env` via `python-dotenv`
  (`load_dotenv()` + `os.getenv("GROQ_API_KEY")`). The `.env` file is gitignored.
- Local-model scripts (`main.py`, `sessions/session-01/test_local.py`) require Ollama running at
  `http://localhost:11434` with the relevant models pulled (e.g. `phi3:mini`, `phi4-mini`,
  `llama3.2:3b`, `gemma3:4b`).
- Scripts requiring user input (e.g. `s3_memory.py`) run an interactive REPL loop in the terminal
  (type `quit` to exit).

## Structure

- `main.py`, `first_agent.py` — root-level early experiments (Ollama chat, LangChain+Groq).
- `sessions/session-NN/` — numbered learning sessions, each containing a handful of `sN_*.py`
  scripts that build on a single concept (e.g. session 01: agent anatomy, system prompt /
  temperature levers, conversation memory via message history + token usage).

## Conventions used across scripts

- Groq scripts standardize on `client.chat.completions.create(...)` with
  `model="llama-3.3-70b-versatile"`.
- Each script is self-contained: top-of-file comments (`# WHAT:` / `# WHY:`) explain the concept
  being demonstrated, and the script prints its own experiment output directly — there's no
  shared runner or assertions to satisfy.

## Security

- API keys live ONLY in the root `.env` file, never hardcoded in script files.
- Before any git commit, scan staged files for leaked key patterns:
  `git diff --cached | grep -i "gsk_"`

## Known issue

`first_agent.py` has a Groq API key hardcoded as a string literal (committed to git). Treat this
as already-leaked/rotated, don't reuse it, and prefer the `.env` + `os.getenv("GROQ_API_KEY")`
pattern used elsewhere if editing this file.
