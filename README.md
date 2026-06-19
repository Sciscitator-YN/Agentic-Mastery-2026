# Agentic Mastery 2026

Building production AI systems from primitives to multi-agent architectures — every layer written from raw API calls up.

## Why this exists

The fastest way to look productive with LLMs is to wire together a framework and trust the abstractions. The fastest way to actually understand them is the opposite: build each piece from scratch — the raw API call, the token accounting, the retrieval funnel, the eval loop — until there are no black boxes left. This repo is that path. Each phase rebuilds a production capability from first principles, so the behavior of the whole system is something I can reason about, not just invoke.

## What's inside

A progression from single API calls to grounded, observable retrieval systems. Each phase adds a capability that production AI work actually depends on.

**Phase 1 — LLM API primitives.** Streaming responses and time-to-first-token, the token economics behind cost and context limits, and full request-level observability with LangFuse tracing.

**Phase 2 — Prompt & context engineering.** Structured outputs validated into typed objects with Pydantic, chain-of-thought reasoning, few-shot prompting, and a custom eval harness that scores prompts against known-answer cases instead of eyeballing them.

**Phase 3 — Retrieval-augmented generation.** Embeddings and semantic search, vector storage, chunking strategies, cross-encoder reranking, and hybrid dense + sparse retrieval — assembled into a complete production retrieval funnel.

## Flagship: `rag/production_rag.py`

A complete RAG funnel with every stage visible and instrumented:

**hybrid retrieval** (dense vectors + BM25 sparse, fused with weighted reciprocal rank fusion) → **cross-encoder reranking** by true relevance → **threshold gate** that drops weak matches → **grounded generation** constrained to the surviving evidence.

The pipeline fails safe: when no chunk clears the relevance threshold, it refuses *without calling the LLM at all* — no relevant source means no answer and no opportunity to hallucinate. Every stage is traced in LangFuse.

### Run it

```bash
git clone https://github.com/<you>/Agentic-Mastery-2026.git
cd Agentic-Mastery-2026

# deps are pinned in requirements.txt (env created with uv)
uv venv && uv pip install -r requirements.txt

# one required key; tracing is optional
echo "GROQ_API_KEY=your_key_here" > .env

.venv/bin/python rag/production_rag.py
```

`GROQ_API_KEY` is the only requirement. LangFuse keys are optional — without them the pipeline runs identically, just untraced. The cross-encoder model (`ms-marco-MiniLM-L-6-v2`, ~80MB) downloads on first run, then caches.

The script runs three built-in queries that exercise every branch of the funnel:

```text
Q: What's the typical cold brew coffee-to-water ratio?
  [STAGE 1] HYBRID retrieval (dense + sparse, weighted RRF)
  [STAGE 2] RERANK (cross-encoder)
  [STAGE 3] THRESHOLD gate -> 2 of 8 chunks survived
  [STAGE 5] GROUNDED generation
  A: A common cold brew concentrate ratio is one part coffee to eight parts water by weight.

Q: How many calories are in a Starbucks iced caramel macchiato?
  [STAGE 3] THRESHOLD gate -> 0 chunks survived
  [STAGE 4] No survivors -> refusing without an LLM call.
  A: I don't have that information.

Q: What year did the Berlin Wall fall?
  [STAGE 4] No survivors -> refusing without an LLM call.
  A: I don't have that information.
```

The first question is answered straight from retrieved evidence; the second is on-topic but absent from the knowledge base; the third is unrelated. Both unanswerable cases are refused at the gate — no LLM call, no hallucination.

## Stack

Python · Groq · Ollama · ChromaDB · sentence-transformers · LangFuse · Claude Code

## Status

Actively building. Phases 1–3 complete; currently entering **Phase 4 — tool calling & agents**.
