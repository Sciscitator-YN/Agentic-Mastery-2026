# s3_chunking.py
# WHAT: Split a long text into pieces two ways - blind char-split vs sentence-aware
# WHY: Chunking is the most underestimated lever in RAG - where most systems silently fail

import chromadb

# A realistic blob of text - multiple facts run together (like real document content)
DOCUMENT = (
    "The Nordwind freight platform launched in 2021 to automate dispatch. "
    "It matches available trucks to incoming loads using a scheduling engine. "
    "The system reduced manual dispatch time by roughly forty percent. "
    "Pricing starts at two thousand dollars per month for small fleets. "
    "Enterprise plans include dedicated support and custom integrations. "
    "A free trial runs for fourteen days with no credit card required."
)

# --- STRATEGY 1: NAIVE - split every 80 characters, no regard for meaning ---
def naive_chunks(text, size=80):
    return [text[i:i+size] for i in range(0, len(text), size)]

# --- STRATEGY 2: SMART - split on sentence boundaries (period + space) ---
def sentence_chunks(text):
    parts = text.split(". ")
    # re-add the period that split removed (except possibly the last)
    return [p if p.endswith(".") else p + "." for p in parts]

naive = naive_chunks(DOCUMENT)
smart = sentence_chunks(DOCUMENT)

print("=" * 70)
print(f"NAIVE CHUNKS (split every 80 chars) - {len(naive)} chunks:")
print("=" * 70)
for i, c in enumerate(naive):
    print(f"[{i}] {c!r}")   # !r shows quotes so you SEE where cuts land

print("\n" + "=" * 70)
print(f"SMART CHUNKS (split on sentences) - {len(smart)} chunks:")
print("=" * 70)
for i, c in enumerate(smart):
    print(f"[{i}] {c!r}")

# Now store BOTH ways and query the same question, compare what comes back
query = "How much does it cost per month?"

def retrieve(chunks, label):
    client = chromadb.Client()
    col = client.create_collection(name=label)
    col.add(documents=chunks, ids=[f"{label}{i}" for i in range(len(chunks))])
    res = col.query(query_texts=[query], n_results=1)
    return res["documents"][0][0], res["distances"][0][0]

print("\n" + "=" * 70)
print(f"QUERY: {query}")
print("=" * 70)
n_doc, n_dist = retrieve(naive, "naive")
s_doc, s_dist = retrieve(smart, "smart")
print(f"\nNAIVE best match (distance {n_dist:.3f}):\n  {n_doc!r}")
print(f"\nSMART best match (distance {s_dist:.3f}):\n  {s_doc!r}")
