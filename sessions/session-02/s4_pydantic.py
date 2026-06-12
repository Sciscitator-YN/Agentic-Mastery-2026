# s4_pydantic.py
# WHAT: Schema-enforced extraction - model output validated into typed Python objects
# WHY: json_mode guarantees syntax; Pydantic guarantees YOUR contract. Foundation of tools/agents.

from dotenv import load_dotenv
load_dotenv()

from langfuse import observe, get_client
from pydantic import BaseModel, Field
from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
langfuse = get_client()

# ---- THE CONTRACT ----
class JobPosting(BaseModel):
    title: str = Field(description="Job title")
    company: str = Field(description="Company name")
    location: str = Field(description="City/region, or 'Remote'")
    salary_min: int = Field(description="Minimum annual salary in USD, plain integer")
    salary_max: int = Field(description="Maximum annual salary in USD, plain integer")
    skills: list[str] = Field(description="Required technical skills")
    remote_ok: bool = Field(description="True if remote work is allowed")

MESSY_POSTING = """
We're hiring!! Senior AI Engineer @ FlowMatic (Series B, SF Bay Area)
Comp: $120k-$150k DOE + equity. Hybrid ok, 2 days in office, rest remote.
Must know: Python, LangGraph, RAG pipelines, vector DBs. Bonus: MCP experience.
Apply at flowmatic.io/careers
"""

@observe(as_type="generation")
def extract(text: str) -> JobPosting:
    schema = JobPosting.model_json_schema()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": f"Extract job posting data. Respond ONLY with json matching this schema, no other text:\n{schema}"},
            {"role": "user", "content": text},
        ],
        temperature=0.0,
        max_tokens=400,
        response_format={"type": "json_object"},
    )
    langfuse.update_current_generation(
        model="llama-3.3-70b-versatile",
        usage_details={
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
        },
    )
    raw = response.choices[0].message.content
    print(f"--- RAW MODEL OUTPUT ---\n{raw}\n")

    # The enforcement moment: string -> validated typed object (or loud failure)
    return JobPosting.model_validate_json(raw)

@observe()
def main():
    job = extract(MESSY_POSTING)

    print("--- VALIDATED OBJECT ---")
    print(f"Type:        {type(job).__name__}")
    print(f"Title:       {job.title}")
    print(f"Company:     {job.company}")
    print(f"Salary max:  {job.salary_max}  (type: {type(job.salary_max).__name__})")
    print(f"Skills:      {job.skills}")
    print(f"Remote:      {job.remote_ok}")

    # Proof it's real typed data, not strings:
    print(f"\nMidpoint salary (math on extracted data): ${(job.salary_min + job.salary_max) // 2:,}")

if __name__ == "__main__":
    main()
    langfuse.flush()
