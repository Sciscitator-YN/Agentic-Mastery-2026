
from dotenv import load_dotenv
load_dotenv()

from langfuse import observe, get_client
from pydantic import BaseModel, Field
from groq import Groq
from enum import Enum
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
langfuse = get_client()


class Urgency(Enum):
    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class JobRequest(BaseModel):
    job_contact:str = Field(description="The contact name who requested  for the job")
    job_company:str = Field(description="The company name who requested the job")
    job_company_size:int = Field(description="The company size in number of employees")
    job_painpoint:str = Field(description="The pain point of the company that is requesting the job")
    job_min_budget:int = Field(description="The minimum budget for the job")
    job_max_budget:int = Field(description="The maximum budget for the job")
    job_urgency:Urgency = Field(description="The urgency level based on the time frame and the description of the urgency")
    job_contact_email:str = Field(description="The email address of the contact who requested the job") 

MESSY_JOB_REQUEST = """Hi, this is Sarah Chen from Nordwind Logistics. We're a mid-size 
freight company, about 200 employees. Our dispatchers waste hours 
every day manually matching trucks to loads. Budget's somewhere 
around 30-50k for the right solution. Pretty urgent - we'd want 
something live within 3 months. You can reach me at s.chen@nordwind.co"""

@observe(as_type="generation")
def extract(text: str) -> JobRequest:
    schema = JobRequest.model_json_schema()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": f"Extract job request data. Respond ONLY with json matching this schema, no other text:\n{schema}"},
                  {"role": "user", "content": text}],
        temperature=0.0,
        max_tokens=400,
        response_format={"type": "json_object"},
    )
    langfuse.update_current_generation(
        model="llama-3.3-70b-versatile",
        usage_details={"input": response.usage.prompt_tokens,
                       "output": response.usage.completion_tokens},
    )
    raw = response.choices[0].message.content
    return JobRequest.model_validate_json(raw)


@observe(as_type="generation")
def ask_summary(job: JobRequest) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"Summarize the following job request and depending on thiere urgency return a short summary of the job request and is there a possibility to incraese budget: {job.model_dump_json()}"}],
        temperature=0.3,
        max_tokens=200,
    )
    # Attach real usage data to the trace
    langfuse.update_current_generation(
        model="llama-3.3-70b-versatile",
        usage_details={
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
        },
    )
    return response.choices[0].message.content




@observe()
def main():

    job = extract(MESSY_JOB_REQUEST)
    print(job)
    print("\n"*4)
    summary = ask_summary(job)
    print(summary)
    print("\n"*4)


if __name__ == "__main__":
    main()