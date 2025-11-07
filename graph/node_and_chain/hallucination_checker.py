from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
load_dotenv()

llm = ChatGoogleGenerativeAI(
  model="gemini-2.5-flash",
  temperature=0
)

class CheckHallucination(BaseModel):

    binary_score: bool = Field(description="Is the answer grounded in the documents? 'yes' or 'no'")

structured_llm_grader = llm.with_structured_output(CheckHallucination)

system = """You are a strict grader evaluating whether an AI-generated answer is grounded in and supported by the provided source documents.

Your task is to assess if the answer contains only information that can be directly found in or reasonably inferred from the given documents.

Grading Criteria:
- **yes**: The answer is fully grounded in the documents. All claims, facts, and statements can be traced back to the source documents.
- **no**: The answer contains information not present in the documents, makes unsupported claims, or includes hallucinated content.

Important Guidelines:
1. Be strict: If ANY part of the answer cannot be verified from the documents, grade it as 'no'
2. Reasonable inference is acceptable, but speculation is not
3. Direct quotes or paraphrasing from documents should be graded as 'yes'
4. Generic knowledge not specific to the documents should be graded as 'no' unless it's clearly supported by the documents
5. If the answer contradicts the documents, grade it as 'no'

Provide a binary score of 'yes' or 'no' to indicate whether the answer is grounded or supported in the provided documents."""

human = """I need you to evaluate if the generated answer is grounded in the source documents.

User's Question:
{question}

Source Documents (Context):
{documents}

Generated Answer to Evaluate:
{generation}

Task: Determine if the answer above is fully supported by the source documents. Grade as 'yes' if grounded, or 'no' if it contains unsupported information."""

hallucination_checker_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", human),
    ]
)


hallucination_checker = hallucination_checker_prompt | structured_llm_grader