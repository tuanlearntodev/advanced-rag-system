from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

class CheckAnswer(BaseModel):

    binary_score: str = Field(description="Does the answer address the question 'yes' or 'no'")
    
    
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)
structured_llm_grader = llm.with_structured_output(CheckAnswer)

system = """You are an expert evaluator assessing the quality and relevance of AI-generated answers.

Your task is to determine whether the generated answer properly addresses and answers the user's question.

Grading Criteria:
- Grade as 'yes' if the answer:
  * Directly addresses the question asked
  * Provides relevant information that helps answer the question
  * Contains a clear response to what was asked
  * May be partial but still attempts to answer the question

- Grade as 'no' if the answer:
  * Completely ignores the question
  * Provides irrelevant or off-topic information
  * States it cannot answer without providing any useful information
  * Discusses something entirely different from what was asked

Important Guidelines:
1. Focus on whether the answer ATTEMPTS to address the question, not whether it's perfect
2. Partial answers that provide some relevant information should be graded as 'yes'
3. Only grade as 'no' if the answer is completely unhelpful or off-topic
4. The answer doesn't need to be comprehensive, just relevant

Provide a binary score: 'yes' (addresses the question) or 'no' (does not address the question)."""

human = """User's Question:
{question}

Generated Answer:
{generation}

Task: Evaluate whether the answer above addresses and attempts to answer the user's question. Grade as 'yes' if it does, or 'no' if it doesn't."""

answer_check_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", human)
])

answer_checker = answer_check_prompt | structured_llm_grader