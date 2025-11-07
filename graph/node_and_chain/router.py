from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

class RouteQuery(BaseModel):
    datasource: Literal["vector_store", "web_search"] = Field(
        description="Given the question, decide whether to route to 'vector_store' or 'web_search' to find relevant information."
)
    
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0
)

structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert query router for a philosophy-focused Retrieval Augmented Generation (RAG) system.

Your task is to analyze the user's question and determine the best data source to answer it.

Routing Decision:
- Route to 'vector_store' if the question is about PHILOSOPHY:
  * Philosophical concepts, theories, or arguments
  * Philosophers and their works (e.g., Mill, Kant, Plato, Aristotle, etc.)
  * Ethics, metaphysics, epistemology, logic, political philosophy
  * Philosophical terms and definitions
  * Moral reasoning, utilitarianism, deontology, virtue ethics
  * Philosophy of mind, religion, science, or language
  * Any philosophical readings, texts, or academic philosophy content
  * Examples: "What is Mill's definition of...", "Explain utilitarianism", "What does Kant say about..."

- Route to 'web_search' for NON-PHILOSOPHY questions:
  * Current events, news, sports, entertainment
  * Science, technology, math (unless philosophy of science)
  * History, geography, culture (unless philosophical history)
  * Personal advice, how-to questions, practical matters
  * General knowledge not related to philosophy
  * Examples: "What's the weather...", "How to cook...", "Latest news about..."

Important Guidelines:
1. If the question mentions ANY philosophical topic, concept, or philosopher → 'vector_store'
2. If the question is clearly not about philosophy → 'web_search'
3. Your vector store contains philosophy course materials and readings
4. When in doubt about philosophical relevance, choose 'vector_store'

Provide your routing decision as either 'vector_store' or 'web_search'."""

human = """User's Question:
{question}

Task: Is this question about philosophy? If yes, route to 'vector_store'. If no (non-philosophy topic), route to 'web_search'."""

route_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", human)
])

question_router = route_prompt | structured_llm_router