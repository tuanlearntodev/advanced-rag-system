from langsmith import Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import Any, Dict
from graph.state import GraphState
from dotenv import load_dotenv
import os
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# Adaptive RAG prompt - adjusts length based on question complexity
system_template = """Answer using ONLY the provided context. Do not add external knowledge.

Adaptive Length Rules:
- Simple/factual questions: 30-50 words
- Complex questions: 200-300 words
- Step-by-step/list questions: numbered steps or bullet points

Requirements:
1. Base answer strictly on context below
2. If context lacks info, state: "Information not available in source material."
3. No headers, reasoning, or commentary—only the answer

Context:
{context}"""

human_template = """Question: {question}

Answer:"""

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", human_template)
])

generation_chain = rag_prompt | llm | StrOutputParser()

def generate_answer(state: GraphState) -> Dict[str, Any]:
    print("--Generate Answer---")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"question": question, "context": documents})
    
    # Check if answer indicates information not found
    answer_found = "information not available" not in generation.lower()
    
    return {"documents": documents, "question": question, "generation": generation, "answer_found": answer_found}
