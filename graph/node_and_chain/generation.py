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

# Custom RAG prompt for high-quality answer generation
system_template = """You are an expert assistant providing accurate, well-structured answers based on retrieved source documents.

Your task is to answer the user's question using ONLY the information provided in the context documents below.

Guidelines for Answer Generation:
1. **Accuracy**: Base your answer strictly on the provided context. Do not add external knowledge or assumptions.
2. **Completeness**: Provide a comprehensive answer that fully addresses the question using all relevant information from the documents.
3. **Clarity**: Write in clear, concise language that is easy to understand.
4. **Structure**: Organize your answer logically with proper paragraphs or bullet points when appropriate.
5. **Honesty**: If the context does not contain enough information to answer the question fully, clearly state what information is missing.
6. **Citations**: When referencing specific information, you may indicate which part of the context it comes from.

Context Documents:
{context}

Instructions:
- If the context fully answers the question, provide a complete and detailed response.
- If the context only partially answers the question, provide what you can and acknowledge any gaps.
- If the context does not answer the question at all, clearly state: "Based on the provided documents, I cannot answer this question as the information is not available in the source material."
- Do not make up or infer information that is not explicitly stated or reasonably implied in the context.

Answer the question below based on the context above."""

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
    return {"documents": documents, "question": question, "generation": generation}
