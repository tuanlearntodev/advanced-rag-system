from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from graph.state import GraphState
from typing import Dict, Any
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

class CheckDocuments(BaseModel):

    binary_score: str = Field(description="Documents are relevant to the question: 'yes' or 'no'")
    

structured_llm_grader = llm.with_structured_output(CheckDocuments)

system = """You are an expert document relevance evaluator for a Retrieval Augmented Generation (RAG) system.

Your task is to assess whether a retrieved document contains information that can help answer the user's question.

Grading Criteria:
- Grade as 'yes' if the document contains:
  * Direct answers or facts relevant to the question
  * Keywords, concepts, or entities mentioned in the question
  * Related context that provides background or supporting information
  * Semantic meaning aligned with the question's intent

- Grade as 'no' if the document:
  * Is completely off-topic or unrelated to the question
  * Only contains tangentially related information that doesn't help answer the question
  * Discusses different aspects of a keyword without addressing the question's intent

Important: Be lenient rather than strict. If there's ANY useful information that could contribute to answering the question, grade it as 'yes'. 
The goal is to retain potentially useful context while filtering out completely irrelevant documents.

Provide a binary score: 'yes' (relevant) or 'no' (not relevant)."""
document_check_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

document_checker = document_check_prompt | structured_llm_grader


def document_check(state: GraphState) -> Dict[str, Any]:
    print("--Grade Document---")
    question = state["question"]
    documents= state["documents"]
    filtered_doc=[]
    web_search = False
    for doc in documents:
        score: CheckDocuments = document_checker.invoke(
            {"question": question, "document": doc}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            print("Document relevant to the question.")
            filtered_doc.append(doc)
        else:
            print("Document not relevant to the question. Triggering web search.")
            web_search = True
            continue

    return {"documents": filtered_doc, "question": question, "web_search": web_search}
