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

class GradeDocuments(BaseModel):

    binary_score: str = Field(description="Documents are relevant to the question: 'yes' or 'no'")
    

structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader


def grade_document(state: GraphState) -> Dict[str, Any]:
    print("--Grade Document---")
    question = state["question"]
    documents= state["documents"]
    filtered_doc=[]
    web_search = False
    for doc in documents:
        score: GradeDocuments = retrieval_grader.invoke(
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
