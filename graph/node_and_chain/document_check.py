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

system = """Evaluate if the document can help answer the question.

Grade 'yes': Document contains relevant answers, keywords, concepts, context, or semantically aligned info.
Grade 'no': Document completely off-topic or unhelpful.

Be lenient - any useful information = 'yes'."""
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
