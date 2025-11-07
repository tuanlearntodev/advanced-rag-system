from langsmith import Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from typing import Any, Dict
from graph.state import GraphState
from dotenv import load_dotenv
import os
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)
client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
prompt = client.pull_prompt("rlm/rag-prompt", include_model=True)

generation_chain = prompt | llm | StrOutputParser()

def generate_answer(state: GraphState) -> Dict[str, Any]:
    print("--Generate Answer---")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"question": question, "context": documents})
    return {"documents": documents, "question": question, "generation": generation}
