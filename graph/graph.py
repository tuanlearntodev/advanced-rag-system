from dotenv import load_dotenv
from langgraph.graph import END,StateGraph
from graph.node_and_chain import hallucination_checker, retrieve, generate_answer, document_check, web_search, answer_checker, question_router, RouteQuery
from graph.state import GraphState
load_dotenv()

RETRIEVE = "retrieve"
GENERATE_ANSWER = "generate_answer"
DOCUMENT_CHECK = "document_check"
WEB_SEARCH = "web_search"


def decide_web_search(state: GraphState) -> str:
    print("--Decide Web Search---")
    if state["web_search"]:
        return WEB_SEARCH
    else:
        return GENERATE_ANSWER

def enable_crag(state: GraphState) -> str:
    print("--Enable CRAG---")
    if state["crag"]:
        return DOCUMENT_CHECK
    else:
        return GENERATE_ANSWER

def check_hallucination_and_answer(state: GraphState) -> str:
    print("--Check Hallucination---")
    question = state["question"]
    generation = state["generation"]
    documents = state["documents"]
    
    score = hallucination_checker.invoke(
        {"question": question, "generation": generation, "documents": documents}
    )
    if score.binary_score:
        print("✓ Answer is grounded in documents")
        print("--Check Answer Relevance---")
        answer_score = answer_checker.invoke(
            {"question": question, "generation": generation}
        )
        if answer_score.binary_score.lower() == "yes":
            print("✓ Answer addresses the question")
            return "relevant"
        else:
            print("✗ Answer does not address the question")
            return "not_relevant"
    else:
        print("✗ Answer contains hallucinations")
        return "not_grounded"


def route_question(state: GraphState) -> str:
    print("--Router---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == "web_search":
        print("--Routing to Web Search---")
        return WEB_SEARCH
    else:
        print("--Routing to Vector Store---")
        return RETRIEVE
      
workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(WEB_SEARCH, web_search)
workflow.add_node(GENERATE_ANSWER, generate_answer)
workflow.add_node(DOCUMENT_CHECK, document_check)

workflow.set_conditional_entry_point(
  route_question,
  {
      RETRIEVE: RETRIEVE,
      WEB_SEARCH: WEB_SEARCH,
  }
)
workflow.add_conditional_edges(
  RETRIEVE,
  enable_crag,
  {
      DOCUMENT_CHECK: DOCUMENT_CHECK,
      GENERATE_ANSWER: GENERATE_ANSWER,
  }
)
workflow.add_conditional_edges(
  DOCUMENT_CHECK,
  decide_web_search,
  {
      WEB_SEARCH: WEB_SEARCH,
      GENERATE_ANSWER: GENERATE_ANSWER,
  }
)

workflow.add_edge(WEB_SEARCH, GENERATE_ANSWER)

workflow.add_conditional_edges(
  GENERATE_ANSWER,
  check_hallucination_and_answer,
  {
      "relevant": END,
      "not_relevant": GENERATE_ANSWER,
      "not_grounded": WEB_SEARCH,
  }
)

workflow.add_edge(GENERATE_ANSWER, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph_workflow.png")