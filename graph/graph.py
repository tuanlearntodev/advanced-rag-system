from dotenv import load_dotenv
from langgraph.graph import END,StateGraph
from graph.node import retrieve, generate_answer, grade_document, web_search
from graph.state import GraphState
load_dotenv()

RETRIEVE = "retrieve"
GENERATE_ANSWER = "generate_answer"
GRADE_DOCUMENT = "grade_document"
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
        return GRADE_DOCUMENT
    else:
        return GENERATE_ANSWER
      
workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(WEB_SEARCH, web_search)
workflow.add_node(GENERATE_ANSWER, generate_answer)
workflow.add_node(GRADE_DOCUMENT, grade_document)

workflow.set_entry_point(RETRIEVE)
workflow.add_conditional_edges(
  RETRIEVE,
  enable_crag,
  {
      GRADE_DOCUMENT: GRADE_DOCUMENT,
      GENERATE_ANSWER: GENERATE_ANSWER,
  }
)
workflow.add_conditional_edges(
  GRADE_DOCUMENT,
  decide_web_search,
  {
      WEB_SEARCH: WEB_SEARCH,
      GENERATE_ANSWER: GENERATE_ANSWER,
  }
)

workflow.add_edge(WEB_SEARCH, GENERATE_ANSWER)
workflow.add_edge(GENERATE_ANSWER, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph_workflow.png")