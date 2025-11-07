# CRAG Demo - Corrective Retrieval Augmented Generation

A production-ready demonstration of Corrective Retrieval Augmented Generation (CRAG) using LangChain 1.0, LangGraph, and enterprise-grade AI infrastructure. This project showcases advanced RAG techniques with self-correcting retrieval, vector database integration, and adaptive web search fallback mechanisms.

## 🎯 Project Highlights

- **Advanced RAG Architecture**: Implements CRAG methodology with intelligent document relevance scoring and adaptive retrieval strategies
- **Production-Grade Stack**: Built with LangChain 1.0, LangGraph state machines, and enterprise AI services (Google Gemini, Pinecone, Tavily)
- **Intelligent Orchestration**: Leverages LangGraph for complex multi-step agent workflows with conditional branching
- **Observability & Monitoring**: Integrated LangSmith tracing for production debugging and performance optimization
- **Scalable Design**: Batch processing, vector indexing, and efficient embedding strategies for large document collections

## Overview

This project implements a Corrective Retrieval Augmented Generation (CRAG) system that intelligently retrieves and grades document relevance before generating answers. When retrieved documents are deemed irrelevant, the system automatically falls back to web search using Tavily to ensure accurate responses—a critical feature for production AI applications requiring high accuracy and reliability.

**Key Technical Achievement**: Reduces AI hallucinations by implementing a self-correcting retrieval mechanism that validates information sources before generation, demonstrating expertise in advanced prompt engineering and RAG optimization.

## 💡 Features & Technical Skills Demonstrated

### AI/ML Engineering
- **Intelligent Document Grading**: Implemented LLM-based relevance scoring using structured outputs and Pydantic validation
- **Corrective Retrieval**: Designed adaptive fallback mechanism with web search integration for improved accuracy
- **Vector Storage & Retrieval**: Architected efficient document embedding pipeline with Pinecone vector database (768-dimensional embeddings)
- **Prompt Engineering**: Crafted domain-specific prompts for document grading and answer generation with LangSmith optimization

### Software Engineering
- **Modern AI Stack**: Production deployment using LangChain 1.0, LangGraph 1.0, and Google Gemini API
- **Graph-Based Workflows**: Built stateful, multi-actor agent orchestration with conditional branching and state management
- **API Integration**: Seamless integration of multiple AI services (Google AI, Pinecone, Tavily, LangSmith)
- **Error Handling**: Implemented robust retry logic and graceful degradation for API failures

### DevOps & Best Practices
- **Dependency Management**: Modern Python packaging with `uv` and semantic versioning
- **Environment Configuration**: Secure credential management with environment variables and `.env` files
- **Code Organization**: Modular architecture with clear separation of concerns (nodes, state, orchestration)
- **Observability**: Full request tracing and debugging with LangSmith integration
- **Version Control**: Git workflow with proper `.gitignore` and secret protection

## Architecture

The CRAG workflow follows these steps:

1. **Retrieve**: Fetch relevant documents from Pinecone vector store
2. **Grade Documents** (optional): Evaluate document relevance using Gemini
3. **Web Search**: If documents are irrelevant, search the web with Tavily
4. **Generate Answer**: Produce final response using retrieved context

## Tech Stack

- **LangChain 1.0**: Framework for building LLM applications
- **LangGraph 1.0**: Orchestration for stateful, multi-actor applications
- **Google Gemini**: LLM (gemini-2.5-flash) and embeddings (gemini-embedding-001)
- **Pinecone**: Vector database for document storage
- **Tavily**: Web search API for corrective retrieval
- **LangSmith**: Tracing and observability
- **Python 3.14**: Built with latest Python features

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd crag_demo
```

2. Install dependencies using `uv`:
```bash
uv sync
```

3. Create a `.env` file with your API keys:
```env
# Google AI API Key
GOOGLE_API_KEY=your_google_api_key

# LangSmith Configuration
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=your_project_name

# Tavily API Key
TAVILY_API_KEY=your_tavily_api_key

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
```

## Usage

### Document Ingestion

First, ingest your documents into Pinecone:

```bash
python ingestion.py
```

This will process PDF files and store them as embeddings in your Pinecone index.

### Running the CRAG System

Run the main application:

```bash
python main.py
```

The system supports two modes:
- **CRAG Mode** (`crag=True`): Grades documents and uses web search as fallback
- **Standard RAG** (`crag=False`): Uses retrieved documents without grading

Example in `main.py`:
```python
# With CRAG (document grading + web search fallback)
app.invoke({"question": "What is Mill's definition of higher pleasure?", "crag": True})

# Without CRAG (standard RAG)
app.invoke({"question": "What is Mill's definition of higher pleasure?", "crag": False})
```

## Project Structure

```
crag_demo/
├── graph/
│   ├── node/
│   │   ├── generation.py      # Answer generation
│   │   ├── grade_document.py  # Document relevance grading
│   │   ├── retrieve.py        # Vector store retrieval
│   │   └── web_search.py      # Tavily web search
│   ├── graph.py               # LangGraph workflow definition
│   └── state.py               # Graph state schema
├── ingestion.py               # Document processing and upload
├── main.py                    # Application entry point
├── .env                       # Environment variables (not committed)
└── pyproject.toml             # Project dependencies
```

## How CRAG Works

Corrective Retrieval Augmented Generation (CRAG) improves upon traditional RAG by:

1. **Self-Correction**: Evaluating whether retrieved documents are actually relevant using LLM-based grading
2. **Knowledge Refinement**: Using web search to supplement or replace low-quality retrievals
3. **Accuracy Improvement**: Reducing hallucinations by validating information sources before generation

This implementation showcases advanced RAG techniques applicable to production systems requiring high accuracy, such as customer support, legal research, and enterprise knowledge management.

## 🚀 Business Impact & Use Cases

This project demonstrates capabilities relevant to:

- **Enterprise RAG Systems**: Building intelligent Q&A systems over proprietary document collections
- **Customer Support Automation**: Accurate answer generation with fallback to live data sources
- **Research Assistants**: Academic and professional research tools with citation validation
- **Knowledge Management**: Corporate knowledge bases with self-correcting retrieval mechanisms

**Measurable Outcomes**: CRAG architecture reduces hallucination rates and improves answer accuracy by validating retrieval quality before generation—critical for production AI deployments.

## Requirements

- Python 3.14+
- Google AI API key (for Gemini)
- Pinecone account and API key
- Tavily API key
- LangSmith API key (optional, for tracing)

## License

MIT

## Acknowledgments

- Built with [LangChain](https://langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/)
- Powered by [Google Gemini](https://ai.google.dev/)
- Vector storage by [Pinecone](https://www.pinecone.io/)
- Web search by [Tavily](https://tavily.com/)

---

**Developer**: Demonstrated advanced AI/ML engineering skills including RAG architecture, LLM orchestration, vector databases, and production-ready AI system design.
