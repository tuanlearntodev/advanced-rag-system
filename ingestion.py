from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
import os
load_dotenv()

file_path = [
  "D:\\advanced-rag-system\\105 Reading Selections.pdf",
  "D:\\advanced-rag-system\\Definitions for PHI 105.pdf"
]
docs = []
for path in file_path:
    loader = PyPDFLoader(path)
    docs.extend(loader.load())
    
    
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=500
)

docs_split = text_splitter.split_documents(docs)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Get index name from .env file
index_name = os.getenv("PINECONE_INDEX_NAME")

print(f"Total documents to upload: {len(docs_split)}")


batch_size = 100
for i in range(0, len(docs_split), batch_size):
    batch = docs_split[i:i + batch_size]
    print(f"Uploading batch {i//batch_size + 1} ({len(batch)} documents)...")
    
    if i == 0:
        vector_store = PineconeVectorStore.from_documents(
            documents=batch,
            embedding=embeddings,
            index_name=index_name
        )
    else:
        vector_store.add_documents(batch)

print(f"Successfully stored {len(docs_split)} documents in Pinecone index '{index_name}'")




