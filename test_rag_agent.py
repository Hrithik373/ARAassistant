import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag.document_loader import load_and_split_pdf
from rag.vector_store import build_vectorstore
from rag.retriever import create_retriever

from agent.agent_core import build_agent


# Load & chunk document
docs = load_and_split_pdf("sample.pdf")

# Build vector DB
vectorstore = build_vectorstore(docs)

# Create retriever tool
retriever_fn = create_retriever(vectorstore)

# Build agent with retriever
agent = build_agent(retriever_fn)

# Ask question
response = agent.run(
    "What does the document say about agentic AI?"
)

print("\nFINAL ANSWER:\n", response)
