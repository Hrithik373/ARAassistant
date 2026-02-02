import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.document_loader import load_and_split_pdf
from rag.vector_store import build_vectorstore
from rag.retriever import create_retriever
from agent.agent_core import build_agent
from evaluation.eval_runner import run_pdf_evaluation


# 1️⃣ Load PDF
docs = load_and_split_pdf("sample.pdf")

# 2️⃣ Build vector store
vectorstore = build_vectorstore(docs)

# 3️⃣ Create retriever
retriever_fn = create_retriever(vectorstore)

# 4️⃣ Build agent
agent = build_agent(retriever_fn)

# 5️⃣ Evaluate on PDF
question = "What is agentic AI?"

report = run_pdf_evaluation(
    question=question,
    agent=agent,
    retriever_fn=retriever_fn,
)

print("\nPDF EVALUATION REPORT")
for k, v in report.items():
    print(f"{k}: {v}")
