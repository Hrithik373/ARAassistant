import time
from evaluation.metrics import (
    relevance_score,
    faithfulness_score,
    groundedness_score,
)


def run_pdf_evaluation(
    question: str,
    agent,
    retriever_fn,
) -> dict:
    """
    Runs agent + RAG + evaluation on a PDF-backed system.
    """

    # Retrieve context from PDF
    retrieved_context = retriever_fn(question)

    # Run agent
    start_time = time.time()
    answer = agent.run(question)
    latency = round(time.time() - start_time, 3)

    # Run evaluation metrics
    relevance = relevance_score(question, answer)
    faithfulness = faithfulness_score(answer, retrieved_context)
    groundedness = groundedness_score(answer, retrieved_context)

    return {
        "question": question,
        "answer": answer,
        "relevance": relevance,
        "faithfulness": faithfulness,
        "groundedness": groundedness,
        "latency_sec": latency,
        "overall_score": round(
            (relevance + faithfulness + groundedness) / 3, 3
        ),
    }
