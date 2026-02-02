import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.agent_core import build_agent


def dummy_retriever(query: str) -> str:
    return (
        "Agentic AI refers to AI systems that can autonomously plan, "
        "make decisions, and take actions toward goals."
    )


agent = build_agent(dummy_retriever)
result = agent.run("What is agentic AI?")

print("\nFINAL ANSWER:\n", result)
