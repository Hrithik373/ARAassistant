import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm import OpenAILLM

llm = OpenAILLM()
out = llm.generate("Explain agentic AI in one sentence")
print(out["text"])
