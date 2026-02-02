from langchain_openai import ChatOpenAI
judge_llm = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0
)
def _score_with_prompt(prompt: str) -> float:
    response = judge_llm.invoke(prompt)
    text = response.content.strip()
    try:
        return float(text)
    except ValueError:
        return 0.0
def relevance_score(question: str, answer: str) -> float:
    prompt = f'''
You are an evaluator.
Score the relevance of the answer to the question from 0 to 1.
ONLY return a number.
Question:
{question}
Answer:
{answer}
'''
    return _score_with_prompt(prompt)
def faithfulness_score(answer: str, context: str) -> float:
    prompt = f'''
You are an evaluator.
Score how factually faithful the answer is to the context from 0 to 1.
ONLY return a number.
Context:
{context}
Answer:
{answer}
'''
    return _score_with_prompt(prompt)
def groundedness_score(answer: str, context: str) -> float:
    prompt = f'''
You are an evaluator.
Score how well the answer is grounded in the given context from 0 to 1.
ONLY return a number.
Context:
{context}
Answer:
{answer}
'''
    return _score_with_prompt(prompt)
