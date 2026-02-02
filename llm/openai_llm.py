import time
from typing import Optional, Dict
from openai import OpenAI
import os


class OpenAILLM:
    """
    OpenAI LLM wrapper for agentic AI projects (Colab-friendly)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not set")

        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Dict:
        """
        Standard text generation
        """
        messages = []

        if system_prompt:
            messages.append(
                {"role": "system", "content": system_prompt}
            )

        messages.append(
            {"role": "user", "content": prompt}
        )

        start_time = time.time()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        latency = round(time.time() - start_time, 3)

        return {
            "text": response.choices[0].message.content,
            "latency": latency,
            "tokens": {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": response.usage.total_tokens,
            },
        }

    def generate_with_context(
        self,
        question: str,
        context: str,
    ) -> Dict:
        """
        Context-grounded generation (RAG)
        """
        system_prompt = (
            "You are a helpful AI assistant. "
            "Answer strictly using the provided context. "
            "If the answer is not in the context, say 'I don't know'."
        )

        prompt = f"""
        Context:
        {context}

        Question:
        {question}
        """

        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt
        )

    def stream(self, prompt: str):
        """
        Token streaming (for Streamlit)
        """
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.get("content"):
                yield delta["content"]
