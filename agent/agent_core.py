from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI


def build_agent(retriever_fn):
    """
    Builds a LangChain ReAct agent using ChatOpenAI.
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2
    )

    tools = [
        Tool(
            name="Document Retriever",
            func=retriever_fn,
            description="Retrieve relevant document chunks for a query"
        )
    ]

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        memory=memory,
        verbose=True
    )

    return agent
