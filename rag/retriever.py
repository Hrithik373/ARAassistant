def create_retriever(vectorstore, k: int = 4):
    """
    Create a retriever function usable as a LangChain Tool.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    def retrieve(query: str) -> str:
        docs = retriever.get_relevant_documents(query)
        return "\n\n".join(doc.page_content for doc in docs)
    return retrieve
