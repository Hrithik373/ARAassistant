from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings


def build_vectorstore(documents):
    """
    Build a FAISS vector store from document chunks.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    return FAISS.from_documents(
        documents,
        embedding=embeddings
    )
