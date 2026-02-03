"""
Document loader with Streamlit Cloud–safe imports.

This file handles LangChain import inconsistencies by
falling back to legacy paths when needed.
"""

# =================================================
# SAFE IMPORT FOR PyPDFLoader (CRITICAL FIX)
# =================================================
try:
    # Preferred import (newer LangChain versions)
    from langchain_community.document_loaders import PyPDFLoader
except ModuleNotFoundError:
    # Fallback for Streamlit Cloud / partial installs
    from langchain.document_loaders import PyPDFLoader


from langchain.text_splitter import RecursiveCharacterTextSplitter


# =================================================
# PDF Loader + Chunker
# =================================================
def load_and_split_pdf(pdf_path: str):
    """
    Loads a PDF file and splits it into text chunks.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        List[Document]: Chunked LangChain documents
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    return splitter.split_documents(documents)
