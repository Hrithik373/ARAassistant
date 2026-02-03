"""
Streamlit Cloud–safe PDF loader.

This implementation avoids LangChain's PDF loaders entirely
and uses pypdf directly, which is 100% reliable on Streamlit Cloud.
"""

from pypdf import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_and_split_pdf(pdf_path: str):
    """
    Load a PDF using pypdf and split into LangChain Documents.

    Args:
        pdf_path (str): Path to PDF file

    Returns:
        List[Document]
    """

    reader = PdfReader(pdf_path)
    full_text = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text.append(text)

    full_text = "\n".join(full_text)

    documents = [
        Document(
            page_content=full_text,
            metadata={"source": pdf_path},
        )
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    return splitter.split_documents(documents)
