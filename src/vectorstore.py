from pathlib import Path
from langchain_chroma import Chroma
from src.embeddings import get_embedding_model
from src.config import VECTORSTORE_DIR

COLLECTION_NAME = "document_qa"


def create_vectorstore(documents):

    embeddings = get_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR)
    )

    return vectorstore


def load_vectorstore():

    if not Path(VECTORSTORE_DIR).exists():
        raise FileNotFoundError(
            "Vector store does not exist. "
            "Run indexing first."
        )

    embeddings = get_embedding_model()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR)
    )

    return vectorstore