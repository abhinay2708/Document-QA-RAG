from src.vectorstore import load_vectorstore
from src.config import TOP_K

def get_retriever():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    return retriever