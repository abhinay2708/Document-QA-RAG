from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL

def get_embedding_model():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32
        },
        show_progress=True
    )

    return embeddings