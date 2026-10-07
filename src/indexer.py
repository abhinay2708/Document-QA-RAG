from src.config import DATA_DIR
from src.loaders import load_documents
from src.splitter import split_documents
from src.vectorstore import create_vectorstore

def build_index():

    print()
    print("=" * 70)
    print("RAG DOCUMENT INDEXING")
    print("=" * 70)

    # STEP 1: LOAD

    print()
    print("[1/3] Loading documents...")

    documents = load_documents(DATA_DIR)

    print(
        f"\nTotal document units: "
        f"{len(documents)}"
    )

    # STEP 2: CHUNK

    print()
    print("[2/3] Splitting documents...")

    chunks = split_documents(documents)

    print(
        f"Total chunks created: "
        f"{len(chunks)}"
    )

    # STEP 3: EMBED + STORE

    print()
    print(
        "[3/3] Creating embeddings "
        "and storing in Chroma..."
    )

    create_vectorstore(chunks)

    print()
    print("=" * 70)
    print("INDEXING COMPLETE")
    print("=" * 70)

    print(f"Documents : {len(documents)}")
    print(f"Chunks    : {len(chunks)}")

    print("Vector DB : vectorstore/")
    print("=" * 70)

if __name__ == "__main__":
    build_index()