from pathlib import Path
from src.config import VECTORSTORE_DIR
from src.rag_pipeline import RAGPipeline

def print_sources(sources):

    print()
    print("=" * 70)
    print("RETRIEVED SOURCES")
    print("=" * 70)

    for index, source in enumerate(sources, start=1):

        filename = source["source"]
        page = source["page"]
        chunk_id = source["chunk_id"]

        print()
        print(
            f"[{index}] "
            f"{filename}"
        )

        print(f"    Page     : {page}")
        print(f"    Chunk ID : {chunk_id}")

        print(
            f"    Content  : "
            f"{source['content'][:300]}..."
        )

    print("=" * 70)

def main():

    print()
    print("=" * 70)
    print("DOCUMENT Q&A BOT")
    print("LangChain + Chroma + HuggingFace + Gemini")
    print("=" * 70)

    # Check vector store

    if not Path(VECTORSTORE_DIR).exists():

        print()
        print("Vector store not found.")

        print()
        print("Run:")
        print("python -m src.indexer")

        return

    # Load RAG system

    print()

    rag = RAGPipeline()

    print()
    print("System ready.")

    print("Type 'exit' or 'quit' to stop.")

    print("-" * 70)

    # Interactive loop

    while True:
        try:
            question = input("\nQuestion: ").strip()

        except KeyboardInterrupt:
            print("\nExiting...")
            break

        if question.lower() in {"exit", "quit"}:
            print("\nGoodbye.")
            break

        if not question:
            continue

        try:
            result = rag.ask(question)

            print()
            print("=" * 70)
            print("ANSWER")
            print("=" * 70)

            print(result["answer"])
            print_sources(result["sources"])

        except Exception as error:
            print()
            print("ERROR:")

            print(error)

if __name__ == "__main__":
    main()