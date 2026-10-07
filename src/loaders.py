from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def load_single_document(file_path: Path) -> list[Document]:

    extension = file_path.suffix.lower()

    # PDF
    if extension == ".pdf":
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()

    # TXT
    elif extension == ".txt":
        loader = TextLoader(
            str(file_path),
            encoding="utf-8"
        )
        documents = loader.load()

    # DOCX
    elif extension == ".docx":
        loader = Docx2txtLoader(str(file_path))
        documents = loader.load()

    else:
        raise ValueError(f"Unsupported file type: {extension}")

    # Add / normalize metadata

    for document in documents:
        document.metadata["source"] = (file_path.name)
        document.metadata["file_path"] = (str(file_path))

        # PyPDFLoader usually gives page as 0-based.
        # Convert to human-readable 1-based page.
        if extension == ".pdf" and "page" in document.metadata:
            document.metadata["page"] = document.metadata["page"] + 1

        else:
            document.metadata["page"] = None

    return documents


def load_documents(data_directory: Path) -> list[Document]:
    if not data_directory.exists():
        raise FileNotFoundError(f"Data directory does not exist: "f"{data_directory}")

    files = sorted(
        [
            file
            for file in data_directory.iterdir()
            if (
                file.is_file()
                and file.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        ]
    )

    if not files:
        raise ValueError("No supported documents found in data/.")
    
    all_documents = []

    for file_path in files:
        print(f"[LOAD] {file_path.name}")
        documents = load_single_document(file_path)
        print(f"       Pages/units: "f"{len(documents)}")
        all_documents.extend(documents)

    return all_documents